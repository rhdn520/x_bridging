"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import argparse
import os, json
from tracemalloc import start
import numpy as np
import torch as th
th.set_printoptions(threshold=float('inf'))
import torch.distributed as dist
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round
from diffuseq.text_datasets import load_data_text
from bridging.bm_retriever import SimpleBM25Retriever
from bridging.make_startpool import process_jsonl_file

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_tokenizer
)
import sys
from tqdm import tqdm

def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def linear_interpolate(p1, p2, t):
    """
    Performs linear interpolation between two points in N-dimensional Euclidean space.

    The function calculates the point that lies a fraction `t` of the way along the
    line segment from `p1` to `p2`.

    Args:
        p1 (torch.Tensor or array-like): The coordinates of the first point.
        p2 (torch.Tensor or array-like): The coordinates of the second point.
                                         Must have the same dimension as p1.
        t (float): The interpolation factor, which should ideally be between 0.0 and 1.0.
                   - A value of 0.0 will return p1.
                   - A value of 1.0 will return p2.
                   - A value of 0.5 will return the midpoint.
                   - Values outside this range perform extrapolation.

    Returns:
        torch.Tensor: A PyTorch tensor representing the coordinates of the interpolated point.

    Raises:
        ValueError: If the input points have different dimensions.
    """
    # Convert inputs to PyTorch tensors, using float32 for standard precision
    # torch.as_tensor avoids a copy if the input is already a tensor of the correct type.
    p1 = th.as_tensor(p1, dtype=th.float32)
    p2 = th.as_tensor(p2, dtype=th.float32)

    # Ensure the points are in the same dimensional space
    if p1.shape != p2.shape:
        raise ValueError("Points must have the same dimensions for interpolation.")

    # The formula for linear interpolation (lerp) works identically with tensors
    # P(t) = (1 - t) * P1 + t * P2
    interpolated_point = (1 - t) * p1 + t * p2
    
    return interpolated_point

def interpolate_points(p1, p2, num_mid_points):
    """
    Generates a list of evenly spaced intermediate points between two points.

    Args:
        p1 (torch.Tensor or array-like): The coordinates of the first point.
        p2 (torch.Tensor or array-like): The coordinates of the second point.
        num_points (int): The number of intermediate points to generate.

    Returns:
        list[torch.Tensor]: A list of PyTorch tensors, where each tensor is an
                            interpolated point. Returns an empty list if
                            num_mid_points is less than 1.
    """
    if num_mid_points < 1:
        return [p1, p2]

    interpolated_points = []
    # There will be num_mid_points + 1 total segments along the line
    total_segments = num_mid_points + 1

    for i in range(1, total_segments):
        t = i / total_segments
        point = linear_interpolate(p1, p2, t)
        interpolated_points.append(point)
        
    return [p1] + interpolated_points + [p2]


@th.no_grad()
def main():
    """
    LOAD MODEL
    """
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    world_size = dist.get_world_size() or 1
    rank = dist.get_rank() or 0

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    print(args)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # logger.log(f'### The parameter count is {pytorch_total_params}')

    model = model.eval().requires_grad_(False).to(dist_util.dev())

    tokenizer = load_tokenizer(args)

    model_emb = th.nn.Embedding(
        num_embeddings=tokenizer.vocab_size, 
        embedding_dim=args.hidden_dim, 
        _weight=model.word_embedding.weight.clone().cpu()
    ).eval().requires_grad_(False)

    model_emb = model_emb.to(dist_util.dev())

    set_seed(args.seed2)


    """
    LOAD START POOL AND RETRIEVER 
    """
    #Data for BM25 Retriever
    with open("bridging/startpool.txt", encoding='utf-8') as f:
        startpool_sents = f.readlines()

    bm25_retriever = SimpleBM25Retriever(startpool_sents)

    """
    LOAD TEST DATA 
    """
    ref_sents = [x['trg'] for x in process_jsonl_file("datasets/qqp/test.jsonl")]
    test_sents = [x['src'] for x in process_jsonl_file("datasets/qqp/test.jsonl")]


    """
    DO BRIDGING INFERENCE
    """
    # count = 0

    start_sents = []
    bridging_recovers = []
    vanilla_recovers = []
    for idx, test_sent in tqdm(enumerate(test_sents)):
        # count += 1 
        # if(count > 100):
        #     break

        start_sent = bm25_retriever.search(test_sent)[0].page_content
        start_sents.append(start_sent)

        # print(start_sent)
        # print(test_sent)
        text_token = tokenizer.encode_token([start_sent, test_sent], is_bridging=True).to(dist_util.dev())
        # print(text_token.shape)
        sep_tokens = th.full((text_token.shape[0],1), tokenizer.sep_token_id).to(dist_util.dev())
        text_token = th.cat([text_token, sep_tokens], dim=1)
        mask_idx = text_token.shape[1]


        x_src = model.get_embeds(text_token)
        # print(x_src.shape)
        x_start = th.randn((x_src.shape[0], args.seq_len, x_src.shape[2]), dtype=th.float32, device=dist_util.dev())
        # print(x_start.shape)
        x_start[:,:x_src.shape[1],:] = x_src

        input_ids_mask = th.zeros_like(x_start, dtype=th.int16, device=dist_util.dev())
        input_ids_mask[:,mask_idx:,:] = 1
        # print(input_ids_mask)
        # print(input_ids_mask[0,:,0])
        # input_ids_mask[:,:x_src.shape[1],:] = 0

        # print("hmm")
        # print(x_start.shape)
        intp_points = interpolate_points(x_start[0], x_start[1], 7)
        # intp_points = [x.to(dist_util.dev()) for x in intp_points]

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step

        sample_fn = diffusion.p_sample_loop_from_to
        schedule = [(1999,1998),(1998,1997),(1997,1996),(1996,1995),(1995,1994),(1994,1993),(1993,1992),(1992,1991),(1991,0)]
        assert len(schedule) == len(intp_points)

        # sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim)
        sample_shape = (1, args.seq_len, args.hidden_dim)

        input_noise = intp_points[0].unsqueeze(0)

        for i in range(len(intp_points)):   
            samples = sample_fn(
                model,
                sample_shape,
                noise=input_noise,
                clip_denoised=args.clip_denoised,
                denoised_fn=partial(denoised_fn_round, args, model_emb),
                model_kwargs=model_kwargs,
                top_p=args.top_p,
                clamp_step=args.clamp_step,
                clamp_first=True,
                mask=input_ids_mask[0].unsqueeze(0),
                x_start=intp_points[i],
                gap=step_gap,
                from_time=schedule[i][0],
                to_time=schedule[i][1],
                progress=False
            )
            #     print("samples[0].shape:::")
            #     print(len(samples)) # samples for each step
                
            sample = samples[-1]

            # print(input_noise.shape)


            if(i+1)<len(intp_points): 
                input_noise = sample
                input_noise[:,:mask_idx,:] = intp_points[i+1].unsqueeze(0)[:,:mask_idx,:]
            

            # print(word_lst_recover)
            if(i+1 == len(intp_points)):
                logits = model.get_logits(sample)  # bsz, seqlen, vocab
                cands = th.topk(logits, k=1, dim=-1)

                word_lst_recover = []

                input_ids_mask_ori = input_ids_mask[0,:,0]
                for seq in cands.indices:
                    
                    len_x = args.seq_len - sum(input_ids_mask_ori).tolist()
                    tokens = tokenizer.decode_token(seq[len_x:])
                    word_lst_recover.append(tokens)
            
                bridging_recovers.append(word_lst_recover[0])

        # baseline: without interpolation
        # print("### Baseline without interpolation")
        samples = sample_fn(
            model,
            sample_shape,
            noise=intp_points[-1].unsqueeze(0),
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask[0].unsqueeze(0),
            x_start=intp_points[-1],
            gap=step_gap,
            from_time=1999,
            to_time=0,
            progress=False
        )
        sample = samples[-1]
        logits = model.get_logits(sample)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)

        word_lst_recover = []

        input_ids_mask_ori = input_ids_mask[0,:,0]
        for seq in cands.indices:
            
            len_x = args.seq_len - sum(input_ids_mask_ori).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)
        
        # print(word_lst_recover)
        vanilla_recovers.append(word_lst_recover[0])

        #append new jsonl line to output file
        if (idx+1)%10 == 0:
            test_result = []
            for b, v, r, s in zip(bridging_recovers, vanilla_recovers, ref_sents[:len(bridging_recovers)], start_sents[:len(bridging_recovers)]):
                tmp = {
                    "start_sent": s,
                    "bridg_recover": b,
                    "vanilla_recover": v,
                    "reference": r
                }
                test_result.append(tmp)
            
            # Save to JSONL
            with open("bridging/outputs/test_result9.jsonl", "w", encoding="utf-8") as f:
                for item in test_result:
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + "\n")

if __name__ == "__main__":
    main()
