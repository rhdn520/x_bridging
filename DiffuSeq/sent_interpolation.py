"""
Do Sentence Interpolation between two sentences using DiffuSeq model.
"""
import argparse
import os, json
from tracemalloc import start
import numpy as np
import torch as th
th.set_printoptions(threshold=float('inf'))
import torch.distributed as dist

from transformers import AutoTokenizer, set_seed
from diffuseq.rounding import denoised_fn_round
from diffuseq.text_datasets import load_data_text
from bridging.bm_retriever import SimpleBM25Retriever
from bridging.make_startpool import process_jsonl_file
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append("../")
sys.path.append("/home/seungwoochoi/data/x_bridging/bridging")
from latent_intp import linear_interpolate, slerp_channel_wise

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

    tokenizer = AutoTokenizer.from_pretrained(args.config_name)

    model_emb = th.nn.Embedding(
        num_embeddings=tokenizer.vocab_size, 
        embedding_dim=args.hidden_dim, 
        _weight=model.word_embedding.weight.clone().cpu()
    ).eval().requires_grad_(False)

    model_emb = model_emb.to(dist_util.dev())

    set_seed(args.seed2)

    if args.step == args.diffusion_steps:
        args.use_ddim = False
        step_gap = 1
    else:
        args.use_ddim = True
        step_gap = args.diffusion_steps//args.step


    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    model_kwargs = {}


    text_0 = ["What is your name?"]
    text_1 = ["I don't like hot coffee."]
    
    intp_paths = []

    for t0, t1 in tqdm(zip(text_0, text_1)):
        print(f"### Interpolating between: '{t0}' AND '{t1}'")

        tokens_0 = tokenizer(
            t0,
            max_length=args.seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(dist_util.dev())

        tokens_1 = tokenizer(
            t1,
            max_length=args.seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(dist_util.dev())

        intp_path = []

        z_0 = model.get_cls_conditioned_embeds(tokens_0.input_ids)
        z_1 = model.get_cls_conditioned_embeds(tokens_1.input_ids)
        mask = th.ones_like(z_0).to(dist_util.dev())
        mask[:,0,:] = 0

        noise = th.randn_like(z_0)
        z_0_noised = th.where(mask == 0, z_0, noise)
        sample_shape = (z_0.shape[0], args.seq_len, args.hidden_dim)

        samples = sample_fn(
            model,
            sample_shape,
            noise=z_0_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=mask,
            x_start=z_0,
            gap=step_gap
        )

        sample = samples[-1]

        logits = model.get_logits(sample)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)
        # print(cands)

        decoded_text = tokenizer.batch_decode(cands.indices.squeeze(-1), skip_special_tokens=True)

        print(f"### Generated from text 0: '{decoded_text[0]}'")

        intp_path.append(decoded_text[0])

        alphas = np.linspace(0, 1, num=5)[1:-1]
        for alpha in alphas:
            #embed_0과 embed_1의 CLS토큰 임베딩을 interpolate
            z_0_cls = z_0[:,0,:].unsqueeze(1)
            z_1_cls = z_1[:,0,:].unsqueeze(1)
            interped_cls = linear_interpolate(z_0_cls, z_1_cls, alpha)
            noise = th.randn_like(z_0)
            z_interped = th.where(mask == 0, interped_cls, noise)
            samples = sample_fn(
                model,
                sample_shape,
                noise=z_interped,
                clip_denoised=args.clip_denoised,
                denoised_fn=partial(denoised_fn_round, args, model_emb),
                model_kwargs=model_kwargs,
                top_p=args.top_p,
                clamp_step=args.clamp_step,
                clamp_first=True,
                mask=mask,
                x_start=z_0,
                gap=step_gap
            )
            sample = samples[-1]
            logits = model.get_logits(sample)  # bsz, seqlen, vocab
            cands = th.topk(logits, k=1, dim=-1)
            decoded_text = tokenizer.batch_decode(cands.indices.squeeze(-1), skip_special_tokens=True)
            print(f"### Generated at alpha {alpha:.2f}: '{decoded_text[0]}'")
            intp_path.append(decoded_text[0])
        noise = th.randn_like(z_1)
        z_1_noised = th.where(mask == 0, z_1, noise)
        samples = sample_fn(
            model,
            sample_shape,
            noise=z_1_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=mask,
            x_start=z_1,
            gap=step_gap
        )   
        sample = samples[-1]
        logits = model.get_logits(sample)  # bsz, seqlen, vocab
        cands = th.topk(logits, k=1, dim=-1)
        decoded_text = tokenizer.batch_decode(cands.indices.squeeze(-1), skip_special_tokens=True)
        print(f"### Generated from text 1: '{decoded_text[0]}'")
        intp_path.append(decoded_text[0])

        intp_paths.append(intp_path)

    print(intp_paths)

if __name__ == "__main__":
    main()
    dist.destroy_process_group()