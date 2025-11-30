"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import argparse
import os, json
from tracemalloc import start
import numpy as np

import torch
import warnings
torch.set_printoptions(threshold=float('inf'))
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
    p1 = torch.as_tensor(p1, dtype=torch.float32)
    p2 = torch.as_tensor(p2, dtype=torch.float32)

    # Ensure the points are in the same dimensional space
    if p1.shape != p2.shape:
        raise ValueError("Points must have the same dimensions for interpolation.")

    # The formula for linear interpolation (lerp) works identically with tensors
    # P(t) = (1 - t) * P1 + t * P2
    interpolated_point = (1 - t) * p1 + t * p2
    
    return interpolated_point


def slerp_vectors_torch(v1, v2, t):
    """
    Performs Spherical Linear Interpolation (SLERP) between two torch tensors.

    This function interpolates the *direction* of the vectors, ensuring
    the interpolated vector moves along the shortest arc on a sphere.
    Assumes v1 and v2 are 1D tensors (single vectors).
    t can be a float or a 1D tensor of interpolation factors.

    Args:
        v1 (torch.Tensor): The starting vector.
        v2 (torch.Tensor): The ending vector.
        t (float or torch.Tensor): The interpolation factor(s), clamped to [0, 1].
                                   A single float or a 1D tensor of values.

    Returns:
        torch.Tensor: The interpolated vector(s). The magnitude will also be
                      interpolated linearly between v1 and v2.
    """
    
    # --- 1. Handle magnitudes (LERP) ---
    # SLERP only deals with direction. We'll LERP the magnitudes separately.
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)
    
    # Handle zero vectors
    if norm1 == 0 or norm2 == 0:
        if norm1 == 0 and norm2 == 0:
            return v1 # Both are zero, just return one
        
        # One is zero, standard LERP is fine
        # Inlined LERP logic for vectors:
        if isinstance(t, torch.Tensor):
            # t.unsqueeze(-1) expands t from shape [n] to [n, 1]
            # for broadcasting with v1 and v2 of shape [3]
            t_reshaped = t.unsqueeze(-1) 
            return (1 - t_reshaped) * v1 + t_reshaped * v2
        else:
            # t is a float, standard LERP
            return (1 - t) * v1 + t * v2

    # Normalize vectors
    v1_norm = v1 / norm1
    v2_norm = v2 / norm2
    
    # LERP the magnitudes
    interp_norm = (1 - t) * norm1 + t * norm2

    # --- 2. Handle directions (SLERP) ---
    
    # Calculate the angle between the vectors
    dot = torch.dot(v1_norm, v2_norm)
    
    # Clip dot product to avoid numerical errors with acos
    dot = torch.clamp(dot, -1.0, 1.0)
    
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    # We need to handle the case where the vectors are nearly collinear.
    # In this case, sin(omega) is close to 0, and we can use LERP
    # as a stable approximation.
    
    # Check if t is a tensor or a float
    if isinstance(t, torch.Tensor):
        # Handle tensor of t values
        t_reshaped = t.unsqueeze(-1) # Shape [n, 1] for broadcasting
        
        if sin_omega < 1e-6:
            # Use LERP for collinear vectors
            lerp_dir = (1 - t_reshaped) * v1_norm + t_reshaped * v2_norm
            # Normalize the LERP results
            interp_dir = lerp_dir / torch.norm(lerp_dir, dim=1, keepdim=True)
        else:
            # Use SLERP formula for all t
            t1 = torch.sin((1 - t_reshaped) * omega) / sin_omega
            t2 = torch.sin(t_reshaped * omega) / sin_omega
            # v1_norm and v2_norm broadcast from [3] to [n, 3]
            interp_dir = t1 * v1_norm + t2 * v2_norm
        
        # Apply the interpolated magnitude
        # interp_norm is [n], needs to be [n, 1] for broadcasting
        return interp_dir * interp_norm.unsqueeze(-1)
        
    else:
        # Handle single float t value
        if sin_omega < 1e-6:
            # Vectors are collinear. Use LERP and re-normalize.
            # This is stable and avoids division by zero.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning) # Ignore divide-by-zero
                result_dir = (1 - t) * v1_norm + t * v2_norm
                result_norm = torch.norm(result_dir)
                if result_norm == 0:
                    interp_dir = v1_norm # Fallback if LERP gives zero vector
                else:
                    interp_dir = result_dir / result_norm
        else:
            # Standard SLERP formula
            t1 = torch.sin((1 - t) * omega) / sin_omega
            t2 = torch.sin(t * omega) / sin_omega
            interp_dir = (t1 * v1_norm) + (t2 * v2_norm)
            # No need to re-normalize, formula guarantees a unit vector

        # Apply the interpolated magnitude (which is a float)
        return interp_dir * interp_norm

@torch.no_grad()
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

    model_emb = torch.nn.Embedding(
        num_embeddings=tokenizer.vocab_size, 
        embedding_dim=args.hidden_dim, 
        _weight=model.word_embedding.weight.clone().cpu()
    ).eval().requires_grad_(False)

    model_emb = model_emb.to(dist_util.dev())

    set_seed(args.seed2)

    """
    LOAD TEST DATA 
    """
    
    """
    DO BRIDGING INFERENCE
    """
    # count = 0

    word0 = ["Happy"]
    word1 = ["Sad"]
    bridging_recovers = []
    vanilla_recovers = []
    for word0, word1 in tqdm(zip(word0, word1)):

        # print(start_sent)
        # print(test_sent)
        text_token = tokenizer.encode_token([word0, word1], is_bridging=True).to(dist_util.dev())
        # print(text_token.shape)
        # sep_tokens = torch.full((text_token.shape[0],1), tokenizer.sep_token_id).to(dist_util.dev())
        # text_token = torch.cat([text_token, sep_tokens], dim=1)
        mask_idx = text_token.shape[1]

        print(text_token)


        x_start = model.get_embeds(text_token)
        # print(x_src)
        # x_start 
        print(x_start.shape) # (문장 개수 x 토큰수(3) x 차원수 (128))
        word_midpoint = slerp_vectors_torch(x_start[0,1,:], x_start[1,1,:], 0.5)
        print(word_midpoint)
        x_start = torch.cat([x_start, x_start[1].unsqueeze(0)], dim=0) #word1를 뒤에 추가하고 
        x_start[1,1,:] = word_midpoint #여기서 mid point로 갈아끼움

        print(x_start.shape)
        
        x_noised = torch.randn((1, args.seq_len, args.hidden_dim), dtype=torch.float32, device=dist_util.dev())
        print(x_start.shape)
        x_noised = x_noised.repeat(3,1,1)
        print(x_noised.shape)
        x_noised[:,:x_start.shape[1],:] = x_start

        input_ids_mask = torch.zeros((x_noised.shape[0], x_noised.shape[1]), dtype=torch.int16, device=dist_util.dev())
        input_ids_mask[:,mask_idx:] = 1
        input_ids_mask_ori = input_ids_mask
        input_ids_mask = torch.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_noised.shape).to(dist_util.dev())
        

        # # print("hmm")
        # # print(x_start.shape)

        """
        DO BRIDGING INFERENCE
        """

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step

        
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], args.seq_len, args.hidden_dim) 
        print(x_noised.shape)
        print(input_ids_mask.shape)

        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model_emb),
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_ids_mask,
            x_start=x_noised,
            gap=step_gap
        )
            
        sample = samples[-1]

        # print('decoding for seq2seq', )
        # print(sample.shape)

        logits = model.get_logits(sample)  # bsz, seqlen, vocab
        # print(logits.shape)
        cands = torch.topk(logits, k=1, dim=-1)


        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []

        for seq, input_mask in zip(cands.indices, input_ids_mask_ori):
            # print(input_mask.shape)
            # print(sum(input_mask))
            len_x = args.seq_len - sum(input_mask).tolist()
            tokens = tokenizer.decode_token(seq[len_x:])
            word_lst_recover.append(tokens)
        
        print(word_lst_recover)
    
if __name__ == "__main__":
    main()
