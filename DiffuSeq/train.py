"""
Train a diffusion model on images.
"""
from dotenv import load_dotenv
load_dotenv()
import argparse
import json, torch, os
import numpy as np
# from compression.zstd import train_dict
from diffuseq.utils import dist_util, logger
from diffuseq.text_datasets import load_data_text, infinite_loader
from diffuseq.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer 
)
from train_util import TrainLoop
from transformers import set_seed
import wandb
import sys

from hf_dataset import TinyStoriesDataset
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"

def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # update latest args according to argparse
    return parser

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    tokenizer = AutoTokenizer.from_pretrained(args.config_name)
    # Ensure vocab_size is populated when skipping load_tokenizer().
    args.vocab_size = tokenizer.vocab_size
    # tokenizer = load_tokenizer(args)
    # model_weight, tokenizer = load_model_emb(args, tokenizer)

    # train_loader = load_data_text(
    #     batch_size=args.batch_size,
    #     seq_len=args.seq_len,
    #     data_args = args,
    #     loaded_vocab=tokenizer,
    #     model_emb=model_weight # use model's weights as init
    # )
    # print(next(train_loader))
        
    # val_loader = load_data_text(
    #     batch_size=args.batch_size,
    #     seq_len=args.seq_len,
    #     data_args=args,
    #     split='valid',
    #     deterministic=True,
    #     loaded_vocab=tokenizer,
    #     model_emb=model_weight # using the same embedding wight with tranining data
    # )
    # Data Limits
    TRAIN_SAMPLES = 3000000
    VAL_SAMPLES = 10000
    TEST_SAMPLES = 10000
    MAX_LEN = args.seq_len

    # Load Data (Identical on all ranks)
    train_dataset = TinyStoriesDataset(tokenizer, split="train", dataset_size=TRAIN_SAMPLES, max_seq_len=MAX_LEN)
    val_dataset = TinyStoriesDataset(tokenizer, split="validation", dataset_size=VAL_SAMPLES, skip_samples=0, max_seq_len=MAX_LEN)

    # Samplers handles the splitting
    dist_sampler_train = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    dist_sampler_val = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    # DataLoaders (num_workers > 0 is important for speed)
    train_loader = infinite_loader(DataLoader(train_dataset, batch_size=args.batch_size, sampler=dist_sampler_train, num_workers=1, pin_memory=True))
    val_loader = infinite_loader(DataLoader(val_dataset, batch_size=args.batch_size, sampler=dist_sampler_val, num_workers=1, pin_memory=True))


    print('#'*30, 'size of vocab', args.vocab_size)

    logger.log("### Creating model and diffusion...")
    # print('#'*30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    print(args_to_dict(args, load_defaults_config().keys()).keys())
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    # print('#'*30, 'cuda', dist_util.dev())
    model.to(dist_util.dev()) #  DEBUG **
    # model.cuda() #  DEBUG **

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "DiffuSeq"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("### Training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=val_loader,
        eval_interval=args.eval_interval
    ).run_loop()

if __name__ == "__main__":
    main()
