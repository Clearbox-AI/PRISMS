"""
Train a diffusion model on image-tabular pairs.
"""

import sys
import os
import json
import argparse
import numpy as np
import torch as th
from glob import glob
import pathlib
from torch.utils.data import Dataset
from PIL import Image

# Assuming mm_diffusion is a package or module available in your environment
from multi_modal_diffusion import dist_util, logger
from multi_modal_diffusion.resample import create_named_schedule_sampler
from diffusion_process.multimodal_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)
from diffusion_process.multimodal_train_util import TrainLoop
from multi_modal_diffusion.common import set_seed_logger_random
from torch.utils.data.distributed import DistributedSampler
from sklearn.preprocessing import StandardScaler
from diffusion_process.dataloaders import (ImageTabularDataset, ToyMNISTDataset, ExpLumirDataset, LDMOneHDataset)


def load_training_data(args):
    # dataset = ImageTabularDataset(args.data_dir, image_size=(64, 64))
    # dataset = ToyMNISTDataset(args.data_dir)
    # dataset = ExpLumirDataset(args.data_dir, image_size=(64, 64))
    dataset = LDMOneHDataset(args.data_dir, image_size=(64, 64), modality="tabular")
    sampler = DistributedSampler(dataset) if dist_util.get_world_size() > 1 else None
    data_loader = th.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
    )
    return data_loader


def main():
    args = create_argparser().parse_args()
    # args.image_size and args.tabular_size are already in the correct format
    logger.configure(args.output_dir)

    if args.devices is None:
        args.devices = "cpu"

    dist_util.setup_dist(args.devices)

    args = set_seed_logger_random(args)

    logger.log("Creating data loader...")
    data_loader = load_training_data(args)

    # Infer image_size and tabular_size from data if not specified
    sample_batch = next(iter(data_loader))
    image_shape = sample_batch['image'].shape  # [batch_size, C, H, W]
    tabular_shape = sample_batch['tabular'].shape  # [batch_size, tabular_size]

    # Update args.image_size and args.tabular_size
    args.image_size = ','.join(map(str, image_shape[1:]))  # Exclude batch dimension
    args.tabular_size = str(tabular_shape[1])  # Exclude batch dimension

    logger.log("Creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
    )

    # Move model to the defined device
    model.to(dist_util.dev())

    # Wrap model with DistributedDataParallel if using GPUs
    # if dist_util.dev().type == 'cuda' and dist_util.get_world_size() > 1:
    #     model = th.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[dist_util.dev()], output_device=dist_util.dev(),
    #         find_unused_parameters=True
    #     )

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Starting training...")

    num_epochs = 2
    eval_interval = 1
    num_eval_samples = 20

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        num_epochs=num_epochs,
        lr=args.lr,
        t_lr=args.t_lr,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        class_cond=args.class_cond,
        sample_fn=args.sample_fn,
        eval_interval=eval_interval,
        num_eval_samples=num_eval_samples,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",  # User should specify the data directory
        schedule_sampler="uniform",
        lr=1e-4,
        t_lr=1e-4,
        seed=42,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        num_workers=0,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        devices=None,  # This argument is retained but not used
        save_interval=2000,
        output_dir="/mnt/storage/lumir_three_stage_exp/stage_two",
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        sample_fn="dpm_solver",
        class_cond=False,
        image_size="",  # Will be inferred from data
        tabular_size=""  # Will be inferred from data
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
