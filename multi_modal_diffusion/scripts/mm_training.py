"""
Train a diffusion model on image-tabular pairs.
"""

import sys
import os
import argparse
import torch as th

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


def load_training_data(args):
    # Generate dummy data
    batch_size = args.batch_size
    image_channels, image_height, image_width = [int(i) for i in args.image_size.split(',')]
    tabular_size = int(args.tabular_size)

    # Create infinite generator of dummy data
    while True:
        image_batch = th.randn(batch_size, image_channels, image_height, image_width)
        tabular_batch = th.randn(batch_size, tabular_size)
        gt_batch = {'image': image_batch, 'tabular': tabular_batch}
        yield gt_batch


def main():
    args = create_argparser().parse_args()
    # args.image_size and args.tabular_size are already in the correct format
    logger.configure(args.output_dir)

    # Commented out distributed setup
    dist_util.setup_dist(args.devices)

    # Define device for single-device training
    # device = th.device("cuda" if th.cuda.is_available() else "cpu")

    args = set_seed_logger_random(args)

    logger.log("Creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
    )

    # Move model to the defined device
    model.to(dist_util.dev())
    # model.to(device)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Creating data loader...")

    data = load_training_data(args)

    logger.log("Starting training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        lr=args.lr,
        t_lr=args.t_lr,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        class_cond=args.class_cond,
        sample_fn=args.sample_fn,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
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
        devices="cpu",  # This argument is retained but not used
        save_interval=100,
        output_dir="output",
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        sample_fn="dpm_solver",
        class_cond=False,
        image_size="3,64,64",  # Default image size (channels, height, width)
        tabular_size="128",  # Default tabular feature size
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
