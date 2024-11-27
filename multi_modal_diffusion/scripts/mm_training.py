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


class ImageTabularDataset(Dataset):
    def __init__(self, data_dir, image_size=(256, 256)):
        self.data_dir = data_dir
        self.image_size = image_size  # Desired image size (width, height)
        # Get list of patient directories
        self.patient_dirs = [
            os.path.join(data_dir, d) for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]
        if not self.patient_dirs:
            raise ValueError(f"No patient directories found in {data_dir}")

        # Compute normalization parameters
        self.tabular_mean, self.tabular_std = self.compute_tabular_normalization()
        self.image_mean, self.image_std = self.compute_image_normalization()

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]

        # Find image file
        image_files = glob(os.path.join(patient_dir, '*.npy'))
        if not image_files:
            raise FileNotFoundError(f"No image .npy files found in {patient_dir}")
        image_path = image_files[0]  # Use the first .npy file found
        image = np.load(image_path).astype(np.float32)  # Shape: [H, W]

        # Resize image to the desired size
        image = self.resize_image(image, self.image_size)  # Now shape is [H, W]

        # Find JSON file
        json_files = glob(os.path.join(patient_dir, '*.json'))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {patient_dir}")
        json_path = json_files[0]  # Use the first .json file found
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        # Get the tabular data
        # Try 'patient_id' key; if not present, use the first value
        tabular_data = json_data.get('patient_id', list(json_data.values())[0])
        if not tabular_data:
            raise ValueError(f"No tabular data found in {json_path}")
        tabular_data = np.array(tabular_data, dtype=np.float32)

        # Normalize tabular data
        tabular_data = (tabular_data - self.tabular_mean) / self.tabular_std

        # Normalize image data
        image = (image - self.image_mean) / self.image_std

        # Convert to torch tensors
        image = th.from_numpy(image)  # Shape: [H, W]
        tabular_data = th.from_numpy(tabular_data)

        # Add channel dimension to image if necessary
        if image.dim() == 2:
            image = image.unsqueeze(0)  # Shape: [1, H, W]

        return {'image': image, 'tabular': tabular_data}

    def resize_image(self, image, size):
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        # Resize image
        pil_image = pil_image.resize(size[::-1], Image.BILINEAR)  # size[::-1] because PIL uses (width, height)
        # Convert back to numpy array
        image_resized = np.array(pil_image).astype(np.float32)
        return image_resized

    def compute_tabular_normalization(self):
        # [Same as before]
        # Collect all tabular data
        all_tabular_data = []
        for patient_dir in self.patient_dirs:
            json_files = glob(os.path.join(patient_dir, '*.json'))
            if not json_files:
                continue
            json_path = json_files[0]
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            tabular_data = json_data.get('patient_id', list(json_data.values())[0])
            if not tabular_data:
                continue
            all_tabular_data.append(tabular_data)
        if not all_tabular_data:
            raise ValueError("No tabular data found in any patient directories.")
        all_tabular_data = np.array(all_tabular_data, dtype=np.float32)
        mean = np.mean(all_tabular_data, axis=0)
        std = np.std(all_tabular_data, axis=0)
        std[std == 0] = 1.0  # Prevent division by zero
        return mean, std

    def compute_image_normalization(self):
        # Collect all image data
        all_image_pixels = []
        for patient_dir in self.patient_dirs:
            image_files = glob(os.path.join(patient_dir, '*.npy'))
            if not image_files:
                continue
            image_path = image_files[0]
            image = np.load(image_path).astype(np.float32)
            if image.shape != (256,256):
                pass
            # Resize image
            image = self.resize_image(image, self.image_size)
            all_image_pixels.append(image.flatten())
        if not all_image_pixels:
            raise ValueError("No image data found in any patient directories.")
        all_image_pixels = np.concatenate(all_image_pixels, axis=0)
        mean = np.mean(all_image_pixels)
        std = np.std(all_image_pixels)
        if std == 0:
            std = 1.0  # Prevent division by zero
        return mean, std

def load_training_data(args):
    dataset = ImageTabularDataset(args.data_dir, image_size=(256, 256))
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
    if dist_util.dev().type == 'cuda' and dist_util.get_world_size() > 1:
        model = th.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist_util.dev()], output_device=dist_util.dev()
        )

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Starting training...")

    num_epochs = 10
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
        save_interval=100,
        output_dir="output",
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        sample_fn="dpm_solver",
        class_cond=False,
        image_size="",  # Will be inferred from data
        tabular_size="",  # Will be inferred from data
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
