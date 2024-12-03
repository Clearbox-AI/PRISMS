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

        # Initialize StandardScaler for tabular data
        self.tabular_scaler = StandardScaler()

        # Compute normalization parameters
        # self.tabular_mean, self.tabular_std = self.compute_tabular_normalization()
        self.compute_tabular_normalization()
        self.image_mean, self.image_std = self.compute_image_normalization()

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        import matplotlib.pyplot as plt

        patient_dir = self.patient_dirs[idx]

        # IMAGE PART
        # Find image file
        image_files = glob(os.path.join(patient_dir, '*.npy'))
        if not image_files:
            raise FileNotFoundError(f"No image .npy files found in {patient_dir}")
        image_path = image_files[0]  # Use the first .npy file found
        image = np.load(image_path).astype(np.float32)  # Shape: [H, W]

        # Per image normalization since their values can vary too much
        # Compute per-image mean and std
        mean = image.mean()
        std = image.std()
        if std < 1e-8:
            std = 1.0  # Avoid division by zero
        image = (image - mean) / std

        # # TODO
        # # Visualize the 2D image
        # plt.figure()
        # plt.imshow(image, cmap='gray')
        # plt.title('Original 2D Image')
        # plt.axis('off')
        # plt.show()

        # Resize image to the desired size
        image = self.resize_image(image, self.image_size)

        # TODO: check for all possibilities
        # Convert image to 3 channels
        if image.ndim == 2:
            # Duplicate the single channel to create a 3-channel image
            image = np.stack([image] * 3, axis=-1)  # Shape: [H, W, 3]
        elif image.shape[2] == 1:
            # If image has a singleton channel dimension
            image = np.concatenate([image] * 3, axis=2)  # Shape: [H, W, 3]
        elif image.shape[2] != 3:
            raise ValueError(f"Unexpected number of channels in image: {image.shape[2]}")

        # #TODO
        # # Visualize the 3D image before normalization
        # plt.figure()
        # image_display = image.copy()
        # plt.imshow(image_display)
        # plt.title('3-Channel Image Before Normalization')
        # plt.axis('off')
        # plt.show()

        # Normalize image data
        # image = (image - self.image_mean) / self.image_std  # Now shape is [H, W, 3]

        # #TODO
        # plt.figure()
        # plt.imshow(image)
        # plt.title('3-Channel Image After Normalization')
        # plt.axis('off')
        # plt.show()

        # Transpose image to [C, H, W] for PyTorch
        image = np.transpose(image, (2, 0, 1))  # Shape: [3, H, W]

        # Convert to torch tensors
        image = th.from_numpy(image)  # Shape: [3, H, W]

        # TABULAR PART
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
        # tabular_data = np.array(tabular_data, dtype=np.float32)
        tabular_data = np.array(tabular_data, dtype=np.float32).reshape(1, -1)

        # Normalize tabular data
        # tabular_data = (tabular_data - self.tabular_mean) / self.tabular_std
        tabular_data = self.tabular_scaler.transform(tabular_data).flatten()

        # Convert tabular data to torch tensor
        tabular_data = th.from_numpy(tabular_data)

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

        # Collect all tabular data
        # all_tabular_data = []
        # for patient_dir in self.patient_dirs:
        #     json_files = glob(os.path.join(patient_dir, '*.json'))
        #     if not json_files:
        #         continue
        #     json_path = json_files[0]
        #     with open(json_path, 'r') as f:
        #         json_data = json.load(f)
        #     tabular_data = json_data.get('patient_id', list(json_data.values())[0])
        #     if not tabular_data:
        #         continue
        #     all_tabular_data.append(tabular_data)
        # if not all_tabular_data:
        #     raise ValueError("No tabular data found in any patient directories.")
        # all_tabular_data = np.array(all_tabular_data, dtype=np.float32)
        # mean = np.mean(all_tabular_data, axis=0)
        # std = np.std(all_tabular_data, axis=0)
        # std[std == 0] = 1.0  # Prevent division by zero
        # return mean, std

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

        # Fit the StandardScaler on the collected tabular data
        self.tabular_scaler.fit(all_tabular_data)

    def compute_image_normalization(self):

        # Collect all image data
        all_image_pixels = []
        for patient_dir in self.patient_dirs:
            image_files = glob(os.path.join(patient_dir, '*.npy'))
            if not image_files:
                continue
            image_path = image_files[0]
            image = np.load(image_path).astype(np.float32)
            # Resize image
            image = self.resize_image(image, self.image_size)
            # Convert to 3 channels
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)  # Shape: [H, W, 3]
            elif image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=2)  # Shape: [H, W, 3]
            elif image.shape[2] != 3:
                raise ValueError(f"Unexpected number of channels in image: {image.shape[2]}")
            all_image_pixels.append(image.reshape(-1, 3))  # Shape: [num_pixels, 3]
        if not all_image_pixels:
            raise ValueError("No image data found in any patient directories.")
        all_image_pixels = np.concatenate(all_image_pixels, axis=0)  # Shape: [total_pixels, 3]
        mean = np.mean(all_image_pixels, axis=0)  # Mean per channel
        std = np.std(all_image_pixels, axis=0)  # Std per channel
        # Prevent division by zero
        std[std == 0] = 1.0
        return mean, std

def load_training_data(args):
    dataset = ImageTabularDataset(args.data_dir, image_size=(64, 64))
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
            model, device_ids=[dist_util.dev()], output_device=dist_util.dev(),
            find_unused_parameters=True
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
