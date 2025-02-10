import json
import torch as th
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from torch.utils.data.distributed import DistributedSampler
from multi_modal_diffusion import dist_util
import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import DictConfig
from enums.latent_type import LatentType
import torch.distributed as dist

from diffusion_process.enums import DatasetType

# def load_training_data(args):
#     """
#     Load training data based on the chosen dataset type.
#     `args.dataset_type` should be a value from DatasetType enum.
#     Additional dataset-specific parameters can be included in args.
#     """
#
#     # You can add dataset-specific argument parsing or defaults in your config if needed
#     # For example, args could have fields like args.image_size, args.max_samples, etc.
#
#     if args.dataset_type == DatasetType.NACC:
#         dataset = NaccDataset(
#             data_dir=args.data_dir,
#             image_size=(args.image_height, args.image_width) if hasattr(args, 'image_height') and hasattr(args, 'image_width')
#             else (64, 64)
#         )
#     elif args.dataset_type == DatasetType.NACC_LATENTS:
#         dataset = NaccLatentsDataset(
#             data_dir=args.data_dir,
#             image_size=(args.image_height, args.image_width) if hasattr(args, 'image_height') and hasattr(args,'image_width')
#             else (64, 64)
#         )
#     elif args.dataset_type == DatasetType.TOY_MNIST:
#         dataset = ToyMNISTDataset(
#             data_dir=args.data_dir,
#             resize_to=(args.toy_resize_height, args.toy_resize_width) if hasattr(args, 'toy_resize_height') else (32, 32)
#         )
#     elif args.dataset_type == DatasetType.EXP_LUMIR:
#         dataset = ExpLumirDataset(
#             data_dir=args.data_dir,
#             image_size=(args.image_height, args.image_width) if hasattr(args, 'image_height') else (64, 64),
#             max_samples=args.max_samples if hasattr(args, 'max_samples') else None
#         )
#     elif args.dataset_type == DatasetType.LDM_ONE_H:
#         dataset = LDMOneHDataset(
#             data_dir=args.data_dir,
#             image_size=(args.image_height, args.image_width) if hasattr(args, 'image_height') else (64, 64),
#             # modality=args.modality if hasattr(args, 'modality') else "tabular"
#             modality=args.modality if hasattr(args, 'modality') else "image"
#         )
#     else:
#         raise ValueError(f"Unsupported dataset type: {args.dataset_type}")
#
#     sampler = DistributedSampler(dataset, shuffle=False, drop_last=True) if dist_util.get_world_size() > 1 else None
#     data_loader = th.utils.data.DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         drop_last=True,
#         sampler=sampler,
#     )
#     return data_loader

# TODO: BEFORE DDP
# def load_training_data(cfg: DictConfig):
#     """
#     Creates a DataLoader for the NACC dataset (or other possible sets)
#     depending on Hydra config values in cfg.data and cfg.model.
#     """
#
#     dataset_type = cfg.data.dataset_type.lower()
#
#     if dataset_type == "nacc":
#         # Decide final image range based on VAE choice
#         # e.g. "sd_xl" => we want [-1,1], or "none" => no shift
#         vae_name = cfg.vae.model_alias
#         if vae_name == LatentType.SD_XL.value:
#             final_range = "minus1to1"
#         else:
#             final_range = "none"
#             # or "0to1", depending on your preference
#
#         dataset = NaccDataset(
#             data_dir=cfg.data.data_dir,
#             image_height=cfg.data.image_height,
#             image_width=cfg.data.image_width,
#             domain=cfg.data.domain,  # "mri" or "ct"
#             do_augment=cfg.data.do_augment,
#             do_image_normalize=cfg.data.do_image_normalize,
#             do_tabular_normalize=cfg.data.do_tabular_normalize,
#             target_channels=cfg.vae.target_channels,
#             final_image_range=final_range,
#             debug=cfg.data.debug
#         )
#     else:
#         raise ValueError(f"Unsupported dataset type: {dataset_type}")
#
#     sampler = DistributedSampler(dataset, shuffle=False, drop_last=True) if dist_util.get_world_size() > 1 else None
#     loader = DataLoader(
#         dataset=dataset,
#         batch_size=cfg.training.batch_size,
#         num_workers=cfg.training.num_workers,
#         pin_memory=True,
#         drop_last=True,
#         sampler=sampler
#     )
#     return loader

def load_training_data(cfg: DictConfig):
    """
    Create a DataLoader for your dataset.
    If multiple GPUs are used (DDP), then we use a DistributedSampler.
    """
    dataset_type = cfg.data.dataset_type.lower()

    vae_name = cfg.vae.model_alias
    if vae_name == LatentType.SD_XL.value:
        final_range = "minus1to1"
    else:
        final_range = "none"
        # or "0to1", depending on your preference

    if dataset_type == "nacc":
        dataset = NaccDataset(
                        data_dir=cfg.data.data_dir,
                        image_height=cfg.data.image_height,
                        image_width=cfg.data.image_width,
                        domain=cfg.data.domain,  # "mri" or "ct"
                        do_augment=cfg.data.do_augment,
                        do_image_normalize=cfg.data.do_image_normalize,
                        do_tabular_normalize=cfg.data.do_tabular_normalize,
                        target_channels=cfg.vae.target_channels,
                        final_image_range=final_range,
                        debug=cfg.data.debug
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Condition for distributed
    world_size = 1
    rank = 0
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    # If we have multiple processes, wrap in DistributedSampler
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    else:
        sampler = None

    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        shuffle=(sampler is None)  # only shuffle if not using a distributed sampler
    )
    return loader

import os
import json
import numpy as np
import torch as th
import random
import cv2
from glob import glob
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# def debug_getitem(func):
#     """
#     Decorator to optionally visualize the image for debugging
#     (only on the first call if self.debug == True).
#     """
#     def wrapper(self, idx):
#         data = func(self, idx)
#         if self.debug and not self._debug_shown:
#             image = data["image"]  # shape [C,H,W], torch tensor
#             self._debug_show_image(image)
#             self._debug_shown = True
#         return data
#     return wrapper

class NaccDataset(Dataset):
    """
    Example dataset for NACC data. Each patient directory:
      - One .npy image file (2D slice)
      - One .json with tabular data
    """
    _debug_shown_global = False

    def __init__(
        self,
        data_dir: str,
        image_height: int = 512,
        image_width: int = 512,
        domain: str = "mri",              # "mri" or "ct" for domain-specific normalization
        do_augment: bool = False,
        do_image_normalize: bool = True,
        do_tabular_normalize: bool = True,
        target_channels: int = 4,
        final_image_range: str = "none",   # "none", "0to1", or "minus1to1"
        debug: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_height = image_height
        self.image_width = image_width
        self.domain = domain.lower()      # e.g. "mri", "ct"
        self.do_augment = do_augment
        self.do_image_normalize = do_image_normalize
        self.do_tabular_normalize = do_tabular_normalize
        self.target_channels = target_channels
        self.final_image_range = final_image_range

        # Debug settings
        self.debug = debug
        self._debug_shown = False  # to ensure we only debug-show once

        self.patient_dirs = [
            os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        ]

        if not self.patient_dirs:
            raise ValueError(f"No patient directories in {data_dir}")

        # Scalers
        self.tabular_scaler = StandardScaler()
        self.image_mean = 0.0
        self.image_std = 1.0

        # Precompute stats if requested
        # TODO: in distributed setting each rank sees each portion of data, so in future handle this for global mean etc
        if do_tabular_normalize or do_image_normalize:
            self._compute_normalization()

    def __len__(self):
        return len(self.patient_dirs)

    # @debug_getitem
    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]

        # 1) Load image
        image_path = self._get_first_file(patient_dir, "*.npy")
        image = np.load(image_path).astype(np.float32)  # shape: [H, W] (2D slice)

        # 2) Domain-specific normalization (mocked)
        image = self._domain_specific_normalization(image, self.domain)

        # 3) Resize/Pad
        image = self._resize_or_pad(image)

        # 4) Convert channels
        image = self._set_num_channels(image)

        # 5) Global mean/std
        if self.do_image_normalize:
            image = (image - self.image_mean) / (self.image_std + 1e-7)

        # 6) Final range
        image = self._map_final_range(image, self.final_image_range)

        # 7) Data augmentation
        if self.do_augment:
            # Put a placeholder for albumentations or other library
            image = self._augment_image(image)

        # 8) Convert to torch tensor
        image_tensor = th.from_numpy(image).float()

        # 9) Debug visualization
        if self.debug and not NaccDataset._debug_shown_global:
            # show only if rank 0
            rank = 0
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            if rank == 0:
                self._debug_show_image(image_tensor)
            NaccDataset._debug_shown_global = True

        # 10) Load tabular
        json_path = self._get_first_file(patient_dir, "*.json")
        with open(json_path, 'r') as f:
            jdata = json.load(f)

        # Extract the tabular array
        tab = jdata.get("patient_id", list(jdata.values())[0])
        tab = np.array(tab, dtype=np.float32)

        # Replace sentinel values here too (important!)
        tab = np.where(tab > 9999, -1, tab)

        # Reshape so scaler expects shape [1, num_features]
        tab = tab.reshape(1, -1)

        # Tabular normalization
        if self.do_tabular_normalize:
            tab = self.tabular_scaler.transform(tab)

        # Future augmentation for tabular
        if self.do_augment:
            tab = self._augment_tabular(tab)

        # Convert to torch
        tab_tensor = th.from_numpy(tab).squeeze(0)  # shape [features]
        return {"image": image_tensor, "tabular": tab_tensor}

    def _debug_show_image(self, image_tensor):
        """
        Show the image using matplotlib for a quick debug.
        We'll handle up to 3 channels for visualization.
        """
        # image_tensor: shape [C,H,W]
        image_np = image_tensor.cpu().numpy()
        C, H, W = image_np.shape
        if C == 1:
            # grayscale
            to_show = image_np[0]
            plt.imshow(to_show, cmap="gray")
        elif C >= 3:
            # show first 3 channels as RGB
            # to_show = image_np[:3]  # shape [3,H,W]
            # # to_show = np.transpose(to_show, (1, 2, 0))  # [H,W,3]
            # plt.imshow(to_show)
            to_show = image_np[0]
            plt.imshow(to_show, cmap="gray")
            pass
        else:
            # fallback: just show first channel
            to_show = image_np[0]
            plt.imshow(to_show, cmap="gray")

        plt.title("Debug: Transformed Image")
        plt.axis("off")
        plt.show(block=True)  # block so we can see it

    def _get_first_file(self, directory, pattern):
        files = glob(os.path.join(directory, pattern))
        if not files:
            raise FileNotFoundError(f"No files found in {directory} with pattern {pattern}")
        return files[0]

    def _domain_specific_normalization(self, image, domain):
        # Mock code
        if domain == "ct":
            # Example: clip to [0, 2048], then divide by 2048
            # image = np.clip(image, 0, 2048) / 2048.0
            pass
        elif domain == "mri":
            # Example: maybe Z-score per-slice or clip outliers
            # image = self._clip_outliers(image, quantiles=(1, 99))
            pass
        return image

    def _resize_or_pad(self, img):
        """
        If the image is bigger than [image_height, image_width], shrink with cv2.
        Otherwise do symmetrical padding to keep it centered.
        """
        H, W = img.shape[:2]
        # If bigger, shrink
        if H > self.image_height or W > self.image_width:
            img = cv2.resize(img, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        else:
            # symmetrical padding
            delta_h = self.image_height - H
            delta_w = self.image_width - W
            pad_top = delta_h // 2
            pad_bottom = delta_h - pad_top
            pad_left = delta_w // 2
            pad_right = delta_w - pad_left

            img = np.pad(
                img,
                pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0
            )
            # final check in case we had odd/even differences
            if img.shape[0] != self.image_height or img.shape[1] != self.image_width:
                img = cv2.resize(img, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        return img

    def _set_num_channels(self, img):
        if len(img.shape) == 2:  # [H, W]
            # Expand to [C, H, W]
            img = img[None, ...]  # => [1, H, W]
        C, H, W = img.shape
        if C == self.target_channels:
            return img
        if C > self.target_channels:
            return img[:self.target_channels, :, :]
        # if C < self.target_channels
        repeats = self.target_channels // C
        remainder = self.target_channels % C
        img = np.concatenate([img]*repeats, axis=0)
        if remainder:
            img = np.concatenate([img[:remainder, :, :], img], axis=0)
        return img

    def _map_final_range(self, image, frange):
        """
        Possible range shifts:
          - none: do nothing
          - 0to1: clamp to [0,1]
          - minus1to1: force to [0,1] then shift to [-1,1]
        """
        if frange == "none":
            return image
        elif frange == "0to1":
            return np.clip(image, 0.0, 1.0)
        elif frange == "minus1to1":
            # If standardization produced negative values, you could skip re-mapping.
            # Or forcibly min-max scale to [0,1], then shift to [-1,1].
            # We'll do the latter as an example:
            mn, mx = image.min(), image.max()
            denom = max(mx - mn, 1e-7)
            image = (image - mn) / denom  # => [0,1]
            image = image*2.0 - 1.0       # => [-1,1]
            return image
        else:
            return image

    def _augment_image(self, image):
        """
        Placeholder for image augmentation with albumentations or others.
        For now, just do a random horizontal flip.
        """
        if random.random() < 0.5:
            # Flip along width dimension
            image = np.flip(image, axis=2)
        return image

    def _augment_tabular(self, tab):
        """
        Placeholder for future tabular augmentation
        (e.g., add noise, or do nothing).
        """
        return tab

    def _compute_normalization(self):
        """
        One pass to gather tabular rows and image pixels for computing
        global mean/std. (Might be memory-heavy for large data.)
        """
        tabular_rows = []
        image_pixels = []

        for pdir in self.patient_dirs:
            # ---- Load tabular data ----
            jpath = self._get_first_file(pdir, "*.json")
            with open(jpath, 'r') as f:
                jdata = json.load(f)
            row = jdata.get("patient_id", list(jdata.values())[0])

            # Convert to numpy
            row = np.array(row, dtype=np.float32)

            # Replace sentinel > 9999 with -1 (or any reasonable placeholder)
            row = np.where(row > 9999, -1, row)

            # Reshape to (1, -1)
            row = row.reshape(1, -1)
            tabular_rows.append(row)

            # ---- Load image data ----
            ipath = self._get_first_file(pdir, "*.npy")
            img = np.load(ipath).astype(np.float32)
            # domain-specific steps if you wish (clip, etc.)
            image_pixels.append(img.flatten())

        if self.do_tabular_normalize:
            big_tab = np.concatenate(tabular_rows, axis=0)  # shape [N, feats]
            self.tabular_scaler.fit(big_tab)

        if self.do_image_normalize:
            all_pixels = np.concatenate(image_pixels, axis=0)
            self.image_mean = float(all_pixels.mean())
            self.image_std = float(all_pixels.std(ddof=1))  # unbiased


# class NaccDataset(Dataset):
#     def __init__(self, data_dir, image_size=(256, 256)):
#         self.data_dir = data_dir
#         self.image_size = (64,64)
#         # self.image_size = image_size  # Desired image size (width, height)
#         # Get list of patient directories
#         self.patient_dirs = [
#             os.path.join(data_dir, d) for d in os.listdir(data_dir)
#             if os.path.isdir(os.path.join(data_dir, d))
#         ]
#         if not self.patient_dirs:
#             raise ValueError(f"No patient directories found in {data_dir}")
#
#         # Initialize StandardScaler for tabular data
#         self.tabular_scaler = StandardScaler()
#
#         # Compute normalization parameters
#         # self.tabular_mean, self.tabular_std = self.compute_tabular_normalization()
#         self.compute_tabular_normalization()
#         self.image_mean, self.image_std = self.compute_image_normalization()
#
#     def __len__(self):
#         return len(self.patient_dirs)
#
#     def __getitem__(self, idx):
#
#         patient_dir = self.patient_dirs[idx]
#
#         # IMAGE PART
#         # Find image file
#         image_files = glob(os.path.join(patient_dir, '*.npy'))
#         if not image_files:
#             raise FileNotFoundError(f"No image .npy files found in {patient_dir}")
#         image_path = image_files[0]  # Use the first .npy file found
#         image = np.load(image_path).astype(np.float32)  # Shape: [H, W]
#
#         # Per image normalization since their values can vary too much
#         # Compute per-image mean and std
#         mean = image.mean()
#         std = image.std()
#         if std < 1e-8:
#             std = 1.0  # Avoid division by zero
#         image = (image - mean) / std
#
#         # # TODO
#         # import matplotlib.pyplot as plt
#         # # Visualize the 2D image
#         # plt.figure()
#         # plt.imshow(image, cmap='gray')
#         # plt.title('Original 2D Image')
#         # plt.axis('off')
#         # plt.show()
#
#         # Resize image to the desired size
#         image = self.resize_image(image, self.image_size)
#
#         # TODO: check for all possibilities
#         # Convert image to 3 channels
#         if image.ndim == 2:
#             # Duplicate the single channel to create a 3-channel image
#             image = np.stack([image] * 3, axis=-1)  # Shape: [H, W, 3]
#         elif image.shape[2] == 1:
#             # If image has a singleton channel dimension
#             image = np.concatenate([image] * 3, axis=2)  # Shape: [H, W, 3]
#         elif image.shape[2] != 3:
#             raise ValueError(f"Unexpected number of channels in image: {image.shape[2]}")
#
#         # #TODO
#         # # Visualize the 3D image before normalization
#         # plt.figure()
#         # image_display = image.copy()
#         # plt.imshow(image_display)
#         # plt.title('3-Channel Image Before Normalization')
#         # plt.axis('off')
#         # plt.show()
#
#         # Normalize image data
#         # image = (image - self.image_mean) / self.image_std  # Now shape is [H, W, 3]
#
#         # #TODO
#         # plt.figure()
#         # plt.imshow(image)
#         # plt.title('3-Channel Image After Normalization')
#         # plt.axis('off')
#         # plt.show()
#
#         # Transpose image to [C, H, W] for PyTorch
#         image = np.transpose(image, (2, 0, 1))  # Shape: [3, H, W]
#
#         # Convert to torch tensors
#         image = th.from_numpy(image)  # Shape: [3, H, W]
#
#         # TABULAR PART
#         # Find JSON file
#         json_files = glob(os.path.join(patient_dir, '*.json'))
#         if not json_files:
#             raise FileNotFoundError(f"No JSON files found in {patient_dir}")
#         json_path = json_files[0]  # Use the first .json file found
#         with open(json_path, 'r') as f:
#             json_data = json.load(f)
#
#         # Get the tabular data
#         # Try 'patient_id' key; if not present, use the first value
#         tabular_data = json_data.get('patient_id', list(json_data.values())[0])
#         if not tabular_data:
#             raise ValueError(f"No tabular data found in {json_path}")
#         # tabular_data = np.array(tabular_data, dtype=np.float32)
#         tabular_data = np.array(tabular_data, dtype=np.float32).reshape(1, -1)
#
#         # Normalize tabular data
#         # tabular_data = (tabular_data - self.tabular_mean) / self.tabular_std
#         tabular_data = self.tabular_scaler.transform(tabular_data).flatten()
#
#         # Convert tabular data to torch tensor
#         tabular_data = th.from_numpy(tabular_data)
#
#         return {'image': image, 'tabular': tabular_data}
#
#     def resize_image(self, image, size):
#         # Convert numpy array to PIL Image
#         pil_image = Image.fromarray(image)
#         # Resize image
#         pil_image = pil_image.resize(size[::-1], Image.BILINEAR)  # size[::-1] because PIL uses (width, height)
#         # Convert back to numpy array
#         image_resized = np.array(pil_image).astype(np.float32)
#         return image_resized
#
#     def compute_tabular_normalization(self):
#
#         # Collect all tabular data
#         all_tabular_data = []
#         for patient_dir in self.patient_dirs:
#             json_files = glob(os.path.join(patient_dir, '*.json'))
#             if not json_files:
#                 continue
#             json_path = json_files[0]
#             with open(json_path, 'r') as f:
#                 json_data = json.load(f)
#             tabular_data = json_data.get('patient_id', list(json_data.values())[0])
#             if not tabular_data:
#                 continue
#             all_tabular_data.append(tabular_data)
#         if not all_tabular_data:
#             raise ValueError("No tabular data found in any patient directories.")
#         all_tabular_data = np.array(all_tabular_data, dtype=np.float32)
#
#         # Fit the StandardScaler on the collected tabular data
#         self.tabular_scaler.fit(all_tabular_data)
#
#     def compute_image_normalization(self):
#
#         # Collect all image data
#         all_image_pixels = []
#         for patient_dir in self.patient_dirs:
#             image_files = glob(os.path.join(patient_dir, '*.npy'))
#             if not image_files:
#                 continue
#             image_path = image_files[0]
#             image = np.load(image_path).astype(np.float32)
#             # Resize image
#             image = self.resize_image(image, self.image_size)
#             # Convert to 3 channels
#             if image.ndim == 2:
#                 image = np.stack([image] * 3, axis=-1)  # Shape: [H, W, 3]
#             elif image.shape[2] == 1:
#                 image = np.concatenate([image] * 3, axis=2)  # Shape: [H, W, 3]
#             elif image.shape[2] != 3:
#                 raise ValueError(f"Unexpected number of channels in image: {image.shape[2]}")
#             all_image_pixels.append(image.reshape(-1, 3))  # Shape: [num_pixels, 3]
#         if not all_image_pixels:
#             raise ValueError("No image data found in any patient directories.")
#         all_image_pixels = np.concatenate(all_image_pixels, axis=0)  # Shape: [total_pixels, 3]
#         mean = np.mean(all_image_pixels, axis=0)  # Mean per channel
#         std = np.std(all_image_pixels, axis=0)  # Std per channel
#         # Prevent division by zero
#         std[std == 0] = 1.0
#         return mean, std



# NACC WITH LATENTS [1, 4, 64, 64]

class NaccLatentsDataset(Dataset):
    """
    A dataset for latents (shape [1, 4, H, W]) which we resize to a square
    and normalize to [-1, 1]. We also load optional JSON tabular data
    and scale it to [-1, 1] as well.

    Steps for latents:
      1) Load .npy of shape [1, 4, H, W].
      2) Remove the first dimension -> [4, H, W].
      3) (Optional) Resize to a fixed square (e.g. 64x64).
      4) Normalize latents to [-1, 1] using a global min/max across the dataset.
      5) Return latents as a torch float32 tensor of shape [4, outH, outW].

    Steps for tabular:
      1) Load JSON data if it exists.
      2) Collect all tabular data across dataset, compute global min/max per feature.
      3) Transform each feature into [-1, 1].
      4) Return as torch float32 tensor.
    """

    def __init__(self, data_dir, image_size=(64, 64)):
        """
        Args:
            data_dir (str): Root directory containing subfolders, each with .npy (latents) and possibly .json (tabular).
            image_size (tuple): (H, W) to resize the latents.
        """
        self.data_dir = data_dir
        self.image_size = image_size

        # List of patient directories
        self.patient_dirs = [
            os.path.join(data_dir, d)
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]
        if not self.patient_dirs:
            raise ValueError(f"No patient directories found in {data_dir}")

        # Create a MinMaxScaler to scale tabular features into [-1, 1]
        self.tabular_scaler = StandardScaler()

        # Fit the tabular scaler on all data (so each sample is scaled consistently)
        self.compute_tabular_normalization()

        # Min–max scalers for latents and tabular
        self.latent_min = float('inf')
        self.latent_max = float('-inf')
        # self.tabular_scaler = MinMaxScalerNeg1to1()

        # Pre-fit the scalers
        # self._compute_tabular_minmax()
        self._compute_latent_minmax()

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        """
        Returns a dict with:
          - 'image': shape [4, H, W] as a torch.Tensor (float32)
          - 'tabular': shape [N] as a torch.Tensor (float32), if found
        """
        patient_dir = self.patient_dirs[idx]

        # 1) Load latents => shape [1, 4, H, W]
        latents_path = self._find_npy_file(patient_dir)
        latents_np = np.load(latents_path)  # shape [1, 4, H, W]
        if latents_np.shape[0] != 1 or latents_np.shape[1] != 4:
            raise ValueError(f"Expected latents shape [1, 4, H, W], got {latents_np.shape}")

        # Drop first dimension => [4, H, W]
        latents_np = latents_np[0]

        # 2) Resize latents (uncomment if you do want resizing)
        # latents_np = self._resize_latents(latents_np)  # => [4, outH, outW]

        # 3) Normalize latents to [-1, 1]
        latents_np = self._minmax_normalize_latents(latents_np)

        # 4) Convert latents to torch tensor
        latents_torch = torch.from_numpy(latents_np.astype(np.float32))

        # 5) Load tabular data and scale to [-1, 1]
        # tabular_data = self._load_tabular_data(patient_dir)

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

        return {
            'image': latents_torch,  # [4, outH, outW]
            'tabular': tabular_data  # [N] or empty
        }

    def _find_npy_file(self, patient_dir):
        """Return the first .npy file found in a patient directory."""
        latents_files = glob(os.path.join(patient_dir, '*.npy'))
        if not latents_files:
            raise FileNotFoundError(f"No .npy file found in {patient_dir}")
        return latents_files[0]

    def _resize_latents(self, latents: np.ndarray) -> np.ndarray:
        """
        latents shape: [4, H, W].
        We want to resize to [4, image_size[0], image_size[1]].

        We'll:
          1) move channels last -> [H, W, 4],
          2) use PIL to resize to image_size,
          3) move channels back -> [4, image_size[0], image_size[1]].
        """
        # Move channels to last dimension: [H, W, 4]
        latents_ch_last = np.transpose(latents, (1, 2, 0))

        # Convert to PIL Image (must be float32, 8-bit, or others; float32 works)
        # But PIL expects channel dimension in [1,3,4]. We have 4, so this is okay
        # (RGBA interpretation if you consider them as an image).
        latents_pil = Image.fromarray(latents_ch_last)

        # Resize to image_size
        latents_pil = latents_pil.resize(self.image_size[::-1], Image.BILINEAR)

        # Convert back to numpy array, shape [out_H, out_W, 4]
        latents_resized_ch_last = np.array(latents_pil, dtype=np.float32)

        # Move channels to front: [4, out_H, out_W]
        latents_resized = np.transpose(latents_resized_ch_last, (2, 0, 1))
        return latents_resized

    def _compute_latent_minmax(self):
        """
        One pass to find the global min and max across all latents,
        so we can scale them consistently to [-1, 1].
        """
        for patient_dir in self.patient_dirs:
            latents_path = self._find_npy_file(patient_dir)
            arr = np.load(latents_path)  # [1, 4, H, W]
            arr = arr[0]  # [4, H, W]

            curr_min = arr.min()
            curr_max = arr.max()
            if curr_min < self.latent_min:
                self.latent_min = curr_min
            if curr_max > self.latent_max:
                self.latent_max = curr_max

        # Avoid divide-by-zero if min == max
        if self.latent_min == self.latent_max:
            self.latent_min -= 1e-6
            self.latent_max += 1e-6

    def _minmax_normalize_latents(self, latents: np.ndarray) -> np.ndarray:
        """
        Scale latents from [4, H, W] into [-1, 1] using global min/max.
        """
        return 2.0 * (latents - self.latent_min) / (self.latent_max - self.latent_min) - 1.0

    def compute_tabular_normalization(self):
        # """
        # Go through each patient directory, gather all tabular arrays, and
        # fit the StandardScaler.
        # """
        # all_tabular = []
        # for patient_dir in self.patient_dirs:
        #     json_files = glob(os.path.join(patient_dir, '*.json'))
        #     if not json_files:
        #         continue
        #     with open(json_files[0], 'r') as f:
        #         json_data = json.load(f)
        #     vals = json_data.get('patient_id', list(json_data.values())[0])
        #     vals = [0 if x >= 9000 else x for x in vals]
        #     if isinstance(vals, (list, tuple)):
        #         all_tabular.append(vals)
        #
        # if not all_tabular:
        #     # No tabular data found, fit on dummy to avoid errors
        #     self.tabular_scaler.fit([[0.0]])
        #     return
        #
        # all_tabular_np = np.array(all_tabular, dtype=np.float32)
        # self.tabular_scaler.fit(all_tabular_np)

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


    def _get_tabular_values(self, patient_dir):
        """Helper to load numeric tabular data as a list/1D array, or None if missing."""
        json_files = glob(os.path.join(patient_dir, '*.json'))
        if not json_files:
            return None
        with open(json_files[0], 'r') as f:
            data = json.load(f)

        # Suppose your numeric data is in data['patient_id'], or just the first values
        vals = data.get('patient_id', list(data.values())[0])
        vals = [0 if x >= 9000 else x for x in vals]

        # Must be a list/tuple of numbers
        if isinstance(vals, (list, tuple)):
            return vals
        return None

    def _load_tabular_data(self, patient_dir):
        """
        Load one row of tabular data and transform it to [-1, 1].
        Returns a torch.Tensor, shape [num_features].
        """
        row = self._get_tabular_values(patient_dir)
        if row is None:
            return torch.empty(0, dtype=torch.float32)

        row = np.array(row, dtype=np.float32).reshape(1, -1)  # shape [1, n_features]
        # row_scaled = self.tabular_scaler.transform(row)       # also shape [1, n_features]
        # return torch.from_numpy(row_scaled.flatten())
        return torch.from_numpy(row.flatten())


class MinMaxScalerNeg1to1:
    """
    Min-max scaler that scales each column into [-1, 1].
    Handles zero-range columns by assigning them to 0.
    Optionally can convert sentinel values > 9999 to NaN and then impute.
    """
    def __init__(self, sentinel_threshold=9999):
        self.min_ = None
        self.max_ = None
        self.sentinel_threshold = sentinel_threshold

    def fit(self, X):
        # X shape: [n_samples, n_features]
        # Optionally replace sentinel > 9999 with np.nan
        if self.sentinel_threshold is not None:
            X = np.where(X > self.sentinel_threshold, np.nan, X)

        # If columns can have NaN, decide how to handle it (drop or fill)
        for c in range(X.shape[1]):
            col = X[:, c]
            # Fill NaN with column mean (simple imputation)
            if np.all(np.isnan(col)):
                # Entire column is missing
                col[:] = 0  # or drop the column, or keep them as 0
            else:
                mean_val = np.nanmean(col)
                col[np.isnan(col)] = mean_val

        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)

        # If min == max for a column, that column is constant => nudge them
        no_range_mask = (self.max_ == self.min_)
        self.max_[no_range_mask] = self.min_[no_range_mask] + 1e-7

    def transform(self, X):
        if self.sentinel_threshold is not None:
            X = np.where(X > self.sentinel_threshold, np.nan, X)
        # Fill NaN again
        for c in range(X.shape[1]):
            col = X[:, c]
            mean_val = np.nanmean(col)
            col[np.isnan(col)] = mean_val

        # Perform min–max with the fitted min_/max_
        denom = (self.max_ - self.min_)
        scaled = 2.0 * (X - self.min_) / denom - 1.0
        return scaled



# TOY MNIST

class ToyMNISTDataset(Dataset):
    def __init__(self, data_dir, resize_to=(32, 32)):
        """
        Args:
            data_dir (str): Path to the directory containing subdirectories for each sample.
                            Each subdirectory should contain an image (as .png) and a corresponding tabular .json file.
            resize_to (tuple): Desired output size of the images (height, width).
        """
        self.data_dir = data_dir
        self.resize_to = resize_to

        # Get list of sample directories
        self.sample_dirs = [
            os.path.join(data_dir, d) for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]
        if not self.sample_dirs:
            raise ValueError(f"No sample directories found in {data_dir}")

        # Initialize StandardScaler for tabular data
        self.tabular_scaler = StandardScaler()

        # Compute tabular normalization parameters
        self.compute_tabular_normalization()

        # Initialize transformation for images
        self.image_transform = transforms.Compose([
            transforms.Resize(self.resize_to),   # Resize images to desired size
            transforms.ToTensor(),              # Convert PIL image to Tensor and scale pixel values to [0, 1]
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Replicate the grayscale channel 3 times
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images to range [-1, 1]
        ])

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        # Load image data
        image_files = glob(os.path.join(sample_dir, '*.png'))
        if not image_files:
            raise FileNotFoundError(f"No image .png files found in {sample_dir}")
        image_path = image_files[0]

        # Open image using PIL and convert to grayscale
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        # Apply transformations to the image
        image = self.image_transform(image)  # Shape: [3, H, W]

        # Load tabular data
        json_files = glob(os.path.join(sample_dir, '*.json'))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {sample_dir}")
        json_path = json_files[0]
        with open(json_path, 'r') as f:
            tabular_data = json.load(f)

        # Convert tabular data to numpy array
        tabular_values = np.array(list(tabular_data.values()), dtype=np.float32)
        tabular_values = self.tabular_scaler.transform(tabular_values.reshape(1, -1)).flatten()

        # Convert to torch tensor
        tabular_tensor = torch.tensor(tabular_values, dtype=torch.float32)

        return {'image': image, 'tabular': tabular_tensor}

    def compute_tabular_normalization(self):
        # Collect all tabular data
        all_tabular_data = []
        for sample_dir in self.sample_dirs:
            json_files = glob(os.path.join(sample_dir, '*.json'))
            if not json_files:
                continue
            json_path = json_files[0]
            with open(json_path, 'r') as f:
                tabular_data = json.load(f)
            all_tabular_data.append(list(tabular_data.values()))
        if not all_tabular_data:
            raise ValueError("No tabular data found in any sample directories.")
        all_tabular_data = np.array(all_tabular_data, dtype=np.float32)

        # Fit the scaler
        self.tabular_scaler.fit(all_tabular_data)



# TOY MNIST

class ToyMNISTDataset(Dataset):
    def __init__(self, data_dir, resize_to=(32, 32)):
        """
        Args:
            data_dir (str): Path to the directory containing subdirectories for each sample.
                            Each subdirectory should contain an image (as .png) and a corresponding tabular .json file.
            resize_to (tuple): Desired output size of the images (height, width).
        """
        self.data_dir = data_dir
        self.resize_to = resize_to

        # Get list of sample directories
        self.sample_dirs = [
            os.path.join(data_dir, d) for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]
        if not self.sample_dirs:
            raise ValueError(f"No sample directories found in {data_dir}")

        # Initialize StandardScaler for tabular data
        self.tabular_scaler = StandardScaler()

        # Compute tabular normalization parameters
        self.compute_tabular_normalization()

        # Initialize transformation for images
        self.image_transform = transforms.Compose([
            transforms.Resize(self.resize_to),   # Resize images to desired size
            transforms.ToTensor(),              # Convert PIL image to Tensor and scale pixel values to [0, 1]
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Replicate the grayscale channel 3 times
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images to range [-1, 1]
        ])

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        # Load image data
        image_files = glob(os.path.join(sample_dir, '*.png'))
        if not image_files:
            raise FileNotFoundError(f"No image .png files found in {sample_dir}")
        image_path = image_files[0]

        # Open image using PIL and convert to grayscale
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        # Apply transformations to the image
        image = self.image_transform(image)  # Shape: [3, H, W]

        # Load tabular data
        json_files = glob(os.path.join(sample_dir, '*.json'))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {sample_dir}")
        json_path = json_files[0]
        with open(json_path, 'r') as f:
            tabular_data = json.load(f)

        # Convert tabular data to numpy array
        tabular_values = np.array(list(tabular_data.values()), dtype=np.float32)
        tabular_values = self.tabular_scaler.transform(tabular_values.reshape(1, -1)).flatten()

        # Convert to torch tensor
        tabular_tensor = torch.tensor(tabular_values, dtype=torch.float32)

        return {'image': image, 'tabular': tabular_tensor}

    def compute_tabular_normalization(self):
        # Collect all tabular data
        all_tabular_data = []
        for sample_dir in self.sample_dirs:
            json_files = glob(os.path.join(sample_dir, '*.json'))
            if not json_files:
                continue
            json_path = json_files[0]
            with open(json_path, 'r') as f:
                tabular_data = json.load(f)
            all_tabular_data.append(list(tabular_data.values()))
        if not all_tabular_data:
            raise ValueError("No tabular data found in any sample directories.")
        all_tabular_data = np.array(all_tabular_data, dtype=np.float32)

        # Fit the scaler
        self.tabular_scaler.fit(all_tabular_data)


# EXP LUMIR

import random

class ExpLumirDataset(Dataset):
    def __init__(self, data_dir, image_size=(128, 128), max_samples=None, drop_last=True):
        """
        Args:
            data_dir (str): Path to the directory containing sample subdirectories.
            image_size (tuple): Desired (H, W) size for the final 2D slice image.
            max_samples (int, optional): Maximum number of random samples to include. If None, use all samples.
        """
        self.data_dir = data_dir
        self.image_size = image_size

        # Get list of sample directories
        self.sample_dirs = [
            os.path.join(data_dir, d) for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]
        if not self.sample_dirs:
            raise ValueError(f"No sample directories found in {data_dir}")

        # If max_samples is specified, randomly select a subset
        if max_samples is not None:
            if max_samples > len(self.sample_dirs):
                raise ValueError(f"max_samples ({max_samples}) cannot exceed total samples ({len(self.sample_dirs)})")
            self.sample_dirs = random.sample(self.sample_dirs, max_samples)

        # Initialize StandardScaler for tabular data
        self.tabular_scaler = StandardScaler()
        self.compute_tabular_normalization()

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        # Load the NIfTI file
        nii_files = glob(os.path.join(sample_dir, '*.nii*'))
        if not nii_files:
            raise FileNotFoundError(f"No NIfTI files found in {sample_dir}")
        nii_path = nii_files[0]
        img_nii = nib.load(nii_path)
        img_data = img_nii.get_fdata(dtype=np.float32)  # shape: [D, W, H]

        if img_data.ndim != 3:
            raise ValueError(f"Expected a 3D MRI volume, got shape {img_data.shape}")

        # Select the middle slice along the D dimension (dimension 0)
        mid_slice_idx = img_data.shape[0] // 2
        img_2d = img_data[mid_slice_idx, ...]  # shape: [W, H]

        # Currently, img_2d is [W, H], we want [H, W]
        img_2d = img_2d.T  # Now [H, W]

        # Resize the 2D slice
        img_2d_resized = self.resize_image(img_2d, self.image_size)

        # Normalize (z-score) after resizing
        mean_val = img_2d_resized.mean()
        std_val = img_2d_resized.std()
        if std_val > 1e-6:
            img_2d_resized = (img_2d_resized - mean_val) / std_val
        else:
            img_2d_resized = img_2d_resized - mean_val

        # Replicate the grayscale channel 3 times to get shape [3, H, W]
        img_tensor = torch.tensor(img_2d_resized, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

        # Load tabular data
        json_files = glob(os.path.join(sample_dir, '*.json'))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {sample_dir}")
        json_path = json_files[0]
        with open(json_path, 'r') as f:
            tabular_data = json.load(f)

        # Convert tabular data to array and normalize
        tabular_values = np.array(list(tabular_data.values()), dtype=np.float32)
        tabular_values = self.tabular_scaler.transform(tabular_values.reshape(1, -1)).flatten()
        tabular_tensor = torch.tensor(tabular_values, dtype=torch.float32)

        return {'image': img_tensor, 'tabular': tabular_tensor}

    def compute_tabular_normalization(self):
        # Collect tabular data to fit scaler
        all_tabular_data = []
        for sample_dir in self.sample_dirs:
            json_files = glob(os.path.join(sample_dir, '*.json'))
            if not json_files:
                continue
            json_path = json_files[0]
            with open(json_path, 'r') as f:
                tabular_data = json.load(f)
            all_tabular_data.append(list(tabular_data.values())[0])
        if not all_tabular_data:
            raise ValueError("No tabular data found in any sample directories.")
        all_tabular_data = np.array(all_tabular_data, dtype=np.float32)
        self.tabular_scaler.fit(all_tabular_data)

    def resize_image(self, image, size):
        # image: 2D numpy array [H, W]
        # size: (H, W) desired
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize(size[::-1], Image.BILINEAR)  # Note: (W, H) for PIL
        image_resized = np.array(pil_image, dtype=np.float32)
        return image_resized



# LDMOneH Dataset

# class LDMOneHDataset(Dataset):
#     def __init__(self, data_dir, image_size=(128, 128), max_samples=None, drop_last=True):
#         """
#         Args:
#             data_dir (str): Path to the directory containing sample subdirectories.
#             image_size (tuple): Desired (H, W) size for the final 2D slice image.
#             max_samples (int, optional): Maximum number of random samples to include. If None, use all samples.
#         """
#         self.data_dir = data_dir
#         self.image_size = image_size
#
#         # Get list of 'anat' directories under each patient folder
#         self.sample_dirs = []
#         rawdata_path = os.path.join(self.data_dir, "rawdata")
#         for patient_folder in os.listdir(rawdata_path):
#             anat_path = os.path.join(rawdata_path, patient_folder, "anat")
#             if os.path.isdir(anat_path):
#                 self.sample_dirs.append(anat_path)
#
#         if not self.sample_dirs:
#             raise ValueError(f"No sample directories found in {data_dir}")
#
#         # If max_samples is specified, randomly select a subset
#         if max_samples is not None:
#             if max_samples > len(self.sample_dirs):
#                 raise ValueError(f"max_samples ({max_samples}) cannot exceed total samples ({len(self.sample_dirs)})")
#             self.sample_dirs = random.sample(self.sample_dirs, max_samples)
#
#     def __len__(self):
#         return len(self.sample_dirs)
#
#     def __getitem__(self, idx):
#         sample_dir = self.sample_dirs[idx]
#
#         # Load the NIfTI file
#         nii_files = glob(os.path.join(sample_dir, '*.nii*'))
#         if not nii_files:
#             raise FileNotFoundError(f"No NIfTI files found in {sample_dir}")
#         nii_path = nii_files[0]
#         img_nii = nib.load(nii_path)
#         img_data = img_nii.get_fdata(dtype=np.float32)  # shape: [D, W, H]
#
#         if img_data.ndim != 3:
#             raise ValueError(f"Expected a 3D MRI volume, got shape {img_data.shape}")
#
#         # Select the middle slice along the D dimension (dimension 0)
#         mid_slice_idx = img_data.shape[0] // 2
#         img_2d = img_data[mid_slice_idx, ...]  # shape: [W, H]
#
#         # import matplotlib.pyplot as plt
#         # plt.imshow(img_data[:, :, mid_slice_idx], cmap="gray")
#         # plt.show()
#
#         # Currently, img_2d is [W, H], we want [H, W]
#         img_2d = img_2d.T  # Now [H, W]
#
#         # Resize the 2D slice
#         img_2d_resized = self.resize_image(img_2d, self.image_size)
#
#         # Normalize (z-score) after resizing
#         mean_val = img_2d_resized.mean()
#         std_val = img_2d_resized.std()
#         if std_val > 1e-6:
#             img_2d_resized = (img_2d_resized - mean_val) / std_val
#         else:
#             img_2d_resized = img_2d_resized - mean_val
#
#         # Replicate the grayscale channel 3 times to get shape [3, H, W]
#         img_tensor = torch.tensor(img_2d_resized, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
#
#         #img_tensor = np.transpose(img_tensor, (1, 2, 0))
#
#         # Fake tabular data: 128 zeros
#         tabular_tensor = torch.zeros(128, dtype=torch.float32)
#
#         return {'image': img_tensor, 'tabular': tabular_tensor}
#
#     def resize_image(self, image, size):
#         # image: 2D numpy array [H, W]
#         # size: (H, W) desired
#         pil_image = Image.fromarray(image)
#         pil_image = pil_image.resize(size[::-1], Image.BILINEAR)  # Note: (W, H) for PIL
#         image_resized = np.array(pil_image, dtype=np.float32)
#         return image_resized
#
#     # def resize_image(self, image, size):
#     #     """
#     #     Resize the image while maintaining aspect ratio.
#     #
#     #     Parameters:
#     #     - image: 2D numpy array [H, W]
#     #     - size: tuple (longer_side_length, _), e.g., (64, 64)
#     #
#     #     Returns:
#     #     - image_resized: 2D numpy array resized with the longer side equal to size[0]
#     #     """
#     #     # Extract original dimensions
#     #     height, width = image.shape
#     #
#     #     # Desired length for the longer side
#     #     longer_side = size[0]
#     #
#     #     # Determine the scaling factor and new dimensions
#     #     if width > height:
#     #         new_width = longer_side
#     #         new_height = int(round((height / width) * longer_side))
#     #     else:
#     #         new_height = longer_side
#     #         new_width = int(round((width / height) * longer_side))
#     #
#     #     # Convert to PIL Image for resizing
#     #     pil_image = Image.fromarray(image)
#     #
#     #     # Resize with the new dimensions
#     #     pil_image = pil_image.resize((new_width, new_height), Image.BILINEAR)
#     #
#     #     # Convert back to numpy array
#     #     image_resized = np.array(pil_image, dtype=np.float32)
#     #
#     #     return image_resized
#
#
#



class LDMOneHDataset(Dataset):
    def __init__(self, data_dir, image_size=(128, 128), max_samples=None, drop_last=True, modality='image'):
        """
        Args:
            data_dir (str): Path to the directory containing patient folders with 'anat' subdirectories.
            image_size (tuple): Desired (H, W) size for the final 2D slice image.
            max_samples (int, optional): Maximum number of random samples to include. If None, use all samples.
            modality (str): 'image' or 'tabular'.
                            'image': load and normalize images, generate fake tabular zeros.
                            'tabular': load and normalize tabular data, generate fake zero images.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.modality = modality

        self.sample_dirs = []

        # behave accordingly for images or tabular
        common_path = os.path.join(self.data_dir, "rawdata") if modality == 'image' else os.path.join(self.data_dir, "tabular_gen_vfa_net")
        # rawdata_path = os.path.join(self.data_dir, "rawdata")

        if modality == 'image':
            # Get list of 'anat' directories under each patient folder
            for patient_folder in os.listdir(common_path):
                anat_path = os.path.join(common_path, patient_folder, "anat")
                if os.path.isdir(anat_path):
                    self.sample_dirs.append(anat_path)

            if not self.sample_dirs:
                raise ValueError(f"No sample directories found in {data_dir}")
        elif modality == 'tabular':
            for patient_folder in os.listdir(common_path):
                patient_path = os.path.join(common_path, patient_folder)
                if os.path.isdir(patient_path):
                    self.sample_dirs.append(patient_path)

        # If max_samples is specified, randomly select a subset
        if max_samples is not None:
            if max_samples > len(self.sample_dirs):
                raise ValueError(f"max_samples ({max_samples}) cannot exceed total samples ({len(self.sample_dirs)})")
            self.sample_dirs = random.sample(self.sample_dirs, max_samples)

        # If in tabular mode, set up a scaler and compute normalization parameters
        if self.modality == 'tabular':
            self.tabular_scaler = StandardScaler()
            self.compute_tabular_normalization()

    def compute_tabular_normalization(self):
        # Collect all tabular data from JSON files to fit the scaler
        all_tabular_data = []
        for sample_dir in self.sample_dirs:
            json_files = glob(os.path.join(sample_dir, '*.json'))
            if not json_files:
                continue
            json_path = json_files[0]
            with open(json_path, 'r') as f:
                tabular_data = json.load(f)
            # Extract the values from the JSON dict
            values = np.array(*list(tabular_data.values()), dtype=np.float32)
            all_tabular_data.append(values)

        if not all_tabular_data:
            raise ValueError("No tabular data found in any sample directories.")

        all_tabular_data = np.array(all_tabular_data, dtype=np.float32)
        self.tabular_scaler.fit(all_tabular_data)

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        if self.modality == 'image':
            # IMAGE MODE: Load and normalize the image, produce fake tabular zeros
            # Load the NIfTI file
            nii_files = glob(os.path.join(sample_dir, '*.nii*'))
            if not nii_files:
                raise FileNotFoundError(f"No NIfTI files found in {sample_dir}")
            nii_path = nii_files[0]
            img_nii = nib.load(nii_path)
            img_data = img_nii.get_fdata(dtype=np.float32)  # shape: [D, W, H]

            if img_data.ndim != 3:
                raise ValueError(f"Expected a 3D MRI volume, got shape {img_data.shape}")

            # Select the middle slice along the D dimension
            mid_slice_idx = img_data.shape[0] // 2
            img_2d = img_data[mid_slice_idx, ...]  # shape: [W, H]

            # Currently, img_2d is [W, H], we want [H, W]
            img_2d = img_2d.T  # Now [H, W]

            # Resize the 2D slice
            img_2d_resized = self.resize_image(img_2d, self.image_size)

            # Normalize (z-score) after resizing
            mean_val = img_2d_resized.mean()
            std_val = img_2d_resized.std()
            if std_val > 1e-6:
                img_2d_resized = (img_2d_resized - mean_val) / std_val
            else:
                img_2d_resized = img_2d_resized - mean_val

            # Replicate the grayscale channel 3 times to get shape [3, H, W]
            img_tensor = torch.tensor(img_2d_resized, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

            # Fake tabular data: 128 zeros
            tabular_tensor = torch.zeros(128, dtype=torch.float32)

            return {'image': img_tensor, 'tabular': tabular_tensor}

        elif self.modality == 'tabular':
            # TABULAR MODE: Load and normalize tabular data, produce fake zero image
            # Load JSON
            json_files = glob(os.path.join(sample_dir, '*.json'))
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {sample_dir}")
            json_path = json_files[0]
            with open(json_path, 'r') as f:
                tabular_data = json.load(f)

            # Convert tabular data to array and normalize
            tabular_values = np.array(list(tabular_data.values()), dtype=np.float32)
            tabular_values = self.tabular_scaler.transform(tabular_values.reshape(1, -1)).flatten()
            tabular_tensor = torch.tensor(tabular_values, dtype=torch.float32)

            # Fake image: zeros of shape [3, H, W]
            img_tensor = torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32)

            return {'image': img_tensor, 'tabular': tabular_tensor}

        else:
            raise ValueError(f"Invalid modality: {self.modality}")

    def resize_image(self, image, size):
        # image: 2D numpy array [H, W]
        # size: (H, W) desired
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize(size[::-1], Image.BILINEAR)  # Note: (W, H) for PIL
        image_resized = np.array(pil_image, dtype=np.float32)
        return image_resized
