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
