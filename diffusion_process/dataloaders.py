from PIL import Image
import os
import json
import numpy as np
from torch.utils.data import Dataset
import torch as th
from glob import glob
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

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



class ToyMNISTDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Path to the directory containing subdirectories for each sample.
                            Each subdirectory should contain an image (as .npy) and a corresponding tabular .json file.
        """
        self.data_dir = data_dir

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
        self.image_transform = transforms.Normalize((0.5,), (0.5,))  # Normalize images to range [-1, 1]

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        # Load image data
        image_files = glob(os.path.join(sample_dir, '*.npy'))
        if not image_files:
            raise FileNotFoundError(f"No image .npy files found in {sample_dir}")
        image_path = image_files[0]
        image = np.load(image_path).astype(np.float32)  # Shape: [28, 28]

        # Normalize image
        image = (image - image.min()) / (image.max() - image.min())  # Scale to [0, 1]
        image = th.tensor(image).unsqueeze(0)  # Add channel dimension, shape: [1, 28, 28]
        image = self.image_transform(image)  # Normalize to [-1, 1]

        # Load tabular data
        json_files = glob(os.path.join(sample_dir, '*.json'))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {sample_dir}")
        json_path = json_files[0]
        with open(json_path, 'r') as f:
            tabular_data = json.load(f)

        # Convert tabular data to numpy array
        tabular_values = np.array(list(tabular_data.values()), dtype=np.float32).reshape(1, -1)
        tabular_values = self.tabular_scaler.transform(tabular_values).flatten()

        # Convert to torch tensor
        tabular_tensor = th.tensor(tabular_values)

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
