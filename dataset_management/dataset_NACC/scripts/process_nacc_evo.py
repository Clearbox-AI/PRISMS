#!/usr/bin/env python

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import pathlib
import cv2
import shutil
import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file
from diffusers.models import AutoencoderKL

from glob import glob


##############################################################################
#                     AUTOENCODER WRAPPER CLASS
##############################################################################
class AutoencoderWrapper:
    """
    A simple wrapper around a diffusers.models.AutoencoderKL (and others in future) initialized from
    a config.json and weights in safetensors format.

    Usage:
        ae = AutoencoderWrapper(config_path="path/to/config.json",
                                weights_path="path/to/diffusion_pytorch_model.safetensors",
                                device="cuda")
        latents = ae.encode(middle_slice_2d)  # returns np.ndarray
    """

    def __init__(self, model_path, device="cuda"):

        self.device = torch.device(device)

        # # Load the config.json
        # with open(config_path, "r") as f:
        #     config_dict = json.load(f)

        # Initialize AutoencoderKL with config
        # self.model = AutoencoderKL(**config_dict)
        self.model = AutoencoderKL.from_pretrained(model_path).to(device)

        # # Load weights from safetensors
        # state_dict = load_file(weights_path)
        # self.model.load_state_dict(state_dict)

        # Move to device and set to eval
        self.model.to(self.device)
        self.model.eval()

    def encode(self, slice_2d: np.ndarray) -> np.ndarray:
        """
        Encode a 2D slice into latent space. The input is assumed to be
        float32, shape [H, W].
        Returns a np.ndarray containing latents, e.g. shape [1, latent_channels, H/8, W/8].
        """

        if slice_2d.ndim == 2:
            slice_2d = slice_2d[np.newaxis, ...]

        slice_2d_norm = self._normalize_complex_image(slice_2d)

        # Convert to 2-channel [real, imag], This will produce shape [2, N, H, W] if input was [N,H,W]
        two_channel_slice = self._complex_to_two_channel_image(slice_2d_norm)

        # If original had shape [1, H, W], then after stack => [2, 1, H, W].
        # PyTorch expects [batch, channels, height, width]. Let's reorder to [1, 2, H, W].
        two_channel_slice = np.transpose(two_channel_slice, (1, 0, 2, 3))  # => [1, 2, H, W]

        # Convert to torch tensor
        slice_tensor = torch.from_numpy(two_channel_slice).float().to(self.device)

        with torch.no_grad():
            encoder_out = self.model.encode(slice_tensor)
            latents = encoder_out.latent_dist.mean  # shape [1, latent_channels, h, w]

            # plt.imshow(latents[0, 0].detach().cpu().numpy(), cmap='gray')
            # plt.show()
        return latents.cpu().numpy()

    def _normalize_complex_image(self, complex_img: np.ndarray) -> np.ndarray:
        """
        Scales the complex-valued image by the 99.5th percentile of its magnitude
        to keep values in a reasonable range for neural networks.
        """
        max_val = np.percentile(np.abs(complex_img), 99.5)
        return complex_img / (max_val + 1e-8)

    def _complex_to_two_channel_image(self, complex_img: np.ndarray) -> np.ndarray:
        """
        Splits a complex-valued image into a 2-channel image [real, imag].

        Example:
          Input shape:  [N, H, W] or [H, W]
          Output shape: [2, N, H, W] if input was [N,H,W];
                        or [2, H, W] if input was [H,W].
        """
        real = np.real(complex_img)
        imag = np.imag(complex_img)
        return np.stack((real, imag), axis=0)  # shape => [2, ...]


##############################################################################
#                     ARGUMENT PARSING
##############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process CSV and images for patients.")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the input CSV file.') #/mnt/dataset_storage/data/nacc_dataset/dataset_manipulation/investigator_mri_nacc65.csv
    parser.add_argument('--input_image_folder', type=str, required=True, help='Path to the input images folder.') #/mnt/dataset_storage/data/nacc_dataset/NACC_ORIGINAL
    parser.add_argument('--output_data_folder', type=str, required=True, help='Path to the output data folder.') #/mnt/dataset_storage/data/nacc_dataset/nacc_subset_latents
    parser.add_argument('--exclude_columns_json', type=str, required=True, #/mnt/dataset_storage/data/nacc_dataset/dataset_manipulation/columns_to_remove_investigator_mri_nacc65.json
                        help='Path to JSON file containing columns to exclude.')
    parser.add_argument('--transform', type=str, default=None,
                        help="If 'encode', encode the middle slice using the autoencoder and save the latents instead.") #encode

    # When transform="encode", we expect these two extra arguments
    parser.add_argument('--autoencoder_path', type=str, default=None,
                        help='Path to the autoencoder model (config + weights).') #/home/PRISMS/model_garden/microsoft_mri_autoencoder_01/weights
    return parser.parse_args()


##############################################################################
#                     LOGGING SETUP
##############################################################################
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


##############################################################################
#                     MAIN FUNCTION
##############################################################################
def main(csv_path,
         input_image_folder,
         output_data_folder,
         exclude_columns_json,
         transform=None,
         autoencoder_path=None):
    setup_logging()

    logging.info("Reading the CSV file.")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read CSV file: {e}")
        sys.exit(1)

    logging.info("Reading columns to exclude from JSON.")
    try:
        with open(exclude_columns_json, 'r') as f:
            exclude_columns = json.load(f)["exclude_columns"]
        if not isinstance(exclude_columns, list):
            raise ValueError("Exclude columns JSON should be a list of column names.")
    except Exception as e:
        logging.error(f"Failed to read exclude_columns JSON: {e}")
        sys.exit(1)

    logging.info("Dropping rows with missing date information.")
    date_columns = ['MRIYR', 'MRIMO', 'MRIDY']
    for col in date_columns:
        if col not in df.columns:
            logging.error(f"Required date column '{col}' not found in CSV.")
            sys.exit(1)
    df = df.dropna(subset=date_columns)

    logging.info("Sorting DataFrame and selecting the latest entry per patient.")
    df = df.sort_values(by=['NACCID'] + date_columns)
    df = df.groupby('NACCID').tail(1)

    logging.info("Filtering rows where NACCMVOL == 1.")
    if 'NACCMVOL' not in df.columns:
        logging.error("Required column 'NACCMVOL' not found in CSV.")
        sys.exit(1)
    df = df[df['NACCMVOL'] == 1]

    logging.info("Excluding specified columns from DataFrame.")
    missing_columns = set(exclude_columns) - set(df.columns)
    if missing_columns:
        logging.warning(f"The following columns to exclude were not found in the DataFrame: {missing_columns}")
    df = df.drop(columns=[col for col in exclude_columns if col in df.columns])

    logging.info("Ensuring output directory exists.")
    os.makedirs(output_data_folder, exist_ok=True)

    # If transform is "encode", instantiate our autoencoder
    ae_wrapper = None
    if transform == "encode":
        if not autoencoder_path:
            logging.error("transform='encode' requires --autoencoder_path.")
            sys.exit(1)
        logging.info("Initializing AutoencoderWrapper.")
        ae_wrapper = AutoencoderWrapper(model_path=autoencoder_path,
                                        device="cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Starting to process each patient.")
    for _, row in df.iterrows():
        patient_id = row['NACCID']
        patient_folder_name = f"sub-{patient_id}"
        input_patient_folder = os.path.join(input_image_folder, patient_folder_name)
        output_patient_folder = os.path.join(output_data_folder, patient_folder_name)

        if not os.path.exists(input_patient_folder):
            logging.warning(f"Input folder for patient {patient_id} does not exist. Skipping.")
            continue

        os.makedirs(output_patient_folder, exist_ok=True)

        anat_folder = os.path.join(input_patient_folder, 'anat')
        nii_gz_files = glob(os.path.join(anat_folder, '*.nii.gz'))

        if not nii_gz_files:
            logging.warning(f"No .nii.gz files found for patient {patient_id}. Skipping.")
            continue

        for nii_file in nii_gz_files:
            try:
                if "sub-NACC774848" in nii_file:
                    bla = 3
                img = nib.load(nii_file)
                img_data = img.get_fdata().astype(np.float32)

                # Middle slice along z-axis
                mid_slice_index = img_data.shape[2] // 2
                middle_slice = img_data[:, :, mid_slice_index]

                base_name = pathlib.Path(pathlib.Path(nii_file).stem).stem

                if transform is None:
                    # just save the middle slice
                    npy_file_name = f"{base_name}_middle_slice.npy"
                    npy_file_path = os.path.join(output_patient_folder, npy_file_name)
                    np.save(npy_file_path, middle_slice)
                    logging.info(f"Saved middle slice of {nii_file} to {npy_file_path}")

                elif transform == "encode":

                    if middle_slice.shape[0] != middle_slice.shape[1]:
                        # -----------------------------------------------------------------
                        # Make the 2D array a square by padding the smaller dimension
                        # to match the larger dimension. No cropping is performed.
                        # E.g., if shape is 177x256 => becomes 256x256
                        # -----------------------------------------------------------------

                        # Make sure it's square by padding if needed
                        middle_slice = pad_to_square(middle_slice)
                    if middle_slice.shape[1] != 256:

                        # Ensure final shape is 256 x 256
                        middle_slice = pad_or_resize_to_256(middle_slice)

                    # Use the autoencoder wrapper to encode the slice
                    latents = ae_wrapper.encode(middle_slice)

                    assert latents.shape[2] == latents.shape[3] and latents.shape[3] == 64

                    latents_npy_file = f"{base_name}_middle_slice_latents.npy"
                    latents_npy_path = os.path.join(output_patient_folder, latents_npy_file)
                    np.save(latents_npy_path, latents)
                    logging.info(f"Encoded and saved latents of middle slice of {nii_file} to {latents_npy_path}")

            except Exception as e:
                logging.error(f"Failed to process {nii_file} for patient {patient_id}: {e}")
                continue

        # Prepare patient data as JSON
        patient_data = row.to_dict()
        patient_data.pop('NACCID', None)
        json_data = {patient_id: list(patient_data.values())}

        json_path = os.path.join(output_patient_folder, f"{patient_id}_data.json")
        try:
            with open(json_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)
            logging.info(f"Saved JSON data for patient {patient_id} at {json_path}")
        except Exception as e:
            logging.error(f"Failed to save JSON data for patient {patient_id}: {e}")

        logging.info(f"Processed patient {patient_id}")

    logging.info("Processing completed.")


def pad_to_square(slice_2d: np.ndarray) -> np.ndarray:
    """
    Zero-pad the smaller dimension so that the output is [side, side],
    where side = max(H, W). Never crops data.
    """
    H, W = slice_2d.shape
    if H == W:
        return slice_2d  # Already square

    side = max(H, W)
    pad_h = side - H
    pad_w = side - W

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return np.pad(
        slice_2d,
        pad_width=((top, bottom), (left, right)),
        mode='constant',
        constant_values=0
    )


def pad_or_resize_to_256(slice_2d: np.ndarray) -> np.ndarray:
    """
    1) If slice_2d is bigger than 256 in any dimension, resize to 256x256 (no cropping).
    2) If it's exactly 256x256, do nothing.
    3) If it's smaller, zero-pad up to 256x256.
    """
    H, W = slice_2d.shape

    # If either dimension is larger than 256, just resize to 256x256
    if H > 256 or W > 256:
        # cv2 expects images in height x width, and we'll specify the output size
        slice_2d_resized = cv2.resize(
            slice_2d,
            (256, 256),
            interpolation=cv2.INTER_AREA  # Good for downsampling
        )
        return slice_2d_resized

    # If already 256x256, do nothing
    if H == 256 and W == 256:
        return slice_2d

    # Otherwise, we zero-pad to 256 in both dimensions
    pad_h = 256 - H
    pad_w = 256 - W

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return np.pad(
        slice_2d,
        pad_width=((top, bottom), (left, right)),
        mode='constant',
        constant_values=0
    )




##############################################################################
#                     ENTRY POINT
##############################################################################
if __name__ == "__main__":

    args = parse_arguments()

    main(csv_path=args.csv_path,
         input_image_folder=args.input_image_folder,
         output_data_folder=args.output_data_folder,
         exclude_columns_json=args.exclude_columns_json,
         transform=args.transform,
         autoencoder_path=args.autoencoder_path)
