import os
import sys
import json
import shutil
import argparse
import pandas as pd
from glob import glob
import logging
import nibabel as nib
import numpy as np
import pathlib
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process CSV and images for patients.")
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--input_image_folder', type=str, required=True, help='Path to the input images folder.')
    parser.add_argument('--output_data_folder', type=str, required=True, help='Path to the output data folder.')
    parser.add_argument('--exclude_columns_json', type=str, required=True,
                        help='Path to JSON file containing columns to exclude.')
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main(csv_path, input_image_folder, output_data_folder, exclude_columns_json):
    # Setup logging
    setup_logging()

    logging.info("Reading the CSV file.")
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read CSV file: {e}")
        sys.exit(1)

    logging.info("Reading columns to exclude from JSON.")
    # Read the exclude columns from JSON file
    try:
        with open(exclude_columns_json, 'r') as f:
            exclude_columns = json.load(f)["exclude_columns"]
        # Ensure it's a list
        if not isinstance(exclude_columns, list):
            raise ValueError("Exclude columns JSON should be a list of column names.")
    except Exception as e:
        logging.error(f"Failed to read exclude columns JSON: {e}")
        sys.exit(1)

    logging.info("Dropping rows with missing date information.")
    # Drop rows with NaN in date columns
    date_columns = ['MRIYR', 'MRIMO', 'MRIDY']
    for col in date_columns:
        if col not in df.columns:
            logging.error(f"Required date column '{col}' not found in CSV.")
            sys.exit(1)
    df = df.dropna(subset=date_columns)

    logging.info("Sorting DataFrame and selecting the latest entry per patient.")
    # Sort and keep the last row per patient based on MRI date
    df = df.sort_values(by=['NACCID'] + date_columns)
    df = df.groupby('NACCID').tail(1)

    logging.info("Filtering rows where NACCMVOL == 1.")
    # Filter rows where NACCMVOL == 1
    if 'NACCMVOL' not in df.columns:
        logging.error("Required column 'NACCMVOL' not found in CSV.")
        sys.exit(1)
    df = df[df['NACCMVOL'] == 1]

    logging.info("Excluding specified columns from DataFrame.")
    # Exclude specified columns
    missing_columns = set(exclude_columns) - set(df.columns)
    if missing_columns:
        logging.warning(f"The following columns to exclude were not found in the DataFrame: {missing_columns}")
    df = df.drop(columns=[col for col in exclude_columns if col in df.columns])

    logging.info("Ensuring output directory exists.")
    # Ensure output directory exists
    os.makedirs(output_data_folder, exist_ok=True)

    logging.info("Starting to process each patient.")
    # Iterate over patients
    for _, row in df.iterrows():
        patient_id = row['NACCID']
        patient_folder_name = f"sub-{patient_id}"
        input_patient_folder = os.path.join(input_image_folder, patient_folder_name)
        output_patient_folder = os.path.join(output_data_folder, patient_folder_name)

        # Check if input patient folder exists
        if not os.path.exists(input_patient_folder):
            logging.warning(f"Input folder for patient {patient_id} does not exist. Skipping.")
            continue

        # Create output patient folder
        os.makedirs(output_patient_folder, exist_ok=True)

        # Copy image file(s)
        anat_folder = os.path.join(input_patient_folder, 'anat')
        nii_gz_files = glob(os.path.join(anat_folder, '*.nii.gz'))

        if not nii_gz_files:
            logging.warning(f"No .nii.gz files found for patient {patient_id}. Skipping.")
            continue

        for nii_file in nii_gz_files:
            try:
                # Load the .nii.gz file
                img = nib.load(nii_file)
                img_data = img.get_fdata()

                # Convert to float32 for consistency
                img_data = img_data.astype(np.float32)

                # Extract the middle slice
                mid_slice_index = img_data.shape[2] // 2  # Middle slice along the third axis
                middle_slice = img_data[:, :, mid_slice_index]

                # Visualize the middle slice
                # plt.imshow(middle_slice, cmap='gray')
                # plt.title(f"Middle slice of {os.path.basename(nii_file)}")
                # plt.axis('off')
                # plt.show()

                # Define the output .npy file name (remove `.nii.gz` using pathlib)
                base_name = pathlib.Path(pathlib.Path(nii_file).stem).stem  # This removes both `.gz` and `.nii`
                npy_file_name = f"{base_name}_middle_slice.npy"
                npy_file_path = os.path.join(output_patient_folder, npy_file_name)

                # Save the middle slice as a NumPy array
                np.save(npy_file_path, middle_slice)
                logging.info(f"Converted and saved middle slice of {nii_file} to {npy_file_path}")
            except Exception as e:
                logging.error(f"Failed to process {nii_file} for patient {patient_id}: {e}")
                continue  # Proceed with next file

        # Prepare patient data as JSON
        patient_data = row.to_dict()
        # Remove NACCID from data
        patient_data.pop('NACCID', None)
        json_data = {patient_id: list(patient_data.values())}

        # Save JSON data
        json_path = os.path.join(output_patient_folder, f"{patient_id}_data.json")
        try:
            with open(json_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)
            logging.info(f"Saved JSON data for patient {patient_id} at {json_path}")
        except Exception as e:
            logging.error(f"Failed to save JSON data for patient {patient_id}: {e}")

        logging.info(f"Processed patient {patient_id}")

    logging.info("Processing completed.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args.csv_path, args.input_image_folder, args.output_data_folder, args.exclude_columns_json)
