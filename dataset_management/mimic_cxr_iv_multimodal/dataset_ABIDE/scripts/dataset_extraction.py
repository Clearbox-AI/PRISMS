import tarfile
from pathlib import Path
import argparse
import shutil
import nibabel as nib
import numpy as np


# Function to extract a tar.gz file while maintaining the directory structure
def extract_tar_gz(tar_path: Path, extract_to: Path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)


# Function to save a .nii.gz file as a NumPy array and remove the original .nii.gz
def save_as_numpy_and_remove(nii_file: Path, remove_nii: bool):
    img = nib.load(nii_file)
    data = img.get_fdata()
    np.save(nii_file.with_suffix('.npy'), data)
    if remove_nii:
        nii_file.unlink()


# Function to convert all .nii.gz files in a directory tree to NumPy arrays
def convert_nii_to_numpy(output_dir: Path, remove_nii: bool):
    # Recursively iterate through all files in the output directory
    for file in output_dir.rglob('*.nii.gz'):
        save_as_numpy_and_remove(file, remove_nii)


# Function to handle the extraction process
def extract_abide_dataset(root_dir: Path, output_dir: Path, remove: bool):
    # Create the extracted directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over the institution folders
    for institution_folder in root_dir.iterdir():

        # Skip if it's not a directory
        if not institution_folder.is_dir():
            continue

        # Define the path to extract the contents for the current institution
        institution_output_dir = output_dir / institution_folder.name
        institution_output_dir.mkdir(parents=True, exist_ok=True)

        # Iterate over files in the institution folder
        for file in institution_folder.iterdir():
            if file.suffix == '.gz' and file.name.endswith('.tar.gz'):
                # Extract .tar.gz files
                tar_file_path = file
                print(f'Extracting {tar_file_path} to {institution_output_dir}')
                extract_tar_gz(tar_file_path, institution_output_dir)
                print(f'Extraction complete for {tar_file_path}')

                # Remove the tar.gz file if --remove is specified
                if remove:
                    tar_file_path.unlink()
                    print(f'Removed {tar_file_path}')

            else:
                # Copy other non-.tar.gz files (e.g., .csv, .txt)
                if file.is_file():
                    shutil.copy(file, institution_output_dir / file.name)
                    print(f'Copied {file} to {institution_output_dir / file.name}')

                    # Remove the file after copying if --remove is specified
                    if remove:
                        file.unlink()
                        print(f'Removed {file}')

    # Remove all the content in the input directory after the process is complete
    print(f"Removing all contents of {root_dir}")
    for item in root_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print(f"All contents of {root_dir} have been removed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract ABIDE dataset tar.gz files and maintain directory structure.")

    parser.add_argument(
        '--input_dir',
        type=Path,
        required=True,
        help="The root directory where the ABIDE dataset is located."
    )

    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help="The directory where the extracted files will be saved"
    )

    parser.add_argument(
        '--remove',
        type=bool,
        required=True,
        help="Remove the original files after extraction and copying."
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['extract', 'convert', 'both'],
        default='both',
        help="Choose whether to just extract ('extract'), just convert nii to numpy ('convert'), or do both ('both')."
    )

    args = parser.parse_args()

    # Run based on the specified mode
    if args.mode == 'extract':
        print("Running in extract-only mode.")
        extract_abide_dataset(args.input_dir, args.output_dir, args.remove)

    elif args.mode == 'convert':
        print("Running in convert-only mode.")
        convert_nii_to_numpy(args.output_dir, args.remove)

    elif args.mode == 'both':
        print("Running in both extraction and conversion mode.")
        extract_abide_dataset(args.input_dir, args.output_dir, args.remove)
        convert_nii_to_numpy(args.output_dir, args.remove)