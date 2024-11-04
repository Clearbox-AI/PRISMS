import tarfile
from pathlib import Path
import argparse
import shutil
import nibabel as nib
import numpy as np
import re


# Function to extract a tar.gz file while maintaining the directory structure
#def extract_tar_gz(tar_path: Path, extract_to: Path):
#    with tarfile.open(tar_path, 'r:gz') as tar:
#        tar.extractall(path=extract_to)

# Function to extract a tar.gz file while unifying contents in a common directory
def extract_tar_gz(tar_path: Path, extract_to: Path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        temp_dir = extract_to / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)  # Create a temporary directory for extraction
        tar.extractall(path=temp_dir)

        # Move files from the temp directory to the main extract_to directory
        for item in temp_dir.iterdir():
            # If there's an intermediate folder like "exam_part1", move its contents up
            if item.is_dir():
                for sub_item in item.iterdir():
                    shutil.move(str(sub_item), extract_to)
                shutil.rmtree(item)  # Remove the intermediate directory after moving
            else:
                shutil.move(str(item), extract_to)

        # Clean up the temp directory
        temp_dir.rmdir()




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



# Function to handle the extraction process with support for both split and regular tar files
def extract_abide_dataset(root_dir: Path, output_dir: Path, remove: bool):
    # Create the extracted directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pattern to match filenames like "name_{number}_part{index}.tar.gz"
    pattern = re.compile(r"(.+)_([0-9]+)_part[0-9]+\.tar\.gz")

    # Dictionary to group tar files by their base name and number (e.g., 'ABIDEII-ONRC_2')
    file_groups = {}

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
                # Check if it's a split part file (e.g., name_{number}_part{index}.tar.gz)
                match = pattern.match(file.name)
                if match:
                    base_name = f"{match.group(1)}_{match.group(2)}"
                    if base_name not in file_groups:
                        file_groups[base_name] = []
                    file_groups[base_name].append(file) #file_groups[base_name].append(file)
                else:
                    # If it's a regular tar.gz file, extract it directly
                    single_output_dir = institution_output_dir / file.with_suffix('').stem  # Use file.stem to create a folder without .tar.gz
                    single_output_dir.mkdir(parents=True, exist_ok=True)
                    print(f'Extracting {file} to {single_output_dir}')
                    extract_tar_gz(file, single_output_dir)
                    print(f'Extraction complete for {file}')

                    # Remove the tar.gz file if --remove is specified
                    if remove:
                        file.unlink()
                        print(f'Removed {file}')

        # Process each group of split tar files (e.g., name_{number}_part{index})
        for base_name, tar_files in file_groups.items():
            base_output_dir = institution_output_dir / base_name
            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Extract each tar file in the group, unifying contents
            for tar_file in sorted(tar_files):  # Sort by part number to maintain the order
                print(f'Extracting {tar_file} to {base_output_dir}')
                extract_tar_gz(tar_file, base_output_dir)  # Uses the updated extract_tar_gz function
                print(f'Extraction complete for {tar_file}')

                # Remove the tar.gz file if --remove is specified
                if remove:
                    tar_file.unlink()
                    print(f'Removed {tar_file}')

        # Copy other non-.tar.gz files (e.g., .csv, .txt)
        for file in institution_folder.iterdir():
            if not file.suffix.endswith('.tar.gz') and file.is_file():
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

    parser = argparse.ArgumentParser(description="Extract ABIDE dataset tar.gz files and convert .nii images in npy arrays.")

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
