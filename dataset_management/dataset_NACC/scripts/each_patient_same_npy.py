import os
import shutil

def replace_npy_files(input_folder):
    # List all subfolders in the input folder
    subfolders = [os.path.join(input_folder, subfolder) for subfolder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, subfolder))]

    if not subfolders:
        print("No subfolders found in the input folder.")
        return

    # Take the first folder and find the .npy file in it
    first_folder = subfolders[0]
    npy_files = [file for file in os.listdir(first_folder) if file.endswith('.npy')]

    if not npy_files:
        print(f"No .npy file found in the first folder: {first_folder}")
        return

    source_npy_file = os.path.join(first_folder, npy_files[0])

    # Replace .npy files in the other subfolders
    for folder in subfolders[1:]:
        # Find and remove existing .npy files in the current folder
        existing_npy_files = [file for file in os.listdir(folder) if file.endswith('.npy')]
        for existing_file in existing_npy_files:
            os.remove(os.path.join(folder, existing_file))

        # Copy the source .npy file to the current folder
        destination_file = os.path.join(folder, os.path.basename(source_npy_file))
        shutil.copy(source_npy_file, destination_file)
        print(f"Replaced .npy file in {folder} with {source_npy_file}")

if __name__ == "__main__":
    input_folder = "/mnt/dataset_storage/data/nacc_dataset/nacc_subset_latents_samelatent"

    if os.path.exists(input_folder) and os.path.isdir(input_folder):
        replace_npy_files(input_folder)
    else:
        print("Invalid folder path. Please provide a valid directory.")