import os
import json
import numpy as np
import nibabel as nib


def process_data(
        input_folder: str,
        output_folder: str,
        train_features_file: str,
        metadata_features_file: str
):
    """
    For each subject sub-{naccid} in input_folder:
        1) Loads image3d.nii.gz, extracts the middle slice, and saves it as image3d.npy.
        2) Loads all_columns.json and uses train_features.json, metadata.json to decide
           which features go into tabular.json and metadata.json.
           Replaces missing values with 88888 and maps SEX from M/F to 0/1.
           Tabular format is now: { "NACC001412": [val1, val2, val3, ...] }
    """

    # 1. Read the lists of features from the two JSON files
    with open(train_features_file, 'r') as tf:
        train_features = json.load(tf)
    with open(metadata_features_file, 'r') as mf:
        metadata_features = json.load(mf)

    # 2. Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # 3. Loop over each folder in input_folder that starts with "sub-"
    for subdir_name in os.listdir(input_folder):
        subdir_path = os.path.join(input_folder, subdir_name)
        if os.path.isdir(subdir_path) and subdir_name.startswith("sub-"):

            # Prepare the corresponding output path
            out_subdir_path = os.path.join(output_folder, subdir_name)
            os.makedirs(out_subdir_path, exist_ok=True)

            # --- Step A: Process the 3D image and save the middle slice ---
            image_path = os.path.join(subdir_path, "image3d.nii.gz")
            if os.path.exists(image_path):
                nii_img = nib.load(image_path)
                image_data = nii_img.get_fdata()  # shape could be (X, Y, Z) or more dims

                # Choose the middle slice along the third dimension
                mid_slice_idx = image_data.shape[2] // 2
                middle_slice = image_data[:, :, mid_slice_idx]

                # Save the middle slice as .npy
                np.save(os.path.join(out_subdir_path, "image.npy"), middle_slice)

            # --- Step B: Process the tabular data ---
            all_columns_path = os.path.join(subdir_path, "all_columns.json")
            if os.path.exists(all_columns_path):
                with open(all_columns_path, 'r') as ac:
                    all_columns = json.load(ac)
            else:
                # If the JSON isn't there, just skip or define an empty dict
                all_columns = {}

            # We extract the subject's naccid from the folder name
            naccid = subdir_name[4:]  # e.g. "sub-NACC001412" -> "NACC001412"

            # Build the tabular array in the order of train_features
            tabular_list = []
            for feature in train_features:
                val = all_columns.get(feature, 88888)

                # If the key exists but is None
                if val is None:
                    val = 88888

                # Special handling for SEX
                if feature == "SEX":
                    if val == "M":
                        val = 0
                    elif val == "F":
                        val = 1
                    else:
                        val = 88888

                tabular_list.append(val)

            # Put that list into a dict with the subject's naccid as key
            tabular_dict = {naccid: tabular_list}

            # Process metadata in the old format
            metadata_dict = {}
            for feature in metadata_features:
                val = all_columns.get(feature, 88888)
                if val is None:
                    val = 88888
                metadata_dict[feature] = val

            # Save to tabular.json and metadata.json
            with open(os.path.join(out_subdir_path, "tabular.json"), 'w') as tj:
                json.dump(tabular_dict, tj)
            with open(os.path.join(out_subdir_path, "metadata.json"), 'w') as mj:
                json.dump(metadata_dict, mj)

# ------------- Example usage -------------
if __name__ == "__main__":
    process_data(
        input_folder="/mnt/dataset_storage/data/adni_nacc_processing_steps/8_cubic",
        output_folder="/mnt/dataset_storage/data/adni_nacc_processing_steps/final_2d",
        train_features_file="/mnt/dataset_storage/data/adni_nacc_processing_steps/list_of_features/train_features.json",
        metadata_features_file="/mnt/dataset_storage/data/adni_nacc_processing_steps/list_of_features/metadata.json"
    )
