import os
import json
import nibabel as nib
from pathlib import Path

def generate_json(data_dir, output_dir, loader="L2R2024LUMIR", n=10000):
    # Initialize the JSON structure
    data = {
        "loader": loader,
        "shape": [],
        "transform": [
            {"class_name": "Nifti2Array"},
            {"class_name": "DatatypeConversion"},
            {"class_name": "ToTensor"}
        ],
        "pairs": []
    }

    # Traverse the data_dir and collect paths
    patient_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))][:n]

    id_counter = 1
    first_image_shape = None
    for patient_id in sorted(patient_folders):
        anat_dir = os.path.join(data_dir, patient_id, "anat")
        if os.path.exists(anat_dir):
            image_files = [f for f in os.listdir(anat_dir) if f.endswith(".nii.gz")]
            if image_files:
                image_path = os.path.join(anat_dir, image_files[0])
                if first_image_shape is None:
                    # Deduce shape from the first image
                    img = nib.load(image_path)
                    first_image_shape = img.shape
                    data["shape"] = [1, first_image_shape[2], first_image_shape[0], first_image_shape[1]]

                pair = {
                    "id": id_counter,
                    "f_img": image_path,
                    "m_img": image_path
                }
                data["pairs"].append(pair)
                id_counter += 1
            else:
                print(f"Warning: No .nii.gz files found in {anat_dir} for patient {patient_id}")
        else:
            print(f"Warning: Anat directory not found for patient {patient_id}")

    # Ensure the output directory exists
    os.makedirs(Path(output_dir).parent, exist_ok=True)

    # Write the JSON to the output file
    with open(output_dir, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"JSON file saved to {output_dir}")

# Example usage
data_dir = "/mnt/dataset_storage/ldm_100k/LDM-data/LDM_100k/rawdata"
output_dir = "/home/PRISMS/experiments_vfa/vfa/data_configs/ldm100k/ldm100k_inference.json"
generate_json(data_dir, output_dir, loader="LDM100kDataset")
