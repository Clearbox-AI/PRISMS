import json
from pathlib import Path
import os


def create_config_from_existing():
    # Input and output file paths
    input_file = Path(Path(__file__).parent.resolve(), "data_configs", "lumir", "lumir_test.json")
    output_file = Path(Path(__file__).parent.resolve(), "data_configs", "lumir", "lumir_inference_old.json")

    # Read the original JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Modify the "pairs" list so that m_img matches f_img
    if 'pairs' in data:
        for pair in data['pairs']:
            if 'f_img' in pair:
                pair['m_img'] = pair['f_img']

    # Write the updated data to a new file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


def create_from_data():
    # Specify the directory containing the files
    input_directory = Path("/mnt", "storage", "data", "utils", "imagesTot")
    output_file = Path(Path(__file__).parent.resolve(), "data_configs", "lumir", "lumir_inference.json")

    # Initialize the JSON structure
    data = {
        "loader": "L2R2024LUMIR",
        "shape": [1, 160, 224, 192],
        "transform": [
            {"class_name": "Nifti2Array"},
            {"class_name": "DatatypeConversion"},
            {"class_name": "ToTensor"}
        ],
        "pairs": []
    }

    # Get all files in the directory
    file_list = sorted([
        os.path.join(input_directory, file)
        for file in os.listdir(input_directory)
        if os.path.isfile(os.path.join(input_directory, file))
    ])

    # Populate the "pairs" list
    for idx, file_path in enumerate(file_list, start=1):
        data["pairs"].append({
            "id": idx,
            "f_img": file_path,
            "m_img": file_path
        })

    # Write the updated JSON to a new file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"JSON file has been created: {output_file}")


if __name__ == '__main__':
    create_from_data()