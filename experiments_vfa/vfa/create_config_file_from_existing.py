import json
from pathlib import Path

# Input and output file paths
input_file = Path(Path(__file__).parent.resolve(), "data_configs", "lumir", "lumir_test.json")
output_file = Path(Path(__file__).parent.resolve(), "data_configs", "lumir", "lumir_inference.json")

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
