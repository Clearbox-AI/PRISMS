import os
import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path
from experiments_vfa.vfa.utils.utils import load_data_configs
import nibabel as nib
from experiments_vfa.vfa.models.vfa import VFA
from experiments_vfa.vfa.datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np

# ------------------------------------------------------------
# Step 1: Define model and dataset parameters
# ------------------------------------------------------------
params_path = Path(Path(__file__).parent.resolve(), "params", "vfa", "vfa_ncc.json")
with open(params_path) as f:
    params = json.load(f)

params['eval_data_configs'] = Path(Path(__file__).parent.resolve(), "data_configs", "lumir", "lumir_val.json")
eval_data_configs = load_data_configs(params['eval_data_configs'])
params['model']['in_channels'] = eval_data_configs['shape'][0]
params['model']['in_shape'] = eval_data_configs['shape'][1:]

configs_path = Path(Path(__file__).parent.resolve(), "data_configs", "ldm100k", "ldm100k_inference.json")
with open(configs_path) as f:
    configs = json.load(f)

DatasetClass = load_dataset('LDM100k')
dataset = DatasetClass(configs, params)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# ------------------------------------------------------------
# Step 2: Load the model and checkpoint
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VFA(params, device)
model.eval()

checkpoint_path = Path("/mnt", "storage", "data", "utils", "net.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

extracted_features = {}

def encoder_hook(module, input, output):
    # Extract bottleneck
    bottleneck_feat = output[-1]  # [B, C, D, H, W]
    # Move it to CPU immediately to free GPU memory
    extracted_features['bottleneck'] = bottleneck_feat.detach().cpu()

hook_handle = model.encoder.register_forward_hook(encoder_hook)

output_folder = Path("/mnt", "dataset_storage", "ldm_100k", "LDM-data", "LDM_100k", "tabular_gen_vfa_net")
os.makedirs(output_folder, exist_ok=True)

with torch.no_grad():
    for batch_idx, sample in enumerate(tqdm(dataloader, total=len(dataloader), desc="Processing")):
        # Move tensors to device
        for k in sample:
            if isinstance(sample[k], torch.Tensor):
                sample[k] = sample[k].to(device, non_blocking=True)

        results = model(sample)  # Inference step

        # Now extract features from the global variable
        bottleneck_feat = extracted_features.get('bottleneck')
        if bottleneck_feat is None:
            # Something went wrong or no features extracted
            continue

        # Compute embeddings (on CPU)
        avg_emb = bottleneck_feat.mean(dim=[2, 3, 4])
        max_emb = bottleneck_feat.amax(dim=[2, 3, 4])
        embedding = torch.cat([avg_emb, max_emb], dim=1)  # CPU tensor
        embedding_np = embedding.numpy()

        for i in range(embedding_np.shape[0]):
            patient_img_path = sample['f_img_path'][i]
            patient_img_name = os.path.basename(patient_img_path)
            patient_id = Path(os.path.splitext(patient_img_name)[0]).stem

            patient_folder = os.path.join(output_folder, patient_id)
            os.makedirs(patient_folder, exist_ok=True)

            # Move image back to CPU if not already
            f_img_cpu = sample['f_img'][i].cpu().numpy().squeeze()
            f_affine_cpu = sample['f_affine'][i].cpu().numpy() if 'f_affine' in sample else np.eye(4)

            out_nii = nib.Nifti1Image(f_img_cpu, f_affine_cpu)
            nii_path = os.path.join(patient_folder, patient_img_name)
            nib.save(out_nii, nii_path)

            embedding_list = embedding_np[i].tolist()
            json_data = {patient_id: embedding_list}

            json_path = os.path.join(patient_folder, 'features.json')
            with open(json_path, 'w') as json_file:
                json.dump(json_data, json_file)

        # Clean up references to free memory
        del bottleneck_feat
        extracted_features.clear()
        del results
        torch.cuda.empty_cache()

hook_handle.remove()
