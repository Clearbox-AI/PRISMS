import json
import os
import numpy as np
import torch
import glob

from typing import Optional, Tuple, Any, Dict, List
from omegaconf import DictConfig
from pathlib import Path
from hydra import compose, initialize_config_dir

from utils.data import infinite_loader, DataLabel
from models.dit.dit_multimodal import load_dit
from models.vae.vae import decode_latents, load_vae
from data.loader import load_training_data
from models.diffusion.diffusion_multimodal import load_diffusion
from data.artifact_store import ArtifactStore

from data.multimodal_dataset import SourceType
from data.multimodal_dataset import DataBucket

def save_generated_data(
    result_bucket: DataBucket,
    cond_mapping: Dict[str, List[int]],
    input_bucket: DataBucket,
    save_path: str
) -> Dict[str, Any]:
    """
    Saves generated images + conditioning tabular data in a custom structure:
      save_path/
        patient_{cond_key}/
          tab_data_i.json
          img_data_i.npy
          ...
    Returns a dict "artifact_ref" with:
      - "save_path"
      - "num_samples"
    which can be used to store this reference in the ArtifactStore if desired.
    """

    os.makedirs(save_path, exist_ok=True)

    # 1) For safety, ensure we are dealing with in-memory lists
    if result_bucket.source_type != SourceType.LIST:
        raise ValueError(
            "save_generated_data currently expects result_bucket.source_type=LIST (in-memory). "
            f"Got {result_bucket.source_type}."
        )
    if input_bucket.source_type != SourceType.LIST:
        raise ValueError(
            "save_generated_data currently expects input_bucket.source_type=LIST. "
            f"Got {input_bucket.source_type}."
        )

    # 2) Grab the underlying lists
    all_images = result_bucket.data_list  # list of Tensors or arrays
    all_tab_data = input_bucket.data_list

    if len(all_images) != len(all_tab_data):
        raise ValueError("Mismatch in length of result_bucket and input_bucket data.")

    # We'll count how many times we actually save an item
    num_saved = 0

    for cond_key, indices in cond_mapping.items():
        folder_name = f"patient_{str(cond_key)}"
        folder_path = os.path.join(save_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # For each index in the list, we create separate tab_data_X.json / img_data_X.npy
        for i, idx in enumerate(indices, start=1):
            # tab_data => tab_data_i.json
            tab_entry = all_tab_data[idx]  # either a Tensor or list/ndarray
            if isinstance(tab_entry, torch.Tensor):
                tab_entry = tab_entry.cpu().numpy().tolist()
            elif isinstance(tab_entry, np.ndarray):
                tab_entry = tab_entry.tolist()

            tab_path = os.path.join(folder_path, f"tab_data_{i}.json")
            with open(tab_path, "w") as f:
                json.dump(tab_entry, f, indent=2)

            # image => img_data_{i}.npy
            img_entry = all_images[idx]
            if isinstance(img_entry, torch.Tensor):
                img_entry = img_entry.cpu().numpy()

            img_path = os.path.join(folder_path, f"img_data_{i}.npy")
            np.save(img_path, img_entry)

            num_saved += 1

    return {
        "save_path": save_path,
        "num_samples": num_saved
    }

# def save_generated_data(
#     result_bucket: DataBucket,
#     cond_mapping: Dict[str, List[int]],
#     input_bucket: DataBucket,
#     save_path: str
# ) -> Dict[str, Any]:
#     """
#     Saves generated images + conditioning tabular data in a custom structure:
#       save_path/
#         patient_{cond_key}/
#           tab_data_i.json
#           img_data_i.npy
#           ...
#     Returns a small dict "artifact_ref" with:
#       - "save_path"
#       - "num_samples"
#     It can be used to store this reference in the ArtifactStore if desired.
#     """
#     os.makedirs(save_path, exist_ok=True)
#
#     # We'll assume result_bucket.data_source is an indexable list of image Tensors
#     all_images = result_bucket.data_source
#     all_tab_data = input_bucket.data_source
#
#     num_saved = 0
#
#     for cond_key, indices in cond_mapping.items():
#         folder_name = f"patient_{str(cond_key)}"
#         folder_path = os.path.join(save_path, folder_name)
#         os.makedirs(folder_path, exist_ok=True)
#
#         for i, idx in enumerate(indices, start=1):
#             # tab_data => tab_data_i.json
#             tab_entry = all_tab_data[idx]
#             tab_path = os.path.join(folder_path, f"tab_data_{i}.json")
#             if isinstance(tab_entry, torch.Tensor):
#                 tab_entry = tab_entry.cpu().numpy().tolist()
#             elif isinstance(tab_entry, np.ndarray):
#                 tab_entry = tab_entry.tolist()
#
#             with open(tab_path, "w") as f:
#                 json.dump(tab_entry, f, indent=2)
#
#             # image => img_data_i.npy
#             img_entry = all_images[idx]
#             if isinstance(img_entry, torch.Tensor):
#                 img_entry = img_entry.cpu().numpy()
#             img_path = os.path.join(folder_path, f"img_data_{i}.npy")
#             np.save(img_path, img_entry)
#             num_saved += 1
#
#     return {
#         "save_path": save_path,
#         "num_samples": num_saved
#     }


# def load_generated_data(
#     artifact_ref: Dict[str, Any],
#     lazy: bool = True
# ) -> DataBucket:
#     """
#     Given an 'artifact_ref' with "save_path", load the generated images.
#     If lazy=True, we create a LazyNpyImageDataset that loads each .npy on demand.
#     If lazy=False, we load everything into memory at once (returns a list of Tensors).
#     For now, only loading the images. TODO: add multiomdality
#     """
#     save_path = artifact_ref["save_path"]
#     npy_files = glob.glob(os.path.join(save_path, "**", "img_data_*.npy"), recursive=True)
#     npy_files.sort()
#
#     if lazy:
#         # Just store a LazyNpyImageDataset that references all .npy files
#         dataset = LazyNpyImageDataset(npy_files)
#         return DataBucket(dataset, label=DataLabel.IMAGE)
#     else:
#         # Eager load: read all .npy into memory
#         all_images = []
#         for npy_f in npy_files:
#             arr = np.load(npy_f)
#             tensor = torch.from_numpy(arr)
#             all_images.append(tensor)
#         return DataBucket(all_images, label=DataLabel.IMAGE)
#
#
# def create_or_load_generated_data(store: ArtifactStore, key: str, lazy: bool) -> Any:
#     """
#     Either fetch data from store if it exists,
#     or generate & save it if not present. Then store a reference (or an in-memory Bucket).
#
#     The 'lazy' parameter controls whether we store a lazy reference (so we can load images on-demand)
#     or we store the entire data in memory.
#     """
#     # 1) If already in memory store, return it
#     if store.has_artifact(key):
#         return store.get_artifact(key)
#
#     # 2) Otherwise, check if we have the data on disk (in some known path).
#     #    If it exists, we skip generation and just build an artifact_ref or load it.
#     save_path = "/tmp/generated_lazy_demo"
#     already_exists = os.path.exists(save_path) and len(os.listdir(save_path)) > 0
#
#     if already_exists:
#         # We just create a reference object
#         artifact_ref = {
#             "save_path": save_path,
#             "num_samples": 9999,  # You could compute if you want
#         }
#         # If we want to store lazy references in the store => load or not:
#         if lazy:
#             lazy_bucket = load_generated_data(artifact_ref, lazy=True)
#             store.put_artifact(key, lazy_bucket, save_to_disk=True)
#             return lazy_bucket
#         else:
#             # Eager load everything
#             full_bucket = load_generated_data(artifact_ref, lazy=False)
#             store.put_artifact(key, full_bucket, save_to_disk=True)
#             return full_bucket
#
#     # 3) If we get here, we generate from scratch
#     diffusion_model = create_diffusion_model()
#     vae = DummyVAE()
#
#     # Maybe we also retrieve the tabular data from the store or create it anew
#     tab_data_bucket = store.get_or_create_artifact(
#         "my_tab_data", creator_fn=create_tab_data_bucket, force=False, save_to_disk=False
#     )
#
#     # Actually generate
#     gen_bucket, cond_map, input_bucket = diffusion_model.generate_samples(
#         n_samples=10,
#         data_bucket=tab_data_bucket,
#         vae=vae,
#         batch_size=5,
#         device="cpu"
#     )
#
#     # Now we can save it to disk with our specialized function
#     artifact_ref = save_generated_data(gen_bucket, cond_map, input_bucket, save_path)
#
#     # Decide if we store a lazy or eager version in the store
#     if lazy:
#         # We'll store a lazy DataBucket in the store
#         lazy_bucket = load_generated_data(artifact_ref, lazy=True)
#         store.put_artifact(key, lazy_bucket, save_to_disk=True)
#         return lazy_bucket
#     else:
#         full_bucket = load_generated_data(artifact_ref, lazy=False)
#         store.put_artifact(key, full_bucket, save_to_disk=True)
#         return full_bucket


# def save_generated_data(
#     result_bucket,
#     cond_mapping: Dict[str, List[int]],
#     input_bucket,
#     save_path: str
# ) -> None:
#     """
#     Saves generated images (from `result_bucket`) and their corresponding
#     input tabular data (from `input_bucket`) on disk based on `cond_mapping`.
#     Each sample is saved in its own pair of files: tab_data_{i}.json and img_data_{i}.npy.
#
#     Args:
#         result_bucket: DataBucket containing generated images (DataLabel.IMAGE),
#                        typically a list (or tensor) of Tensors, each shaped [C,H,W].
#         cond_mapping: Dict mapping condition_key -> list of generated sample indices.
#         input_bucket: DataBucket containing the tabular data (DataLabel.TAB)
#                       that was actually used for generation.
#         save_path: Root directory where data will be saved. If it doesn't exist,
#                    this function will create it.
#     """
#     os.makedirs(save_path, exist_ok=True)
#
#     # We'll assume `result_bucket.data_source` and `input_bucket.data_source` are lists
#     # (or similar) of length = total #samples. The i-th item in each corresponds
#     # to sample i.
#     all_images = result_bucket.data_source
#     all_tab_data = input_bucket.data_source
#
#     # For each condition key and all its associated sample indices
#     for key, indices in cond_mapping.items():
#         folder_name = f"patient_{str(key)}" if str(key).isdigit() else os.path.basename(str(key))
#         folder_path = os.path.join(save_path, folder_name)
#         os.makedirs(folder_path, exist_ok=True)
#
#         # For each sample index for this condition key, save tab_data_{i}.json and img_data_{i}.npy
#         for i, idx in enumerate(indices, start=1):
#             # 1) Save the tabular row to tab_data_{i}.json
#             tab_data_path = os.path.join(folder_path, f"tab_data_{i}.json")
#
#             # Grab the i-th row for that key (assuming torch.Tensor or np.array)
#             tab_entry = all_tab_data[idx]
#             if isinstance(tab_entry, torch.Tensor):
#                 tab_list = tab_entry.detach().cpu().numpy().tolist()
#             elif isinstance(tab_entry, np.ndarray):
#                 tab_list = tab_entry.tolist()
#             else:
#                 # If it's already a Python list or some other structure
#                 tab_list = tab_entry
#
#             with open(tab_data_path, "w") as f:
#                 json.dump(tab_list, f, indent=2)
#
#             # 2) Save the corresponding generated image to img_data_{i}.npy
#             img_npy_path = os.path.join(folder_path, f"img_data_{i}.npy")
#             image_entry = all_images[idx]
#             if isinstance(image_entry, torch.Tensor):
#                 image_np = image_entry.detach().cpu().numpy()
#             else:
#                 image_np = image_entry
#             np.save(img_npy_path, image_np)

if __name__ == "__main__":
    from utils.configurations import set_project_root
    set_project_root()

    # 1) Load config(s)
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
        dit_cfg = compose(config_name="base_dit_training")
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
        diffusion_cfg = compose(config_name="diffusion")

    # 2) Load the underlying DiT model
    dit_model = load_dit(dit_cfg)

    # 3) Build the diffusion model
    diffusion_model = load_diffusion(diffusion_cfg, dit_model).cuda()

    # 4) Load VAE
    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "models"))):
        vae_cfg = compose(config_name="vae")
    vae = load_vae(vae_cfg)
    vae.requires_grad_(False)
    vae.eval()
    vae.to("cuda")

    # 5) Load train dataloader
    train_dataloader = load_training_data(dit_cfg)

    # --------------------------
    # EXAMPLE 1: Using a DataLoader
    # --------------------------
    tab_bucket_dataloader = DataBucket(data_source=train_dataloader, label=DataLabel.TAB)

    # Generate 5 images from this DataLoader
    gen_bucket_dl, cond_map_dl, source_data_bucket = diffusion_model.generate_samples(
        n_samples=9,
        data_bucket=tab_bucket_dataloader,
        vae=vae,
        batch_size=2,
        cfg=1.5,
        steps=6,
        device='cuda'
    )

    save_generated_data(
        result_bucket=gen_bucket_dl,
        cond_mapping=cond_map_dl,
        input_bucket=source_data_bucket,
        save_path="/mnt/dataset_storage/data/nacc_dataset/generated_data"
    )