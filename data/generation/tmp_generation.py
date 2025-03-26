import os
from enum import Enum
from typing import Any, Dict, List, Optional
from utils.configurations import load_hydra_config
from models.vae.vae import decode_latents, load_vae
from utils.model import load_checkpoint
from models.utils.model_loader import load_model
from enums.models.model_types import ModelType
from enums.training_versions import DiTTrainingVersion


from data.artifact_store import ArtifactStore
from data.multimodal_dataset import DataBucket
from enums.models.diffusion import DataLabel
from data.nacc_dataset import NaccDataset

from enums.generation import SourceType, DataLabel, StorageFormat




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
    all_images = result_bucket.data_list
    all_tab_data = input_bucket.data_list

    if len(all_images) != len(all_tab_data):
        raise ValueError("Mismatch in length of result_bucket and input_bucket data.")

    # count how many times actually save an item
    num_saved = 0

    for cond_key, indices in cond_mapping.items():
        folder_name = f"patient_{str(cond_key)}"
        folder_path = os.path.join(save_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # For each index in the list, create separate tab_data_X.json / img_data_X.npy
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


def load_data_from_config(
    store: ArtifactStore,
    cfg: Dict[str, Any],
    force: bool = False
) -> DataBucket:
    """
    If artifact_key is in store, return it (unless force=True).
    Else create a DataBucket from the config. We FOLDER.
    """
    artifact_key = cfg["artifact_key"]
    if not force:
        existing = store.get_artifact(artifact_key) or store.load_artifact(artifact_key)
        if existing is not None:
            return existing

    # Not found => create
    source_type = SourceType(cfg["source_type"])
    label = DataLabel(cfg["data_label"])

    if source_type == SourceType.FOLDER:
        folder_path = cfg["folder_path"]
        bucket = DataBucket(SourceType.FOLDER, label, folder_path=folder_path)
    elif source_type == SourceType.NACC:
        # build the underlying NaccDataset from config
        nacc_params = cfg.get("nacc_params", {})
        nacc_ds = NaccDataset(**nacc_params)
        bucket = DataBucket(
            source_type=SourceType.NACC,
            label=label,
            nacc_dataset=nacc_ds
        )
    else:
        raise ValueError(f"load_data_from_config only handles FOLDER. NACC and LIST are not available from config. Got {source_type}.")

    store.put_artifact(artifact_key, bucket, save_to_disk=True)
    return bucket


def load_generated_data(
        store: ArtifactStore,
        condition_bucket: Optional[DataBucket],
        cfg: Optional[Dict[str, Any]],
        force: bool = False
) -> DataBucket:
    """
    """
    artifact_key = cfg["artifact_key"]
    if not force:
        existing = store.get_artifact(artifact_key) or store.load_artifact(artifact_key)
        if existing is not None:
            return existing

    # Not found => generate
    label = DataLabel(cfg["data_label"])
    n_samples = cfg.get("n_samples", 10)
    batch_size = cfg.get("batch_size", 4)
    device = cfg.get("device", "cuda")

    # 1) Possibly load conditioning data
    cond_cfg = cfg.get("conditioning_data", None)
    cond_bucket = None

    """
    an example of cond_cfg might look like:
    cond_cfg = {
        "folder_path": "blabla",
        "artifact_key": "some_artifact_key",
        "source_type": "folder",
        "data_label": "tab"
    }
    """

    if cond_cfg is not None and not condition_bucket:
        if cond_cfg["source_type"] == "folder":
            cond_bucket = load_data_from_config(store, cond_cfg, force=False)

    dif_cfg = load_hydra_config("metrics", "FID").diffusion
    vae_cfg = load_hydra_config("models", "vae")

    vae = load_vae(vae_cfg)
    vae.requires_grad_(False), vae.eval(), vae.to(device)

    diffusion_model = load_model(
        model_type=ModelType.DIFFUSION,
        model_variant=DiTTrainingVersion.base_dit_training
    )
    diffusion_model.to(device)
    load_checkpoint(diffusion_model, dif_cfg.ckpt, device)


    # 3) Actually generate
    gen_list = diffusion_model.generate_samples(
        n_samples=n_samples,
        data_bucket=cond_bucket,
        batch_size=batch_size,
        device=device,
        vae = vae,
    )

    # 4) Build a DataBucket
    # Actually the generate_samples will create samples in two ways: saving data on disk and so when loaded of type FOLDER, or directly returning the Databucket with in memory lists so of type LIST (using the save_generated_data function)
    bucket = DataBucket(
        source_type=SourceType.LIST,
        label=label,
        data_list=gen_list
    )

    # 5) Put in store
    store.put_artifact(artifact_key, bucket, save_to_disk=True)
    return bucket


###############################################################################
# 6) get_data_for_metric => typical usage from the "metric manager"
###############################################################################
def get_data_for_metric(
    store: ArtifactStore,
    metric_name: str,
    config: Dict[str, Any],
    force: bool = False
) -> DataBucket:
    """
    Suppose your "metric manager" calls this function with a config specifying
    how to get data. The config might say:
      {
        "artifact_key": "...",
        "source_type": "nacc" | "folder" | "list",
        "data_label": "image" | "tab" | "both",
        "folder_path": "blabla"
        ...
      }
    """
    source_type_str = config["source_type"]


    cond_data = load_data_from_config(store, config, force=force)

    generated_data = load_generated_data(store=store, condition_bucket=cond_data, cfg=None, force=force)

    return ...


def get_model_for_metric(
    store: ArtifactStore,
    model_cfg: Dict[str, Any],
    training_data: Optional[DataBucket] = None,
    force: bool = False
):
    """
    Stub function: loads or trains an ML model based on the config.
    - model_cfg should contain keys like "artifact_key" and instructions
      about training or loading a checkpoint.
    - training_data (DataBucket) is used if we need to do training.
    """

    def load_my_model_from_disk():
        ...
    def train_my_model():
        ...

    artifact_key = model_cfg["artifact_key"]
    if not force:
        existing = store.get_artifact(artifact_key) or store.load_artifact(artifact_key)
        if existing is not None:
            return existing

    # If we reach here, no artifact in store => we either train or load
    load_ckpt_path = model_cfg.get("load_checkpoint_path", None)
    if load_ckpt_path is not None:
        # Actually load from disk
        model = load_my_model_from_disk(load_ckpt_path)
    else:
        # Possibly train
        model = train_my_model(training_data, model_cfg)

    # Put into store
    store.put_artifact(artifact_key, model, save_to_disk=True)
    return model


"""
example of chaining requests:

cond_data = get_data_for_metric(...)
gen_data = load_generated_data(store, cond_data, gen_cfg, force=False)
model = get_model_for_metric(store, model_cfg, training_data=gen_data)
metric_val= compute_my_metric(model, gen_data, additional_args...)
"""

import os
import shutil
import tempfile
import pickle
import json
import numpy as np
import torch

def test_load_data_from_config_folder():
    """
    Tests the load_data_from_config function with a FOLDER source type.
    """
    print("[test_load_data_from_config_folder] Starting...")
    tmp_dir = tempfile.mkdtemp(prefix="test_folder_")

    try:
        # Create a small subfolder to simulate data
        patient_0_path = os.path.join(tmp_dir, "patient_000")
        os.makedirs(patient_0_path, exist_ok=True)

        # Save a .npy image
        np.save(os.path.join(patient_0_path, "img.npy"), np.random.randn(3, 32, 32))
        # Save a .json tab file
        with open(os.path.join(patient_0_path, "data.json"), "w") as f:
            json.dump([0.1, 0.2], f)

        cfg = {
            "artifact_key": "some_folder_data",
            "source_type": "folder",
            "data_label": "tab",
            "folder_path": tmp_dir
        }

        store = ArtifactStore()
        bucket = load_data_from_config(store, cfg, force=False)

        assert isinstance(bucket, DataBucket), "Returned object must be a DataBucket."
        assert len(bucket) == 1, "We expect exactly one 'patient_000' directory."
        # Check that the artifact is now in store
        assert store.has_artifact("some_folder_data"), "Artifact should be stored."
        item = bucket[0]
        # item["tabular"] is a torch.Tensor of shape [2], item["image"] might be None because data_label=tab
        assert item["tabular"] is not None, "Should have loaded tabular data."
        print("[test_load_data_from_config_folder] Passed.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_save_generated_data():
    """
    Tests that we can save in-memory data from a result_bucket and input_bucket
    to a folder using save_generated_data.
    """
    print("[test_save_generated_data] Starting...")
    tmp_dir = tempfile.mkdtemp(prefix="generated_data_test_")

    try:
        # Make a DataBucket with images
        images = [torch.randn(3, 64, 64) for _ in range(5)]  # 5 images
        result_bucket = DataBucket(
            source_type=SourceType.LIST,
            label=DataLabel.IMAGE,
            data_list=images
        )
        # Make a DataBucket with tabular data
        tabs = [torch.randn(10) for _ in range(5)]  # 5 tab items
        input_bucket = DataBucket(
            source_type=SourceType.LIST,
            label=DataLabel.TAB,
            data_list=tabs
        )

        # cond_mapping => let's say we have "patientA" => indices [0,1], "patientB" => [2,3,4]
        cond_mapping = {
            "patientA": [0,1],
            "patientB": [2,3,4]
        }

        artifact_ref = save_generated_data(
            result_bucket=result_bucket,
            cond_mapping=cond_mapping,
            input_bucket=input_bucket,
            save_path=tmp_dir
        )

        # Check the structure
        assert "save_path" in artifact_ref
        assert "num_samples" in artifact_ref
        assert artifact_ref["num_samples"] == 5, "We saved 5 total items"

        patientA_path = os.path.join(tmp_dir, "patient_patientA")
        patientB_path = os.path.join(tmp_dir, "patient_patientB")
        assert os.path.isdir(patientA_path), "Folder for patientA must exist"
        assert os.path.isdir(patientB_path), "Folder for patientB must exist"

        # For patientA, we have tab_data_1.json, tab_data_2.json, img_data_1.npy, img_data_2.npy
        filesA = sorted(os.listdir(patientA_path))
        assert len(filesA) == 4, "Expect 2 tab + 2 npy in patientA's folder"
        print("[test_save_generated_data] Passed.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():

    """
    Entry point to run all tests from a single script.
    """
    print("===== Running All Tests =====")
    # 2. Data loading test
    test_load_data_from_config_folder()
    # 3. Generated data saving test
    test_save_generated_data()

    print("===== All Tests Passed Successfully! =====")


if __name__ == "__main__":
    main()