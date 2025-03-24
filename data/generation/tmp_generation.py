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
from data.multimodal_dataset import SourceType
from data.nacc_dataset import NaccDataset

class StorageFormat(Enum):
    DATASET = "dataset"     # e.g. a DataBucket with a Dataset
    LIST = "list"           # e.g. a DataBucket with a list of Tensors
    REFERENCE = "reference" # a dictionary with path pointers, etc.

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
    vae.requires_grad_(False), vae.eval(), vae.to("cuda")

    diffusion_model = load_model(
        model_type=ModelType.DIFFUSION,
        model_variant=DiTTrainingVersion.base_dit_training
    )
    diffusion_model.to("cuda")
    load_checkpoint(diffusion_model, dif_cfg.ckpt, "cuda")


    # 3) Actually generate
    gen_list = diffusion_model.generate_samples(
        n_samples=n_samples,
        data_bucket=cond_bucket,
        batch_size=batch_size,
        device=device,
        vae = vae,
    )

    # 4) Build a DataBucket
    # Actually the generate_samples will create samples in two ways: saving data on disk and so when loaded of type FOLDER, or directly returning the Databucket with in memory lists so of type LIST. For now, just LIST
    bucket = DataBucket(
        source_type=SourceType.LIST,
        label=label,
        data_list=gen_list
    )

    # 5) Put in store
    store.put_artifact(artifact_key, bucket, save_to_disk=True)
    return bucket

###############################################################################
# 5C) Single function that merges the above (Pythonic approach with a flag)
###############################################################################
def get_data_bucket(
    store: ArtifactStore,
    config: Dict[str, Any],
    generate: bool = False,
    force: bool = False
) -> DataBucket:
    """
    A single function that decides if we do "normal data" (NACC, FOLDER, LIST)
    or "generated data" (GENERATE) based on the bool 'generate'.

    If generate=True => we call load_generated_data(...).
    Else => we call load_data_from_config(...).

    This is optional. If you prefer, you can call those two separate functions
    directly from get_data_for_metric(...) or your pipeline code.
    """
    if generate:
        return load_generated_data(store, config, force=force)
    else:
        return load_data_from_config(store, config, force=force)

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