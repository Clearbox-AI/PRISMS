import os
import json
import numpy as np
import torch.distributed as dist
import torch
import glob
from typing import Optional, Tuple

from torch.utils.data import DataLoader, DistributedSampler
from sklearn.preprocessing import StandardScaler
from typing import Union
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from data.nacc_dataset import NaccDataset
from data.nacc_dataset_synth import NaccSynthDataset
from enums.data import DatasetType, ImageRange
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def _make_sampler(ds, *, shuffle: bool, drop_last: bool = True) -> Optional[torch.utils.data.Sampler]:
    if dist.is_available() and dist.is_initialized():
        return DistributedSampler(
            ds,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
    return None

def _extract_group_labels(dataset) -> np.ndarray:
    """
    Walk once through the dataset *without touching disk again* and return
    an array of 0 ( "CN") / 1 ( "AD") labels drawn from the metadata.
    """
    labels = []
    for itm in dataset:
        group = itm["metadata"].get("GROUP", "CN")
        labels.append(0 if group == "CN" else 1)
    return np.array(labels, dtype=np.int64)


def load_training_data(cfg: DictConfig, *, real: bool = True) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Build DataLoader(s) for training (and, optionally, validation).

    Behaviour
    ---------
    • If `cfg.data.split.enable` is **missing or False**  →  returns **one**
      DataLoader with the full dataset (100 % train)
    • If `cfg.data.split.enable` is **True**  →  returns a pair
      (**train_loader, val_loader**) produced with
      `sklearn.model_selection.train_test_split`, using **all** keyword
      arguments found under `cfg.data.split.kwargs` (and sensible defaults).
    """

    if real:
        if cfg.data.dataset_type.lower() == DatasetType.NACC.value:
            data_dir = cfg.data.data_dir
            stats_dir = cfg.data.get("stats_file") or Path(Path(__file__).resolve().parent, "computations")
            stats_path = Path(stats_dir, "nacc_stats.json")
            compute_dataset_stats(data_dir, stats_path)
            meta_path = Path(stats_dir, "nacc_meta.json")
            ft_path = Path(stats_dir, "tab_ft.pkl") if cfg.data.tf_train else None
            features_path = Path(stats_dir, "feature_desc.json")
            compute_dataset_meta(data_dir, meta_path, features_path)

            batch_size = cfg.data.batch_size
            num_workers = cfg.data.num_workers
            shuffle_flag = cfg.data.shuffle

            dataset = NaccDataset(
                data_dir=data_dir,
                image_height=cfg.data.image_height,
                image_width=cfg.data.image_width,
                domain=cfg.data.domain,
                do_augment=cfg.data.do_augment,
                do_image_normalize=cfg.data.do_image_normalize,
                do_tabular_normalize=cfg.data.do_tabular_normalize,
                target_channels=cfg.data.target_channels,
                final_image_range=ImageRange(cfg.data.final_image_range),
                debug=cfg.data.debug,
                stats_file=stats_path,
                meta_json=meta_path,
                tab_ft_path=ft_path,
            )
        else:
            raise NotImplementedError
    else:
        if cfg.data_synth.dataset_type.lower() == DatasetType.NACC_SYNTH.value:
            data_dir = cfg.data_synth.data_dir
            batch_size = cfg.data_synth.batch_size
            num_workers = cfg.data_synth.num_workers
            shuffle_flag = cfg.data_synth.shuffle

            dataset = NaccSynthDataset(
                data_dir=data_dir,
                debug=cfg.data_synth.debug,
            )
        else:
            raise NotImplementedError

    # ------------------------------------------------------------------ #
    # B. Decide whether we must do a train/val split
    # ------------------------------------------------------------------ #
    if real:
        split_cfg = cfg.data.get("split")
        if not split_cfg or not split_cfg.get("enable", False):
            # 100 % TRAIN
            sampler = _make_sampler(dataset, shuffle=True)
            loader  = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                sampler=sampler,
                shuffle=sampler is None and shuffle_flag,
            )
            return loader
    else:
        split_cfg = cfg.data_synth.get("split")
        if not split_cfg or not split_cfg.get("enable", False):
            # 100 % TRAIN
            sampler = _make_sampler(dataset, shuffle=True)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                sampler=sampler,
                shuffle=sampler is None and shuffle_flag,
            )
            return loader

    # ------------------------------------------------------------------ #
    # C. Build TRAIN / VAL subsets
    # ------------------------------------------------------------------ #
    # ------ 1. collect kwargs for train_test_split -----------------------
    default_kwargs = {"test_size": 0.2, "random_state": 1234, "shuffle": True}
    user_kwargs = OmegaConf.to_container(split_cfg.get("kwargs", {}), resolve=True)
    tts_kwargs = {**default_kwargs, **(user_kwargs or {})}

    # ------ 2. stratify handling (bool → labels array or None) ----------

    stratify_flag = tts_kwargs.pop("stratify", True)
    if stratify_flag:
        tts_kwargs["stratify"] = np.array([0 if el == "CN" else 1 for el in dataset.labels], dtype=np.int64)
    else:
        tts_kwargs["stratify"] = None

    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, **tts_kwargs)

    train_subset = Subset(dataset, train_idx)
    val_subset   = Subset(dataset, val_idx)

    # ------------------------------------------------------------------ #
    # D. Build DataLoaders with DDP-aware samplers
    # ------------------------------------------------------------------ #
    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    train_sampler = _make_sampler(train_subset, shuffle=True)
    val_sampler = _make_sampler(val_subset,   shuffle=False)

    train_loader = DataLoader(
        train_subset,
        sampler=train_sampler,
        shuffle=train_sampler is None and shuffle_flag,
        **common_kwargs,
    )

    val_loader = DataLoader(
        val_subset,
        sampler=val_sampler,
        shuffle=False,
        **common_kwargs,
    )

    return train_loader, val_loader


def compute_dataset_stats(data_dir: Union[Path, str], stats_path: Union[Path, str]):
    """
    Checks if 'stats_path' already exists.
    - If it does NOT exist, compute stats from 'data_dir' and write them to stats_path.
    - If it exists, do nothing (we assume it's already computed).
    """
    if os.path.exists(stats_path):
        print(f"[INFO] Stats file '{stats_path}' already exists; skipping computation.")
        return

    print(f"[INFO] Stats file '{stats_path}' not found. Computing stats...")

    patient_dirs = [
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    if not patient_dirs:
        raise ValueError(f"No patient directories found in {data_dir}")

    all_image_pixels = []
    tabular_rows = []
    tabular_scaler = StandardScaler()

    for pdir in patient_dirs:
        # 1) Load the first .npy file
        npy_files = glob.glob(os.path.join(pdir, "*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy file found in {pdir}")
        image = np.load(npy_files[0]).astype(np.float32)
        all_image_pixels.append(image.flatten())

        # 2) Load the first .json
        json_files = glob.glob(os.path.join(pdir, "*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json file found in {pdir}")
        with open(json_files[0], 'r') as f:
            jdata = json.load(f)
        tab = jdata.get("patient_id", list(jdata.values())[0])
        tab = np.array(tab, dtype=np.float32)
        tab = np.where(tab >= 9999, -1, tab)  # sentinel replacement
        tabular_rows.append(tab.reshape(1, -1))

    # Compute global image mean/std
    all_pixels = np.concatenate(all_image_pixels, axis=0)
    image_mean = float(all_pixels.mean())
    image_std = float(all_pixels.std(ddof=1))  # unbiased

    # Fit tabular scaler
    big_tab = np.concatenate(tabular_rows, axis=0)  # shape [N, feats]
    tabular_scaler.fit(big_tab)

    # Save to JSON
    stats_dict = {
        "image_mean": image_mean,
        "image_std": image_std,
        "tabular_scaler_mean_": tabular_scaler.mean_.tolist(),
        "tabular_scaler_scale_": tabular_scaler.scale_.tolist()
    }
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=4)

    print(f"[INFO] Stats file saved to {stats_path}")

import glob
import json
import os
from pathlib import Path
from typing import Union, List, Dict, Any
import numpy as np

SENTINEL_NUMERIC = -1          # -1 is used in the loader as “missing value”
SENTINEL_RAW_MIN = 9_999       # any raw value ≥ 9999 is mapped to SENTINEL_NUMERIC

def _load_feature_desc(features_path: Path) -> List[Dict[str, Any]]:
    """Load and sanity-check the feature description JSON."""
    if not features_path.exists():
        raise FileNotFoundError(f"Feature description file not found: {features_path}")

    with features_path.open("r", encoding="utf-8") as f:
        desc = json.load(f)

    if not isinstance(desc, list) or not all("feature" in d for d in desc):
        raise ValueError(
            f"{features_path} must contain a list of objects with at least a "
            "'feature' field"
        )
    return desc


def _collect_tabular_data(data_dir: Path, num_feats: int) -> np.ndarray:
    """Read every patient *.json file under data_dir into one big array."""
    patient_dirs = [
        p for p in (d for d in data_dir.iterdir() if d.is_dir())
        if any(p.glob("*.json"))
    ]
    if not patient_dirs:
        raise ValueError(f"No patient directories with JSON files found in {data_dir}")

    rows: List[np.ndarray] = []
    for pdir in patient_dirs:
        json_files = list(pdir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No *.json file found in {pdir}")

        with json_files[0].open("r", encoding="utf-8") as f:
            jdata = json.load(f)

        # Robust handling: if the patient JSON uses a keyed structure, grab the
        # first value; if it is already a flat list, use it directly.
        if isinstance(jdata, dict):
            patient_tab = next(iter(jdata.values()))
        else:
            patient_tab = jdata

        arr = np.asarray(patient_tab, dtype=np.float32)

        # Replace out-of-band sentinels (raw ≥ 9999) with the common sentinel -1
        arr = np.where(arr >= SENTINEL_RAW_MIN, SENTINEL_NUMERIC, arr)

        if arr.ndim != 1 or arr.size != num_feats:
            raise ValueError(
                f"Expected {num_feats} features but got array with shape {arr.shape} "
                f"in file {json_files[0]}"
            )

        rows.append(arr.reshape(1, -1))

    return np.concatenate(rows, axis=0)   # shape (N, num_feats)


def _infer_sign(min_val: float, max_val: float) -> str:
    """Return 'non-negative', 'non-positive', or 'mixed'."""
    if min_val >= 0:
        return "non-negative"
    if max_val <= 0:
        return "non-positive"
    return "mixed"


def compute_dataset_meta(
    data_dir: Union[Path, str],
    meta_path: Union[Path, str],
    features_path: Union[Path, str],
) -> None:
    """
    Build a *meta.json* file containing basic statistics for every tabular
    feature.
    The JSON structure is a list of dictionaries, **preserving the order**
    from *feature_desc.json*:

    ```json
    [
      {
        "feature": "AGE",
        "description": "Age in years at MRI session",
        "min": 40.0,
        "max": 92.0,
        "sign": "non-negative"
      },
      …
    ]
    ```

    Parameters
    ----------
    data_dir
        Root directory containing one sub-directory per patient with a single
        `*.json` file inside.
    meta_path
        Where to write the generated metadata.  If the file already exists,
        the function exits early.
    features_path
        Path to *feature_desc.json* with at least
        `[{"feature": "...", "description": "..."}]`.
    """
    data_dir = Path(data_dir)
    meta_path = Path(meta_path)
    features_path = Path(features_path)

    # 1) Bail out if meta already exists
    if meta_path.exists():
        print(f"[compute_dataset_meta] Meta file already present → {meta_path}")
        return

    # 2) Read feature descriptions
    feature_desc = _load_feature_desc(features_path)
    feature_names = [d["feature"] for d in feature_desc]

    # 3) Gather the entire tabular matrix
    big_tab = _collect_tabular_data(data_dir, len(feature_names))   # shape (N, F)

    # 4) Build the meta information per feature
    meta: List[Dict[str, Any]] = []
    for idx, fd in enumerate(feature_desc):
        col = big_tab[:, idx]

        # Mask out sentinel/missing entries
        valid = col != SENTINEL_NUMERIC
        if not valid.any():
            raise ValueError(
                f"All values missing for feature '{fd['feature']}' – cannot "
                "compute statistics."
            )

        col_valid = col[valid]
        fmin = float(col_valid.min())
        fmax = float(col_valid.max())

        meta.append(
            {
                "feature": fd["feature"],
                "description": fd.get("description", ""),
                "min": fmin,
                "max": fmax,
                "sign": _infer_sign(fmin, fmax),
            }
        )

    # 5) Persist to disk
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[compute_dataset_meta] Wrote {len(meta)} feature entries to {meta_path}")

if __name__ == "__main__":

    from utils.configurations import set_project_root
    set_project_root()

    def compute_stats(loader, max_batches=20):
        """
        Iterates through `max_batches` of the DataLoader, stores all images in a large tensor,
        and computes:
          - Global mean & variance
          - Per-channel mean & variance
        """
        latents = []

        for i, batch in enumerate(loader):
            images = batch["image"]  # Shape: [B, C, H, W]
            latents.append(images)

            if i + 1 >= max_batches:
                break  # Stop after `max_batches`

        # Stack all collected images into a single tensor
        latents = torch.cat(latents, dim=0)  # Shape: [Total_B, C, H, W]

        # Compute statistics
        global_mean = torch.mean(latents)
        global_var = torch.var(latents, unbiased=True)
        channel_mean = torch.mean(latents, dim=(0, 2, 3))  # Shape: [C]
        channel_var = torch.var(latents, dim=(0, 2, 3), unbiased=True)  # Shape: [C]

        return global_mean, global_var, channel_mean, channel_var


    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "trainers"))):
        cfg = compose(config_name="base_dit_training")  # Adjust if needed
        OmegaConf.set_struct(cfg, False)

        print("[INFO] Loading DataLoader...")
        loader = load_training_data(cfg)

        # Compute statistics over 20 batches
        global_mean, global_var, channel_mean, channel_var = compute_stats(loader, max_batches=2000)

        print(f"[INFO] DataLoader Global Mean: {global_mean.item()}")
        print(f"[INFO] DataLoader Global Variance: {global_var.item()}")
        print(f"[INFO] DataLoader Per-Channel Mean: {channel_mean.tolist()}")
        print(f"[INFO] DataLoader Per-Channel Variance: {channel_var.tolist()}")