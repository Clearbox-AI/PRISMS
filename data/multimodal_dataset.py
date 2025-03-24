import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from enum import Enum
from typing import Union, List, Optional, Dict, Any
from data.nacc_dataset import NaccDataset
from torch.utils.data import DataLoader

class SourceType(Enum):
    NACC = "nacc"   # wraps a NaccDataset
    FOLDER = "folder"  # raw "patient_xxx" folder approach
    LIST = "list"      # in-memory list of items

class DataLabel(Enum):
    IMAGE = "image"
    TAB = "tab"
    BOTH = "both"



# TODO: maybe lazy when the input is a dataset and a path, impossible when it's already some tensor list
# TODO: not os.path.join but pathlib.Path


class DataBucket(Dataset):
    """
    This single Dataset class can unify:
      1) an existing NaccDataset,
      2) a folder on disk (patient_x subdirs) => path-based,
      3) a list of items in memory.

    So, the constructor takes:
      source_type: one of (NACC, PATH, LIST)
      label: one of (IMAGE, TAB, BOTH)
      nacc_dataset: an already-initialized NaccDataset (if source_type=NACC)
      folder_path: a str path to the folder (if source_type=PATH)
      data_list:   a list of items (if source_type=LIST)

    Then, in __getitem__ and __len__, we unify all 3 approaches to
    produce a dict: {"image":..., "tabular":..., "dir":...} as needed.
    """

    def __init__(
            self,
            source_type: SourceType,
            label: DataLabel,
            nacc_dataset: Optional[NaccDataset] = None,
            folder_path: Optional[str] = None,
            data_list: Optional[List[Any]] = None,
            metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.source_type = source_type
        self.label = label

        self.nacc_dataset = nacc_dataset
        self.folder_path = folder_path
        self.data_list = data_list
        self.metadata = metadata or {}

        # we might store some "index metadata" if path-based, etc.
        if self.source_type == SourceType.NACC:
            if not self.nacc_dataset:
                raise ValueError("source_type=NACC but nacc_dataset not provided")

        elif self.source_type == SourceType.FOLDER:
            if not self.folder_path:
                raise ValueError("source_type=FOLDER but folder_path not provided")
            # discover subfolders
            self.patient_dirs = [
                os.path.join(self.folder_path, d)
                for d in sorted(os.listdir(self.folder_path))
                if os.path.isdir(os.path.join(self.folder_path, d))
            ]
        elif self.source_type == SourceType.LIST:
            if self.data_list is None:
                raise ValueError("source_type=LIST but data_list not provided")

        else:
            raise ValueError(f"Unknown source_type: {self.source_type}")

    def __len__(self) -> int:
        if self.source_type == SourceType.NACC:
            return len(self.nacc_dataset)
        elif self.source_type == SourceType.FOLDER:
            return len(self.patient_dirs)
        elif self.source_type == SourceType.LIST:
            return len(self.data_list)
        return 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dict with:
          {
            "image": torch.Tensor or None,
            "tabular": torch.Tensor or None,
            "dir": str or None
          }
        """
        # We'll unify the result in a dict: "image", "tabular", "dir"
        out = {"image": None, "tabular": None, "dir": None}

        if self.source_type == SourceType.NACC:
            # We rely on the nacc_dataset to yield a dict
            sample = self.nacc_dataset[idx]
            # sample might have "image", "tabular", "dir". We unify:
            out["image"] = sample.get("image", None) if self.label in (DataLabel.IMAGE, DataLabel.BOTH) else None
            out["tabular"] = sample.get("tabular", None) if self.label in (DataLabel.TAB, DataLabel.BOTH) else None
            out["dir"] = sample.get("dir", None)

        elif self.source_type == SourceType.FOLDER:
            # We'll do a "folder approach" like the typical "patient_x" logic
            dir_path = self.patient_dirs[idx]
            out["dir"] = dir_path
            # If label=IMAGE or BOTH, read .npy
            if self.label in (DataLabel.IMAGE, DataLabel.BOTH):
                npy_files = [f for f in os.listdir(dir_path) if f.endswith(".npy")]
                if len(npy_files) == 0:
                    raise FileNotFoundError(f"No .npy found in {dir_path}")
                img_path = os.path.join(dir_path, npy_files[0])
                arr = np.load(img_path)
                out["image"] = torch.from_numpy(arr).float()

            # If label=TAB or BOTH, read .json
            if self.label in (DataLabel.TAB, DataLabel.BOTH):
                json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
                if len(json_files) == 0:
                    raise FileNotFoundError(f"No .json found in {dir_path}")
                j_path = os.path.join(dir_path, json_files[0])
                with open(j_path, "r") as f:
                    jdata = json.load(f)
                if isinstance(jdata, dict):
                    # pick the first value or do your logic
                    jdata = next(iter(jdata.values()))
                tab_tensor = torch.tensor(jdata, dtype=torch.float32)
                out["tabular"] = tab_tensor

        elif self.source_type == SourceType.LIST:
            item = self.data_list[idx]
            # If label=IMAGE, item might be an image Tensor or a dict
            if self.label == DataLabel.IMAGE:
                if isinstance(item, dict):
                    out["image"] = item.get("image", None)
                    out["dir"] = item.get("dir", None)
                else:
                    # treat item as an image Tensor
                    out["image"] = item
            elif self.label == DataLabel.TAB:
                if isinstance(item, dict):
                    out["tabular"] = item.get("tabular", None)
                    out["dir"] = item.get("dir", None)
                else:
                    out["tabular"] = item
            elif self.label == DataLabel.BOTH:
                # item might be a dict with "image" / "tabular"
                # or a tuple (img, tab)
                if isinstance(item, dict):
                    out["image"] = item.get("image", None)
                    out["tabular"] = item.get("tabular", None)
                    out["dir"] = item.get("dir", None)
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    out["image"] = item[0]
                    out["tabular"] = item[1]
                else:
                    raise ValueError("Expected a dict or 2-tuple for BOTH label in LIST mode.")
            else:
                raise ValueError(f"Unsupported label: {self.label}")

        return out

    def get_dataloader(self, batch_size: int = 4, shuffle: bool = False) -> DataLoader:
        """
        Convenience method to get a DataLoader from this dataset.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


# class DataBucket:
#     """
#     A wrapper around a MultimodalDataset.
#     Other code can just call get_dataset() or get_dataloader().
#     """
#     def __init__(self, dataset: MultimodalDataset):
#         self.dataset = dataset
#         self.label = dataset.label  # optional, just for reference
#
#     def get_dataset(self) -> Dataset:
#         return self.dataset
#
#     def get_dataloader(self, batch_size: int = 4, shuffle: bool = False) -> DataLoader:
#         return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
#
#     def __len__(self):
#         return len(self.dataset)


def main():
    from utils.configurations import set_project_root
    from utils.configurations import load_hydra_config
    from data.loader import load_training_data

    set_project_root()
    dit_cfg = load_hydra_config("trainers", "base_dit_training")
    train_dataloader = load_training_data(dit_cfg)

    # -------------------------------------------------------
    # A) Test the NACC source mode (using the existing NaccDataset)
    # -------------------------------------------------------
    # The dataloader itself has an underlying dataset, which should be a NaccDataset.
    nacc_dataset = train_dataloader.dataset  # assuming it's a NaccDataset instance

    print("=== NACC mode: IMAGE only ===")
    ds_nacc_image = MultimodalDataset(
        source_type=SourceType.NACC,
        label=DataLabel.IMAGE,
        nacc_dataset=nacc_dataset
    )
    print(f"Length of NACC IMAGE dataset: {len(ds_nacc_image)}")
    sample = ds_nacc_image[0]
    print("Sample keys:", sample.keys())
    print("image shape:", None if sample["image"] is None else sample["image"].shape)
    print("tabular:", sample["tabular"])  # should be None in IMAGE-only mode
    print("dir:", sample["dir"], "\n")

    print("=== NACC mode: TAB only ===")
    ds_nacc_tab = MultimodalDataset(
        source_type=SourceType.NACC,
        label=DataLabel.TAB,
        nacc_dataset=nacc_dataset
    )
    print(f"Length of NACC TAB dataset: {len(ds_nacc_tab)}")
    sample = ds_nacc_tab[0]
    print("Sample keys:", sample.keys())
    print("image:", sample["image"])  # should be None in TAB-only mode
    print("tabular shape:", None if sample["tabular"] is None else sample["tabular"].shape)
    print("dir:", sample["dir"], "\n")

    print("=== NACC mode: BOTH ===")
    ds_nacc_both = MultimodalDataset(
        source_type=SourceType.NACC,
        label=DataLabel.BOTH,
        nacc_dataset=nacc_dataset
    )
    print(f"Length of NACC BOTH dataset: {len(ds_nacc_both)}")
    sample = ds_nacc_both[0]
    print("Sample keys:", sample.keys())
    print("image shape:", None if sample["image"] is None else sample["image"].shape)
    print("tabular shape:", None if sample["tabular"] is None else sample["tabular"].shape)
    print("dir:", sample["dir"], "\n")

    # -------------------------------------------------------
    # B) Test the PATH source mode
    # -------------------------------------------------------
    path_to_data = "/mnt/dataset_storage/data/nacc_dataset/nacc_subset/middle_slice"

    print("=== PATH mode: IMAGE only ===")
    ds_path_image = MultimodalDataset(
        source_type=SourceType.PATH,
        label=DataLabel.IMAGE,
        folder_path=path_to_data
    )
    print(f"Length of PATH IMAGE dataset: {len(ds_path_image)}")
    sample = ds_path_image[0]
    print("Sample keys:", sample.keys())
    print("image shape:", None if sample["image"] is None else sample["image"].shape)
    print("tabular:", sample["tabular"])
    print("dir:", sample["dir"], "\n")

    print("=== PATH mode: TAB only ===")
    ds_path_tab = MultimodalDataset(
        source_type=SourceType.PATH,
        label=DataLabel.TAB,
        folder_path=path_to_data
    )
    print(f"Length of PATH TAB dataset: {len(ds_path_tab)}")
    sample = ds_path_tab[0]
    print("Sample keys:", sample.keys())
    print("image:", sample["image"])
    print("tabular shape:", None if sample["tabular"] is None else sample["tabular"].shape)
    print("dir:", sample["dir"], "\n")

    print("=== PATH mode: BOTH ===")
    ds_path_both = MultimodalDataset(
        source_type=SourceType.PATH,
        label=DataLabel.BOTH,
        folder_path=path_to_data
    )
    print(f"Length of PATH BOTH dataset: {len(ds_path_both)}")
    sample = ds_path_both[0]
    print("Sample keys:", sample.keys())
    print("image shape:", None if sample["image"] is None else sample["image"].shape)
    print("tabular shape:", None if sample["tabular"] is None else sample["tabular"].shape)
    print("dir:", sample["dir"], "\n")

    # -------------------------------------------------------
    # C) Test the LIST source mode
    #    We'll grab items from the dataloader (image & tab)
    #    and build separate lists to test the three label types
    # -------------------------------------------------------
    list_of_image_tensors = []
    list_of_tab_tensors = []
    list_of_both = []

    for i, batch in enumerate(train_dataloader):
        # each batch is typically a dict with keys like ['image', 'tabular', ...]
        imgs = batch["image"]   # shape: (B, channels, height, width) or similar
        tabs = batch["tabular"] # shape: (B, 174) or similar

        for im, tab in zip(imgs, tabs):
            list_of_image_tensors.append(im.cpu())
            list_of_tab_tensors.append(tab.cpu())
            # For BOTH, you can store a dictionary or a tuple:
            list_of_both.append({"image": im.cpu(), "tabular": tab.cpu()})

        # Just to keep the list short for demonstration, break after 6
        if len(list_of_image_tensors) >= 6:
            break

    print("=== LIST mode: IMAGE only ===")
    ds_list_image = MultimodalDataset(
        source_type=SourceType.LIST,
        label=DataLabel.IMAGE,
        data_list=list_of_image_tensors
    )
    print(f"Length of LIST IMAGE dataset: {len(ds_list_image)}")
    sample = ds_list_image[0]
    print("Sample keys:", sample.keys())
    print("image shape:", None if sample["image"] is None else sample["image"].shape)
    print("tabular:", sample["tabular"], "\n")

    print("=== LIST mode: TAB only ===")
    ds_list_tab = MultimodalDataset(
        source_type=SourceType.LIST,
        label=DataLabel.TAB,
        data_list=list_of_tab_tensors
    )
    print(f"Length of LIST TAB dataset: {len(ds_list_tab)}")
    sample = ds_list_tab[0]
    print("Sample keys:", sample.keys())
    print("image:", sample["image"])
    print("tabular shape:", None if sample["tabular"] is None else sample["tabular"].shape, "\n")

    print("=== LIST mode: BOTH ===")
    ds_list_both = MultimodalDataset(
        source_type=SourceType.LIST,
        label=DataLabel.BOTH,
        data_list=list_of_both
    )
    print(f"Length of LIST BOTH dataset: {len(ds_list_both)}")
    sample = ds_list_both[0]
    print("Sample keys:", sample.keys())
    print("image shape:", None if sample["image"] is None else sample["image"].shape)
    print("tabular shape:", None if sample["tabular"] is None else sample["tabular"].shape, "\n")


if __name__ == "__main__":
    main()
