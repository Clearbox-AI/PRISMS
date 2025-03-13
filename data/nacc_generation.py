import os
import json
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Optional, Union, List
from pathlib import Path
from hydra import compose, initialize_config_dir
from data.loader import load_training_data
from models.utils.model_loader import load_model
from enums.models.model_types import ModelType
from enums.training_versions import DiTTrainingVersion


def save_conditional_data(
    images: torch.Tensor,
    tabs: torch.Tensor,
    sample_mapping: Dict[str, list],
    output_dir: str
) -> None:
    """
    Saves the generated images/tabular data to disk in a structure based on the
    sample_mapping. For each patient's path in sample_mapping, we:
      1) Extract the last part of the path (e.g. 'patient001').
      2) Create a folder for that patient if it does not exist.
      3) For each generated sample index in the mapping, we figure out how many
         files already exist in that folder, and we name the new files accordingly.
    """

    # 1) Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if not sample_mapping:
        print("[WARNING] sample_mapping is empty or None. Nothing to save.")
        return

    # 2) Loop through each patient path -> list of sample indices
    for full_path, gen_indices in sample_mapping.items():
        # We'll parse out the last part of the path for a folder name
        patient_id = os.path.basename(full_path)  # e.g. patient001

        patient_folder = os.path.join(output_dir, patient_id)
        os.makedirs(patient_folder, exist_ok=True)

        # 3) For each generated sample index, save:
        #    - One .npy for the image
        #    - One .json for the tab data
        for idx in gen_indices:
            # Count how many existing npy files in that folder that match "image_*.npy"
            existing_images = [
                f for f in os.listdir(patient_folder)
                if f.startswith("image_") and f.endswith(".npy")
            ]
            # We'll create new image index = number of existing image files
            new_img_index = len(existing_images)

            # Save the image in .npy
            img_filename = os.path.join(patient_folder, f"image_{new_img_index:03d}.npy")
            np.save(img_filename, images[idx].cpu().numpy())

            # Similarly for the tab data in .json
            # Count how many "tab_*.json" exist
            existing_tabs = [
                f for f in os.listdir(patient_folder)
                if f.startswith("tab_") and f.endswith(".json")
            ]
            new_tab_index = len(existing_tabs)

            # Convert tab tensor to python list, then save as JSON
            tab_data_list = tabs[idx].cpu().numpy().tolist()
            tab_filename = os.path.join(patient_folder, f"tab_{new_tab_index:03d}.json")
            with open(tab_filename, "w") as f:
                json.dump(tab_data_list, f)

    print("[INFO] Saved generated samples to:", output_dir)


class ConfigBucket:
    """
    A container ensuring that either:
      (1) `generation_cfg` and `loader_cfg`, OR
      (2) `generation_params` and `loader_params`
    is provided (but not both).
    """

    def __init__(
        self,
        generation_cfg: Optional[Union[DictConfig, dict]] = None,
        loader_cfg: Optional[Union[DictConfig, dict]] = None,
        generation_params: Optional[dict] = None,
        loader_params: Optional[dict] = None,
    ):
        # Either we have (generation_cfg & loader_cfg) OR (generation_params & loader_params)
        # Not both or neither.
        have_hydra_conf = (generation_cfg is not None) and (loader_cfg is not None)
        have_params_conf = (generation_params is not None) and (loader_params is not None)

        if have_hydra_conf == have_params_conf:
            # True == True means both sets are provided; False == False means neither is provided
            raise ValueError(
                "A valid ConfigBucket must have exactly one of:\n"
                "  (1) generation_cfg and loader_cfg, OR\n"
                "  (2) generation_params and loader_params.\n"
                "You provided:\n"
                f"  generation_cfg={generation_cfg}\n  loader_cfg={loader_cfg}\n"
                f"  generation_params={generation_params}\n  loader_params={loader_params}"
            )

        self.generation_cfg = generation_cfg
        self.loader_cfg = loader_cfg
        self.generation_params = generation_params
        self.loader_params = loader_params


class SrcDataBucket:
    """
    A container ensuring that `condition_data` is always present, and
    `patient_dirs` can optionally be provided.
    """

    def __init__(
        self,
        condition_data: torch.Tensor,
        patient_dirs: Optional[List[str]] = None
    ):
        if condition_data is None:
            raise ValueError("`condition_data` cannot be None for a valid SrcDataBucket.")
        self.condition_data = condition_data
        # If no patient_dirs provided, default to an empty list.
        self.patient_dirs = patient_dirs if patient_dirs is not None else []


def generate_conditional_data(
        config_bucket: Optional[ConfigBucket] = None,
        src_data_bucket: Optional[SrcDataBucket] = None,

) -> None:
    """
    Flexible function to generate data using a diffusion model in a conditional scenario,
    then save the generated samples on disk.

    Two main usage modes:

      1) Using a config-based approach (pass `config_bucket`):
         - The `config_bucket` can contain:
             (a) A Hydra DictConfig (with, e.g., `generation_cfg` and `loader_cfg` fields), OR
             (b) A plain Python dict with `generation_cfg` / `loader_cfg` keys
         - If present, it's one of these two cases, we assume we are using
           a **dataloader** approach ("conditional generation from dataloader").

      2) Using direct data approach (pass `src_data_bucket`):
         - We expect `src_data_bucket` to contain:
             `condition_data`: a Torch tensor with shape [num_conditions, ...]
             `patient_dirs`:   a list of patient directory strings (optional but recommended)
           - In this case, no dataloader is used. If `patient_dirs` is not passed,
             you must generate some default patient naming to save them.

    Dataloader and direct data are mutually exclusive:
      - If `config_bucket` is provided, we do the dataloader approach.
      - Otherwise, we do direct data from `src_data_bucket`.

    Returns:
        None. (It saves data to disk as a side effect.)
    """

    # Check mutual exclusivity
    if config_bucket is not None and src_data_bucket is not None:
        raise ValueError(
            "Both `config_bucket` and `src_data_bucket` were provided, "
            "but they are mutually exclusive. Please provide only one."
        )
    if config_bucket is None and src_data_bucket is None:
        raise ValueError(
            "Neither `config_bucket` nor `src_data_bucket` was provided. "
            "Please provide at least one."
        )


    # -----------------------------------------
    # 1. Parse the input approach
    # -----------------------------------------
    if config_bucket is not None:
        # We are in the dataloader approach.
        # config_bucket ensures we have EITHER (generation_cfg + loader_cfg) OR
        # (generation_params + loader_params).
        if config_bucket.generation_cfg is not None:
            generation_section = config_bucket.generation_cfg
            loader_section = config_bucket.loader_cfg
        else:
            generation_section = config_bucket.generation_params
            loader_section = config_bucket.loader_params

        # Example: extract some parameters from generation_section
        # This part depends on your actual structure
        device = generation_section.get("device", "cuda")
        model_type = generation_section["model_type"]
        model_variant = generation_section["model_variant"]

        # Load the diffusion model (function not shown)
        diff_model = load_model(model_type=model_type, model_variant=model_variant)
        diff_model.to(device)

        # Prepare Dataloader
        dataloader = load_training_data(loader_section)

        # Extract the generation parameters
        n_samples = generation_section["n_samples"]
        condition_modality = generation_section.get("condition_modality", "multi")
        partial_condition = generation_section.get("partial_condition", False)
        partial_noise_factor = generation_section.get("partial_noise_factor", 0.0)
        cfg_scale = generation_section.get("cfg_scale", 7.0)
        height = generation_section.get("height", 256)
        width = generation_section.get("width", 256)
        n_tab = generation_section.get("n_tab", 10)
        batch_size = generation_section.get("batch_size", 1)
        output_dir = generation_section["output_dir"]

        # -----------------------------------------
        # 2. Generate samples from the diffusion model
        # -----------------------------------------
        images, tabs, mapping = diff_model.generate_samples(
            n_samples=n_samples,
            device=device,
            condition_modality=condition_modality,
            partial_condition=partial_condition,
            partial_noise_factor=partial_noise_factor,
            cfg_scale=cfg_scale,
            height=height,
            width=width,
            n_tab=n_tab,
            batch_size=batch_size,
            dataloader=dataloader,
            condition_data=None,  # because we are using the dataloader approach
            patient_dirs=None
        )

    else:
        # We are in the direct data approach
        condition_data = src_data_bucket.condition_data
        patient_dirs = src_data_bucket.patient_dirs

        # You'd define these values in practice or pass them as arguments somewhere
        device = "cuda"
        model_type = "example_model"
        model_variant = "example_variant"
        diff_model = load_model(model_type=model_type, model_variant=model_variant)
        diff_model.to(device)

        n_samples = condition_data.size(0)
        condition_modality = "multi"
        partial_condition = False
        partial_noise_factor = 0.0
        cfg_scale = 7.0
        height = 256
        width = 256
        n_tab = 10
        batch_size = 1
        output_dir = "./output"  # or whichever

        # -----------------------------------------
        # Generate the samples (no dataloader here)
        # -----------------------------------------
        images, tabs, mapping = diff_model.generate_samples(
            n_samples=n_samples,
            device=device,
            condition_modality=condition_modality,
            partial_condition=partial_condition,
            partial_noise_factor=partial_noise_factor,
            cfg_scale=cfg_scale,
            height=height,
            width=width,
            n_tab=n_tab,
            batch_size=batch_size,
            dataloader=None,
            condition_data=condition_data,
            patient_dirs=patient_dirs,
        )

    # -----------------------------------------
    # 3. Save the generated data to disk
    # -----------------------------------------
    if mapping:
        save_conditional_data(images, tabs, mapping, output_dir)
    else:
        print("[WARNING] No sample mapping was returned. "
              "Nothing to save or no conditional data provided.")



def generate_conditional_data(...):

    from utils.configurations import set_project_root
    set_project_root()

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "generation"))):
        generation_cfg = compose(config_name="tab_conditioning")
        OmegaConf.set_struct(generation_cfg, False)

    with initialize_config_dir(config_dir=str(Path(os.environ["PROJECT_ROOT"], "configs", "datasets"))):
        loader_cfg = compose(config_name="nacc")
        OmegaConf.set_struct(loader_cfg, False)

    dataloader = load_training_data(loader_cfg)

    # Load Diffusion model
    diff_model = load_model(
        model_type=ModelType.DIFFUSION,
        model_variant=DiTTrainingVersion.base_dit_training
    )
    diff_model.to(generation_cfg.generation.device)

    # --- Generate the samples ---
    images, tabs, mapping = diff_model.generate_samples(
        n_samples=generation_cfg.generation.n_samples,
        device=generation_cfg.generation.device,
        condition_modality=generation_cfg.generation.condition_modality,
        partial_condition=generation_cfg.generation.partial_condition,
        partial_noise_factor=generation_cfg.generation.partial_noise_factor,
        cfg_scale=generation_cfg.generation.cfg_scale,
        height=generation_cfg.generation.height,
        width=generation_cfg.generation.width,
        n_tab=174, # TODO: maybe make a "load base configurations" function, loader_cfg.n_tab,
        batch_size=generation_cfg.generation.batch_size,
        dataloader=dataloader,
        condition_data=None,  # We are using the dataloader approach here
        patient_dirs=None      # In dataloader mode, the `generate_samples` typically uses 'dir' from batch
    )

    # 'mapping' is a dict like: { full_patient_path: [list_of_generated_indices] }

    # --- Save the generated data to disk ---
    save_conditional_data(
        images=images,
        tabs=tabs,
        sample_mapping=mapping,
        output_dir=generation_cfg.generation.output_dir
    )

if __name__ == "__main__":
    main()