import os
import datetime
from typing import Optional, Union
from pathlib import Path

def setup_storage_directory(base_path: Union[Path, str], label: Optional[str] = None) -> Path:
    """
    Checks if `base_path` exists, creates it if not, then creates a date-based
    subfolder for saving results.

    Returns:
        str: The path to the newly created directory.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    # Create a subdirectory with current date+time to avoid collisions
    date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = Path(base_path, f"{date_str}_{label}" if label else date_str)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def get_main_save_directory(cfg) -> Path:
    """
    Determines the main save directory for training outputs.
    """
    # Determine the base save path (keep original fallback semantics as much as possible)
    default_base = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent)) / "training_outputs"
    base_save_path = Path(cfg.training.get("base_save_path", default_base))

    if cfg.training.resume_training:
        main_save_dir = cfg.training.get("resume_checkpoint_dir")

        if not main_save_dir:
            # Find the most recent directory if no checkpoint dir is provided
            try:
                main_save_dir = max(
                    (p for p in base_save_path.iterdir() if p.is_dir()),
                    key=lambda p: p.stat().st_mtime
                )
            except ValueError:
                raise FileNotFoundError("No existing checkpoint directories found for resuming training.")
    else:
        main_save_dir = setup_storage_directory(base_save_path, label=cfg.training.get("save_label"))

    # Create necessary subdirectories
    for subdir in ["samples", "checkpoints"]:
        os.makedirs(Path(main_save_dir, subdir), exist_ok=True)

    return main_save_dir
