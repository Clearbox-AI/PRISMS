import os
import datetime
from typing import Optional
from pathlib import Path

def setup_exp_directory(base_path: str, label: Optional[str] = None) -> Path:
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