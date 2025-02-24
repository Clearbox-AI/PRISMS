import os
import datetime

def setup_exp_directory(base_path: str) -> str:
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
    save_dir = os.path.join(base_path, date_str)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir