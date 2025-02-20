import os

# HYDRA SETUP

def get_project_root():
    """Finds the PRISMS root dynamically based on the script's absolute path."""
    script_path = os.path.abspath(__file__)  # Get the absolute path of this file
    parts = script_path.split(os.sep)  # Split path into components

    if "PRISMS" in parts:
        idx = parts.index("PRISMS")  # Find the first occurrence of PRISMS
        return os.sep.join(parts[:idx + 1])  # Join back up to PRISMS
    else:
        raise RuntimeError("PRISMS root not found in the script path.")

# Get the project root and export it as an environment variable for Hydra
PROJECT_ROOT = get_project_root()
os.environ["PROJECT_ROOT"] = PROJECT_ROOT  # Ensure Hydra sees this variable