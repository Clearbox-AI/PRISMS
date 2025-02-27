import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from glob import glob

class BaseNaccDataset(Dataset):
    """
    Base dataset class that handles:
      - listing data directories
      - debug functionality
      - file searching, etc.
    """
    _debug_shown_global = False  # For one-time global debug (if desired)

    def __init__(self,
                 data_dir: str,
                 debug: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.debug = debug
        self._debug_shown = False

        # Collect subdirectories for each patient
        self.patient_dirs = [
            os.path.join(self.data_dir, d) for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ]
        if not self.patient_dirs:
            raise ValueError(f"No patient directories in {self.data_dir}")

    def __len__(self):
        return len(self.patient_dirs)

    def _get_patient_dir(self, idx: int) -> str:
        return self.patient_dirs[idx]

    def _get_first_file(self, directory: str, pattern: str) -> str:
        files = glob(os.path.join(directory, pattern))
        if not files:
            raise FileNotFoundError(
                f"No files found in {directory} with pattern {pattern}")
        return files[0]

    def _debug_show_image(self, image_tensor: torch.Tensor):
        """
        Show the image using matplotlib for a quick debug.
        We'll handle up to 3 channels for visualization.
        """
        image_np = image_tensor.cpu().numpy()
        C, H, W = image_np.shape
        if C == 1:
            plt.imshow(image_np[0], cmap="gray")
        else:
            # if C >= 3, or any other fallback
            plt.imshow(image_np[0], cmap="gray")
        plt.title("Debug: Image")
        plt.axis("off")
        plt.show(block=True)
