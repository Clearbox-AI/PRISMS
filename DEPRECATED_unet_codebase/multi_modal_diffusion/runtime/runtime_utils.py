import json
import os
from pathlib import Path
import tempfile

class ShapeManager:
    def __init__(self):
        # Create a temporary file path
        self.filename = Path(tempfile.gettempdir()) / "execution_factors.json"
        self.shapes = {"1d": [], "2d": []}  # Separate storage for 1D and 2D shapes

    def save_downsample_shape(self, shape, dims):
        """Save a new shape during downsampling if it's smaller than the last recorded shape."""
        if dims == 1:
            shape_list = self.shapes["1d"]
        elif dims == 2:
            shape_list = self.shapes["2d"]
        else:
            raise ValueError("Unsupported dimensions: only 1D (tabular) and 2D (image) are supported.")

        # Check if shape_list is empty or if the current shape's last dimension is smaller than the last saved shape's
        if not shape_list or shape[-1] <= shape_list[-1][-1]:
            shape_list.append(shape)
            self._write_shapes()

    def load_upsample_shape(self, dims):
        """Retrieve and remove the next shape for upsampling in reverse order."""
        if dims == 1:
            shape_list = self.shapes["1d"]
        elif dims == 2:
            shape_list = self.shapes["2d"]
        else:
            raise ValueError("Unsupported dimensions: only 1D (tabular) and 2D (image) are supported.")

        if shape_list:
            shape = shape_list.pop()
            self._write_shapes()
            return shape
        return None

    def _write_shapes(self):
        """Write the shapes dictionary to the JSON file."""
        with open(self.filename, "w") as file:
            json.dump(self.shapes, file)

    def load_shapes_from_file(self):
        """Load shapes from JSON file if it exists."""
        if self.filename.exists():
            with open(self.filename, "r") as file:
                self.shapes = json.load(file)

    def delete_file(self):
        """Delete the JSON file after the iteration."""
        if self.filename.exists():
            os.remove(self.filename)
