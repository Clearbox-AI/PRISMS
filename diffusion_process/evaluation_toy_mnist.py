import os
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.ndimage import center_of_mass, label
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import re
from natsort import natsorted
from PIL import Image
from tqdm import tqdm
import argparse

# List of metric names in order
metric_names = [
    "average_pixel_intensity",
    "non_zero_pixels",
    "pixel_intensity_std",
    "aspect_ratio",
    "median_pixel_intensity",
    "pixel_intensity_skewness",
    "pixel_intensity_kurtosis",
    "pixel_intensity_range",
    "max_pixel_intensity",
    "min_pixel_intensity",
    "pixel_intensity_entropy",
    "normalized_pixel_variance",
    "sum_pixel_intensities",
    "center_of_mass_x",
    "center_of_mass_y",
    "bounding_box_area",
    "pixel_density",
    "connected_components",
    "eccentricity",
    "perimeter",
    "compactness",
]

def calculate_metrics(image_array):
    """
    Calculate enhanced metrics for a given image.
    :param image_array: numpy array of the image.
    :return: Array of metrics in the specified order.
    """
    flattened = image_array.flatten()  # Flatten for statistical calculations
    labeled_array, num_features = label(image_array > 0)  # For connected components and bounding box
    regions = regionprops(labeled_array)
    bounding_box_area = regions[0].area if regions else 0

    metrics = {
        # Statistical Metrics
        "average_pixel_intensity": float(np.mean(flattened)),
        "non_zero_pixels": int(np.count_nonzero(flattened)),
        "pixel_intensity_std": float(np.std(flattened)),
        "aspect_ratio": float(image_array.shape[1] / image_array.shape[0]),
        "median_pixel_intensity": float(np.median(flattened)),
        "pixel_intensity_skewness": float(skew(flattened)),
        "pixel_intensity_kurtosis": float(kurtosis(flattened)),
        "pixel_intensity_range": float(np.ptp(flattened)),
        "max_pixel_intensity": float(np.max(flattened)),
        "min_pixel_intensity": float(np.min(flattened)),
        "pixel_intensity_entropy": float(-np.sum((flattened / 255.0) * np.log2(flattened / 255.0 + 1e-10))),
        "normalized_pixel_variance": float(np.var(flattened) / (np.mean(flattened) + 1e-10)),
        "sum_pixel_intensities": float(np.sum(flattened)),

        # Spatial Metrics
        "center_of_mass_x": float(center_of_mass(image_array)[1]),
        "center_of_mass_y": float(center_of_mass(image_array)[0]),
        "bounding_box_area": int(bounding_box_area),
        "pixel_density": float(np.count_nonzero(image_array) / image_array.size),
        "connected_components": int(num_features),

        # Shape-Based Metrics
        "eccentricity": float(regions[0].eccentricity if regions else 0),
        "perimeter": float(regions[0].perimeter if regions else 0),
        "compactness": float(
            (regions[0].perimeter ** 2) / bounding_box_area if regions and bounding_box_area > 0 else 0),
    }

    # Extract the metrics in the specified order
    metric_values = [metrics[name] for name in metric_names]

    return np.array(metric_values)

def extract_index(filename):
    """
    Extracts the index from the filename.
    Assumes that the index is the last number before the extension.
    """
    basename = os.path.basename(filename)
    # Use regex to find the last number before the file extension
    match = re.search(r'(\d+)(?=\.\w+$)', basename)
    if match:
        return int(match.group(1))
    else:
        return None

def main(input_path):
    # Get the list of folders in the input path, ordered by name
    folder_names = [d for d in os.listdir(input_path)
                    if os.path.isdir(os.path.join(input_path, d)) and d.startswith("samples_")]
    folder_names = natsorted(folder_names)

    if not folder_names:
        print(f"No folders starting with 'samples_' found in {input_path}")
        return

    # For plotting, collect comparison values (e.g., MSEs) for each folder
    folder_mses = []

    # For each folder
    for folder_name in folder_names:
        folder_path = os.path.join(input_path, folder_name)
        print(f"Processing folder: {folder_name}")

        # Get the list of image files
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        image_files_full = [os.path.join(folder_path, f) for f in image_files]

        # Extract indices from filenames
        image_indices = []
        for f in image_files_full:
            idx = extract_index(f)
            if idx is not None:
                image_indices.append((f, idx))
            else:
                print(f"Warning: Could not extract index from filename {f}")

        # Sort images by index
        image_indices_sorted = sorted(image_indices, key=lambda x: x[1])

        # Get the sorted list of image files
        sorted_image_files = [f for f, idx in image_indices_sorted]

        # Read the .npy file
        npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        if len(npy_files) != 1:
            print(f"Error: Expected exactly one .npy file in folder {folder_path}, found {len(npy_files)}")
            folder_mses.append(None)
            continue
        npy_file = os.path.join(folder_path, npy_files[0])
        npy_data = np.load(npy_file)

        # Check that the number of images matches the number of rows in the .npy file
        if len(sorted_image_files) != npy_data.shape[0]:
            print(f"Error: Number of images ({len(sorted_image_files)}) does not match number of rows in .npy file ({npy_data.shape[0]}) in folder {folder_path}")
            folder_mses.append(None)
            continue

        # For each image and corresponding row
        mses = []
        for i in tqdm(range(len(sorted_image_files)), desc=f"Processing images in {folder_name}"):
            image_file = sorted_image_files[i]
            try:
                image = Image.open(image_file).convert('L')
                image_array = np.array(image)
                computed_metrics = calculate_metrics(image_array)
                # Get the corresponding row from npy_data
                row_metrics = npy_data[i]
                # Ensure that both computed_metrics and row_metrics are arrays of the same shape
                if computed_metrics.shape != row_metrics.shape:
                    print(f"Error: Shape mismatch between computed metrics and row metrics at index {i} in folder {folder_name}")
                    continue
                # Compute MSE between computed_metrics and row_metrics
                mse = np.mean((computed_metrics - row_metrics) ** 2)
                mses.append(mse)
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
                continue

        # For this folder, compute the average MSE
        if mses:
            average_mse = np.mean(mses)
            folder_mses.append(average_mse)
            print(f"Average MSE for folder {folder_name}: {average_mse}")
        else:
            print(f"No MSEs computed for folder {folder_name}")
            folder_mses.append(None)

    # Plot the MSEs over the folders
    plt.figure(figsize=(10,6))
    valid_indices = [i for i, mse in enumerate(folder_mses) if mse is not None]
    valid_mses = [folder_mses[i] for i in valid_indices]
    valid_folder_names = [folder_names[i] for i in valid_indices]
    plt.plot(valid_indices, valid_mses, marker='o')
    plt.xlabel('Folder Index (Time)')
    plt.ylabel('Average MSE')
    plt.title('Average MSE over Folders')
    plt.xticks(valid_indices, valid_folder_names, rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process images and compare metrics.')
    parser.add_argument('--input_path', type=str, help='Path to the input directory containing folders.')
    args = parser.parse_args()
    main(args.input_path)
