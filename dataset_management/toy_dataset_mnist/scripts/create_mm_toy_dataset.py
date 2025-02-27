import os
import json
import numpy as np
from torchvision import datasets, transforms
from scipy.stats import kurtosis, skew
from scipy.ndimage import center_of_mass, label
from skimage.measure import regionprops

def calculate_metrics(image_array):
    """
    Calculate enhanced metrics for a given image.
    :param image_array: numpy array of the image.
    :return: Dictionary with metrics.
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
    return metrics

def create_multimodal_dataset(output_dir, dataset):
    """
    Create a multimodal dataset from MNIST.
    :param output_dir: Directory to save the dataset.
    :param dataset: MNIST dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, (image, label) in enumerate(dataset):
        # Create a folder for each sample
        sample_folder = os.path.join(output_dir, f"sample_{idx}")
        os.makedirs(sample_folder, exist_ok=True)

        # Save the image as PNG
        image_path = os.path.join(sample_folder, "image.png")
        image.save(image_path)

        # Convert image to numpy array
        image_array = np.array(image)

        # Calculate metrics
        metrics = calculate_metrics(image_array)

        # Save metrics as JSON
        json_path = os.path.join(sample_folder, "metrics.json")
        with open(json_path, "w") as json_file:
            json.dump(metrics, json_file, indent=4)

        print(f"Processed sample {idx + 1}/{len(dataset)}")

if __name__ == "__main__":
    # Directory to save the multimodal dataset
    output_dir = os.path.join("..", "data")

    # Download MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte()),  # Convert to uint8
        transforms.ToPILImage()
    ])

    mnist_dataset = datasets.MNIST(root=os.path.join(".", "tmp_data"), train=True, download=True, transform=transform)

    # Create the multimodal dataset
    create_multimodal_dataset(output_dir, mnist_dataset)

    print(f"Multimodal dataset created in '{output_dir}'")
