import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from utils import load_model_from_config
from omegaconf import OmegaConf
import nibabel as nib
from scipy.ndimage import zoom
import argparse
import numpy as np


# Define a function to display an image
def show_image(img, title="Image"):
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Load the pretrained autoencoder model
def load_autoencoder(checkpoint_path):

    config = OmegaConf.load(".\model_garden\config.yaml")
    model = load_model_from_config(config, checkpoint_path)

    return model

# Load the image and prepare it for the model
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image for model compatibility
        transforms.ToTensor()
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image, image_tensor


def load_image_npy(image_path):
    # Load the .npy file
    image_array = np.load(image_path)

    # Convert the NumPy array to a PyTorch tensor
    image_tensor = torch.from_numpy(image_array)

    # Ensure the tensor is in the correct format for your model
    # Assuming the image_array has shape (height, width) for a grayscale image
    if len(image_array.shape) == 2:
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension

    # Convert grayscale to RGB by replicating the single channel three times
    image_tensor = image_tensor.repeat(3, 1, 1)

    # Resize the image to 256x256
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image for model compatibility
        transforms.ToTensor()
    ])

    # Convert the tensor back to a PIL image for resizing
    image_pil = transforms.ToPILImage()(image_tensor)
    image_tensor = preprocess(image_pil).unsqueeze(0)  # Add batch dimension

    return image_array, image_tensor


def load_nii_image(image_path, target_shape=(1, 3, 256, 256)):
    # Load the .nii.gz image using nibabel
    nii_image = nib.load(image_path)
    image_data = nii_image.get_fdata()  # Get the image data as a numpy array

    # Take a middle slice along the z-axis to get a 2D image
    middle_slice = image_data[:, :, image_data.shape[2] // 2]

    # Resize the 2D slice to (256, 256)
    zoom_factors = (target_shape[2] / middle_slice.shape[0], target_shape[3] / middle_slice.shape[1])
    resized_image = zoom(middle_slice, zoom_factors)

    # Normalize and convert to tensor
    image_tensor = torch.tensor(resized_image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

    # Expand to 3 channels
    image_tensor = image_tensor.expand(target_shape)

    return image_tensor



# Encode and decode the image
def encode_decode(model, image_tensor):
    with torch.no_grad():
        encoded = model.encode(image_tensor).sample()  # Encoding step
        # encoded = model.encode(image_tensor)[0]
        decoded = model.decode(encoded)       # Decoding step
    return encoded, decoded


parser = argparse.ArgumentParser(description="Try autoencoder vq vae.")
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the autoencoder model')
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
args = parser.parse_args()


checkpoint_path = args.checkpoint_path
image_path = args.image_path

# Load model and image
model = load_autoencoder(checkpoint_path)
# original_image, image_tensor = load_image(image_path)
original_image, image_tensor = load_image_npy(image_path)
# image_tensor = load_nii_image(image_path)

# Show the original image
show_image(original_image, title="Original Image")

# Encode and decode
encoded, decoded = encode_decode(model, image_tensor)

# Show encoded (latent representation as tensor) and decoded images
print("Encoded representation:", encoded)
show_image(transforms.ToPILImage()(decoded.squeeze(0)), title="Decoded Image")
