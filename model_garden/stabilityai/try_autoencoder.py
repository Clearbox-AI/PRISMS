import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL
import torch.nn.functional as F


# Configuration flag for latent augmentations
APPLY_LATENT_AUGMENTATIONS = True  # Set to False to disable perturbations
PERTURBATION_TYPES = ["gaussian_noise", "scaling", "drop_channels"]
PERTURBATION_STRENGTHS = [0.1, 0.5, 1.0, 2.0, 5.0]  # Perturbation levels


def prepare_for_plot(tensor):
    """Converts a tensor with values in [-1, 1] to an image in [0, 1] format."""
    image = tensor[0].cpu().clamp(-1, 1)
    image = (image + 1) / 2.0
    image = image.permute(1, 2, 0).numpy()
    return image


def pad_to_size(image_tensor: torch.Tensor, target_size: int = 512) -> torch.Tensor:
    """
    Pads a (C, H, W) image tensor to (C, target_size, target_size) with a constant 0 background.
    Preserves the original aspect ratio in the center.
    """
    _, H, W = image_tensor.shape
    pad_h = max(0, (target_size - H) // 2)
    pad_w = max(0, (target_size - W) // 2)
    return F.pad(image_tensor, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)


def normalize_for_sd(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalizes an image tensor to the range [-1, 1], handling edge cases.
    """
    min_val = image_tensor.min()
    max_val = image_tensor.max()
    if torch.isclose(min_val, max_val):  # Handle constant images
        print("[Warning] Image is effectively constant. Setting it to zeros.")
        return torch.zeros_like(image_tensor)
    image_tensor = (image_tensor - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    return 2.0 * image_tensor - 1.0  # Shift to [-1, 1]


# Perturbation Methods
def add_gaussian_noise(latent, std_dev):
    noise = torch.randn_like(latent) * std_dev
    return latent + noise


def scale_latent(latent, alpha):
    return latent * alpha


def drop_latent_channels(latent, drop_prob):
    mask = torch.bernoulli(torch.full(latent.shape, 1 - drop_prob, device=latent.device))
    return latent * mask


# ---------------------
# 1. Load the VAE model
# ---------------------
vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="vae",
    torch_dtype=torch.float32
).to("cuda")


# ---------------------
# 2. Load the .npy file
# ---------------------
image_path = "/mnt/dataset_storage/data/nacc_dataset/nacc_subset/middle_slice/sub-NACC022031/sub-NACC022031_T1w_middle_slice.npy"
image = np.load(image_path)

if image.ndim == 2:  # If 2D, make it 3-channel
    image = np.repeat(image[None, ...], 3, axis=0)

image_tensor = torch.tensor(image, dtype=torch.float32, device="cuda")
image_tensor = pad_to_size(image_tensor, target_size=512)  # [3, 512, 512]
image_tensor = normalize_for_sd(image_tensor).unsqueeze(0)  # [1, 3, 512, 512]


# ---------------------
# 5. Encode -> Decode
# ---------------------
with torch.inference_mode():
    latent_dist = vae.encode(image_tensor).latent_dist
    latent = latent_dist.sample()  # Latent representation
    decoded_image = vae.decode(latent).sample  # Basic reconstruction


# ---------------------
# 6. Main visualization
# ---------------------
num_cols = 2 + (len(PERTURBATION_STRENGTHS) if APPLY_LATENT_AUGMENTATIONS else 0)
fig, axes = plt.subplots(1, num_cols, figsize=(16, 6))

# Plot original and basic reconstruction
axes[0].imshow(prepare_for_plot(image_tensor))
axes[0].set_title("Original Image", fontsize=10)
axes[0].axis('off')

axes[1].imshow(prepare_for_plot(decoded_image))
axes[1].set_title("Basic Reconstruction", fontsize=10)
axes[1].axis('off')

# Apply perturbations
if APPLY_LATENT_AUGMENTATIONS:
    perturb_fn_map = {
        "gaussian_noise": add_gaussian_noise,
        "scaling": scale_latent,
        "drop_channels": drop_latent_channels
    }

    perturb_fn = perturb_fn_map["gaussian_noise"]  # Select perturbation type here
    for idx, strength in enumerate(PERTURBATION_STRENGTHS):
        perturbed_latent = perturb_fn(latent, strength)
        with torch.inference_mode():
            perturbed_decoded = vae.decode(perturbed_latent).sample

        # Visualize the perturbed reconstruction
        axes[idx + 2].imshow(prepare_for_plot(perturbed_decoded))
        axes[idx + 2].set_title(f"Strength: {strength}", fontsize=10)
        axes[idx + 2].axis('off')

plt.tight_layout()
plt.show()
