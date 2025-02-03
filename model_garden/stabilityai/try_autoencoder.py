import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL
import torch.nn.functional as F


def pad_to_size(image_tensor: torch.Tensor, target_size: int = 512) -> torch.Tensor:
    """
    Pads a (C, H, W) image tensor to (C, target_size, target_size) with a constant 0 background.
    Preserves the original aspect ratio in the center.
    """
    # image_tensor shape: [C, H, W]
    _, H, W = image_tensor.shape

    # Amount to pad on each side
    pad_h = max(0, (target_size - H) // 2)
    pad_w = max(0, (target_size - W) // 2)

    # Pad with zeros (background=0); shape becomes [C, target_size, target_size]
    padded_image = F.pad(image_tensor, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)

    return padded_image


def normalize_for_sd(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalizes an image tensor to the range [-1, 1], handling edge cases.
    Expects float32 or float64 tensor.
    """
    min_val = image_tensor.min()
    max_val = image_tensor.max()

    # Handle constant or near-constant images to avoid divide-by-zero
    if torch.isclose(min_val, max_val):
        # If the slice is constant, just set it to 0.0 => which maps to -1 after the shift.
        # Or you can choose to keep it at 0.0 entirely.
        print("[Warning] Image is effectively constant. Setting it to zeros.")
        return torch.zeros_like(image_tensor)

    # First bring to [0, 1]
    image_tensor = (image_tensor - min_val) / (max_val - min_val)
    # Then shift to [-1, 1]
    image_tensor = 2.0 * image_tensor - 1.0

    return image_tensor


# ---------------------
# 1. Load the VAE model
# ---------------------
vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="vae",
    torch_dtype=torch.float32  # using float32 to avoid potential half-precision NaNs
).to("cuda")


# ---------------------
# 2. Load the .npy file
# ---------------------
image_path = "/mnt/dataset_storage/data/nacc_dataset/nacc_subset/middle_slice/sub-NACC022031/sub-NACC022031_T1w_middle_slice.npy"
image = np.load(image_path)

# Suppose 'image' is 2D. Make it 3-channel by repeating.
# If your data is already 3D in shape [3, H, W], you can skip this step.
if image.ndim == 2:
    # image: [H, W]
    image = image[None, ...]         # => [1, H, W]
    image = np.repeat(image, 3, axis=0)  # => [3, H, W]

# Convert to torch tensor on CUDA, float32
image_tensor = torch.tensor(image, dtype=torch.float32, device="cuda")

# ---------------------------
# 3. Pad and normalize for VAE
# ---------------------------
image_tensor = pad_to_size(image_tensor, target_size=512)   # => [3, 512, 512]
image_tensor = normalize_for_sd(image_tensor)               # => range ~[-1, 1]
image_tensor = image_tensor.unsqueeze(0)                    # => [1, 3, 512, 512]


# -----------------------
# 4. Visualize input image
# -----------------------
# For matplotlib, convert back to CPU/numpy in [0,1] range to show as an RGB image
image_for_plot = (image_tensor[0].clone().cpu() + 1) / 2.0  # bring from [-1,1] to [0,1]
image_for_plot = image_for_plot.clamp(0, 1).numpy()         # shape [3, 512, 512]
image_for_plot = image_for_plot.transpose(1, 2, 0)          # shape [512, 512, 3]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image_for_plot)
plt.title("Original (Padded) Image")
plt.axis("off")


# ---------------------
# 5. Encode -> Decode
# ---------------------
with torch.inference_mode():
    latent_dist = vae.encode(image_tensor).latent_dist
    latent = latent_dist.sample()               # shape [1, latent_channels, H//8, W//8] typically
    decoded_image = vae.decode(latent).sample # shape [1, 3, 512, 512]

# Bring decoded image to [0,1] for plotting
decoded_image_for_plot = (decoded_image[0].cpu() + 1) / 2.0
decoded_image_for_plot = decoded_image_for_plot.clamp(0, 1).numpy()  # [3, 512, 512]
decoded_image_for_plot = decoded_image_for_plot.transpose(1, 2, 0)   # [512, 512, 3]

plt.subplot(1, 3, 2)
plt.imshow(decoded_image_for_plot)
plt.title("Decoded Image")
plt.axis("off")


# ---------------------------------------------------
# 6. Re-encode the decoded image to visualize latents
# ---------------------------------------------------
# We must re-normalize to [-1,1] if we want to feed it back in
decoded_image_tensor = torch.tensor(decoded_image_for_plot.transpose(2, 0, 1),
                                    dtype=torch.float32, device="cuda")
decoded_image_tensor = decoded_image_tensor.unsqueeze(0)  # => [1, 3, 512, 512]
decoded_image_tensor = 2.0 * decoded_image_tensor - 1.0    # map from [0,1] to [-1,1]

with torch.inference_mode():
    re_encoded_latent = vae.encode(decoded_image_tensor).latent_dist.sample()
    # Typically shape: [1, latent_channels, 64, 64] if 512 input (depends on model downsampling)

# Visualize just the first latent channel
latent_first_channel = re_encoded_latent[0, 0].cpu().numpy()
plt.subplot(1, 3, 3)
plt.imshow(latent_first_channel, cmap="viridis")
plt.title("Re-Encoded Latent (ch=0)")
plt.axis("off")

plt.tight_layout()
plt.show()
