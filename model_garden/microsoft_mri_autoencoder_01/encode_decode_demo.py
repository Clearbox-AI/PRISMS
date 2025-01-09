#!/usr/bin/env python

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL


##############################################################################
#                           UTILITY FUNCTIONS
##############################################################################

def normalize_complex_coil_image(complex_img: np.ndarray) -> np.ndarray:
    """
    Scales the complex-valued image by the 99.5th percentile of its magnitude
    to keep values in a reasonable range for neural networks.
    """
    max_val = np.percentile(np.abs(complex_img), 99.5)
    return complex_img / (max_val + 1e-8)


def complex_to_two_channel_image(complex_img: np.ndarray) -> np.ndarray:
    """
    Splits a complex-valued image into a 2-channel image [real, imag].

    Example:
      Input shape:  [N, H, W] or [H, W]
      Output shape: [2, N, H, W] if input was [N,H,W]; or [2,H,W] if input was [H,W].
    """
    real = np.real(complex_img)
    imag = np.imag(complex_img)
    return np.stack((real, imag), axis=0)  # shape => [2, ...]


def two_channel_to_complex_image(two_ch_img: np.ndarray) -> np.ndarray:
    """
    Converts a 2-channel image [real, imag] back to a complex-valued image.

    If two_ch_img has shape [1, 2, H, W], output shape is [1, H, W] (complex).
    """
    # Expect shape [batch=1, 2, H, W]
    real = two_ch_img[0, 0]
    imag = two_ch_img[0, 1]
    return (real + 1j * imag)[np.newaxis, ...]


##############################################################################
#                           MAIN FUNCTION
##############################################################################

def main(args):
    """
    1. Load the MRI image from args.image_path
    2. Normalize
    3. Convert to 2-channel
    4. Encode
    5. Decode
    6. Display
    """
    # 1) Load complex MRI image from npy
    coil_complex_image = np.load(args.image_path)  # shape could be [H,W] or [N,H,W]

    # If it's just a single 2D slice with shape [H,W], add an extra dimension [1,H,W]
    if coil_complex_image.ndim == 2:
        coil_complex_image = coil_complex_image[np.newaxis, ...]

    # 2) Normalize
    coil_complex_image_norm = normalize_complex_coil_image(coil_complex_image)

    # 3) Convert to 2-channel [real, imag]
    #    This will produce shape [2, N, H, W] if input was [N,H,W].
    two_channel_image = complex_to_two_channel_image(coil_complex_image_norm)

    # If original had shape [1, H, W], then after stack => [2, 1, H, W].
    # Pytorch expects [batch, channels, height, width]. Let's reorder to [1, 2, H, W].
    two_channel_image = np.transpose(two_channel_image, (1, 0, 2, 3))  # => [1, 2, H, W]

    # Convert to torch tensor
    two_channel_tensor = torch.from_numpy(two_channel_image).float()

    # 4) Load the pretrained MRI autoencoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = AutoencoderKL.from_pretrained(args.checkpoint_path).to(device)
    autoencoder.eval()

    two_channel_tensor = two_channel_tensor.to(device)

    # 5) Encode
    with torch.no_grad():
        encoder_out = autoencoder.encode(two_channel_tensor)
        latents = encoder_out.latent_dist.mean  # shape: [1, latent_channels, H/8, W/8] (approx)

    # For display, grab one channel of the latents
    latents_for_display = latents[0, 0].detach().cpu().numpy()

    # 6) Decode
    with torch.no_grad():
        decoded = autoencoder.decode(latents).sample  # shape: [1, 2, H, W]

    # Convert 2-channel decoded back to complex
    decoded_np = decoded.detach().cpu().numpy()
    recon_complex_image = two_channel_to_complex_image(decoded_np)

    # 7) Display side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Original magnitude
    axs[0].imshow(np.abs(coil_complex_image[0]), cmap="gray")
    axs[0].set_title("Original Image (Magnitude)")

    # Latent channel
    axs[1].imshow(latents_for_display, cmap="gray")
    axs[1].set_title("Latent (One Channel)")

    # Decoded magnitude
    axs[2].imshow(np.abs(recon_complex_image[0]), cmap="gray")
    axs[2].set_title("Decoded (Magnitude)")

    plt.tight_layout()
    plt.show()


##############################################################################
#                           ENTRY POINT
##############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode and decode a complex MRI image using a pretrained AutoencoderKL."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path or identifier of the pretrained autoencoder (e.g. 'microsoft/mri-autoencoder-v0.1')."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to your .npy file containing a complex MRI image."
    )
    args = parser.parse_args()
    main(args)
