import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from generative.networks.nets import AutoencoderKL
from skimage.transform import resize

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

##############################################################################
#                           MAIN FUNCTION
##############################################################################

def main(args):
    """
    1) Load the 2D complex MRI image
    2) Convert to magnitude & normalize
    3) Embed that 2D slice into a (256,256,256) volume
    4) Expand to shape (1,1,256,256,256)
    5) Encode & Decode with AutoencoderKL
    """
    # -----------------------------------------------------------------
    # 1) Load complex MRI image from .npy
    # -----------------------------------------------------------------
    coil_complex_image = np.load(args.image_path)  # shape could be [H,W] or [1,H,W]

    # If it's just a single 2D slice with shape [H,W], add an extra dimension [1,H,W]
    if coil_complex_image.ndim == 2:
        coil_complex_image = coil_complex_image[np.newaxis, ...]  # => [1,H,W]

    # coil_complex_image now shape => [1,H,W]

    # Display original magnitude
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(np.abs(coil_complex_image[0]), cmap="gray")
    axs[0, 0].set_title("1) Original Magnitude (2D Slice)")
    axs[0, 0].axis("off")

    # -----------------------------------------------------------------
    # 2) Normalize magnitude
    # -----------------------------------------------------------------
    coil_complex_image_norm = normalize_complex_coil_image(coil_complex_image)
    # shape still => [1,H,W], complex values but normalized in magnitude

    # Convert to magnitude only (single channel)
    magnitude_2d = np.abs(coil_complex_image_norm[0])  # shape => [H,W]

    axs[0, 1].imshow(magnitude_2d, cmap="gray")
    axs[0, 1].set_title("2) Normalized Magnitude (2D)")
    axs[0, 1].axis("off")

    # -----------------------------------------------------------------
    # 3) Embed the 2D slice into a 3D volume of shape [256,256,256]
    # -----------------------------------------------------------------

    magnitude_2d_resized = resize(
        magnitude_2d,
        (128, 128),
        order=1,  # Bilinear interpolation
        mode='reflect',  # Border mode
        anti_aliasing=True,  # Anti-aliasing filter
        preserve_range=True  # Keep original range
    )

    # Display resized magnitude
    axs[1, 0].imshow(magnitude_2d_resized, cmap="gray")
    axs[1, 0].set_title("3) Resized Magnitude (128x128)")
    axs[1, 0].axis("off")

    D, H, W = 128, 128, 128

    # Create an empty 3D volume
    volume_3d = np.zeros((D, H, W), dtype=magnitude_2d_resized.dtype)

    # Let's place our 2D slice at depth=0 (or anywhere you like, e.g., the middle).
    # You must ensure your 2D slice is 256x256. If not, you'll need resizing.
    # For demonstration, assume coil_complex_image_norm is already 256x256.
    # volume_3d[0, :, :] = magnitude_2d_resized  # place the 2D slice at the first slice of depth dimension
    volume_3d = np.repeat(magnitude_2d_resized[np.newaxis, :, :], D, axis=0)

    # Display one slice of the 3D volume to confirm
    axs[1, 1].imshow(volume_3d[0], cmap="gray")
    axs[1, 1].set_title("4) Embedded Resized Slice in 3D Volume")
    axs[1, 1].axis("off")

    # Now volume_3d => shape [256,256,256], single channel in "data" sense

    # -----------------------------------------------------------------
    # 4) Expand to final shape [1,1,256,256,256] => [batch, channels, D, H, W]
    # -----------------------------------------------------------------
    volume_5d = volume_3d[np.newaxis, np.newaxis, ...]  # => (1,1,256,256,256)

    # Check shape
    print("volume_5d shape:", volume_5d.shape)

    # -----------------------------------------------------------------
    # 5) Load the pretrained MONAI Autoencoder & Encode->Decode
    # -----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    monai_autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=8,
        num_channels=[64, 128, 256],
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False
    ).to(device)

    # Load pretrained weights
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    monai_autoencoder.load_state_dict(state_dict)
    monai_autoencoder.eval()

    # Convert numpy volume to torch
    volume_tensor = torch.from_numpy(volume_5d).float().to(device)  # [1,1,256,256,256]

    with torch.no_grad():
        # The forward pass returns (recon, z_mu, z_sigma)
        # We'll do it manually to show each piece:
        z_mu, z_sigma = monai_autoencoder.encode(volume_tensor)  # => encode
        z = monai_autoencoder.sampling(z_mu, z_sigma)            # => sample
        reconstruction = monai_autoencoder.decode(z)             # => decode

    # # reconstruction => shape [1,1,256,256,256]
    # # We'll visualize the first slice [depth=0] of the reconstruction
    # reconstruction_np = reconstruction.detach().cpu().numpy()[0, 0]  # => shape [256,256,256]
    # recon_slice_0 = reconstruction_np[0]                             # => shape [256,256]
    #
    # axs[1, 0].imshow(recon_slice_0, cmap="gray")
    # axs[1, 0].set_title("Reconstruction (depth=0)")
    # axs[1, 0].axis("off")
    #
    # # And one more slice, say depth=128 (the middle)
    # recon_slice_mid = reconstruction_np[128]
    # axs[1, 1].imshow(recon_slice_mid, cmap="gray")
    # axs[1, 1].set_title("Reconstruction (depth=128)")
    # axs[1, 1].axis("off")
    #
    # plt.tight_layout()
    # plt.show()

    # reconstruction => shape [1,1,128,128,128]
    # We'll visualize two slices: depth=64 (center) and depth=0
    reconstruction_np = reconstruction.detach().cpu().numpy()[0, 0]  # => shape [128,128,128]
    recon_slice_center = reconstruction_np[1]  # => shape [128,128]
    recon_slice_0 = reconstruction_np[0]  # => shape [128,128]

    # Display reconstruction slices
    fig_recon, axs_recon = plt.subplots(1, 2, figsize=(10, 5))
    axs_recon[0].imshow(recon_slice_center, cmap="gray")
    axs_recon[0].set_title("Reconstruction (center slice)")
    axs_recon[0].axis("off")

    axs_recon[1].imshow(recon_slice_0, cmap="gray")
    axs_recon[1].set_title("Reconstruction (depth=0)")
    axs_recon[1].axis("off")

    plt.tight_layout()
    plt.show()


##############################################################################
#                           ENTRY POINT
##############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Encode and decode a complex MRI image using a pretrained MONAI AutoencoderKL, "
            "filling a (1,1,256,256,256) volume for a 3D model."
        )
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the .pt file containing the pretrained MONAI AutoencoderKL weights."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to your .npy file containing a (H,W) or (1,H,W) complex MRI image."
    )
    args = parser.parse_args()
    main(args)
