import argparse
import nibabel as nib
import numpy as np

def check_imaginary_part(nii_file):
    """
    Checks if a NIfTI file contains an imaginary part.

    Args:
        nii_file (str): Path to the NIfTI file.

    Returns:
        bool: True if the data has an imaginary part, False otherwise.
    """
    # Load the NIfTI file
    print(f"Loading NIfTI file: {nii_file}")
    img = nib.load(nii_file)
    data = img.get_fdata()

    # Check the dimensionality of the data
    print(f"Image shape: {data.shape} (3D scan)")

    # Check if the data is a complex object
    if np.iscomplexobj(data):
        # Check if there are any non-zero imaginary parts
        has_imaginary = np.any(np.imag(data) != 0)
        if has_imaginary:
            print("The 3D scan contains non-zero imaginary parts.")
        else:
            print("The 3D scan is complex but has no significant imaginary parts (all zeros).")
        return has_imaginary
    else:
        print("The 3D scan does not contain complex data (no imaginary parts).")
        return False

def main():
    parser = argparse.ArgumentParser(description="Check if a NIfTI (.nii) file contains an imaginary part.")
    parser.add_argument(
        "--nii_file",
        type=str,
        required=True,
        help="Path to the NIfTI file (e.g., a T1-weighted 3D brain scan)."
    )
    args = parser.parse_args()

    # Run the imaginary part check
    has_imaginary = check_imaginary_part(args.nii_file)

    # Print the result
    if has_imaginary:
        print("Imaginary parts are present in the input 3D scan.")
    else:
        print("No imaginary parts found in the input 3D scan.")

if __name__ == "__main__":
    main()
