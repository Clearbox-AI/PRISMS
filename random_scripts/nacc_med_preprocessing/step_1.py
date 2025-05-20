import os
import shutil
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform


def reorient_nifti(
        input_nifti_path: str,
        output_nifti_path: str,
        target_orientation: tuple = ("R", "A", "S")
) -> tuple:
    """
    Load a NIfTI image, reorient it to the specified target_orientation,
    and save it to the specified output path.

    Returns
    -------
    (orig_ornt_axcodes, final_ornt_axcodes) : tuple of strings
        The original orientation and the final orientation as 3-letter codes.
    """
    img = nib.load(input_nifti_path)

    # Determine the current orientation
    orig_ornt = io_orientation(img.affine)
    orig_ornt_axcodes = "".join(nib.orientations.ornt2axcodes(orig_ornt))

    # Construct the orientation transform
    final_ornt = axcodes2ornt(target_orientation)
    transform = ornt_transform(orig_ornt, final_ornt)

    # Apply the transform
    reoriented_img = img.as_reoriented(transform)

    # Save the reoriented image
    nib.save(reoriented_img, output_nifti_path)

    final_ornt_axcodes = "".join(target_orientation)
    return orig_ornt_axcodes, final_ornt_axcodes


def step_reorient(
        main_folder_input: str,
        orientation_codes: tuple = ("R", "A", "S"),
        in_place: bool = True,
        staging_dir: str = None,
        final_output_folder: str = None
):
    """
    A pipeline step to reorient images to a specified orientation (default: RAS).
    It uses a staging directory to store intermediate files only if in_place=False.

    Parameters
    ----------
    main_folder_input : str
        Path to the directory containing the images to reorient.
        This directory is scanned for sub-folders like sub-XXX.
    orientation_codes : tuple, optional
        The desired orientation (e.g., ('R','A','S')).
    in_place : bool, optional
        If True, the final reoriented images overwrite the original files
        (no staging directory is used).
    staging_dir : str, optional
        Path to an existing or new directory where intermediate files are written
        *if in_place=False*. If None, defaults to "reorient_staging" in the CWD.
    final_output_folder : str, optional
        If in_place=False, the final output will be copied here, preserving subfolder structure.
        This must not match main_folder_input (otherwise it's effectively in-place).
        If in_place=True, this must be None or the same as main_folder_input.

    Returns
    -------
    None
        Prints progress and a summary of how many images were reoriented from
        each original orientation to the final orientation.
    """

    # --------------------------------------------------------------------------
    # Validate the combination of arguments
    # --------------------------------------------------------------------------
    if in_place:
        # If in-place, final_output_folder must be either None or the same as input
        if final_output_folder is not None and final_output_folder != main_folder_input:
            raise ValueError(
                "If in_place=True, 'final_output_folder' must either be None "
                "or the same as 'main_folder_input'."
            )
        # We don't actually need a staging directory here, but the user might
        # have provided one. We'll simply ignore it (or you could raise an error).
        if staging_dir is not None:
            print("Warning: 'staging_dir' is ignored when in_place=True.")
    else:
        # If not in-place, final_output_folder is required and must differ from input
        if final_output_folder is None:
            raise ValueError(
                "If in_place=False, you must specify 'final_output_folder'."
            )
        if final_output_folder == main_folder_input:
            raise ValueError(
                "If in_place=False, 'final_output_folder' cannot be the same "
                "as 'main_folder_input'."
            )
        # Ensure staging_dir exists or create a default if needed
        if staging_dir is None:
            staging_dir = "/mnt/dataset_storage/staging_general"
        os.makedirs(staging_dir, exist_ok=True)
        print(f"Using staging directory: {staging_dir}")

    # --------------------------------------------------------------------------
    # Identify subfolders that start with 'sub-'
    # --------------------------------------------------------------------------
    subfolders = [
        sf for sf in sorted(os.listdir(main_folder_input))
        if sf.startswith("sub-") and os.path.isdir(os.path.join(main_folder_input, sf))
    ]
    total_subfolders = len(subfolders)
    if total_subfolders == 0:
        print("No sub-XXX folders found. Exiting.")
        return

    # Stats dictionary: track how many scans came from each original orientation
    stats = {}
    processed_count = 0

    # --------------------------------------------------------------------------
    # Main logic: reorient either in-place or to staging
    # --------------------------------------------------------------------------
    for i, subject_folder in enumerate(subfolders, start=1):
        input_subdir = os.path.join(main_folder_input, subject_folder)
        input_nifti = os.path.join(input_subdir, "image3d.nii.gz")
        input_json = os.path.join(input_subdir, "all_columns.json")

        if not os.path.isfile(input_nifti):
            print(f"\nWarning: no NIfTI found for {subject_folder}, skipping.")
            continue

        # Reorient in-place or to staging
        if in_place:
            # Overwrite in-place
            # (orig -> same path)
            orig_axcodes, final_axcodes = reorient_nifti(
                input_nifti,
                input_nifti,
                orientation_codes
            )

            # JSON is already there, so no need to copy
            staged_nifti = input_nifti  # for the progress print only

        else:
            # Create subfolder in staging
            staging_subdir = os.path.join(staging_dir, subject_folder)
            os.makedirs(staging_subdir, exist_ok=True)

            # Paths for the reoriented file
            staged_nifti = os.path.join(staging_subdir, "image3d.nii.gz")
            staged_json = os.path.join(staging_subdir, "all_columns.json")

            # Reorient and save to staging
            orig_axcodes, final_axcodes = reorient_nifti(
                input_nifti,
                staged_nifti,
                orientation_codes
            )
            # Copy JSON if it exists
            if os.path.isfile(input_json):
                shutil.copy2(input_json, staged_json)

        # Update stats
        if orig_axcodes not in stats:
            stats[orig_axcodes] = 0
        stats[orig_axcodes] += 1
        processed_count += 1

        # Simple progress bar
        fraction = i / total_subfolders
        bar_len = 20
        filled_len = int(bar_len * fraction)
        bar = "#" * filled_len + "-" * (bar_len - filled_len)
        print(
            f"\rReorienting: |{bar}| {i}/{total_subfolders} "
            f"({fraction * 100:.1f}%) {subject_folder} {orig_axcodes} -> {final_axcodes}",
            end="", flush=True
        )

    # --------------------------------------------------------------------------
    # If not in-place, move staged results into final_output_folder, then clean up
    # --------------------------------------------------------------------------
    print()
    if not in_place:
        print("\nMoving staged files into final location...\n")
        for subject_folder in subfolders:
            staging_subdir = os.path.join(staging_dir, subject_folder)
            if not os.path.isdir(staging_subdir):
                continue

            final_subdir = os.path.join(final_output_folder, subject_folder)
            os.makedirs(final_subdir, exist_ok=True)

            # Move each file from staging to final
            for f in os.listdir(staging_subdir):
                src = os.path.join(staging_subdir, f)
                dest = os.path.join(final_subdir, f)
                shutil.move(src, dest)

        # Optionally remove the staging directory
        shutil.rmtree(staging_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # Print summary
    # --------------------------------------------------------------------------
    print("Reorientation step complete. Below is the summary of orientation changes:")
    for original_orientation, count in stats.items():
        print(f"  {original_orientation} -> {''.join(orientation_codes)} : {count} scans")


if __name__ == "__main__":
    """
    Example usage:
    - If you want to overwrite in place:
        step_reorient(
            main_folder_input="/path/to/input_folder",
            in_place=True  # final_output_folder ignored
        )
    - If you want to reorient to a different output folder:
        step_reorient(
            main_folder_input="/path/to/input_folder",
            in_place=False,
            final_output_folder="/path/to/desired_output_folder"
        )
    """
    # Edit below to your actual folders for a real run.
    input_folder = "/mnt/dataset_storage/data/adni_nacc_processing_steps/1_select_available_mri"
    final_output_folder = "/mnt/dataset_storage/data/adni_nacc_processing_steps/2_reorient_ras"

    # Example: do NOT overwrite in place, reorient to a new folder
    step_reorient(
        main_folder_input=input_folder,
        orientation_codes=("R", "A", "S"),
        in_place=False,
        staging_dir=None,
        final_output_folder=final_output_folder
    )
