#!/usr/bin/env python
# crop_or_pad_step.py
"""
Step: Crop or pad each 3D brain image to a fixed 256x256x256 volume,
while shifting the origin so the center remains the same in physical space.

Features:
  - If dimension < 256 => symmetric pad
  - If dimension > 256 => symmetric crop
  - Recalculates the image origin to keep the same physical center
  - Parallel execution
  - in_place vs. staging logic

In your use case, you said scans never exceed 256 in any dimension,
so practically you'll only get padding. But this code is robust to both.
"""

import os
import shutil
import ants
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


###############################################################################
#             Helper: Center Crop/Pad to 256³ with Origin Recalc             #
###############################################################################
def center_crop_or_pad_256_with_origin_shift(img: ants.ANTsImage) -> ants.ANTsImage:
    """
    Given a 3D ANTsImage, produce a new ANTsImage of shape (256,256,256).
    - If dimension < 256 => symmetric pad
    - If dimension > 256 => symmetric crop
    - The center in *physical* space remains the same,
      so that the 'middle slice' in voxel space truly corresponds
      to the same anatomical plane as before.

    Implementation Details
    ----------------------
    - We compute how many voxels we add/remove before and after the center,
      then build a new NumPy array of size 256^3.
    - We also shift the image origin so that the new image center lines up
      with the old center in real-world (millimeter) coordinates.
    - We preserve direction (orientation matrix) and spacing.

    Returns
    -------
    ants.ANTsImage
        A new image of shape (256,256,256) with physically consistent center.
    """

    target_size = (256, 256, 256)

    # Convert to NumPy
    arr = img.numpy()
    original_shape = arr.shape  # (Dx, Dy, Dz)
    assert len(original_shape) == 3, "Expected a 3D image."

    # We'll build a new array of shape 256^3
    new_arr = np.zeros(target_size, dtype=arr.dtype)

    # -------------------------------------------------------------------------
    # 1. Compute the new origin shift so that the center remains in the same
    #    physical location.
    # -------------------------------------------------------------------------
    # The 'center' in voxel indices for the old image:
    #    center_in[i] = (original_dim[i] - 1) / 2
    # The 'center' in voxel indices for the new (256^3) image:
    #    center_out[i] = (256 - 1) / 2 = 127.5
    # The voxel shift: shift_vox[i] = center_out[i] - center_in[i]
    # Then the physical shift = shift_vox[i] * spacing[i].
    # We want new_origin = old_origin - physical_shift
    # so that the old center_in maps to the new center_out in physical space.

    old_origin = np.array(img.origin)
    spacing = np.array(img.spacing)

    # We'll accumulate how many voxels of shift we do in each dimension
    # (positive shift_vox => we are effectively adding padding =>
    #  we move the origin 'back' in real space).
    shift_vox = [0.0, 0.0, 0.0]

    center_out = [(target_size[i] - 1) / 2.0 for i in range(3)]

    in_dim = original_shape
    for i in range(3):
        center_in = (in_dim[i] - 1) / 2.0
        shift_vox[i] = center_out[i] - center_in  # could be positive (pad) or negative (crop)

    # This is how many mm we shift the origin by (voxel_shift * spacing)
    shift_mm = shift_vox * spacing  # element-wise
    new_origin = old_origin - shift_mm

    # -------------------------------------------------------------------------
    # 2. Build the slices for copy/paste
    #    We'll do symmetrical crop/pad so the new center matches old center.
    # -------------------------------------------------------------------------
    # For dimension i:
    #   diff = 256 - in_dim[i]
    #   if diff >= 0 => pad
    #   if diff < 0 => crop
    #
    # We'll define pad_before = diff//2, pad_after = diff - pad_before
    # or if diff < 0 => we interpret negative as crop.
    #
    # We'll figure out an in_start, in_end in the old array,
    # and out_start, out_end in the new array.

    def get_slices_for_dim(in_size, out_size):
        diff = out_size - in_size
        if diff == 0:
            # Perfectly matching dimension
            return (0, in_size), (0, out_size)
        elif diff > 0:
            # Pad
            pad_before = diff // 2
            pad_after = diff - pad_before
            # old array covers [0, in_size)
            in_start, in_end = 0, in_size
            # new array covers [pad_before, pad_before + in_size)
            out_start, out_end = pad_before, pad_before + in_size
        else:
            # Crop
            diff_abs = abs(diff)
            crop_before = diff_abs // 2
            crop_after = diff_abs - crop_before
            # We'll skip some from the front and the back
            in_start, in_end = crop_before, in_size - crop_after
            out_start, out_end = 0, out_size

        return (in_start, in_end), (out_start, out_end)

    slices_dim = []
    for i in range(3):
        (in_start, in_end), (out_start, out_end) = get_slices_for_dim(
            in_dim[i], target_size[i]
        )
        slices_dim.append(((in_start, in_end), (out_start, out_end)))

    # Copy the data
    x_in, x_out = slices_dim[0]
    y_in, y_out = slices_dim[1]
    z_in, z_out = slices_dim[2]

    new_arr[x_out[0]:x_out[1], y_out[0]:y_out[1], z_out[0]:z_out[1]] = \
        arr[x_in[0]:x_in[1],     y_in[0]:y_in[1],     z_in[0]:z_in[1]]

    # -------------------------------------------------------------------------
    # 3. Build a new ANTsImage with updated origin
    # -------------------------------------------------------------------------
    new_img = ants.from_numpy(new_arr, origin=tuple(new_origin), spacing=tuple(spacing), direction=img.direction)
    return new_img


###############################################################################
#                        Worker Function (Parallel)                           #
###############################################################################
def _process_single_subject_crop_or_pad(
    subject_folder: str,
    main_folder_input: str,
    output_base: str
) -> str:
    """
    Worker function to crop/pad a single subject's "image3d.nii.gz" to 256^3,
    recalculating the origin so that the center remains the same in real space.
    Returns subject_folder if successful.
    """
    subject_dir = os.path.join(main_folder_input, subject_folder)
    nifti_path  = os.path.join(subject_dir, "image3d.nii.gz")
    json_path   = os.path.join(subject_dir, "all_columns.json")

    if not os.path.isfile(nifti_path):
        raise FileNotFoundError(f"[{subject_folder}] {nifti_path} not found. Skipping.")

    # Output directory for this subject
    out_subdir = os.path.join(output_base, subject_folder)
    os.makedirs(out_subdir, exist_ok=True)

    out_nifti_path = os.path.join(out_subdir, "image3d.nii.gz")
    out_json_path  = os.path.join(out_subdir, "all_columns.json")

    # Read, crop/pad to 256, shifting origin so physical center remains consistent
    img = ants.image_read(nifti_path)
    new_img = center_crop_or_pad_256_with_origin_shift(img)
    ants.image_write(new_img, out_nifti_path)

    # Copy JSON if present
    if os.path.isfile(json_path):
        shutil.copy2(json_path, out_json_path)

    return subject_folder


###############################################################################
#                     Main Pipeline Function (Step)                           #
###############################################################################
def step_crop_or_pad_to_256_in_pipeline(
    main_folder_input: str,
    in_place: bool = True,
    staging_dir: str = "/mnt/dataset_storage/staging_general",
    final_output_folder: str = None,
    n_jobs: int = 1
):
    """
    Step: For each sub-XXX, load "image3d.nii.gz", center-crop or pad it
    to a fixed 256^3 volume, then save it. The origin is shifted so the
    old and new physical centers align (important if you plan to pick
    the same slice # across scans for analysis).

    in_place logic:
      - If True, overwrite each subject's 'image3d.nii.gz' in main_folder_input.
        => final_output_folder must be None or same as main_folder_input.
      - If False, write to 'staging_dir' then move to final_output_folder
        => final_output_folder must differ from main_folder_input.

    Parameters
    ----------
    main_folder_input : str
        The parent folder containing sub-XXX subfolders.
    in_place : bool
        Whether to overwrite the input data or do a staged approach.
    staging_dir : str
        Used only if in_place=False, defaults to /mnt/dataset_storage/staging_general.
    final_output_folder : str
        If in_place=False, must be a different folder path.
    n_jobs : int
        Number of parallel processes. If 1, run serially.
    """

    # -------------------------------------------------------------------------
    # Validate logic for in_place vs final_output_folder
    # -------------------------------------------------------------------------
    if in_place:
        if final_output_folder is not None and final_output_folder != main_folder_input:
            raise ValueError(
                "in_place=True => final_output_folder must be None or same as main_folder_input.\n"
                f"Got final_output_folder={final_output_folder}, main_folder_input={main_folder_input}."
            )
        output_base = main_folder_input
        use_staging = False
    else:
        if not final_output_folder:
            raise ValueError("in_place=False => Must provide final_output_folder.")
        if final_output_folder == main_folder_input:
            raise ValueError(
                "in_place=False => final_output_folder must differ from main_folder_input."
            )
        output_base = staging_dir
        use_staging = True
        os.makedirs(staging_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Collect subfolders
    # -------------------------------------------------------------------------
    subfolders = [
        sf for sf in sorted(os.listdir(main_folder_input))
        if sf.startswith("sub-") and os.path.isdir(os.path.join(main_folder_input, sf))
    ]
    total = len(subfolders)
    if total == 0:
        print("No sub-XXX folders found in input directory.")
        return

    print(f"\nStarting crop/pad to 256^3 (with origin shift) for {total} subject(s).")
    print(f"in_place={in_place}, n_jobs={n_jobs}")

    # -------------------------------------------------------------------------
    # Parallel or Serial
    # -------------------------------------------------------------------------
    completed_subjects = 0
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_sub = {}
            for subject in subfolders:
                fut = executor.submit(
                    _process_single_subject_crop_or_pad,
                    subject,
                    main_folder_input,
                    output_base
                )
                future_to_sub[fut] = subject

            for fut in as_completed(future_to_sub):
                sb = future_to_sub[fut]
                try:
                    fut.result()
                    completed_subjects += 1
                    print(f"[{completed_subjects}/{total}] Completed subject: {sb}")
                except Exception as ex:
                    print(f"ERROR in subject {sb}: {ex}")
    else:
        # Serial approach
        for i, subject in enumerate(subfolders, start=1):
            try:
                _process_single_subject_crop_or_pad(
                    subject,
                    main_folder_input,
                    output_base
                )
                completed_subjects += 1
                print(f"[{i}/{total}] Completed subject: {subject}")
            except Exception as ex:
                print(f"ERROR in subject {subject}: {ex}")

    # -------------------------------------------------------------------------
    # If in-place, done
    # -------------------------------------------------------------------------
    if not use_staging:
        print(f"\nCrop/Pad step done (in-place). {completed_subjects}/{total} processed.")
        return

    # -------------------------------------------------------------------------
    # Otherwise, move from staging to final_output_folder, then cleanup
    # -------------------------------------------------------------------------
    print("\nAll parallel tasks finished. Moving staged results to final location...")

    os.makedirs(final_output_folder, exist_ok=True)

    for subject_folder in subfolders:
        stage_subdir = os.path.join(staging_dir, subject_folder)
        if not os.path.isdir(stage_subdir):
            continue
        final_subdir = os.path.join(final_output_folder, subject_folder)
        os.makedirs(final_subdir, exist_ok=True)

        for f in os.listdir(stage_subdir):
            src = os.path.join(stage_subdir, f)
            dst = os.path.join(final_subdir, f)
            shutil.move(src, dst)

    # Cleanup staging
    shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"Crop/Pad to 256^3 step completed: {completed_subjects}/{total} subjects.")
    print(f"Results are in: {final_output_folder}")


###############################################################################
#                         Example Usage (if run as main)                       #
###############################################################################
if __name__ == "__main__":
    """
    Example usage:
      python crop_or_pad_step.py <input_folder> --in_place=False ...
    """
    main_folder_input = "/mnt/dataset_storage/data/adni_nacc_processing_steps/5_registration"  # after registration
    final_output_folder = "/mnt/dataset_storage/data/adni_nacc_processing_steps/8_cubic"

    # Example: staging mode
    step_crop_or_pad_to_256_in_pipeline(
        main_folder_input=main_folder_input,
        in_place=False,               # do not overwrite original
        final_output_folder=final_output_folder,
        n_jobs=30
    )

    # Example: in-place (overwrites input images)
    # step_crop_or_pad_to_256_in_pipeline(
    #     main_folder_input=main_folder_input,
    #     in_place=True,
    #     final_output_folder=None,
    #     n_jobs=1
    # )
