#!/usr/bin/env python
# bias_correction_step.py
"""
Step: ANTs N4 Bias-Field Correction with parallel processing and
a staging vs. in-place approach, supporting separate scan & mask folders.
"""

import os
import shutil
import ants
from concurrent.futures import ProcessPoolExecutor, as_completed

###############################################################################
#                               Worker Function                               #
###############################################################################
def _process_single_subject_n4(
    subject_folder: str,
    main_folder_input: str,
    mask_folder_input: str,
    output_base: str,      # Where corrected results get written (in-place or staging)
    mask_filename: str,    # e.g. "brain_mask.nii.gz" if present
    shrink_factor: int,
    iterations: tuple,
    tolerance: float
) -> str:
    """
    Worker function to be executed in parallel, performing N4 bias correction
    for a single subject folder (sub-XXX).

    Parameters
    ----------
    subject_folder : str
        The folder name (e.g. "sub-001").
    main_folder_input : str
        Path to the parent folder that has the subject's brain scan subfolders.
    mask_folder_input : str
        Path to the parent folder that has the subject's mask subfolders
        (could be the same as main_folder_input).
    output_base : str
        The base directory where results will be written (in-place or staging).
    mask_filename : str
        If provided, we attempt to find it in mask_folder_input/sub-XXX.
    shrink_factor, iterations, tolerance
        N4 bias correction parameters.

    Returns
    -------
    str
        The `subject_folder` to indicate successful completion.
    """
    # -------------------------------------------------------------------------
    # Determine input paths
    # -------------------------------------------------------------------------
    subject_dir_scans = os.path.join(main_folder_input, subject_folder)
    nifti_path        = os.path.join(subject_dir_scans, "image3d.nii.gz")
    json_path         = os.path.join(subject_dir_scans, "all_columns.json")

    if not os.path.isfile(nifti_path):
        raise FileNotFoundError(f"[{subject_folder}] '{nifti_path}' not found. Skipping.")

    # If masks are in a different folder structure, fetch them from mask_folder_input
    subject_dir_masks = os.path.join(mask_folder_input, subject_folder)
    mask_path = None
    if mask_filename:
        candidate_mask = os.path.join(subject_dir_masks, mask_filename)
        if os.path.isfile(candidate_mask):
            mask_path = candidate_mask
        # else we do not have a mask for this subject

    # -------------------------------------------------------------------------
    # Determine output path (depends on in-place vs staging)
    # -------------------------------------------------------------------------
    out_subdir = os.path.join(output_base, subject_folder)
    os.makedirs(out_subdir, exist_ok=True)

    out_nifti_path = os.path.join(out_subdir, "image3d.nii.gz")
    out_json_path  = os.path.join(out_subdir, "all_columns.json")

    # -------------------------------------------------------------------------
    # Perform N4 Bias Correction using ANTs
    # -------------------------------------------------------------------------
    img = ants.image_read(nifti_path)

    mask_img = None
    if mask_path:
        mask_img = ants.image_read(mask_path)

    conv_dict = {'iters': list(iterations), 'tol': tolerance}

    corrected_img = ants.n4_bias_field_correction(
        image=img,
        mask=mask_img,
        shrink_factor=shrink_factor,
        convergence=conv_dict,
        verbose=False
    )

    # Save corrected image
    ants.image_write(corrected_img, out_nifti_path)

    # Copy JSON if it exists
    if os.path.isfile(json_path):
        shutil.copy2(json_path, out_json_path)

    return subject_folder


###############################################################################
#                 Main Pipeline Function (with Parallel Support)              #
###############################################################################
def step_bias_correction_ants_in_pipeline(
    main_folder_input: str,
    mask_folder_input: str = None,
    in_place: bool = True,
    # Always use this staging directory if in_place=False:
    staging_dir: str = "/mnt/dataset_storage/staging_general",
    final_output_folder: str = None,
    mask_filename: str = None,
    shrink_factor: int = 4,
    iterations: tuple = (50, 50, 30, 20),
    tolerance: float = 1e-7,
    n_jobs: int = 1
):
    """
    Pipeline step: For each sub-XXX, run N4 bias correction with ANTsPy, either in-place
    or via a staging directory. Now supports separate input folders for scans vs. masks.

    Parameters
    ----------
    main_folder_input : str
        Path to the parent folder with subject sub-XXX subfolders containing the T1/scan.
    mask_folder_input : str, optional
        Path to the parent folder with subject sub-XXX subfolders containing the masks.
        If None or identical to `main_folder_input`, we treat them as being in the same place.
    in_place : bool
        If True, overwrite the existing data in `main_folder_input`.
        Otherwise, use staging -> final_output_folder.
    staging_dir : str
        Path to the staging directory (used only if in_place=False).
        Defaults to /mnt/dataset_storage/staging_general.
    final_output_folder : str
        If in_place=False, must be provided and must differ from main_folder_input.
        If in_place=True, must be None or exactly main_folder_input.
    mask_filename : str
        If provided, we look for this mask file in the `mask_folder_input/sub-XXX`.
    shrink_factor, iterations, tolerance
        N4 bias correction parameters.
    n_jobs : int
        Number of parallel processes to use.

    Raises
    ------
    ValueError
        If the combination of `in_place` and `final_output_folder` is inconsistent.
    """
    # -------------------------------------------------------------------------
    # Determine which folder to use for masks
    # -------------------------------------------------------------------------
    if mask_folder_input is None:
        mask_folder_input = main_folder_input

    # -------------------------------------------------------------------------
    # Validate 'in_place' logic
    # -------------------------------------------------------------------------
    if in_place:
        # final_output_folder must be None or match input
        if final_output_folder is not None and final_output_folder != main_folder_input:
            raise ValueError(
                "in_place=True => final_output_folder must be None or the same as main_folder_input.\n"
                f"Got final_output_folder={final_output_folder}, main_folder_input={main_folder_input}."
            )
        output_base = main_folder_input
        use_staging = False
    else:
        # If in_place=False, final_output_folder must be set and must differ
        if not final_output_folder:
            raise ValueError(
                "in_place=False => Must provide a final_output_folder."
            )
        if final_output_folder == main_folder_input:
            raise ValueError(
                "in_place=False => final_output_folder must differ from main_folder_input."
            )
        output_base = staging_dir  # We'll write corrected images here first
        use_staging = True
        os.makedirs(staging_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Collect subject folders from main_folder_input
    # -------------------------------------------------------------------------
    subfolders = [
        sf for sf in sorted(os.listdir(main_folder_input))
        if sf.startswith("sub-") and os.path.isdir(os.path.join(main_folder_input, sf))
    ]
    total = len(subfolders)
    if total == 0:
        print("No sub-XXX folders found in input directory.")
        return

    print(f"\nStarting N4 Bias Correction for {total} subject(s).")
    print(f"in_place={in_place}, n_jobs={n_jobs}")
    if mask_folder_input != main_folder_input:
        print(f"Using a separate mask folder: {mask_folder_input}")

    # -------------------------------------------------------------------------
    # Parallel or Serial Execution
    # -------------------------------------------------------------------------
    completed_subjects = 0
    if n_jobs > 1:
        # Parallel mode
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_sub = {}
            for subject in subfolders:
                fut = executor.submit(
                    _process_single_subject_n4,
                    subject,
                    main_folder_input,
                    mask_folder_input,
                    output_base,
                    mask_filename,
                    shrink_factor,
                    iterations,
                    tolerance
                )
                future_to_sub[fut] = subject

            for fut in as_completed(future_to_sub):
                sb = future_to_sub[fut]
                try:
                    fut.result()  # or store the return if needed
                    completed_subjects += 1
                    print(f"[{completed_subjects}/{total}] Completed subject: {sb}")
                except Exception as ex:
                    print(f"ERROR in subject {sb}: {ex}")
    else:
        # Serial mode
        for i, subject in enumerate(subfolders, start=1):
            try:
                _process_single_subject_n4(
                    subject,
                    main_folder_input,
                    mask_folder_input,
                    output_base,
                    mask_filename,
                    shrink_factor,
                    iterations,
                    tolerance
                )
                completed_subjects += 1
                print(f"[{i}/{total}] Completed subject: {subject}")
            except Exception as ex:
                print(f"ERROR in subject {subject}: {ex}")

    # -------------------------------------------------------------------------
    # If we did in-place, we are done now
    # -------------------------------------------------------------------------
    if not use_staging:
        print(f"\nN4 Bias Correction done (in-place). {completed_subjects}/{total} subjects processed.")
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

    # Finally, remove the entire staging directory
    shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"N4 Bias Correction step completed: {completed_subjects}/{total} subjects processed.")
    print(f"Results saved under: {final_output_folder}")


###############################################################################
#                            Example usage stub                                #
###############################################################################
if __name__ == "__main__":
    """
    Example usage:
      python bias_correction_step.py <input_folder_scans> <input_folder_masks> [mask_filename]

    If you want to pass a mask filename, e.g. 'brain_mask.nii.gz', do so in the call.
    """
    main_folder_scans = "/mnt/dataset_storage/data/adni_nacc_processing_steps/2_reorient_ras"  # T1 scans
    main_folder_masks = "/mnt/dataset_storage/data/adni_nacc_processing_steps/3_skull_strip"  # If the same as scans
    final_output_folder = "/mnt/dataset_storage/data/adni_nacc_processing_steps/4_field_correction"
    mask_filename = "brain_mask.nii.gz"

    # Example: Run with staging (in_place=False), using a separate or same folder for masks
    step_bias_correction_ants_in_pipeline(
        main_folder_input=main_folder_scans,
        mask_folder_input=main_folder_masks,
        mask_filename=mask_filename,
        in_place=False,
        final_output_folder=final_output_folder,
        shrink_factor=4,
        iterations=(50, 50, 30, 20),
        tolerance=1e-7,
        n_jobs=30  # parallel processes
    )

    # Example: Run in-place (overwrites scans directly), still can have masks in same or different folder
    # step_bias_correction_ants_in_pipeline(
    #     main_folder_input=main_folder_scans,
    #     mask_folder_input=main_folder_masks,
    #     mask_filename=mask_filename,
    #     in_place=True,
    #     final_output_folder=None,  # or the same as scans
    #     n_jobs=2
    # )
