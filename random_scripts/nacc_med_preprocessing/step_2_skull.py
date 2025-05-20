#!/usr/bin/env python
# skull_stripping_step.py
"""
Skull Stripping pipeline step with optional 9-DOF affine transforms,
parallelization, and special rules for 'in_place' vs. staging usage.
"""

import os
import shutil
import ants
from concurrent.futures import ProcessPoolExecutor, as_completed

###############################################################################
#                        Worker function (parallel)                           #
###############################################################################
def _process_single_subject(
    subject_folder: str,
    main_folder_input: str,
    output_base: str,             # either the input dir (in_place) or the staging dir
    template_t1_path: str,
    template_mask_path: str,
    restrict_9dof: bool,
    reg_iterations: tuple,
    use_histogram_match: bool
):
    """
    Worker function to be run in parallel via ProcessPoolExecutor.

    Parameters
    ----------
    subject_folder : str
        Name of the subject subfolder (e.g. "sub-001").
    main_folder_input : str
        Path containing the subject subfolder for input.
    output_base : str
        Where to write the results (brain_mask, stripped image, etc.).
        This is either the same as main_folder_input (if in_place=True)
        or a staging directory if in_place=False.
    ...
    """

    subject_dir = os.path.join(main_folder_input, subject_folder)
    nifti_path  = os.path.join(subject_dir, "image3d.nii.gz")
    json_path   = os.path.join(subject_dir, "all_columns.json")

    if not os.path.isfile(nifti_path):
        raise FileNotFoundError(f"[{subject_folder}] '{nifti_path}' not found. Skipping.")

    # Output folder for this subject: either main_folder_input/subject_folder (in_place)
    # or staging_dir/subject_folder (not in_place).
    out_subdir = os.path.join(output_base, subject_folder)
    os.makedirs(out_subdir, exist_ok=True)

    out_mask_path     = os.path.join(out_subdir, "brain_mask.nii.gz")
    out_stripped_path = os.path.join(out_subdir, "image3d.nii.gz")   # We'll overwrite the name

    # Run the registration + mask warp
    subject_img  = ants.image_read(nifti_path)
    template_img = ants.image_read(template_t1_path)
    template_msk = ants.image_read(template_mask_path)

    # Restrict to 9 DOF => lock shear (Kx,Ky,Kz)
    if restrict_9dof:
        restrict_params = [0,0,0, 0,0,0, 0,0,0, 1,1,1]  # lock shear
    else:
        restrict_params = [0]*12  # full affine

    reg = ants.registration(
        fixed=subject_img,
        moving=template_img,
        type_of_transform="Affine",
        restrict_transforms=[restrict_params],
        reg_iterations=reg_iterations,
        use_histogram_matching=use_histogram_match,
        verbose=False
    )

    # Warp template mask -> subject space
    warped_mask = ants.apply_transforms(
        fixed=subject_img,
        moving=template_msk,
        transformlist=reg['fwdtransforms'],
        interpolator='nearestNeighbor'
    )

    # Binarize the warped mask
    final_mask = warped_mask.threshold_image(0.5, 1.0).iMath_fill_holes()
    ants.image_write(final_mask, out_mask_path)

    # Save stripped T1
    stripped = subject_img * final_mask
    ants.image_write(stripped, out_stripped_path)

    # Copy JSON if present
    if os.path.isfile(json_path):
        out_json = os.path.join(out_subdir, "all_columns.json")
        shutil.copy2(json_path, out_json)

    return subject_folder  # just so the caller knows which subject was done


###############################################################################
#                Main pipeline function with the requested logic              #
###############################################################################
def step_skull_stripping_in_pipeline(
    main_folder_input: str,
    template_t1_path: str = "/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz",
    template_mask_path: str = "/usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz",
    in_place: bool = True,
    # We always use /mnt/dataset_storage/staging_general for staging if needed:
    staging_dir: str = "/mnt/dataset_storage/staging_general",
    final_output_folder: str = None,
    restrict_9dof: bool = True,
    reg_iterations: tuple = (100, 50, 20),
    use_histogram_match: bool = True,
    n_jobs: int = 1
):
    """
    Pipeline step: For each sub-XXX, run linear skull stripping using the MNI152 1mm template by default.

    Required logic:
      - If in_place=True, we overwrite the input data. No staging is used.
        => final_output_folder must be either None or the same as main_folder_input.
      - If in_place=False, we require a final_output_folder different from main_folder_input.
        => We'll place intermediate results in staging_dir, then move them to final_output_folder.
      - staging_dir is always "/mnt/dataset_storage/staging_general" (unused if in_place=True).
      - If final_output_folder == main_folder_input but in_place=False, we abort (inconsistent).
    ...
    """
    # Validation checks
    if in_place:
        # If in_place=True => final_output_folder must be None or same as main_folder_input
        if final_output_folder is not None and final_output_folder != main_folder_input:
            raise ValueError(
                "in_place=True => 'final_output_folder' must be None or match 'main_folder_input'. "
                f"Got final_output_folder='{final_output_folder}', main_folder_input='{main_folder_input}'."
            )
        output_base = main_folder_input  # We'll write results directly here
        use_staging = False
    else:
        # If in_place=False => final_output_folder must be provided and different from main_folder_input
        if final_output_folder is None:
            raise ValueError("in_place=False => Must provide a 'final_output_folder'.")
        if final_output_folder == main_folder_input:
            raise ValueError(
                "in_place=False => 'final_output_folder' must differ from 'main_folder_input' to avoid overwriting."
            )
        output_base = staging_dir  # We'll do all processing in staging
        use_staging = True

    # Collect subfolders
    subfolders = [
        sf for sf in sorted(os.listdir(main_folder_input))
        if sf.startswith("sub-") and os.path.isdir(os.path.join(main_folder_input, sf))
    ]
    total = len(subfolders)
    if total == 0:
        print("No sub-XXX folders found in input.")
        return

    print(f"\nFound {total} subject(s). in_place={in_place}. n_jobs={n_jobs}")

    # Ensure staging dir if needed
    if use_staging:
        # Always /mnt/dataset_storage/staging_general (as per your requirement)
        staging_dir = "/mnt/dataset_storage/staging_general"
        os.makedirs(staging_dir, exist_ok=True)

    # Parallel processing
    completed_subjects = 0
    tasks = []
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_sub = {}
            for subject_folder in subfolders:
                future = executor.submit(
                    _process_single_subject,
                    subject_folder,
                    main_folder_input,
                    output_base,
                    template_t1_path,
                    template_mask_path,
                    restrict_9dof,
                    reg_iterations,
                    use_histogram_match
                )
                future_to_sub[future] = subject_folder

            for future in as_completed(future_to_sub):
                sb = future_to_sub[future]
                try:
                    _ = future.result()
                    completed_subjects += 1
                    print(f"[{completed_subjects}/{total}] Completed: {sb}")
                except Exception as ex:
                    print(f"[{sb}] ERROR: {ex}")
    else:
        # Serial loop
        for i, subject_folder in enumerate(subfolders, start=1):
            try:
                _process_single_subject(
                    subject_folder,
                    main_folder_input,
                    output_base,
                    template_t1_path,
                    template_mask_path,
                    restrict_9dof,
                    reg_iterations,
                    use_histogram_match
                )
                completed_subjects += 1
                print(f"[{i}/{total}] Completed: {subject_folder}")
            except Exception as ex:
                print(f"[{subject_folder}] ERROR: {ex}")

    # If in_place=True => we are done (no staging to handle)
    if not use_staging:
        print(f"\nSkull-stripping pipeline completed. {completed_subjects}/{total} subjects processed in-place.")
        return

    # Otherwise move from staging -> final_output_folder
    print("\nAll parallel tasks finished. Moving from staging to final output...")

    for subject_folder in subfolders:
        stage_subdir = os.path.join(staging_dir, subject_folder)
        if not os.path.isdir(stage_subdir):
            continue

        final_subdir = os.path.join(final_output_folder, subject_folder)
        os.makedirs(final_subdir, exist_ok=True)

        for f in os.listdir(stage_subdir):
            src = os.path.join(stage_subdir, f)
            dest = os.path.join(final_subdir, f)
            shutil.move(src, dest)

    # Cleanup staging if desired
    shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"Skull-stripping pipeline step completed.\n"
          f" => {completed_subjects}/{total} subjects processed.\n"
          f" => Results moved to {final_output_folder}.")


###############################################################################
#                      Example command-line usage (if any)                    #
###############################################################################
if __name__ == "__main__":
    # Example usage
    # main_folder_input = "/mnt/dataset_storage/data/adni_nacc_processing_steps/2_reorient_ras"
    # final_output_folder = "/mnt/dataset_storage/data/adni_nacc_processing_steps/3_skull_strip"
    main_folder_input = "/mnt/dataset_storage/data/adni_nacc_processing_steps/4_field_correction"
    final_output_folder = "/mnt/dataset_storage/data/adni_nacc_processing_steps/3_skull_strip_after_field_correction"
    template_t1 = "/mnt/venvs/fsl/data/standard/MNI152_T1_1mm.nii.gz"
    template_msk= "/mnt/venvs/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"

    # Example: run in-place
    # step_skull_stripping_in_pipeline(
    #     main_folder_input=main_folder_input,
    #     template_t1_path=template_t1,
    #     template_mask_path=template_msk,
    #     in_place=True,        # Overwrites original data
    #     final_output_folder=None,  # Must be None or same as input
    #     restrict_9dof=True,
    #     n_jobs=4
    # )

    # Example: run with staging
    step_skull_stripping_in_pipeline(
        main_folder_input=main_folder_input,
        template_t1_path=template_t1,
        template_mask_path=template_msk,
        in_place=False,
        final_output_folder=final_output_folder, # must be different from input
        restrict_9dof=True,
        n_jobs=30
    )
