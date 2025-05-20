#!/usr/bin/env python
# registration_step.py
"""
Step 5: Registration to a template (e.g., MNI152_T1_1mm),
with 9-DOF linear (affine) transform (locking out shear).
Parallelized and respecting in-place vs. staging logic.
"""

import os
import shutil
import ants
from concurrent.futures import ProcessPoolExecutor, as_completed

###############################################################################
#                              Worker Function                                #
###############################################################################
def _process_single_subject_registration(
    subject_folder: str,
    main_folder_input: str,
    output_base: str,       # Where results are written (in-place or staging)
    template_t1_path: str,
    reg_iterations: tuple,
    use_histogram_match: bool
) -> str:
    """
    Worker function to be executed in parallel, performing a 9-DOF linear
    registration of sub-XXX's 'image3d.nii.gz' onto the template.
    We then warp the subject image into template space.

    Parameters
    ----------
    subject_folder : str
        The folder name (e.g. "sub-001").
    main_folder_input : str
        Path to the parent folder that has the subject's subfolders with images.
    output_base : str
        Base directory for output (in-place or staging).
    template_t1_path : str
        Path to the template T1 (e.g., MNI152_T1_1mm.nii.gz).
    reg_iterations : tuple
        Registration iterations, e.g. (100,50,20).
    use_histogram_match : bool
        Whether to use histogram matching in registration.

    Returns
    -------
    str
        The `subject_folder`, indicating successful completion.
    """
    # -------------------------------------------------------------------------
    # Identify paths
    # -------------------------------------------------------------------------
    subject_dir = os.path.join(main_folder_input, subject_folder)
    nifti_path  = os.path.join(subject_dir, "image3d.nii.gz")
    json_path   = os.path.join(subject_dir, "all_columns.json")

    if not os.path.isfile(nifti_path):
        raise FileNotFoundError(f"[{subject_folder}] '{nifti_path}' not found. Skipping.")

    # Output directory for this subject
    out_subdir = os.path.join(output_base, subject_folder)
    os.makedirs(out_subdir, exist_ok=True)

    out_warped_path = os.path.join(out_subdir, "image3d.nii.gz")  # resampled into template space
    out_json_path   = os.path.join(out_subdir, "all_columns.json")

    # -------------------------------------------------------------------------
    # Load images
    # -------------------------------------------------------------------------
    subject_img  = ants.image_read(nifti_path)
    template_img = ants.image_read(template_t1_path)

    # -------------------------------------------------------------------------
    # 9-DOF Affine => Lock out shear
    # (transform parameters: [Tx,Ty,Tz, Rx,Ry,Rz, Sx,Sy,Sz, Kx,Ky,Kz],
    #  1=locked, 0=free. We lock shear = Kx,Ky,Kz)
    # -------------------------------------------------------------------------
    restrict_9dof = [0,0,0, 0,0,0, 0,0,0, 1,1,1]

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------
    reg = ants.registration(
        fixed=template_img,        # template is 'fixed'
        moving=subject_img,        # subject is 'moving'
        type_of_transform="Affine",
        restrict_transforms=[restrict_9dof],
        reg_iterations=reg_iterations,
        use_histogram_matching=use_histogram_match,
        verbose=False
    )

    # -------------------------------------------------------------------------
    # Warp the subject image into template space
    # -------------------------------------------------------------------------
    warped = ants.apply_transforms(
        fixed=template_img,
        moving=subject_img,
        transformlist=reg['fwdtransforms'],
        interpolator='linear'
    )

    ants.image_write(warped, out_warped_path)

    # Copy the JSON if present
    if os.path.isfile(json_path):
        shutil.copy2(json_path, out_json_path)

    # Optionally, you could also save the transform(s) if you want them for later
    # e.g. copy reg['fwdtransforms'] into out_subdir. For now we skip that.

    return subject_folder


###############################################################################
#                     Main Pipeline Function (Parallel)                       #
###############################################################################
def step_linear_registration_in_pipeline(
    main_folder_input: str,
    template_t1_path: str = "/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz",
    in_place: bool = True,
    staging_dir: str = "/mnt/dataset_storage/staging_general",
    final_output_folder: str = None,
    reg_iterations: tuple = (100, 50, 20),
    use_histogram_match: bool = True,
    n_jobs: int = 1
):
    """
    Step 5: For each sub-XXX, run 9-DOF (shear-locked) linear registration
    to a given template (e.g. MNI152_T1_1mm).

    Results:
      - We produce image3d.nii.gz in template space.

    The in_place vs. staging logic is the same as in previous steps:
      - If in_place=True, we overwrite each subject's 'image3d.nii.gz' directly.
        => final_output_folder must be None or exactly main_folder_input.
      - If in_place=False, we use a staging folder for intermediate results
        => final_output_folder must be specified and different from main_folder_input.
        => Once done, we move from staging to final_output_folder.

    Parameters
    ----------
    main_folder_input : str
        Directory with sub-XXX subfolders, each containing 'image3d.nii.gz'.
    template_t1_path : str
        Path to the template T1 image (e.g., MNI152_T1_1mm.nii.gz).
    in_place : bool
        Overwrite data in the input folder?
    staging_dir : str
        Used only if in_place=False. Typically '/mnt/dataset_storage/staging_general'.
    final_output_folder : str
        If in_place=False, must be provided and not equal to main_folder_input.
    reg_iterations : tuple
        E.g. (100,50,20).
    use_histogram_match : bool
        Whether to use histogram matching in registration.
    n_jobs : int
        Number of parallel processes for the registration tasks.
    """

    # -------------------------------------------------------------------------
    # Validate in_place logic
    # -------------------------------------------------------------------------
    if in_place:
        if final_output_folder is not None and final_output_folder != main_folder_input:
            raise ValueError(
                "in_place=True => final_output_folder must be None or the same as main_folder_input.\n"
                f"Got final_output_folder={final_output_folder}, main_folder_input={main_folder_input}."
            )
        output_base = main_folder_input
        use_staging = False
    else:
        if not final_output_folder:
            raise ValueError("in_place=False => Must provide a final_output_folder.")
        if final_output_folder == main_folder_input:
            raise ValueError(
                "in_place=False => final_output_folder must differ from main_folder_input."
            )
        # We'll write everything to staging first, then move to final output
        output_base = staging_dir
        use_staging = True
        os.makedirs(staging_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Collect subject folders
    # -------------------------------------------------------------------------
    subfolders = [
        sf for sf in sorted(os.listdir(main_folder_input))
        if sf.startswith("sub-") and os.path.isdir(os.path.join(main_folder_input, sf))
    ]
    total = len(subfolders)
    if total == 0:
        print("No sub-XXX folders found in input directory.")
        return

    print(f"\nStarting 9-DOF linear registration for {total} subject(s).")
    print(f"in_place={in_place}, n_jobs={n_jobs}")
    print(f"Template: {template_t1_path}")

    # -------------------------------------------------------------------------
    # Parallel or Serial Execution
    # -------------------------------------------------------------------------
    completed_subjects = 0
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_sub = {}
            for subject in subfolders:
                fut = executor.submit(
                    _process_single_subject_registration,
                    subject,
                    main_folder_input,
                    output_base,
                    template_t1_path,
                    reg_iterations,
                    use_histogram_match
                )
                future_to_sub[fut] = subject

            for fut in as_completed(future_to_sub):
                sb = future_to_sub[fut]
                try:
                    fut.result()  # We only need to catch exceptions
                    completed_subjects += 1
                    print(f"[{completed_subjects}/{total}] Completed subject: {sb}")
                except Exception as ex:
                    print(f"ERROR in subject {sb}: {ex}")
    else:
        # Serial
        for i, subject in enumerate(subfolders, start=1):
            try:
                _process_single_subject_registration(
                    subject,
                    main_folder_input,
                    output_base,
                    template_t1_path,
                    reg_iterations,
                    use_histogram_match
                )
                completed_subjects += 1
                print(f"[{i}/{total}] Completed subject: {subject}")
            except Exception as ex:
                print(f"ERROR in subject {subject}: {ex}")

    # -------------------------------------------------------------------------
    # If we did in-place, we are done
    # -------------------------------------------------------------------------
    if not use_staging:
        print(f"\nRegistration done (in-place). {completed_subjects}/{total} subjects processed.")
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

    print(f"Registration step completed: {completed_subjects}/{total} subjects processed.")
    print(f"Results saved under: {final_output_folder}")


###############################################################################
#                            Example usage stub                                #
###############################################################################
if __name__ == "__main__":
    """
    Example usage:
      python registration_step.py <input_folder> [<final_output_folder>]

    - If in_place=True, we overwrite the scans in input_folder (no final_output_folder needed).
    - If in_place=False, specify final_output_folder (must differ from input_folder).
    """
    main_folder_input = "/mnt/dataset_storage/data/adni_nacc_processing_steps/4_field_correction"  # after step 4
    final_output_folder = "/mnt/dataset_storage/data/adni_nacc_processing_steps/5_registration"

    template_t1 = "/mnt/venvs/fsl/data/standard/MNI152_T1_1mm.nii.gz"

    # Example: run with staging (recommended for large datasets)
    step_linear_registration_in_pipeline(
        main_folder_input=main_folder_input,
        template_t1_path=template_t1,
        in_place=False,
        final_output_folder=final_output_folder,
        reg_iterations=(100, 50, 20),  # example
        use_histogram_match=True,
        n_jobs=30
    )

    # Example: run in-place (overwrites your input data!)
    # step_linear_registration_in_pipeline(
    #     main_folder_input=main_folder_input,
    #     template_t1_path=template_t1,
    #     in_place=True,
    #     final_output_folder=None,  # or same as main_folder_input
    #     n_jobs=2
    # )
