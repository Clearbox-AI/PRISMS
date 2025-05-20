import os
import math
import nibabel as nib
import matplotlib.pyplot as plt


def plot_middle_slices(folder_path, naccid_list=None):
    """
    Look in `folder_path` for subdirectories named 'sub-XXXX', each containing
    a 3D scan file 'image3d.nii.gz'. Extract the middle slice of each scan
    and display them in plots (64 slices per figure by default), labeled by subject ID.

    If `naccid_list` is provided, only display subjects matching those IDs.
    """
    # 1) Collect all 'sub-XXXX' directories
    all_subjects = [
        d for d in sorted(os.listdir(folder_path))
        if d.startswith("sub-") and os.path.isdir(os.path.join(folder_path, d))
    ]

    # 2) Filter by `naccid_list` if provided
    if naccid_list is not None:
        subjects = [
            d for d in all_subjects
            if d.replace("sub-", "") in naccid_list
        ]
    else:
        subjects = all_subjects

    if not subjects:
        print("No matching subjects found.")
        return

    # 3) Break the list into chunks
    chunk_size = 64
    num_chunks = math.ceil(len(subjects) / chunk_size)

    for chunk_idx, start_idx in enumerate(range(0, len(subjects), chunk_size)):
        end_idx = start_idx + chunk_size
        chunk_subjects = subjects[start_idx:end_idx]

        fig, axes = plt.subplots(8, 8, figsize=(20, 20))
        axes = axes.flatten()

        for i, subject_dir in enumerate(chunk_subjects):
            subject_id = subject_dir.replace("sub-", "")  # e.g., "1234"
            nii_path = os.path.join(folder_path, subject_dir, "image3d.nii.gz")

            if not os.path.isfile(nii_path):
                print(f"Missing file for {subject_id}, skipping.")
                continue

            img = nib.load(nii_path)
            data = img.get_fdata()

            # Pick the middle slice in the 3rd dimension
            mid_slice_index = data.shape[2] // 2
            middle_slice = data[:, :, mid_slice_index]

            ax = axes[i]
            ax.imshow(middle_slice, cmap='gray')
            ax.set_title(subject_id)
            ax.axis("off")

        plt.suptitle(f"Chunk {chunk_idx + 1} (subjects {start_idx + 1} to {min(end_idx, len(subjects))})")
        plt.tight_layout()
        plt.show()


def main():
    folder_path = "/mnt/dataset_storage/data/adni_nacc_processing_steps/8_cubic"

    from random_scripts.check_stast_dataset import check_field_manu_model
    field_dict, manu_dict, model_dict = check_field_manu_model()

    # Example list of NACC IDs
    naccid_list = None

    # Call with or without the list
    plot_middle_slices(folder_path, naccid_list=naccid_list)


if __name__ == "__main__":
    main()
