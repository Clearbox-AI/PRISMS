import os
import json
import shutil
import pandas as pd
from tqdm import tqdm

def main():
    # -------------------------------------------------------------------------
    # 1) Define your paths
    # -------------------------------------------------------------------------
    csv1 = "/mnt/dataset_storage/data/nacc_dataset/additional_files/investigator_mri_final_ADNI_diagnosis.csv"  # ADNI
    csv2 = "/mnt/dataset_storage/data/nacc_dataset/additional_files/investigator_mri_final_NACC_diagnosis.csv"  # NACC
    csv_both = "/mnt/dataset_storage/data/nacc_dataset/additional_files/investigator_mri_nacc65.csv"

    main_folder_input = "/mnt/dataset_storage/data/adni_nacc_processing_steps/0_start"
    adni_input_dir = os.path.join(main_folder_input, "BIDS_ADNI_diagnosys")
    nacc_input_dir = os.path.join(main_folder_input, "BIDS_NACC_diagnosys")

    # The merged output directory
    main_folder_output = "/mnt/dataset_storage/data/adni_nacc_processing_steps/1_select_available_mri"

    # -------------------------------------------------------------------------
    # 2) Read the CSVs into pandas
    # -------------------------------------------------------------------------
    df_adni = pd.read_csv(csv1)     # CSV for ADNI
    df_nacc = pd.read_csv(csv2)     # CSV for NACC
    df_both = pd.read_csv(csv_both) # Combined data (subject to NACCMVOL filter)

    # Keep only rows where NACCMVOL == 1 in df_both
    df_both = df_both[df_both["NACCMVOL"] == 1]

    # -------------------------------------------------------------------------
    # 3) Index ADNI/NACC data by NACCID
    # -------------------------------------------------------------------------
    df_adni.set_index("NACCID", inplace=True)
    df_nacc.set_index("NACCID", inplace=True)

    # -------------------------------------------------------------------------
    # 4) Determine valid NACCIDs (those in df_both + df_adni or df_nacc)
    # -------------------------------------------------------------------------
    adni_naccids = set(df_adni.index)
    nacc_naccids = set(df_nacc.index)
    both_naccids = set(df_both["NACCID"])

    final_adni_naccids = adni_naccids & both_naccids
    final_nacc_naccids = nacc_naccids & both_naccids

    # -------------------------------------------------------------------------
    # 5) Create a 'scan_date' column in df_both for sorting only (we will drop it later)
    # -------------------------------------------------------------------------
    df_both["scan_date"] = pd.to_datetime(
        df_both["MRIYR"].astype(str) + "-" +
        df_both["MRIMO"].astype(str) + "-" +
        df_both["MRIDY"].astype(str),
        errors="coerce"  # invalid combos -> NaT
    )

    # -------------------------------------------------------------------------
    # 6) Helper to list 'sub-XXXX' directories in a given base folder
    # -------------------------------------------------------------------------
    def get_subject_dirs(base_dir):
        """
        Return a dict { 'sub-XXXX': /full/path/to/sub-XXXX, ... }
        for each sub-XXXX folder under base_dir.
        """
        subjects = {}
        if not os.path.isdir(base_dir):
            return subjects

        for entry in os.listdir(base_dir):
            if entry.startswith("sub-"):
                sub_path = os.path.join(base_dir, entry)
                if os.path.isdir(sub_path):
                    subjects[entry] = sub_path
        return subjects

    adni_subject_dirs = get_subject_dirs(adni_input_dir)
    nacc_subject_dirs = get_subject_dirs(nacc_input_dir)

    # -------------------------------------------------------------------------
    # 7) Function to process subjects for ADNI or NACC:
    #    - Find row from df_main
    #    - Find LATEST row from df_both
    #    - Merge them in a single row
    #    - Remove 'scan_date'
    #    - Copy the .nii.gz file + store the merged row in JSON
    #
    #    Returns the count of successfully processed subjects.
    # -------------------------------------------------------------------------
    def process_subjects(naccids, df_main, subject_dirs, label="ADNI"):
        count_saved = 0

        # Wrap naccids with tqdm for a progress bar
        for naccid in tqdm(naccids, desc=f"Processing {label}", unit="subject"):
            sub_folder = f"sub-{naccid}"
            if sub_folder not in subject_dirs:
                continue

            subject_path = subject_dirs[sub_folder]
            anat_path = os.path.join(subject_path, "anat")
            if not os.path.isdir(anat_path):
                continue

            # Pick a .nii.gz file if present
            nii_candidates = [fn for fn in os.listdir(anat_path) if fn.endswith(".nii.gz")]
            if not nii_candidates:
                continue

            nii_name = nii_candidates[0]
            nii_source_path = os.path.join(anat_path, nii_name)

            # Single row from df_main (ADNI or NACC)
            if naccid not in df_main.index:
                continue
            row_main = df_main.loc[naccid].to_dict()

            # All rows from df_both for that NACCID
            subset_both = df_both[df_both["NACCID"] == naccid]
            if subset_both.empty:
                continue

            # Sort by scan_date, take the last (latest row)
            subset_both_sorted = subset_both.sort_values("scan_date")
            latest_row = subset_both_sorted.iloc[-1].to_dict()

            # Convert each to a one-row DataFrame
            df_main_row = pd.DataFrame([row_main])
            df_both_row = pd.DataFrame([latest_row])

            # Concatenate horizontally => single row with combined columns
            df_combined = pd.concat([df_main_row, df_both_row], axis=1)

            # Remove the 'scan_date' column (if present)
            if "scan_date" in df_combined.columns:
                df_combined.drop(columns=["scan_date"], inplace=True)

            # Convert to dict
            combined_dict = df_combined.to_dict(orient="records")[0]

            # -----------------------------------------------------------------
            # 8) Create output directory + copy .nii.gz + write JSON
            # -----------------------------------------------------------------
            out_sub_dir = os.path.join(main_folder_output, sub_folder)
            os.makedirs(out_sub_dir, exist_ok=True)

            json_output_path = os.path.join(out_sub_dir, "all_columns.json")
            with open(json_output_path, "w") as f_out:
                json.dump(combined_dict, f_out, indent=2)

            nii_output_path = os.path.join(out_sub_dir, "image3d.nii.gz")
            shutil.copy2(nii_source_path, nii_output_path)

            # If we get here, we've successfully processed this subject
            count_saved += 1

        return count_saved

    # -------------------------------------------------------------------------
    # 8) Process ADNI subjects, then NACC subjects
    #    and show how many were processed
    # -------------------------------------------------------------------------
    adni_saved_count = process_subjects(final_adni_naccids, df_adni, adni_subject_dirs, label="ADNI")
    nacc_saved_count = process_subjects(final_nacc_naccids, df_nacc, nacc_subject_dirs, label="NACC")

    total_saved = adni_saved_count + nacc_saved_count

    print(f"\nSummary:")
    print(f"  - ADNI subjects processed: {adni_saved_count}")
    print(f"  - NACC subjects processed: {nacc_saved_count}")
    print(f"  - Total processed subjects: {total_saved}")


if __name__ == "__main__":
    main()
