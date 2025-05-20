import os
import pandas as pd
from collections import defaultdict

def check_field_manu_model():
    """
        Processes nacc and adni MRI metadata and filters for final selected subjects with available MRI data.

        Steps:
        1. Loads a directory of selected subjects (named as sub-XXXX) and extracts NACCIDs.
        2. Loads an MRI metadata CSV with manufacturer, model, and field strength information.
        3. Filters the metadata to only include volumetric MRIs (NACCMVOL == 1).
        4. Parses the scan date from year, month, and day columns.
        5. Keeps only the latest scan entry per NACCID based on scan date.
        6. Filters rows to only include NACCIDs that are found in the selected subject directory.
        7. Builds and returns three dictionaries:
           - `field_dict`: maps each unique MRIFIELD value to a list of NACCIDs.
           - `manu_dict`: maps each unique MRIMANU value to a list of NACCIDs.
           - `model_dict`: maps each unique MRIMODL value to a list of NACCIDs.

        Returns:
            Tuple[Dict[Any, List[str]], Dict[Any, List[str]], Dict[Any, List[str]]]:
                - field_dict, manu_dict, model_dict
        """

    # -------------------------------------------------------------------------
    # 1) Define the paths
    # -------------------------------------------------------------------------
    output_dir = "/mnt/dataset_storage/data/adni_nacc_processing_steps/1_select_available_mri"
    csv_nacc = "/mnt/dataset_storage/data/nacc_dataset/additional_files/investigator_mri_nacc65.csv"

    # -------------------------------------------------------------------------
    # 2) Gather the final subject IDs from the output directory
    #    (i.e., sub-XXXX folders)
    # -------------------------------------------------------------------------
    final_naccids = []
    for entry in os.listdir(output_dir):
        if entry.startswith("sub-"):
            sub_id = entry.replace("sub-", "")
            final_naccids.append(sub_id)

    final_naccids = set(final_naccids)
    print(f"Found {len(final_naccids)} subjects in {output_dir}")

    # -------------------------------------------------------------------------
    # 3) Load the CSV that has MRIFIELD, MRIMANU, MRIMODL (and time info)
    # -------------------------------------------------------------------------
    df_nacc = pd.read_csv(csv_nacc)

    df_nacc = df_nacc[df_nacc["NACCMVOL"] == 1]
    df_nacc["NACCID"] = df_nacc["NACCID"].astype(str)

    # -------------------------------------------------------------------------
    # 4) Create a 'scan_date' column
    # -------------------------------------------------------------------------
    df_nacc["scan_date"] = pd.to_datetime(
        df_nacc["MRIYR"].astype(str) + "-" +
        df_nacc["MRIMO"].astype(str) + "-" +
        df_nacc["MRIDY"].astype(str),
        errors="coerce"
    )

    # -------------------------------------------------------------------------
    # 5) Keep latest scan per subject
    # -------------------------------------------------------------------------
    df_nacc_sorted = df_nacc.sort_values(["NACCID", "scan_date"])
    df_latest = df_nacc_sorted.groupby("NACCID").tail(1).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 6) Filter to only subjects found in the output directory
    # -------------------------------------------------------------------------
    df_final = df_latest[df_latest["NACCID"].isin(final_naccids)]
    print(f"DataFrame reduced to {len(df_final)} rows after filtering.\n")

    # -------------------------------------------------------------------------
    # 7) Create dictionaries mapping each feature value to list of NACCIDs
    # -------------------------------------------------------------------------
    field_dict = defaultdict(list)
    manu_dict = defaultdict(list)
    model_dict = defaultdict(list)

    for _, row in df_final.iterrows():
        field_dict[row["MRIFIELD"]].append(row["NACCID"])
        manu_dict[row["MRIMANU"]].append(row["NACCID"])
        model_dict[row["MRIMODL"]].append(row["NACCID"])

    return dict(field_dict), dict(manu_dict), dict(model_dict)

if __name__ == "__main__":
    field_dict, manu_dict, model_dict = check_field_manu_model()
