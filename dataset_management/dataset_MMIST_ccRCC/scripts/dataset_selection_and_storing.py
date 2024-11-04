from pathlib import Path
import argparse
import shutil
import pandas as pd
import pydicom
import numpy as np


def convert_dicom_to_npy(dicom_dir: Path, output_dir: Path) -> None:
    """
    Converts DICOM files in a directory to .npy format and saves them to the output directory.

    Parameters:
        dicom_dir (Path): Path to the directory containing DICOM files.
        output_dir (Path): Path to the directory where the .npy files will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for dicom_file in dicom_dir.glob("*.dcm"):
        dicom_data = pydicom.dcmread(dicom_file)
        npy_data = dicom_data.pixel_array
        npy_output_path = output_dir / f"{dicom_file.stem}.npy"
        np.save(npy_output_path, npy_data)


def process_patient_selection(selection_path: Path, single_selection_path: Path,
                              file_column: str = 'File', chosen_exam_column: str = 'chosen_exam') -> pd.DataFrame:
    """
    Processes patient selection by removing duplicates and prioritizing specified selection files.

    Parameters:
        selection_path (Path): Path to the main selection CSV file.
        single_selection_path (Path): Path to the secondary selection CSV file.
        file_column (str): Column name in the main selection file to rename.
        chosen_exam_column (str): Desired column name for the exam selection.

    Returns:
        pd.DataFrame: Merged and filtered patient selection DataFrame.
    """
    patient_selection = pd.read_csv(selection_path).rename(columns={file_column: chosen_exam_column})
    single_selection = pd.read_csv(single_selection_path)
    patient_selection = patient_selection[~patient_selection['case_id'].isin(single_selection['case_id'])]
    return pd.concat([patient_selection, single_selection], ignore_index=True)


def aggregate_parts(patient_output_dir: Path, remove_original: bool = False) -> None:
    """
    Aggregates part files for each exam in a patient's directory and saves as a single .npy file.

    Parameters:
        patient_output_dir (Path): Path to the patient's output directory containing .npy files.
        remove_original (bool): If True, remove the original part files after aggregation.
    """
    npy_files = sorted(patient_output_dir.glob("*.npy"))
    grouped_files = {}

    # Group files by prefix (before the '-')
    for file in npy_files:
        prefix = file.stem.split('-')[0]
        grouped_files.setdefault(prefix, []).append(file)

    # Process each group of parts and aggregate them
    for prefix, parts in grouped_files.items():
        # Load each slice and stack them into a single 3D numpy array
        parts_data = [np.load(part) for part in sorted(parts)]  # Ensure the parts are sorted correctly
        aggregated_data = np.stack(parts_data, axis=0)  # Stack along a new axis for slices

        # Save the aggregated data as a new .npy file
        aggregated_file_path = patient_output_dir / f"{prefix}.npy"
        np.save(aggregated_file_path, aggregated_data)

        # Optionally, remove the original part files
        if remove_original:
            for part in parts:
                part.unlink()


def process_and_convert_data(patient_selection: pd.DataFrame, exam_type: str,
                             root_dir: Path, output_dir: Path, remove_original: bool = False) -> None:
    """
    Processes patient data by converting DICOM to .npy files and aggregating parts if needed.

    Parameters:
        patient_selection (pd.DataFrame): DataFrame containing patient IDs and chosen exam paths.
        exam_type (str): Either 'CT' or 'MRI', used for directory naming.
        root_dir (Path): Path to the root directory of the dataset.
        output_dir (Path): Path to the directory where processed files will be saved.
        remove_original (bool): If True, remove the original .npy part files after aggregation.
    """
    for patient_id, exam_path in zip(patient_selection['case_id'], patient_selection['chosen_exam']):
        exam_full_path = root_dir / exam_path[1:]
        patient_output_dir = output_dir / patient_id / exam_type

        if exam_full_path.exists():
            convert_dicom_to_npy(exam_full_path, patient_output_dir)
            aggregate_parts(patient_output_dir, remove_original=remove_original)


def select_mmist_ccrcc_dataset(root_dir: Path, output_dir: Path,
                               ct_single_exam_selection: Path, mri_single_exam_selection: Path,
                               ct_selection_patients: Path, mri_selection_patients: Path,
                               remove_original: bool) -> None:
    """
    Selects and processes the MMIST_ccRCC dataset by filtering patient data and converting images to .npy format.

    Parameters:
        root_dir (Path): Path to the root directory of the dataset.
        output_dir (Path): Path to the directory where processed files will be saved.
        ct_single_exam_selection (Path): Path to the CT single exam selection CSV file.
        mri_single_exam_selection (Path): Path to the MRI single exam selection CSV file.
        ct_selection_patients (Path): Path to the CT patient selection CSV file.
        mri_selection_patients (Path): Path to the MRI patient selection CSV file.
        remove_original (bool): If True, remove the original .npy part files after aggregation.
    """
    ct_patient_selection = process_patient_selection(ct_selection_patients, ct_single_exam_selection)
    mri_patient_selection = process_patient_selection(mri_selection_patients, mri_single_exam_selection)

    process_and_convert_data(ct_patient_selection, 'CT', root_dir, output_dir, remove_original)
    process_and_convert_data(mri_patient_selection, 'MRI', root_dir, output_dir, remove_original)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select patient and images according to MMIST_ccRCC dataset, handle DICOM format, and convert images to .npy arrays."
    )

    parser.add_argument('--ct_single_exam_selection', type=Path, required=True,
                        help="Path to the CT single exam selection CSV file.")
    parser.add_argument('--mri_single_exam_selection', type=Path, required=True,
                        help="Path to the MRI single exam selection CSV file.")
    parser.add_argument('--ct_selection_patients', type=Path, required=True,
                        help="Path to the CT patient selection CSV file.")
    parser.add_argument('--mri_selection_patients', type=Path, required=True,
                        help="Path to the MRI patient selection CSV file.")
    parser.add_argument('--input_dir', type=Path, required=True,
                        help="Root directory where the MMIST_ccRCC dataset is located.")
    parser.add_argument('--output_dir', type=Path, required=True,
                        help="Directory where the converted files will be saved.")
    parser.add_argument('--remove', type=bool, required=True, help="Remove original part files after aggregation.")

    args = parser.parse_args()

    select_mmist_ccrcc_dataset(args.input_dir, args.output_dir,
                               args.ct_single_exam_selection, args.mri_single_exam_selection,
                               args.ct_selection_patients, args.mri_selection_patients, args.remove)
