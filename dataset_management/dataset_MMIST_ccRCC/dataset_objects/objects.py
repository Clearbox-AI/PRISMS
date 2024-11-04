import os
import numpy as np
from typing import List, Dict, Optional


class Exam:
    """Represents a medical exam file for a patient."""

    def __init__(self, file_path: str, lazy: bool = True):
        self.file_path = file_path
        self.lazy = lazy
        self._data = None if lazy else self._load_data()

    def _load_data(self):
        """Load the exam file (.npy)"""
        return np.load(self.file_path)

    def load(self):
        """Lazy load the exam file (.npy) if not already loaded."""
        if self.lazy and self._data is None:
            self._data = self._load_data()
        return self._data

    def get_slice(self, index: Optional[int] = None):
        """
        Get a specific slice from the exam, if the exam is multi-slice.
        If no index is specified, return the whole exam.
        """
        data = self.load()
        if index is not None and len(data.shape) > 2:  # Assuming multi-slice exams have more than 2 dimensions
            return data[index]
        return data

    def __repr__(self):
        return f"Exam(data_shape={self._data.shape if self._data is not None else 'Not Loaded'})"


class Patient:
    """Represents a patient with CT and/or MRI exams."""

    def __init__(self, patient_id: str, exams: Dict[str, List[Exam]]):
        self.patient_id = patient_id
        self.exams = exams  # Dictionary with keys 'CT' and/or 'MRI'

    def get_exams_by_type(self, exam_type: str) -> Optional[List[Exam]]:
        """Retrieve exams for a given type (CT or MRI)."""
        return self.exams.get(exam_type)

    def get_all_exams(self) -> List[Exam]:
        """Retrieve all exams for the patient."""
        all_exams = []
        for exam_list in self.exams.values():
            all_exams.extend(exam_list)
        return all_exams

    def __repr__(self):
        return f"Patient(patient_id={self.patient_id}, exam_types={list(self.exams.keys())})"


class Dataset:
    """Represents the entire dataset with optional lazy loading."""

    def __init__(self, dataset_dir: str, lazy: bool = True):
        self.dataset_dir = dataset_dir
        self.lazy = lazy
        self.patients = self._load_patients()

    def _load_patients(self) -> Dict[str, Patient]:
        """Load patients and their exams from the dataset directory."""
        patients = {}
        for patient_id in os.listdir(self.dataset_dir):
            patient_path = os.path.join(self.dataset_dir, patient_id)
            if os.path.isdir(patient_path):
                exams = self._load_exams_for_patient(patient_path)
                patients[patient_id] = Patient(patient_id, exams)
        return patients

    def _load_exams_for_patient(self, patient_path: str) -> Dict[str, List[Exam]]:
        """Load CT and/or MRI exams for a patient."""
        exams = {}
        for exam_type in ['CT', 'MRI']:
            exam_path = os.path.join(patient_path, exam_type)
            if os.path.exists(exam_path):
                exam_files = [
                    Exam(os.path.join(exam_path, f), lazy=self.lazy)
                    for f in os.listdir(exam_path) if f.endswith('.npy')
                ]
                if exam_files:
                    exams[exam_type] = exam_files
        return exams

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """Retrieve a specific patient by ID."""
        return self.patients.get(patient_id)

    def get_patients_by_exam_type(self, exam_type: str) -> List[Patient]:
        """Retrieve all patients that have a specific exam type (CT or MRI)."""
        return [p for p in self.patients.values() if exam_type in p.exams]

    def __repr__(self):
        return f"Dataset(patients_count={len(self.patients)}, lazy_loading={self.lazy})"


# Example usage
dataset_dir = "..."
dataset = Dataset(dataset_dir, lazy=True)

# Accessing a specific patient
patient = dataset.get_patient("...")
print(patient)

# Accessing exams of a specific type (CT or MRI) for a patient
if patient:
    ct_exams = patient.get_exams_by_type("CT")
    if ct_exams:
        # Get the whole exam or a specific slice (loads the file on demand)
        full_exam = ct_exams[0].get_slice()  # Full exam
        slice_exam = ct_exams[0].get_slice(10)  # Specific slice, if multi-slice
        print(full_exam.shape)
        print(slice_exam.shape)
