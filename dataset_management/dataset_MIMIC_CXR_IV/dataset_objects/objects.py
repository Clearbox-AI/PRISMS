import os
import pandas as pd
from glob import glob


class Dataset:
    def __init__(self, root_dir, lazy_load=True):
        self.root_dir = root_dir
        self.lazy_load = lazy_load
        self.splits = ['train', 'test']

        # Store patient IDs for each split at initialization
        self.patient_ids = {split: self._get_patient_ids(split) for split in self.splits}

        # Lazy load patients or load all data depending on the flag
        if not self.lazy_load:
            self.patients = {split: [Patient(os.path.join(self.root_dir, split, pid), lazy_load=False)
                                     for pid in self.patient_ids[split]]
                             for split in self.splits}
        else:
            self.patients = {split: [] for split in self.splits}
            x =5

    def _get_patient_ids(self, split):
        """Return a list of patient IDs for a specific split."""
        split_dir = os.path.join(self.root_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist.")

        # List all directories representing patients
        patient_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        return patient_dirs

    def get_patient(self, patient_id):
        """Retrieve a specific patient by ID."""
        for split in self.splits:
            # Check if the patient is already loaded in self.patients (for lazy_load=False case)
            if not self.lazy_load:
                if patient_id in [p.patient_id for p in self.patients[split]]:
                    return next(p for p in self.patients[split] if p.patient_id == patient_id)
            # Lazy load behavior: use self.patient_ids to check and load the patient
            elif patient_id in self.patient_ids[split]:
                return Patient(os.path.join(self.root_dir, split, patient_id), lazy_load=True)

        raise ValueError(f"Patient ID {patient_id} not found in any split.")

    def get_patients_by_split(self, split):
        """Retrieve all patients in a specific split."""
        if self.lazy_load:
            # Lazy load: instantiate patients on demand using stored patient IDs
            return [Patient(os.path.join(self.root_dir, split, pid), lazy_load=True)
                    for pid in self.patient_ids[split]]
        # Return already loaded patients for non-lazy mode
        return self.patients[split]

    def get_patients_by_ids(self, ids):
        """Retrieve a list of patients by their IDs."""
        found_patients = []
        for patient_id in ids:
            try:
                found_patients.append(self.get_patient(patient_id))
            except ValueError as e:
                print(e)
        return found_patients


class Patient:
    def __init__(self, patient_dir, lazy_load=True):
        self.patient_dir = patient_dir
        self.lazy_load = lazy_load
        self.patient_id = os.path.basename(patient_dir)
        self._clinical_record = None

        if not lazy_load:
            self._clinical_record = self._load_clinical_record()

    @property
    def clinical_record(self):
        if self.lazy_load and self._clinical_record is None:
            self._clinical_record = self._load_clinical_record()
        return self._clinical_record

    def _load_clinical_record(self):
        """Load the patient's clinical record (diagnoses, events, stays, and episodes)."""
        return ClinicalRecord(self.patient_dir, self.lazy_load)


class ClinicalRecord:
    def __init__(self, patient_dir, lazy_load=True):
        self.patient_dir = patient_dir
        self.lazy_load = lazy_load
        self._diagnoses = None
        self._events = None
        self._stays = None
        self._episodes = None

        if not lazy_load:
            self._diagnoses = self._load_csv('diagnoses.csv')
            self._events = self._load_csv('events.csv')
            self._stays = self._load_csv('stays.csv')
            self._episodes = self._load_episodes()

    @property
    def diagnoses(self):
        if self.lazy_load and self._diagnoses is None:
            self._diagnoses = self._load_csv('diagnoses.csv')
        return self._diagnoses

    @property
    def events(self):
        if self.lazy_load and self._events is None:
            self._events = self._load_csv('events.csv')
        return self._events

    @property
    def stays(self):
        if self.lazy_load and self._stays is None:
            self._stays = self._load_csv('stays.csv')
        return self._stays

    @property
    def episodes(self):
        if self.lazy_load and self._episodes is None:
            self._episodes = self._load_episodes()
        return self._episodes

    def _load_csv(self, filename):
        """Load a CSV file only when accessed (if lazy)."""
        file_path = os.path.join(self.patient_dir, filename)
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return None

    def _load_episodes(self):
        """Load episodes and corresponding timeseries files."""
        episode_files = glob(os.path.join(self.patient_dir, 'episode*.csv'))
        episodes = []
        for episode_file in episode_files:
            episode_num = os.path.splitext(os.path.basename(episode_file))[0].replace('episode', '')
            timeseries_file = os.path.join(self.patient_dir, f'episode{episode_num}_timeseries.csv')
            episodes.append(Episode(episode_file, timeseries_file, self.lazy_load))
        return episodes


class Episode:
    def __init__(self, episode_file, timeseries_file, lazy_load=True):
        self.episode_file = episode_file
        self.timeseries_file = timeseries_file
        self.lazy_load = lazy_load
        self.episode_num = os.path.splitext(os.path.basename(episode_file))[0].replace('episode', '')
        self._episode_data = None
        self._timeseries_data = None

        if not lazy_load:
            self._episode_data = pd.read_csv(self.episode_file)
            self._timeseries_data = pd.read_csv(self.timeseries_file) if os.path.exists(self.timeseries_file) else None

    @property
    def episode_data(self):
        if self.lazy_load and self._episode_data is None:
            self._episode_data = pd.read_csv(self.episode_file)
        return self._episode_data

    @property
    def timeseries_data(self):
        if self.lazy_load and self._timeseries_data is None and os.path.exists(self.timeseries_file):
            self._timeseries_data = pd.read_csv(self.timeseries_file)
        return self._timeseries_data


# Example usage:
dataset = Dataset(r'...', lazy_load=True)
x = 4
train_patients = dataset.get_patients_by_split('test')
specific_patient = dataset.get_patient('10001884')
episode_record = specific_patient.clinical_record
episode_record.diagnoses
