from enum import Enum

class SourceType(Enum):
    NACC = "nacc"   # wraps a NaccDataset
    FOLDER = "folder"  # raw "patient_xxx" folder approach
    LIST = "list"      # in-memory list of items

class DataLabel(Enum):
    IMAGE = "image"
    TAB = "tab"
    BOTH = "both"

class StorageFormat(Enum):
    DATASET = "dataset"     # e.g. a DataBucket with a Dataset
    LIST = "list"           # e.g. a DataBucket with a list of Tensors
    REFERENCE = "reference" # a dictionary with path pointers, etc.