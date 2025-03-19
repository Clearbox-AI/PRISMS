from enum import Enum

class ScenarioType(Enum):
    UNCOND = "uncond"
    COND_IMAGE = "cond_image"
    COND_TABLE = "cond_table"
    COND_BOTH = "cond_both"

class DataLabel(Enum):
    IMAGE = "image"
    TAB = "tab"
    BOTH = "both"