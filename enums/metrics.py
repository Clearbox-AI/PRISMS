from enum import Enum

class MetricType(Enum):
    """
    List of available metrics.
    """
    FID = "FID"
    Other = "Other"

class MetricRequirementType(Enum):
    """
    Enumerates possible requirements for a metric.
    Each metric may need:
      - Real dataset
      - Generated dataset
      - Ground truth labels (in metadata)
      - ML model (trained or untrained)
      - etc.
    """
    REAL_DATA = "real_data"
    GENERATED_DATA = "gen_data"
    GROUND_TRUTH_LABELS = "labels"
    ML_MODEL = "ml_model"