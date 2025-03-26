from evaluation.metrics.base_metric import BaseMetric
from enums.metrics import MetricType, MetricRequirementType
from typing import Dict, List, Any

class OtherMetric(BaseMetric):
    """
    Example metric that might depend on real_data, ground_truth_labels, and ml_model.
    """

    @property
    def metric_type(self) -> MetricType:
        return MetricType.Other

    @property
    def dag(self) -> Dict[MetricRequirementType, List[MetricRequirementType]]:
        """
        - REAL_DATA has no dependencies
        - GROUND_TRUTH_LABELS depends on REAL_DATA (if you want that logic)
        - ML_MODEL depends on both REAL_DATA and GROUND_TRUTH_LABELS
        """
        return {
            MetricRequirementType.REAL_DATA: [],
            MetricRequirementType.GROUND_TRUTH_LABELS: [MetricRequirementType.REAL_DATA],
            MetricRequirementType.ML_MODEL: [
                MetricRequirementType.REAL_DATA,
                MetricRequirementType.GROUND_TRUTH_LABELS
            ]
        }

    def compute(self, resources: Dict[MetricRequirementType, Any]) -> float:
        """
        Example: compute 'Other' metric using the ML model, real_data, and ground_truth labels.
        """
        real_data = resources[MetricRequirementType.REAL_DATA]
        ground_truth = resources[MetricRequirementType.GROUND_TRUTH_LABELS]
        model = resources[MetricRequirementType.ML_MODEL]
        print("[OtherMetric] compute() called with real_data, ground_truth, model.")

        return ...
