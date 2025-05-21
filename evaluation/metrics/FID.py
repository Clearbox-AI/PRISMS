from evaluation.metrics.base_metric import BaseMetric
from enums.metrics import MetricType, MetricRequirementType
from typing import Dict, List, Any

class FidMetric(BaseMetric):
    """
    Class for FID.
    """
    @property
    def metric_type(self) -> MetricType:
        return MetricType.FID

    @property
    def dag(self) -> Dict[MetricRequirementType, List[MetricRequirementType]]:
        """
        - REAL_DATA has no dependencies
        - GENERATED_DATA depends on REAL_DATA
        """
        return {
            MetricRequirementType.REAL_DATA: [],
            MetricRequirementType.GENERATED_DATA: [MetricRequirementType.REAL_DATA]
        }

    def compute(self, resources: Dict[MetricRequirementType, Any]) -> float:
        """
        Actually compute the FID from the real_data and generated_data in resources.
        """
        real_data = resources[MetricRequirementType.REAL_DATA]
        generated_data = resources[MetricRequirementType.GENERATED_DATA]
        print("[FidMetric] compute() called with real_data and generated_data.")

        #TODO: FID implementation
        return ...