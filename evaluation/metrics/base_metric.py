import abc
from typing import Dict, List, Any
from enums.metrics import MetricType, MetricRequirementType

class BaseMetric(abc.ABC):
    """
    Abstract base for all metrics.
    Each subclass must provide:
      - metric_type: MetricType
      - dag: Dict[MetricRequirementType, List[MetricRequirementType]]
      - compute(resources: Dict[MetricRequirementType, Any]) -> float
    """

    @property
    @abc.abstractmethod
    def metric_type(self) -> MetricType:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dag(self) -> Dict[MetricRequirementType, List[MetricRequirementType]]:
        """
        Return a DAG where each key is a requirement, and its list
        of dependencies must be fetched first.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute(self, resources: Dict[MetricRequirementType, Any]) -> float:
        """
        Once the manager has fetched all the resources in the DAG,
        do the final metric calculation.
        """
        raise NotImplementedError