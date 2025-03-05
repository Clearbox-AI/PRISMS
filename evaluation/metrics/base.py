from abc import ABC, abstractmethod
from typing import Any, Dict

from evaluation.metrics.artifact_store import ArtifactStore

class BaseMetric(ABC):
    """
    A base interface for all metrics.
    """
    def __init__(self, config: Dict[str, Any], artifact_store: ArtifactStore):
        """
        :param config: The configuration dictionary for this metric.
        :param artifact_store: A shared ArtifactStore for caching or reusing data/models.
        """
        self.config = config
        self.artifact_store = artifact_store

    @abstractmethod
    def prepare(self):
        """
        Prepare data/models that will be needed for the metric.
        (e.g., generate or load data, train or load a model, etc.)
        """

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        """
        Perform the actual metric computation.
        Return a dictionary of results, e.g. {"FID": 12.3}.
        """

    def run(self) -> Dict[str, Any]:
        """
        Template method that runs the entire procedure for this metric.
        """
        self.prepare()
        return self.compute()
