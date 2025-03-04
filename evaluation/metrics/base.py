from abc import ABC, abstractmethod
from typing import Any, Dict
from evaluation.metrics.artifact_store import ArtifactStore

class BaseMetric(ABC):
    """
    A base interface for all metrics.
    """
    def __init__(self, config: Dict[str, Any], artifact_store: ArtifactStore):
        """
        :param config: Configuration dict for this metric.
        :param artifact_store: A shared artifact store for caching.
        """
        self.config = config
        self.artifact_store = artifact_store

    @abstractmethod
    def prepare(self):
        """
        Prepare data/models for this metric, if needed.
        E.g., load or generate data, train or load a model, etc.
        """

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        """
        Perform the metric calculation and return results as a dict.
        E.g., {"FID": 12.3}.
        """

    def run(self) -> Dict[str, Any]:
        """
        Template method that runs prepare() then compute().
        """
        self.prepare()
        return self.compute()