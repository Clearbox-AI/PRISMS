from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseMetric(ABC):
    """
    A base interface for all metrics.
    """
    def __init__(self, config: Dict[str, Any], shared_state: Dict[str, Any]):
        """
        :param config: The configuration dictionary for this metric.
        :param shared_state: A shared dictionary that can be used to
                             cache or store intermediate artifacts
                             needed by multiple metrics.
        """
        self.config = config
        self.shared_state = shared_state

    @abstractmethod
    def prepare(self):
        """
        (Optional) Load or prepare data/models that will be needed
        to compute the metric. This may also look up or populate
        items in self.shared_state to avoid re-computation.
        """

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        """
        Perform the actual metric computation and return the result
        as a dictionary. For example:
        {
          "FID": 12.3
        }
        """

    def run(self) -> Dict[str, Any]:
        """
        Template method that runs the entire procedure for this metric.
        """
        self.prepare()
        return self.compute()