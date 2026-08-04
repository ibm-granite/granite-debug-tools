"""
LogitScope: Base Metrics

Abstract base class for all metric implementations.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logitscope.results import Results


class BaseMetrics:
    """
    Abstract base class for all metrics.

    All metric implementations inherit from this class and must implement
    the compute() method to calculate their specific metric values.

    Attributes:
        scope: Results object containing model outputs and probabilities
    """

    def __init__(self, scope: "Results"):
        """
        Initialize the metric with a Results object.

        Args:
            scope: Results object containing model outputs to analyze
        """
        self.scope = scope

    def compute(self) -> list[float]:
        """
        Compute the metric values for all token positions.

        This method must be implemented by all subclasses.

        Returns:
            List of metric values, one per token position (length: seq_len - 1)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError
