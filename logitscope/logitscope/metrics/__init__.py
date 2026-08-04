"""
LogitScope: Metrics Registry

This module provides the registry of available metrics and imports all metric classes.

Available metrics:
    - surprisal: Information content of actual tokens
    - entropy: Shannon entropy measuring prediction uncertainty
    - varentropy: Variance of the surprisal distribution
    - skewentropy: Skewness of the surprisal distribution
    - perplexity: Cumulative measure of model prediction quality

"""

from logitscope.metrics.entropy import Entropy
from logitscope.metrics.perplexity import Perplexity
from logitscope.metrics.skewentropy import Skewentropy
from logitscope.metrics.surprisal import Surprisal
from logitscope.metrics.varentropy import Varentropy

# Map metric names to their implementation classes
METRICS_REGISTRY = {
    "surprisal": Surprisal,
    "entropy": Entropy,
    "varentropy": Varentropy,
    "skewentropy": Skewentropy,
    "perplexity": Perplexity,
}

__all__ = [
    "METRICS_REGISTRY",
    "Surprisal",
    "Entropy",
    "Varentropy",
    "Skewentropy",
    "Perplexity",
]
