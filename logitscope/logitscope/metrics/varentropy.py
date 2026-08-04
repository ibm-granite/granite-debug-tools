"""
LogitScope: Varentropy (Variance of Entropy) Metric

Computes the variance of surprisal values across the probability distribution.

"""

import torch

from logitscope.metrics.base import BaseMetrics


class Varentropy(BaseMetrics):
    """
    Varentropy: Variance of the entropy distribution.

    Varentropy measures the variability or spread of surprisal values across
    the probability distribution. It captures the second moment of the surprisal
    distribution, providing insight into how concentrated or dispersed the
    probability mass is.

    Formula: Var[I(X)] = E[I(X)²] - E[I(X)]²
             where I(X) = -log(p(X)) is surprisal

    Interpretation:
        - High varentropy: Wide spread in surprisal values, indicates a probability
          distribution with both very likely and very unlikely tokens
        - Low varentropy: Narrow spread, indicates more uniform certainty across
                          tokens
        - Zero varentropy: All tokens with non-zero probability have identical
                           surprisal

    Use cases:
        - Detecting bimodal or multimodal distributions
        - Identifying when model is uncertain between multiple distinct options
        - Complementary to entropy for understanding prediction characteristics
    """

    def __init__(self, scope):
        """
        Initialize Varentropy metric.

        Args:
            scope: Results object containing probabilities and log probabilities
        """
        super().__init__(scope)

    def compute(self) -> list[float]:
        """
        Compute varentropy for each token position.

        Returns:
            List of varentropy values, one per token position
        """
        log_probs = self.scope.log_probs  # [batch_size, seq_len-1, vocab_size]
        probs = self.scope.probs  # [batch_size, seq_len-1, vocab_size]

        # Compute entropy (first moment): E[I(X)]
        entropy = -torch.sum(probs * log_probs, dim=-1)

        # Compute E[I(X)²]: expected value of squared surprisal
        sum_p_log_sq = torch.sum(probs * (log_probs**2), dim=-1)

        # Varentropy = E[I(X)²] - (E[I(X)])²
        varentropy = sum_p_log_sq - (entropy**2)

        return varentropy.squeeze(0).cpu().tolist()
