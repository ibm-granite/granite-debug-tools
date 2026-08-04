"""
LogitScope: Skewentropy (Skewness of Entropy) Metric

Computes the skewness of the surprisal distribution to detect asymmetry.

"""

import torch

from logitscope.metrics.base import BaseMetrics


class Skewentropy(BaseMetrics):
    """
    Skewentropy: Skewness of the surprisal distribution.

    Skewentropy measures the asymmetry of the surprisal distribution, capturing
    the third moment of surprisal. It indicates whether the distribution has a
    longer tail on one side.

    Formula: Skew[I(X)] = E[(I(X) - μ)³] / σ³
             where μ = E[I(X)] (entropy) and σ² = Var[I(X)] (varentropy)

    Interpretation:
        - Positive skewness: Distribution has a long tail toward high surprisal values
          (few very unlikely tokens, most tokens relatively probable)
        - Negative skewness: Distribution has a long tail toward low surprisal values
          (few very likely tokens, most tokens relatively improbable)
        - Zero skewness: Symmetric distribution around the mean

    Use cases:
        - Detecting when model has high confidence in a few tokens (negative skew)
        - Identifying when most tokens are likely but a few are very unlikely
          (positive skew)
        - Understanding the shape and tail behavior of probability distributions
    """

    def __init__(self, scope):
        """
        Initialize Skewentropy metric.

        Args:
            scope: Results object containing probabilities and log probabilities
        """
        super().__init__(scope)

    def compute(self) -> list[float]:
        """
        Compute skewness of surprisal distribution for each token position.

        Returns:
            List of skewness values, one per token position

        Note:
            Includes small epsilon (1e-10) to prevent division by zero for
            degenerate distributions.
        """
        log_probs = self.scope.log_probs  # [batch_size, seq_len-1, vocab_size]
        probs = self.scope.probs  # [batch_size, seq_len-1, vocab_size]

        # Compute surprisal for all tokens: I(X) = -log(p(X))
        surprisal = -log_probs  # [batch_size, seq_len-1, vocab_size]

        # Compute mean surprisal (entropy)
        entropy = torch.sum(
            probs * surprisal, dim=-1, keepdim=True
        )  # [batch_size, seq_len-1, 1]

        # Center surprisal values around the mean
        centered_surprisal = surprisal - entropy  # [batch_size, seq_len-1, vocab_size]

        # Compute third moment: E[(X - μ)³]
        third_moment = torch.sum(
            probs * (centered_surprisal**3), dim=-1
        )  # [batch_size, seq_len-1]

        # Compute variance and standard deviation
        variance = torch.sum(
            probs * (centered_surprisal**2), dim=-1
        )  # [batch_size, seq_len-1]
        sigma = torch.sqrt(variance)

        # Compute standardized skewness: E[(X - μ)³] / σ³
        epsilon = 1e-10  # Small value to prevent division by zero
        skewness = third_moment / (sigma**3 + epsilon)

        return skewness.squeeze(0).cpu().tolist()
