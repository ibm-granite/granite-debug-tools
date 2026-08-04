"""
LogitScope: Entropy Metric

Computes Shannon entropy to measure uncertainty in the model's predictions.

"""

import torch

from logitscope.metrics.base import BaseMetrics


class Entropy(BaseMetrics):
    """
    Shannon entropy metric for measuring prediction uncertainty.

    Entropy quantifies the uncertainty or randomness in a probability distribution.
    Higher entropy indicates more uniform probabilities (higher uncertainty),
    while lower entropy indicates peaked distributions (more confident predictions).

    Formula: H(X) = -Σ p(x) * log(p(x))

    Example:
        - High entropy: Model assigns similar probabilities to many tokens
        - Low entropy: Model is confident about specific token(s)
    """

    def __init__(self, scope):
        """
        Initialize Entropy metric.

        Args:
            scope: Results object containing probabilities and log probabilities
        """
        super().__init__(scope)

    def compute(self) -> list[float]:
        """
        Calculate entropy for each token position.

        Entropy is the expected value of surprisal across all possible tokens.
        It measures how uncertain the model is about the next token.

        Returns:
            List of entropy values (in nats), one per token position

        Note:
            Returns values in natural logarithm units (nats). To convert to bits,
            divide by ln(2) ≈ 0.693.
        """
        probs = self.scope.probs  # [batch_size, seq_len-1, vocab_size]
        log_probs = self.scope.log_probs  # [batch_size, seq_len-1, vocab_size]

        # Calculate entropy: -sum(p_i * log(p_i))
        entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch_size, seq_len-1]

        return entropy.squeeze(0).cpu().tolist()
