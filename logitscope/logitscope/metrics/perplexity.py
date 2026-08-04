"""
LogitScope: Perplexity Metric

Computes cumulative perplexity as a measure of model performance.

"""

import torch

from logitscope.metrics.base import BaseMetrics


class Perplexity(BaseMetrics):
    """
    Perplexity: Cumulative measure of model prediction quality.

    Perplexity is the exponential of the average surprisal up to each position.
    It represents roughly how many choices the model is "confused" between at
    each step. Lower perplexity indicates better prediction performance.

    Formula: PPL(x₁:ᵢ) = exp(avg(I(x₁), I(x₂), ..., I(xᵢ)))
             where I(x) = -log(p(x)) is surprisal

    Interpretation:
        - Perplexity of 1: Perfect predictions (probability 1 for actual tokens)
        - Perplexity of 10: Equivalent to being uncertain between ~10 equally
                            likely options
        - Perplexity of 100: Equivalent to being uncertain between ~100 equally
                             likely options
        - Perplexity of vocab_size: Equivalent to uniform distribution (worst case)

    Note:
        This implementation computes cumulative perplexity, meaning the perplexity
        at position i reflects the model's average performance from the start of
        the sequence up to position i.

    Use cases:
        - Comparing model performance across different texts
        - Measuring model confidence/quality over the course of generation
        - Benchmarking language models
    """

    def __init__(self, scope):
        """
        Initialize Perplexity metric.

        Args:
            scope: Results object containing log probabilities and input IDs
        """
        super().__init__(scope)

    def compute(self) -> list[float]:
        """
        Compute cumulative perplexity for each token position.

        At each position i, perplexity reflects the average surprisal from
        the beginning of the sequence up to position i.

        Returns:
            List of cumulative perplexity values, one per token position
        """
        log_probs = self.scope.log_probs  # [batch_size, seq_len-1, vocab_size]
        input_ids = self.scope.input_ids  # [batch_size, seq_len]

        # Get the actual next tokens
        nxt_token_ids = input_ids[:, 1:]  # [batch_size, seq_len-1]

        # Extract log probabilities of actual tokens and compute surprisal
        act_log_probs = log_probs.gather(-1, nxt_token_ids.unsqueeze(-1)).squeeze(-1)
        surprisal = -act_log_probs  # [batch_size, seq_len-1]

        # Compute cumulative average surprisal
        cumulative_sum = torch.cumsum(surprisal, dim=1)  # [batch_size, seq_len-1]
        positions = torch.arange(
            1, surprisal.size(1) + 1, device=surprisal.device
        ).unsqueeze(0)
        avg_surprisal = cumulative_sum / positions  # [batch_size, seq_len-1]

        # Compute perplexity as exponential of average surprisal
        perplexity = torch.exp(avg_surprisal)

        return perplexity.squeeze(0).cpu().tolist()
