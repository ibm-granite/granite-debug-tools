"""
LogitScope: Surprisal Metric

Computes surprisal (information content) for each predicted token.

"""

from logitscope.metrics.base import BaseMetrics


class Surprisal(BaseMetrics):
    """
    Surprisal (information content) metric for individual tokens.

    Surprisal quantifies how unexpected or surprising a token is given the
    model's predictions. It is the negative log probability of the actual token.

    Formula: I(x) = -log(p(x))

    Interpretation:
        - High surprisal: Token was unlikely/unexpected
        - Low surprisal: Token was likely/expected
        - Surprisal of 0: Token had probability 1 (completely expected)

    Example:
        If a token has probability 0.5, surprisal ≈ 0.69 nats
        If a token has probability 0.01, surprisal ≈ 4.61 nats
    """

    def __init__(self, scope):
        """
        Initialize Surprisal metric.

        Args:
            scope: Results object containing log probabilities and input IDs
        """
        super().__init__(scope)

    def compute(self) -> list[float]:
        """
        Compute surprisal for each actual token in the sequence.

        Returns:
            List of surprisal values (in nats), one per token position

        Note:
            This computes surprisal for the actual tokens that appeared, not the
            average surprisal across all possible tokens (which would be entropy).
        """
        log_probs = self.scope.log_probs  # [batch_size, seq_len-1, vocab_size]
        input_ids = self.scope.input_ids  # [batch_size, seq_len]

        # Get the actual next tokens
        nxt_token_ids = input_ids[:, 1:]  # [batch_size, seq_len-1]

        # Extract log probabilities of the actual tokens that appeared
        act_log_probs = log_probs.gather(-1, nxt_token_ids.unsqueeze(-1)).squeeze(-1)

        # Compute negative log probabilities (surprisal)
        surprisal = -act_log_probs  # [batch_size, seq_len-1]

        return surprisal.cpu().tolist()[0]
