"""
LogitScope: Results Object

This module contains the Results class which stores model outputs and provides
access to computed metrics through lazy evaluation and caching.

"""

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from logitscope.metrics import METRICS_REGISTRY


class Results:
    """
    Container for model outputs and computed metrics.

    This class stores the results of analyzing text with LogitScope, including
    token IDs, probabilities, and lazily-computed metrics. Metrics are computed
    on-demand when accessed and then cached for efficiency.

    Tokens are decoded on-demand using tokenizer.decode() to ensure proper
    handling of all special characters across different tokenizer types.

    Attributes:
        text: Input text that was analyzed
        input_ids: Token IDs as tensors
        tokenizer: Tokenizer used for analysis
        logits: Model logits [batch_size, seq_len, vocab_size]
        log_probs: Log probabilities for next-token predictions
                   [batch_size, seq_len-1, vocab_size]
        probs: Probabilities for next-token predictions
               [batch_size, seq_len-1, vocab_size]
        metrics: Cache of computed metrics

    Example:
        >>> results = scope.measure('Hello world')
        >>> print(results.entropy)  # Automatically computed and cached
        >>> for token_info in results.iter_tokens(['entropy', 'surprisal']):
        ...     print(token_info['token'], token_info['entropy'])
    """

    def __init__(
        self,
        text: str,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Initialize Results with model outputs and tokenization.

        Args:
            text: Input text that was analyzed
            input_ids: Token IDs as tensors
            logits: Model logits from forward pass
            tokenizer: Tokenizer used for analysis
        """
        self.text = text
        self.input_ids = input_ids
        self.tokenizer = tokenizer
        self.logits = logits

        # Shift logits to align with next-token prediction
        # shift_logits[i] contains predictions for token at position i+1
        shift_logits = self.logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]

        # Compute log probabilities and probabilities
        self.log_probs = F.log_softmax(shift_logits, dim=-1)
        self.probs = torch.exp(self.log_probs)

        # Initialize metrics cache (metrics are computed lazily on access)
        self.metrics: dict[str, list[float]] = {}

    def __getattr__(self, item: str) -> list[float]:
        """
        Lazily compute and cache metrics when accessed as attributes.

        This allows accessing metrics like `results.entropy` or `results.surprisal`.
        The metric is computed once and then cached for subsequent accesses.

        Args:
            item: Name of the metric to compute

        Returns:
            List of metric values, one per token position

        Raises:
            ValueError: If the requested metric is not in the registry

        Example:
            >>> results = scope.measure('Hello world')
            >>> entropy = results.entropy  # Computed and cached
            >>> entropy_again = results.entropy  # Retrieved from cache
        """
        if item not in METRICS_REGISTRY:
            raise ValueError(f"{item} is not a valid metric.")

        if item not in self.metrics:
            metric = METRICS_REGISTRY[item](self)
            self.metrics[item] = metric.compute()
        return self.metrics[item]

    def iter_tokens(self, metrics: list[str] | None = None) -> Iterator[dict[str, Any]]:
        """
        Iterate over tokens with their associated metrics.

        Yields dictionaries containing token information and requested metric values.
        Note that the first token (index 0) has no metrics since metrics are computed
        for next-token predictions.

        Tokens are automatically decoded using the tokenizer to show actual text
        with proper whitespace (spaces, newlines, tabs) instead of special encoding.

        Args:
            metrics: List of metric names to include. If None or empty, only token
                    information is included.

        Yields:
            Dictionary with keys:
                - index: Token position in sequence
                - token: Decoded token string with actual whitespace
                - token_id: Token ID
                - [metric_name]: Metric value (for tokens at index > 0)

        Example:
            >>> for token in results.iter_tokens(['entropy', 'surprisal']):
            ...     if token['index'] > 0:
            ...         print(f"{token['token']}: {token['entropy']:.2f}")
        """
        if metrics is None:
            metrics = []

        seq_length = self.input_ids.shape[1]
        for i in range(seq_length):
            token_id = int(self.input_ids[0][i].item())
            # Use tokenizer.decode() for proper decoding of all special characters
            decoded_token = self.tokenizer.decode([token_id])

            token_info = {"index": i, "token": decoded_token, "token_id": token_id}

            # Add metrics for all tokens except the first
            if i > 0:
                for m in metrics:
                    if m in METRICS_REGISTRY:
                        token_info[m] = self.__getattr__(m)[i - 1]

            yield token_info

    def top_k(self, index: int, k: int = 5) -> list[tuple[str, float]]:
        """
        Get the top-k most probable tokens at a given position.

        Returns the k tokens with highest predicted probability at the specified
        position, along with their probabilities. Tokens are automatically decoded
        using the tokenizer to show actual text with proper whitespace.

        Args:
            index: Token position to query (1-indexed to match token positions)
            k: Number of top tokens to return

        Returns:
            List of (token, probability) tuples, sorted by descending probability.
            Tokens are properly decoded with whitespace (spaces, newlines, tabs).

        Raises:
            IndexError: If index is out of range for the sequence

        Example:
            >>> results = scope.measure('Hello world')
            >>> top_tokens = results.top_k(index=2, k=5)
            >>> for token, prob in top_tokens:
            ...     print(f"{token}: {prob:.4f}")
        """
        # Convert to 0-indexed for internal arrays
        index -= 1
        if index < 0 or index >= self.probs.shape[1]:
            raise IndexError(f"Index {index} is out of range for top-k lookup.")

        probs_at_index = self.probs[0, index]  # Shape: [vocab_size]
        top_probs, top_indices = torch.topk(probs_at_index, k)

        # Decode tokens properly using the tokenizer
        decoded_tokens = [
            str(self.tokenizer.decode([int(token_id)]))
            for token_id in top_indices.tolist()
        ]

        return [(token, float(prob)) for token, prob in zip(decoded_tokens, top_probs)]
