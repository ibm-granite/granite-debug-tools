"""
LogitScope: Main module for analyzing language model predictions with token-level
information metrics.

This module provides the primary interface for computing information-theoretic metrics
on language model outputs, enabling debugging and interpretability analysis.
"""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from logitscope.results import Results


class LogitScope:
    """
    Main class for performing token-level analysis of language model predictions.

    LogitScope computes various information-theoretic metrics (entropy, surprisal,
    varentropy, etc.) on model outputs to provide quantitative insights into how
    LLMs process and predict sequences.

    Attributes:
        tokenizer: HuggingFace tokenizer for the model
        model: HuggingFace language model for analysis
        device: Device to run computations on ('cuda' or 'cpu')
        seed: Random seed for reproducibility

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from logitscope import LogitScope
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained('gpt2')
        >>> tokenizer = AutoTokenizer.from_pretrained('gpt2')
        >>> scope = LogitScope(tokenizer, model, device='cpu')
        >>> results = scope.measure('Once upon a time')
        >>> print(results.entropy)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        device: str = "cpu",
        seed: int | None = None,
    ):
        """
        Initialize LogitScope with a model and tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer compatible with the model
            model: HuggingFace causal language model
            device: Device to run computations on ('cuda' or 'cpu')
            seed: Random seed for reproducible results
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.seed = seed

        # Set PRNG seed for reproducibility
        if seed is not None and isinstance(seed, int):
            torch.manual_seed(seed)

        # Set model to evaluation mode
        model.eval()

    def measure(self, input_text: str) -> Results:
        """
        Analyze input text and compute token-level metrics.

        This method tokenizes the input, runs it through the model, and returns
        a Results object containing probabilities, logits, and lazy-computed metrics.

        Args:
            input_text: Text to analyze

        Returns:
            Results object containing tokens, probabilities, and computed metrics

        Example:
            >>> results = scope.measure('The quick brown fox')
            >>> print(results.entropy)
            >>> for token in results.iter_tokens(['entropy', 'surprisal']):
            ...     print(token['token'], token['entropy'])
        """
        # Tokenize input and move to device
        tokenizer_output = self.tokenizer(input_text, return_tensors="pt")
        input_ids = tokenizer_output["input_ids"].to(self.device)

        # Run model inference without gradient computation
        with torch.no_grad():
            outputs = self.model(input_ids)

        return Results(
            text=input_text,
            input_ids=input_ids,
            logits=outputs.logits,
            tokenizer=self.tokenizer,
        )
