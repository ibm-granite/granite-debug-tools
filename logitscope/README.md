# LogitScope

LogitScope is a Python framework for analyzing large language models through information metrics computed from token probability distributions. By examining how models distribute probability mass across vocabulary at each position, LogitScope provides objective, quantitative insights into model behavior, uncertainty, and decision-making without relying on semantic interpretation.

## What It Does

LogitScope transforms opaque LLM outputs into interpretable measurements by:

- **Quantifying uncertainty**: Measures how confident or uncertain the model is at each token position
- **Detecting distribution patterns**: Identifies when models are choosing between multiple options vs. making confident predictions
- **Revealing surprising outputs**: Flags tokens that are statistically unexpected given the context
- **Tracking prediction quality**: Monitors cumulative model performance through perplexity and other metrics
- **Exposing alternative paths**: Shows top-k alternative tokens the model considered at each position

This enables debugging hallucinations, optimizing prompts, evaluating fine-tuning results, and understanding model behavior quantitatively.

## Key Features

- 📊 **Information-theoretic metrics:** Entropy, varentropy, skewentropy, surprisal, perplexity, and probability
- 🔌 **Universal compatibility:** Works with any HuggingFace causal language model
- 🚀 **Efficient computation:** Lazy evaluation with intelligent caching
- 💻 **Flexible APIs:** Python library for programmatic access or interactive web UI
- ⚡ **Multi-platform:** CPU, CUDA, and MPS (Apple Silicon) support
- 🔧 **Extensible:** Plugin architecture for custom metrics

## Installation

### Basic Installation (Python API only)

For programmatic use of LogitScope without the web UI:

```bash
git clone https://github.ibm.com/Granite-debug/logitscope.git
cd logitscope
pip install -e .
```

### Full Installation (with Web UI)

To use the interactive web interface:

```bash
pip install -e ".[ui]"
```

### Development Installation

For contributing to LogitScope:

```bash
pip install -e ".[ui,dev]"
```

### Requirements

LogitScope uses Python 3.11+ in addition to the following packages.

**Core dependencies** (always installed):

- PyTorch 2.7+
- Transformers 4.51+
- NumPy 1.17+

**UI dependencies** (optional, installed with `[ui]`):

- FastAPI 0.115+
- Uvicorn 0.34+

**Development dependencies** (optional, installed with `[dev]`):

- Ruff 0.8+

## Quick Start

### Python Library

Use LogitScope programmatically in your code for analysis pipelines, experiments, or integration into existing workflows:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from logitscope import LogitScope

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct')

# Initialize LogitScope
scope = LogitScope(tokenizer, model, device='cpu')

# Analyze text
results = scope.measure('The quick brown fox jumps over the lazy dog')

# Access metrics as arrays
print(f"Surprisal: {results.surprisal}")
print(f"Entropy: {results.entropy}")
print(f"Varentropy: {results.varentropy}")

# Iterate over tokens with metrics
# Use repr() to show escape sequences (\n, \t, etc.) in CLI output
for token in results.iter_tokens(['surprisal', 'entropy', 'perplexity']):
    print(
        f"{repr(token['token']):15s} | "
        f"surprisal={token['surprisal']:.3f} | "
        f"entropy={token['entropy']:.3f} | "
        f"perplexity={token['perplexity']:.3f}"
    )

# Inspect top-k alternatives at specific positions
# repr() makes special characters visible
top_tokens = results.top_k(index=5, k=10)
for token, prob in top_tokens:
    print(f"{repr(token):15s} {prob:.4f}")
```

View the [examples/](examples/) directory for more examples on how to use the framework.

### Interactive Web UI

**Note:** The web UI requires the optional UI dependencies. Install with `pip install -e ".[ui]"` if not already installed.

Launch the web interface for visual exploration and real-time analysis:

```bash
python -m logitscope.ui --model HuggingFaceTB/SmolLM2-135M-Instruct --device cpu
```

Then open `http://0.0.0.0:8000` in your browser.

**UI Features:**

- Color-coded token visualization by metric
- Live statistics with running averages
- Interactive token explorer with top-k alternatives
- Distribution plots and metric toggles
- Real-time WebSocket streaming

**Command line options:**

```bash
# Custom port
python -m logitscope.ui --model MODEL_NAME --port 8080

# GPU acceleration
python -m logitscope.ui --model MODEL_NAME --device cuda

# Apple Silicon
python -m logitscope.ui --model MODEL_NAME --device mps
```

## How It Works

At each token position, language models produce a probability distribution over their entire vocabulary. LogitScope analyzes these distributions to extract meaningful signals about model behavior:

1. **Tokenization**: Input text is converted into tokens using the model's tokenizer
2. **Model inference**: The model generates logits (pre-softmax scores) for next-token predictions
3. **Probability computation**: Logits are transformed into probability distributions via softmax
4. **Metric calculation**: Information-theoretic properties are computed from the distributions
5. **Analysis**: Metrics reveal uncertainty patterns, distribution shapes, and prediction quality

### Metrics Reference

Each metric captures a different aspect of the probability distribution:

| Metric | Formula | What It Reveals | When to Use |
| ------ | ------- | --------------- | ----------- |
| **Surprisal** | `-log p(x)` | How unexpected the chosen token was | Find outputs that are statistically unlikely given context |
| **Entropy** | `-Σ p(x) log p(x)` | Overall uncertainty—how spread out probabilities are | Identify where model lacks confidence; high entropy = uncertain |
| **Varentropy** | `Var(surprisal)` | Distribution spread—variance of surprisal values | Detect when model is torn between multiple distinct options (bimodal) |
| **Skewentropy** | `Skew(surprisal)` | Distribution asymmetry | Understand whether probability mass is concentrated or dispersed |
| **Perplexity** | `exp(avg surprisal)` | Cumulative prediction quality over sequence | Overall model performance metric; lower = better |
| **Probability** | `p(x)` | Direct confidence in chosen token | Simple confidence measure (0 to 1) |
| **Log Probability** | `log p(x)` | Log-space confidence | Numerically stable alternative to probability |

## Use Cases

### Debugging & Quality Assurance

**Hallucination detection**: Regions with high entropy and varentropy often indicate fabricated content. When a model generates facts with high confidence but also high uncertainty in surrounding tokens, it may be hallucinating.

**Production monitoring**: Track perplexity and average entropy over time to detect model degradation or unexpected behavior in deployment.

**Output comparison**: Quantitatively compare multiple model responses to the same prompt—lower perplexity and higher token probabilities indicate more "natural" outputs.

### Model Development

**Fine-tuning evaluation**: Measure before/after metrics on validation sets. Improved models show lower perplexity and higher probability on expected outputs without increased entropy on known-answer questions.

**Prompt engineering**: Observe how different prompt phrasings affect model confidence. Good prompts reduce entropy while maintaining high probability on correct tokens.

**Ablation studies**: When changing model architecture or training procedures, LogitScope provides objective metrics beyond just accuracy to quantify the impact.

### Research & Analysis

**Decision pattern analysis**: Study how models make choices by examining entropy/varentropy patterns across different text types, domains, or generation strategies.

**Uncertainty quantification**: Measure whether model confidence scores correlate with actual correctness—essential for calibration studies.

**Distribution characteristics**: Investigate how probability distributions differ across languages, domains, or model sizes to understand architectural and training effects.

## API Reference

### Core Classes

#### `LogitScope(tokenizer, model, device='cpu', seed=None)`

Main analysis interface that wraps any HuggingFace causal language model.

**Parameters:**

- `tokenizer` (PreTrainedTokenizer) - HuggingFace tokenizer matching the model
- `model` (PreTrainedModel) - HuggingFace causal language model
- `device` (str) - Computation device: `'cpu'`, `'cuda'`, or `'mps'`
- `seed` (int, optional) - Random seed for reproducibility

**Methods:**

- `measure(text: str) -> Results` - Analyzes input text and returns a Results object with computed metrics

#### `Results`

Container for analysis results with lazy-evaluated metrics. Metrics are computed on-demand when accessed and cached for efficiency.

**Metric Properties** (computed lazily):

- `surprisal` - List of surprisal values for actual tokens chosen
- `entropy` - List of entropy values, one per token position
- `varentropy` - List of varentropy values (distribution variance)
- `skewentropy` - List of skewentropy values (distribution asymmetry)
- `perplexity` - List of perplexity values (cumulative quality)

**Methods:**

- `iter_tokens(metrics: list[str]) -> Iterator[dict]` - Iterates over tokens with their text, IDs, and requested metric values
- `top_k(index: int, k: int) -> list[tuple[str, float]]` - Returns the top-k most probable tokens at a given position with their probabilities

**Properties:**

- `text` - Original input text
- `input_ids` - Token IDs as tensors
- `logits` - Raw model logits (pre-softmax scores)
- `probs` - Probability distributions over vocabulary
- `log_probs` - Log probabilities (for numerical stability)
- `tokenizer` - Tokenizer instance used for decoding

**Token Decoding:**

LogitScope automatically decodes tokens using `tokenizer.decode()` to produce human-readable text. This means:

- Tokens from `iter_tokens()` are properly decoded with actual whitespace
- Tokens from `top_k()` are decoded the same way
- Works correctly with any tokenizer (GPT-2, LLaMA, etc.)
- No manual string processing needed in your code or UI

```python
# Tokens are already decoded!
for token in results.iter_tokens():
    print(token['token'])  # Properly decoded by tokenizer
```

## Development

LogitScope uses modern Python tooling for development:

```bash
# Install with UI and dev dependencies
pip install -e ".[ui,dev]"

# Format code
ruff format .

# Lint code
ruff check .

# Auto-fix issues
ruff check --fix .
```

## Citation

```bibtex
@misc{ahmed2026logitscopeframeworkanalyzingllm,
      title={LogitScope: A Framework for Analyzing LLM Uncertainty Through Information Metrics}, 
      author={Farhan Ahmed and Yuya Jeremy Ong and Chad DeLuca},
      year={2026},
      eprint={2603.24929},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.24929}, 
}
```
