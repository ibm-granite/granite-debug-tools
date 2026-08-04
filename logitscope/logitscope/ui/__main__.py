"""
LogitScope Web UI Entry Point

Command-line interface for launching the LogitScope web UI.
Run this module directly or via: python -m entroscope.ui

"""

import argparse

from logitscope.ui import LogitScopeUIHandler


def main():
    """
    Parse command-line arguments and launch the LogitScope web UI.

    Command-line Arguments:
        --model: HuggingFace model identifier or path (required)
        --tokenizer: Optional tokenizer identifier (defaults to model name)
        --device: Device to run on ('cpu', 'cuda', or 'mps', default: 'cpu')
        --port: Port number for web server (default: 8000)

    Example:
        $ python -m entroscope.ui --model gpt2 --device cpu --port 8000
    """
    parser = argparse.ArgumentParser(
        description="LogitScope - LLM Debugging and Interpretability Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch with GPT-2 on CPU
  python -m entroscope.ui --model gpt2 --device cpu

  # Launch with custom model on GPU
  python -m entroscope.ui --model HuggingFaceTB/SmolLM2-135M-Instruct --device cuda

  # Launch with custom port
  python -m entroscope.ui --model gpt2 --port 8080
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model identifier or path to local model",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        default=None,
        help="Tokenizer identifier (defaults to model name if not specified)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        required=False,
        choices=["cpu", "cuda", "mps"],
        help="Device to run the model on (default: cpu)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        required=False,
        help="Port number for the web server (default: 8000)",
    )

    args = parser.parse_args()

    # Initialize and launch UI handler
    ui_handler = LogitScopeUIHandler(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        device=args.device,
        port=args.port,
    )
    ui_handler.run()


if __name__ == "__main__":
    main()
