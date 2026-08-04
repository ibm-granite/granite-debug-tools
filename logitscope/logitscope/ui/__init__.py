"""
LogitScope UI: Web interface handler for running LogitScope analysis server.

This module provides a web UI handler that launches a FastAPI server for
interactive analysis of language model predictions.

"""

try:
    import uvicorn

    from logitscope.ui.api import LogitScopeAPIHandler

    _UI_AVAILABLE = True
except ImportError as e:
    _UI_AVAILABLE = False
    _IMPORT_ERROR = e


class LogitScopeUIHandler:
    """
    Handler for launching the LogitScope web UI server.

    This class manages the configuration and launching of the web interface,
    which provides interactive analysis of language model outputs through a
    browser-based UI.

    Attributes:
        model_name: HuggingFace model identifier or path
        tokenizer_name: Optional tokenizer identifier (defaults to model_name)
        device: Device to run model on ('cpu', 'cuda', or 'mps')
        port: Port number for the web server
        log_level: Uvicorn logging level

    Example:
        >>> handler = LogitScopeUIHandler(
        ...     model_name='gpt2',
        ...     device='cpu',
        ...     port=8000
        ... )
        >>> handler.run()  # Starts server on http://0.0.0.0:8000

    Raises:
        ImportError: If UI dependencies are not installed
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: str | None = None,
        device: str = "cpu",
        port: int = 8000,
        log_level: str = "info",
    ):
        """
        Initialize the UI handler.

        Args:
            model_name: HuggingFace model identifier or local path
            tokenizer_name: Optional tokenizer identifier (defaults to model_name)
            device: Device to run computations on ('cpu', 'cuda', or 'mps')
            port: Port number for the web server (default: 8000)
            log_level: Uvicorn log level
                       ('critical', 'error', 'warning', 'info', 'debug', 'trace')

        Raises:
            ImportError: If UI dependencies (fastapi, uvicorn) are not installed
        """
        if not _UI_AVAILABLE:
            raise ImportError(
                "LogitScope UI dependencies are not installed. "
                "Install them with: pip install logitscope[ui]"
            ) from _IMPORT_ERROR

        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.device = device
        self.port = port
        self.log_level = log_level

    def run(self):
        """
        Launch the LogitScope web UI server.

        This method initializes the API handler and starts the Uvicorn server,
        making the web interface accessible at http://0.0.0.0:{port}.

        The server will run until interrupted (Ctrl+C).
        """
        # Initialize LogitScope API handler
        api = LogitScopeAPIHandler(self.model_name, self.tokenizer_name, self.device)

        # Launch LogitScope API via Uvicorn
        uvicorn.run(api.app, host="0.0.0.0", port=self.port, log_level=self.log_level)


__all__ = ["LogitScopeUIHandler"]
