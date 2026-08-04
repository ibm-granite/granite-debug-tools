"""
LogitScope API

FastAPI-based REST API and WebSocket server for LogitScope analysis.
Provides HTTP endpoints and real-time WebSocket connections for analyzing
language model predictions.

"""

import json
import math
import os
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from logitscope.logitscope import LogitScope


# Custom JSON encoder for handling NaN/Infinity values
class SafeJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        return super().encode(self._sanitize_for_json(obj))

    def _sanitize_for_json(self, obj):
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj):
                return 0.0
            elif math.isinf(obj):
                return 1000.0 if obj > 0 else -1000.0
            return obj
        else:
            return obj


# NaN and Infinity handling utilities
def safe_float_convert(
    value: float | int | torch.Tensor | Any,
    fallback: float = 0.0,
    metric_name: str = "unknown",
) -> float:
    """
    Safely convert a value to float, handling NaN, Infinity, and tensor types.

    Args:
        value: The value to convert
        fallback: The fallback value to use if conversion fails
        metric_name: Name of the metric for logging purposes

    Returns:
        A safe float value
    """
    try:
        # Handle torch tensors
        if torch.is_tensor(value):
            if value.numel() == 1:
                value = value.item()
            else:
                print(f"Warning: Multi-element tensor for {metric_name}, using mean")
                value = value.mean().item()

        # Convert to float
        if isinstance(value, (int, float)):
            float_value = float(value)
        else:
            try:
                float_value = float(value)
            except (ValueError, TypeError):
                print(
                    f"Warning: Could not convert {type(value)} to float for "
                    f"{metric_name}, using fallback {fallback}"
                )
                return fallback

        # Check for NaN
        if math.isnan(float_value):
            print(f"Warning: NaN detected for {metric_name}, using fallback {fallback}")
            return fallback

        # Check for infinity
        if math.isinf(float_value):
            if float_value > 0:
                # Positive infinity - use reasonable maximums based on metric type
                max_values = {
                    "entropy": 20.0,
                    "varentropy": 10.0,
                    "skewentropy": 10.0,
                    "perplexity": 1000.0,
                    "surprisal": 50.0,
                    "probability": 1.0,
                    "logprob": 0.0,
                }
                max_val = max_values.get(metric_name.lower(), 100.0)
                print(
                    f"Warning: +Infinity detected for {metric_name}, "
                    f"clamping to {max_val}"
                )
                return max_val
            else:
                # Negative infinity
                min_values = {"logprob": -50.0, "log_probability": -50.0}
                min_val = min_values.get(metric_name.lower(), fallback)
                print(
                    f"Warning: -Infinity detected for {metric_name}, "
                    f"clamping to {min_val}"
                )
                return min_val

        # Additional range checks for specific metrics
        if metric_name.lower() == "probability" and (
            float_value < 0 or float_value > 1
        ):
            clamped = max(0.0, min(1.0, float_value))
            print(
                f"Warning: Probability {float_value} out of range [0,1] for "
                f"{metric_name}, clamping to {clamped}"
            )
            return clamped

        if metric_name.lower() == "perplexity" and float_value < 1.0:
            print(
                f"Warning: Perplexity {float_value} < 1.0 for {metric_name}, "
                f"clamping to 1.0"
            )
            return max(1.0, float_value)

        return float_value

    except Exception as e:
        print(
            f"Error in safe_float_convert for {metric_name}: {e}, "
            f"using fallback {fallback}"
        )
        return fallback


def sanitize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """
    Sanitize a dictionary of metrics, ensuring all values are safe floats.

    Args:
        metrics: Dictionary of metric name -> value pairs

    Returns:
        Dictionary with all values converted to safe floats
    """
    sanitized = {}
    for metric_name, value in metrics.items():
        sanitized[metric_name] = safe_float_convert(
            value, fallback=0.0, metric_name=metric_name
        )
    return sanitized


def sanitize_response_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively sanitize response data to handle NaN/Infinity values.

    Args:
        data: Response data dictionary

    Returns:
        Sanitized data dictionary
    """
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if key == "metrics" and isinstance(value, dict):
                sanitized[key] = sanitize_metrics(value)
            elif isinstance(value, (dict, list)):
                sanitized[key] = sanitize_response_data(value)
            elif isinstance(value, (float, int, torch.Tensor)):
                sanitized[key] = safe_float_convert(
                    value, fallback=0.0, metric_name=key
                )
            else:
                sanitized[key] = value
        return sanitized
    elif isinstance(data, list):
        return [sanitize_response_data(item) for item in data]
    elif isinstance(data, (float, int, torch.Tensor)):
        return safe_float_convert(data, fallback=0.0, metric_name="list_item")
    else:
        return data


class AnalysisRequest(BaseModel):
    text: str
    metrics: list[str] | None = ["entropy", "surprisal", "varentropy"]
    top_k: int | None = 5
    generate: bool | None = False


class GenerationRequest(BaseModel):
    prompt: str
    strategy: str = "greedy"
    max_tokens: int | None = 100
    # Universal parameters
    seed: int | None = 42
    # Strategy-specific parameters
    temperature: float | None = 1.0
    top_p: float | None = 0.9
    top_k: int | None = 50
    beam_size: int | None = 5
    length_penalty: float | None = 1.0
    early_stopping: bool | None = True
    typical_p: float | None = 0.9
    mirostat_tau: float | None = 5.0
    mirostat_eta: float | None = 0.1
    mirostat_version: int | None = 2
    contrastive_k: int | None = 4
    contrastive_alpha: float | None = 0.6
    # New decoder parameters
    expert_temperature: float | None = 1.0
    amateur_temperature: float | None = 2.0
    num_beams: int | None = 10
    num_groups: int | None = 5
    diversity_strength: float | None = 0.5
    anti_lm_alpha: float | None = 0.6
    n_gram: int | None = 4
    discrete: bool | None = False
    num_candidates: int | None = 5
    lookahead_steps: int | None = 2
    num_heads: int | None = 2
    microstat_tau: float | None = 5.0
    microstat_eta: float | None = 0.1
    beam_width: int | None = 5


class TokenAnalysis(BaseModel):
    index: int
    token: str
    token_id: int
    metrics: dict[str, float]
    top_k_candidates: list[dict[str, Any]] | None = None


class AnalysisResponse(BaseModel):
    text: str
    tokens: list[TokenAnalysis]
    total_tokens: int
    model_name: str


class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    strategy: str
    total_tokens: int
    generation_time: float


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                self.disconnect(connection)


class LogitScopeAPIHandler:
    def __init__(
        self, model_name: str, tokenizer_name: str | None = None, device: str = "cuda"
    ):
        # Initialize Parameters
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.device = device

        # Application Filepaths
        self.static_dir = os.path.join(os.path.dirname(__file__), "static")

        # Initialize Model and Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                use_cache=False,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(self.device)

            # Initialize LogitScope
            self.entroscope = LogitScope(
                tokenizer=self.tokenizer, model=self.model, device=self.device
            )

            # Initialize generator (for dex integration)
            self.generator = None

            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")
            self.model_loaded = False
            self.model = None
            self.tokenizer = None
            self.entroscope = None

        # WebSocket connection manager
        self.manager = ConnectionManager()

        # Initialize FastAPI app
        self.app = FastAPI(title="LogitScope API", version="0.1.0")

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Mount static files
        self.app.mount("/static", StaticFiles(directory=self.static_dir), name="static")

        # Initialize API routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""
        self.app.add_api_route(path="/", endpoint=self.index, methods=["GET"])
        self.app.add_api_route(path="/analyze", endpoint=self.analyze, methods=["POST"])
        self.app.add_api_route(path="/health", endpoint=self.health, methods=["GET"])
        self.app.add_api_route(
            path="/model/info", endpoint=self.model_info, methods=["GET"]
        )

        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_handler(websocket)

    def index(self):
        """Serve the main HTML page"""
        return FileResponse(os.path.join(self.static_dir, "index.html"))

    def health(self):
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model_loaded": self.model_loaded,
            "model_name": self.model_name if self.model_loaded else None,
            "device": self.device,
        }

    def model_info(self):
        """Get model information"""
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return {
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "device": self.device,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
            "model_type": self.model.config.model_type if self.model else None,
        }

    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Analyze text and return metrics"""
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            # Run entroscope analysis (generate parameter removed in refactoring)
            results = self.entroscope.measure(request.text)

            # Extract token analyses
            token_analyses = []

            for token_info in results.iter_tokens(metrics=request.metrics):
                # Get metrics for this token with NaN protection
                token_metrics = {}
                for metric in request.metrics:
                    if metric in token_info:
                        value = token_info[metric]
                        # Use safe conversion with NaN/Infinity handling
                        token_metrics[metric] = safe_float_convert(
                            value, fallback=0.0, metric_name=metric
                        )

                # Get top-k candidates if requested
                top_k_candidates = None
                if request.top_k > 0 and token_info["index"] > 0:
                    try:
                        top_k_data = results.top_k(token_info["index"], k=request.top_k)
                        top_k_candidates = []
                        for token, prob in top_k_data:
                            # Sanitize probability values
                            safe_prob = safe_float_convert(
                                prob, fallback=0.0, metric_name="probability"
                            )
                            top_k_candidates.append(
                                {"token": token, "probability": safe_prob}
                            )
                    except (IndexError, ValueError) as e:
                        print(
                            f"Warning: Error getting top-k candidates for token "
                            f"{token_info.get('index', 'unknown')}: {e}"
                        )
                        top_k_candidates = None

                token_analysis = TokenAnalysis(
                    index=token_info["index"],
                    token=token_info["token"],
                    token_id=int(token_info["token_id"]),
                    metrics=token_metrics,
                    top_k_candidates=top_k_candidates,
                )
                token_analyses.append(token_analysis)

            # Create response with additional sanitization
            response = AnalysisResponse(
                text=results.text,
                tokens=token_analyses,
                total_tokens=len(token_analyses),
                model_name=self.model_name,
            )

            # Final sanitization pass to catch any remaining NaN/Infinity values
            response_dict = response.dict()
            sanitized_dict = sanitize_response_data(response_dict)

            # Convert back to response object
            return AnalysisResponse(**sanitized_dict)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    async def websocket_handler(self, websocket: WebSocket):
        """Handle WebSocket connections for real-time analysis"""
        await self.manager.connect(websocket)
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "analyze":
                    # Perform real-time analysis
                    try:
                        request = AnalysisRequest(**message.get("data", {}))
                        response = await self.analyze(request)

                        # Include request ID for handling out-of-order responses
                        response_data = response.model_dump()
                        # Extra sanitization for WebSocket responses
                        sanitized_data = sanitize_response_data(response_data)

                        response_msg = {
                            "type": "analysis_result",
                            "data": sanitized_data,
                        }

                        if "requestId" in message:
                            response_msg["requestId"] = message["requestId"]

                        await self.manager.send_personal_message(
                            response_msg, websocket
                        )

                    except Exception as e:
                        await self.manager.send_personal_message(
                            {"type": "error", "message": str(e)}, websocket
                        )

                elif message.get("type") == "ping":
                    # Health check
                    await self.manager.send_personal_message(
                        {"type": "pong", "data": {"status": "connected"}}, websocket
                    )

        except WebSocketDisconnect:
            self.manager.disconnect(websocket)
