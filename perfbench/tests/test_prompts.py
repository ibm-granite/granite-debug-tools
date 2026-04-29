"""Tests for MCP prompt templates."""

import pytest

from perfbench.prompts import (
    benchmark_summary,
    compare_models,
    full_benchmark_suite,
    hardware_benchmark,
    latency_investigation,
    ollama_benchmark,
    quick_benchmark,
)


def test_benchmark_summary():
    """Test that benchmark_summary interpolates the model name."""
    result = benchmark_summary("granite-4.0-micro")
    assert isinstance(result, str)
    assert "granite-4.0-micro" in result


def test_quick_benchmark():
    """Test quick_benchmark references the correct tools and parameters."""
    result = quick_benchmark(
        model="ibm-granite/granite-4.0-micro",
        base_url="http://localhost:8000",
    )
    assert isinstance(result, str)
    assert "ibm-granite/granite-4.0-micro" in result
    assert "http://localhost:8000" in result
    assert "ping" in result
    assert "run_vllm_benchmark" in result
    assert "check_vllm_benchmark_status" in result
    assert "1)" in result and "4)" in result


def test_full_benchmark_suite():
    """Test full_benchmark_suite references all three runners."""
    result = full_benchmark_suite(
        model="ibm-granite/granite-4.0-micro",
        base_url="http://localhost:8000",
    )
    assert isinstance(result, str)
    assert "ibm-granite/granite-4.0-micro" in result
    assert "http://localhost:8000" in result
    assert "run_vllm_benchmark" in result
    assert "check_vllm_benchmark_status" in result
    assert "run_aiperf_benchmark" in result
    assert "check_aiperf_benchmark_status" in result
    assert "run_guidellm_benchmark" in result
    assert "check_guidellm_benchmark_status" in result
    assert "run_streamlit_dashboard" in result
    assert "compare_results" in result
    assert "1)" in result and "5)" in result


def test_compare_models():
    """Test compare_models references both models and comparison tools."""
    result = compare_models(
        model_a="granite-4.0-micro",
        model_b="granite-4.0-tiny",
        base_url="http://localhost:8000",
    )
    assert isinstance(result, str)
    assert "granite-4.0-micro" in result
    assert "granite-4.0-tiny" in result
    assert "http://localhost:8000" in result
    assert "run_benchmark_preset" in result
    assert "list_results" in result
    assert "compare_results" in result
    assert "check_vllm_benchmark_status" in result
    assert "1)" in result and "5)" in result


def test_latency_investigation():
    """Test latency_investigation references aiperf tools and metrics."""
    result = latency_investigation(
        model="ibm-granite/granite-4.0-micro",
        base_url="http://localhost:8000",
    )
    assert isinstance(result, str)
    assert "ibm-granite/granite-4.0-micro" in result
    assert "http://localhost:8000" in result
    assert "run_aiperf_benchmark" in result
    assert "check_aiperf_benchmark_status" in result
    assert "list_results" in result
    assert "compare_results" in result
    assert "TTFT" in result
    assert "ITL" in result
    assert "1)" in result and "5)" in result


def test_hardware_benchmark():
    """Test hardware_benchmark references llama-bench tools."""
    result = hardware_benchmark(model_path="/models/model.gguf")
    assert isinstance(result, str)
    assert "/models/model.gguf" in result
    assert "run_llama_bench" in result
    assert "check_llama_bench_status" in result
    assert "1)" in result and "4)" in result


def test_ollama_benchmark():
    """Test ollama_benchmark references ollama tools and metrics."""
    result = ollama_benchmark(model="granite3.3:8b")
    assert isinstance(result, str)
    assert "granite3.3:8b" in result
    assert "run_ollama_benchmark" in result
    assert "check_ollama_benchmark_status" in result
    assert "eval rate" in result
    assert "1)" in result and "4)" in result


ALL_PROMPTS = [
    ("benchmark_summary", benchmark_summary, {"model_name": "test-model"}),
    (
        "quick_benchmark",
        quick_benchmark,
        {"model": "test-model", "base_url": "http://localhost:8000"},
    ),
    (
        "full_benchmark_suite",
        full_benchmark_suite,
        {"model": "test-model", "base_url": "http://localhost:8000"},
    ),
    (
        "compare_models",
        compare_models,
        {
            "model_a": "model-a",
            "model_b": "model-b",
            "base_url": "http://localhost:8000",
        },
    ),
    (
        "latency_investigation",
        latency_investigation,
        {"model": "test-model", "base_url": "http://localhost:8000"},
    ),
    (
        "hardware_benchmark",
        hardware_benchmark,
        {"model_path": "/models/model.gguf"},
    ),
    (
        "ollama_benchmark",
        ollama_benchmark,
        {"model": "granite3.3:8b"},
    ),
]


@pytest.mark.parametrize(
    "name,fn,kwargs",
    ALL_PROMPTS,
    ids=[name for name, _, _ in ALL_PROMPTS],
)
def test_all_prompts_return_str(name, fn, kwargs):
    """Every prompt must return a non-empty string."""
    result = fn(**kwargs)
    assert isinstance(result, str)
    assert len(result) > 0
