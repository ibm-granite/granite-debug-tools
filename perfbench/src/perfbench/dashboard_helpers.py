"""Helper functions for the Streamlit benchmark dashboard."""

import json
import pathlib

import pandas as pd

try:
    import streamlit as st

    _cache = st.cache_data(ttl=60)
except Exception:

    def _cache(f):
        return f


def fmt(v):
    """Format a value for display."""
    if isinstance(v, float):
        return f"{v:,.4f}" if abs(v) < 1 else f"{v:,.2f}"
    return v


def metric_val(data, key, stat="avg"):
    """Extract a scalar value from a flat or nested metric dict."""
    raw = data.get(key)
    if raw is None:
        return "\u2014"
    if isinstance(raw, dict):
        return raw.get(stat, raw.get("value", "\u2014"))
    return raw


def guidellm_stat(metrics, key, stat="mean", category="successful"):
    """Extract a stat from GuideLLM's nested metrics structure."""
    metric = metrics.get(key, {})
    cat = metric.get(category, {})
    val = cat.get(stat)
    if val is not None:
        return val
    val = cat.get("percentiles", {}).get(stat)
    if val is not None:
        return val
    return "\u2014"


def guidellm_strategy_label(benchmark):
    """Build a human-readable label for a GuideLLM benchmark strategy."""
    config = benchmark.get("config", {})
    strategy = config.get("strategy", {})
    stype = strategy.get("type_", "unknown")
    rate = strategy.get("rate")
    if rate is not None:
        return f"{stype}@{rate:.2f}"
    return stype


# ── Data-loading helpers for cross-run comparison ───────────────────

VLLM_COMPARISON_METRICS = [
    ("request_throughput", "Req/s"),
    ("output_throughput", "Tok/s (output)"),
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("median_tpot_ms", "Median TPOT (ms)"),
]

AIPERF_COMPARISON_METRICS = [
    ("request_throughput", "Req/s"),
    ("output_token_throughput", "Tok/s (output)"),
    ("time_to_first_token", "Mean TTFT (ms)"),
    ("inter_token_latency", "Mean ITL (ms)"),
]

GUIDELLM_COMPARISON_METRICS = [
    ("requests_per_second", "Req/s"),
    ("output_tokens_per_second", "Tok/s (output)"),
    ("time_to_first_token_ms", "Mean TTFT (ms)"),
    ("inter_token_latency_ms", "Mean ITL (ms)"),
]

VLLM_LATENCY_PERCENTILES = [
    ("ttft", {"p50": "median_ttft_ms", "p99": "p99_ttft_ms"}),
    ("tpot", {"p50": "median_tpot_ms", "p99": "p99_tpot_ms"}),
    ("itl", {"p50": "median_itl_ms", "p99": "p99_itl_ms"}),
]

AIPERF_LATENCY_PERCENTILES = [
    ("ttft", "time_to_first_token", ["p50", "p90", "p99"]),
    ("itl", "inter_token_latency", ["p50", "p90", "p99"]),
]

GUIDELLM_LATENCY_PERCENTILES = [
    ("ttft", "time_to_first_token_ms", ["p50", "p90", "p99"]),
    ("itl", "inter_token_latency_ms", ["p50", "p90", "p99"]),
]

LLAMABENCH_COMPARISON_METRICS = [
    ("avg_ts_pp", "Prompt eval (tok/s)"),
    ("avg_ts_tg", "Generation (tok/s)"),
]

OLLAMA_COMPARISON_METRICS = [
    ("avg_eval_rate", "Generation (tok/s)"),
    ("avg_prompt_eval_rate", "Prompt eval (tok/s)"),
    ("avg_total_duration_ms", "Avg total duration (ms)"),
]


def split_pp_tg(
    entries: list[dict],
) -> tuple[list[float], list[float]]:
    """Split llama-bench entries into pp and tg avg_ts values."""
    pp = [
        e["avg_ts"]
        for e in entries
        if e.get("n_prompt", 0) > 0 and e.get("n_gen", 0) == 0
    ]
    tg = [e["avg_ts"] for e in entries if e.get("n_gen", 0) > 0]
    return pp, tg


@_cache
def load_vllm_runs(vllm_dir: pathlib.Path) -> list[dict]:
    """Load all vLLM benchmark runs into a flat list of dicts."""
    runs = []
    if not vllm_dir.exists():
        return runs
    for model_dir in sorted(d for d in vllm_dir.iterdir() if d.is_dir()):
        for fpath in sorted(model_dir.glob("*.json")):
            with open(fpath) as f:
                data = json.load(f)
            concurrency = data.get("max_concurrency", 1)
            run = {
                "label": fpath.stem,
                "model": model_dir.name,
                "concurrency": concurrency,
            }
            for key, _ in VLLM_COMPARISON_METRICS:
                run[key] = data.get(key)
            for metric_label, pct_map in VLLM_LATENCY_PERCENTILES:
                for pct_name, json_key in pct_map.items():
                    run[f"{metric_label}_{pct_name}"] = data.get(json_key)
            runs.append(run)
    return runs


@_cache
def load_aiperf_runs(aiperf_dir: pathlib.Path) -> list[dict]:
    """Load all AIPerf benchmark runs into a flat list of dicts."""
    runs = []
    if not aiperf_dir.exists():
        return runs
    for model_dir in sorted(d for d in aiperf_dir.iterdir() if d.is_dir()):
        for run_dir in sorted(d for d in model_dir.iterdir() if d.is_dir()):
            fpath = run_dir / "profile_export_aiperf.json"
            if not fpath.exists():
                continue
            with open(fpath) as f:
                data = json.load(f)
            concurrency = (
                data.get("input_config", {}).get("loadgen", {}).get("concurrency", 1)
            )
            run = {
                "label": run_dir.name,
                "model": model_dir.name,
                "concurrency": concurrency,
            }
            for key, _ in AIPERF_COMPARISON_METRICS:
                raw = data.get(key)
                if isinstance(raw, dict):
                    run[key] = raw.get("avg")
                else:
                    run[key] = raw
            for metric_label, json_key, pcts in AIPERF_LATENCY_PERCENTILES:
                raw = data.get(json_key)
                if isinstance(raw, dict):
                    for pct_name in pcts:
                        run[f"{metric_label}_{pct_name}"] = raw.get(pct_name)
            runs.append(run)
    return runs


@_cache
def load_guidellm_runs(guidellm_dir: pathlib.Path) -> list[dict]:
    """Load all GuideLLM benchmark runs, one row per strategy."""
    runs = []
    if not guidellm_dir.exists():
        return runs
    for model_dir in sorted(d for d in guidellm_dir.iterdir() if d.is_dir()):
        for run_dir in sorted(d for d in model_dir.iterdir() if d.is_dir()):
            fpath = run_dir / "benchmarks.json"
            if not fpath.exists():
                continue
            with open(fpath) as f:
                report = json.load(f)
            benchmarks = report.get("benchmarks", [])
            for bench in benchmarks:
                strategy = guidellm_strategy_label(bench)
                metrics = bench.get("metrics", {})
                run = {
                    "label": run_dir.name,
                    "model": model_dir.name,
                    "strategy": strategy,
                }
                for key, _ in GUIDELLM_COMPARISON_METRICS:
                    run[key] = guidellm_stat(metrics, key, "mean")
                for metric_label, json_key, pcts in GUIDELLM_LATENCY_PERCENTILES:
                    for pct_name in pcts:
                        run[f"{metric_label}_{pct_name}"] = guidellm_stat(
                            metrics, json_key, pct_name
                        )
                runs.append(run)
    return runs


def build_comparison_df(
    runs: list[dict],
    metric_keys: list[tuple[str, str]],
) -> pd.DataFrame:
    """Melt runs into a long-form DataFrame for Altair charts.

    Returns DataFrame with columns: Run, Model, Metric, Value
    (plus Concurrency/Strategy if present in the input).
    """
    rows = []
    for run in runs:
        for key, label in metric_keys:
            val = run.get(key)
            if val is None or val == "\u2014":
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            row = {
                "Run": run.get("display_label", run["label"]),
                "Model": run["model"],
                "Metric": label,
                "Value": val,
            }
            if "concurrency" in run:
                row["Concurrency"] = run["concurrency"]
            if "strategy" in run:
                row["Strategy"] = run["strategy"]
            rows.append(row)
    return pd.DataFrame(rows)


def build_percentile_df(
    runs: list[dict],
    latency_metric: str,
    percentiles: list[str],
) -> pd.DataFrame:
    """Build a long-form DataFrame of latency percentile values per run.

    Returns DataFrame with columns: Run, Model, Percentile, Value
    (plus Strategy if present).
    """
    rows = []
    for run in runs:
        for pct in percentiles:
            col_key = f"{latency_metric}_{pct}"
            val = run.get(col_key)
            if val is None or val == "\u2014":
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            row = {
                "Run": run.get("display_label", run["label"]),
                "Model": run["model"],
                "Percentile": pct.upper(),
                "Value": val,
            }
            if "strategy" in run:
                row["Strategy"] = run["strategy"]
            rows.append(row)
    return pd.DataFrame(rows)


@_cache
def load_llamabench_runs(llamabench_dir: pathlib.Path) -> list[dict]:
    """Load all llama-bench runs into a flat list of dicts."""
    runs = []
    if not llamabench_dir.exists():
        return runs
    for model_dir in sorted(d for d in llamabench_dir.iterdir() if d.is_dir()):
        for fpath in sorted(model_dir.glob("*.json")):
            with open(fpath) as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else []
            if not entries:
                continue
            pp_vals, tg_vals = split_pp_tg(entries)
            hw = entries[0]
            run = {
                "label": fpath.stem,
                "model": model_dir.name,
                "avg_ts_pp": (sum(pp_vals) / len(pp_vals)) if pp_vals else None,
                "avg_ts_tg": (sum(tg_vals) / len(tg_vals)) if tg_vals else None,
                "n_gpu_layers": hw.get("n_gpu_layers"),
                "n_batch": hw.get("n_batch"),
                "n_threads": hw.get("n_threads"),
                "backends": hw.get("backends", ""),
                "model_type": hw.get("model_type", ""),
                "model_size": hw.get("model_size"),
            }
            runs.append(run)
    return runs


@_cache
def load_ollama_runs(ollama_dir: pathlib.Path) -> list[dict]:
    """Load all Ollama benchmark runs into a flat list of dicts."""
    runs = []
    if not ollama_dir.exists():
        return runs
    for model_dir in sorted(d for d in ollama_dir.iterdir() if d.is_dir()):
        for fpath in sorted(model_dir.glob("*.json")):
            with open(fpath) as f:
                data = json.load(f)
            agg = data.get("aggregated", {})
            if not agg:
                continue
            run = {
                "label": fpath.stem,
                "model": model_dir.name,
                "avg_eval_rate": agg.get("avg_eval_rate"),
                "avg_prompt_eval_rate": agg.get("avg_prompt_eval_rate"),
                "avg_total_duration_ms": agg.get("avg_total_duration_ms"),
                "avg_load_duration_ms": agg.get("avg_load_duration_ms"),
                "num_prompts": data.get("num_prompts"),
                "num_iterations": data.get("num_iterations"),
                "category": data.get("category", ""),
            }
            runs.append(run)
    return runs
