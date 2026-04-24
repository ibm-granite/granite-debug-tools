"""Tests for dashboard helper functions."""

import json
import pathlib

from perfbench.dashboard_helpers import (
    build_comparison_df,
    build_percentile_df,
    fmt,
    guidellm_stat,
    guidellm_strategy_label,
    load_aiperf_runs,
    load_guidellm_runs,
    load_ollama_runs,
    load_vllm_runs,
    metric_val,
)

# ── fmt tests ───────────────────────────────────────────────────────


class TestFmt:
    def test_small_float(self):
        assert fmt(0.001234) == "0.0012"

    def test_very_small_float(self):
        assert fmt(0.0) == "0.0000"

    def test_negative_small_float(self):
        assert fmt(-0.5) == "-0.5000"

    def test_large_float(self):
        assert fmt(1234.5678) == "1,234.57"

    def test_exactly_one(self):
        assert fmt(1.0) == "1.00"

    def test_negative_large_float(self):
        assert fmt(-99.999) == "-100.00"

    def test_integer(self):
        assert fmt(42) == 42

    def test_string(self):
        assert fmt("hello") == "hello"

    def test_none(self):
        assert fmt(None) is None

    def test_dash(self):
        assert fmt("\u2014") == "\u2014"


# ── metric_val tests ────────────────────────────────────────────────


class TestMetricVal:
    def test_flat_value(self):
        assert metric_val({"throughput": 42.5}, "throughput") == 42.5

    def test_missing_key(self):
        assert metric_val({}, "throughput") == "\u2014"

    def test_nested_dict_with_stat(self):
        data = {"ttft": {"avg": 120.5, "p99": 250.0}}
        assert metric_val(data, "ttft", "avg") == 120.5
        assert metric_val(data, "ttft", "p99") == 250.0

    def test_nested_dict_fallback_to_value(self):
        data = {"ttft": {"value": 100.0}}
        assert metric_val(data, "ttft", "avg") == 100.0

    def test_nested_dict_no_stat_no_value(self):
        data = {"ttft": {"other": 999}}
        assert metric_val(data, "ttft", "avg") == "\u2014"

    def test_none_value(self):
        data = {"throughput": None}
        assert metric_val(data, "throughput") == "\u2014"


# ── guidellm_stat tests ────────────────────────────────────────────


class TestGuidellmStat:
    def test_basic_extraction(self):
        metrics = {"requests_per_second": {"successful": {"mean": 12.34, "p50": 11.0}}}
        assert guidellm_stat(metrics, "requests_per_second", "mean") == 12.34
        assert guidellm_stat(metrics, "requests_per_second", "p50") == 11.0

    def test_missing_metric(self):
        assert guidellm_stat({}, "requests_per_second") == "\u2014"

    def test_missing_category(self):
        metrics = {"requests_per_second": {"errored": {"mean": 1.0}}}
        assert (
            guidellm_stat(metrics, "requests_per_second", "mean", "successful")
            == "\u2014"
        )

    def test_missing_stat(self):
        metrics = {"requests_per_second": {"successful": {"p99": 5.0}}}
        assert guidellm_stat(metrics, "requests_per_second", "mean") == "\u2014"

    def test_custom_category(self):
        metrics = {"latency": {"errored": {"mean": 99.9}}}
        assert guidellm_stat(metrics, "latency", "mean", "errored") == 99.9

    def test_percentile_from_nested_dict(self):
        metrics = {
            "ttft": {
                "successful": {
                    "mean": 100.0,
                    "percentiles": {"p50": 95.0, "p90": 150.0, "p99": 200.0},
                }
            }
        }
        assert guidellm_stat(metrics, "ttft", "p50") == 95.0
        assert guidellm_stat(metrics, "ttft", "p90") == 150.0
        assert guidellm_stat(metrics, "ttft", "p99") == 200.0

    def test_top_level_takes_priority_over_percentiles(self):
        metrics = {
            "rps": {
                "successful": {
                    "p50": 10.0,
                    "percentiles": {"p50": 99.0},
                }
            }
        }
        assert guidellm_stat(metrics, "rps", "p50") == 10.0

    def test_percentile_missing_returns_dash(self):
        metrics = {"rps": {"successful": {"mean": 5.0, "percentiles": {}}}}
        assert guidellm_stat(metrics, "rps", "p90") == "\u2014"


# ── guidellm_strategy_label tests ──────────────────────────────────


class TestGuidellmStrategyLabel:
    def test_with_rate(self):
        bench = {"config": {"strategy": {"type_": "constant", "rate": 5.0}}}
        assert guidellm_strategy_label(bench) == "constant@5.00"

    def test_without_rate(self):
        bench = {"config": {"strategy": {"type_": "synchronous"}}}
        assert guidellm_strategy_label(bench) == "synchronous"

    def test_missing_strategy(self):
        bench = {"config": {}}
        assert guidellm_strategy_label(bench) == "unknown"

    def test_missing_config(self):
        assert guidellm_strategy_label({}) == "unknown"

    def test_rate_zero(self):
        bench = {"config": {"strategy": {"type_": "constant", "rate": 0.0}}}
        assert guidellm_strategy_label(bench) == "constant@0.00"


# ── load_vllm_runs tests ──────────────────────────────────────────────


def _write_json(path: pathlib.Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


class TestLoadVllmRuns:
    def test_loads_runs(self, tmp_path):
        model_dir = tmp_path / "model-a"
        _write_json(
            model_dir / "run1.json",
            {
                "max_concurrency": 1,
                "request_throughput": 10.0,
                "output_throughput": 100.0,
                "mean_ttft_ms": 50.0,
                "median_tpot_ms": 5.0,
            },
        )
        _write_json(
            model_dir / "run2.json",
            {
                "max_concurrency": 10,
                "request_throughput": 45.0,
                "output_throughput": 400.0,
                "mean_ttft_ms": 80.0,
                "median_tpot_ms": 8.0,
            },
        )
        runs = load_vllm_runs(tmp_path)
        assert len(runs) == 2
        assert runs[0]["model"] == "model-a"
        assert runs[0]["concurrency"] == 1
        assert runs[1]["concurrency"] == 10

    def test_missing_dir(self, tmp_path):
        runs = load_vllm_runs(tmp_path / "nonexistent")
        assert runs == []

    def test_empty_dir(self, tmp_path):
        (tmp_path / "model-a").mkdir()
        runs = load_vllm_runs(tmp_path)
        assert runs == []

    def test_default_concurrency(self, tmp_path):
        model_dir = tmp_path / "model-a"
        _write_json(
            model_dir / "run1.json",
            {"request_throughput": 10.0},
        )
        runs = load_vllm_runs(tmp_path)
        assert runs[0]["concurrency"] == 1

    def test_loads_percentile_fields(self, tmp_path):
        model_dir = tmp_path / "model-a"
        _write_json(
            model_dir / "run1.json",
            {
                "request_throughput": 10.0,
                "median_ttft_ms": 45.0,
                "p99_ttft_ms": 120.0,
                "median_tpot_ms": 5.0,
                "p99_tpot_ms": 15.0,
                "median_itl_ms": 3.0,
                "p99_itl_ms": 10.0,
            },
        )
        runs = load_vllm_runs(tmp_path)
        assert runs[0]["ttft_p50"] == 45.0
        assert runs[0]["ttft_p99"] == 120.0
        assert runs[0]["tpot_p50"] == 5.0
        assert runs[0]["tpot_p99"] == 15.0
        assert runs[0]["itl_p50"] == 3.0
        assert runs[0]["itl_p99"] == 10.0


# ── load_aiperf_runs tests ────────────────────────────────────────────


class TestLoadAiperfRuns:
    def test_loads_runs(self, tmp_path):
        run_dir = tmp_path / "model-b" / "20260101120000"
        _write_json(
            run_dir / "profile_export_aiperf.json",
            {
                "request_throughput": {"unit": "req/s", "avg": 12.5},
                "output_token_throughput": {"unit": "tok/s", "avg": 200.0},
                "time_to_first_token": {"unit": "ms", "avg": 50.0},
                "inter_token_latency": {"unit": "ms", "avg": 10.0},
                "input_config": {"loadgen": {"concurrency": 5}},
            },
        )
        runs = load_aiperf_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]["model"] == "model-b"
        assert runs[0]["concurrency"] == 5
        assert runs[0]["request_throughput"] == 12.5

    def test_missing_dir(self, tmp_path):
        assert load_aiperf_runs(tmp_path / "nonexistent") == []

    def test_flat_metric(self, tmp_path):
        run_dir = tmp_path / "model-c" / "ts1"
        _write_json(
            run_dir / "profile_export_aiperf.json",
            {"request_throughput": 99.0, "input_config": {"loadgen": {}}},
        )
        runs = load_aiperf_runs(tmp_path)
        assert runs[0]["request_throughput"] == 99.0
        assert runs[0]["concurrency"] == 1

    def test_loads_percentile_fields(self, tmp_path):
        run_dir = tmp_path / "model-b" / "ts1"
        _write_json(
            run_dir / "profile_export_aiperf.json",
            {
                "time_to_first_token": {
                    "unit": "ms",
                    "avg": 50.0,
                    "p50": 45.0,
                    "p90": 80.0,
                    "p99": 120.0,
                },
                "inter_token_latency": {
                    "unit": "ms",
                    "avg": 10.0,
                    "p50": 8.0,
                    "p90": 15.0,
                    "p99": 25.0,
                },
                "input_config": {"loadgen": {"concurrency": 1}},
            },
        )
        runs = load_aiperf_runs(tmp_path)
        assert runs[0]["ttft_p50"] == 45.0
        assert runs[0]["ttft_p90"] == 80.0
        assert runs[0]["ttft_p99"] == 120.0
        assert runs[0]["itl_p50"] == 8.0
        assert runs[0]["itl_p90"] == 15.0
        assert runs[0]["itl_p99"] == 25.0


# ── load_guidellm_runs tests ─────────────────────────────────────────


class TestLoadGuidellmRuns:
    def test_loads_strategies(self, tmp_path):
        run_dir = tmp_path / "model-d" / "ts1"
        _write_json(
            run_dir / "benchmarks.json",
            {
                "benchmarks": [
                    {
                        "config": {"strategy": {"type_": "synchronous"}},
                        "metrics": {
                            "requests_per_second": {"successful": {"mean": 5.0}},
                            "output_tokens_per_second": {"successful": {"mean": 100.0}},
                            "time_to_first_token_ms": {"successful": {"mean": 30.0}},
                            "inter_token_latency_ms": {"successful": {"mean": 8.0}},
                        },
                    },
                    {
                        "config": {"strategy": {"type_": "constant", "rate": 2.0}},
                        "metrics": {
                            "requests_per_second": {"successful": {"mean": 2.0}},
                            "output_tokens_per_second": {"successful": {"mean": 50.0}},
                            "time_to_first_token_ms": {"successful": {"mean": 40.0}},
                            "inter_token_latency_ms": {"successful": {"mean": 12.0}},
                        },
                    },
                ]
            },
        )
        runs = load_guidellm_runs(tmp_path)
        assert len(runs) == 2
        assert runs[0]["strategy"] == "synchronous"
        assert runs[1]["strategy"] == "constant@2.00"
        assert runs[0]["requests_per_second"] == 5.0

    def test_missing_dir(self, tmp_path):
        assert load_guidellm_runs(tmp_path / "nonexistent") == []

    def test_loads_percentile_fields(self, tmp_path):
        run_dir = tmp_path / "model-d" / "ts1"
        _write_json(
            run_dir / "benchmarks.json",
            {
                "benchmarks": [
                    {
                        "config": {"strategy": {"type_": "synchronous"}},
                        "metrics": {
                            "requests_per_second": {"successful": {"mean": 5.0}},
                            "output_tokens_per_second": {"successful": {"mean": 100.0}},
                            "time_to_first_token_ms": {
                                "successful": {
                                    "mean": 30.0,
                                    "percentiles": {
                                        "p50": 28.0,
                                        "p90": 50.0,
                                        "p99": 80.0,
                                    },
                                }
                            },
                            "inter_token_latency_ms": {
                                "successful": {
                                    "mean": 8.0,
                                    "percentiles": {
                                        "p50": 7.0,
                                        "p90": 12.0,
                                        "p99": 18.0,
                                    },
                                }
                            },
                        },
                    }
                ]
            },
        )
        runs = load_guidellm_runs(tmp_path)
        assert runs[0]["ttft_p50"] == 28.0
        assert runs[0]["ttft_p90"] == 50.0
        assert runs[0]["ttft_p99"] == 80.0
        assert runs[0]["itl_p50"] == 7.0
        assert runs[0]["itl_p90"] == 12.0
        assert runs[0]["itl_p99"] == 18.0


# ── build_comparison_df tests ─────────────────────────────────────────


class TestBuildComparisonDf:
    def test_basic_dataframe(self):
        runs = [
            {"label": "run1", "model": "m1", "concurrency": 1, "throughput": 10.0},
            {"label": "run2", "model": "m1", "concurrency": 10, "throughput": 45.0},
        ]
        df = build_comparison_df(runs, [("throughput", "Req/s")])
        assert len(df) == 2
        assert list(df.columns) == ["Run", "Model", "Metric", "Value", "Concurrency"]
        assert df["Value"].tolist() == [10.0, 45.0]

    def test_skips_none_and_dash(self):
        runs = [
            {"label": "r1", "model": "m1", "throughput": None},
            {"label": "r2", "model": "m1", "throughput": "\u2014"},
            {"label": "r3", "model": "m1", "throughput": 5.0},
        ]
        df = build_comparison_df(runs, [("throughput", "Req/s")])
        assert len(df) == 1
        assert df.iloc[0]["Value"] == 5.0

    def test_includes_strategy(self):
        runs = [
            {"label": "r1", "model": "m1", "strategy": "sync", "rps": 10.0},
        ]
        df = build_comparison_df(runs, [("rps", "Req/s")])
        assert "Strategy" in df.columns
        assert df.iloc[0]["Strategy"] == "sync"

    def test_uses_display_label(self):
        runs = [
            {
                "label": "r1",
                "model": "m1",
                "display_label": "m1 / r1",
                "throughput": 10.0,
            },
        ]
        df = build_comparison_df(runs, [("throughput", "Req/s")])
        assert df.iloc[0]["Run"] == "m1 / r1"

    def test_empty_runs(self):
        df = build_comparison_df([], [("throughput", "Req/s")])
        assert df.empty


# ── build_percentile_df tests ────────────────────────────────────────


class TestBuildPercentileDf:
    def test_basic(self):
        runs = [
            {"label": "r1", "model": "m1", "ttft_p50": 45.0, "ttft_p99": 120.0},
            {"label": "r2", "model": "m1", "ttft_p50": 50.0, "ttft_p99": 130.0},
        ]
        df = build_percentile_df(runs, "ttft", ["p50", "p99"])
        assert len(df) == 4
        assert set(df["Percentile"].unique()) == {"P50", "P99"}

    def test_skips_missing(self):
        runs = [
            {"label": "r1", "model": "m1", "ttft_p50": 45.0, "ttft_p99": None},
        ]
        df = build_percentile_df(runs, "ttft", ["p50", "p99"])
        assert len(df) == 1

    def test_uses_display_label(self):
        runs = [
            {
                "label": "r1",
                "model": "m1",
                "display_label": "m1 / r1",
                "ttft_p50": 45.0,
            },
        ]
        df = build_percentile_df(runs, "ttft", ["p50"])
        assert df.iloc[0]["Run"] == "m1 / r1"

    def test_includes_strategy(self):
        runs = [
            {
                "label": "r1",
                "model": "m1",
                "strategy": "sync",
                "ttft_p50": 10.0,
            },
        ]
        df = build_percentile_df(runs, "ttft", ["p50"])
        assert "Strategy" in df.columns
        assert df.iloc[0]["Strategy"] == "sync"

    def test_empty_runs(self):
        df = build_percentile_df([], "ttft", ["p50", "p99"])
        assert df.empty


# ── load_ollama_runs tests ──────────────────────────────────────────


class TestLoadOllamaRuns:
    def test_loads_runs(self, tmp_path):
        model_dir = tmp_path / "llama3.1_8b"
        _write_json(
            model_dir / "20260401120000.json",
            {
                "model": "llama3.1:8b",
                "num_prompts": 5,
                "num_iterations": 3,
                "category": "general",
                "aggregated": {
                    "avg_eval_rate": 42.5,
                    "avg_prompt_eval_rate": 680.2,
                    "avg_total_duration_ms": 5043.5,
                    "avg_load_duration_ms": 5.0,
                },
                "per_prompt": [],
            },
        )
        runs = load_ollama_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]["model"] == "llama3.1_8b"
        assert runs[0]["avg_eval_rate"] == 42.5
        assert runs[0]["avg_prompt_eval_rate"] == 680.2
        assert runs[0]["num_prompts"] == 5
        assert runs[0]["category"] == "general"

    def test_missing_dir(self, tmp_path):
        runs = load_ollama_runs(tmp_path / "nonexistent")
        assert runs == []

    def test_empty_dir(self, tmp_path):
        (tmp_path / "model-a").mkdir()
        runs = load_ollama_runs(tmp_path)
        assert runs == []

    def test_skips_empty_aggregated(self, tmp_path):
        model_dir = tmp_path / "model-a"
        _write_json(
            model_dir / "bad.json",
            {"model": "test", "aggregated": {}},
        )
        runs = load_ollama_runs(tmp_path)
        assert runs == []
