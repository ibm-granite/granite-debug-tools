"""Streamlit dashboard for AI benchmark results (vLLM, AIPerf & GuideLLM)."""

import datetime
import json
import pathlib
import time

import altair as alt
import pandas as pd
import streamlit as st

from perfbench.dashboard_helpers import (
    AIPERF_COMPARISON_METRICS,
    GUIDELLM_COMPARISON_METRICS,
    LLAMABENCH_COMPARISON_METRICS,
    OLLAMA_COMPARISON_METRICS,
    VLLM_COMPARISON_METRICS,
    build_comparison_df,
    build_percentile_df,
    load_aiperf_runs,
    load_guidellm_runs,
    load_llamabench_runs,
    load_ollama_runs,
    load_vllm_runs,
    split_pp_tg,
)
from perfbench.dashboard_helpers import (
    fmt as _fmt,
)
from perfbench.dashboard_helpers import (
    guidellm_stat as _guidellm_stat,
)
from perfbench.dashboard_helpers import (
    guidellm_strategy_label as _guidellm_strategy_label,
)
from perfbench.dashboard_helpers import (
    metric_val as _metric_val,
)

ROOT = pathlib.Path(__file__).parent
VLLM_DIR = ROOT / "results_vllm_bench"
AIPERF_DIR = ROOT / "results_aiperf"
GUIDELLM_DIR = ROOT / "results_guidellm"
LLAMABENCH_DIR = ROOT / "results_llama_bench"
OLLAMA_DIR = ROOT / "results_ollama_bench"


# ── Shared comparison renderer ───────────────────────────────────────


def render_comparison_section(
    runs,
    metrics,
    key_prefix,
    *,
    throughput_key,
    latency_key=None,
    available_percentiles=None,
    label_builder=None,
    color_field="Model:N",
    show_concurrency=True,
):
    """Render cross-run comparison charts (scatter, bar, percentiles, line)."""
    build_label = label_builder or (lambda r: f"{r['model']} / {r['label']}")

    run_labels = [build_label(r) for r in runs]
    seen = set()
    unique_labels = [x for x in run_labels if not (x in seen or seen.add(x))]

    selected = st.multiselect(
        "Select runs to compare",
        options=unique_labels,
        default=unique_labels[: min(5, len(unique_labels))],
        key=f"{key_prefix}_compare_select",
    )
    if len(selected) < 2:
        st.info("Select at least 2 runs to compare.")
        return

    # Copy dicts to avoid mutating cached objects
    filtered = [
        {**r, "display_label": build_label(r)}
        for r in runs
        if build_label(r) in selected
    ]

    # ── Throughput vs Latency scatter ────────────────────────────
    if latency_key is not None:
        scatter_rows = []
        for r in filtered:
            tp = r.get(throughput_key)
            lat = r.get(latency_key)
            if tp is None or lat is None or tp == "\u2014" or lat == "\u2014":
                continue
            try:
                scatter_rows.append(
                    {
                        "Run": r["display_label"],
                        "Model": r["model"],
                        "Throughput (Req/s)": float(tp),
                        "TTFT (ms)": float(lat),
                    }
                )
            except (TypeError, ValueError):
                continue
        if scatter_rows:
            st.markdown("#### Throughput vs Latency")
            df_scatter = pd.DataFrame(scatter_rows)
            scatter = (
                alt.Chart(df_scatter)
                .mark_circle(size=100)
                .encode(
                    x=alt.X("Throughput (Req/s):Q"),
                    y=alt.Y("TTFT (ms):Q", title="Mean TTFT (ms)"),
                    color=color_field,
                    tooltip=["Run", "Model", "Throughput (Req/s)", "TTFT (ms)"],
                )
                .properties(height=350)
            )
            st.altair_chart(scatter, width="stretch")
            st.caption("\u2196 Northwest = better (higher throughput, lower latency)")

    # ── Single-metric bar chart ──────────────────────────────────
    df_cmp = build_comparison_df(filtered, metrics)
    if df_cmp.empty:
        return

    metric_labels = [m[1] for m in metrics]
    chosen_metric = st.selectbox(
        "Metric",
        options=metric_labels,
        key=f"{key_prefix}_metric_select",
    )
    df_bar = df_cmp[df_cmp["Metric"] == chosen_metric]

    tooltip_fields = ["Run", "Model", "Value"]
    if "Strategy" in df_bar.columns:
        tooltip_fields = ["Run", "Model", "Strategy", "Value"]

    bar = (
        alt.Chart(df_bar)
        .mark_bar()
        .encode(
            x=alt.X("Run:N", sort=None, title="Run"),
            y=alt.Y("Value:Q", title=chosen_metric),
            color=color_field,
            tooltip=tooltip_fields,
        )
        .properties(height=350)
    )
    st.altair_chart(bar, width="stretch")

    # ── Latency percentile grouped bars ──────────────────────────
    if available_percentiles:
        pct_options = ["TTFT", "ITL"]
        pct_choice = st.radio(
            "Latency metric for percentiles",
            options=pct_options,
            horizontal=True,
            key=f"{key_prefix}_pct_metric",
        )
        df_pct = build_percentile_df(
            filtered,
            pct_choice.lower(),
            available_percentiles,
        )
        if not df_pct.empty:
            st.markdown(f"#### {pct_choice} Latency Percentiles")
            pct_chart = (
                alt.Chart(df_pct)
                .mark_bar()
                .encode(
                    x=alt.X("Run:N", sort=None, title="Run"),
                    y=alt.Y("Value:Q", title=f"{pct_choice} (ms)"),
                    color="Percentile:N",
                    xOffset="Percentile:N",
                    tooltip=["Run", "Model", "Percentile", "Value"],
                )
                .properties(height=350)
            )
            st.altair_chart(pct_chart, width="stretch")
        else:
            st.info("No percentile data available for the selected runs.")

    # ── Concurrency scaling line chart ───────────────────────────
    if show_concurrency and "Concurrency" in df_bar.columns:
        concurrency_vals = df_bar["Concurrency"].nunique()
        if concurrency_vals > 1:
            st.markdown(f"#### {chosen_metric} vs Concurrency")
            line = (
                alt.Chart(df_bar.sort_values("Concurrency"))
                .mark_line(point=True)
                .encode(
                    x=alt.X("Concurrency:Q", title="Concurrency"),
                    y=alt.Y("Value:Q", title=chosen_metric),
                    color=color_field,
                    tooltip=["Model", "Concurrency", "Value"],
                )
                .properties(height=350)
            )
            st.altair_chart(line, width="stretch")


# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Benchmark Results",
    page_icon="⚡",
    layout="wide",
)

# ── Custom styling ───────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    h1 {
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    .stDataFrame { border-radius: 10px; overflow: hidden; }

    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
    }

    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.1);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────
header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown("# ⚡ AI Benchmark Results")
    st.markdown(
        "<p style='color:#888; margin-top:-0.8rem; margin-bottom:0.5rem;'>"
        "Visualise and compare benchmark runs from "
        "<b>vLLM bench</b>, <b>AIPerf</b>, <b>GuideLLM</b>, <b>llama-bench</b>"
        " and <b>Ollama Bench</b>"
        "</p>",
        unsafe_allow_html=True,
    )
    now = datetime.datetime.now().strftime("%H:%M:%S")
    st.caption(f"Last updated: {now}")

with header_right:
    auto_refresh = st.toggle("Auto-refresh", value=False)
    refresh_interval = st.selectbox("Interval (seconds)", [5, 10, 30, 60], index=1)
    if st.button("Refresh now"):
        st.rerun()


# ═════════════════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════════════════
tab_vllm, tab_aiperf, tab_guidellm, tab_llama, tab_ollama = st.tabs(
    ["🚀 vLLM Bench", "📊 AIPerf", "🔬 GuideLLM", "🦙 llama-bench", "🦙 Ollama Bench"]
)

# ── vLLM tab ─────────────────────────────────────────────────────────
VLLM_HIGHLIGHT = [
    ("request_throughput", "Req/s"),
    ("output_throughput", "Tok/s (output)"),
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("median_tpot_ms", "Median TPOT (ms)"),
    ("completed", "Completed"),
    ("failed", "Failed"),
]

with tab_vllm:
    if not VLLM_DIR.exists():
        st.warning(
            f"Results directory **{VLLM_DIR}** does not exist. "
            "Run a vLLM benchmark first to generate results."
        )
    else:
        model_dirs = sorted([d for d in VLLM_DIR.iterdir() if d.is_dir()])
        total_files = len(list(VLLM_DIR.glob("*/*.json")))
        if total_files == 0:
            st.info("No JSON result files found in `results_vllm_bench/<model>/`.")
        else:
            st.markdown(f"**{total_files}** result file(s) found.\n")
            for model_dir in model_dirs:
                json_files = sorted(
                    model_dir.glob("*.json"),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )
                if not json_files:
                    continue

                st.markdown(f"### 🤖 {model_dir.name}")
                for i, fpath in enumerate(json_files):
                    with open(fpath) as f:
                        data = json.load(f)
                    file_label = fpath.stem
                    with st.expander(f"📄  {file_label}", expanded=True):
                        cols = st.columns(len(VLLM_HIGHLIGHT))
                        for col, (key, label) in zip(cols, VLLM_HIGHLIGHT):
                            with col:
                                st.metric(label=label, value=_fmt(data.get(key, "—")))
                        st.markdown(
                            "<div style='height:0.5rem'></div>",
                            unsafe_allow_html=True,
                        )
                        model_name = fpath.parent.name
                        if st.toggle(
                            f"Show all metrics for {model_name}",
                            key=f"vllm_toggle_{file_label}_{model_name}",
                        ):
                            df = pd.DataFrame(
                                [
                                    {"Metric": k, "Value": str(_fmt(v))}
                                    for k, v in data.items()
                                ]
                            )
                            st.dataframe(
                                df,
                                width="stretch",
                                hide_index=True,
                                height=min(40 * len(df), 600),
                            )

            # ── vLLM cross-run comparison ────────────────────────────
            vllm_runs = load_vllm_runs(VLLM_DIR)
            if len(vllm_runs) >= 2:
                st.markdown("---")
                st.markdown("### \U0001f4ca Cross-Run Comparison")
                render_comparison_section(
                    vllm_runs,
                    VLLM_COMPARISON_METRICS,
                    "vllm",
                    throughput_key="request_throughput",
                    latency_key="mean_ttft_ms",

                    available_percentiles=["p50", "p99"],
                )

# ── AIPerf tab ───────────────────────────────────────────────────────
AIPERF_HIGHLIGHT = [
    ("request_throughput", "Req/s", "avg"),
    ("output_token_throughput", "Tok/s (output)", "avg"),
    ("time_to_first_token", "Mean TTFT (ms)", "avg"),
    ("inter_token_latency", "Mean ITL (ms)", "avg"),
    ("request_count", "Requests", "avg"),
    ("request_latency", "p50 Latency (ms)", "p50"),
]

with tab_aiperf:
    if not AIPERF_DIR.exists():
        st.warning(
            f"Results directory **{AIPERF_DIR}** does not exist. "
            "Run an AIPerf benchmark first to generate results."
        )
    else:
        # Each run is a timestamped subdirectory inside a model directory
        # containing profile_export_aiperf.json
        model_dirs = sorted([d for d in AIPERF_DIR.iterdir() if d.is_dir()])
        total_runs = sum(
            1
            for md in model_dirs
            for rd in md.iterdir()
            if rd.is_dir() and (rd / "profile_export_aiperf.json").exists()
        )

        if total_runs == 0:
            st.info(
                "No AIPerf result files found in `results_aiperf/<model>/<timestamp>/`."
            )
        else:
            st.markdown(f"**{total_runs}** run(s) found.\n")
            for model_dir in model_dirs:
                run_dirs = sorted(
                    [d for d in model_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime,
                    reverse=True,
                )
                json_files = [
                    d / "profile_export_aiperf.json"
                    for d in run_dirs
                    if (d / "profile_export_aiperf.json").exists()
                ]
                if not json_files:
                    continue

                st.markdown(f"### 🤖 {model_dir.name}")
                for i, fpath in enumerate(json_files):
                    with open(fpath) as f:
                        data = json.load(f)

                    run_label = fpath.parent.name
                    model = ", ".join(
                        data.get("input_config", {})
                        .get("endpoint", {})
                        .get("model_names", ["—"])
                    )

                    with st.expander(f"📄  {run_label}  —  {model}", expanded=True):
                        cols = st.columns(len(AIPERF_HIGHLIGHT))
                        for col, (key, label, stat) in zip(cols, AIPERF_HIGHLIGHT):
                            with col:
                                st.metric(
                                    label=label,
                                    value=_fmt(_metric_val(data, key, stat)),
                                )

                        st.markdown(
                            "<div style='height:0.5rem'></div>",
                            unsafe_allow_html=True,
                        )

                        model_name = fpath.parent.parent.name
                        if st.toggle(
                            f"Show all metrics for {model_name}",
                            key=f"aiperf_toggle_{run_label}_{model_name}",
                        ):
                            rows = []
                            for k, v in data.items():
                                if isinstance(v, dict) and "unit" in v:
                                    unit = v.get("unit", "")
                                    for stat_name in (
                                        "avg",
                                        "p50",
                                        "p90",
                                        "p99",
                                        "min",
                                        "max",
                                        "std",
                                    ):
                                        if stat_name in v:
                                            rows.append(
                                                {
                                                    "Metric": f"{k} ({stat_name})",
                                                    "Value": _fmt(v[stat_name]),
                                                    "Unit": unit,
                                                }
                                            )
                                elif not isinstance(v, (dict, list)):
                                    rows.append(
                                        {"Metric": k, "Value": str(v), "Unit": ""}
                                    )
                            df = pd.DataFrame(rows)
                            st.dataframe(
                                df,
                                width="stretch",
                                hide_index=True,
                                height=min(40 * len(df), 600),
                            )

            # ── AIPerf cross-run comparison ──────────────────────────
            aiperf_runs = load_aiperf_runs(AIPERF_DIR)
            if len(aiperf_runs) >= 2:
                st.markdown("---")
                st.markdown("### \U0001f4ca Cross-Run Comparison")
                render_comparison_section(
                    aiperf_runs,
                    AIPERF_COMPARISON_METRICS,
                    "aiperf",
                    throughput_key="request_throughput",
                    latency_key="time_to_first_token",

                    available_percentiles=["p50", "p90", "p99"],
                )

# ── GuideLLM tab ─────────────────────────────────────────────────────
GUIDELLM_HIGHLIGHT = [
    ("requests_per_second", "Req/s", "mean"),
    ("output_tokens_per_second", "Tok/s (output)", "mean"),
    ("time_to_first_token_ms", "Mean TTFT (ms)", "mean"),
    ("inter_token_latency_ms", "Mean ITL (ms)", "mean"),
    ("request_latency", "Mean Latency (s)", "mean"),
    ("request_concurrency", "Concurrency", "mean"),
]

GUIDELLM_DETAIL_METRICS = [
    ("requests_per_second", "Req/s"),
    ("output_tokens_per_second", "Output Tok/s"),
    ("tokens_per_second", "Total Tok/s"),
    ("time_to_first_token_ms", "TTFT (ms)"),
    ("time_per_output_token_ms", "TPOT (ms)"),
    ("inter_token_latency_ms", "ITL (ms)"),
    ("request_latency", "Latency (s)"),
    ("request_concurrency", "Concurrency"),
    ("prompt_token_count", "Prompt Tokens"),
    ("output_token_count", "Output Tokens"),
]

with tab_guidellm:
    if not GUIDELLM_DIR.exists():
        st.warning(
            f"Results directory **{GUIDELLM_DIR}** does not exist. "
            "Run a GuideLLM benchmark first to generate results."
        )
    else:
        model_dirs = sorted([d for d in GUIDELLM_DIR.iterdir() if d.is_dir()])
        total_runs = sum(
            1
            for md in model_dirs
            for rd in md.iterdir()
            if rd.is_dir() and (rd / "benchmarks.json").exists()
        )

        if total_runs == 0:
            st.info(
                "No GuideLLM result files found in "
                "`results_guidellm/<model>/<timestamp>/`."
            )
        else:
            st.markdown(f"**{total_runs}** run(s) found.\n")
            for model_dir in model_dirs:
                run_dirs = sorted(
                    [d for d in model_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime,
                    reverse=True,
                )
                json_files = [
                    d / "benchmarks.json"
                    for d in run_dirs
                    if (d / "benchmarks.json").exists()
                ]
                if not json_files:
                    continue

                st.markdown(f"### 🤖 {model_dir.name}")
                for i, fpath in enumerate(json_files):
                    with open(fpath) as f:
                        report = json.load(f)

                    run_label = fpath.parent.name
                    target = report.get("args", {}).get("target", "—")
                    profile = report.get("args", {}).get("profile", "—")
                    benchmarks = report.get("benchmarks", [])

                    with st.expander(
                        f"📄  {run_label}  —  {profile} ({len(benchmarks)} strategies)",
                        expanded=True,
                    ):
                        # Show each benchmark strategy as a row of metrics
                        for i, bench in enumerate(benchmarks):
                            strategy_label = _guidellm_strategy_label(bench)
                            metrics = bench.get("metrics", {})
                            totals = metrics.get("request_totals", {})
                            successful = totals.get("successful", 0)
                            errored = totals.get("errored", 0)

                            st.markdown(
                                f"**{strategy_label}** — "
                                f"✅ {successful} completed, ❌ {errored} errors"
                            )

                            cols = st.columns(len(GUIDELLM_HIGHLIGHT))
                            for col, (key, label, stat) in zip(
                                cols, GUIDELLM_HIGHLIGHT
                            ):
                                with col:
                                    val = _guidellm_stat(metrics, key, stat)
                                    st.metric(label=label, value=_fmt(val))

                            st.markdown(
                                "<div style='height:0.5rem'></div>",
                                unsafe_allow_html=True,
                            )

                        # Detailed comparison table across all strategies
                        model_name = fpath.parent.parent.name
                        if st.toggle(
                            f"Show detailed comparison for {model_name}",
                            key=f"guidellm_toggle_{run_label}_{model_name}",
                        ):
                            rows = []
                            for bench in benchmarks:
                                strategy_label = _guidellm_strategy_label(bench)
                                metrics = bench.get("metrics", {})
                                row = {"Strategy": strategy_label}
                                for key, label in GUIDELLM_DETAIL_METRICS:
                                    for stat in ("mean", "p50", "p90", "p99"):
                                        val = _guidellm_stat(metrics, key, stat)
                                        if val != "—":
                                            row[f"{label} ({stat})"] = _fmt(val)
                                        else:
                                            row[f"{label} ({stat})"] = "—"
                                rows.append(row)
                            df = pd.DataFrame(rows)
                            st.dataframe(
                                df,
                                width="stretch",
                                hide_index=True,
                                height=min(40 * len(df), 600),
                            )

            # ── GuideLLM cross-run comparison ────────────────────────
            guidellm_runs = load_guidellm_runs(GUIDELLM_DIR)
            if len(guidellm_runs) >= 2:
                st.markdown("---")
                st.markdown("### \U0001f4ca Cross-Run Comparison")
                render_comparison_section(
                    guidellm_runs,
                    GUIDELLM_COMPARISON_METRICS,
                    "guidellm",
                    throughput_key="requests_per_second",
                    latency_key="time_to_first_token_ms",

                    available_percentiles=["p50", "p90", "p99"],
                    label_builder=lambda r: (
                        f"{r['model']} / {r['label']} / {r['strategy']}"
                    ),
                    color_field="Strategy:N",
                    show_concurrency=False,
                )

# ── llama-bench tab ─────────────────────────────────────────────────

with tab_llama:
    if not LLAMABENCH_DIR.exists():
        st.warning(
            f"Results directory **{LLAMABENCH_DIR}** does not exist. "
            "Run a llama-bench benchmark first to generate results."
        )
    else:
        model_dirs = sorted([d for d in LLAMABENCH_DIR.iterdir() if d.is_dir()])
        total_files = len(list(LLAMABENCH_DIR.glob("*/*.json")))
        if total_files == 0:
            st.info("No JSON result files found in `results_llama_bench/<model>/`.")
        else:
            st.markdown(f"**{total_files}** result file(s) found.\n")
            for model_dir in model_dirs:
                json_files = sorted(
                    model_dir.glob("*.json"),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )
                if not json_files:
                    continue

                st.markdown(f"### {model_dir.name}")
                for fpath in json_files:
                    with open(fpath) as f:
                        data = json.load(f)

                    entries = data if isinstance(data, list) else []
                    if not entries:
                        continue

                    file_label = fpath.stem
                    pp_vals, tg_vals = split_pp_tg(entries)
                    hw = entries[0]

                    with st.expander(f"📄  {file_label}", expanded=True):
                        cols = st.columns(6)
                        with cols[0]:
                            avg_pp = (
                                sum(pp_vals) / len(pp_vals) if pp_vals else None
                            )
                            st.metric(
                                label="Prompt eval (tok/s)",
                                value=_fmt(avg_pp) if avg_pp else "—",
                            )
                        with cols[1]:
                            avg_tg = (
                                sum(tg_vals) / len(tg_vals) if tg_vals else None
                            )
                            st.metric(
                                label="Generation (tok/s)",
                                value=_fmt(avg_tg) if avg_tg else "—",
                            )
                        with cols[2]:
                            st.metric(
                                label="GPU Layers",
                                value=hw.get("n_gpu_layers", "—"),
                            )
                        with cols[3]:
                            st.metric(
                                label="Batch Size",
                                value=hw.get("n_batch", "—"),
                            )
                        with cols[4]:
                            st.metric(
                                label="Model Type",
                                value=hw.get("model_type", "—"),
                            )
                        with cols[5]:
                            model_size = hw.get("model_size")
                            if model_size and isinstance(model_size, (int, float)):
                                size_str = f"{model_size / 1e9:.1f} GB"
                            else:
                                size_str = str(model_size) if model_size else "—"
                            st.metric(label="Model Size", value=size_str)

                        st.markdown(
                            "<div style='height:0.5rem'></div>",
                            unsafe_allow_html=True,
                        )
                        model_name = fpath.parent.name
                        if st.toggle(
                            f"Show all entries for {model_name}",
                            key=f"llama_toggle_{file_label}_{model_name}",
                        ):
                            rows = []
                            for e in entries:
                                test_type = (
                                    "pp"
                                    if e.get("n_prompt", 0) > 0
                                    and e.get("n_gen", 0) == 0
                                    else "tg"
                                )
                                rows.append(
                                    {
                                        "Type": test_type,
                                        "avg_ts": _fmt(e.get("avg_ts")),
                                        "n_prompt": e.get("n_prompt", 0),
                                        "n_gen": e.get("n_gen", 0),
                                        "n_batch": e.get("n_batch"),
                                        "n_threads": e.get("n_threads"),
                                        "n_gpu_layers": e.get("n_gpu_layers"),
                                        "backends": e.get("backends", ""),
                                    }
                                )
                            df = pd.DataFrame(rows)
                            st.dataframe(
                                df,
                                width="stretch",
                                hide_index=True,
                                height=min(40 * len(df), 600),
                            )

            # ── llama-bench cross-run comparison ────────────────────────
            llama_runs = load_llamabench_runs(LLAMABENCH_DIR)
            if len(llama_runs) >= 2:
                st.markdown("---")
                st.markdown("### \U0001f4ca Cross-Run Comparison")
                render_comparison_section(
                    llama_runs,
                    LLAMABENCH_COMPARISON_METRICS,
                    "llamabench",
                    throughput_key="avg_ts_tg",
                    show_concurrency=False,
                )

# ── Ollama Bench tab ──────────────────────────────────────────────

OLLAMA_HIGHLIGHT = [
    ("avg_eval_rate", "Generation (tok/s)"),
    ("avg_prompt_eval_rate", "Prompt eval (tok/s)"),
    ("avg_total_duration_ms", "Avg Duration (ms)"),
    ("avg_load_duration_ms", "Avg Load (ms)"),
    ("total_tokens_generated", "Tokens Generated"),
    ("total_prompt_tokens", "Prompt Tokens"),
]

with tab_ollama:
    if not OLLAMA_DIR.exists():
        st.warning(
            f"Results directory **{OLLAMA_DIR}** does not exist. "
            "Run an Ollama benchmark first to generate results."
        )
    else:
        model_dirs = sorted([d for d in OLLAMA_DIR.iterdir() if d.is_dir()])
        total_files = len(list(OLLAMA_DIR.glob("*/*.json")))
        if total_files == 0:
            st.info("No JSON result files found in `results_ollama_bench/<model>/`.")
        else:
            st.markdown(f"**{total_files}** result file(s) found.\n")
            for model_dir in model_dirs:
                json_files = sorted(
                    model_dir.glob("*.json"),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )
                if not json_files:
                    continue

                st.markdown(f"### 🦙 {model_dir.name}")
                for fpath in json_files:
                    with open(fpath) as f:
                        data = json.load(f)

                    agg = data.get("aggregated", {})
                    if not agg:
                        continue

                    file_label = fpath.stem
                    category = data.get("category", "")
                    num_prompts = data.get("num_prompts", 0)
                    num_iterations = data.get("num_iterations", 0)

                    with st.expander(
                        f"📄  {file_label}  —  {category} "
                        f"({num_prompts} prompts × {num_iterations} iterations)",
                        expanded=True,
                    ):
                        cols = st.columns(len(OLLAMA_HIGHLIGHT))
                        for col, (key, label) in zip(cols, OLLAMA_HIGHLIGHT):
                            with col:
                                st.metric(
                                    label=label,
                                    value=_fmt(agg.get(key, "—")),
                                )

                        st.markdown(
                            "<div style='height:0.5rem'></div>",
                            unsafe_allow_html=True,
                        )

                        model_name = fpath.parent.name
                        if st.toggle(
                            f"Show per-prompt breakdown for {model_name}",
                            key=f"ollama_toggle_{file_label}_{model_name}",
                        ):
                            per_prompt = data.get("per_prompt", [])
                            if per_prompt:
                                rows = []
                                for pp in per_prompt:
                                    rows.append(
                                        {
                                            "Prompt": pp.get("prompt", ""),
                                            "Avg Eval Rate": _fmt(
                                                pp.get("avg_eval_rate")
                                            ),
                                            "Avg Prompt Eval Rate": _fmt(
                                                pp.get("avg_prompt_eval_rate")
                                            ),
                                            "Iterations": len(
                                                pp.get("iterations", [])
                                            ),
                                        }
                                    )
                                df = pd.DataFrame(rows)
                                st.dataframe(
                                    df,
                                    width="stretch",
                                    hide_index=True,
                                    height=min(40 * len(df), 600),
                                )
                            else:
                                st.info("No per-prompt data available.")

            # ── Ollama cross-run comparison ────────────────────────
            ollama_runs = load_ollama_runs(OLLAMA_DIR)
            if len(ollama_runs) >= 2:
                st.markdown("---")
                st.markdown("### \U0001f4ca Cross-Run Comparison")
                render_comparison_section(
                    ollama_runs,
                    OLLAMA_COMPARISON_METRICS,
                    "ollama",
                    throughput_key="avg_eval_rate",
                    show_concurrency=False,
                )

# ── Auto-refresh ────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
