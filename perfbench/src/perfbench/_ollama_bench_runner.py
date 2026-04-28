"""Subprocess wrapper that benchmarks an Ollama model via its REST API.

Invoked by ``run_ollama_benchmark`` as::

    python -m perfbench._ollama_bench_runner \
        --model llama3.1:8b --base-url http://localhost:11434 \
        --prompts '["Hello"]' --num-iterations 3 --category general

Progress lines are written to **stderr** (visible via ``check_*_status``).
The final aggregated JSON result is written to **stdout** (captured by
``_save_stdout_result``).
"""

import argparse
import json
import math
import sys
import urllib.error
import urllib.request

DEFAULT_PROMPTS = [
    "Write a step-by-step guide on how to bake a chocolate cake from scratch.",
    "Explain the key differences between classical and operant conditioning.",
    "Develop a Python function that solves a sudoku puzzle.",
    "What are the main causes of the American Civil War?",
    "Translate the following into French: AI is transforming industries.",
]


def _post_generate(base_url: str, model: str, prompt: str) -> dict:
    """Call Ollama's /api/generate endpoint and return the JSON response."""
    url = f"{base_url}/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def _safe_rate(count: int, duration_ns: int) -> float:
    """Compute tokens/second from count and duration in nanoseconds."""
    if duration_ns <= 0:
        return 0.0
    return count / (duration_ns / 1e9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--prompts", default="[]")
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--category", default="general")
    args = parser.parse_args()

    if args.num_iterations < 1:
        print("Error: --num-iterations must be at least 1.", file=sys.stderr)
        sys.exit(1)

    prompts = json.loads(args.prompts)
    if not prompts:
        prompts = DEFAULT_PROMPTS

    all_eval_rates: list[float] = []
    all_prompt_eval_rates: list[float] = []
    total_tokens_generated = 0
    total_prompt_tokens = 0
    all_total_durations: list[float] = []
    all_load_durations: list[float] = []
    per_prompt: list[dict] = []

    for pi, prompt in enumerate(prompts, 1):
        prompt_excerpt = prompt[:40] + ("..." if len(prompt) > 40 else "")
        iterations: list[dict] = []
        prompt_eval_rates: list[float] = []
        prompt_gen_rates: list[float] = []

        for it in range(1, args.num_iterations + 1):
            try:
                data = _post_generate(args.base_url, args.model, prompt)
            except urllib.error.URLError as exc:
                print(
                    f"Error: cannot connect to Ollama at {args.base_url}: {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
            except Exception as exc:
                print(f"Error calling Ollama API: {exc}", file=sys.stderr)
                sys.exit(1)

            eval_count = data.get("eval_count", 0)
            eval_duration = data.get("eval_duration", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)
            prompt_eval_duration = data.get("prompt_eval_duration", 0)
            total_duration = data.get("total_duration", 0)
            load_duration = data.get("load_duration", 0)

            eval_rate = _safe_rate(eval_count, eval_duration)
            prompt_eval_rate = _safe_rate(prompt_eval_count, prompt_eval_duration)

            iteration_result = {
                "eval_count": eval_count,
                "eval_duration_ms": eval_duration / 1e6,
                "eval_rate": round(eval_rate, 2),
                "prompt_eval_count": prompt_eval_count,
                "prompt_eval_duration_ms": round(prompt_eval_duration / 1e6, 2),
                "prompt_eval_rate": round(prompt_eval_rate, 2),
                "total_duration_ms": round(total_duration / 1e6, 2),
                "load_duration_ms": round(load_duration / 1e6, 2),
            }
            iterations.append(iteration_result)

            all_eval_rates.append(eval_rate)
            all_prompt_eval_rates.append(prompt_eval_rate)
            total_tokens_generated += eval_count
            total_prompt_tokens += prompt_eval_count
            all_total_durations.append(total_duration / 1e6)
            all_load_durations.append(load_duration / 1e6)
            prompt_eval_rates.append(prompt_eval_rate)
            prompt_gen_rates.append(eval_rate)

            print(
                f"[{pi}/{len(prompts)}] [{prompt_excerpt}] "
                f"iter {it}/{args.num_iterations}: "
                f"eval={eval_rate:.1f} t/s, prompt={prompt_eval_rate:.1f} t/s",
                file=sys.stderr,
            )

        avg_gen = (
            sum(prompt_gen_rates) / len(prompt_gen_rates) if prompt_gen_rates else 0
        )
        avg_pe = (
            sum(prompt_eval_rates) / len(prompt_eval_rates) if prompt_eval_rates else 0
        )
        per_prompt.append(
            {
                "prompt": prompt,
                "iterations": iterations,
                "avg_eval_rate": round(avg_gen, 2),
                "avg_prompt_eval_rate": round(avg_pe, 2),
            }
        )

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def _stddev(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        m = _mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

    result = {
        "model": args.model,
        "ollama_url": args.base_url,
        "category": args.category,
        "num_prompts": len(prompts),
        "num_iterations": args.num_iterations,
        "aggregated": {
            "avg_eval_rate": round(_mean(all_eval_rates), 2),
            "stddev_eval_rate": round(_stddev(all_eval_rates), 2),
            "avg_prompt_eval_rate": round(_mean(all_prompt_eval_rates), 2),
            "stddev_prompt_eval_rate": round(_stddev(all_prompt_eval_rates), 2),
            "avg_total_duration_ms": round(_mean(all_total_durations), 2),
            "avg_load_duration_ms": round(_mean(all_load_durations), 2),
            "total_tokens_generated": total_tokens_generated,
            "total_prompt_tokens": total_prompt_tokens,
        },
        "per_prompt": per_prompt,
    }

    print(json.dumps(result))
    print("Benchmark complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
