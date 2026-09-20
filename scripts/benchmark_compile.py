"""Benchmark eager-mode vs. torch.compile inference latency (Phase 3).

Loads the same checkpoint profiled in Phase 2, times a batch of forward passes in
eager mode, then wraps the model with torch.compile and times it again, reporting
the speedup. Uses the same sample pair and device selection as scripts/profile_model.py
so results are comparable to the Phase 2 baseline.

Usage:
    python -m scripts.benchmark_compile --model-dir checkpoints/distilbert-ranker-50k-1ep
"""

import argparse
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SAMPLE_PAIR = (
    "Great place, food was amazing and service was fast. Will definitely come back!",
    "Not impressed. Waited 40 minutes and the order was still wrong when it arrived.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/distilbert-ranker-50k-1ep"))
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("profiling"),
        help="if set, also profiles eager vs. compiled with torch.profiler and writes traces here",
    )
    return parser.parse_args()


def profile_forward(model, inputs, num_warmup: int, num_iters: int, label: str, out_dir: Path) -> None:
    with torch.no_grad():
        for _ in range(num_warmup):
            model(**inputs)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with torch.no_grad():
            for _ in range(num_iters):
                model(**inputs)

    print(f"\n--- {label} op breakdown ---")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"{label}_trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"trace written to {trace_path}")


def load_model_and_inputs(model_dir: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    inputs = tokenizer(
        *SAMPLE_PAIR,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    ).to(device)

    return model, inputs


def benchmark(model, inputs, num_warmup: int, num_iters: int) -> dict:
    with torch.no_grad():
        for _ in range(num_warmup):
            model(**inputs)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_iters):
            start = time.perf_counter()
            model(**inputs)
            latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms.sort()
    return {
        "mean_ms": statistics.mean(latencies_ms),
        "p50_ms": latencies_ms[len(latencies_ms) // 2],
        "p95_ms": latencies_ms[int(len(latencies_ms) * 0.95)],
        "min_ms": latencies_ms[0],
        "max_ms": latencies_ms[-1],
    }


def print_stats(label: str, stats: dict) -> None:
    print(
        f"{label:>10} | mean {stats['mean_ms']:6.2f}ms  p50 {stats['p50_ms']:6.2f}ms  "
        f"p95 {stats['p95_ms']:6.2f}ms  min {stats['min_ms']:6.2f}ms  max {stats['max_ms']:6.2f}ms"
    )


def main() -> None:
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    model, inputs = load_model_and_inputs(args.model_dir, device)

    eager_stats = benchmark(model, inputs, args.num_warmup, args.num_iters)
    print_stats("eager", eager_stats)

    compiled_model = torch.compile(model)
    compiled_stats = benchmark(compiled_model, inputs, args.num_warmup, args.num_iters)
    print_stats("compiled", compiled_stats)

    speedup = eager_stats["mean_ms"] / compiled_stats["mean_ms"]
    print(f"\nspeedup (mean latency): {speedup:.2f}x")

    if args.profile_dir:
        profile_forward(model, inputs, args.num_warmup, args.num_iters, "eager", args.profile_dir)
        profile_forward(compiled_model, inputs, args.num_warmup, args.num_iters, "compiled", args.profile_dir)


if __name__ == "__main__":
    main()
