"""Benchmark fixed batch-size x window configs vs. the adaptive batcher (Phase 5).

Sweeps a grid of (max_batch_size, fixed_window_ms) configurations under a fixed
concurrent load, measuring throughput and p95 latency for each -- the
latency/throughput Pareto frontier. Then runs the adaptive strategy (a window that
shrinks under queue pressure, grows when idle) across several different
concurrency levels, to check it lands near that frontier without needing to be
manually retuned per load level.

Usage:
    python -m scripts.benchmark_batching --model-dir checkpoints/distilbert-ranker-50k-1ep
"""

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.serving.batching import AdaptiveBatcher

SAMPLE_PAIR = (
    "Great place, food was amazing and service was fast. Will definitely come back!",
    "Not impressed. Waited 40 minutes and the order was still wrong when it arrived.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/distilbert-ranker-50k-1ep"))
    parser.add_argument("--duration-s", type=float, default=1.5)
    parser.add_argument("--warmup-s", type=float, default=0.3)
    parser.add_argument("--fixed-concurrency", type=int, default=30)
    parser.add_argument("--out-path", type=Path, default=Path("profiling/batching_pareto.png"))
    return parser.parse_args()


def make_predict_fn(model, tokenizer, device):
    def predict_fn(pairs: list) -> list:
        text_a_list = [p[0] for p in pairs]
        text_b_list = [p[1] for p in pairs]
        inputs = tokenizer(
            text_a_list, text_b_list, truncation=True, max_length=256, padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        return [None] * len(pairs)  # result value doesn't matter for this benchmark

    return predict_fn


async def run_load(batcher: AdaptiveBatcher, concurrency: int, duration_s: float, warmup_s: float) -> dict:
    batcher.start()
    latencies_ms: list = []
    stop_flag = {"stop": False}

    async def client_loop():
        while not stop_flag["stop"]:
            start = time.perf_counter()
            await batcher.submit(SAMPLE_PAIR)
            latencies_ms.append((time.perf_counter() - start) * 1000)

    tasks = [asyncio.create_task(client_loop()) for _ in range(concurrency)]
    await asyncio.sleep(warmup_s)
    latencies_ms.clear()
    start_time = time.perf_counter()
    await asyncio.sleep(duration_s)
    elapsed = time.perf_counter() - start_time
    stop_flag["stop"] = True
    await asyncio.gather(*tasks, return_exceptions=True)
    await batcher.stop()

    latencies_ms.sort()
    return {
        "throughput_rps": len(latencies_ms) / elapsed,
        "p50_ms": latencies_ms[len(latencies_ms) // 2] if latencies_ms else float("nan"),
        "p95_ms": latencies_ms[int(len(latencies_ms) * 0.95)] if latencies_ms else float("nan"),
    }


async def main_async(args: argparse.Namespace) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()
    predict_fn = make_predict_fn(model, tokenizer, device)

    fixed_configs = [
        (batch_size, window_ms)
        for batch_size in (1, 4, 8, 16)
        for window_ms in (1, 5, 20, 50)
    ]

    fixed_results = []
    for batch_size, window_ms in fixed_configs:
        batcher = AdaptiveBatcher(
            predict_fn=predict_fn, max_batch_size=batch_size, min_window_ms=window_ms, max_window_ms=window_ms
        )
        stats = await run_load(batcher, args.fixed_concurrency, args.duration_s, args.warmup_s)
        fixed_results.append({"batch_size": batch_size, "window_ms": window_ms, **stats})
        print(
            f"fixed  batch={batch_size:2d} window={window_ms:3d}ms  "
            f"throughput={stats['throughput_rps']:6.1f} req/s  p95={stats['p95_ms']:6.1f}ms"
        )

    adaptive_results = []
    for concurrency in (5, 15, 30):
        batcher = AdaptiveBatcher(predict_fn=predict_fn, max_batch_size=16, min_window_ms=2.0, max_window_ms=20.0)
        stats = await run_load(batcher, concurrency, args.duration_s, args.warmup_s)
        adaptive_results.append({"concurrency": concurrency, **stats})
        print(
            f"adaptive concurrency={concurrency:3d}  "
            f"throughput={stats['throughput_rps']:6.1f} req/s  p95={stats['p95_ms']:6.1f}ms"
        )

    plot_pareto(fixed_results, adaptive_results, args.out_path)


def plot_pareto(fixed_results: list, adaptive_results: list, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    fixed_x = [r["throughput_rps"] for r in fixed_results]
    fixed_y = [r["p95_ms"] for r in fixed_results]
    ax.scatter(fixed_x, fixed_y, color="#2a78d6", label="fixed batch/window configs", s=40, zorder=2)

    adaptive_x = [r["throughput_rps"] for r in adaptive_results]
    adaptive_y = [r["p95_ms"] for r in adaptive_results]
    ax.scatter(
        adaptive_x, adaptive_y, color="#eb6834", marker="D", s=70,
        label="adaptive (across load levels)", zorder=3,
    )

    ax.set_xlabel("throughput (req/s)")
    ax.set_ylabel("p95 latency (ms)")
    ax.set_title("Batching: throughput vs. p95 latency")
    ax.legend()
    ax.grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nchart written to {out_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
