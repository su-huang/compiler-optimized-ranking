"""Profile the cross-encoder's forward pass to find compute bottlenecks (Phase 2).

Usage:
    python -m scripts.profile_model --model-dir checkpoints/distilbert-ranker-50k-1ep
"""

import argparse
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
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--trace-path", type=Path, default=Path("profiling/baseline_trace.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    inputs = tokenizer(
        *SAMPLE_PAIR,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        for _ in range(args.num_warmup):
            model(**inputs)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True) as prof:
        with torch.no_grad():
            for _ in range(args.num_iters):
                model(**inputs)

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

    args.trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.trace_path))
    print(f"\nChrome trace written to {args.trace_path} (view at chrome://tracing)")


if __name__ == "__main__":
    main()
