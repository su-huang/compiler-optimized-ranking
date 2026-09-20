"""Compile the cross-encoder through TVM's Relax IR and benchmark it (Phase 3).

Exports the model with torch.export, lowers it into TVM's Relax IR via the
from_exported_program frontend, runs the default optimization+build pipeline
(fusion, legalization) targeting LLVM/CPU, and times the resulting compiled
module. Verifies numerical correctness against the eager PyTorch output first,
since a compiled model that disagrees with eager mode isn't useful regardless
of speed.

Note: TVM's default_build pipeline uses untuned generic schedules. Unlike
torch.compile (which reuses PyTorch's already-tuned MPS/CPU kernels), getting a
real speedup out of TVM requires its MetaSchedule autotuner searching over
schedule variants for this specific hardware -- that's a separate, much
longer-running step, not something this script does by default.

Usage:
    python -m scripts.benchmark_tvm --model-dir checkpoints/distilbert-ranker-50k-1ep
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SAMPLE_PAIR = (
    "Great place, food was amazing and service was fast. Will definitely come back!",
    "Not impressed. Waited 40 minutes and the order was still wrong when it arrived.",
)


class LogitsOnly(torch.nn.Module):
    """torch.export needs a plain tensor output, not a HF ModelOutput dataclass."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/distilbert-ranker-50k-1ep"))
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    wrapped = LogitsOnly(model)

    inputs = tokenizer(
        *SAMPLE_PAIR,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    )
    example_args = (inputs["input_ids"], inputs["attention_mask"])

    with torch.no_grad():
        eager_output = wrapped(*example_args).numpy()

    exported = torch.export.export(wrapped, example_args)
    ir_module = from_exported_program(exported)

    target = tvm.target.Target("llvm")
    with target:
        ir_module = relax.get_pipeline("default_build")(ir_module)

    executable = relax.build(ir_module, target=target)
    device = tvm.cpu()
    vm = relax.VirtualMachine(executable, device)

    tvm_input_ids = tvm.runtime.tensor(inputs["input_ids"].numpy(), device)
    tvm_attention_mask = tvm.runtime.tensor(inputs["attention_mask"].numpy(), device)

    tvm_output = vm["main"](tvm_input_ids, tvm_attention_mask)[0].numpy()

    max_abs_diff = float(np.max(np.abs(tvm_output - eager_output)))
    print(f"eager output:  {eager_output}")
    print(f"tvm output:    {tvm_output}")
    print(f"max abs diff:  {max_abs_diff:.2e}")
    if max_abs_diff > 1e-3:
        raise RuntimeError("TVM output diverges from eager PyTorch output beyond tolerance")

    for _ in range(args.num_warmup):
        vm["main"](tvm_input_ids, tvm_attention_mask)

    start = time.perf_counter()
    for _ in range(args.num_iters):
        vm["main"](tvm_input_ids, tvm_attention_mask)
    mean_ms = (time.perf_counter() - start) / args.num_iters * 1000

    print(f"\nTVM (llvm/cpu, default_build pipeline) mean latency: {mean_ms:.2f}ms over {args.num_iters} iters")


if __name__ == "__main__":
    main()
