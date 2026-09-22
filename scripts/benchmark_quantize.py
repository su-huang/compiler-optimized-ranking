"""IR-level INT8 quantization benchmark for the pairwise ranking cross-encoder (Phase 4).

Uses PyTorch's pt2e quantization flow (torch.export -> prepare_pt2e -> calibrate ->
convert_pt2e) with ArmInductorQuantizer, since this operates on the exported graph
(the same IR used for the TVM benchmark), not a black-box module-swap call.

TVM's Relax IR (used in benchmark_tvm.py) has no quantization passes in this build
(the old relay.quantize API from the original plan doesn't exist in TVM 0.26), so this
uses PyTorch's own graph-level quantization instead -- still IR-level, just a
different IR than TVM's.

Reports:
  - latency: eager fp32 vs. quantized int8
  - model size on disk: fp32 vs. quantized state dict
  - task metric: pairwise_accuracy on real held-out val pairs, fp32 vs. quantized

Known result on this hardware: convert_pt2e produces a "reference" quantized graph
with explicit quantize/dequantize nodes that still compute in fp32 under the hood, so
uncompiled it is *slower* and *larger* than fp32 (extra Q/DQ ops and scale/zero-point
buffers). Wrapping the quantized graph with torch.compile (Inductor) narrows this --
fuses the Q/DQ pattern into real int8 kernels -- but on this Apple Silicon Mac, even
compiled-int8 (~19ms) doesn't beat compiled-fp32 (~17ms): Apple's ARM CPU cores don't
expose the same int8 dot-product acceleration server ARM/x86 chips do, so there's no
real hardware win to unlock here. Included as an honest finding, not hidden.

Usage:
    python -m scripts.benchmark_quantize --model-dir checkpoints/distilbert-ranker-50k-1ep
"""

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import torch
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer.arm_inductor_quantizer import (
    ArmInductorQuantizer,
    get_default_arm_inductor_quantization_config,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.ranking.dataset import load_pairs_jsonl
from src.ranking.metrics import pairwise_accuracy

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
    parser.add_argument("--data-dir", type=Path, default=Path("data/pairs"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-calibration", type=int, default=32)
    parser.add_argument("--num-eval", type=int, default=500)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=50)
    return parser.parse_args()


def tokenize_fixed(tokenizer, text_a: str, text_b: str, max_length: int):
    """Fixed-shape tokenization (padding to max_length) so every input matches the
    shape used for torch.export/calibration -- pt2e-exported graphs are shape-specialized."""
    return tokenizer(
        text_a,
        text_b,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )


def benchmark_latency(model, inputs, num_warmup: int, num_iters: int) -> dict:
    with torch.no_grad():
        for _ in range(num_warmup):
            model(**inputs) if isinstance(inputs, dict) else model(*inputs)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_iters):
            start = time.perf_counter()
            model(**inputs) if isinstance(inputs, dict) else model(*inputs)
            latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms.sort()
    return {
        "mean_ms": statistics.mean(latencies_ms),
        "p50_ms": latencies_ms[len(latencies_ms) // 2],
        "p95_ms": latencies_ms[int(len(latencies_ms) * 0.95)],
    }


def model_size_mb(state_dict) -> float:
    with tempfile.NamedTemporaryFile() as f:
        torch.save(state_dict, f.name)
        return Path(f.name).stat().st_size / (1024 * 1024)


def eval_pairwise_accuracy(predict_fn, pairs, tokenizer, max_length: int) -> float:
    predictions = []
    labels = []
    for pair in pairs:
        inputs = tokenize_fixed(tokenizer, pair["text_a"], pair["text_b"], max_length)
        with torch.no_grad():
            logits = predict_fn(inputs)
        predictions.append(int(torch.argmax(logits, dim=-1).item()))
        labels.append(pair["label"])
    return pairwise_accuracy(predictions, labels)


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    fp32_model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    fp32_model.eval()
    wrapped_fp32 = LogitsOnly(fp32_model)

    example_inputs = tokenize_fixed(tokenizer, *SAMPLE_PAIR, args.max_length)
    example_args = (example_inputs["input_ids"], example_inputs["attention_mask"])

    train_pairs = load_pairs_jsonl(args.data_dir / "train.jsonl")
    val_pairs = load_pairs_jsonl(args.data_dir / "val.jsonl")
    calibration_pairs = train_pairs[: args.num_calibration]
    eval_pairs = val_pairs[: args.num_eval]

    print(f"calibrating on {len(calibration_pairs)} pairs, evaluating on {len(eval_pairs)} pairs\n")

    exported = torch.export.export(wrapped_fp32, example_args)

    quantizer = ArmInductorQuantizer()
    quantizer.set_global(get_default_arm_inductor_quantization_config())
    prepared = prepare_pt2e(exported.module(), quantizer)

    with torch.no_grad():
        for pair in calibration_pairs:
            inputs = tokenize_fixed(tokenizer, pair["text_a"], pair["text_b"], args.max_length)
            prepared(inputs["input_ids"], inputs["attention_mask"])

    quantized_model = convert_pt2e(prepared)

    # --- latency ---
    fp32_latency = benchmark_latency(wrapped_fp32, example_args, args.num_warmup, args.num_iters)
    quant_latency = benchmark_latency(quantized_model, example_args, args.num_warmup, args.num_iters)
    speedup = fp32_latency["mean_ms"] / quant_latency["mean_ms"]

    print("--- latency ---")
    print(f"fp32:      mean {fp32_latency['mean_ms']:6.2f}ms  p50 {fp32_latency['p50_ms']:6.2f}ms  p95 {fp32_latency['p95_ms']:6.2f}ms")
    print(f"int8:      mean {quant_latency['mean_ms']:6.2f}ms  p50 {quant_latency['p50_ms']:6.2f}ms  p95 {quant_latency['p95_ms']:6.2f}ms")
    print(f"speedup:   {speedup:.2f}x\n")

    # --- model size ---
    fp32_size = model_size_mb(fp32_model.state_dict())
    quant_size = model_size_mb(quantized_model.state_dict())
    print("--- model size on disk ---")
    print(f"fp32:      {fp32_size:.2f} MB")
    print(f"int8:      {quant_size:.2f} MB")
    print(f"reduction: {(1 - quant_size / fp32_size) * 100:.1f}%\n")

    # --- task metric ---
    fp32_acc = eval_pairwise_accuracy(
        lambda inputs: wrapped_fp32(inputs["input_ids"], inputs["attention_mask"]),
        eval_pairs, tokenizer, args.max_length,
    )
    quant_acc = eval_pairwise_accuracy(
        lambda inputs: quantized_model(inputs["input_ids"], inputs["attention_mask"]),
        eval_pairs, tokenizer, args.max_length,
    )
    print("--- pairwise_accuracy ---")
    print(f"fp32:      {fp32_acc:.4f}")
    print(f"int8:      {quant_acc:.4f}")
    print(f"delta:     {quant_acc - fp32_acc:+.4f}")


if __name__ == "__main__":
    main()
