"""Verify batched inference gives the same result as running each pair alone (Phase 5).

Padding shorter sequences to match the longest one in a batch changes the raw
input tensors compared to running that sequence alone, so this checks the model's
predicted label/confidence is unaffected (attention masks should make padding a
no-op) rather than assuming it.

Usage:
    python -m scripts.check_batching_correctness --model-dir checkpoints/distilbert-ranker-50k-1ep
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PAIRS = [
    ("Great food and service!", "Terrible, would not recommend."),
    (
        "This place exceeded every expectation, from the ambiance to the incredible flavors in every dish.",
        "Waited an hour, the food was cold, and the staff seemed to not care at all.",
    ),
    ("Decent.", "Fine, nothing special but got the job done for a quick lunch."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/distilbert-ranker-50k-1ep"))
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser.parse_args()


def predict_batch(pairs, model, tokenizer) -> torch.Tensor:
    text_a_list = [p[0] for p in pairs]
    text_b_list = [p[1] for p in pairs]
    inputs = tokenizer(
        text_a_list, text_b_list, truncation=True, max_length=256, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=-1)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    batched_probs = predict_batch(PAIRS, model, tokenizer)

    max_abs_diff = 0.0
    for i, pair in enumerate(PAIRS):
        single_probs = predict_batch([pair], model, tokenizer)[0]
        diff = (batched_probs[i] - single_probs).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)
        print(f"pair {i}: batched={batched_probs[i].tolist()} single={single_probs.tolist()} diff={diff:.2e}")

    print(f"\nmax abs diff across all pairs: {max_abs_diff:.2e} (tolerance {args.tolerance:.2e})")
    if max_abs_diff > args.tolerance:
        raise AssertionError("batched inference diverges from single-item inference beyond tolerance")
    print("OK: batching does not change predictions")


if __name__ == "__main__":
    main()
