"""Tokenization and dataset wrapper for review-pair ranking."""

import json
from pathlib import Path

from transformers import PreTrainedTokenizerBase


def load_pairs_jsonl(path: Path) -> "list[dict]":
    """Load {text_a, text_b, label} records from a pairs jsonl file."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class PairDataset:
    """HF Dataset-compatible wrapper that tokenizes text_a [SEP] text_b for the cross-encoder."""

    def __init__(self, pairs: "list[dict]", tokenizer: PreTrainedTokenizerBase, max_length: int = 256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        """Return tokenized input_ids/attention_mask/label for one pair."""
        record = self.pairs[idx]

        encoded = self.tokenizer(
            record["text_a"],
            record["text_b"],
            truncation=True,
            max_length=self.max_length,
            padding=False,  # deferred to the collator at batch time
        )

        item = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"]

        label = record["label"]
        item["labels"] = float(label) if isinstance(label, float) else int(label)

        return item
