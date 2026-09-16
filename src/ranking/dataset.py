"""Tokenization and dataset wrapper for review-pair ranking."""

from pathlib import Path

from transformers import PreTrainedTokenizerBase


def load_pairs_jsonl(path: Path) -> "list[dict]":
    """Load {text_a, text_b, label} records from a pairs jsonl file."""
    raise NotImplementedError


class PairDataset:
    """HF Dataset-compatible wrapper that tokenizes text_a [SEP] text_b for the cross-encoder."""

    def __init__(self, pairs: "list[dict]", tokenizer: PreTrainedTokenizerBase, max_length: int = 256):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """Return tokenized input_ids/attention_mask/label for one pair."""
        raise NotImplementedError
