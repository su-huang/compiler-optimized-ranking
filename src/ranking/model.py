"""Model loading for the pairwise ranking cross-encoder."""

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_model_and_tokenizer(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
) -> "tuple[PreTrainedModel, PreTrainedTokenizerBase]":
    """Load AutoModelForSequenceClassification + AutoTokenizer for the cross-encoder ranker."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer
