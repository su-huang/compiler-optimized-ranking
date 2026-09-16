"""Model loading for the pairwise ranking cross-encoder."""

from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_model_and_tokenizer(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
) -> "tuple[PreTrainedModel, PreTrainedTokenizerBase]":
    """Load AutoModelForSequenceClassification + AutoTokenizer for the cross-encoder ranker."""
    raise NotImplementedError
