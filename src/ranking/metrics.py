"""Evaluation metrics for the pairwise ranking model."""


def pairwise_accuracy(predictions: "list[int]", labels: "list[int]") -> float:
    """Fraction of pairs where the model picked the higher-starred review."""
    raise NotImplementedError


def kendall_tau_by_user(
    user_ids: "list[str]",
    scores: "list[float]",
    stars: "list[int]",
) -> float:
    """Reconstruct each user's predicted vs. true review ordering and average Kendall's tau across users."""
    raise NotImplementedError
