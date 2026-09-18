"""Evaluation metrics for the pairwise ranking model."""

import math
from collections import defaultdict

from scipy.stats import kendalltau


def pairwise_accuracy(predictions: "list[int]", labels: "list[int]") -> float:
    """Fraction of pairs where the model picked the higher-starred review."""
    correct = sum(int(p == l) for p, l in zip(predictions, labels))
    return correct / len(labels)


def kendall_tau_by_user(
    user_ids: "list[str]",
    scores: "list[float]",
    stars: "list[int]",
) -> float:
    """Reconstruct each user's predicted vs. true review ordering and average Kendall's tau across users.

    scores are per-review model outputs (e.g. from a bi-encoder scoring each review
    independently) and stars are the ground-truth ratings, both aligned to user_ids by
    index. Users with fewer than 2 reviews (no ordering to compare) are skipped.
    """
    reviews_by_user = defaultdict(lambda: ([], []))
    for user_id, score, star in zip(user_ids, scores, stars):
        user_scores, user_stars = reviews_by_user[user_id]
        user_scores.append(score)
        user_stars.append(star)

    taus = []
    for user_scores, user_stars in reviews_by_user.values():
        if len(user_scores) < 2:
            continue
        tau, _ = kendalltau(user_scores, user_stars)
        if tau is not None and not math.isnan(tau):
            taus.append(tau)

    return sum(taus) / len(taus) if taus else 0.0
