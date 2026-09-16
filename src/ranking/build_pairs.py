"""Build user-grouped preference pairs from the raw Yelp Open Dataset."""

from pathlib import Path
import json
from collections import defaultdict
import random
from itertools import combinations
from sklearn.model_selection import train_test_split

def load_reviews(reviews_path: Path) -> "list[dict]":
    """Load line-delimited review JSON and return one dict per review."""

    reviews = []
    if reviews_path.exists():
        with reviews_path.open("r", encoding="utf-8") as file:
            for line in file:
                review = json.loads(line.strip())
                reviews.append(review)
        
    return reviews


def group_by_user(reviews: "list[dict]", min_reviews: int = 2) -> "dict[str, list[dict]]":
    """Group reviews by user_id, keeping only users with >= min_reviews reviews."""

    user_to_reviews = defaultdict(list)
    for review in reviews:
        user_to_reviews[review["user_id"]].append(review)

    users_to_remove = []
    for user_id, user_reviews in user_to_reviews.items():
        if len(user_reviews) < min_reviews:
            users_to_remove.append(user_id)

    for user_id in users_to_remove:
        del user_to_reviews[user_id]

    return dict(user_to_reviews)


def make_pairs_for_user(user_reviews: "list[dict]", max_pairs_per_user: int = 20) -> "list[dict]":
    """Sample up to max_pairs_per_user (text_a, text_b, label) pairs for one user, skipping ties.
        label: 1 if text_a has a higher star rating than text_b, 0 if text_b has a higher star rating than text_a.
    """

    pairs = []
    for review_a, review_b in combinations(user_reviews, 2): 
        if review_a["stars"] == review_b["stars"]:
            continue

        if review_a["stars"] > review_b["stars"]:
            label = 1
        else:
            label = 0 

        pairs.append({
            "text_a": review_a["text"],
            "text_b": review_b["text"],
            "label": label
        })

    if len(pairs) > max_pairs_per_user:
        pairs = random.sample(pairs, max_pairs_per_user)

    return pairs 

def split_by_user(
    user_to_reviews: "dict[str, list[dict]]",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> "tuple[list[str], list[str], list[str]]":
    """Split user_ids (not pairs) into train/val/test to avoid leakage."""

    items = list(user_to_reviews.items())

    val_test_frac = 1 - train_frac
    train_items, val_test_items = train_test_split(items, test_size=(val_test_frac), random_state=42)

    test_frac = val_test_frac - val_frac
    relative_test_frac = test_frac / val_test_frac
    val_items, test_items = train_test_split(val_test_items, test_size=relative_test_frac, random_state=42)

    return train_items, val_items, test_items


def main(raw_dir: Path, out_dir: Path, max_pairs_per_user: int = 20) -> None:
    """Load raw reviews, build pairs, split by user, and write train/val/test.jsonl to out_dir."""
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError("wire up argparse for raw_dir/out_dir/max_pairs_per_user")
