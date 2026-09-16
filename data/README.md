# Data

This directory is not committed to git (see `.gitignore`) — download the
raw Yelp Open Dataset yourself: https://www.yelp.com/dataset

Expected layout:

```
data/
  raw/
    yelp_academic_dataset_review.json
    yelp_academic_dataset_user.json
  pairs/
    train.jsonl
    val.jsonl
    test.jsonl
```

- `raw/` holds the unmodified Yelp JSON files (line-delimited JSON).
- `pairs/` holds the output of `src/ranking/build_pairs.py`: one
  `{text_a, text_b, label}` record per line, split by `user_id` so no user's
  reviews appear in more than one split.
