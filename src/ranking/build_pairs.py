"""Build user-grouped preference pairs from the raw Yelp Open Dataset, via DuckDB SQL.

Query logic lives in src/ranking/sql/*.sql (edit those, not this file's SQL calls)
so the actual grouping/pairing/sampling/split logic stays hand-written.
"""

import argparse
from pathlib import Path

import duckdb

SQL_DIR = Path(__file__).parent / "sql"


def get_connection(db_path: "Path | None" = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection (in-memory if db_path is None, else a persistent .duckdb file)."""
    return duckdb.connect(str(db_path) if db_path else ":memory:")


def load_sql(name: str) -> str:
    """Read a .sql file from SQL_DIR by name (e.g. 'build_pairs' -> sql/build_pairs.sql)."""
    return (SQL_DIR / f"{name}.sql").read_text()


def register_raw_reviews(con: duckdb.DuckDBPyConnection, reviews_path: Path) -> None:
    """Create a DuckDB view/table over the raw Yelp review JSONL at reviews_path."""
    con.execute(f"CREATE OR REPLACE VIEW reviews AS SELECT * FROM read_json_auto('{reviews_path}')")


def sample_reviews_table(
    con: duckdb.DuckDBPyConnection,
    min_reviews_per_user: int = 2,
    max_reviews_per_user: int = 30,
) -> None:
    """Run sql/sample_reviews.sql to filter to eligible users and cap reviews per user."""
    con.execute(
        f"CREATE OR REPLACE TABLE sampled_reviews AS {load_sql('sample_reviews')}",
        {"min_reviews_per_user": min_reviews_per_user, "max_reviews_per_user": max_reviews_per_user},
    )


def build_pairs_table(
    con: duckdb.DuckDBPyConnection,
    max_pairs_per_user: int = 20,
) -> None:
    """Run sql/build_pairs.sql to self-join sampled_reviews into non-tied, capped-per-user pairs."""
    con.execute(
        f"CREATE OR REPLACE TABLE pairs AS {load_sql('build_pairs')}",
        {"max_pairs_per_user": max_pairs_per_user},
    )


def split_by_user(
    con: duckdb.DuckDBPyConnection,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> None:
    """Run sql/split_by_user.sql to assign each user_id (not each pair) to train/val/test."""
    con.execute(
        f"CREATE OR REPLACE TABLE user_splits AS {load_sql('split_by_user')}",
        {"train_frac": train_frac, "val_frac": val_frac},
    )


def export_splits(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    """COPY each split (joined pairs + user split assignment) out to out_dir/{train,val,test}.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        out_path = out_dir / f"{split}.jsonl"
        con.execute(
            f"""
            COPY (
                SELECT p.text_a, p.text_b, p.label
                FROM pairs p
                JOIN user_splits s ON s.user_id = p.user_id
                WHERE s.split = '{split}'
            ) TO '{out_path}' (FORMAT JSON)
            """
        )


def export_reviews_splits(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    """COPY sampled_reviews (joined to user_splits) out to out_dir/{train,val,test}_reviews.jsonl.

    Unlike export_splits, this is review-level (not pair-level): one row per
    (user_id, text, stars), needed to compute per-user Kendall's tau between
    predicted and true review rankings via metrics.kendall_tau_by_user.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        out_path = out_dir / f"{split}_reviews.jsonl"
        con.execute(
            f"""
            COPY (
                SELECT r.user_id, r.text, r.stars
                FROM sampled_reviews r
                JOIN user_splits s ON s.user_id = r.user_id
                WHERE s.split = '{split}'
            ) TO '{out_path}' (FORMAT JSON)
            """
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/pairs"))
    parser.add_argument("--db-path", type=Path, default=None, help="optional persistent .duckdb file")
    parser.add_argument("--min-reviews-per-user", type=int, default=2)
    parser.add_argument("--max-pairs-per-user", type=int, default=20)
    parser.add_argument(
        "--max-reviews-per-user",
        type=int,
        default=30,
        help="cap on reviews sampled per user before self-joining, to bound join size for power users",
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    """Wire the above steps together: connect, register raw data, build pairs, split, export."""
    args = parse_args()

    con = get_connection(args.db_path)
    register_raw_reviews(con, args.raw_dir / "yelp_academic_dataset_review.json")
    sample_reviews_table(con, args.min_reviews_per_user, args.max_reviews_per_user)
    build_pairs_table(con, args.max_pairs_per_user)
    split_by_user(con, args.train_frac, args.val_frac)
    export_splits(con, args.out_dir)
    export_reviews_splits(con, args.out_dir)


if __name__ == "__main__":
    main()
