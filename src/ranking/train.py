"""Training entrypoint: wires dataset -> model -> Trainer for the pairwise ranker."""

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/pairs"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/phase1"))
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    """Load config, build datasets/model, run HF Trainer, evaluate with pairwise_accuracy, save checkpoint."""
    args = parse_args()
    config = load_config(args.config)
    raise NotImplementedError


if __name__ == "__main__":
    main()
