"""Training entrypoint: wires dataset -> model -> Trainer for the pairwise ranker."""

import argparse
from pathlib import Path

import numpy as np
import yaml
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

from src.ranking.dataset import PairDataset, load_pairs_jsonl
from src.ranking.metrics import pairwise_accuracy
from src.ranking.model import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/pairs"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/phase1"))
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"pairwise_accuracy": pairwise_accuracy(predictions.tolist(), labels.tolist())}


def main() -> None:
    """Load config, build datasets/model, run HF Trainer, evaluate with pairwise_accuracy, save checkpoint."""
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["seed"])

    model, tokenizer = load_model_and_tokenizer(config["model_name"], config["num_labels"])

    train_pairs = load_pairs_jsonl(args.data_dir / "train.jsonl")
    val_pairs = load_pairs_jsonl(args.data_dir / "val.jsonl")

    max_train_samples = config.get("max_train_samples")
    if max_train_samples is not None:
        train_pairs = train_pairs[:max_train_samples]

    max_eval_samples = config.get("max_eval_samples")
    if max_eval_samples is not None:
        val_pairs = val_pairs[:max_eval_samples]

    train_dataset = PairDataset(train_pairs, tokenizer, max_length=config["max_length"])
    val_dataset = PairDataset(val_pairs, tokenizer, max_length=config["max_length"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=config["num_epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pairwise_accuracy",
        logging_steps=50,
        seed=config["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
