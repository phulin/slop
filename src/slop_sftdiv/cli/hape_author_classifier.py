from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


@dataclass(frozen=True)
class AuthorRow:
    text: str
    label: int
    author: str
    source_model: str
    base_doc_id: str
    record_id: str
    split: str


class AuthorDataset(Dataset[AuthorRow]):
    def __init__(self, rows: list[AuthorRow]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> AuthorRow:
        return self.rows[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ModernBERT to classify the author/model of compiled HAP-E continuations."
    )
    parser.add_argument("--dataset-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="answerdotai/ModernBERT-large")
    parser.add_argument("--holdout-percent", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "wsd"],
        default="constant",
        help="Learning-rate schedule. WSD is linear warmup, stable plateau, linear decay.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--decay-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument(
        "--label-mode",
        choices=["author", "binary"],
        default="author",
        help="Use the row model as a multiclass label, or collapse rows to human=0 and AI=1.",
    )
    parser.add_argument("--class-weighting", choices=["none", "inverse_frequency"], default="none")
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="hape-author-classifier")
    parser.add_argument("--wandb-job-type", default="train-author-classifier")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    return parser


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _load_author_rows(
    path: Path,
    *,
    holdout_percent: float,
    seed: int,
    label_mode: str = "author",
) -> tuple[list[AuthorRow], list[AuthorRow], dict[str, int], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if holdout_percent < 0.0 or holdout_percent >= 100.0:
        raise ValueError("--holdout-percent must be >= 0 and < 100")
    frame = pd.read_parquet(path)
    required = {"base_doc_id", "record_id", "role", "model", "text"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset parquet is missing columns: {missing}")
    frame = frame[frame["role"].isin(["human_continuation", "llm_continuation"])].copy()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["model"] = frame["model"].fillna("").astype(str).str.strip()
    frame["base_doc_id"] = frame["base_doc_id"].fillna("").astype(str).str.strip()
    frame["record_id"] = frame["record_id"].fillna("").astype(str).str.strip()
    frame = frame[(frame["text"] != "") & (frame["model"] != "") & (frame["base_doc_id"] != "")]
    source_models = sorted(frame["model"].unique().tolist())
    if label_mode == "author":
        authors = source_models
        label_by_author = {author: index for index, author in enumerate(authors)}
    elif label_mode == "binary":
        authors = ["human", "ai"]
        label_by_author = {"human": 0, "ai": 1}
    else:
        raise ValueError("--label-mode must be 'author' or 'binary'")

    rng = random.Random(seed)
    base_doc_ids = sorted(frame["base_doc_id"].unique().tolist())
    rng.shuffle(base_doc_ids)
    holdout_count = int(len(base_doc_ids) * holdout_percent / 100.0)
    if holdout_percent > 0.0 and holdout_count == 0 and base_doc_ids:
        holdout_count = 1
    heldout = set(base_doc_ids[:holdout_count])
    train_rows: list[AuthorRow] = []
    eval_rows: list[AuthorRow] = []
    for row in frame.itertuples(index=False):
        role = _string_or_empty(getattr(row, "role"))
        source_model = _string_or_empty(getattr(row, "model"))
        author = source_model if label_mode == "author" else ("human" if role == "human_continuation" else "ai")
        base_doc_id = _string_or_empty(getattr(row, "base_doc_id"))
        split = "eval" if base_doc_id in heldout else "train"
        item = AuthorRow(
            text=_string_or_empty(getattr(row, "text")),
            label=label_by_author[author],
            author=author,
            source_model=source_model,
            base_doc_id=base_doc_id,
            record_id=_string_or_empty(getattr(row, "record_id")),
            split=split,
        )
        if split == "eval":
            eval_rows.append(item)
        else:
            train_rows.append(item)
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    train_counts = Counter(row.author for row in train_rows)
    eval_counts = Counter(row.author for row in eval_rows)
    train_source_counts = Counter(row.source_model for row in train_rows)
    eval_source_counts = Counter(row.source_model for row in eval_rows)
    summary = {
        "dataset_parquet": str(path),
        "label_mode": label_mode,
        "total_continuation_rows": int(len(frame)),
        "authors": authors,
        "author_count": len(authors),
        "source_models": source_models,
        "source_model_count": len(source_models),
        "base_doc_count": len(base_doc_ids),
        "train_doc_count": len(base_doc_ids) - len(heldout),
        "eval_doc_count": len(heldout),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_examples_by_author": dict(sorted(train_counts.items())),
        "eval_examples_by_author": dict(sorted(eval_counts.items())),
        "train_examples_by_source_model": dict(sorted(train_source_counts.items())),
        "eval_examples_by_source_model": dict(sorted(eval_source_counts.items())),
    }
    return train_rows, eval_rows, label_by_author, summary


def _collate(tokenizer: Any, rows: list[AuthorRow], *, max_length: int) -> dict[str, Any]:
    encoded = tokenizer(
        [row.text for row in rows],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor([row.label for row in rows], dtype=torch.long)
    encoded["metadata"] = rows
    return encoded


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _class_weights(
    rows: list[AuthorRow],
    *,
    label_count: int,
    mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = Counter(row.label for row in rows)
    total = max(1, len(rows))
    weights = [
        total / max(1, label_count * counts.get(label, 0))
        for label in range(label_count)
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _max_steps(loader_len: int, *, epochs: float, max_train_steps: int) -> int:
    planned = max(1, int(loader_len * epochs))
    if max_train_steps > 0:
        return min(planned, max_train_steps)
    return planned


def _wsd_lr_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    min_lr_ratio: float,
) -> float:
    if step <= 0:
        return 0.0 if warmup_steps > 0 else 1.0
    if warmup_steps > 0 and step <= warmup_steps:
        return step / warmup_steps
    decay_start = max(warmup_steps, total_steps - decay_steps)
    if decay_steps > 0 and step > decay_start:
        remaining = max(0, total_steps - step)
        decay_progress = remaining / decay_steps
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay_progress
    return 1.0


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    total_steps: int,
) -> tuple[torch.optim.lr_scheduler.LambdaLR | None, dict[str, Any]]:
    if args.warmup_ratio < 0.0 or args.decay_ratio < 0.0:
        raise ValueError("--warmup-ratio and --decay-ratio must be non-negative")
    if args.warmup_ratio + args.decay_ratio > 1.0:
        raise ValueError("--warmup-ratio + --decay-ratio must be <= 1")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be between 0 and 1")
    if args.lr_schedule == "constant":
        return None, {
            "lr_schedule": "constant",
            "warmup_steps": 0,
            "stable_steps": total_steps,
            "decay_steps": 0,
            "min_lr_ratio": 1.0,
        }
    warmup_steps = int(total_steps * args.warmup_ratio)
    decay_steps = int(total_steps * args.decay_ratio)
    stable_steps = max(0, total_steps - warmup_steps - decay_steps)

    def lr_lambda(step: int) -> float:
        return _wsd_lr_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler, {
        "lr_schedule": "wsd",
        "warmup_steps": warmup_steps,
        "stable_steps": stable_steps,
        "decay_steps": decay_steps,
        "min_lr_ratio": args.min_lr_ratio,
    }


def _train(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[AuthorRow],
    args: argparse.Namespace,
    device: torch.device,
    class_weights: torch.Tensor | None,
    wandb_run: Any,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        AuthorDataset(rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(tokenizer, batch, max_length=args.max_length),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    steps = _max_steps(len(loader), epochs=args.epochs, max_train_steps=args.max_train_steps)
    scheduler, scheduler_summary = _build_lr_scheduler(optimizer, args=args, total_steps=steps)
    model.train()
    train_log: list[dict[str, Any]] = []
    start = time.time()
    step = 0
    while step < steps:
        for batch in loader:
            step += 1
            if step > steps:
                break
            batch = _move_batch(batch, device)
            batch.pop("metadata")
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(**batch).logits
                loss = loss_fn(logits.float(), labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if step == 1 or step % args.log_every == 0 or step == steps:
                current_lr = float(optimizer.param_groups[0]["lr"])
                row = {
                    "step": step,
                    "examples_seen": min(step * args.batch_size, len(rows)),
                    "loss": float(loss.detach().cpu()),
                    "learning_rate": current_lr,
                    "elapsed_seconds": time.time() - start,
                }
                train_log.append(row)
                wandb_run.log(
                    {
                        "train/loss": row["loss"],
                        "train/learning_rate": current_lr,
                        "train/examples_seen": row["examples_seen"],
                        "train/elapsed_seconds": row["elapsed_seconds"],
                    },
                    step=step,
                )
                print(f"step {step}/{steps} loss={row['loss']:.6f}", flush=True)
    if train_log:
        train_log[-1].update(scheduler_summary)
    return train_log


@torch.inference_mode()
def _evaluate(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[AuthorRow],
    args: argparse.Namespace,
    device: torch.device,
    id_to_author: dict[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    loader = DataLoader(
        AuthorDataset(rows),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate(tokenizer, batch, max_length=args.max_length),
    )
    model.eval()
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    confusion: Counter[tuple[str, str]] = Counter()
    predictions: list[dict[str, Any]] = []
    ai_label_id = next((index for index, author in id_to_author.items() if author == "ai"), None)
    for batch in loader:
        batch = _move_batch(batch, device)
        metadata = batch.pop("metadata")
        labels = batch.pop("labels")
        logits = model(**batch).logits.float()
        probabilities = torch.softmax(logits, dim=-1)
        predicted = probabilities.argmax(dim=-1)
        for row, label, prediction, probability_row in zip(
            metadata,
            labels.cpu().tolist(),
            predicted.cpu().tolist(),
            probabilities.cpu().tolist(),
        ):
            author = id_to_author[int(label)]
            pred_author = id_to_author[int(prediction)]
            correct = int(label == prediction)
            grouped[(row.source_model, author)]["examples"] += 1
            grouped[(row.source_model, author)]["correct"] += correct
            grouped[(row.source_model, author)]["label_sum"] += int(label)
            grouped[(row.source_model, author)]["prediction_sum"] += int(prediction)
            confusion[(author, pred_author)] += 1
            prediction_row = {
                "split": row.split,
                "record_id": row.record_id,
                "base_doc_id": row.base_doc_id,
                "source_model": row.source_model,
                "author": author,
                "pred_author": pred_author,
                "label": int(label),
                "prediction": int(prediction),
                "correct": correct,
                "pred_probability": probability_row[int(prediction)],
                "gold_probability": probability_row[int(label)],
            }
            if ai_label_id is not None:
                prediction_row["ai_probability"] = probability_row[ai_label_id]
            predictions.append(prediction_row)
    metric_rows = []
    for (source_model, author), counts in sorted(grouped.items()):
        examples = counts["examples"]
        metric_rows.append(
            {
                "source_model": source_model,
                "author": author,
                "examples": examples,
                "accuracy": counts["correct"] / max(1, examples),
                "positive_label_rate": counts["label_sum"] / max(1, examples),
                "positive_prediction_rate": counts["prediction_sum"] / max(1, examples),
            }
        )
    confusion_rows = [
        {"author": author, "pred_author": pred_author, "count": count}
        for (author, pred_author), count in sorted(confusion.items())
    ]
    return metric_rows, confusion_rows, predictions


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "step",
        "loss",
        "learning_rate",
        "lr_schedule",
        "warmup_steps",
        "stable_steps",
        "decay_steps",
        "author",
        "source_model",
        "pred_author",
        "examples",
        "accuracy",
        "positive_label_rate",
        "positive_prediction_rate",
        "count",
        "record_id",
        "base_doc_id",
        "label",
        "prediction",
        "correct",
    ]
    columns = [column for column in preferred if column in columns] + [
        column for column in columns if column not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wandb_config(args: argparse.Namespace, data_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_parquet": str(args.dataset_parquet),
        "output_dir": str(args.output_dir),
        "model": args.model,
        "holdout_percent": args.holdout_percent,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "max_train_steps": args.max_train_steps,
        "lr_schedule": args.lr_schedule,
        "warmup_ratio": args.warmup_ratio,
        "decay_ratio": args.decay_ratio,
        "min_lr_ratio": args.min_lr_ratio,
        "label_mode": args.label_mode,
        "class_weighting": args.class_weighting,
        "seed": args.seed,
        "authors": data_summary["authors"],
        "train_examples": data_summary["train_examples"],
        "eval_examples": data_summary["eval_examples"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, eval_rows, label_by_author, data_summary = _load_author_rows(
        args.dataset_parquet,
        holdout_percent=args.holdout_percent,
        seed=args.seed,
        label_mode=args.label_mode,
    )
    id_to_author = {index: author for author, index in label_by_author.items()}
    wandb_run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage4", "phase4", f"hape-{args.label_mode}", *args.wandb_tag],
        config=_wandb_config(args, data_summary),
    )
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = cast(Any, AutoTokenizer.from_pretrained(args.model))
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=len(label_by_author),
            ignore_mismatched_sizes=True,
        )
        model.config.label2id = label_by_author
        model.config.id2label = id_to_author
        model.to(device)
        weights = _class_weights(
            train_rows,
            label_count=len(label_by_author),
            mode=args.class_weighting,
            device=device,
        )
        train_log = _train(
            model=model,
            tokenizer=tokenizer,
            rows=train_rows,
            args=args,
            device=device,
            class_weights=weights,
            wandb_run=wandb_run,
        )
        metric_rows, confusion_rows, prediction_rows = _evaluate(
            model=model,
            tokenizer=tokenizer,
            rows=eval_rows,
            args=args,
            device=device,
            id_to_author=id_to_author,
        )
        total_examples = sum(int(row["examples"]) for row in metric_rows)
        total_correct = sum(int(row["examples"]) * float(row["accuracy"]) for row in metric_rows)
        overall_accuracy = total_correct / max(1, total_examples)
        summary = {
            **data_summary,
            "model": args.model,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "label_by_author": label_by_author,
            "id_to_author": id_to_author,
            "holdout_percent": args.holdout_percent,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "lr_schedule": args.lr_schedule,
            "warmup_ratio": args.warmup_ratio,
            "decay_ratio": args.decay_ratio,
            "min_lr_ratio": args.min_lr_ratio,
            "class_weighting": args.class_weighting,
            "label_mode": args.label_mode,
            "overall_eval_accuracy": overall_accuracy,
            "checkpoint_dir": str(output_dir / "checkpoint") if args.save_checkpoint else "",
        }
        if train_log:
            for key in ["warmup_steps", "stable_steps", "decay_steps"]:
                if key in train_log[-1]:
                    summary[key] = train_log[-1][key]
        _write_csv(output_dir / "hape_author_train_log.csv", train_log)
        _write_csv(output_dir / "hape_author_metrics.csv", metric_rows)
        _write_csv(output_dir / "hape_author_confusion.csv", confusion_rows)
        _write_csv(output_dir / "hape_author_predictions.csv", prediction_rows)
        _write_json(output_dir / "hape_author_summary.json", summary)
        wandb_run.log({"eval/overall_accuracy": overall_accuracy})
        for row in metric_rows:
            metric_name = row.get("source_model") or row["author"]
            wandb_run.log({f"eval_accuracy/{metric_name}": row["accuracy"]})
        log_summary_table(wandb_run, "hape_author_train_log", train_log)
        log_summary_table(wandb_run, "hape_author_metrics", metric_rows)
        log_summary_table(wandb_run, "hape_author_confusion", confusion_rows)
        if args.save_checkpoint:
            checkpoint_dir = output_dir / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
        return summary
    finally:
        wandb_run.finish()


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        "Wrote HAP-E author classifier outputs to "
        f"{args.output_dir} ({summary['train_examples']} train, "
        f"{summary['eval_examples']} eval, accuracy={summary['overall_eval_accuracy']:.4f})"
    )


if __name__ == "__main__":
    main()
