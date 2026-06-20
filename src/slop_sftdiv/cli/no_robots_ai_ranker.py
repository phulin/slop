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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from slop_sftdiv.cli.hape_author_classifier import (
    _build_lr_scheduler,
    _max_steps,
    _string_or_empty,
)
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


@dataclass(frozen=True)
class AIPromptPair:
    base_doc_id: str
    human_record_id: str
    human_text: str
    ai_records: tuple[tuple[str, str, str], ...]
    split: str


class AIPromptPairDataset(Dataset[AIPromptPair]):
    def __init__(self, rows: list[AIPromptPair]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> AIPromptPair:
        return self.rows[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a scalar ModernBERT AI-score head with a per-prompt pairwise "
            "ranking loss: sum_model softplus(-(s(model) - s(human)))."
        )
    )
    parser.add_argument("--dataset-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="answerdotai/ModernBERT-large")
    parser.add_argument("--holdout-percent", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument(
        "--pad-to-max-length",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pad every batch to --max-length. Recommended when using --compile-model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Prompt groups per optimizer step. Each prompt contributes one human plus all AI continuations.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "wsd"],
        default="wsd",
        help="Learning-rate schedule. WSD is linear warmup, stable plateau, linear decay.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--decay-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Console progress interval. Train loss is written to CSV and W&B every step.",
    )
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="no-robots-ai-ranker")
    parser.add_argument("--wandb-job-type", default="train-ai-ranker")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    return parser


def _load_prompt_pairs(
    path: Path,
    *,
    holdout_percent: float,
    seed: int,
) -> tuple[list[AIPromptPair], list[AIPromptPair], dict[str, Any]]:
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
    for column in ["base_doc_id", "record_id", "role", "model", "text"]:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame = frame[
        (frame["base_doc_id"] != "")
        & (frame["record_id"] != "")
        & (frame["model"] != "")
        & (frame["text"] != "")
    ]

    rng = random.Random(seed)
    base_doc_ids = sorted(frame["base_doc_id"].unique().tolist())
    rng.shuffle(base_doc_ids)
    holdout_count = int(len(base_doc_ids) * holdout_percent / 100.0)
    if holdout_percent > 0.0 and holdout_count == 0 and base_doc_ids:
        holdout_count = 1
    heldout = set(base_doc_ids[:holdout_count])

    train_rows: list[AIPromptPair] = []
    eval_rows: list[AIPromptPair] = []
    skipped: Counter[str] = Counter()
    ai_models: set[str] = set()
    for base_doc_id, group in frame.groupby("base_doc_id", sort=True):
        human_group = group[group["role"] == "human_continuation"]
        ai_group = group[group["role"] == "llm_continuation"]
        if human_group.empty:
            skipped["missing_human"] += 1
            continue
        if ai_group.empty:
            skipped["missing_ai"] += 1
            continue
        human_row = human_group.sort_values("record_id").iloc[0]
        ai_records: list[tuple[str, str, str]] = []
        for row in ai_group.sort_values(["model", "record_id"]).itertuples(index=False):
            model = _string_or_empty(getattr(row, "model"))
            ai_models.add(model)
            ai_records.append(
                (
                    model,
                    _string_or_empty(getattr(row, "record_id")),
                    _string_or_empty(getattr(row, "text")),
                )
            )
        split = "eval" if str(base_doc_id) in heldout else "train"
        item = AIPromptPair(
            base_doc_id=str(base_doc_id),
            human_record_id=_string_or_empty(human_row["record_id"]),
            human_text=_string_or_empty(human_row["text"]),
            ai_records=tuple(ai_records),
            split=split,
        )
        if split == "eval":
            eval_rows.append(item)
        else:
            train_rows.append(item)
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    train_pair_count = sum(len(row.ai_records) for row in train_rows)
    eval_pair_count = sum(len(row.ai_records) for row in eval_rows)
    summary = {
        "dataset_parquet": str(path),
        "objective": "pairwise_ai_score",
        "loss": "mean_prompt_sum_model_softplus_negative_ai_minus_human_margin",
        "total_continuation_rows": int(len(frame)),
        "base_doc_count": len(base_doc_ids),
        "train_prompt_count": len(train_rows),
        "eval_prompt_count": len(eval_rows),
        "train_pair_count": train_pair_count,
        "eval_pair_count": eval_pair_count,
        "ai_models": sorted(ai_models),
        "ai_model_count": len(ai_models),
        "skipped_prompt_groups": dict(sorted(skipped.items())),
    }
    return train_rows, eval_rows, summary


def _collate(
    tokenizer: Any,
    rows: list[AIPromptPair],
    *,
    max_length: int,
    pad_to_max_length: bool = False,
) -> dict[str, Any]:
    texts: list[str] = []
    group_sizes: list[int] = []
    pair_human_indices: list[int] = []
    pair_ai_indices: list[int] = []
    pair_metadata: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        human_index = len(texts)
        texts.append(row.human_text)
        group_sizes.append(1 + len(row.ai_records))
        for model, record_id, text in row.ai_records:
            ai_index = len(texts)
            texts.append(text)
            pair_human_indices.append(human_index)
            pair_ai_indices.append(ai_index)
            pair_metadata.append(
                {
                    "row_index": row_index,
                    "base_doc_id": row.base_doc_id,
                    "human_record_id": row.human_record_id,
                    "ai_record_id": record_id,
                    "ai_model": model,
                    "split": row.split,
                }
            )
    encoded = tokenizer(
        texts,
        padding="max_length" if pad_to_max_length else True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded["group_sizes"] = group_sizes
    encoded["pair_human_indices"] = torch.tensor(pair_human_indices, dtype=torch.long)
    encoded["pair_ai_indices"] = torch.tensor(pair_ai_indices, dtype=torch.long)
    encoded["prompt_count"] = torch.tensor(len(rows), dtype=torch.float32)
    encoded["metadata"] = rows
    encoded["pair_metadata"] = pair_metadata
    return encoded


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _pairwise_loss(scores: torch.Tensor, group_sizes: list[int]) -> tuple[torch.Tensor, dict[str, float]]:
    scores = scores.reshape(-1).float()
    losses: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    offset = 0
    for group_size in group_sizes:
        if group_size < 2:
            offset += group_size
            continue
        human_score = scores[offset]
        ai_scores = scores[offset + 1 : offset + group_size]
        margin = ai_scores - human_score
        losses.append(F.softplus(-margin).sum())
        margins.append(margin)
        offset += group_size
    if not losses:
        raise ValueError("batch has no AI-human pairs")
    all_margins = torch.cat(margins)
    return torch.stack(losses).mean(), {
        "pair_accuracy": float((all_margins > 0).float().mean().detach().cpu()),
        "mean_margin": float(all_margins.mean().detach().cpu()),
        "mean_ai_score_minus_human": float(all_margins.mean().detach().cpu()),
    }


class AIPairwiseTrainingModule(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pair_human_indices: torch.Tensor,
        pair_ai_indices: torch.Tensor,
        prompt_count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.reshape(-1).float()
        margins = scores[pair_ai_indices] - scores[pair_human_indices]
        loss = F.softplus(-margins).sum() / prompt_count.float().clamp_min(1.0)
        pair_accuracy = (margins > 0).float().mean()
        mean_margin = margins.mean()
        return loss, pair_accuracy, mean_margin


def _compile_training_module(
    module: AIPairwiseTrainingModule,
    *,
    enabled: bool,
    mode: str,
    device: torch.device,
) -> nn.Module:
    if not enabled or device.type != "cuda":
        return module
    if not hasattr(torch, "compile"):
        return module
    print(f"Compiling AI pairwise training module with torch.compile(mode={mode})", flush=True)
    return cast(nn.Module, torch.compile(module, mode=mode))


def _train(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[AIPromptPair],
    args: argparse.Namespace,
    device: torch.device,
    wandb_run: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loader = DataLoader(
        AIPromptPairDataset(rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(
            tokenizer,
            batch,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length,
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps = _max_steps(len(loader), epochs=args.epochs, max_train_steps=args.max_train_steps)
    scheduler, scheduler_summary = _build_lr_scheduler(optimizer, args=args, total_steps=steps)
    model.train()
    train_module = AIPairwiseTrainingModule(cast(nn.Module, model)).to(device)
    compiled_train_module = _compile_training_module(
        train_module,
        enabled=bool(args.compile_model),
        mode=args.compile_mode,
        device=device,
    )
    compile_fallback = False
    train_log: list[dict[str, Any]] = []
    start = time.time()
    step = 0
    pair_count = sum(len(row.ai_records) for row in rows)
    while step < steps:
        for batch in loader:
            step += 1
            if step > steps:
                break
            batch = _move_batch(batch, device)
            batch.pop("group_sizes")
            batch.pop("metadata")
            batch.pop("pair_metadata")
            pair_human_indices = batch.pop("pair_human_indices")
            pair_ai_indices = batch.pop("pair_ai_indices")
            prompt_count = batch.pop("prompt_count")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                try:
                    loss, pair_accuracy, mean_margin = compiled_train_module(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pair_human_indices=pair_human_indices,
                        pair_ai_indices=pair_ai_indices,
                        prompt_count=prompt_count,
                    )
                except RuntimeError as exc:
                    if not args.compile_model or compile_fallback:
                        raise
                    print(
                        f"torch.compile pairwise training failed ({type(exc).__name__}: {exc}); "
                        "falling back to eager training",
                        flush=True,
                    )
                    compile_fallback = True
                    compiled_train_module = train_module
                    loss, pair_accuracy, mean_margin = compiled_train_module(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pair_human_indices=pair_human_indices,
                        pair_ai_indices=pair_ai_indices,
                        prompt_count=prompt_count,
                    )
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            current_lr = float(optimizer.param_groups[0]["lr"])
            row = {
                "step": step,
                "prompt_groups_seen": min(step * args.batch_size, len(rows)),
                "pairs_seen_upper_bound": min(
                    step * args.batch_size * max(1, pair_count // max(1, len(rows))),
                    pair_count,
                ),
                "loss": float(loss.detach().cpu()),
                "pair_accuracy": float(pair_accuracy.detach().cpu()),
                "mean_margin": float(mean_margin.detach().cpu()),
                "learning_rate": current_lr,
                "elapsed_seconds": time.time() - start,
            }
            train_log.append(row)
            wandb_run.log(
                {
                    "train/loss": row["loss"],
                    "train/pair_accuracy": row["pair_accuracy"],
                    "train/mean_margin": row["mean_margin"],
                    "train/learning_rate": current_lr,
                    "train/prompt_groups_seen": row["prompt_groups_seen"],
                    "train/elapsed_seconds": row["elapsed_seconds"],
                },
                step=step,
            )
            if step == 1 or step % args.log_every == 0 or step == steps:
                print(
                    f"step {step}/{steps} loss={row['loss']:.6f} "
                    f"pair_acc={row['pair_accuracy']:.4f} margin={row['mean_margin']:.4f}",
                    flush=True,
                )
    if train_log:
        train_log[-1].update(scheduler_summary)
    return train_log, {
        **scheduler_summary,
        "compile_model_requested": bool(args.compile_model),
        "compile_model_fallback": compile_fallback,
        "compile_mode": args.compile_mode,
    }


@torch.inference_mode()
def _evaluate(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[AIPromptPair],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    loader = DataLoader(
        AIPromptPairDataset(rows),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate(
            tokenizer,
            batch,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length,
        ),
    )
    model.eval()
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    predictions: list[dict[str, Any]] = []
    prompt_all_correct: Counter[str] = Counter()
    for batch in loader:
        batch = _move_batch(batch, device)
        group_sizes = batch.pop("group_sizes")
        metadata: list[AIPromptPair] = batch.pop("metadata")
        batch.pop("pair_metadata")
        batch.pop("pair_human_indices")
        batch.pop("pair_ai_indices")
        batch.pop("prompt_count")
        scores = model(**batch).logits.reshape(-1).float().cpu().tolist()
        offset = 0
        for row, group_size in zip(metadata, group_sizes):
            human_score = float(scores[offset])
            prompt_correct = 1
            for index, (ai_model, ai_record_id, _text) in enumerate(row.ai_records, start=1):
                ai_score = float(scores[offset + index])
                margin = ai_score - human_score
                correct = int(margin > 0.0)
                prompt_correct &= correct
                grouped[ai_model]["pairs"] += 1
                grouped[ai_model]["correct"] += correct
                grouped[ai_model]["margin_sum"] += margin
                grouped[ai_model]["ai_score_sum"] += ai_score
                grouped[ai_model]["human_score_sum"] += human_score
                predictions.append(
                    {
                        "split": row.split,
                        "base_doc_id": row.base_doc_id,
                        "human_record_id": row.human_record_id,
                        "ai_record_id": ai_record_id,
                        "ai_model": ai_model,
                        "human_score": human_score,
                        "ai_score": ai_score,
                        "margin": margin,
                        "correct": correct,
                    }
                )
            prompt_all_correct["prompts"] += 1
            prompt_all_correct["all_correct"] += prompt_correct
            offset += group_size
    metric_rows = []
    total_pairs = 0
    total_correct = 0
    margin_sum = 0.0
    for ai_model, counts in sorted(grouped.items()):
        pairs = counts["pairs"]
        total_pairs += pairs
        total_correct += counts["correct"]
        margin_sum += counts["margin_sum"]
        metric_rows.append(
            {
                "ai_model": ai_model,
                "pairs": pairs,
                "pairwise_accuracy": counts["correct"] / max(1, pairs),
                "mean_margin": counts["margin_sum"] / max(1, pairs),
                "mean_ai_score": counts["ai_score_sum"] / max(1, pairs),
                "mean_human_score": counts["human_score_sum"] / max(1, pairs),
            }
        )
    overall = {
        "eval_pair_count": total_pairs,
        "overall_pairwise_accuracy": total_correct / max(1, total_pairs),
        "overall_mean_margin": margin_sum / max(1, total_pairs),
        "prompt_all_correct_accuracy": prompt_all_correct["all_correct"] / max(1, prompt_all_correct["prompts"]),
    }
    return metric_rows, predictions, overall


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "step",
        "loss",
        "pair_accuracy",
        "mean_margin",
        "learning_rate",
        "lr_schedule",
        "warmup_steps",
        "stable_steps",
        "decay_steps",
        "ai_model",
        "pairs",
        "pairwise_accuracy",
        "mean_ai_score",
        "mean_human_score",
        "base_doc_id",
        "human_record_id",
        "ai_record_id",
        "human_score",
        "ai_score",
        "margin",
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
        "objective": "pairwise_ai_score",
        "holdout_percent": args.holdout_percent,
        "max_length": args.max_length,
        "pad_to_max_length": bool(args.pad_to_max_length),
        "batch_size_prompt_groups": args.batch_size,
        "eval_batch_size_prompt_groups": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "max_train_steps": args.max_train_steps,
        "lr_schedule": args.lr_schedule,
        "warmup_ratio": args.warmup_ratio,
        "decay_ratio": args.decay_ratio,
        "min_lr_ratio": args.min_lr_ratio,
        "compile_model": bool(args.compile_model),
        "compile_mode": args.compile_mode,
        "seed": args.seed,
        **data_summary,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, eval_rows, data_summary = _load_prompt_pairs(
        args.dataset_parquet,
        holdout_percent=args.holdout_percent,
        seed=args.seed,
    )
    wandb_run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage4", "phase4", "no-robots", "pairwise-ai-score", *args.wandb_tag],
        config=_wandb_config(args, data_summary),
    )
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        tokenizer = cast(Any, AutoTokenizer.from_pretrained(args.model))
        config = AutoConfig.from_pretrained(args.model)
        config.num_labels = 1
        config.problem_type = "regression"
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            config=config,
            ignore_mismatched_sizes=True,
        )
        model.config.problem_type = "regression"
        model.config.num_labels = 1
        model.to(device)
        train_log, train_summary = _train(
            model=model,
            tokenizer=tokenizer,
            rows=train_rows,
            args=args,
            device=device,
            wandb_run=wandb_run,
        )
        metric_rows, prediction_rows, eval_summary = _evaluate(
            model=model,
            tokenizer=tokenizer,
            rows=eval_rows,
            args=args,
            device=device,
        )
        summary = {
            **data_summary,
            **eval_summary,
            "model": args.model,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "holdout_percent": args.holdout_percent,
            "max_length": args.max_length,
            "pad_to_max_length": bool(args.pad_to_max_length),
            "batch_size_prompt_groups": args.batch_size,
            "eval_batch_size_prompt_groups": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "lr_schedule": args.lr_schedule,
            "warmup_ratio": args.warmup_ratio,
            "decay_ratio": args.decay_ratio,
            "min_lr_ratio": args.min_lr_ratio,
            "compile_model": bool(args.compile_model),
            "compile_mode": args.compile_mode,
            "compile_model_fallback": bool(train_summary["compile_model_fallback"]),
            "checkpoint_dir": str(output_dir / "checkpoint") if args.save_checkpoint else "",
        }
        if train_log:
            for key in ["warmup_steps", "stable_steps", "decay_steps"]:
                if key in train_summary:
                    summary[key] = train_summary[key]
        _write_csv(output_dir / "no_robots_ai_ranker_train_log.csv", train_log)
        _write_csv(output_dir / "no_robots_ai_ranker_metrics.csv", metric_rows)
        _write_csv(output_dir / "no_robots_ai_ranker_predictions.csv", prediction_rows)
        _write_json(output_dir / "no_robots_ai_ranker_summary.json", summary)
        wandb_run.log(
            {
                "eval/overall_pairwise_accuracy": summary["overall_pairwise_accuracy"],
                "eval/overall_mean_margin": summary["overall_mean_margin"],
                "eval/prompt_all_correct_accuracy": summary["prompt_all_correct_accuracy"],
            }
        )
        for row in metric_rows:
            name = row["ai_model"]
            wandb_run.log(
                {
                    f"eval_pairwise_accuracy/{name}": row["pairwise_accuracy"],
                    f"eval_mean_margin/{name}": row["mean_margin"],
                }
            )
        log_summary_table(wandb_run, "no_robots_ai_ranker_train_log", train_log)
        log_summary_table(wandb_run, "no_robots_ai_ranker_metrics", metric_rows)
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
        "Wrote no_robots AI ranker outputs to "
        f"{args.output_dir} ({summary['train_prompt_count']} train prompts, "
        f"{summary['eval_prompt_count']} eval prompts, "
        f"pairwise_accuracy={summary['overall_pairwise_accuracy']:.4f})"
    )


if __name__ == "__main__":
    main()
