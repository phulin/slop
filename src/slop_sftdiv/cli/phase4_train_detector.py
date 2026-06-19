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

import torch
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from slop_sftdiv.generation_text import (
    FeatureTextMode,
    feature_text_for_mode,
    strip_chat_role_markers,
)
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


DEFAULT_TRAIN_AUTHORS = (
    "gpt-4o-2024-08-06",
    "Meta-Llama-3-8B-Instruct",
)
DEFAULT_HELDOUT_AUTHORS = (
    "gpt-4o-mini-2024-07-18",
    "Meta-Llama-3-70B-Instruct",
)
HAPE_CHUNK_AUTHORS = {"chunk_1", "chunk_2", "unknown"}


@dataclass(frozen=True)
class TextRow:
    text: str
    label: int
    source: str
    doc_id: str
    split: str


class TextDataset(Dataset[TextRow]):
    def __init__(self, rows: list[TextRow]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> TextRow:
        return self.rows[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a bounded ModernBERT Phase 4 human-vs-LLM detector."
    )
    parser.add_argument("--model", default="answerdotai/ModernBERT-large")
    parser.add_argument("--hape-dataset", default="browndw/human-ai-parallel-corpus")
    parser.add_argument(
        "--hape-texts-parquet",
        type=Path,
        default=None,
        help=(
            "Optional compiled HAP-E long-table Parquet. If provided, the trainer "
            "uses this instead of loading --hape-dataset from Hugging Face."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hape-split-mode",
        choices=["doc_percent", "author"],
        default="doc_percent",
        help=(
            "doc_percent splits HAP-E by base document ID across selected authors; "
            "author preserves the original train-author/heldout-author split."
        ),
    )
    parser.add_argument(
        "--hape-holdout-percent",
        type=float,
        default=10.0,
        help="Percentage of HAP-E base documents held out for eval in doc_percent mode.",
    )
    parser.add_argument(
        "--hape-author",
        action="append",
        default=[],
        help=(
            "HAP-E LLM author to include in doc_percent mode. Repeat to select multiple. "
            "If omitted, all non-human HAP-E authors present in the dataset are used."
        ),
    )
    parser.add_argument("--train-author", action="append", default=list(DEFAULT_TRAIN_AUTHORS))
    parser.add_argument("--heldout-author", action="append", default=list(DEFAULT_HELDOUT_AUTHORS))
    parser.add_argument(
        "--max-train-pairs-per-author",
        type=int,
        default=0,
        help="Maximum train pairs per HAP-E author; 0 means no cap.",
    )
    parser.add_argument(
        "--max-eval-pairs-per-author",
        type=int,
        default=0,
        help="Maximum eval pairs per HAP-E author; 0 means no cap.",
    )
    parser.add_argument("--max-external-train-docs-per-source", type=int, default=0)
    parser.add_argument("--max-external-docs-per-source", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--external-reference-jsonl",
        action="append",
        default=[],
        metavar="LABEL=PATH[:FIELD]",
        help="External reference JSONL evaluated as human/reference label 0.",
    )
    parser.add_argument(
        "--external-generation-jsonl",
        action="append",
        default=[],
        metavar="LABEL=PATH[:FIELD]",
        help="External generation JSONL evaluated as LLM label 1.",
    )
    parser.add_argument(
        "--feature-text-mode",
        choices=["raw", "final_answer"],
        default="final_answer",
    )
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase4-detector")
    parser.add_argument("--wandb-job-type", default="train-detector")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    return parser


def _author_from_doc_id(doc_id: str) -> tuple[str, str]:
    if "@" in doc_id:
        base, author = doc_id.rsplit("@", 1)
        return base, author
    if doc_id.endswith("_chunk_1"):
        return doc_id.removesuffix("_chunk_1"), "chunk_1"
    if doc_id.endswith("_chunk_2"):
        return doc_id.removesuffix("_chunk_2"), "chunk_2"
    parts = doc_id.rsplit("_", 2)
    if len(parts) >= 2 and parts[-2] == "chunk":
        return "_".join(parts[:-2]), f"chunk_{parts[-1]}"
    return doc_id, "unknown"


def _load_hape_rows(
    *,
    dataset_name: str,
    texts_parquet: Path | None,
    split_mode: str,
    hape_authors: list[str],
    train_authors: set[str],
    heldout_authors: set[str],
    holdout_percent: float,
    max_train_pairs_per_author: int,
    max_eval_pairs_per_author: int,
    seed: int,
) -> tuple[list[TextRow], list[TextRow], dict[str, Any]]:
    source_summary: dict[str, Any]
    if texts_parquet is not None:
        by_doc, source_summary = _load_hape_by_doc_from_texts_parquet(texts_parquet)
    else:
        dataset = load_dataset(dataset_name, split="train")
        by_doc = defaultdict(dict)
        for row in dataset:
            doc_id = str(row["doc_id"])
            base_doc_id, author = _author_from_doc_id(doc_id)
            by_doc[base_doc_id][author] = str(row["text"])
        source_summary = {
            "hape_source": "hf_dataset",
            "hape_dataset": dataset_name,
            "hape_rows": len(dataset),
        }

    train_rows, eval_rows, summary = _build_hape_rows_from_by_doc(
        by_doc=by_doc,
        split_mode=split_mode,
        hape_authors=hape_authors,
        train_authors=train_authors,
        heldout_authors=heldout_authors,
        holdout_percent=holdout_percent,
        max_train_pairs_per_author=max_train_pairs_per_author,
        max_eval_pairs_per_author=max_eval_pairs_per_author,
        seed=seed,
    )
    summary.update(source_summary)
    summary["parallel_doc_count"] = len(by_doc)
    return train_rows, eval_rows, summary


def _load_hape_by_doc_from_texts_parquet(
    path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return _hape_by_doc_from_texts_frame(pd.read_parquet(path), source_path=path)


def _hape_by_doc_from_texts_frame(
    frame: pd.DataFrame,
    *,
    source_path: Path | str | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    required_columns = {"base_doc_id", "role", "model", "text"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"compiled HAP-E parquet is missing columns: {missing_columns}")

    by_doc: dict[str, dict[str, str]] = defaultdict(dict)
    role_counts: Counter[str] = Counter()
    llm_authors: set[str] = set()
    for row in frame.itertuples(index=False):
        base_doc_id = _string_or_empty(getattr(row, "base_doc_id"))
        role = _string_or_empty(getattr(row, "role"))
        model = _string_or_empty(getattr(row, "model"))
        text = _string_or_empty(getattr(row, "text"))
        if not base_doc_id or not role or not text:
            continue
        role_counts[role] += 1
        if role == "prompt":
            by_doc[base_doc_id]["chunk_1"] = text
        elif role == "human_continuation":
            by_doc[base_doc_id]["chunk_2"] = text
        elif role == "llm_continuation" and model:
            by_doc[base_doc_id][model] = text
            llm_authors.add(model)

    summary: dict[str, Any] = {
        "hape_source": "compiled_parquet",
        "hape_rows": int(len(frame)),
        "hape_role_counts": dict(sorted(role_counts.items())),
        "compiled_llm_authors": sorted(llm_authors),
    }
    if source_path is not None:
        summary["hape_texts_parquet"] = str(source_path)
    return by_doc, summary


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _build_hape_rows_from_by_doc(
    *,
    by_doc: dict[str, dict[str, str]],
    split_mode: str,
    hape_authors: list[str],
    train_authors: set[str],
    heldout_authors: set[str],
    holdout_percent: float,
    max_train_pairs_per_author: int,
    max_eval_pairs_per_author: int,
    seed: int,
) -> tuple[list[TextRow], list[TextRow], dict[str, Any]]:
    rng = random.Random(seed)
    train_rows: list[TextRow] = []
    eval_rows: list[TextRow] = []
    summary: dict[str, Any] = {
        "hape_split_mode": split_mode,
        "hape_holdout_percent": holdout_percent if split_mode == "doc_percent" else None,
        "train_pairs_by_author": {},
        "heldout_pairs_by_author": {},
    }
    if split_mode == "author":
        summary["train_authors"] = sorted(train_authors)
        summary["heldout_authors"] = sorted(heldout_authors)
        for author in sorted(train_authors):
            pairs = _parallel_pairs(by_doc, author=author)
            rng.shuffle(pairs)
            pairs = _cap_pairs(pairs, max_train_pairs_per_author)
            summary["train_pairs_by_author"][author] = len(pairs)
            for base_doc_id, human_text, llm_text in pairs:
                train_rows.append(
                    TextRow(human_text, 0, f"hape_human_for_{author}", base_doc_id, "train")
                )
                train_rows.append(TextRow(llm_text, 1, f"hape_{author}", base_doc_id, "train"))
        for author in sorted(heldout_authors):
            pairs = _parallel_pairs(by_doc, author=author)
            rng.shuffle(pairs)
            pairs = _cap_pairs(pairs, max_eval_pairs_per_author)
            summary["heldout_pairs_by_author"][author] = len(pairs)
            for base_doc_id, human_text, llm_text in pairs:
                eval_rows.append(
                    TextRow(human_text, 0, f"hape_human_for_{author}", base_doc_id, "hape_heldout")
                )
                eval_rows.append(TextRow(llm_text, 1, f"hape_{author}", base_doc_id, "hape_heldout"))
        rng.shuffle(train_rows)
        rng.shuffle(eval_rows)
        return train_rows, eval_rows, summary

    if split_mode != "doc_percent":
        raise ValueError(f"unsupported HAP-E split mode: {split_mode}")
    if holdout_percent < 0.0 or holdout_percent >= 100.0:
        raise ValueError("--hape-holdout-percent must be >= 0 and < 100")

    selected_authors = sorted(hape_authors or _available_hape_llm_authors(by_doc))
    summary["hape_authors"] = selected_authors
    eligible_doc_ids = [
        base_doc_id
        for base_doc_id, texts in by_doc.items()
        if texts.get("chunk_2") and any(texts.get(author) for author in selected_authors)
    ]
    rng.shuffle(eligible_doc_ids)
    heldout_count = int(len(eligible_doc_ids) * holdout_percent / 100.0)
    if holdout_percent > 0.0 and heldout_count == 0 and eligible_doc_ids:
        heldout_count = 1
    heldout_doc_ids = set(eligible_doc_ids[:heldout_count])
    train_doc_ids = set(eligible_doc_ids[heldout_count:])
    summary["train_doc_count"] = len(train_doc_ids)
    summary["heldout_doc_count"] = len(heldout_doc_ids)
    for author in selected_authors:
        train_pairs = _parallel_pairs(by_doc, author=author, allowed_doc_ids=train_doc_ids)
        eval_pairs = _parallel_pairs(by_doc, author=author, allowed_doc_ids=heldout_doc_ids)
        rng.shuffle(train_pairs)
        rng.shuffle(eval_pairs)
        train_pairs = _cap_pairs(train_pairs, max_train_pairs_per_author)
        eval_pairs = _cap_pairs(eval_pairs, max_eval_pairs_per_author)
        summary["train_pairs_by_author"][author] = len(train_pairs)
        summary["heldout_pairs_by_author"][author] = len(eval_pairs)
        for base_doc_id, human_text, llm_text in train_pairs:
            train_rows.append(
                TextRow(human_text, 0, f"hape_human_for_{author}", base_doc_id, "train")
            )
            train_rows.append(TextRow(llm_text, 1, f"hape_{author}", base_doc_id, "train"))
        for base_doc_id, human_text, llm_text in eval_pairs:
            eval_rows.append(
                TextRow(human_text, 0, f"hape_human_for_{author}", base_doc_id, "hape_heldout")
            )
            eval_rows.append(TextRow(llm_text, 1, f"hape_{author}", base_doc_id, "hape_heldout"))
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    if train_authors:
        summary.setdefault("legacy_train_authors_ignored", sorted(train_authors))
    if heldout_authors:
        summary.setdefault("legacy_heldout_authors_ignored", sorted(heldout_authors))
    return train_rows, eval_rows, summary


def _parallel_pairs(
    by_doc: dict[str, dict[str, str]],
    *,
    author: str,
    allowed_doc_ids: set[str] | None = None,
) -> list[tuple[str, str, str]]:
    pairs = []
    for base_doc_id, texts in by_doc.items():
        if allowed_doc_ids is not None and base_doc_id not in allowed_doc_ids:
            continue
        human_text = texts.get("chunk_2")
        llm_text = texts.get(author)
        if human_text and llm_text:
            pairs.append((base_doc_id, human_text, llm_text))
    return pairs


def _cap_pairs(
    pairs: list[tuple[str, str, str]],
    cap: int,
) -> list[tuple[str, str, str]]:
    if cap <= 0:
        return pairs
    return pairs[:cap]


def _available_hape_llm_authors(by_doc: dict[str, dict[str, str]]) -> list[str]:
    authors = set()
    for texts in by_doc.values():
        authors.update(texts)
    return sorted(author for author in authors if author not in HAPE_CHUNK_AUTHORS)


def _parse_spec(raw: str, *, default_field: str) -> tuple[str, Path, str]:
    if "=" not in raw:
        raise ValueError(f"expected LABEL=PATH[:FIELD], got {raw!r}")
    label, rest = raw.split("=", 1)
    path = Path(rest)
    if path.exists():
        return label, path, default_field
    if ":" in rest:
        path_text, field = rest.rsplit(":", 1)
        return label, Path(path_text), field
    return label, Path(rest), default_field


def _load_jsonl_rows(
    specs: list[str],
    *,
    label: int,
    split: str,
    default_field: str,
    offset_docs_per_source: int = 0,
    max_docs_per_source: int,
    feature_text_mode: str,
) -> list[TextRow]:
    rows: list[TextRow] = []
    for raw in specs:
        source, path, field = _parse_spec(raw, default_field=default_field)
        loaded = 0
        seen = 0
        with path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if loaded >= max_docs_per_source:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                text = str(row.get(field, "") or "")
                if label == 1:
                    text = strip_chat_role_markers(
                        feature_text_for_mode(text, cast(FeatureTextMode, feature_text_mode))
                    )
                text = text.strip()
                if not text:
                    continue
                if seen < offset_docs_per_source:
                    seen += 1
                    continue
                rows.append(
                    TextRow(
                        text=text,
                        label=label,
                        source=source,
                        doc_id=str(
                            row.get("record_id")
                            or row.get("doc_id")
                            or row.get("phase2_prompt_id")
                            or row_index
                        ),
                        split=split,
                    )
                )
                loaded += 1
                seen += 1
    return rows


def _collate(tokenizer: Any, rows: list[TextRow], *, max_length: int) -> dict[str, Any]:
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


def _train(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[TextRow],
    args: argparse.Namespace,
    device: torch.device,
    wandb_run: Any | None = None,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        TextDataset(rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(tokenizer, batch, max_length=args.max_length),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    update_rows: list[dict[str, Any]] = []
    max_steps = max(1, math_ceil(len(loader) * args.epochs))
    start = time.time()
    for step_index, batch in enumerate(loader, start=1):
        if step_index > max_steps:
            break
        batch = _move_batch(batch, device)
        metadata = batch.pop("metadata")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(**batch)
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        if step_index == 1 or step_index % 25 == 0 or step_index == max_steps:
            update_row = {
                "step": step_index,
                "examples_seen": min(step_index * args.batch_size, len(rows)),
                "loss": float(loss.detach().cpu()),
                "elapsed_seconds": time.time() - start,
                "batch_sources": ",".join(sorted({row.source for row in metadata})),
            }
            update_rows.append(update_row)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": update_row["loss"],
                        "train/examples_seen": update_row["examples_seen"],
                        "train/elapsed_seconds": update_row["elapsed_seconds"],
                    },
                    step=step_index,
                )
    return update_rows


def math_ceil(value: float) -> int:
    return int(value) if int(value) == value else int(value) + 1


@torch.inference_mode()
def _evaluate(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[TextRow],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    loader = DataLoader(
        TextDataset(rows),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate(tokenizer, batch, max_length=args.max_length),
    )
    model.eval()
    prediction_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for batch in loader:
        batch = _move_batch(batch, device)
        metadata = batch.pop("metadata")
        labels = batch.pop("labels")
        outputs = model(**batch)
        probabilities = torch.softmax(outputs.logits.float(), dim=-1)
        predictions = probabilities.argmax(dim=-1)
        for row, label, prediction, probability in zip(
            metadata,
            labels.cpu().tolist(),
            predictions.cpu().tolist(),
            probabilities[:, 1].cpu().tolist(),
        ):
            correct = int(label == prediction)
            key = (row.split, row.source)
            grouped[key]["examples"] += 1
            grouped[key]["correct"] += correct
            grouped[key]["label_sum"] += int(label)
            grouped[key]["prediction_sum"] += int(prediction)
            grouped[key]["llm_probability_sum"] += probability
            prediction_rows.append(
                {
                    "split": row.split,
                    "source": row.source,
                    "doc_id": row.doc_id,
                    "label": label,
                    "prediction": prediction,
                    "llm_probability": probability,
                    "correct": correct,
                }
            )
    metric_rows = []
    for (split, source), counts in sorted(grouped.items()):
        examples = counts["examples"]
        metric_rows.append(
            {
                "split": split,
                "source": source,
                "examples": examples,
                "accuracy": counts["correct"] / max(1, examples),
                "positive_label_rate": counts["label_sum"] / max(1, examples),
                "positive_prediction_rate": counts["prediction_sum"] / max(1, examples),
                "mean_llm_probability": counts["llm_probability_sum"] / max(1, examples),
            }
        )
    return metric_rows, prediction_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "split",
        "source",
        "examples",
        "accuracy",
        "positive_label_rate",
        "positive_prediction_rate",
        "mean_llm_probability",
    ]
    columns = [column for column in preferred if column in columns] + [
        column for column in columns if column not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wandb_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "hape_dataset": args.hape_dataset,
        "hape_texts_parquet": str(args.hape_texts_parquet) if args.hape_texts_parquet else None,
        "output_dir": str(args.output_dir),
        "hape_split_mode": args.hape_split_mode,
        "hape_holdout_percent": args.hape_holdout_percent,
        "hape_author": args.hape_author,
        "train_author": args.train_author,
        "heldout_author": args.heldout_author,
        "max_train_pairs_per_author": args.max_train_pairs_per_author,
        "max_eval_pairs_per_author": args.max_eval_pairs_per_author,
        "max_external_train_docs_per_source": args.max_external_train_docs_per_source,
        "max_external_docs_per_source": args.max_external_docs_per_source,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "seed": args.seed,
        "save_checkpoint": args.save_checkpoint,
        "external_reference_jsonl": args.external_reference_jsonl,
        "external_generation_jsonl": args.external_generation_jsonl,
        "feature_text_mode": args.feature_text_mode,
    }


def _summary_scalars(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        f"summary/{key}": value
        for key, value in summary.items()
        if value is None or isinstance(value, str | int | float | bool)
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage4", "phase4", "detector", *args.wandb_tag],
        config=_wandb_config(args),
    )
    try:
        train_rows, hape_eval_rows, data_summary = _load_hape_rows(
            dataset_name=args.hape_dataset,
            texts_parquet=args.hape_texts_parquet,
            split_mode=args.hape_split_mode,
            hape_authors=args.hape_author,
            train_authors=set(args.train_author),
            heldout_authors=set(args.heldout_author),
            holdout_percent=args.hape_holdout_percent,
            max_train_pairs_per_author=args.max_train_pairs_per_author,
            max_eval_pairs_per_author=args.max_eval_pairs_per_author,
            seed=args.seed,
        )
        external_train_rows = []
        if args.max_external_train_docs_per_source > 0:
            external_train_rows = [
                *_load_jsonl_rows(
                    args.external_reference_jsonl,
                    label=0,
                    split="external_train_reference",
                    default_field="text",
                    max_docs_per_source=args.max_external_train_docs_per_source,
                    feature_text_mode=args.feature_text_mode,
                ),
                *_load_jsonl_rows(
                    args.external_generation_jsonl,
                    label=1,
                    split="external_train_generation",
                    default_field="generation",
                    max_docs_per_source=args.max_external_train_docs_per_source,
                    feature_text_mode=args.feature_text_mode,
                ),
            ]
            random.Random(args.seed).shuffle(external_train_rows)
            train_rows.extend(external_train_rows)

        external_rows = [
            *_load_jsonl_rows(
                args.external_reference_jsonl,
                label=0,
                split="external_reference",
                default_field="text",
                offset_docs_per_source=args.max_external_train_docs_per_source,
                max_docs_per_source=args.max_external_docs_per_source,
                feature_text_mode=args.feature_text_mode,
            ),
            *_load_jsonl_rows(
                args.external_generation_jsonl,
                label=1,
                split="external_generation",
                default_field="generation",
                offset_docs_per_source=args.max_external_train_docs_per_source,
                max_docs_per_source=args.max_external_docs_per_source,
                feature_text_mode=args.feature_text_mode,
            ),
        ]
        tokenizer = cast(Any, AutoTokenizer.from_pretrained(args.model))
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
        model.to(device)
        train_log = _train(
            model=model,
            tokenizer=tokenizer,
            rows=train_rows,
            args=args,
            device=device,
            wandb_run=wandb_run,
        )
        metric_rows, prediction_rows = _evaluate(
            model=model,
            tokenizer=tokenizer,
            rows=[*hape_eval_rows, *external_rows],
            args=args,
            device=device,
        )
        output_dir: Path = args.output_dir
        _write_csv(output_dir / "phase4_detector_train_log.csv", train_log)
        _write_csv(output_dir / "phase4_detector_metrics.csv", metric_rows)
        _write_csv(output_dir / "phase4_detector_predictions.csv", prediction_rows)
        summary = {
            **data_summary,
            "model": args.model,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "train_examples": len(train_rows),
            "external_train_examples": len(external_train_rows),
            "hape_eval_examples": len(hape_eval_rows),
            "external_examples": len(external_rows),
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "checkpoint_dir": str(output_dir / "checkpoint") if args.save_checkpoint else "",
        }
        _write_json(output_dir / "phase4_detector_summary.json", summary)
        wandb_run.log(_summary_scalars(summary))
        log_summary_table(wandb_run, "phase4_detector_train_log", train_log)
        log_summary_table(wandb_run, "phase4_detector_metrics", metric_rows)
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
        "Wrote Phase 4 detector outputs to "
        f"{args.output_dir} ({summary['train_examples']} train examples, "
        f"{summary['hape_eval_examples']} HAP-E eval examples, "
        f"{summary['external_examples']} external examples)"
    )


if __name__ == "__main__":
    main()
