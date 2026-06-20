from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from slop_sftdiv.cli.phase4_batchtopk_sae import BatchTopKSAE, _activation_mask, _load_sae


@dataclass(frozen=True)
class ScoringDoc:
    base_doc_id: str
    record_id: str
    source: str
    role: str
    label: int
    text: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score BatchTopK SAE latents as AI-style candidates on a sampled prompt "
            "set using source enrichment, specificity, and first-order scalar-score effects."
        )
    )
    parser.add_argument("--dataset-parquet", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sae-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-prompts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pad-to-max-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="reduce-overhead")
    parser.add_argument("--candidate-latents", type=int, default=128)
    parser.add_argument("--top-output-latents", type=int, default=100)
    parser.add_argument("--max-examples-per-latent", type=int, default=12)
    parser.add_argument("--context-chars", type=int, default=280)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--ubiquitous-token-rate-threshold",
        type=float,
        default=0.95,
        help="Latents above this active-token rate are penalized as broad reconstruction axes.",
    )
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


def _load_sample_docs(path: Path, *, sample_prompts: int, seed: int) -> list[ScoringDoc]:
    frame = pd.read_parquet(path)
    required = {"base_doc_id", "record_id", "role", "model", "text"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset parquet is missing columns: {missing}")
    frame = frame[frame["role"].isin(["human_continuation", "llm_continuation"])].copy()
    for column in required:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame = frame[
        (frame["base_doc_id"] != "")
        & (frame["record_id"] != "")
        & (frame["model"] != "")
        & (frame["text"] != "")
    ].copy()
    prompt_ids = sorted(frame["base_doc_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(prompt_ids)
    selected = set(prompt_ids[:sample_prompts])
    frame = frame[frame["base_doc_id"].isin(selected)].copy()

    docs: list[ScoringDoc] = []
    for row in frame.sort_values(["base_doc_id", "role", "model", "record_id"]).itertuples(index=False):
        role = _string_or_empty(getattr(row, "role"))
        source = _string_or_empty(getattr(row, "model"))
        docs.append(
            ScoringDoc(
                base_doc_id=_string_or_empty(getattr(row, "base_doc_id")),
                record_id=_string_or_empty(getattr(row, "record_id")),
                source=source,
                role=role,
                label=0 if role == "human_continuation" else 1,
                text=_string_or_empty(getattr(row, "text")),
            )
        )
    return docs


def _batches(items: list[ScoringDoc], batch_size: int) -> list[list[ScoringDoc]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _encode(tokenizer: Any, docs: list[ScoringDoc], *, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    encoded = tokenizer(
        [doc.text for doc in docs],
        padding="max_length" if args.pad_to_max_length else True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }


def _compile_model(model: nn.Module, *, args: argparse.Namespace, device: torch.device) -> nn.Module:
    if not args.compile_model or device.type != "cuda" or not hasattr(torch, "compile"):
        return model
    print(f"Compiling detector with torch.compile(mode={args.compile_mode})", flush=True)
    return cast(nn.Module, torch.compile(model, mode=args.compile_mode))


def _token_context(tokenizer: Any, encoded: dict[str, Any], *, doc_index: int, token_index: int, context_chars: int) -> tuple[str, str]:
    token_id = int(encoded["input_ids"][doc_index, token_index].detach().cpu())
    token = tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(token_id)]).strip()
    ids = encoded["input_ids"][doc_index].detach().cpu().tolist()
    left = max(0, token_index - 32)
    right = min(len(ids), token_index + 33)
    text = tokenizer.decode(ids[left:right], skip_special_tokens=True)
    text = " ".join(text.split())
    if len(text) > context_chars:
        half = max(20, context_chars // 2)
        text = text[:half].rstrip() + " ... " + text[-half:].lstrip()
    return token, text


def _source_key(doc: ScoringDoc) -> str:
    return "human" if doc.label == 0 else doc.source


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "rank",
        "latent_id",
        "slop_candidate_score",
        "ai_human_log_odds",
        "ai_mass",
        "human_mass",
        "ai_mass_share",
        "first_order_ai_score_drop",
        "first_order_human_score_drop",
        "first_order_ai_minus_human_drop",
        "active_token_rate",
        "ai_active_doc_rate",
        "human_active_doc_rate",
        "max_activation",
        "category_hint",
        "example_rank",
        "source",
        "activation",
        "doc_id",
        "base_doc_id",
        "token_index",
        "token",
        "context",
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


def _category_hint(row: dict[str, Any]) -> str:
    if float(row["active_token_rate"]) >= 0.95:
        return "broad_or_ubiquitous"
    if float(row["first_order_ai_minus_human_drop"]) <= 0:
        return "not_ai_score_positive"
    if float(row["ai_human_log_odds"]) >= 2.0:
        return "high_ai_enrichment"
    if float(row["ai_human_log_odds"]) >= 1.0:
        return "moderate_ai_enrichment"
    return "weak_ai_enrichment"


@torch.inference_mode()
def _source_stats_pass(
    *,
    model: nn.Module,
    tokenizer: Any,
    sae: BatchTopKSAE,
    docs: list[ScoringDoc],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    latent_dim = sae.latent_dim
    source_mass: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(latent_dim, dtype=torch.float64))
    source_active_tokens: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(latent_dim, dtype=torch.int64))
    source_token_totals: Counter[str] = Counter()
    source_doc_totals: Counter[str] = Counter()
    source_active_docs: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(latent_dim, dtype=torch.int64))
    latent_max = torch.zeros(latent_dim, dtype=torch.float32)
    examples: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_example_docs: dict[int, set[str]] = defaultdict(set)
    batches = _batches(docs, args.batch_size)
    start = time.time()
    for batch_index, batch_docs in enumerate(batches, start=1):
        encoded = _encode(tokenizer, batch_docs, args=args, device=device)
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-2]
        token_mask = _activation_mask(encoded)
        flat_hidden = hidden[token_mask].float()
        codes = sae.encode(flat_hidden).detach().cpu()
        flat_doc_indexes = (
            torch.arange(len(batch_docs))
            .unsqueeze(1)
            .expand(-1, token_mask.shape[1])
            .reshape(-1)[token_mask.detach().cpu().reshape(-1)]
        )
        latent_max = torch.maximum(latent_max, codes.max(dim=0).values.float())
        for doc_index, doc in enumerate(batch_docs):
            source = _source_key(doc)
            doc_rows = flat_doc_indexes == doc_index
            doc_codes = codes[doc_rows]
            if doc_codes.numel() == 0:
                continue
            source_doc_totals[source] += 1
            source_token_totals[source] += int(doc_codes.shape[0])
            source_mass[source] += doc_codes.sum(dim=0, dtype=torch.float64)
            active_tokens = doc_codes > 0
            source_active_tokens[source] += active_tokens.sum(dim=0).to(torch.int64)
            source_active_docs[source] += active_tokens.any(dim=0).to(torch.int64)

        top_values, top_indexes = torch.topk(
            codes.flatten(),
            k=min(codes.numel(), max(1, args.candidate_latents * args.max_examples_per_latent)),
        )
        kept_doc_indexes = flat_doc_indexes.tolist()
        kept_token_indexes = (
            torch.arange(token_mask.shape[1])
            .unsqueeze(0)
            .expand(len(batch_docs), -1)
            .reshape(-1)[token_mask.detach().cpu().reshape(-1)]
            .tolist()
        )
        for value, flat_index in zip(top_values.tolist(), top_indexes.tolist()):
            if value <= 0:
                continue
            token_row = flat_index // latent_dim
            latent_id = int(flat_index % latent_dim)
            doc_index = int(kept_doc_indexes[token_row])
            doc = batch_docs[doc_index]
            if doc.record_id in seen_example_docs[latent_id]:
                continue
            token_index = int(kept_token_indexes[token_row])
            token, context = _token_context(
                tokenizer,
                encoded,
                doc_index=doc_index,
                token_index=token_index,
                context_chars=args.context_chars,
            )
            examples[latent_id].append(
                {
                    "latent_id": latent_id,
                    "activation": float(value),
                    "source": _source_key(doc),
                    "label": doc.label,
                    "doc_id": doc.record_id,
                    "base_doc_id": doc.base_doc_id,
                    "token_index": token_index,
                    "token": token,
                    "context": context,
                }
            )
            seen_example_docs[latent_id].add(doc.record_id)
            examples[latent_id].sort(key=lambda row: float(row["activation"]), reverse=True)
            del examples[latent_id][args.max_examples_per_latent :]
        if args.progress_every > 0 and (batch_index == 1 or batch_index % args.progress_every == 0):
            print(
                f"stats pass {min(batch_index * args.batch_size, len(docs)):,}/{len(docs):,} docs "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    return (
        {
            "source_mass": source_mass,
            "source_active_tokens": source_active_tokens,
            "source_token_totals": source_token_totals,
            "source_doc_totals": source_doc_totals,
            "source_active_docs": source_active_docs,
            "latent_max": latent_max,
        },
        examples,
    )


def _initial_rows(stats: dict[str, Any], *, latent_dim: int, ubiquitous_threshold: float) -> list[dict[str, Any]]:
    source_mass: dict[str, torch.Tensor] = stats["source_mass"]
    source_active_tokens: dict[str, torch.Tensor] = stats["source_active_tokens"]
    source_token_totals: Counter[str] = stats["source_token_totals"]
    source_doc_totals: Counter[str] = stats["source_doc_totals"]
    source_active_docs: dict[str, torch.Tensor] = stats["source_active_docs"]
    latent_max: torch.Tensor = stats["latent_max"]
    ai_sources = [source for source in source_mass if source != "human"]
    eps = 1e-6
    rows: list[dict[str, Any]] = []
    for latent_id in range(latent_dim):
        human_mass = float(source_mass["human"][latent_id]) if "human" in source_mass else 0.0
        ai_mass = sum(float(source_mass[source][latent_id]) for source in ai_sources)
        total_mass = human_mass + ai_mass
        human_tokens = int(source_token_totals["human"])
        ai_tokens = sum(int(source_token_totals[source]) for source in ai_sources)
        human_active_tokens = (
            int(source_active_tokens["human"][latent_id]) if "human" in source_active_tokens else 0
        )
        ai_active_tokens = sum(int(source_active_tokens[source][latent_id]) for source in ai_sources)
        human_docs = int(source_doc_totals["human"])
        ai_docs = sum(int(source_doc_totals[source]) for source in ai_sources)
        human_active_docs = (
            int(source_active_docs["human"][latent_id]) if "human" in source_active_docs else 0
        )
        ai_active_docs = sum(int(source_active_docs[source][latent_id]) for source in ai_sources)
        ai_rate_mass = ai_mass / max(eps, ai_tokens)
        human_rate_mass = human_mass / max(eps, human_tokens)
        active_token_rate = (ai_active_tokens + human_active_tokens) / max(1, ai_tokens + human_tokens)
        ai_active_doc_rate = ai_active_docs / max(1, ai_docs)
        human_active_doc_rate = human_active_docs / max(1, human_docs)
        log_odds = math.log((ai_rate_mass + eps) / (human_rate_mass + eps))
        specificity = max(0.0, 1.0 - min(1.0, active_token_rate / ubiquitous_threshold))
        preliminary_score = log_odds * specificity * math.log1p(total_mass)
        rows.append(
            {
                "latent_id": latent_id,
                "ai_mass": ai_mass,
                "human_mass": human_mass,
                "total_mass": total_mass,
                "ai_mass_share": ai_mass / max(eps, total_mass),
                "ai_human_log_odds": log_odds,
                "active_token_rate": active_token_rate,
                "ai_active_doc_rate": ai_active_doc_rate,
                "human_active_doc_rate": human_active_doc_rate,
                "max_activation": float(latent_max[latent_id]),
                "preliminary_score": preliminary_score,
            }
        )
    rows.sort(key=lambda row: float(row["preliminary_score"]), reverse=True)
    return rows


def _gradient_effect_pass(
    *,
    model: nn.Module,
    tokenizer: Any,
    sae: BatchTopKSAE,
    docs: list[ScoringDoc],
    candidate_latents: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    if not candidate_latents:
        return {}
    effects: dict[int, Counter[str]] = {latent_id: Counter() for latent_id in candidate_latents}
    latent_index = torch.tensor(candidate_latents, dtype=torch.long, device=device)
    decoder_vectors = sae.decoder.weight[:, latent_index].detach().to(device=device, dtype=torch.float32)
    batches = _batches(docs, args.batch_size)
    start = time.time()
    model.eval()
    for batch_index, batch_docs in enumerate(batches, start=1):
        encoded = _encode(tokenizer, batch_docs, args=args, device=device)
        with torch.enable_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-2]
            hidden.retain_grad()
            scores = outputs.logits.reshape(-1).float()
            model.zero_grad(set_to_none=True)
            scores.sum().backward()
            grad = hidden.grad
            if grad is None:
                raise RuntimeError("failed to capture gradient for penultimate hidden states")
        token_mask = _activation_mask(encoded)
        flat_hidden = hidden.detach()[token_mask].float()
        flat_grad = grad.detach()[token_mask].float()
        codes = sae.encode(flat_hidden)[:, latent_index].float()
        projected_grad = flat_grad @ decoder_vectors
        token_effect = (codes * projected_grad).detach().cpu()
        flat_doc_indexes = (
            torch.arange(len(batch_docs))
            .unsqueeze(1)
            .expand(-1, token_mask.shape[1])
            .reshape(-1)[token_mask.detach().cpu().reshape(-1)]
        )
        for doc_index, doc in enumerate(batch_docs):
            doc_rows = flat_doc_indexes == doc_index
            doc_effect = token_effect[doc_rows].sum(dim=0)
            bucket = "human" if doc.label == 0 else "ai"
            for offset, latent_id in enumerate(candidate_latents):
                value = float(doc_effect[offset])
                effects[latent_id][f"{bucket}_drop_sum"] += value
                effects[latent_id][f"{bucket}_docs"] += 1
                if value > 0:
                    effects[latent_id][f"{bucket}_positive_docs"] += 1
        if args.progress_every > 0 and (batch_index == 1 or batch_index % args.progress_every == 0):
            print(
                f"gradient pass {min(batch_index * args.batch_size, len(docs)):,}/{len(docs):,} docs "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    out: dict[int, dict[str, float]] = {}
    for latent_id, counts in effects.items():
        ai_docs = max(1.0, float(counts["ai_docs"]))
        human_docs = max(1.0, float(counts["human_docs"]))
        ai_drop = float(counts["ai_drop_sum"]) / ai_docs
        human_drop = float(counts["human_drop_sum"]) / human_docs
        out[latent_id] = {
            "first_order_ai_score_drop": ai_drop,
            "first_order_human_score_drop": human_drop,
            "first_order_ai_minus_human_drop": ai_drop - human_drop,
            "first_order_ai_positive_doc_rate": float(counts["ai_positive_docs"]) / ai_docs,
            "first_order_human_positive_doc_rate": float(counts["human_positive_docs"]) / human_docs,
        }
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    docs = _load_sample_docs(args.dataset_parquet, sample_prompts=args.sample_prompts, seed=args.seed)
    source_counts = Counter(_source_key(doc) for doc in docs)
    print(f"Loaded {len(docs):,} docs from {len({doc.base_doc_id for doc in docs}):,} prompts: {dict(source_counts)}", flush=True)

    tokenizer = cast(Any, AutoTokenizer.from_pretrained(args.checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    model.eval()
    stats_model = _compile_model(cast(nn.Module, model), args=args, device=device)
    sae, sae_summary = _load_sae(args.sae_checkpoint, device=device)
    sae.eval()

    stats, examples = _source_stats_pass(
        model=stats_model,
        tokenizer=tokenizer,
        sae=sae,
        docs=docs,
        args=args,
        device=device,
    )
    rows = _initial_rows(
        stats,
        latent_dim=sae.latent_dim,
        ubiquitous_threshold=args.ubiquitous_token_rate_threshold,
    )
    candidate_pool = [row for row in rows if float(row["total_mass"]) > 0]
    candidate_pool.sort(
        key=lambda row: (
            float(row["preliminary_score"]),
            float(row["ai_human_log_odds"]),
            float(row["ai_mass_share"]),
            float(row["total_mass"]),
        ),
        reverse=True,
    )
    candidate_latents = [int(row["latent_id"]) for row in candidate_pool[: args.candidate_latents]]
    effects = _gradient_effect_pass(
        model=cast(nn.Module, model),
        tokenizer=tokenizer,
        sae=sae,
        docs=docs,
        candidate_latents=candidate_latents,
        args=args,
        device=device,
    )
    final_rows: list[dict[str, Any]] = []
    effect_by_latent = effects
    for row in rows:
        latent_id = int(row["latent_id"])
        effect = effect_by_latent.get(
            latent_id,
            {
                "first_order_ai_score_drop": 0.0,
                "first_order_human_score_drop": 0.0,
                "first_order_ai_minus_human_drop": 0.0,
                "first_order_ai_positive_doc_rate": 0.0,
                "first_order_human_positive_doc_rate": 0.0,
            },
        )
        combined = {
            **row,
            **effect,
        }
        combined["slop_candidate_score"] = (
            max(0.0, float(combined["ai_human_log_odds"]))
            * max(0.0, 1.0 - float(combined["active_token_rate"]))
            * max(0.0, float(combined["first_order_ai_minus_human_drop"]))
            * math.log1p(float(combined["total_mass"]))
        )
        combined["category_hint"] = _category_hint(combined)
        final_rows.append(combined)
    final_rows.sort(
        key=lambda row: (
            float(row["slop_candidate_score"]),
            float(row["first_order_ai_minus_human_drop"]),
            float(row["ai_human_log_odds"]),
            float(row["total_mass"]),
        ),
        reverse=True,
    )
    output_rows = [
        row for row in final_rows if float(row["total_mass"]) > 0
    ][: args.top_output_latents]
    for rank, row in enumerate(output_rows, start=1):
        row["rank"] = rank

    ranked_ids = {int(row["latent_id"]) for row in output_rows}
    example_rows: list[dict[str, Any]] = []
    for row in output_rows:
        latent_id = int(row["latent_id"])
        for example_rank, example in enumerate(examples.get(latent_id, []), start=1):
            example_rows.append(
                {
                    "rank": row["rank"],
                    "latent_id": latent_id,
                    "example_rank": example_rank,
                    **example,
                }
            )
    # Include examples for 2829 when present, even if it does not make this
    # stricter top list on the small sample.
    if 2829 not in ranked_ids and 2829 in examples:
        for example_rank, example in enumerate(examples[2829], start=1):
            example_rows.append(
                {
                    "rank": "",
                    "latent_id": 2829,
                    "example_rank": example_rank,
                    **example,
                }
            )

    summary = {
        "dataset_parquet": str(args.dataset_parquet),
        "checkpoint": str(args.checkpoint),
        "sae_checkpoint": str(args.sae_checkpoint),
        "sample_prompts": args.sample_prompts,
        "documents": len(docs),
        "source_counts": dict(sorted(source_counts.items())),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "candidate_latents": len(candidate_latents),
        "top_output_latents": len(output_rows),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "compile_model": bool(args.compile_model),
        "compile_mode": args.compile_mode,
        "sae_summary": sae_summary,
    }
    _write_csv(output_dir / "latent_slop_candidates.csv", output_rows)
    _write_csv(output_dir / "latent_slop_candidate_examples.csv", example_rows)
    _write_json(output_dir / "latent_slop_candidate_summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        f"Wrote slop latent candidates to {args.output_dir} "
        f"({summary['documents']} docs, {summary['candidate_latents']} causal candidates)",
        flush=True,
    )


if __name__ == "__main__":
    main()
