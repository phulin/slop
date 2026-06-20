from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from torch import nn

from slop_sftdiv.cli.pangram_llama_sae import (
    LLAMA_BASE_ID,
    PANGRAM_ADAPTER_ID,
    _device,
    _encode_docs,
    load_pangram_model,
)
from slop_sftdiv.cli.phase4_batchtopk_sae import SAEDoc, _load_sae, _write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Causally ablate top Pangram Llama SAE latents on sampled docs."
    )
    parser.add_argument(
        "--dataset-parquet",
        type=Path,
        default=Path(
            "artifacts/no_robots_batch_generations/"
            "categorical_openai_gemini_triplet_cap_640/data/train.parquet"
        ),
    )
    parser.add_argument(
        "--sae-path",
        type=Path,
        default=Path(
            "/tmp/slop_pangram_llama_sae_full_cap640_k64_e50/"
            "pangram_llama_batchtopk_sae.pt"
        ),
    )
    parser.add_argument(
        "--latents-csv",
        type=Path,
        default=Path(
            "/tmp/slop_pangram_llama_sae_full_cap640_k64_e50/"
            "pangram_llama_sae_latents.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/slop_pangram_llama_sae_full_cap640_k64_e50/causal_ablation_top50_sample"),
    )
    parser.add_argument("--top-latents", type=int, default=50)
    parser.add_argument("--docs-per-source", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--base-model", default=LLAMA_BASE_ID)
    parser.add_argument("--adapter-model", default=PANGRAM_ADAPTER_ID)
    parser.add_argument(
        "--local-model-dir",
        type=Path,
        default=Path("artifacts/models/pangram_editlens_Llama-3.2-3B_unsloth_base"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--adapter-revision", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--no-hf-token", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--progress-every", type=int, default=5)
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


def _load_sampled_docs(path: Path, *, docs_per_source: int, seed: int) -> list[SAEDoc]:
    frame = pd.read_parquet(path)
    required = {"role", "model", "text"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame = frame[frame["role"].isin(["human_continuation", "llm_continuation"])].copy()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["model"] = frame["model"].fillna("").astype(str).str.strip()
    if "record_id" not in frame.columns:
        frame["record_id"] = [str(index) for index in range(len(frame))]

    docs: list[SAEDoc] = []
    rng = random.Random(seed)
    for source, source_frame in frame.groupby("model", sort=True):
        rows = list(source_frame.itertuples(index=False))
        rng.shuffle(rows)
        loaded = 0
        for row in rows:
            if loaded >= docs_per_source:
                break
            text = _string_or_empty(getattr(row, "text"))
            role = _string_or_empty(getattr(row, "role"))
            if not text:
                continue
            label = 0 if role == "human_continuation" else 1
            docs.append(
                SAEDoc(
                    source=_string_or_empty(source),
                    doc_id=_string_or_empty(getattr(row, "record_id", loaded)),
                    text=text,
                    label=label,
                )
            )
            loaded += 1
    random.Random(seed).shuffle(docs)
    return docs


def _load_candidate_latents(path: Path, *, top_latents: int) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = rows[:top_latents]
    if not rows:
        raise ValueError(f"no candidate latents found in {path}")
    return rows


def _score_head(model: Any) -> nn.Module:
    candidates = [
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]
    for candidate in candidates:
        score = getattr(candidate, "score", None)
        if isinstance(score, nn.Module):
            return score
    raise TypeError("could not find Pangram sequence-classification score head")


def _module_dtype(module: nn.Module) -> torch.dtype:
    for parameter in module.parameters():
        return parameter.dtype
    return torch.float32


def _batched(items: list[SAEDoc], batch_size: int) -> list[list[SAEDoc]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _softmax_target(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)[:, target_class]


def _last_non_pad_indexes(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    masked_positions = attention_mask.long() * positions.unsqueeze(0)
    return masked_positions.max(dim=1).values


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    docs = _load_sampled_docs(
        args.dataset_parquet,
        docs_per_source=args.docs_per_source,
        seed=args.seed,
    )
    candidate_rows = _load_candidate_latents(args.latents_csv, top_latents=args.top_latents)
    candidate_ids = [int(row["latent_id"]) for row in candidate_rows]
    candidate_by_id = {int(row["latent_id"]): row for row in candidate_rows}

    model, tokenizer = load_pangram_model(args, device=device)
    score_head = _score_head(model)
    score_dtype = _module_dtype(score_head)
    sae, _sae_summary = _load_sae(args.sae_path, device=device)
    sae.eval()

    baseline_logits_parts: list[torch.Tensor] = []
    model_logits_parts: list[torch.Tensor] = []
    labels: list[int] = []
    sources: list[str] = []
    pooled_batches: list[torch.Tensor] = []
    sparse_batches: list[torch.Tensor] = []

    for batch_index, batch_docs in enumerate(_batched(docs, args.batch_size), start=1):
        encoded = _encode_docs(
            tokenizer,
            batch_docs,
            max_length=args.max_length,
            device=device,
            pad_to_max_length=True,
        )
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Pangram/Llama forward pass did not return hidden_states")
        hidden = cast(torch.Tensor, hidden_states[-1])
        last_token_indexes = _last_non_pad_indexes(encoded["attention_mask"])
        batch_indexes = torch.arange(hidden.shape[0], device=device)
        pooled = hidden[batch_indexes, last_token_indexes].float()
        sparse = sae.encode(pooled).detach()
        baseline_logits = score_head(pooled.to(dtype=score_dtype)).float()
        baseline_logits_parts.append(baseline_logits.detach().cpu())
        model_logits_parts.append(cast(torch.Tensor, outputs.logits).detach().float().cpu())
        pooled_batches.append(pooled.detach())
        sparse_batches.append(sparse)
        labels.extend(int(doc.label) for doc in batch_docs)
        sources.extend(doc.source for doc in batch_docs)
        if args.progress_every > 0 and (batch_index == 1 or batch_index % args.progress_every == 0):
            print(
                f"Collected baseline logits/pooled activations for "
                f"{min(batch_index * args.batch_size, len(docs)):,}/{len(docs):,} docs",
                flush=True,
            )

    baseline_logits = torch.cat(baseline_logits_parts, dim=0)
    model_logits = torch.cat(model_logits_parts, dim=0)
    max_model_logit_abs_diff = float((baseline_logits - model_logits).abs().max().item())
    mean_model_logit_abs_diff = float((baseline_logits - model_logits).abs().mean().item())
    label_tensor = torch.tensor(labels, dtype=torch.long)
    class_diffs = []
    for class_id in range(int(baseline_logits.shape[1])):
        ai_mean = float(baseline_logits[label_tensor == 1, class_id].mean())
        human_mean = float(baseline_logits[label_tensor == 0, class_id].mean())
        class_diffs.append(
            {
                "class_id": class_id,
                "mean_ai_logit": ai_mean,
                "mean_human_logit": human_mean,
                "ai_minus_human_logit": ai_mean - human_mean,
            }
        )
    target_class = int(max(class_diffs, key=lambda row: float(row["ai_minus_human_logit"]))["class_id"])

    baseline_target_logits = baseline_logits[:, target_class]
    baseline_target_probs = _softmax_target(baseline_logits, target_class)
    decoder_weight = sae.decoder.weight.detach()

    effect_sums: dict[int, float] = defaultdict(float)
    effect_abs_sums: dict[int, float] = defaultdict(float)
    prob_effect_sums: dict[int, float] = defaultdict(float)
    ai_effect_sums: dict[int, float] = defaultdict(float)
    human_effect_sums: dict[int, float] = defaultdict(float)
    active_counts: Counter[int] = Counter()
    ai_active_counts: Counter[int] = Counter()
    human_active_counts: Counter[int] = Counter()
    doc_count = 0

    offset = 0
    for batch_index, (pooled, sparse) in enumerate(zip(pooled_batches, sparse_batches, strict=True), start=1):
        batch_size = int(pooled.shape[0])
        batch_labels = labels[offset : offset + batch_size]
        baseline_batch_logits = baseline_target_logits[offset : offset + batch_size].to(device)
        baseline_batch_probs = baseline_target_probs[offset : offset + batch_size].to(device)
        for latent_id in candidate_ids:
            codes = sparse[:, latent_id]
            active = codes > 0
            active_counts[latent_id] += int(active.sum().detach().cpu())
            ai_active_counts[latent_id] += sum(
                1 for value, label in zip(active.detach().cpu().tolist(), batch_labels, strict=True) if value and label == 1
            )
            human_active_counts[latent_id] += sum(
                1 for value, label in zip(active.detach().cpu().tolist(), batch_labels, strict=True) if value and label == 0
            )
            contribution = codes.unsqueeze(1) * decoder_weight[:, latent_id].unsqueeze(0)
            ablated_pooled = pooled - contribution
            ablated_logits = score_head(ablated_pooled.to(dtype=score_dtype)).float()
            effects = baseline_batch_logits - ablated_logits[:, target_class]
            prob_effects = baseline_batch_probs - _softmax_target(ablated_logits, target_class)
            effect_sums[latent_id] += float(effects.detach().sum().cpu())
            effect_abs_sums[latent_id] += float(effects.detach().abs().sum().cpu())
            prob_effect_sums[latent_id] += float(prob_effects.detach().sum().cpu())
            for effect, label in zip(effects.detach().cpu().tolist(), batch_labels, strict=True):
                if int(label) == 1:
                    ai_effect_sums[latent_id] += float(effect)
                else:
                    human_effect_sums[latent_id] += float(effect)
        offset += batch_size
        doc_count += batch_size
        if args.progress_every > 0 and (batch_index == 1 or batch_index % args.progress_every == 0):
            print(
                f"Ablated {len(candidate_ids)} latents on {doc_count:,}/{len(docs):,} docs",
                flush=True,
            )

    label_counts = Counter(str(label) for label in labels)
    source_counts = Counter(sources)
    ai_docs = max(1, int(label_counts["1"]))
    human_docs = max(1, int(label_counts["0"]))
    rows: list[dict[str, Any]] = []
    for latent_id in candidate_ids:
        association = candidate_by_id[latent_id]
        rows.append(
            {
                "latent_id": latent_id,
                "association_rank": association.get("rank", ""),
                "association_ai_human_log_odds": association.get("ai_human_log_odds", ""),
                "association_ai_mass_share": association.get("ai_mass_share", ""),
                "association_total_mass": association.get("total_mass", ""),
                "mean_target_logit_drop": effect_sums[latent_id] / max(1, len(docs)),
                "mean_abs_target_logit_change": effect_abs_sums[latent_id] / max(1, len(docs)),
                "mean_target_probability_drop": prob_effect_sums[latent_id] / max(1, len(docs)),
                "mean_ai_doc_logit_drop": ai_effect_sums[latent_id] / ai_docs,
                "mean_human_doc_logit_drop": human_effect_sums[latent_id] / human_docs,
                "pooled_active_docs": int(active_counts[latent_id]),
                "pooled_active_doc_share": active_counts[latent_id] / max(1, len(docs)),
                "pooled_active_ai_docs": int(ai_active_counts[latent_id]),
                "pooled_active_human_docs": int(human_active_counts[latent_id]),
                "sample_docs": len(docs),
                "target_class": target_class,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["mean_target_logit_drop"]),
            float(row["pooled_active_doc_share"]),
            float(row["mean_abs_target_logit_change"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["causal_rank"] = rank

    summary = {
        "dataset_parquet": str(args.dataset_parquet),
        "sae_path": str(args.sae_path),
        "latents_csv": str(args.latents_csv),
        "sample_docs": len(docs),
        "docs_per_source": args.docs_per_source,
        "source_counts": dict(sorted(source_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "top_latents": args.top_latents,
        "target_class": target_class,
        "class_diffs": class_diffs,
        "baseline_target_logit_mean": float(baseline_target_logits.mean()),
        "baseline_target_probability_mean": float(baseline_target_probs.mean()),
        "max_model_logit_abs_diff": max_model_logit_abs_diff,
        "mean_model_logit_abs_diff": mean_model_logit_abs_diff,
        "intervention": "pooled_final_token_subtract_sparse_latent_decoder_contribution",
    }
    _write_csv(args.output_dir / "pangram_llama_sae_causal_ablation_latents.csv", rows)
    (args.output_dir / "pangram_llama_sae_causal_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        "Wrote Pangram SAE causal ablation outputs to "
        f"{args.output_dir} (target_class={summary['target_class']}, "
        f"sample_docs={summary['sample_docs']})"
    )


if __name__ == "__main__":
    main()
