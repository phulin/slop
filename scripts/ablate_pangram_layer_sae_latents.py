from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from ablate_pangram_sae_latents import (
    _batched,
    _load_candidate_latents,
    _load_sampled_docs,
    _softmax_target,
)
from slop_sftdiv.cli.pangram_llama_sae import (
    LLAMA_BASE_ID,
    PANGRAM_ADAPTER_ID,
    _device,
    _encode_docs,
    load_pangram_model,
)
from slop_sftdiv.cli.phase4_batchtopk_sae import _load_sae, _write_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablate Pangram SAE latents inside a Llama decoder layer."
    )
    parser.add_argument(
        "--dataset-parquet",
        type=Path,
        default=Path(
            "artifacts/no_robots_batch_generations/"
            "categorical_openai_gemini_triplet_cap_640/data/train.parquet"
        ),
    )
    parser.add_argument("--sae-path", type=Path, required=True)
    parser.add_argument("--latents-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--activation-layer", type=int, required=True)
    parser.add_argument("--top-latents", type=int, default=50)
    parser.add_argument("--docs-per-source", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--use-dense-codes", action="store_true")
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


def _llama_layer(model: Any, layer_index: int) -> nn.Module:
    layers = getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None)
    layer_list = getattr(layers, "layers", None)
    if layer_list is None:
        raise TypeError("could not find Llama decoder layers on Pangram model")
    if not 0 <= layer_index < len(layer_list):
        raise ValueError(f"--activation-layer {layer_index} is out of range for {len(layer_list)} layers")
    return cast(nn.Module, layer_list[layer_index])


@torch.inference_mode()
def _baseline_logits(
    *,
    model: Any,
    tokenizer: Any,
    docs: list[Any],
    batch_size: int,
    max_length: int,
    device: torch.device,
    progress_every: int,
) -> tuple[torch.Tensor, list[int], list[str]]:
    logits_parts: list[torch.Tensor] = []
    labels: list[int] = []
    sources: list[str] = []
    for batch_index, batch_docs in enumerate(_batched(docs, batch_size), start=1):
        encoded = _encode_docs(
            tokenizer,
            batch_docs,
            max_length=max_length,
            device=device,
            pad_to_max_length=True,
        )
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        )
        logits_parts.append(cast(torch.Tensor, outputs.logits).detach().float().cpu())
        labels.extend(int(doc.label) for doc in batch_docs)
        sources.extend(str(doc.source) for doc in batch_docs)
        if progress_every > 0 and (batch_index == 1 or batch_index % progress_every == 0):
            print(
                f"Collected baseline logits for {min(batch_index * batch_size, len(docs)):,}/{len(docs):,} docs",
                flush=True,
            )
    return torch.cat(logits_parts, dim=0), labels, sources


def _target_class_from_baseline(logits: torch.Tensor, labels: list[int]) -> tuple[int, list[dict[str, Any]]]:
    label_tensor = torch.tensor(labels, dtype=torch.long)
    class_diffs: list[dict[str, Any]] = []
    for class_id in range(int(logits.shape[1])):
        ai_mean = float(logits[label_tensor == 1, class_id].mean())
        human_mean = float(logits[label_tensor == 0, class_id].mean())
        class_diffs.append(
            {
                "class_id": class_id,
                "mean_ai_logit": ai_mean,
                "mean_human_logit": human_mean,
                "ai_minus_human_logit": ai_mean - human_mean,
            }
        )
    return int(max(class_diffs, key=lambda row: float(row["ai_minus_human_logit"]))["class_id"]), class_diffs


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
    layer = _llama_layer(model, args.activation_layer)
    sae, _sae_summary = _load_sae(args.sae_path, device=device)
    sae.eval()
    decoder_weight = sae.decoder.weight.detach()

    baseline_logits, labels, sources = _baseline_logits(
        model=model,
        tokenizer=tokenizer,
        docs=docs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        progress_every=args.progress_every,
    )
    target_class, class_diffs = _target_class_from_baseline(baseline_logits, labels)
    baseline_target_logits = baseline_logits[:, target_class]
    baseline_target_probs = _softmax_target(baseline_logits, target_class)

    effect_sums: dict[int, float] = defaultdict(float)
    effect_abs_sums: dict[int, float] = defaultdict(float)
    prob_effect_sums: dict[int, float] = defaultdict(float)
    ai_effect_sums: dict[int, float] = defaultdict(float)
    human_effect_sums: dict[int, float] = defaultdict(float)
    active_doc_counts: Counter[int] = Counter()
    ai_active_doc_counts: Counter[int] = Counter()
    human_active_doc_counts: Counter[int] = Counter()
    active_token_counts: Counter[int] = Counter()

    for latent_offset, latent_id in enumerate(candidate_ids, start=1):
        doc_offset = 0
        for batch_index, batch_docs in enumerate(_batched(docs, args.batch_size), start=1):
            encoded = _encode_docs(
                tokenizer,
                batch_docs,
                max_length=args.max_length,
                device=device,
                pad_to_max_length=True,
            )
            attention_mask = encoded["attention_mask"].bool()
            active_doc_indexes: set[int] = set()
            active_token_count = 0

            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
                nonlocal active_doc_indexes, active_token_count
                hidden = output[0] if isinstance(output, tuple) else output
                if not isinstance(hidden, torch.Tensor):
                    return output
                hidden_float = hidden.float()
                token_vectors = hidden_float[attention_mask]
                if token_vectors.numel() == 0:
                    return output
                codes = sae.encode_dense(token_vectors) if args.use_dense_codes else sae.encode(token_vectors)
                latent_codes = codes[:, latent_id]
                active_positions = latent_codes > 0
                active_token_count += int(active_positions.sum().detach().cpu())
                if active_positions.any():
                    token_doc_indexes = torch.nonzero(attention_mask, as_tuple=False)[:, 0]
                    active_doc_indexes.update(
                        int(index)
                        for index in token_doc_indexes[active_positions].detach().cpu().tolist()
                    )
                contribution = latent_codes.unsqueeze(1) * decoder_weight[:, latent_id].unsqueeze(0)
                modified = hidden_float.clone()
                modified[attention_mask] = token_vectors - contribution
                modified = modified.to(dtype=hidden.dtype)
                if isinstance(output, tuple):
                    return (modified, *output[1:])
                return modified

            handle = layer.register_forward_hook(hook)
            try:
                outputs = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    return_dict=True,
                )
            finally:
                handle.remove()

            ablated_logits = cast(torch.Tensor, outputs.logits).detach().float().cpu()
            batch_size = len(batch_docs)
            baseline_batch_logits = baseline_target_logits[doc_offset : doc_offset + batch_size]
            baseline_batch_probs = baseline_target_probs[doc_offset : doc_offset + batch_size]
            effects = baseline_batch_logits - ablated_logits[:, target_class]
            prob_effects = baseline_batch_probs - _softmax_target(ablated_logits, target_class)
            effect_sums[latent_id] += float(effects.sum())
            effect_abs_sums[latent_id] += float(effects.abs().sum())
            prob_effect_sums[latent_id] += float(prob_effects.sum())
            active_token_counts[latent_id] += active_token_count
            for local_doc_index in active_doc_indexes:
                doc = batch_docs[local_doc_index]
                active_doc_counts[latent_id] += 1
                if int(doc.label) == 1:
                    ai_active_doc_counts[latent_id] += 1
                else:
                    human_active_doc_counts[latent_id] += 1
            for effect, doc in zip(effects.tolist(), batch_docs, strict=True):
                if int(doc.label) == 1:
                    ai_effect_sums[latent_id] += float(effect)
                else:
                    human_effect_sums[latent_id] += float(effect)
            doc_offset += batch_size

        if args.progress_every > 0 and (
            latent_offset == 1 or latent_offset % args.progress_every == 0
        ):
            print(
                f"Ablated layer-{args.activation_layer} latent {latent_id} "
                f"({latent_offset:,}/{len(candidate_ids):,}) on {len(docs):,} docs",
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
                "active_docs": int(active_doc_counts[latent_id]),
                "active_doc_share": active_doc_counts[latent_id] / max(1, len(docs)),
                "active_ai_docs": int(ai_active_doc_counts[latent_id]),
                "active_human_docs": int(human_active_doc_counts[latent_id]),
                "active_tokens": int(active_token_counts[latent_id]),
                "sample_docs": len(docs),
                "target_class": target_class,
                "activation_layer": args.activation_layer,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["mean_target_logit_drop"]),
            float(row["active_doc_share"]),
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
        "activation_layer": args.activation_layer,
        "sample_docs": len(docs),
        "docs_per_source": args.docs_per_source,
        "source_counts": dict(sorted(source_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "top_latents": args.top_latents,
        "target_class": target_class,
        "class_diffs": class_diffs,
        "baseline_target_logit_mean": float(baseline_target_logits.mean()),
        "baseline_target_probability_mean": float(baseline_target_probs.mean()),
        "use_dense_codes": bool(args.use_dense_codes),
        "intervention": "decoder_layer_all_nonpad_tokens_subtract_latent_decoder_contribution",
    }
    _write_csv(args.output_dir / "pangram_layer_sae_causal_ablation_latents.csv", rows)
    (args.output_dir / "pangram_layer_sae_causal_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        "Wrote layer SAE causal ablation outputs to "
        f"{args.output_dir} (layer={summary['activation_layer']}, "
        f"target_class={summary['target_class']}, sample_docs={summary['sample_docs']})"
    )


if __name__ == "__main__":
    main()
