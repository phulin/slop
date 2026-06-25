from __future__ import annotations

import argparse
import heapq
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from rerank_latent_explorer_by_causal_ablation import (
    ExplorerDoc,
    _as_sae_doc,
    _baseline_logits,
    _batched,
    _llama_layer,
    _load_payloads,
    _softmax_target,
    _target_class_from_docs,
    _write_csv,
    _write_json,
)
from slop_sftdiv.cli.pangram_llama_sae import (
    LLAMA_BASE_ID,
    PANGRAM_ADAPTER_ID,
    _device,
    _encode_docs,
    load_pangram_model,
)
from slop_sftdiv.cli.phase4_batchtopk_sae import _load_sae


@dataclass(frozen=True)
class LinearizedDocImpact:
    doc_id: str
    estimated_logit_drop: float
    estimated_abs_logit_change: float
    turn_id: str
    human_estimated_logit_drop: float
    mean_ai_estimated_logit_drop: float
    peak_activation: float = 0.0
    peak_token_index: int | str = ""
    candidate_source: str = "linearized_impact"
    human_peak_activation: float = 0.0
    max_ai_peak_activation: float = 0.0
    activation_difference: float = 0.0
    paired_activation_effect: float = 0.0
    human_activation_score: float = 0.0
    mean_ai_activation_score: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select latent-explorer candidates with a gradient-linearized causal "
            "ablation estimate, then exact-ablate the top prompt groups."
        )
    )
    parser.add_argument("--browser-index", type=Path, required=True)
    parser.add_argument("--doc-tokens-parquet", type=Path, required=True)
    parser.add_argument(
        "--sparse-topk-parquet",
        type=Path,
        default=None,
        help="Browser-aligned sparse top-k parquet, required for --candidate-doc-source max_activation.",
    )
    parser.add_argument("--sae-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-browser-index", type=Path, default=None)
    parser.add_argument("--activation-layer", type=int, default=20)
    parser.add_argument("--linearized-top-latents", type=int, default=200)
    parser.add_argument("--linearized-top-generations-per-latent", type=int, default=50)
    parser.add_argument(
        "--latent-selection-source",
        choices=["linearized", "max_ai_human_activation_difference", "paired_activation_effect_size"],
        default="linearized",
        help="How to select candidate latents before exact prompt-level ablation.",
    )
    parser.add_argument(
        "--candidate-doc-source",
        choices=[
            "linearized_impact",
            "max_activation",
            "max_ai_human_activation_difference",
            "paired_activation_effect_size",
        ],
        default="linearized_impact",
        help=(
            "How to choose prompt groups within each selected latent. Latent selection always uses "
            "the linearized global impact ranking."
        ),
    )
    parser.add_argument(
        "--activation-candidates-per-latent",
        type=int,
        default=100,
        help="Number of peak-activation documents to keep per latent when --candidate-doc-source max_activation.",
    )
    parser.add_argument(
        "--sparse-scan-batch-size",
        type=int,
        default=65536,
        help="Rows per batch while scanning the sparse top-k parquet for peak activations.",
    )
    parser.add_argument("--examples-per-latent", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--ablation-batch-size",
        type=int,
        default=None,
        help="Batch size for exact ablation passes. Defaults to --batch-size.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--baseline-logits-cache",
        type=Path,
        default=None,
        help=(
            "Optional torch cache for baseline logits. Reused only when doc order, max length, "
            "class count, and dataset paths match."
        ),
    )
    parser.add_argument(
        "--refresh-baseline-logits-cache",
        action="store_true",
        help="Ignore any existing --baseline-logits-cache and overwrite it after recomputing.",
    )
    parser.add_argument(
        "--browser-top-k",
        type=int,
        default=64,
        help="Per-token dense-code top-k activation definition used by the browser sparse parquet.",
    )
    parser.add_argument("--max-gradient-docs", type=int, default=None)
    parser.add_argument(
        "--effect-target",
        choices=["target_class", "ai_human_margin"],
        default="target_class",
        help="Exact ablation target: single target-class logit or AI-vs-human classifier margin.",
    )
    parser.add_argument("--paired-min-active-prompts", type=int, default=100)
    parser.add_argument("--paired-min-sign-consistency", type=float, default=0.65)
    parser.add_argument("--candidate-score", choices=["abs", "signed"], default="abs")
    parser.add_argument(
        "--final-latent-rank-score",
        choices=[
            "linearized_abs_mean",
            "linearized_signed_mean",
            "exact_max_abs",
            "exact_mean_abs",
            "activation_difference_abs_mean",
            "activation_difference_max_abs",
            "paired_activation_effect_size",
            "paired_causal_consistent",
        ],
        default="linearized_abs_mean",
        help=(
            "Primary ordering for the final explorer latent list. "
            "The default ranks by average absolute linearized score impact over all scored docs."
        ),
    )
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
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def _restore_grad_flags(module: nn.Module, flags: list[bool]) -> None:
    for parameter, flag in zip(module.parameters(), flags, strict=True):
        parameter.requires_grad_(flag)


def _snippet(text: str, limit: int = 260) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "..."


def _activation_mask(encoded: dict[str, Any]) -> torch.Tensor:
    mask = encoded["attention_mask"].bool()
    special = encoded.get("special_tokens_mask")
    if isinstance(special, torch.Tensor):
        mask = mask & ~special.bool()
    return mask


def _class_groups_from_docs(logits: torch.Tensor, docs: list[ExplorerDoc]) -> tuple[int, list[int], list[dict[str, Any]]]:
    labels = torch.tensor([doc.label for doc in docs], dtype=torch.long)
    rows: list[dict[str, Any]] = []
    for class_id in range(int(logits.shape[1])):
        ai_mean = float(logits[labels == 1, class_id].mean())
        human_mean = float(logits[labels == 0, class_id].mean())
        rows.append(
            {
                "class_id": class_id,
                "mean_ai_logit": ai_mean,
                "mean_human_logit": human_mean,
                "ai_minus_human_logit": ai_mean - human_mean,
                "human_minus_ai_logit": human_mean - ai_mean,
            }
        )
    human_class = int(max(rows, key=lambda row: float(row["human_minus_ai_logit"]))["class_id"])
    ai_classes = [class_id for class_id in range(int(logits.shape[1])) if class_id != human_class]
    return human_class, ai_classes, rows


def _ai_human_margin(logits: torch.Tensor, *, human_class: int, ai_classes: list[int]) -> torch.Tensor:
    logits = logits.float()
    ai_index = torch.tensor(ai_classes, dtype=torch.long, device=logits.device)
    return torch.logsumexp(logits.index_select(dim=-1, index=ai_index), dim=-1) - logits[:, int(human_class)]


def _effect_values(
    *,
    baseline_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    target_class: int,
    human_class: int,
    ai_classes: list[int],
    effect_target: str,
) -> tuple[float, float]:
    if effect_target == "ai_human_margin":
        baseline_value = float(
            _ai_human_margin(
                baseline_logits.unsqueeze(0),
                human_class=int(human_class),
                ai_classes=ai_classes,
            )[0]
        )
        ablated_value = float(
            _ai_human_margin(
                ablated_logits.unsqueeze(0),
                human_class=int(human_class),
                ai_classes=ai_classes,
            )[0]
        )
        return baseline_value - ablated_value, 0.0
    baseline_prob = float(_softmax_target(baseline_logits.unsqueeze(0), target_class)[0])
    ablated_prob = float(_softmax_target(ablated_logits.unsqueeze(0), target_class)[0])
    return float(baseline_logits[target_class] - ablated_logits[target_class]), baseline_prob - ablated_prob


def _browser_sparse_codes(sae: Any, token_vectors: torch.Tensor, *, top_k: int) -> torch.Tensor:
    dense_codes = sae.encode_dense(token_vectors.float()).float()
    keep = min(int(top_k), int(dense_codes.shape[1]))
    values, indexes = torch.topk(dense_codes, k=keep, dim=1)
    sparse_codes = torch.zeros_like(dense_codes)
    sparse_codes.scatter_(1, indexes, values)
    return sparse_codes


def _browser_latent_codes(sae: Any, token_vectors: torch.Tensor, *, latent_id: int, top_k: int) -> torch.Tensor:
    dense_codes = sae.encode_dense(token_vectors.float()).float()
    keep = min(int(top_k), int(dense_codes.shape[1]))
    values, indexes = torch.topk(dense_codes, k=keep, dim=1)
    matches = indexes == int(latent_id)
    if not matches.any():
        return torch.zeros(dense_codes.shape[0], dtype=torch.float32, device=dense_codes.device)
    matched_values = torch.zeros(dense_codes.shape[0], dtype=torch.float32, device=dense_codes.device)
    matched_values[matches.any(dim=1)] = values[matches]
    return matched_values


@torch.inference_mode()
def _ablate_latent_on_docs_browser_aligned(
    *,
    model: Any,
    tokenizer: Any,
    layer: nn.Module,
    sae: Any,
    latent_id: int,
    docs: list[ExplorerDoc],
    baseline_logits_by_doc: dict[str, torch.Tensor],
    target_class: int,
    human_class: int,
    ai_classes: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    decoder_weight = sae.decoder.weight.detach()
    results: dict[str, dict[str, Any]] = {}
    ablation_batch_size = int(args.ablation_batch_size or args.batch_size)
    for batch_docs in _batched(docs, ablation_batch_size):
        encoded = _encode_docs(
            tokenizer,
            [_as_sae_doc(doc) for doc in batch_docs],
            max_length=int(args.max_length),
            device=device,
            pad_to_max_length=True,
        )
        token_mask = _activation_mask(encoded)
        active_doc_indexes: set[int] = set()
        active_token_counts = [0 for _doc in batch_docs]

        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                return output
            hidden_float = hidden.float()
            token_vectors = hidden_float[token_mask]
            if token_vectors.numel() == 0:
                return output
            latent_codes = _browser_latent_codes(
                sae,
                token_vectors,
                latent_id=int(latent_id),
                top_k=int(args.browser_top_k),
            )
            active_positions = latent_codes > 0
            if not active_positions.any():
                return output
            token_doc_indexes = torch.nonzero(token_mask, as_tuple=False)[:, 0]
            for index in token_doc_indexes[active_positions].detach().cpu().tolist():
                active_doc_indexes.add(int(index))
                active_token_counts[int(index)] += 1
            contribution = latent_codes.unsqueeze(1) * decoder_weight[:, latent_id].unsqueeze(0)
            modified = hidden_float.clone()
            modified[token_mask] = token_vectors - contribution
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
        for doc_index, doc in enumerate(batch_docs):
            baseline_logits = baseline_logits_by_doc[doc.doc_id]
            if doc_index not in active_doc_indexes:
                logit_drop = 0.0
                prob_drop = 0.0
            else:
                logit_drop, prob_drop = _effect_values(
                    baseline_logits=baseline_logits,
                    ablated_logits=ablated_logits[doc_index],
                    target_class=int(target_class),
                    human_class=int(human_class),
                    ai_classes=ai_classes,
                    effect_target=str(args.effect_target),
                )
            results[doc.doc_id] = {
                "causal_logit_drop": logit_drop,
                "causal_abs_logit_change": abs(logit_drop),
                "causal_probability_drop": prob_drop,
                "causal_abs_probability_change": abs(prob_drop),
                "causal_active_tokens": int(active_token_counts[doc_index]),
                "causal_active": doc_index in active_doc_indexes,
                "causal_effect_target": str(args.effect_target),
            }
    return results


def _prompt_contrast_from_effects(
    *,
    turn_id: str,
    sibling_doc_ids: list[str],
    documents: dict[str, Any],
    effects: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    human_values: list[float] = []
    human_doc_id = ""
    ai_values: list[float] = []
    ai_doc_ids: list[str] = []
    representative_doc_id = ""
    representative_abs_effect = -1.0
    for doc_id in sibling_doc_ids:
        if doc_id not in effects:
            continue
        value = float(effects[doc_id]["causal_logit_drop"])
        abs_value = abs(value)
        if bool(effects[doc_id].get("causal_active")) and abs_value > representative_abs_effect:
            representative_doc_id = doc_id
            representative_abs_effect = abs_value
        model = str(documents.get(doc_id, {}).get("model", ""))
        if model == "human":
            human_values.append(value)
            human_doc_id = doc_id
        else:
            ai_values.append(value)
            ai_doc_ids.append(doc_id)
    if not human_values or not ai_values:
        return None
    human = sum(human_values) / len(human_values)
    mean_ai = sum(ai_values) / len(ai_values)
    contrast = human - mean_ai
    return {
        "turn_id": turn_id,
        "doc_id": representative_doc_id or human_doc_id or ai_doc_ids[0],
        "human_doc_id": human_doc_id,
        "ai_doc_ids": "|".join(ai_doc_ids),
        "prompt_contrast_logit_drop": contrast,
        "prompt_abs_contrast_logit_change": abs(contrast),
        "prompt_human_logit_drop": human,
        "prompt_mean_ai_logit_drop": mean_ai,
        "prompt_ai_docs": len(ai_values),
    }


def _linearized_doc_impacts(
    *,
    model: Any,
    tokenizer: Any,
    layer: nn.Module,
    sae: Any,
    docs: list[ExplorerDoc],
    documents: dict[str, Any],
    prompt_groups: dict[str, list[str]],
    target_class: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[int, list[LinearizedDocImpact]]]:
    latent_dim = int(sae.latent_dim)
    decoder_weight = sae.decoder.weight.detach().float()
    top_docs_per_latent = int(args.linearized_top_generations_per_latent)
    turn_ids = sorted(
        {
            str(documents.get(doc.doc_id, {}).get("turn_id", ""))
            for doc in docs
            if str(documents.get(doc.doc_id, {}).get("turn_id", ""))
        }
    )
    turn_index_by_id = {turn_id: index for index, turn_id in enumerate(turn_ids)}
    human_effects = torch.zeros(len(turn_ids), latent_dim, dtype=torch.float32)
    ai_effect_sums = torch.zeros(len(turn_ids), latent_dim, dtype=torch.float32)
    ai_counts = torch.zeros(len(turn_ids), dtype=torch.float32)
    human_doc_by_turn: dict[str, str] = {}
    fallback_doc_by_turn: dict[str, str] = {}

    model_flags = [parameter.requires_grad for parameter in model.parameters()]
    sae_flags = [parameter.requires_grad for parameter in sae.parameters()]
    model.requires_grad_(False)
    sae.requires_grad_(False)
    batches = _batched(docs, int(args.batch_size))
    try:
        for batch_index, batch_docs in enumerate(batches, start=1):
            captured: dict[str, torch.Tensor] = {}
            encoded = _encode_docs(
                tokenizer,
                [_as_sae_doc(doc) for doc in batch_docs],
                max_length=int(args.max_length),
                device=device,
                pad_to_max_length=True,
            )
            token_mask = _activation_mask(encoded)

            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                if not isinstance(hidden, torch.Tensor):
                    return output
                hidden_leaf = hidden.detach().float().requires_grad_(True)
                captured["hidden"] = hidden_leaf
                replacement = hidden_leaf.to(dtype=hidden.dtype)
                if isinstance(output, tuple):
                    return (replacement, *output[1:])
                return replacement

            handle = layer.register_forward_hook(hook)
            try:
                outputs = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    return_dict=True,
                )
                score = cast(torch.Tensor, outputs.logits)[:, target_class].sum()
                score.backward()
            finally:
                handle.remove()

            hidden = captured.get("hidden")
            if hidden is None or hidden.grad is None:
                raise RuntimeError("failed to capture hooked hidden-state gradients")
            token_doc_indexes = torch.nonzero(token_mask, as_tuple=False)[:, 0]
            token_vectors = hidden.detach()[token_mask]
            token_grads = hidden.grad.detach()[token_mask].float()
            with torch.no_grad():
                codes = _browser_sparse_codes(
                    sae,
                    token_vectors.float(),
                    top_k=int(args.browser_top_k),
                )
                gradient_projection = token_grads @ decoder_weight
                token_effects = codes * gradient_projection
                doc_effects = torch.zeros(
                    len(batch_docs),
                    latent_dim,
                    dtype=torch.float32,
                    device=device,
                )
                doc_effects.index_add_(0, token_doc_indexes, token_effects)
                doc_effects_cpu = doc_effects.detach().cpu()
            for doc_position, doc in enumerate(batch_docs):
                turn_id = str(documents.get(doc.doc_id, {}).get("turn_id", ""))
                if not turn_id or turn_id not in turn_index_by_id:
                    continue
                turn_index = turn_index_by_id[turn_id]
                fallback_doc_by_turn.setdefault(turn_id, doc.doc_id)
                if str(documents.get(doc.doc_id, {}).get("model", doc.source)) == "human":
                    human_effects[turn_index] += doc_effects_cpu[doc_position]
                    human_doc_by_turn[turn_id] = doc.doc_id
                else:
                    ai_effect_sums[turn_index] += doc_effects_cpu[doc_position]
                    ai_counts[turn_index] += 1

            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if args.progress_every > 0 and (
                batch_index == 1 or batch_index % int(args.progress_every) == 0
            ):
                print(
                    f"Linearized causal pass {min(batch_index * int(args.batch_size), len(docs)):,}/"
                    f"{len(docs):,} docs",
                    flush=True,
                )
    finally:
        _restore_grad_flags(model, model_flags)
        _restore_grad_flags(sae, sae_flags)

    valid_prompt_mask = ai_counts > 0
    ai_means = ai_effect_sums / ai_counts.clamp_min(1).unsqueeze(1)
    prompt_contrasts = human_effects - ai_means
    prompt_contrasts = prompt_contrasts[valid_prompt_mask]
    valid_turn_ids = [
        turn_id
        for turn_id in turn_ids
        if bool(valid_prompt_mask[turn_index_by_id[turn_id]])
    ]
    denom = max(1, len(valid_turn_ids))
    signed_sums = prompt_contrasts.sum(dim=0, dtype=torch.float64)
    abs_sums = prompt_contrasts.abs().sum(dim=0, dtype=torch.float64)
    active_prompt_counts = (prompt_contrasts.abs() > 0).sum(dim=0)
    values, prompt_positions = torch.topk(
        prompt_contrasts.abs(),
        k=min(top_docs_per_latent, prompt_contrasts.shape[0]),
        dim=0,
    )
    rows: list[dict[str, Any]] = []
    top_doc_rows: dict[int, list[LinearizedDocImpact]] = {}
    for latent_id in range(latent_dim):
        mean_signed = float(signed_sums[latent_id] / denom)
        mean_abs = float(abs_sums[latent_id] / denom)
        rows.append(
            {
                "latent_id": latent_id,
                "linearized_mean_estimated_logit_drop": mean_signed,
                "linearized_mean_abs_estimated_logit_change": mean_abs,
                "linearized_mean_prompt_contrast_logit_drop": mean_signed,
                "linearized_mean_abs_prompt_contrast_logit_change": mean_abs,
                "linearized_active_docs": int(active_prompt_counts[latent_id]),
                "linearized_active_doc_share": float(active_prompt_counts[latent_id] / denom),
                "linearized_active_prompts": int(active_prompt_counts[latent_id]),
                "linearized_active_prompt_share": float(active_prompt_counts[latent_id] / denom),
                "linearized_scored_docs": len(docs),
                "linearized_scored_prompts": denom,
                "target_class": target_class,
                "activation_layer": int(args.activation_layer),
            }
        )
        estimates: list[LinearizedDocImpact] = []
        for row_index in range(values.shape[0]):
            abs_value = float(values[row_index, latent_id])
            if abs_value <= 0:
                continue
            prompt_index = int(prompt_positions[row_index, latent_id])
            turn_id = valid_turn_ids[prompt_index]
            source_turn_index = turn_index_by_id[turn_id]
            signed_value = float(
                human_effects[source_turn_index, latent_id]
                - ai_means[source_turn_index, latent_id]
            )
            doc_id = human_doc_by_turn.get(turn_id) or fallback_doc_by_turn.get(turn_id, "")
            estimates.append(
                LinearizedDocImpact(
                    doc_id=doc_id,
                    turn_id=turn_id,
                    estimated_logit_drop=signed_value,
                    estimated_abs_logit_change=abs_value,
                    human_estimated_logit_drop=float(human_effects[source_turn_index, latent_id]),
                    mean_ai_estimated_logit_drop=float(ai_means[source_turn_index, latent_id]),
                )
            )
        top_doc_rows[latent_id] = estimates
    return rows, top_doc_rows


def _rank_linearized_rows(rows: list[dict[str, Any]], *, candidate_score: str) -> list[dict[str, Any]]:
    if candidate_score == "signed":
        key = lambda row: (
            float(row["linearized_mean_estimated_logit_drop"]),
            float(row["linearized_mean_abs_estimated_logit_change"]),
            float(row["linearized_active_doc_share"]),
        )
    else:
        key = lambda row: (
            float(row["linearized_mean_abs_estimated_logit_change"]),
            abs(float(row["linearized_mean_estimated_logit_drop"])),
            float(row["linearized_active_doc_share"]),
        )
    ranked = sorted(rows, key=key, reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["linearized_rank"] = rank
    return ranked


def _baseline_cache_metadata(args: argparse.Namespace, docs: list[ExplorerDoc]) -> dict[str, Any]:
    return {
        "doc_ids": [doc.doc_id for doc in docs],
        "max_length": int(args.max_length),
        "base_model": str(args.base_model),
        "adapter_model": str(args.adapter_model),
        "local_model_dir": str(args.local_model_dir),
        "doc_tokens_parquet": str(args.doc_tokens_parquet),
    }


def _metadata_mismatch(expected: dict[str, Any], observed: dict[str, Any]) -> str | None:
    for key, expected_value in expected.items():
        observed_value = observed.get(key)
        if observed_value != expected_value:
            if key == "doc_ids" and isinstance(observed_value, list):
                return f"doc_ids differ: expected {len(expected_value):,}, found {len(observed_value):,}"
            return f"{key} differs: expected {expected_value!r}, found {observed_value!r}"
    return None


def _baseline_logits_cached(
    *,
    model: Any,
    tokenizer: Any,
    docs: list[ExplorerDoc],
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    cache_path = args.baseline_logits_cache
    expected_metadata = _baseline_cache_metadata(args, docs)
    if cache_path is not None and cache_path.exists() and not bool(args.refresh_baseline_logits_cache):
        try:
            payload = torch.load(cache_path, map_location="cpu")
            if not isinstance(payload, dict):
                raise TypeError("cache payload is not a dict")
            mismatch = _metadata_mismatch(expected_metadata, dict(payload.get("metadata", {})))
            logits = payload.get("logits")
            if mismatch is None and isinstance(logits, torch.Tensor) and logits.shape[0] == len(docs):
                print(f"Loaded baseline logits cache from {cache_path}", flush=True)
                return logits.detach().float().cpu()
            reason = mismatch or "logits tensor missing or wrong first dimension"
            print(f"Ignoring baseline logits cache at {cache_path}: {reason}", flush=True)
        except Exception as exc:
            print(f"Ignoring unreadable baseline logits cache at {cache_path}: {exc}", flush=True)

    logits = _baseline_logits(
        model=model,
        tokenizer=tokenizer,
        docs=docs,
        args=args,
        device=device,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
        torch.save(
            {
                "metadata": {
                    **expected_metadata,
                    "shape": list(logits.shape),
                    "dtype": str(logits.dtype),
                },
                "logits": logits.detach().float().cpu(),
            },
            tmp_path,
        )
        tmp_path.replace(cache_path)
        print(f"Wrote baseline logits cache to {cache_path}", flush=True)
    return logits


def _max_activation_doc_candidates(
    *,
    sparse_topk_parquet: Path,
    latent_ids: list[int],
    documents: dict[str, Any],
    docs_by_id: dict[str, ExplorerDoc],
    candidates_per_latent: int,
    batch_size: int,
    progress_every: int,
) -> tuple[dict[int, list[LinearizedDocImpact]], dict[int, dict[str, dict[str, float | int]]]]:
    if not sparse_topk_parquet.exists():
        raise FileNotFoundError(f"missing sparse top-k parquet: {sparse_topk_parquet}")
    doc_id_by_doc_index = {
        int(metadata["doc_index"]): str(doc_id)
        for doc_id, metadata in documents.items()
        if doc_id in docs_by_id and "doc_index" in metadata
    }
    if not doc_id_by_doc_index:
        raise ValueError("browser index does not contain doc_index metadata for sparse parquet lookup")

    selected_latents = sorted({int(latent_id) for latent_id in latent_ids})
    if not selected_latents:
        return {}, {}
    lookup_size = max(65536, max(selected_latents) + 1)
    selected_lookup = np.zeros(lookup_size, dtype=np.bool_)
    selected_lookup[np.asarray(selected_latents, dtype=np.int64)] = True

    maxes: dict[int, dict[str, dict[str, float | int]]] = defaultdict(dict)
    parquet = pq.ParquetFile(sparse_topk_parquet)
    processed = 0
    for batch_index, batch in enumerate(
        parquet.iter_batches(
            batch_size=int(batch_size),
            columns=["latent_ids", "activations", "doc_index", "token_index"],
        ),
        start=1,
    ):
        doc_indices = batch.column("doc_index").to_numpy(zero_copy_only=False)
        rows = len(doc_indices)
        latent_column = batch.column("latent_ids")
        value_column = batch.column("activations")
        width = int(latent_column.type.list_size)
        latent_matrix = latent_column.values.to_numpy(zero_copy_only=False).reshape(rows, width)
        value_matrix = value_column.values.to_numpy(zero_copy_only=False).reshape(rows, width)
        token_indices = batch.column("token_index").to_numpy(zero_copy_only=False)
        hit_rows, hit_offsets = np.nonzero(selected_lookup[latent_matrix])
        for row_offset, latent_offset in zip(hit_rows.tolist(), hit_offsets.tolist(), strict=True):
            doc_id = doc_id_by_doc_index.get(int(doc_indices[row_offset]))
            if doc_id is None:
                continue
            latent_id = int(latent_matrix[row_offset, latent_offset])
            value = float(value_matrix[row_offset, latent_offset])
            token_index = int(token_indices[row_offset])
            current = maxes[latent_id].get(doc_id)
            if current is None or value > float(current["peak_activation"]):
                maxes[latent_id][doc_id] = {
                    "peak_activation": value,
                    "peak_token_index": token_index,
                }
        processed += rows
        if progress_every > 0 and (batch_index == 1 or batch_index % int(progress_every) == 0):
            print(
                f"Scanned {processed:,}/{parquet.metadata.num_rows:,} sparse rows for peak-activation candidates",
                flush=True,
            )

    candidates: dict[int, list[LinearizedDocImpact]] = {}
    for latent_id in selected_latents:
        ranked_doc_maxes = sorted(
            maxes.get(latent_id, {}).items(),
            key=lambda item: (float(item[1]["peak_activation"]), str(item[0])),
            reverse=True,
        )
        estimates: list[LinearizedDocImpact] = []
        for doc_id, peak in ranked_doc_maxes[: int(candidates_per_latent)]:
            metadata = documents.get(doc_id, {})
            estimates.append(
                LinearizedDocImpact(
                    doc_id=doc_id,
                    turn_id=str(metadata.get("turn_id", "")),
                    estimated_logit_drop=0.0,
                    estimated_abs_logit_change=0.0,
                    human_estimated_logit_drop=0.0,
                    mean_ai_estimated_logit_drop=0.0,
                    peak_activation=float(peak["peak_activation"]),
                    peak_token_index=int(peak["peak_token_index"]),
                    candidate_source="max_activation",
                )
            )
        candidates[latent_id] = estimates
    return candidates, dict(maxes)


def _activation_difference_latent_rows(
    *,
    sparse_topk_parquet: Path,
    latent_dim: int,
    documents: dict[str, Any],
    docs_by_id: dict[str, ExplorerDoc],
    batch_size: int,
    progress_every: int,
    target_class: int,
    activation_layer: int,
) -> list[dict[str, Any]]:
    doc_rows: list[tuple[int, str, str, bool]] = []
    for doc_id, metadata in documents.items():
        if doc_id not in docs_by_id or "doc_index" not in metadata:
            continue
        turn_id = str(metadata.get("turn_id", ""))
        if not turn_id:
            continue
        model = str(metadata.get("model", docs_by_id[doc_id].source))
        doc_rows.append((int(metadata["doc_index"]), str(doc_id), turn_id, model == "human"))
    if not doc_rows:
        raise ValueError("browser index does not contain doc_index/turn_id metadata")

    turn_ids = sorted({turn_id for _doc_index, _doc_id, turn_id, _is_human in doc_rows})
    turn_index_by_id = {turn_id: index for index, turn_id in enumerate(turn_ids)}
    max_doc_index = max(doc_index for doc_index, _doc_id, _turn_id, _is_human in doc_rows)
    doc_index_to_turn = np.full(max_doc_index + 1, -1, dtype=np.int32)
    doc_index_to_is_human = np.zeros(max_doc_index + 1, dtype=np.bool_)
    for doc_index, _doc_id, turn_id, is_human in doc_rows:
        doc_index_to_turn[doc_index] = turn_index_by_id[turn_id]
        doc_index_to_is_human[doc_index] = is_human

    human_max = np.full((len(turn_ids), int(latent_dim)), -np.inf, dtype=np.float32)
    ai_max = np.full((len(turn_ids), int(latent_dim)), -np.inf, dtype=np.float32)
    parquet = pq.ParquetFile(sparse_topk_parquet)
    processed = 0
    for batch_index, batch in enumerate(
        parquet.iter_batches(
            batch_size=int(batch_size),
            columns=["latent_ids", "activations", "doc_index"],
        ),
        start=1,
    ):
        doc_indices = batch.column("doc_index").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        valid_rows = doc_indices <= max_doc_index
        if valid_rows.any():
            row_turns = np.full(len(doc_indices), -1, dtype=np.int32)
            row_human = np.zeros(len(doc_indices), dtype=np.bool_)
            valid_doc_indices = doc_indices[valid_rows]
            row_turns[valid_rows] = doc_index_to_turn[valid_doc_indices]
            row_human[valid_rows] = doc_index_to_is_human[valid_doc_indices]
            valid_rows &= row_turns >= 0
        if valid_rows.any():
            rows = len(doc_indices)
            latent_column = batch.column("latent_ids")
            value_column = batch.column("activations")
            width = int(latent_column.type.list_size)
            latent_matrix = latent_column.values.to_numpy(zero_copy_only=False).reshape(rows, width)[valid_rows]
            value_matrix = value_column.values.to_numpy(zero_copy_only=False).reshape(rows, width)[valid_rows]
            turn_flat = np.repeat(row_turns[valid_rows], width)
            human_flat = np.repeat(row_human[valid_rows], width)
            latent_flat = latent_matrix.reshape(-1).astype(np.int64, copy=False)
            value_flat = value_matrix.reshape(-1).astype(np.float32, copy=False)
            human_mask = human_flat
            ai_mask = ~human_flat
            if human_mask.any():
                np.maximum.at(human_max, (turn_flat[human_mask], latent_flat[human_mask]), value_flat[human_mask])
            if ai_mask.any():
                np.maximum.at(ai_max, (turn_flat[ai_mask], latent_flat[ai_mask]), value_flat[ai_mask])
        processed += len(doc_indices)
        if progress_every > 0 and (batch_index == 1 or batch_index % int(progress_every) == 0):
            print(
                f"Scanned {processed:,}/{parquet.metadata.num_rows:,} sparse rows for activation-difference latent scores",
                flush=True,
            )

    valid = np.isfinite(human_max) & np.isfinite(ai_max)
    differences = np.zeros_like(human_max)
    np.subtract(human_max, ai_max, out=differences, where=valid)
    abs_differences = np.abs(differences)
    prompt_counts = valid.sum(axis=0)
    mean_abs = abs_differences.sum(axis=0) / np.maximum(prompt_counts, 1)
    mean_signed = differences.sum(axis=0) / np.maximum(prompt_counts, 1)
    max_abs = abs_differences.max(axis=0)
    rows: list[dict[str, Any]] = []
    for latent_id in range(int(latent_dim)):
        rows.append(
            {
                "latent_id": latent_id,
                "linearized_mean_estimated_logit_drop": 0.0,
                "linearized_mean_abs_estimated_logit_change": 0.0,
                "linearized_mean_prompt_contrast_logit_drop": 0.0,
                "linearized_mean_abs_prompt_contrast_logit_change": 0.0,
                "linearized_active_docs": int(prompt_counts[latent_id]),
                "linearized_active_doc_share": float(prompt_counts[latent_id] / max(1, len(turn_ids))),
                "linearized_active_prompts": int(prompt_counts[latent_id]),
                "linearized_active_prompt_share": float(prompt_counts[latent_id] / max(1, len(turn_ids))),
                "linearized_scored_docs": len(docs_by_id),
                "linearized_scored_prompts": len(turn_ids),
                "activation_difference_max_abs": float(max_abs[latent_id]),
                "activation_difference_mean_abs": float(mean_abs[latent_id]),
                "activation_difference_mean_signed": float(mean_signed[latent_id]),
                "activation_difference_prompt_count": int(prompt_counts[latent_id]),
                "target_class": target_class,
                "activation_layer": int(activation_layer),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["activation_difference_max_abs"]),
            float(row["activation_difference_mean_abs"]),
            int(row["activation_difference_prompt_count"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["linearized_rank"] = rank
        row["activation_difference_rank"] = rank
    return rows


def _activation_difference_doc_candidates(
    *,
    peak_activations_by_latent: dict[int, dict[str, dict[str, float | int]]],
    latent_ids: list[int],
    documents: dict[str, Any],
    prompt_groups: dict[str, list[str]],
    docs_by_id: dict[str, ExplorerDoc],
    candidates_per_latent: int,
) -> tuple[dict[int, list[LinearizedDocImpact]], dict[int, dict[str, Any]]]:
    candidates: dict[int, list[LinearizedDocImpact]] = {}
    latent_stats: dict[int, dict[str, Any]] = {}
    for latent_id in latent_ids:
        doc_peaks = peak_activations_by_latent.get(int(latent_id), {})
        estimates: list[LinearizedDocImpact] = []
        abs_differences: list[float] = []
        signed_differences: list[float] = []
        for turn_id, sibling_ids in prompt_groups.items():
            human_doc_id = ""
            human_peak = 0.0
            max_ai_doc_id = ""
            max_ai_peak = 0.0
            for raw_doc_id in sibling_ids:
                doc_id = str(raw_doc_id)
                if doc_id not in docs_by_id:
                    continue
                peak = float(doc_peaks.get(doc_id, {}).get("peak_activation", 0.0))
                model = str(documents.get(doc_id, {}).get("model", docs_by_id[doc_id].source))
                if model == "human":
                    if not human_doc_id or peak > human_peak:
                        human_doc_id = doc_id
                        human_peak = peak
                elif peak > max_ai_peak:
                    max_ai_doc_id = doc_id
                    max_ai_peak = peak
            if not human_doc_id or not max_ai_doc_id:
                continue
            difference = human_peak - max_ai_peak
            abs_difference = abs(difference)
            if abs_difference <= 0:
                continue
            representative_doc_id = human_doc_id if difference >= 0 else max_ai_doc_id
            peak = doc_peaks.get(representative_doc_id, {})
            estimates.append(
                LinearizedDocImpact(
                    doc_id=representative_doc_id,
                    turn_id=str(turn_id),
                    estimated_logit_drop=difference,
                    estimated_abs_logit_change=abs_difference,
                    human_estimated_logit_drop=human_peak,
                    mean_ai_estimated_logit_drop=max_ai_peak,
                    peak_activation=float(peak.get("peak_activation", max(human_peak, max_ai_peak))),
                    peak_token_index=peak.get("peak_token_index", ""),
                    candidate_source="max_ai_human_activation_difference",
                    human_peak_activation=human_peak,
                    max_ai_peak_activation=max_ai_peak,
                    activation_difference=difference,
                )
            )
            signed_differences.append(difference)
            abs_differences.append(abs_difference)
        estimates.sort(
            key=lambda estimate: (
                estimate.estimated_abs_logit_change,
                estimate.peak_activation,
                estimate.turn_id,
            ),
            reverse=True,
        )
        candidates[int(latent_id)] = estimates[: int(candidates_per_latent)]
        latent_stats[int(latent_id)] = {
            "activation_difference_max_abs": max(abs_differences, default=0.0),
            "activation_difference_mean_abs": sum(abs_differences) / max(1, len(abs_differences)),
            "activation_difference_mean_signed": sum(signed_differences) / max(1, len(signed_differences)),
            "activation_difference_prompt_count": len(abs_differences),
        }
    return candidates, latent_stats


def _paired_activation_effect_rows(
    *,
    sparse_topk_parquet: Path,
    latent_dim: int,
    documents: dict[str, Any],
    prompt_groups: dict[str, list[str]],
    docs_by_id: dict[str, ExplorerDoc],
    candidates_per_latent: int,
    batch_size: int,
    progress_every: int,
    min_active_prompts: int,
    min_sign_consistency: float,
    target_class: int,
    activation_layer: int,
) -> tuple[list[dict[str, Any]], dict[int, list[LinearizedDocImpact]], dict[int, dict[str, Any]]]:
    doc_rows: list[tuple[int, str, str, bool, float]] = []
    for doc_id, metadata in documents.items():
        if doc_id not in docs_by_id or "doc_index" not in metadata:
            continue
        turn_id = str(metadata.get("turn_id", ""))
        if not turn_id:
            continue
        model = str(metadata.get("model", docs_by_id[doc_id].source))
        token_len = max(
            1.0,
            float(
                metadata.get(
                    "token_len",
                    int(metadata.get("sparse_row_end", 1)) - int(metadata.get("sparse_row_start", 0)),
                )
                or 1
            ),
        )
        doc_rows.append((int(metadata["doc_index"]), str(doc_id), turn_id, model == "human", token_len))
    if not doc_rows:
        raise ValueError("browser index does not contain doc_index/turn_id metadata")

    doc_rows.sort()
    doc_index_to_pos = {doc_index: pos for pos, (doc_index, _doc_id, _turn_id, _is_human, _token_len) in enumerate(doc_rows)}
    max_doc_index = max(doc_index_to_pos)
    doc_lookup = np.full(max_doc_index + 1, -1, dtype=np.int32)
    for doc_index, pos in doc_index_to_pos.items():
        doc_lookup[doc_index] = pos

    doc_ids = [doc_id for _doc_index, doc_id, _turn_id, _is_human, _token_len in doc_rows]
    turn_ids_by_doc = [turn_id for _doc_index, _doc_id, turn_id, _is_human, _token_len in doc_rows]
    is_human_by_doc = np.asarray([is_human for _doc_index, _doc_id, _turn_id, is_human, _token_len in doc_rows], dtype=np.bool_)
    token_lens = np.asarray([token_len for _doc_index, _doc_id, _turn_id, _is_human, token_len in doc_rows], dtype=np.float32)

    activation_sums = np.zeros((len(doc_rows), int(latent_dim)), dtype=np.float32)
    parquet = pq.ParquetFile(sparse_topk_parquet)
    processed = 0
    for batch_index, batch in enumerate(
        parquet.iter_batches(
            batch_size=int(batch_size),
            columns=["latent_ids", "activations", "doc_index"],
        ),
        start=1,
    ):
        doc_indices = batch.column("doc_index").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        valid_rows = doc_indices <= max_doc_index
        if valid_rows.any():
            doc_positions = np.full(len(doc_indices), -1, dtype=np.int32)
            doc_positions[valid_rows] = doc_lookup[doc_indices[valid_rows]]
            valid_rows &= doc_positions >= 0
        if valid_rows.any():
            rows = len(doc_indices)
            latent_column = batch.column("latent_ids")
            value_column = batch.column("activations")
            width = int(latent_column.type.list_size)
            latent_matrix = latent_column.values.to_numpy(zero_copy_only=False).reshape(rows, width)[valid_rows]
            value_matrix = value_column.values.to_numpy(zero_copy_only=False).reshape(rows, width)[valid_rows]
            doc_flat = np.repeat(doc_positions[valid_rows], width)
            latent_flat = latent_matrix.reshape(-1).astype(np.int64, copy=False)
            value_flat = value_matrix.reshape(-1).astype(np.float32, copy=False)
            np.add.at(activation_sums, (doc_flat, latent_flat), value_flat)
        processed += len(doc_indices)
        if progress_every > 0 and (batch_index == 1 or batch_index % int(progress_every) == 0):
            print(
                f"Scanned {processed:,}/{parquet.metadata.num_rows:,} sparse rows for paired activation scores",
                flush=True,
            )

    doc_scores = np.log1p(activation_sums) / np.sqrt(token_lens).reshape(-1, 1)
    prompt_ids = [
        turn_id
        for turn_id, sibling_ids in prompt_groups.items()
        if any(str(doc_id) in docs_by_id for doc_id in sibling_ids)
    ]
    prompt_effects = np.zeros((len(prompt_ids), int(latent_dim)), dtype=np.float32)
    prompt_human_scores = np.zeros_like(prompt_effects)
    prompt_ai_scores = np.zeros_like(prompt_effects)
    representative_doc_pos = np.full((len(prompt_ids), int(latent_dim)), -1, dtype=np.int32)
    prompt_valid = np.zeros((len(prompt_ids), int(latent_dim)), dtype=np.bool_)
    doc_pos_by_id = {doc_id: pos for pos, doc_id in enumerate(doc_ids)}
    for prompt_index, turn_id in enumerate(prompt_ids):
        sibling_positions = [
            doc_pos_by_id[str(doc_id)]
            for doc_id in prompt_groups.get(turn_id, [])
            if str(doc_id) in doc_pos_by_id
        ]
        if not sibling_positions:
            continue
        human_positions = [pos for pos in sibling_positions if bool(is_human_by_doc[pos])]
        ai_positions = [pos for pos in sibling_positions if not bool(is_human_by_doc[pos])]
        if not human_positions or not ai_positions:
            continue
        human_score = doc_scores[human_positions].mean(axis=0)
        ai_score = doc_scores[ai_positions].mean(axis=0)
        effect = ai_score - human_score
        prompt_effects[prompt_index] = effect
        prompt_human_scores[prompt_index] = human_score
        prompt_ai_scores[prompt_index] = ai_score
        prompt_valid[prompt_index] = (doc_scores[sibling_positions] > 0).any(axis=0)
        ai_matrix = doc_scores[ai_positions]
        ai_best_offsets = ai_matrix.argmax(axis=0)
        human_matrix = doc_scores[human_positions]
        human_best_offsets = human_matrix.argmax(axis=0)
        ai_best_positions = np.asarray(ai_positions, dtype=np.int32)[ai_best_offsets]
        human_best_positions = np.asarray(human_positions, dtype=np.int32)[human_best_offsets]
        representative_doc_pos[prompt_index] = np.where(effect >= 0, ai_best_positions, human_best_positions)

    valid_counts = prompt_valid.sum(axis=0)
    signed_sums = prompt_effects.sum(axis=0, dtype=np.float64)
    abs_sums = np.abs(prompt_effects).sum(axis=0, dtype=np.float64)
    positive_counts = (prompt_effects > 0).sum(axis=0)
    negative_counts = (prompt_effects < 0).sum(axis=0)
    sign_consistency = np.maximum(positive_counts, negative_counts) / np.maximum(valid_counts, 1)
    mean_signed = signed_sums / np.maximum(valid_counts, 1)
    mean_abs = abs_sums / np.maximum(valid_counts, 1)
    paired_score = np.abs(mean_signed) * sign_consistency * np.log1p(valid_counts)
    paired_score = np.where(
        (valid_counts >= int(min_active_prompts)) & (sign_consistency >= float(min_sign_consistency)),
        paired_score,
        0.0,
    )

    rows: list[dict[str, Any]] = []
    top_doc_rows: dict[int, list[LinearizedDocImpact]] = {}
    latent_stats: dict[int, dict[str, Any]] = {}
    for latent_id in range(int(latent_dim)):
        rows.append(
            {
                "latent_id": latent_id,
                "paired_activation_effect_score": float(paired_score[latent_id]),
                "paired_activation_mean_signed_effect": float(mean_signed[latent_id]),
                "paired_activation_mean_abs_effect": float(mean_abs[latent_id]),
                "paired_activation_sign_consistency": float(sign_consistency[latent_id]),
                "paired_activation_active_prompts": int(valid_counts[latent_id]),
                "linearized_mean_estimated_logit_drop": 0.0,
                "linearized_mean_abs_estimated_logit_change": 0.0,
                "linearized_mean_prompt_contrast_logit_drop": 0.0,
                "linearized_mean_abs_prompt_contrast_logit_change": 0.0,
                "linearized_active_docs": int(valid_counts[latent_id]),
                "linearized_active_doc_share": float(valid_counts[latent_id] / max(1, len(prompt_ids))),
                "linearized_active_prompts": int(valid_counts[latent_id]),
                "linearized_active_prompt_share": float(valid_counts[latent_id] / max(1, len(prompt_ids))),
                "linearized_scored_docs": len(docs_by_id),
                "linearized_scored_prompts": len(prompt_ids),
                "target_class": target_class,
                "activation_layer": int(activation_layer),
            }
        )
        latent_stats[latent_id] = {
            "paired_activation_effect_score": float(paired_score[latent_id]),
            "paired_activation_mean_signed_effect": float(mean_signed[latent_id]),
            "paired_activation_mean_abs_effect": float(mean_abs[latent_id]),
            "paired_activation_sign_consistency": float(sign_consistency[latent_id]),
            "paired_activation_active_prompts": int(valid_counts[latent_id]),
        }
        effects = np.abs(prompt_effects[:, latent_id])
        if effects.size == 0:
            top_doc_rows[latent_id] = []
            continue
        count = min(int(candidates_per_latent), int((effects > 0).sum()))
        if count <= 0:
            top_doc_rows[latent_id] = []
            continue
        top_positions = np.argpartition(effects, -count)[-count:]
        top_positions = top_positions[np.argsort(effects[top_positions])[::-1]]
        estimates: list[LinearizedDocImpact] = []
        for prompt_index in top_positions.tolist():
            doc_pos = int(representative_doc_pos[prompt_index, latent_id])
            if doc_pos < 0:
                continue
            signed = float(prompt_effects[prompt_index, latent_id])
            estimates.append(
                LinearizedDocImpact(
                    doc_id=doc_ids[doc_pos],
                    turn_id=prompt_ids[prompt_index],
                    estimated_logit_drop=signed,
                    estimated_abs_logit_change=abs(signed),
                    human_estimated_logit_drop=float(prompt_human_scores[prompt_index, latent_id]),
                    mean_ai_estimated_logit_drop=float(prompt_ai_scores[prompt_index, latent_id]),
                    peak_activation=float(doc_scores[doc_pos, latent_id]),
                    peak_token_index="",
                    candidate_source="paired_activation_effect_size",
                    paired_activation_effect=signed,
                    human_activation_score=float(prompt_human_scores[prompt_index, latent_id]),
                    mean_ai_activation_score=float(prompt_ai_scores[prompt_index, latent_id]),
                )
            )
        top_doc_rows[latent_id] = estimates

    rows.sort(
        key=lambda row: (
            float(row["paired_activation_effect_score"]),
            float(row["paired_activation_mean_abs_effect"]),
            int(row["paired_activation_active_prompts"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["linearized_rank"] = rank
        row["paired_activation_rank"] = rank
    return rows, top_doc_rows, latent_stats


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    index, docs_by_id = _load_payloads(args)
    all_docs = [docs_by_id[doc_id] for doc_id in sorted(docs_by_id)]
    if args.max_gradient_docs is not None:
        all_docs = all_docs[: int(args.max_gradient_docs)]
    if not all_docs:
        raise ValueError("no documents available for reranking")
    documents = index.get("documents", {})
    prompt_groups = index.get("promptGroups", {})

    model, tokenizer = load_pangram_model(args, device=device)
    layer = _llama_layer(model, int(args.activation_layer))
    sae, sae_summary = _load_sae(args.sae_path, device=device)
    sae.eval()

    print(f"Collecting baseline logits for {len(all_docs):,} docs", flush=True)
    baseline_logits = _baseline_logits_cached(
        model=model,
        tokenizer=tokenizer,
        docs=all_docs,
        args=args,
        device=device,
    )
    baseline_by_doc = {
        doc.doc_id: baseline_logits[index_offset].clone()
        for index_offset, doc in enumerate(all_docs)
    }
    target_class, class_diffs = _target_class_from_docs(baseline_logits, all_docs)
    human_class, ai_classes, margin_class_diffs = _class_groups_from_docs(baseline_logits, all_docs)

    if str(args.latent_selection_source) == "max_ai_human_activation_difference":
        if args.sparse_topk_parquet is None:
            raise ValueError("--sparse-topk-parquet is required for activation-difference latent selection")
        print(
            f"Selecting latents by max human-vs-AI peak activation difference "
            f"against target class {target_class}",
            flush=True,
        )
        ranked_linearized_rows = _activation_difference_latent_rows(
            sparse_topk_parquet=args.sparse_topk_parquet,
            latent_dim=int(sae.latent_dim),
            documents=documents,
            docs_by_id=docs_by_id,
            batch_size=int(args.sparse_scan_batch_size),
            progress_every=int(args.progress_every),
            target_class=target_class,
            activation_layer=int(args.activation_layer),
        )
        top_docs_by_latent: dict[int, list[LinearizedDocImpact]] = {}
    elif str(args.latent_selection_source) == "paired_activation_effect_size":
        if args.sparse_topk_parquet is None:
            raise ValueError("--sparse-topk-parquet is required for paired activation-effect selection")
        print(
            "Selecting latents by paired prompt-level aggregate activation effect size",
            flush=True,
        )
        ranked_linearized_rows, top_docs_by_latent, paired_activation_stats = _paired_activation_effect_rows(
            sparse_topk_parquet=args.sparse_topk_parquet,
            latent_dim=int(sae.latent_dim),
            documents=documents,
            prompt_groups=prompt_groups,
            docs_by_id=docs_by_id,
            candidates_per_latent=int(args.activation_candidates_per_latent),
            batch_size=int(args.sparse_scan_batch_size),
            progress_every=int(args.progress_every),
            min_active_prompts=int(args.paired_min_active_prompts),
            min_sign_consistency=float(args.paired_min_sign_consistency),
            target_class=target_class,
            activation_layer=int(args.activation_layer),
        )
    else:
        print(
            f"Estimating linearized causal impact for {len(all_docs):,} docs "
            f"against target class {target_class}",
            flush=True,
        )
        linearized_rows, top_docs_by_latent = _linearized_doc_impacts(
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            sae=sae,
            docs=all_docs,
            documents=documents,
            prompt_groups=prompt_groups,
            target_class=target_class,
            args=args,
            device=device,
        )
        ranked_linearized_rows = _rank_linearized_rows(
            linearized_rows,
            candidate_score=str(args.candidate_score),
        )
    selected_linearized_rows = ranked_linearized_rows[: int(args.linearized_top_latents)]
    selected_ids = [int(row["latent_id"]) for row in selected_linearized_rows]
    selected_id_texts = {str(latent_id) for latent_id in selected_ids}
    linearized_by_id = {int(row["latent_id"]): row for row in ranked_linearized_rows}
    original_latents_by_id = {
        str(int(row["latent_id"])): dict(row)
        for row in index.get("latents", [])
    }
    peak_activations_by_latent: dict[int, dict[str, dict[str, float | int]]] = {}
    activation_difference_stats: dict[int, dict[str, Any]] = {}
    paired_activation_stats = locals().get("paired_activation_stats", {})
    if str(args.candidate_doc_source) in {"max_activation", "max_ai_human_activation_difference"}:
        if args.sparse_topk_parquet is None:
            raise ValueError(
                "--sparse-topk-parquet is required with activation-based --candidate-doc-source"
            )
        print(
            f"Selecting top {int(args.activation_candidates_per_latent):,} docs per latent by peak activation",
            flush=True,
        )
        peak_doc_candidates, peak_activations_by_latent = _max_activation_doc_candidates(
            sparse_topk_parquet=args.sparse_topk_parquet,
            latent_ids=selected_ids,
            documents=documents,
            docs_by_id=docs_by_id,
            candidates_per_latent=(
                len(all_docs)
                if str(args.candidate_doc_source) == "max_ai_human_activation_difference"
                else int(args.activation_candidates_per_latent)
            ),
            batch_size=int(args.sparse_scan_batch_size),
            progress_every=int(args.progress_every),
        )
        if str(args.candidate_doc_source) == "max_activation":
            top_docs_by_latent = peak_doc_candidates
        else:
            print(
                "Aggregating candidates by max human-vs-AI activation difference per prompt",
                flush=True,
            )
            top_docs_by_latent, activation_difference_stats = _activation_difference_doc_candidates(
                peak_activations_by_latent=peak_activations_by_latent,
                latent_ids=selected_ids,
                documents=documents,
                prompt_groups=prompt_groups,
                docs_by_id=docs_by_id,
                candidates_per_latent=int(args.activation_candidates_per_latent),
            )
    elif str(args.candidate_doc_source) == "paired_activation_effect_size":
        if args.sparse_topk_parquet is None:
            raise ValueError("--sparse-topk-parquet is required with paired activation-effect candidates")
        if str(args.latent_selection_source) != "paired_activation_effect_size":
            _rows, top_docs_by_latent, paired_activation_stats = _paired_activation_effect_rows(
                sparse_topk_parquet=args.sparse_topk_parquet,
                latent_dim=int(sae.latent_dim),
                documents=documents,
                prompt_groups=prompt_groups,
                docs_by_id=docs_by_id,
                candidates_per_latent=int(args.activation_candidates_per_latent),
                batch_size=int(args.sparse_scan_batch_size),
                progress_every=int(args.progress_every),
                min_active_prompts=int(args.paired_min_active_prompts),
                min_sign_consistency=float(args.paired_min_sign_consistency),
                target_class=target_class,
                activation_layer=int(args.activation_layer),
            )

    latent_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    generation_impact_rows: list[dict[str, Any]] = []
    generation_impacts: dict[str, dict[str, dict[str, Any]]] = {}

    for latent_offset, latent_id in enumerate(selected_ids, start=1):
        latent_id_text = str(latent_id)
        prompt_order: list[str] = []
        seen_turns: set[str] = set()
        candidate_estimate_by_turn: dict[str, LinearizedDocImpact] = {}
        candidate_limit = (
            int(args.activation_candidates_per_latent)
            if str(args.candidate_doc_source) != "linearized_impact"
            else int(args.linearized_top_generations_per_latent)
        )
        for estimate in top_docs_by_latent[latent_id][:candidate_limit]:
            turn_id = estimate.turn_id or str(documents.get(estimate.doc_id, {}).get("turn_id", ""))
            if not turn_id:
                turn_id = estimate.doc_id
            if turn_id in seen_turns:
                continue
            seen_turns.add(turn_id)
            candidate_estimate_by_turn[turn_id] = estimate
            prompt_order.append(turn_id)

        latent_doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()
        for turn_id in prompt_order:
            sibling_ids = prompt_groups.get(turn_id, [turn_id])
            for sibling_doc_id in sibling_ids:
                sibling_doc_id = str(sibling_doc_id)
                if sibling_doc_id in docs_by_id and sibling_doc_id not in seen_doc_ids:
                    latent_doc_ids.append(sibling_doc_id)
                    seen_doc_ids.add(sibling_doc_id)
        latent_docs = [docs_by_id[doc_id] for doc_id in latent_doc_ids]
        effects = _ablate_latent_on_docs_browser_aligned(
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            sae=sae,
            latent_id=latent_id,
            docs=latent_docs,
            baseline_logits_by_doc=baseline_by_doc,
            target_class=target_class,
            human_class=human_class,
            ai_classes=ai_classes,
            args=args,
            device=device,
        )
        generation_impacts[latent_id_text] = effects
        prompt_contrast_rows: list[dict[str, Any]] = []
        for turn_id in prompt_order:
            prompt_contrast = _prompt_contrast_from_effects(
                turn_id=turn_id,
                sibling_doc_ids=[str(doc_id) for doc_id in prompt_groups.get(turn_id, [])],
                documents=documents,
                effects=effects,
            )
            if prompt_contrast is not None:
                candidate_estimate = candidate_estimate_by_turn.get(turn_id)
                if candidate_estimate is not None and candidate_estimate.doc_id in effects:
                    prompt_contrast["doc_id"] = candidate_estimate.doc_id
                    prompt_contrast["candidate_peak_activation"] = candidate_estimate.peak_activation
                    prompt_contrast["candidate_peak_token_index"] = candidate_estimate.peak_token_index
                    prompt_contrast["candidate_source"] = candidate_estimate.candidate_source
                prompt_contrast_rows.append(prompt_contrast)
        drops = [float(row["prompt_contrast_logit_drop"]) for row in prompt_contrast_rows]
        abs_drops = [abs(drop) for drop in drops]
        positives = [drop for drop in drops if drop > 0]
        active_prompts = [drop for drop in drops if abs(drop) > 0]
        base_latent = original_latents_by_id.get(latent_id_text, {})
        linearized = linearized_by_id[latent_id]
        activation_difference = activation_difference_stats.get(latent_id, {})
        paired_activation = paired_activation_stats.get(latent_id, {})
        causal_sign_consistency = 0.0
        causal_consistent_score = 0.0
        if drops:
            positive_prompt_count = sum(1 for value in drops if value > 0)
            negative_prompt_count = sum(1 for value in drops if value < 0)
            causal_sign_consistency = max(positive_prompt_count, negative_prompt_count) / max(1, len(drops))
            causal_consistent_score = abs(sum(drops) / max(1, len(drops))) * causal_sign_consistency * (len(drops) ** 0.5)
        latent_rows.append(
            {
                **base_latent,
                **linearized,
                **activation_difference,
                **paired_activation,
                "latent_id": latent_id,
                "original_rank": int(base_latent.get("rank", linearized.get("linearized_rank", latent_offset))),
                "causal_max_abs_target_logit_change": max(abs_drops, default=0.0),
                "causal_max_target_logit_drop": max(
                    drops,
                    key=lambda value: abs(value),
                    default=0.0,
                ),
                "causal_mean_abs_target_logit_change": sum(abs_drops) / max(1, len(abs_drops)),
                "causal_mean_target_logit_drop": sum(drops) / max(1, len(drops)),
                "causal_mean_positive_target_logit_drop": sum(positives) / max(1, len(positives)),
                "causal_mean_abs_target_probability_change": "",
                "causal_mean_target_probability_drop": "",
                "causal_positive_doc_rate": len(positives) / max(1, len(drops)),
                "causal_active_doc_rate": len(active_prompts) / max(1, len(drops)),
                "causal_active_prompt_rate": len(active_prompts) / max(1, len(drops)),
                "causal_prompt_sign_consistency": causal_sign_consistency,
                "causal_prompt_consistent_score": causal_consistent_score,
                "causal_docs": len(effects),
                "causal_prompts": len(prompt_contrast_rows),
                "exact_max_abs_prompt_contrast_logit_change": max(abs_drops, default=0.0),
                "exact_mean_abs_prompt_contrast_logit_change": sum(abs_drops) / max(1, len(abs_drops)),
                "exact_mean_prompt_contrast_logit_drop": sum(drops) / max(1, len(drops)),
                "causal_prompt_candidates": len(prompt_order),
                "target_class": target_class,
                "activation_layer": int(args.activation_layer),
            }
        )

        prompt_examples: list[dict[str, Any]] = []
        for prompt_contrast in prompt_contrast_rows:
            turn_id = str(prompt_contrast["turn_id"])
            doc_id = str(prompt_contrast["doc_id"])
            effect = effects[doc_id]
            metadata = documents.get(doc_id, {})
            doc = docs_by_id[doc_id]
            peak_activation = float(prompt_contrast.get("candidate_peak_activation", 0.0))
            peak_token_index = prompt_contrast.get("candidate_peak_token_index", "")
            prompt_examples.append(
                {
                    "latent_id": latent_id,
                    "doc_id": doc_id,
                    "turn_id": turn_id,
                    "source": metadata.get("model", doc.source),
                    "model": metadata.get("model", doc.source),
                    "context": _snippet(doc.text),
                    "activation": peak_activation,
                    "peak_activation": peak_activation,
                    "token_index": peak_token_index,
                    "candidate_source": prompt_contrast.get("candidate_source", args.candidate_doc_source),
                    "human_peak_activation": getattr(candidate_estimate_by_turn.get(turn_id), "human_peak_activation", ""),
                    "max_ai_peak_activation": getattr(candidate_estimate_by_turn.get(turn_id), "max_ai_peak_activation", ""),
                    "activation_difference": getattr(candidate_estimate_by_turn.get(turn_id), "activation_difference", ""),
                    "paired_activation_effect": getattr(candidate_estimate_by_turn.get(turn_id), "paired_activation_effect", ""),
                    "human_activation_score": getattr(candidate_estimate_by_turn.get(turn_id), "human_activation_score", ""),
                    "mean_ai_activation_score": getattr(candidate_estimate_by_turn.get(turn_id), "mean_ai_activation_score", ""),
                    "prompt_contrast_logit_drop": prompt_contrast["prompt_contrast_logit_drop"],
                    "prompt_abs_contrast_logit_change": prompt_contrast["prompt_abs_contrast_logit_change"],
                    "prompt_human_logit_drop": prompt_contrast["prompt_human_logit_drop"],
                    "prompt_mean_ai_logit_drop": prompt_contrast["prompt_mean_ai_logit_drop"],
                    "prompt_ai_docs": prompt_contrast["prompt_ai_docs"],
                    **effect,
                    "causal_logit_drop": prompt_contrast["prompt_contrast_logit_drop"],
                    "causal_abs_logit_change": prompt_contrast["prompt_abs_contrast_logit_change"],
                }
            )
        if str(args.candidate_doc_source) == "max_activation":
            prompt_examples.sort(
                key=lambda row: (
                    float(row["peak_activation"]),
                    float(row["prompt_abs_contrast_logit_change"]),
                    abs(float(row["prompt_human_logit_drop"])),
                    abs(float(row["prompt_mean_ai_logit_drop"])),
                ),
                reverse=True,
            )
        elif str(args.candidate_doc_source) == "max_ai_human_activation_difference":
            prompt_examples.sort(
                key=lambda row: (
                    abs(float(row["activation_difference"] or 0.0)),
                    float(row["peak_activation"]),
                    float(row["prompt_abs_contrast_logit_change"]),
                ),
                reverse=True,
            )
        elif str(args.candidate_doc_source) == "paired_activation_effect_size":
            prompt_examples.sort(
                key=lambda row: (
                    abs(float(row["paired_activation_effect"] or 0.0)),
                    float(row["prompt_abs_contrast_logit_change"]),
                    float(row["peak_activation"]),
                ),
                reverse=True,
            )
        else:
            prompt_examples.sort(
                key=lambda row: (
                    float(row["prompt_abs_contrast_logit_change"]),
                    abs(float(row["prompt_human_logit_drop"])),
                    abs(float(row["prompt_mean_ai_logit_drop"])),
                ),
                reverse=True,
            )
        for example_rank, example in enumerate(prompt_examples[: int(args.examples_per_latent)], start=1):
            example["example_rank"] = example_rank
            example_rows.append(example)
        for doc_id, effect in effects.items():
            metadata = documents.get(doc_id, {})
            peak = peak_activations_by_latent.get(latent_id, {}).get(doc_id, {})
            generation_impact_rows.append(
                {
                    "latent_id": latent_id,
                    "doc_id": doc_id,
                    "turn_id": metadata.get("turn_id", ""),
                    "model": metadata.get("model", docs_by_id[doc_id].source),
                    "peak_activation": peak.get("peak_activation", ""),
                    "peak_token_index": peak.get("peak_token_index", ""),
                    **effect,
                }
            )
        if args.progress_every > 0 and (
            latent_offset == 1 or latent_offset % int(args.progress_every) == 0
        ):
            print(
                f"Exact-ablated {latent_offset:,}/{len(selected_ids):,} "
                "linearized-selected latents",
                flush=True,
            )

    if args.final_latent_rank_score == "exact_max_abs":
        latent_sort_key = lambda row: (
            float(row["causal_max_abs_target_logit_change"]),
            abs(float(row["causal_max_target_logit_drop"])),
            float(row["causal_mean_abs_target_logit_change"]),
            float(row["linearized_mean_abs_estimated_logit_change"]),
        )
    elif args.final_latent_rank_score == "exact_mean_abs":
        latent_sort_key = lambda row: (
            float(row["causal_mean_abs_target_logit_change"]),
            float(row["causal_max_abs_target_logit_change"]),
            float(row.get("paired_activation_effect_score", 0.0) or 0.0),
        )
    elif args.final_latent_rank_score == "linearized_signed_mean":
        latent_sort_key = lambda row: (
            float(row["linearized_mean_estimated_logit_drop"]),
            float(row["linearized_mean_abs_estimated_logit_change"]),
            float(row["causal_max_abs_target_logit_change"]),
        )
    elif args.final_latent_rank_score == "activation_difference_abs_mean":
        latent_sort_key = lambda row: (
            float(row.get("activation_difference_mean_abs", 0.0) or 0.0),
            float(row.get("activation_difference_max_abs", 0.0) or 0.0),
            float(row["causal_max_abs_target_logit_change"]),
        )
    elif args.final_latent_rank_score == "activation_difference_max_abs":
        latent_sort_key = lambda row: (
            float(row.get("activation_difference_max_abs", 0.0) or 0.0),
            float(row.get("activation_difference_mean_abs", 0.0) or 0.0),
            float(row["causal_max_abs_target_logit_change"]),
        )
    elif args.final_latent_rank_score == "paired_activation_effect_size":
        latent_sort_key = lambda row: (
            float(row.get("paired_activation_effect_score", 0.0) or 0.0),
            float(row.get("paired_activation_mean_abs_effect", 0.0) or 0.0),
            float(row.get("causal_prompt_consistent_score", 0.0) or 0.0),
        )
    elif args.final_latent_rank_score == "paired_causal_consistent":
        latent_sort_key = lambda row: (
            float(row.get("causal_prompt_consistent_score", 0.0) or 0.0),
            float(row.get("causal_mean_abs_target_logit_change", 0.0) or 0.0),
            float(row.get("paired_activation_effect_score", 0.0) or 0.0),
        )
    else:
        latent_sort_key = lambda row: (
            float(row["linearized_mean_abs_estimated_logit_change"]),
            abs(float(row["linearized_mean_estimated_logit_drop"])),
            float(row["causal_max_abs_target_logit_change"]),
        )
    latent_rows.sort(key=latent_sort_key, reverse=True)
    for rank, row in enumerate(latent_rows, start=1):
        row["causal_rank"] = rank
        row["rank"] = rank

    reranked_latents: list[dict[str, Any]] = []
    reranked_latent_docs: dict[str, list[dict[str, Any]]] = {}
    for row in latent_rows:
        latent_id_text = str(int(row["latent_id"]))
        reranked_latents.append(dict(row))
        docs = [
            example
            for example in example_rows
            if str(int(example["latent_id"])) == latent_id_text
        ]
        for example in docs:
            example["rank"] = int(row["rank"])
        reranked_latent_docs[latent_id_text] = docs

    untouched_latents = [
        row for row in index.get("latents", []) if str(int(row["latent_id"])) not in selected_id_texts
    ]
    for offset, row in enumerate(untouched_latents, start=len(reranked_latents) + 1):
        row["rank"] = offset
        row["causal_rank"] = offset
    index["latents"] = reranked_latents + untouched_latents
    for latent_id, docs in reranked_latent_docs.items():
        index["latentDocs"][latent_id] = docs
    index["generationImpacts"] = generation_impacts
    rerank_method = (
        "browser_aligned_paired_activation_effect_then_exact_prompt_contrast_ablation"
        if str(args.latent_selection_source) == "paired_activation_effect_size"
        else
        "browser_aligned_activation_difference_selection_then_exact_prompt_contrast_ablation"
        if str(args.latent_selection_source) == "max_ai_human_activation_difference"
        else "browser_aligned_linearized_prompt_contrast_then_exact_prompt_contrast_ablation"
    )
    index["causalRerank"] = {
        "method": rerank_method,
        "activation_layer": int(args.activation_layer),
        "target_class": target_class,
        "linearized_top_latents": int(args.linearized_top_latents),
        "linearized_top_generations_per_latent": int(args.linearized_top_generations_per_latent),
        "latent_selection_source": str(args.latent_selection_source),
        "candidate_doc_source": str(args.candidate_doc_source),
        "activation_candidates_per_latent": int(args.activation_candidates_per_latent),
        "examples_per_latent": int(args.examples_per_latent),
        "linearized_scored_documents": len(all_docs),
        "effect_target": str(args.effect_target),
        "human_class": int(human_class),
        "ai_classes": ai_classes,
        "paired_min_active_prompts": int(args.paired_min_active_prompts),
        "paired_min_sign_consistency": float(args.paired_min_sign_consistency),
        "candidate_score": str(args.candidate_score),
        "final_latent_rank_score": str(args.final_latent_rank_score),
        "browser_top_k": int(args.browser_top_k),
        "activation_definition": "dense_relu_per_token_topk_non_special_tokens",
        "linearized_score": (
            "prompt contrast of grad(target_logit) dot latent_decoder_contribution: "
            "human_estimated_drop - mean(ai_estimated_drop)"
        ),
        "candidate_doc_score": (
            "per-prompt normalized aggregate activation effect size: mean(ai) - human"
            if str(args.candidate_doc_source) == "paired_activation_effect_size"
            else
            "per-prompt absolute difference between human peak activation and max AI peak activation"
            if str(args.candidate_doc_source) == "max_ai_human_activation_difference"
            else "per-document peak browser sparse activation"
            if str(args.candidate_doc_source) == "max_activation"
            else "linearized prompt contrast score"
        ),
        "exact_score": (
            "prompt contrast of exact browser-aligned ablation effect: "
            "human_effect_drop - mean(ai_effect_drop)"
        ),
    }

    output_index = args.output_browser_index or args.browser_index
    output_index.write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "pangram_llama_sae_linearized_candidate_latents.csv", ranked_linearized_rows)
    linearized_top_doc_rows: list[dict[str, Any]] = []
    for latent_id in selected_ids:
        for rank, estimate in enumerate(top_docs_by_latent.get(latent_id, []), start=1):
            metadata = documents.get(estimate.doc_id, {})
            linearized_top_doc_rows.append(
                {
                    "latent_id": latent_id,
                    "candidate_doc_rank": rank,
                    "candidate_source": estimate.candidate_source,
                    "doc_id": estimate.doc_id,
                    "turn_id": estimate.turn_id or metadata.get("turn_id", ""),
                    "model": metadata.get("model", docs_by_id[estimate.doc_id].source),
                    "peak_activation": estimate.peak_activation,
                    "peak_token_index": estimate.peak_token_index,
                    "human_peak_activation": estimate.human_peak_activation,
                    "max_ai_peak_activation": estimate.max_ai_peak_activation,
                    "activation_difference": estimate.activation_difference,
                    "paired_activation_effect": estimate.paired_activation_effect,
                    "human_activation_score": estimate.human_activation_score,
                    "mean_ai_activation_score": estimate.mean_ai_activation_score,
                    "linearized_estimated_logit_drop": estimate.estimated_logit_drop,
                    "linearized_estimated_abs_logit_change": estimate.estimated_abs_logit_change,
                    "linearized_prompt_contrast_logit_drop": estimate.estimated_logit_drop,
                    "linearized_abs_prompt_contrast_logit_change": estimate.estimated_abs_logit_change,
                    "linearized_human_logit_drop": estimate.human_estimated_logit_drop,
                    "linearized_mean_ai_logit_drop": estimate.mean_ai_estimated_logit_drop,
                }
            )
    _write_csv(args.output_dir / "pangram_llama_sae_linearized_top_generations.csv", linearized_top_doc_rows)
    _write_csv(args.output_dir / "pangram_llama_sae_causal_reranked_latents.csv", latent_rows)
    _write_csv(args.output_dir / "pangram_llama_sae_causal_reranked_examples.csv", example_rows)
    _write_csv(args.output_dir / "pangram_llama_sae_causal_generation_impacts.csv", generation_impact_rows)
    summary = {
        "browser_index": str(args.browser_index),
        "doc_tokens_parquet": str(args.doc_tokens_parquet),
        "sparse_topk_parquet": str(args.sparse_topk_parquet) if args.sparse_topk_parquet is not None else "",
        "sae_path": str(args.sae_path),
        "output_browser_index": str(output_index),
        "output_dir": str(args.output_dir),
        "activation_layer": int(args.activation_layer),
        "target_class": target_class,
        "class_diffs": class_diffs,
        "linearized_top_latents": int(args.linearized_top_latents),
        "linearized_top_generations_per_latent": int(args.linearized_top_generations_per_latent),
        "latent_selection_source": str(args.latent_selection_source),
        "candidate_doc_source": str(args.candidate_doc_source),
        "activation_candidates_per_latent": int(args.activation_candidates_per_latent),
        "examples_per_latent": int(args.examples_per_latent),
        "linearized_scored_documents": len(all_docs),
        "effect_target": str(args.effect_target),
        "human_class": int(human_class),
        "ai_classes": ai_classes,
        "margin_class_diffs": margin_class_diffs,
        "final_latent_rank_score": str(args.final_latent_rank_score),
        "browser_top_k": int(args.browser_top_k),
        "activation_definition": "dense_relu_per_token_topk_non_special_tokens",
        "exact_generation_impacts": len(generation_impact_rows),
        "source_counts": dict(sorted(Counter(doc.source for doc in all_docs).items())),
        "label_counts": dict(sorted(Counter(str(doc.label) for doc in all_docs).items())),
        "sae_summary": sae_summary,
    }
    _write_json(args.output_dir / "pangram_llama_sae_linearized_causal_rerank_summary.json", summary)
    return summary


def main() -> None:
    summary = run(build_parser().parse_args())
    print(
        "Wrote linearized-causal-reranked latent explorer index to "
        f"{summary['output_browser_index']} "
        f"({summary['linearized_top_latents']} latents, "
        f"{summary['linearized_scored_documents']} gradient-scored docs)",
        flush=True,
    )


if __name__ == "__main__":
    main()
