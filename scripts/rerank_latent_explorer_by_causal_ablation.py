from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
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
from slop_sftdiv.cli.phase4_batchtopk_sae import SAEDoc, _load_sae


@dataclass(frozen=True)
class ExplorerDoc:
    doc_id: str
    source: str
    label: int
    text: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rerank latent-explorer latents and examples by exact target-score "
            "change when each displayed latent is ablated."
        )
    )
    parser.add_argument("--browser-index", type=Path, required=True)
    parser.add_argument("--doc-tokens-parquet", type=Path, required=True)
    parser.add_argument("--sae-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-browser-index", type=Path, default=None)
    parser.add_argument("--activation-layer", type=int, default=20)
    parser.add_argument("--top-latents", type=int, default=100)
    parser.add_argument("--examples-per-latent", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=640)
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "causal_rank",
        "rank",
        "latent_id",
        "causal_mean_abs_target_logit_change",
        "causal_mean_target_logit_drop",
        "causal_mean_abs_target_probability_change",
        "causal_mean_target_probability_drop",
        "causal_positive_doc_rate",
        "causal_active_doc_rate",
        "causal_docs",
        "original_rank",
        "ai_human_log_odds",
        "ai_mass_share",
        "total_mass",
        "example_rank",
        "causal_abs_logit_change",
        "causal_logit_drop",
        "causal_abs_probability_change",
        "causal_probability_drop",
        "causal_active_tokens",
        "source",
        "doc_id",
        "turn_id",
        "model",
        "activation",
        "peak_activation",
        "token_index",
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


def _llama_layer(model: Any, layer_index: int) -> nn.Module:
    layers = getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None)
    layer_list = getattr(layers, "layers", None)
    if layer_list is None:
        raise TypeError("could not find Llama decoder layers on Pangram model")
    if not 0 <= layer_index < len(layer_list):
        raise ValueError(f"--activation-layer {layer_index} is out of range for {len(layer_list)} layers")
    return cast(nn.Module, layer_list[layer_index])


def _batched(items: list[ExplorerDoc], batch_size: int) -> list[list[ExplorerDoc]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _as_sae_doc(doc: ExplorerDoc) -> SAEDoc:
    return SAEDoc(source=doc.source, doc_id=doc.doc_id, text=doc.text, label=doc.label)


def _load_payloads(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, ExplorerDoc]]:
    index = json.loads(args.browser_index.read_text(encoding="utf-8"))
    doc_rows = pd.read_parquet(args.doc_tokens_parquet, columns=["doc_id", "model", "text"])
    docs: dict[str, ExplorerDoc] = {}
    for row in doc_rows.itertuples(index=False):
        doc_id = str(row.doc_id)
        source = str(row.model)
        docs[doc_id] = ExplorerDoc(
            doc_id=doc_id,
            source=source,
            label=0 if source == "human" else 1,
            text=str(row.text),
        )
    return index, docs


@torch.inference_mode()
def _baseline_logits(
    *,
    model: Any,
    tokenizer: Any,
    docs: list[ExplorerDoc],
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for batch_docs in _batched(docs, int(args.batch_size)):
        encoded = _encode_docs(
            tokenizer,
            [_as_sae_doc(doc) for doc in batch_docs],
            max_length=int(args.max_length),
            device=device,
            pad_to_max_length=True,
        )
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        )
        parts.append(cast(torch.Tensor, outputs.logits).detach().float().cpu())
    return torch.cat(parts, dim=0)


def _target_class_from_docs(logits: torch.Tensor, docs: list[ExplorerDoc]) -> tuple[int, list[dict[str, Any]]]:
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
            }
        )
    target = int(max(rows, key=lambda row: float(row["ai_minus_human_logit"]))["class_id"])
    return target, rows


def _softmax_target(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)[:, target_class]


@torch.inference_mode()
def _ablate_latent_on_docs(
    *,
    model: Any,
    tokenizer: Any,
    layer: nn.Module,
    sae: Any,
    latent_id: int,
    docs: list[ExplorerDoc],
    baseline_logits_by_doc: dict[str, torch.Tensor],
    target_class: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    decoder_weight = sae.decoder.weight.detach()
    results: dict[str, dict[str, Any]] = {}
    for batch_docs in _batched(docs, int(args.batch_size)):
        encoded = _encode_docs(
            tokenizer,
            [_as_sae_doc(doc) for doc in batch_docs],
            max_length=int(args.max_length),
            device=device,
            pad_to_max_length=True,
        )
        attention_mask = encoded["attention_mask"].bool()
        active_doc_indexes: set[int] = set()
        active_token_counts = [0 for _doc in batch_docs]

        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                return output
            hidden_float = hidden.float()
            token_vectors = hidden_float[attention_mask]
            if token_vectors.numel() == 0:
                return output
            codes = sae.encode(token_vectors)
            latent_codes = codes[:, latent_id]
            active_positions = latent_codes > 0
            if active_positions.any():
                token_doc_indexes = torch.nonzero(attention_mask, as_tuple=False)[:, 0]
                for index in token_doc_indexes[active_positions].detach().cpu().tolist():
                    active_doc_indexes.add(int(index))
                    active_token_counts[int(index)] += 1
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
        ablated_probs = _softmax_target(ablated_logits, target_class)
        for doc_index, doc in enumerate(batch_docs):
            baseline_logits = baseline_logits_by_doc[doc.doc_id]
            baseline_prob = float(_softmax_target(baseline_logits.unsqueeze(0), target_class)[0])
            logit_drop = float(baseline_logits[target_class] - ablated_logits[doc_index, target_class])
            prob_drop = float(baseline_prob - ablated_probs[doc_index])
            results[doc.doc_id] = {
                "causal_logit_drop": logit_drop,
                "causal_abs_logit_change": abs(logit_drop),
                "causal_probability_drop": prob_drop,
                "causal_abs_probability_change": abs(prob_drop),
                "causal_active_tokens": int(active_token_counts[doc_index]),
                "causal_active": doc_index in active_doc_indexes,
            }
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    index, docs_by_id = _load_payloads(args)
    selected_latents = index.get("latents", [])[: int(args.top_latents)]
    selected_ids = [str(int(row["latent_id"])) for row in selected_latents]
    documents = index.get("documents", {})
    prompt_groups = index.get("promptGroups", {})
    selected_doc_ids = sorted(
        {
            str(sibling_doc_id)
            for latent_id in selected_ids
            for example in index.get("latentDocs", {}).get(latent_id, [])[: int(args.examples_per_latent)]
            for sibling_doc_id in prompt_groups.get(
                str(documents.get(str(example.get("doc_id", "")), {}).get("turn_id", "")),
                [],
            )
            if str(sibling_doc_id) in docs_by_id
        }
    )
    selected_docs = [docs_by_id[doc_id] for doc_id in selected_doc_ids]
    if not selected_docs:
        raise ValueError("no explorer documents selected for causal reranking")

    model, tokenizer = load_pangram_model(args, device=device)
    layer = _llama_layer(model, int(args.activation_layer))
    sae, sae_summary = _load_sae(args.sae_path, device=device)
    sae.eval()

    print(f"Collecting baseline logits for {len(selected_docs):,} explorer docs", flush=True)
    baseline_logits = _baseline_logits(
        model=model,
        tokenizer=tokenizer,
        docs=selected_docs,
        args=args,
        device=device,
    )
    baseline_by_doc = {
        doc.doc_id: baseline_logits[index].clone()
        for index, doc in enumerate(selected_docs)
    }
    target_class, class_diffs = _target_class_from_docs(baseline_logits, selected_docs)

    latent_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    generation_impact_rows: list[dict[str, Any]] = []
    generation_impacts: dict[str, dict[str, dict[str, Any]]] = {}
    latents_by_id = {str(int(row["latent_id"])): dict(row) for row in selected_latents}

    for latent_offset, latent_id_text in enumerate(selected_ids, start=1):
        latent_id = int(latent_id_text)
        examples = index.get("latentDocs", {}).get(latent_id_text, [])[: int(args.examples_per_latent)]
        latent_doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()
        for example in examples:
            example_doc_id = str(example.get("doc_id", ""))
            turn_id = str(documents.get(example_doc_id, {}).get("turn_id", ""))
            for sibling_doc_id in prompt_groups.get(turn_id, []):
                sibling_doc_id = str(sibling_doc_id)
                if sibling_doc_id in docs_by_id and sibling_doc_id not in seen_doc_ids:
                    latent_doc_ids.append(sibling_doc_id)
                    seen_doc_ids.add(sibling_doc_id)
        latent_docs = [docs_by_id[doc_id] for doc_id in latent_doc_ids]
        effects = _ablate_latent_on_docs(
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            sae=sae,
            latent_id=latent_id,
            docs=latent_docs,
            baseline_logits_by_doc=baseline_by_doc,
            target_class=target_class,
            args=args,
            device=device,
        )
        generation_impacts[latent_id_text] = effects
        drops = [float(row["causal_logit_drop"]) for row in effects.values()]
        prob_drops = [float(row["causal_probability_drop"]) for row in effects.values()]
        abs_drops = [abs(drop) for drop in drops]
        abs_prob_drops = [abs(drop) for drop in prob_drops]
        positives = [drop for drop in drops if drop > 0]
        active_docs = [row for row in effects.values() if bool(row["causal_active"])]
        latent = latents_by_id[latent_id_text]
        latent_rows.append(
            {
                **latent,
                "latent_id": latent_id,
                "original_rank": int(latent.get("rank", latent_offset)),
                "causal_mean_abs_target_logit_change": sum(abs_drops) / max(1, len(abs_drops)),
                "causal_mean_target_logit_drop": sum(drops) / max(1, len(drops)),
                "causal_mean_positive_target_logit_drop": sum(positives) / max(1, len(positives)),
                "causal_mean_abs_target_probability_change": sum(abs_prob_drops) / max(1, len(abs_prob_drops)),
                "causal_mean_target_probability_drop": sum(prob_drops) / max(1, len(prob_drops)),
                "causal_positive_doc_rate": len(positives) / max(1, len(drops)),
                "causal_active_doc_rate": len(active_docs) / max(1, len(drops)),
                "causal_docs": len(drops),
                "target_class": target_class,
                "activation_layer": int(args.activation_layer),
            }
        )
        ranked_examples = []
        for example in examples:
            doc_id = str(example.get("doc_id", ""))
            effect = effects.get(doc_id)
            if effect is None:
                continue
            ranked_examples.append({**example, **effect})
        ranked_examples.sort(
            key=lambda row: (
                float(row["causal_abs_logit_change"]),
                abs(float(row["causal_probability_drop"])),
                float(row.get("peak_activation", row.get("activation", 0.0))),
            ),
            reverse=True,
        )
        for example_rank, example in enumerate(ranked_examples, start=1):
            example["example_rank"] = example_rank
            example_rows.append(
                {
                    "latent_id": latent_id,
                    "original_rank": latent.get("rank", latent_offset),
                    **example,
                }
            )
        for doc_id, effect in effects.items():
            metadata = documents.get(doc_id, {})
            generation_impact_rows.append(
                {
                    "latent_id": latent_id,
                    "doc_id": doc_id,
                    "turn_id": metadata.get("turn_id", ""),
                    "model": metadata.get("model", docs_by_id[doc_id].source),
                    **effect,
                }
            )
        if args.progress_every > 0 and (
            latent_offset == 1 or latent_offset % int(args.progress_every) == 0
        ):
            print(
                f"Ablated {latent_offset:,}/{len(selected_ids):,} latents "
                f"on displayed explorer docs",
                flush=True,
            )

    latent_rows.sort(
        key=lambda row: (
            float(row["causal_mean_abs_target_logit_change"]),
            abs(float(row["causal_mean_target_logit_drop"])),
            float(row["causal_active_doc_rate"]),
            float(row.get("ai_human_log_odds", 0.0)),
        ),
        reverse=True,
    )
    for rank, row in enumerate(latent_rows, start=1):
        row["causal_rank"] = rank
        row["rank"] = rank

    reranked_latents = []
    reranked_latent_docs: dict[str, list[dict[str, Any]]] = {}
    for row in latent_rows:
        latent_id_text = str(int(row["latent_id"]))
        payload = dict(row)
        reranked_latents.append(payload)
        docs = [
            example
            for example in example_rows
            if str(int(example["latent_id"])) == latent_id_text
        ]
        for example in docs:
            example["rank"] = int(row["rank"])
        reranked_latent_docs[latent_id_text] = docs

    untouched_latents = [
        row for row in index.get("latents", []) if str(int(row["latent_id"])) not in set(selected_ids)
    ]
    index["latents"] = reranked_latents + untouched_latents
    for latent_id, docs in reranked_latent_docs.items():
        index["latentDocs"][latent_id] = docs
    index["generationImpacts"] = generation_impacts
    index["causalRerank"] = {
        "method": "exact_layer_ablation_on_displayed_prompt_generations",
        "activation_layer": int(args.activation_layer),
        "target_class": target_class,
        "top_latents": int(args.top_latents),
        "examples_per_latent": int(args.examples_per_latent),
        "selected_documents": len(selected_docs),
        "score": "absolute_baseline_target_logit_minus_ablated_target_logit",
        "signed_score": "baseline_target_logit_minus_ablated_target_logit",
    }

    output_index = args.output_browser_index or args.browser_index
    output_index.write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "pangram_llama_sae_causal_reranked_latents.csv", latent_rows)
    _write_csv(args.output_dir / "pangram_llama_sae_causal_reranked_examples.csv", example_rows)
    _write_csv(args.output_dir / "pangram_llama_sae_causal_generation_impacts.csv", generation_impact_rows)
    summary = {
        "browser_index": str(args.browser_index),
        "doc_tokens_parquet": str(args.doc_tokens_parquet),
        "sae_path": str(args.sae_path),
        "output_browser_index": str(output_index),
        "output_dir": str(args.output_dir),
        "activation_layer": int(args.activation_layer),
        "target_class": target_class,
        "class_diffs": class_diffs,
        "top_latents": int(args.top_latents),
        "examples_per_latent": int(args.examples_per_latent),
        "selected_documents": len(selected_docs),
        "generation_impacts": sum(len(rows) for rows in generation_impacts.values()),
        "source_counts": dict(sorted(Counter(doc.source for doc in selected_docs).items())),
        "label_counts": dict(sorted(Counter(str(doc.label) for doc in selected_docs).items())),
        "sae_summary": sae_summary,
    }
    _write_json(args.output_dir / "pangram_llama_sae_causal_rerank_summary.json", summary)
    return summary


def main() -> None:
    summary = run(build_parser().parse_args())
    print(
        "Wrote causal-reranked latent explorer index to "
        f"{summary['output_browser_index']} "
        f"({summary['top_latents']} latents, {summary['selected_documents']} docs)",
        flush=True,
    )


if __name__ == "__main__":
    main()
