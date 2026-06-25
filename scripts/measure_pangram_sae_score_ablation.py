from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from measure_pangram_sae_chain_intervention import (
    DEFAULT_SAE_PATHS,
    _activation_mask,
    _browser_latent_codes,
    _load_target_docs,
    _parse_node,
    _parse_sae_map,
)
from rerank_latent_explorer_by_causal_ablation import _as_sae_doc, _batched, _llama_layer, _softmax_target
from slop_sftdiv.cli.pangram_llama_sae import (
    LLAMA_BASE_ID,
    PANGRAM_ADAPTER_ID,
    _device,
    _encode_docs,
    load_pangram_model,
)
from slop_sftdiv.cli.phase4_batchtopk_sae import _load_sae


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


@torch.inference_mode()
def _run_logits(
    *,
    model: Any,
    tokenizer: Any,
    docs: list[Any],
    args: argparse.Namespace,
    device: torch.device,
    layer_modules: dict[int, nn.Module],
    saes: dict[int, Any],
    nodes_by_layer: dict[int, list[int]],
) -> torch.Tensor:
    logits_parts: list[torch.Tensor] = []
    decoder_weights = {
        layer: saes[layer].decoder.weight.detach()
        for layer in nodes_by_layer
    }
    for batch_docs in _batched(docs, int(args.batch_size)):
        encoded = _encode_docs(
            tokenizer,
            [_as_sae_doc(doc) for doc in batch_docs],
            max_length=int(args.max_length),
            device=device,
            pad_to_max_length=True,
        )
        token_mask = _activation_mask(encoded)

        def make_hook(layer_index: int):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                if not isinstance(hidden, torch.Tensor):
                    return output
                token_vectors = hidden.float()[token_mask]
                if token_vectors.numel() == 0:
                    return output
                modified_tokens = token_vectors
                for latent_id in nodes_by_layer[layer_index]:
                    latent_codes = _browser_latent_codes(
                        saes[layer_index],
                        token_vectors,
                        latent_id=int(latent_id),
                        top_k=int(args.browser_top_k),
                    )
                    contribution = (
                        latent_codes.unsqueeze(1)
                        * decoder_weights[layer_index][:, int(latent_id)].unsqueeze(0)
                    )
                    modified_tokens = modified_tokens - contribution
                modified = hidden.float().clone()
                modified[token_mask] = modified_tokens
                modified = modified.to(dtype=hidden.dtype)
                if isinstance(output, tuple):
                    return (modified, *output[1:])
                return modified

            return hook

        handles = [
            layer_modules[layer].register_forward_hook(make_hook(layer))
            for layer in sorted(nodes_by_layer)
        ]
        try:
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        logits_parts.append(cast(torch.Tensor, outputs.logits).detach().float().cpu())
    return torch.cat(logits_parts, dim=0)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "doc_id",
        "source",
        "nodes",
        "target_class",
        "baseline_target_logit",
        "ablated_target_logit",
        "target_logit_drop",
        "baseline_target_probability",
        "ablated_target_probability",
        "target_probability_drop",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure detector score effects from ablating one or more Pangram SAE latents.")
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--doc-tokens-parquet", type=Path, default=Path("apps/latent-explorer/public/data/layer19/pangram_llama_sae_doc_tokens.parquet"))
    parser.add_argument("--node", action="append", required=True, help="SAE node to ablate, e.g. L19:3450.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sae", action="append", default=[], metavar="LAYER=PATH")
    parser.add_argument("--target-class", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--browser-top-k", type=int, default=64)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--base-model", default=LLAMA_BASE_ID)
    parser.add_argument("--adapter-model", default=PANGRAM_ADAPTER_ID)
    parser.add_argument("--local-model-dir", type=Path, default=Path("artifacts/models/pangram_editlens_Llama-3.2-3B_unsloth_base"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--adapter-revision", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--no-hf-token", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes_by_layer: defaultdict[int, list[int]] = defaultdict(list)
    for node in args.node:
        layer, latent = _parse_node(node)
        nodes_by_layer[layer].append(latent)
    sae_paths = _parse_sae_map(args.sae)
    for layer in nodes_by_layer:
        if not sae_paths.get(layer, DEFAULT_SAE_PATHS.get(layer, Path())).is_file():
            raise FileNotFoundError(f"Missing SAE checkpoint for layer {layer}: {sae_paths.get(layer)}")

    docs = _load_target_docs(args.target_csv, args.doc_tokens_parquet, limit_docs=args.limit_docs)
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model, tokenizer = load_pangram_model(args, device=device)
    model.eval()
    layer_modules = {layer: _llama_layer(model, layer) for layer in nodes_by_layer}
    saes: dict[int, Any] = {}
    for layer in nodes_by_layer:
        sae, _summary = _load_sae(sae_paths[layer], device=device)
        sae.eval()
        saes[layer] = sae

    baseline_logits = _run_logits(
        model=model,
        tokenizer=tokenizer,
        docs=docs,
        args=args,
        device=device,
        layer_modules={},
        saes={},
        nodes_by_layer={},
    )
    ablated_logits = _run_logits(
        model=model,
        tokenizer=tokenizer,
        docs=docs,
        args=args,
        device=device,
        layer_modules=layer_modules,
        saes=saes,
        nodes_by_layer=dict(nodes_by_layer),
    )
    target = int(args.target_class)
    baseline_probs = _softmax_target(baseline_logits, target)
    ablated_probs = _softmax_target(ablated_logits, target)
    node_label = "+".join(args.node)
    rows: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        rows.append(
            {
                "doc_id": doc.doc_id,
                "source": doc.source,
                "nodes": node_label,
                "target_class": target,
                "baseline_target_logit": float(baseline_logits[index, target]),
                "ablated_target_logit": float(ablated_logits[index, target]),
                "target_logit_drop": float(baseline_logits[index, target] - ablated_logits[index, target]),
                "baseline_target_probability": float(baseline_probs[index]),
                "ablated_target_probability": float(ablated_probs[index]),
                "target_probability_drop": float(baseline_probs[index] - ablated_probs[index]),
            }
        )
    summary = {
        "target_csv": str(args.target_csv),
        "docs": len(docs),
        "nodes": args.node,
        "target_class": target,
        "mean_target_logit_drop": _mean([float(row["target_logit_drop"]) for row in rows]),
        "mean_abs_target_logit_drop": _mean([abs(float(row["target_logit_drop"])) for row in rows]),
        "mean_target_probability_drop": _mean([float(row["target_probability_drop"]) for row in rows]),
        "mean_abs_target_probability_drop": _mean([abs(float(row["target_probability_drop"])) for row in rows]),
    }
    _write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
