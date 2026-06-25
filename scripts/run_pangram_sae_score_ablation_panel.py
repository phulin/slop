from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from measure_pangram_sae_chain_intervention import (
    DEFAULT_SAE_PATHS,
    _load_target_docs,
    _parse_node,
    _parse_sae_map,
)
from measure_pangram_sae_score_ablation import _mean, _run_logits
from rerank_latent_explorer_by_causal_ablation import _llama_layer, _softmax_target
from slop_sftdiv.cli.pangram_llama_sae import LLAMA_BASE_ID, PANGRAM_ADAPTER_ID, _device, load_pangram_model
from slop_sftdiv.cli.phase4_batchtopk_sae import _load_sae


def _parse_nodes(value: str) -> list[str]:
    nodes = [node for node in re.split(r"[+;, ]+", value.strip()) if node]
    if not nodes:
        raise ValueError("empty nodes field")
    return nodes


def _nodes_by_layer(nodes: list[str]) -> dict[int, list[int]]:
    by_layer: defaultdict[int, list[int]] = defaultdict(list)
    for node in nodes:
        layer, latent = _parse_node(node)
        by_layer[layer].append(latent)
    return dict(by_layer)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "target_name",
        "target_csv",
        "ablation_label",
        "control_type",
        "nodes",
        "anchor_nodes",
        "doc_id",
        "source",
        "target_class",
        "baseline_target_logit",
        "ablated_target_logit",
        "target_logit_drop",
        "baseline_target_probability",
        "ablated_target_probability",
        "target_probability_drop",
    ]
    columns = preferred + [column for column in columns if column not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **config,
        "docs": len(rows),
        "mean_target_logit_drop": _mean([float(row["target_logit_drop"]) for row in rows]),
        "mean_abs_target_logit_drop": _mean([abs(float(row["target_logit_drop"])) for row in rows]),
        "mean_target_probability_drop": _mean([float(row["target_probability_drop"]) for row in rows]),
        "mean_abs_target_probability_drop": _mean([abs(float(row["target_probability_drop"])) for row in rows]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run many Pangram SAE score ablations with one model load.")
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--doc-tokens-parquet", type=Path, default=Path("apps/latent-explorer/public/data/layer19/pangram_llama_sae_doc_tokens.parquet"))
    parser.add_argument("--output-details-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path, required=True)
    parser.add_argument("--sae", action="append", default=[], metavar="LAYER=PATH")
    parser.add_argument("--target-class", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--browser-top-k", type=int, default=64)
    parser.add_argument("--limit-rows", type=int, default=None)
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
    panel = pd.read_csv(args.panel_csv)
    if args.limit_rows is not None:
        panel = panel.head(int(args.limit_rows))

    needed_layers: set[int] = set()
    parsed_nodes: list[list[str]] = []
    for row in panel.to_dict(orient="records"):
        nodes = _parse_nodes(str(row["nodes"]))
        parsed_nodes.append(nodes)
        needed_layers.update(_nodes_by_layer(nodes))

    sae_paths = _parse_sae_map(args.sae)
    missing = [layer for layer in sorted(needed_layers) if not sae_paths.get(layer, DEFAULT_SAE_PATHS.get(layer, Path())).is_file()]
    if missing:
        raise FileNotFoundError(f"missing SAE checkpoints for layers: {missing}")

    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model, tokenizer = load_pangram_model(args, device=device)
    model.eval()
    layer_modules = {layer: _llama_layer(model, layer) for layer in sorted(needed_layers)}
    saes: dict[int, Any] = {}
    for layer in sorted(needed_layers):
        sae, _summary = _load_sae(sae_paths[layer], device=device)
        sae.eval()
        saes[layer] = sae

    docs_cache: dict[str, list[Any]] = {}
    baseline_cache: dict[str, torch.Tensor] = {}
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    panel_rows = panel.to_dict(orient="records")
    for row_index, (row, nodes) in enumerate(zip(panel_rows, parsed_nodes, strict=True), start=1):
        target_csv = str(row["target_csv"])
        docs = docs_cache.get(target_csv)
        if docs is None:
            docs = _load_target_docs(Path(target_csv), args.doc_tokens_parquet, limit_docs=args.limit_docs)
            docs_cache[target_csv] = docs
        baseline_logits = baseline_cache.get(target_csv)
        if baseline_logits is None:
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
            baseline_cache[target_csv] = baseline_logits
        nodes_by_layer = _nodes_by_layer(nodes)
        print(f"[{row_index}/{len(panel_rows)}] {target_csv} :: {'+'.join(nodes)}", flush=True)
        ablated_logits = _run_logits(
            model=model,
            tokenizer=tokenizer,
            docs=docs,
            args=args,
            device=device,
            layer_modules=layer_modules,
            saes=saes,
            nodes_by_layer=nodes_by_layer,
        )
        target = int(args.target_class)
        baseline_probs = _softmax_target(baseline_logits, target)
        ablated_probs = _softmax_target(ablated_logits, target)
        config = {key: row.get(key, "") for key in panel.columns}
        rows: list[dict[str, Any]] = []
        for doc_index, doc in enumerate(docs):
            detail = {
                **config,
                "doc_id": doc.doc_id,
                "source": doc.source,
                "target_class": target,
                "baseline_target_logit": float(baseline_logits[doc_index, target]),
                "ablated_target_logit": float(ablated_logits[doc_index, target]),
                "target_logit_drop": float(baseline_logits[doc_index, target] - ablated_logits[doc_index, target]),
                "baseline_target_probability": float(baseline_probs[doc_index]),
                "ablated_target_probability": float(ablated_probs[doc_index]),
                "target_probability_drop": float(baseline_probs[doc_index] - ablated_probs[doc_index]),
            }
            rows.append(detail)
        detail_rows.extend(rows)
        summary_rows.append(_summarize(config, rows))

    _write_csv(args.output_details_csv, detail_rows)
    _write_csv(args.output_summary_csv, summary_rows)
    payload = {
        "panel_csv": str(args.panel_csv),
        "rows": len(summary_rows),
        "detail_rows": len(detail_rows),
        "target_class": int(args.target_class),
        "summary": summary_rows,
    }
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ["panel_csv", "rows", "detail_rows", "target_class"]}, indent=2))


if __name__ == "__main__":
    main()
