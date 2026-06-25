from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch

from measure_pangram_sae_chain_intervention import (
    DEFAULT_SAE_PATHS,
    _load_target_docs,
    _parse_node,
    _parse_sae_map,
    _run_pass,
    _write_csv,
)
from rerank_latent_explorer_by_causal_ablation import _llama_layer, _softmax_target
from slop_sftdiv.cli.pangram_llama_sae import (
    LLAMA_BASE_ID,
    PANGRAM_ADAPTER_ID,
    _device,
    load_pangram_model,
)
from slop_sftdiv.cli.phase4_batchtopk_sae import _load_sae


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _details_for_pair(
    *,
    config: dict[str, Any],
    docs: list[Any],
    baseline_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    downstream_base: list[dict[str, float | int]],
    downstream_ablated: list[dict[str, float | int]],
    upstream_ablated: list[dict[str, float | int]],
    target_class: int,
) -> list[dict[str, Any]]:
    baseline_probs = _softmax_target(baseline_logits, target_class)
    ablated_probs = _softmax_target(ablated_logits, target_class)
    rows: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        base_down = downstream_base[index]
        ablated_down = downstream_ablated[index]
        up = upstream_ablated[index]
        rows.append(
            {
                **config,
                "doc_id": doc.doc_id,
                "source": doc.source,
                "target_class": target_class,
                "baseline_target_logit": float(baseline_logits[index, target_class]),
                "ablated_target_logit": float(ablated_logits[index, target_class]),
                "target_logit_drop": float(baseline_logits[index, target_class] - ablated_logits[index, target_class]),
                "baseline_target_probability": float(baseline_probs[index]),
                "ablated_target_probability": float(ablated_probs[index]),
                "target_probability_drop": float(baseline_probs[index] - ablated_probs[index]),
                "upstream_baseline_mass": float(up["mass"]),
                "upstream_baseline_max": float(up["max"]),
                "upstream_baseline_active_tokens": int(up["active_tokens"]),
                "upstream_peak_token_index": int(up["peak_token_index"]),
                "downstream_baseline_mass": float(base_down["mass"]),
                "downstream_ablated_mass": float(ablated_down["mass"]),
                "downstream_mass_delta": float(ablated_down["mass"]) - float(base_down["mass"]),
                "downstream_baseline_max": float(base_down["max"]),
                "downstream_ablated_max": float(ablated_down["max"]),
                "downstream_max_delta": float(ablated_down["max"]) - float(base_down["max"]),
                "downstream_baseline_active_tokens": int(base_down["active_tokens"]),
                "downstream_ablated_active_tokens": int(ablated_down["active_tokens"]),
                "downstream_active_tokens_delta": int(ablated_down["active_tokens"]) - int(base_down["active_tokens"]),
                "downstream_baseline_peak_token_index": int(base_down["peak_token_index"]),
                "downstream_ablated_peak_token_index": int(ablated_down["peak_token_index"]),
            }
        )
    return rows


def _summarize(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **config,
        "docs": len(rows),
        "mean_target_logit_drop": _mean([float(row["target_logit_drop"]) for row in rows]),
        "mean_abs_target_logit_drop": _mean([abs(float(row["target_logit_drop"])) for row in rows]),
        "mean_downstream_mass_delta": _mean([float(row["downstream_mass_delta"]) for row in rows]),
        "mean_abs_downstream_mass_delta": _mean([abs(float(row["downstream_mass_delta"])) for row in rows]),
        "mean_downstream_max_delta": _mean([float(row["downstream_max_delta"]) for row in rows]),
        "mean_abs_downstream_max_delta": _mean([abs(float(row["downstream_max_delta"])) for row in rows]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Pangram SAE intervention control panel with one model load."
    )
    parser.add_argument("--control-csv", type=Path, required=True)
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
    panel = pd.read_csv(args.control_csv)
    if args.limit_rows is not None:
        panel = panel.head(int(args.limit_rows))
    sae_paths = _parse_sae_map(args.sae)
    needed_layers: set[int] = set()
    for row in panel.itertuples(index=False):
        upstream_layer, _ = _parse_node(str(row.upstream_node))
        downstream_layer, _ = _parse_node(str(row.downstream_node))
        needed_layers.update([upstream_layer, downstream_layer])
    missing = [layer for layer in sorted(needed_layers) if not sae_paths.get(layer, DEFAULT_SAE_PATHS.get(layer, Path())).is_file()]
    if missing:
        raise FileNotFoundError(f"missing SAE checkpoints for layers: {missing}")

    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model, tokenizer = load_pangram_model(args, device=device)
    model.eval()
    layers = {layer: _llama_layer(model, layer) for layer in needed_layers}
    saes: dict[int, Any] = {}
    for layer in sorted(needed_layers):
        sae, _summary = _load_sae(sae_paths[layer], device=device)
        sae.eval()
        saes[layer] = sae

    docs_cache: dict[str, list[Any]] = {}
    baseline_cache: dict[tuple[str, str], tuple[torch.Tensor, list[dict[str, float | int]]]] = {}
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(panel.to_dict(orient="records"), start=1):
        target_csv = str(row["target_csv"])
        docs = docs_cache.get(target_csv)
        if docs is None:
            docs = _load_target_docs(Path(target_csv), args.doc_tokens_parquet, limit_docs=args.limit_docs)
            docs_cache[target_csv] = docs
        upstream_layer_index, upstream_latent = _parse_node(str(row["upstream_node"]))
        downstream_layer_index, downstream_latent = _parse_node(str(row["downstream_node"]))
        if upstream_layer_index >= downstream_layer_index:
            raise ValueError(f"invalid row {row_index}: upstream layer must be earlier than downstream layer")
        config = {
            key: row.get(key, "")
            for key in [
                "target_name",
                "target_csv",
                "control_type",
                "upstream_node",
                "downstream_node",
                "chain_upstream_node",
                "chain_downstream_node",
                "candidate_rank",
                "chain_rank",
                "rank_distance",
                "shared_prompts_with_anchor",
                "chain_shared_prompts_with_anchor",
                "selection_score",
            ]
        }
        print(f"[{row_index}/{len(panel)}] {config['upstream_node']} -> {config['downstream_node']}", flush=True)
        ablated_logits, downstream_ablated, upstream_ablated = _run_pass(
            model=model,
            tokenizer=tokenizer,
            docs=docs,
            args=args,
            device=device,
            downstream_layer=layers[downstream_layer_index],
            downstream_sae=saes[downstream_layer_index],
            downstream_latent=downstream_latent,
            upstream_layer=layers[upstream_layer_index],
            upstream_sae=saes[upstream_layer_index],
            upstream_latents=[upstream_latent],
        )
        baseline_key = (target_csv, str(row["downstream_node"]))
        if baseline_key not in baseline_cache:
            baseline_logits, downstream_base, _unused = _run_pass(
                model=model,
                tokenizer=tokenizer,
                docs=docs,
                args=args,
                device=device,
                downstream_layer=layers[downstream_layer_index],
                downstream_sae=saes[downstream_layer_index],
                downstream_latent=downstream_latent,
            )
            baseline_cache[baseline_key] = (baseline_logits, downstream_base)
        baseline_logits, downstream_base = baseline_cache[baseline_key]
        rows = _details_for_pair(
            config=config,
            docs=docs,
            baseline_logits=cast(torch.Tensor, baseline_logits),
            ablated_logits=ablated_logits,
            downstream_base=downstream_base,
            downstream_ablated=downstream_ablated,
            upstream_ablated=upstream_ablated,
            target_class=int(args.target_class),
        )
        detail_rows.extend(rows)
        summary_rows.append(_summarize(config, rows))

    _write_csv(args.output_details_csv, detail_rows)
    _write_csv(args.output_summary_csv, summary_rows)
    payload = {
        "control_csv": str(args.control_csv),
        "rows": len(summary_rows),
        "detail_rows": len(detail_rows),
        "target_class": int(args.target_class),
        "summary": summary_rows,
    }
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ["control_csv", "rows", "detail_rows", "target_class"]}, indent=2))


if __name__ == "__main__":
    main()
