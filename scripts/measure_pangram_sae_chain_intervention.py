from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from torch import nn

from rerank_latent_explorer_by_causal_ablation import (
    ExplorerDoc,
    _as_sae_doc,
    _batched,
    _llama_layer,
    _softmax_target,
)
from slop_sftdiv.cli.pangram_llama_sae import (
    LLAMA_BASE_ID,
    PANGRAM_ADAPTER_ID,
    _device,
    _encode_docs,
    load_pangram_model,
)
from slop_sftdiv.cli.phase4_batchtopk_sae import _load_sae


DEFAULT_SAE_PATHS = {
    16: Path("artifacts/pangram_llama_sae/layer16_full_glm52_cap640_k64_e50/pangram_llama_batchtopk_sae.pt"),
    19: Path("artifacts/pangram_llama_sae/layer19_full_neuralwatt_cap640_k64_e50/pangram_llama_batchtopk_sae.pt"),
    20: Path("artifacts/pangram_llama_sae/layer20_full_glm52_cap640_k64_e50/pangram_llama_batchtopk_sae.pt"),
    24: Path("artifacts/pangram_llama_sae/layer24_full_neuralwatt_cap640_k64_e50/pangram_llama_batchtopk_sae.pt"),
}


def _parse_node(raw: str) -> tuple[int, int]:
    raw = raw.strip()
    if not raw.startswith("L") or ":" not in raw:
        raise ValueError(f"invalid node {raw!r}; expected L19:1234")
    layer_text, latent_text = raw.split(":", 1)
    return int(layer_text[1:]), int(latent_text)


def _parse_sae_map(values: list[str]) -> dict[int, Path]:
    out = dict(DEFAULT_SAE_PATHS)
    for value in values:
        layer_text, path_text = value.split("=", 1)
        out[int(layer_text)] = Path(path_text)
    return out


def _activation_mask(encoded: dict[str, Any]) -> torch.Tensor:
    mask = encoded["attention_mask"].bool()
    special = encoded.get("special_tokens_mask")
    if isinstance(special, torch.Tensor):
        mask = mask & ~special.bool()
    return mask


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


def _empty_stats(count: int) -> list[dict[str, float | int]]:
    return [
        {
            "mass": 0.0,
            "max": 0.0,
            "active_tokens": 0,
            "peak_token_index": -1,
        }
        for _ in range(count)
    ]


def _accumulate_stats(
    stats: list[dict[str, float | int]],
    *,
    token_doc_indexes: torch.Tensor,
    token_indexes: torch.Tensor,
    latent_codes: torch.Tensor,
) -> None:
    codes_cpu = latent_codes.detach().float().cpu()
    doc_cpu = token_doc_indexes.detach().cpu()
    tok_cpu = token_indexes.detach().cpu()
    for code, doc_index, token_index in zip(codes_cpu.tolist(), doc_cpu.tolist(), tok_cpu.tolist(), strict=True):
        if code <= 0:
            continue
        row = stats[int(doc_index)]
        row["mass"] = float(row["mass"]) + float(code)
        row["active_tokens"] = int(row["active_tokens"]) + 1
        if float(code) > float(row["max"]):
            row["max"] = float(code)
            row["peak_token_index"] = int(token_index)


def _load_target_docs(target_csv: Path, doc_tokens_parquet: Path, *, limit_docs: int | None) -> list[ExplorerDoc]:
    targets = pd.read_csv(target_csv)
    if "text" in targets.columns:
        rows = targets.to_dict(orient="records")
        if limit_docs is not None:
            rows = rows[: int(limit_docs)]
        docs: list[ExplorerDoc] = []
        for row in rows:
            source = str(row.get("model") or row.get("source") or "unknown")
            docs.append(
                ExplorerDoc(
                    doc_id=str(row["doc_id"]),
                    source=source,
                    label=0 if source == "human" else 1,
                    text=str(row["text"]),
                )
            )
        return docs
    wanted_doc_ids = list(dict.fromkeys(str(value) for value in targets["doc_id"].tolist()))
    if limit_docs is not None:
        wanted_doc_ids = wanted_doc_ids[: int(limit_docs)]
    table = pd.read_parquet(doc_tokens_parquet, columns=["doc_id", "model", "text"])
    rows_by_id = {str(row.doc_id): row for row in table.itertuples(index=False)}
    missing = [doc_id for doc_id in wanted_doc_ids if doc_id not in rows_by_id]
    if missing:
        raise ValueError(f"{len(missing)} target docs are missing from {doc_tokens_parquet}: {missing[:3]}")
    docs: list[ExplorerDoc] = []
    for doc_id in wanted_doc_ids:
        row = rows_by_id[doc_id]
        source = str(row.model)
        docs.append(
            ExplorerDoc(
                doc_id=doc_id,
                source=source,
                label=0 if source == "human" else 1,
                text=str(row.text),
            )
        )
    return docs


@torch.inference_mode()
def _run_pass(
    *,
    model: Any,
    tokenizer: Any,
    docs: list[ExplorerDoc],
    args: argparse.Namespace,
    device: torch.device,
    downstream_layer: nn.Module,
    downstream_sae: Any,
    downstream_latent: int,
    upstream_layer: nn.Module | None = None,
    upstream_sae: Any | None = None,
    upstream_latents: list[int] | None = None,
) -> tuple[torch.Tensor, list[dict[str, float | int]], list[dict[str, float | int]]]:
    logits_parts: list[torch.Tensor] = []
    downstream_all: list[dict[str, float | int]] = []
    upstream_all: list[dict[str, float | int]] = []
    decoder_weight = upstream_sae.decoder.weight.detach() if upstream_sae is not None else None
    for batch_docs in _batched(docs, int(args.batch_size)):
        encoded = _encode_docs(
            tokenizer,
            [_as_sae_doc(doc) for doc in batch_docs],
            max_length=int(args.max_length),
            device=device,
            pad_to_max_length=True,
        )
        token_mask = _activation_mask(encoded)
        positions = torch.nonzero(token_mask, as_tuple=False)
        token_doc_indexes = positions[:, 0] if positions.numel() else torch.empty(0, dtype=torch.long, device=device)
        token_indexes = positions[:, 1] if positions.numel() else torch.empty(0, dtype=torch.long, device=device)
        downstream_stats = _empty_stats(len(batch_docs))
        upstream_stats = _empty_stats(len(batch_docs))

        def upstream_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            if upstream_sae is None or not upstream_latents or decoder_weight is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                return output
            token_vectors = hidden.float()[token_mask]
            if token_vectors.numel() == 0:
                return output
            contribution = torch.zeros_like(token_vectors)
            combined_codes = torch.zeros(token_vectors.shape[0], dtype=torch.float32, device=token_vectors.device)
            for upstream_latent in upstream_latents:
                latent_codes = _browser_latent_codes(
                    upstream_sae,
                    token_vectors,
                    latent_id=int(upstream_latent),
                    top_k=int(args.browser_top_k),
                )
                combined_codes = combined_codes + latent_codes
                contribution = contribution + latent_codes.unsqueeze(1) * decoder_weight[:, int(upstream_latent)].unsqueeze(0)
            _accumulate_stats(
                upstream_stats,
                token_doc_indexes=token_doc_indexes,
                token_indexes=token_indexes,
                latent_codes=combined_codes,
            )
            modified = hidden.float().clone()
            modified[token_mask] = token_vectors - contribution
            modified = modified.to(dtype=hidden.dtype)
            if isinstance(output, tuple):
                return (modified, *output[1:])
            return modified

        def downstream_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                return output
            token_vectors = hidden.float()[token_mask]
            if token_vectors.numel() == 0:
                return output
            latent_codes = _browser_latent_codes(
                downstream_sae,
                token_vectors,
                latent_id=int(downstream_latent),
                top_k=int(args.browser_top_k),
            )
            _accumulate_stats(
                downstream_stats,
                token_doc_indexes=token_doc_indexes,
                token_indexes=token_indexes,
                latent_codes=latent_codes,
            )
            return output

        handles = []
        if upstream_layer is not None:
            handles.append(upstream_layer.register_forward_hook(upstream_hook))
        handles.append(downstream_layer.register_forward_hook(downstream_hook))
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
        downstream_all.extend(downstream_stats)
        upstream_all.extend(upstream_stats)
    return torch.cat(logits_parts, dim=0), downstream_all, upstream_all


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "doc_id",
        "source",
        "upstream_node",
        "downstream_node",
        "target_class",
        "baseline_target_logit",
        "ablated_target_logit",
        "target_logit_drop",
        "baseline_target_probability",
        "ablated_target_probability",
        "target_probability_drop",
        "upstream_baseline_mass",
        "downstream_baseline_mass",
        "downstream_ablated_mass",
        "downstream_mass_delta",
        "downstream_baseline_max",
        "downstream_ablated_max",
        "downstream_max_delta",
        "downstream_baseline_active_tokens",
        "downstream_ablated_active_tokens",
        "downstream_active_tokens_delta",
    ]
    columns = [column for column in preferred if column in columns] + [
        column for column in columns if column not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whether ablating an upstream Pangram SAE latent changes downstream SAE latent activity."
    )
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--doc-tokens-parquet", type=Path, default=Path("apps/latent-explorer/public/data/layer19/pangram_llama_sae_doc_tokens.parquet"))
    parser.add_argument("--upstream-node", required=True)
    parser.add_argument("--extra-upstream-node", action="append", default=[], help="Additional same-layer upstream node to ablate jointly.")
    parser.add_argument("--downstream-node", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sae", action="append", default=[], metavar="LAYER=PATH")
    parser.add_argument("--target-class", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
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
    upstream_layer_index, upstream_latent = _parse_node(args.upstream_node)
    extra_upstream_nodes = list(args.extra_upstream_node or [])
    extra_upstream_latents: list[int] = []
    for raw_node in extra_upstream_nodes:
        extra_layer_index, extra_latent = _parse_node(raw_node)
        if extra_layer_index != upstream_layer_index:
            raise ValueError("--extra-upstream-node must be on the same layer as --upstream-node")
        extra_upstream_latents.append(extra_latent)
    upstream_latents = [upstream_latent, *extra_upstream_latents]
    upstream_node_label = "+".join([args.upstream_node, *extra_upstream_nodes])
    downstream_layer_index, downstream_latent = _parse_node(args.downstream_node)
    if upstream_layer_index >= downstream_layer_index:
        raise ValueError("--upstream-node layer must be earlier than --downstream-node layer")
    sae_paths = _parse_sae_map(args.sae)
    for layer_index in [upstream_layer_index, downstream_layer_index]:
        if not sae_paths[layer_index].is_file():
            raise FileNotFoundError(
                f"Missing SAE checkpoint for layer {layer_index}: {sae_paths[layer_index]}"
            )
    docs = _load_target_docs(args.target_csv, args.doc_tokens_parquet, limit_docs=args.limit_docs)
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model, tokenizer = load_pangram_model(args, device=device)
    model.eval()
    upstream_layer = _llama_layer(model, upstream_layer_index)
    downstream_layer = _llama_layer(model, downstream_layer_index)
    upstream_sae, upstream_summary = _load_sae(sae_paths[upstream_layer_index], device=device)
    downstream_sae, downstream_summary = _load_sae(sae_paths[downstream_layer_index], device=device)
    upstream_sae.eval()
    downstream_sae.eval()

    ablated_logits, downstream_ablated, upstream_ablated = _run_pass(
        model=model,
        tokenizer=tokenizer,
        docs=docs,
        args=args,
        device=device,
        downstream_layer=downstream_layer,
        downstream_sae=downstream_sae,
        downstream_latent=downstream_latent,
        upstream_layer=upstream_layer,
        upstream_sae=upstream_sae,
        upstream_latents=upstream_latents,
    )
    # The two passes both apply upstream ablation above; rerun baseline without the upstream hook.
    baseline_logits, downstream_base, _unused = _run_pass(
        model=model,
        tokenizer=tokenizer,
        docs=docs,
        args=args,
        device=device,
        downstream_layer=downstream_layer,
        downstream_sae=downstream_sae,
        downstream_latent=downstream_latent,
    )
    target = int(args.target_class)
    baseline_probs = _softmax_target(baseline_logits, target)
    ablated_probs = _softmax_target(ablated_logits, target)
    rows: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        base_down = downstream_base[index]
        ablated_down = downstream_ablated[index]
        up = upstream_ablated[index]
        rows.append(
            {
                "doc_id": doc.doc_id,
                "source": doc.source,
                "upstream_node": upstream_node_label,
                "downstream_node": args.downstream_node,
                "target_class": target,
                "baseline_target_logit": float(baseline_logits[index, target]),
                "ablated_target_logit": float(ablated_logits[index, target]),
                "target_logit_drop": float(baseline_logits[index, target] - ablated_logits[index, target]),
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
    summary = {
        "target_csv": str(args.target_csv),
        "doc_tokens_parquet": str(args.doc_tokens_parquet),
        "docs": len(docs),
        "upstream_node": upstream_node_label,
        "upstream_nodes": [args.upstream_node, *extra_upstream_nodes],
        "downstream_node": args.downstream_node,
        "upstream_sae": str(sae_paths[upstream_layer_index]),
        "downstream_sae": str(sae_paths[downstream_layer_index]),
        "target_class": target,
        "mean_target_logit_drop": sum(row["target_logit_drop"] for row in rows) / max(1, len(rows)),
        "mean_abs_target_logit_drop": sum(abs(row["target_logit_drop"]) for row in rows) / max(1, len(rows)),
        "mean_downstream_mass_delta": sum(row["downstream_mass_delta"] for row in rows) / max(1, len(rows)),
        "mean_abs_downstream_mass_delta": sum(abs(row["downstream_mass_delta"]) for row in rows) / max(1, len(rows)),
        "mean_downstream_max_delta": sum(row["downstream_max_delta"] for row in rows) / max(1, len(rows)),
        "mean_abs_downstream_max_delta": sum(abs(row["downstream_max_delta"]) for row in rows) / max(1, len(rows)),
        "upstream_sae_summary": upstream_summary,
        "downstream_sae_summary": downstream_summary,
    }
    _write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
