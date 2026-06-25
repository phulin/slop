from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pandas as pd
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
from rerank_latent_explorer_by_causal_ablation import _as_sae_doc, _batched, _llama_layer
from slop_sftdiv.cli.pangram_llama_sae import (
    LLAMA_BASE_ID,
    PANGRAM_ADAPTER_ID,
    _device,
    _encode_docs,
    load_pangram_model,
)
from slop_sftdiv.cli.phase4_batchtopk_sae import _load_sae


CONTRACTION_RE = re.compile(r"(n't|n\u2019t|didn|wasn|weren|couldn|wouldn|shouldn|don|doesn|hadn)", re.I)
EXPANDED_NEGATION_RE = re.compile(r"^(not|no|never|cannot|did|do|does|was|were|had|could|would|should)$", re.I)
DISCOURSE_MARKER_RE = re.compile(
    r"^(however|therefore|thus|moreover|furthermore|additionally|consequently|meanwhile|"
    r"instead|otherwise|nevertheless|nonetheless|similarly|likewise|finally|first|second|"
    r"third|overall|ultimately|specifically|notably|indeed|for|example|instance)$",
    re.I,
)
STYLE_FUNCTION_RE = re.compile(
    r"^(also|then|when|while|because|although|though|since|if|as|which|that|this|these|those|"
    r"can|may|might|must|should|would|could|will|feel|feels|help|helps|make|makes)$",
    re.I,
)


def _token_class(token_text: str) -> str:
    if "\n" in token_text or "\r" in token_text:
        return "line_break"
    if token_text and token_text.strip() == "":
        return "whitespace"
    token = token_text.strip()
    if not token:
        return "empty"
    if CONTRACTION_RE.search(token):
        return "contraction_piece"
    if EXPANDED_NEGATION_RE.search(token):
        return "expanded_negation_word"
    if DISCOURSE_MARKER_RE.search(token):
        return "discourse_marker_word"
    if STYLE_FUNCTION_RE.search(token):
        return "style_function_word"
    if re.fullmatch(r"[.!?]+", token):
        return "sentence_boundary"
    if re.fullmatch(r"[^\w]+", token, flags=re.UNICODE):
        return "punctuation"
    if re.search(r"\w", token, flags=re.UNICODE):
        return "word"
    return "other"


def _decode_context(tokenizer: Any, input_ids: list[int], token_index: int, *, context_tokens: int) -> str:
    left = max(0, int(token_index) - int(context_tokens))
    right = min(len(input_ids), int(token_index) + int(context_tokens) + 1)
    return " ".join(str(tokenizer.decode(input_ids[left:right], skip_special_tokens=True)).split())


def _shorten(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, detail_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Pangram SAE Token Loci",
        "",
        "Top token activations for selected SAE nodes on explicit target rows.",
        "",
        "## Summary",
        "",
        "| Node | Doc | Source | Active tokens | Total mass | Max activation | Class counts |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| `{node}` | `{doc_id}` | `{source}` | {active_tokens} | `{mass:.2f}` | `{max_activation:.2f}` | {class_counts} |".format(
                **row
            )
        )
    lines.extend(["", "## Top Tokens", ""])
    for row in detail_rows:
        lines.extend(
            [
                "- `{node}` `{doc_id}` token `{token_index}` activation `{activation:.3f}` class `{token_class}` token `{token_text}`".format(
                    **row
                ),
                f"  - {row['context']}",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def _collect_loci(
    *,
    model: Any,
    tokenizer: Any,
    docs: list[Any],
    args: argparse.Namespace,
    device: torch.device,
    node: str,
    layer: int,
    latent_id: int,
    layer_module: nn.Module,
    sae: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for batch_docs in _batched(docs, int(args.batch_size)):
        encoded = _encode_docs(
            tokenizer,
            [_as_sae_doc(doc) for doc in batch_docs],
            max_length=int(args.max_length),
            device=device,
            pad_to_max_length=True,
        )
        input_ids_cpu = encoded["input_ids"].detach().cpu().tolist()
        token_mask = _activation_mask(encoded)
        positions = torch.nonzero(token_mask, as_tuple=False)
        captured_codes: list[torch.Tensor] = []

        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                return output
            token_vectors = hidden.float()[token_mask]
            if token_vectors.numel() == 0:
                captured_codes.append(torch.empty(0, dtype=torch.float32, device=hidden.device))
                return output
            captured_codes.append(
                _browser_latent_codes(
                    sae,
                    token_vectors,
                    latent_id=int(latent_id),
                    top_k=int(args.browser_top_k),
                )
            )
            return output

        handle = layer_module.register_forward_hook(hook)
        try:
            model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                return_dict=True,
            )
        finally:
            handle.remove()
        if not captured_codes:
            continue
        codes = captured_codes[0].detach().float().cpu()
        pos_cpu = positions.detach().cpu()
        per_doc: dict[int, list[tuple[float, int]]] = {index: [] for index in range(len(batch_docs))}
        for code, position in zip(codes.tolist(), pos_cpu.tolist(), strict=True):
            if code <= 0:
                continue
            doc_index, token_index = int(position[0]), int(position[1])
            per_doc[doc_index].append((float(code), token_index))
        for doc_index, doc in enumerate(batch_docs):
            activations = sorted(per_doc[doc_index], reverse=True)
            class_counts: Counter[str] = Counter()
            mass = sum(value for value, _token_index in activations)
            for value, token_index in activations:
                token_text = tokenizer.convert_tokens_to_string(
                    [tokenizer.convert_ids_to_tokens(int(input_ids_cpu[doc_index][token_index]))]
                )
                class_counts[_token_class(token_text)] += 1
            summary_rows.append(
                {
                    "node": node,
                    "layer": layer,
                    "latent_id": latent_id,
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "active_tokens": len(activations),
                    "mass": mass,
                    "max_activation": activations[0][0] if activations else 0.0,
                    "class_counts": "|".join(f"{key}:{value}" for key, value in class_counts.most_common()),
                }
            )
            for rank, (activation, token_index) in enumerate(activations[: int(args.top_tokens_per_doc)], start=1):
                token_id = int(input_ids_cpu[doc_index][token_index])
                token_text = tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(token_id)])
                detail_rows.append(
                    {
                        "node": node,
                        "layer": layer,
                        "latent_id": latent_id,
                        "doc_id": doc.doc_id,
                        "source": doc.source,
                        "rank": rank,
                        "token_index": token_index,
                        "token_id": token_id,
                        "token_text": token_text,
                        "token_class": _token_class(token_text),
                        "activation": activation,
                        "context": _shorten(
                            _decode_context(
                                tokenizer,
                                input_ids_cpu[doc_index],
                                token_index,
                                context_tokens=int(args.context_tokens),
                            ),
                            int(args.max_context_chars),
                        ),
                    }
                )
    return detail_rows, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect token-level SAE activation loci for selected Pangram nodes.")
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--doc-tokens-parquet", type=Path, default=Path("apps/latent-explorer/public/data/layer19/pangram_llama_sae_doc_tokens.parquet"))
    parser.add_argument("--node", action="append", required=True)
    parser.add_argument("--output-details-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--sae", action="append", default=[], metavar="LAYER=PATH")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--browser-top-k", type=int, default=64)
    parser.add_argument("--top-tokens-per-doc", type=int, default=12)
    parser.add_argument("--context-tokens", type=int, default=10)
    parser.add_argument("--max-context-chars", type=int, default=260)
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
    nodes = []
    for raw in args.node:
        layer, latent_id = _parse_node(raw)
        nodes.append((raw, layer, latent_id))
    sae_paths = _parse_sae_map(args.sae)
    for _node, layer, _latent_id in nodes:
        if not sae_paths.get(layer, DEFAULT_SAE_PATHS.get(layer, Path())).is_file():
            raise FileNotFoundError(f"Missing SAE checkpoint for layer {layer}: {sae_paths.get(layer)}")
    docs = _load_target_docs(args.target_csv, args.doc_tokens_parquet, limit_docs=args.limit_docs)
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model, tokenizer = load_pangram_model(args, device=device)
    model.eval()
    layers = {layer: _llama_layer(model, layer) for _node, layer, _latent_id in nodes}
    saes: dict[int, Any] = {}
    for _node, layer, _latent_id in nodes:
        if layer in saes:
            continue
        sae, _summary = _load_sae(sae_paths[layer], device=device)
        sae.eval()
        saes[layer] = sae
    all_details: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    for node, layer, latent_id in nodes:
        details, summaries = _collect_loci(
            model=model,
            tokenizer=tokenizer,
            docs=docs,
            args=args,
            device=device,
            node=node,
            layer=layer,
            latent_id=latent_id,
            layer_module=layers[layer],
            sae=saes[layer],
        )
        all_details.extend(details)
        all_summaries.extend(summaries)
    _write_csv(args.output_details_csv, all_details)
    _write_csv(args.output_summary_csv, all_summaries)
    _write_md(args.output_md, all_details, all_summaries)
    print(
        json.dumps(
            {
                "target_csv": str(args.target_csv),
                "nodes": [node for node, _layer, _latent_id in nodes],
                "details": len(all_details),
                "summaries": len(all_summaries),
                "output_details_csv": str(args.output_details_csv),
                "output_summary_csv": str(args.output_summary_csv),
                "output_md": str(args.output_md),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
