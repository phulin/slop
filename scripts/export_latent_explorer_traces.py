from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from slop_sftdiv.cli.phase4_batchtopk_sae import _load_docs, _load_sae
from slop_sftdiv.cli.pangram_llama_sae import load_pangram_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export compact token-level SAE traces for the static latent explorer."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset-parquet", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--base-model", default="unsloth/Llama-3.2-3B")
    parser.add_argument(
        "--local-model-dir",
        type=Path,
        default=Path("artifacts/models/pangram_editlens_Llama-3.2-3B_unsloth_base"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--no-hf-token", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-docs-per-source", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-latents", type=int, default=10)
    parser.add_argument("--examples-per-latent", type=int, default=10)
    parser.add_argument("--context-tokens", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def _device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _loader_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        reference_jsonl=[],
        generation_jsonl=[],
        dataset_parquet=args.dataset_parquet,
        max_docs_per_source=args.max_docs_per_source,
        seed=args.seed,
        feature_text_mode="final_answer",
        base_model=args.base_model,
        adapter_model="pangram/editlens_Llama-3.2-3B",
        local_model_dir=args.local_model_dir,
        cache_dir=args.cache_dir,
        revision=args.revision,
        adapter_revision=None,
        hf_token_env=args.hf_token_env,
        no_hf_token=args.no_hf_token,
        local_files_only=args.local_files_only,
    )


def _read_selected_examples(
    artifact_dir: Path,
    *,
    top_latents: int,
    examples_per_latent: int,
) -> tuple[list[int], pd.DataFrame]:
    latents = pd.read_csv(artifact_dir / "pangram_llama_sae_latents.csv")
    selected_latents = latents.sort_values("rank").head(top_latents)["latent_id"].astype(int).tolist()
    examples = pd.read_csv(artifact_dir / "pangram_llama_sae_examples.csv")
    examples = examples[examples["latent_id"].astype(int).isin(selected_latents)].copy()
    examples = (
        examples.sort_values(["rank", "example_rank"])
        .groupby("latent_id", as_index=False, sort=False)
        .head(examples_per_latent)
        .reset_index(drop=True)
    )
    return selected_latents, examples


def _token_text(tokenizer: Any, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    return str(tokenizer.convert_tokens_to_string([token]))


def _encode_doc(tokenizer: Any, text: str, *, max_length: int) -> dict[str, list[int]]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
        return_special_tokens_mask=True,
    )
    return {
        "input_ids": [int(value) for value in encoded.get("input_ids", [])],
        "attention_mask": [int(value) for value in encoded.get("attention_mask", [])],
        "special_tokens_mask": [int(value) for value in encoded.get("special_tokens_mask", [])],
    }


@torch.inference_mode()
def _collect_selected_activations(
    *,
    cache_path: Path,
    sae_path: Path,
    selected_latents: list[int],
    target_doc_latents: dict[int, set[int]],
    batch_size: int,
    device: torch.device,
) -> dict[tuple[int, int], dict[int, float]]:
    sae, _summary = _load_sae(sae_path, device=device)
    sae.eval()
    selected = torch.tensor(selected_latents, dtype=torch.long, device=device)
    selected_offsets = {latent_id: offset for offset, latent_id in enumerate(selected_latents)}
    target_array = np.array(sorted(target_doc_latents), dtype=np.int64)
    by_doc_latent: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    parquet = pq.ParquetFile(cache_path)
    scanned = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["activation", "doc_index", "token_index"],
    ):
        doc_indices = batch.column("doc_index").to_numpy(zero_copy_only=False)
        keep_mask = np.isin(doc_indices, target_array, assume_unique=False)
        scanned += len(doc_indices)
        if not keep_mask.any():
            continue
        activation_column = batch.column("activation")
        hidden_dim = int(activation_column.type.list_size)
        activations = (
            activation_column.values.to_numpy(zero_copy_only=False)
            .reshape(len(doc_indices), hidden_dim)[keep_mask]
            .copy()
        )
        token_indices = batch.column("token_index").to_numpy(zero_copy_only=False)[keep_mask]
        kept_doc_indices = doc_indices[keep_mask]
        codes = sae.encode_dense(
            torch.from_numpy(activations).to(device=device, dtype=torch.float32)
        )[:, selected].detach().cpu().numpy()
        for row_offset, doc_index in enumerate(kept_doc_indices.tolist()):
            token_index = int(token_indices[row_offset])
            for latent_id in target_doc_latents.get(int(doc_index), set()):
                latent_offset = selected_offsets[latent_id]
                value = float(codes[row_offset, latent_offset])
                if value > 0.0:
                    by_doc_latent[(int(doc_index), latent_id)][token_index] = value
        if scanned % (batch_size * 100) == 0:
            print(f"Scanned {scanned:,} activation rows", flush=True)
    return by_doc_latent


def export_traces(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = args.artifact_dir
    cache_path = artifact_dir / "pangram_llama_sae_activation_cache.parquet"
    sae_path = artifact_dir / "pangram_llama_batchtopk_sae.pt"
    selected_latents, examples = _read_selected_examples(
        artifact_dir,
        top_latents=args.top_latents,
        examples_per_latent=args.examples_per_latent,
    )
    loader_args = _loader_args(args)
    docs = _load_docs(loader_args)
    doc_index_by_id = {doc.doc_id: index for index, doc in enumerate(docs)}
    target_doc_latents: dict[int, set[int]] = defaultdict(set)
    for row in examples.itertuples(index=False):
        doc_id = str(row.doc_id)
        if doc_id in doc_index_by_id:
            target_doc_latents[doc_index_by_id[doc_id]].add(int(row.latent_id))
    missing = sorted(set(examples["doc_id"].astype(str)) - set(doc_index_by_id))
    if missing:
        print(f"Warning: {len(missing)} selected example docs were not found in the loaded dataset", flush=True)
    device = _device(args.device)
    activations = _collect_selected_activations(
        cache_path=cache_path,
        sae_path=sae_path,
        selected_latents=selected_latents,
        target_doc_latents=target_doc_latents,
        batch_size=args.batch_size,
        device=device,
    )
    tokenizer = load_pangram_tokenizer(loader_args)
    traces: dict[str, dict[str, Any]] = {}
    for row in examples.itertuples(index=False):
        doc_id = str(row.doc_id)
        if doc_id not in doc_index_by_id:
            continue
        doc_index = doc_index_by_id[doc_id]
        latent_id = int(row.latent_id)
        peak_token_index = int(row.token_index)
        encoded = _encode_doc(tokenizer, docs[doc_index].text, max_length=args.max_length)
        attention_mask = encoded["attention_mask"]
        special_mask = encoded["special_tokens_mask"]
        valid_indexes = [
            index
            for index, attention in enumerate(attention_mask)
            if attention and (index >= len(special_mask) or not special_mask[index])
        ]
        left = max(0, peak_token_index - int(args.context_tokens))
        right = min(len(encoded["input_ids"]), peak_token_index + int(args.context_tokens) + 1)
        context_indexes = [index for index in valid_indexes if left <= index < right]
        latent_activations = activations.get((doc_index, latent_id), {})
        token_values = [float(latent_activations.get(index, 0.0)) for index in context_indexes]
        if peak_token_index in context_indexes:
            peak_index = context_indexes.index(peak_token_index)
        elif token_values:
            peak_index = max(range(len(token_values)), key=lambda index: token_values[index])
        else:
            peak_index = 0
        traces[f"{latent_id}:{doc_id}"] = {
            "tokens": [_token_text(tokenizer, encoded["input_ids"][index]) for index in context_indexes],
            "activations": token_values,
            "peak_index": int(peak_index),
            "token_indices": context_indexes,
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(traces, separators=(",", ":")), encoding="utf-8")
    return {
        "traces": len(traces),
        "latents": selected_latents,
        "output": str(args.output_json),
    }


def main() -> None:
    args = build_parser().parse_args()
    summary = export_traces(args)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
