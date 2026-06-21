from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from slop_sftdiv.cli.phase4_batchtopk_sae import (
    ActivationCacheParquetStream,
    SAEDoc,
    _activation_document_rows,
    _activation_mask,
)
from slop_sftdiv.cli.pangram_llama_sae import (
    LLAMA_BASE_ID,
    PANGRAM_ADAPTER_ID,
    _device,
    _encode_docs,
    _load_docs,
    _pangram_activation_cache_metadata,
    load_pangram_model,
)


GLM_ALIASES = {
    "accounts/fireworks/models/glm-5p2": "glm-5.2",
    "zai-org/GLM-5.2-FP8": "glm-5.2",
    "neuralwatt/glm-5.2-short": "glm-5.2",
    "glm-5.2": "glm-5.2",
    "glm-5.2-short": "glm-5.2",
    "glm-5.2-fp8": "glm-5.2",
    "qwen3.6-35b-fast": "qwen3.6-35b",
    "qwen3.6-35b": "qwen3.6-35b",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Pangram/Llama activation cache for a rebuilt dataset by "
            "copying cached activations for unchanged docs and collecting only missing docs."
        )
    )
    parser.add_argument("--old-cache", type=Path, required=True)
    parser.add_argument("--old-dataset-parquet", type=Path, required=True)
    parser.add_argument("--new-dataset-parquet", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument(
        "--target-dataset-output",
        type=Path,
        default=None,
        help="Optional filtered target dataset parquet used to define target doc order.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["full-new-dataset", "old-cache-prompts-plus-source"],
        default="old-cache-prompts-plus-source",
    )
    parser.add_argument(
        "--new-source",
        action="append",
        default=["qwen3.6-35b"],
        help="Source/model to force-collect in old-cache-prompts-plus-source mode.",
    )
    parser.add_argument("--max-docs-per-source", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--collect-batch-size", type=int, default=32)
    parser.add_argument("--pad-to-max-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--activation-layer", type=int, default=20)
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
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--copy-batch-size", type=int, default=8192)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _loader_args(args: argparse.Namespace, dataset_parquet: Path) -> argparse.Namespace:
    return argparse.Namespace(
        reference_jsonl=[],
        generation_jsonl=[],
        dataset_parquet=dataset_parquet,
        feature_text_mode="final_answer",
        max_docs_per_source=int(args.max_docs_per_source),
        seed=int(args.seed),
    )


def _model_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        base_model=args.base_model,
        adapter_model=args.adapter_model,
        local_model_dir=args.local_model_dir,
        cache_dir=args.cache_dir,
        revision=args.revision,
        adapter_revision=args.adapter_revision,
        hf_token_env=args.hf_token_env,
        no_hf_token=args.no_hf_token,
        local_files_only=args.local_files_only,
        torch_dtype=args.torch_dtype,
    )


def _cache_metadata_args(args: argparse.Namespace, dataset_parquet: Path) -> argparse.Namespace:
    return argparse.Namespace(
        reference_jsonl=[],
        generation_jsonl=[],
        dataset_parquet=dataset_parquet,
        activation_layer=int(args.activation_layer),
        base_model=args.base_model,
        adapter_model=args.adapter_model,
        local_model_dir=args.local_model_dir,
        max_length=int(args.max_length),
        pad_to_max_length=bool(args.pad_to_max_length),
        max_activations=0,
    )


def _string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def _norm_model(value: Any) -> str:
    raw = _string(value)
    return GLM_ALIASES.get(raw, raw)


def _text_hash(value: Any) -> str:
    return hashlib.sha1(_string(value).encode("utf-8")).hexdigest()


def _sidecar(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}.{suffix}")


def _read_cache_documents(cache_path: Path) -> pd.DataFrame:
    path = _sidecar(cache_path, "documents.parquet")
    if not path.is_file():
        raise FileNotFoundError(f"missing activation-cache document sidecar: {path}")
    return pd.read_parquet(path)


def _row_by_record_id(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        rows.setdefault(_string(row.get("record_id")), row)
    return rows


def _filter_target_dataset(
    *,
    args: argparse.Namespace,
    old_cache_docs: pd.DataFrame,
    old_dataset: pd.DataFrame,
    new_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, Path]:
    if args.target_mode == "full-new-dataset":
        target = new_dataset.copy()
    else:
        old_cached_ids = set(old_cache_docs["doc_id"].fillna("").astype(str))
        old_turns = set(
            old_dataset.loc[
                old_dataset["record_id"].fillna("").astype(str).isin(old_cached_ids),
                "turn_id",
            ]
            .fillna("")
            .astype(str)
        )
        wanted_sources = {_norm_model(source) for source in args.new_source}
        target = new_dataset[
            new_dataset["turn_id"].fillna("").astype(str).isin(old_turns)
            & (
                new_dataset["model"].map(_norm_model).isin(wanted_sources)
                | (new_dataset["model"].map(_norm_model) != "")
            )
        ].copy()
    output = args.target_dataset_output or args.new_dataset_parquet
    if args.target_dataset_output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        target.to_parquet(output, index=False)
    return target, output


def _doc_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _string(row.get("turn_id")),
        _norm_model(row.get("model")),
        _string(row.get("role")),
        _text_hash(row.get("text")),
    )


def _build_reuse_plan(
    *,
    args: argparse.Namespace,
    old_cache_docs: pd.DataFrame,
    old_dataset: pd.DataFrame,
    target_dataset: pd.DataFrame,
    target_docs: list[SAEDoc],
) -> tuple[dict[int, list[int]], list[tuple[int, SAEDoc]], dict[str, Any]]:
    old_rows = _row_by_record_id(old_dataset)
    target_rows = _row_by_record_id(target_dataset)
    old_doc_indices_by_id: dict[str, list[int]] = defaultdict(list)
    for row in old_cache_docs.itertuples(index=False):
        old_doc_indices_by_id[_string(row.doc_id)].append(int(row.document_index))
    exact_old: dict[tuple[str, str], deque[int]] = defaultdict(deque)
    keyed_old: dict[tuple[str, str, str, str], deque[int]] = defaultdict(deque)
    for doc_id, old_doc_indexes in old_doc_indices_by_id.items():
        old_row = old_rows.get(doc_id)
        if old_row is None:
            continue
        for old_doc_index in old_doc_indexes:
            exact_old[(doc_id, _text_hash(old_row.get("text")))].append(old_doc_index)
            keyed_old[_doc_key(old_row)].append(old_doc_index)

    force_collect_sources = {_norm_model(source) for source in args.new_source}
    old_to_new: dict[int, list[int]] = defaultdict(list)
    missing: list[tuple[int, SAEDoc]] = []
    reuse_reasons: Counter[str] = Counter()
    missing_sources: Counter[str] = Counter()
    reused_sources: Counter[str] = Counter()
    reused_docs = 0

    for target_index, doc in enumerate(target_docs):
        target_row = target_rows.get(doc.doc_id)
        source = _norm_model(doc.source)
        if target_row is None:
            missing.append((target_index, doc))
            missing_sources[source] += 1
            continue
        if source in force_collect_sources:
            missing.append((target_index, doc))
            missing_sources[source] += 1
            continue
        text_hash = _text_hash(target_row.get("text"))
        exact_matches = exact_old.get((doc.doc_id, text_hash))
        if exact_matches:
            old_doc_index = exact_matches.popleft()
            old_to_new[old_doc_index].append(target_index)
            reuse_reasons["exact_doc_id"] += 1
            reused_sources[source] += 1
            reused_docs += 1
            continue
        keyed_matches = keyed_old.get(_doc_key(target_row))
        if keyed_matches:
            old_doc_index = keyed_matches.popleft()
            old_to_new[old_doc_index].append(target_index)
            reuse_reasons["turn_model_role_text"] += 1
            reused_sources[source] += 1
            reused_docs += 1
            continue
        missing.append((target_index, doc))
        missing_sources[source] += 1

    summary = {
        "target_docs": len(target_docs),
        "reused_docs": reused_docs,
        "reused_unique_cache_docs": len(old_to_new),
        "missing_docs": len(missing),
        "reuse_reasons": dict(sorted(reuse_reasons.items())),
        "reused_sources": dict(sorted(reused_sources.items())),
        "missing_sources": dict(sorted(missing_sources.items())),
    }
    return old_to_new, missing, summary


def _copy_cached_rows(
    *,
    writer: ActivationCacheParquetStream,
    old_cache: Path,
    old_to_new: dict[int, list[int]],
    batch_size: int,
    progress_every: int,
) -> int:
    if not old_to_new:
        return 0
    parquet = pq.ParquetFile(old_cache)
    activation_type = parquet.schema_arrow.field("activation").type
    if not isinstance(activation_type, pa.FixedSizeListType):
        raise ValueError("old cache activation column must be fixed-size-list")
    hidden_dim = int(activation_type.list_size)
    max_old_index = max(old_to_new)
    lookup = np.full(max_old_index + 1, -1, dtype=np.int64)
    for old_index, new_indexes in old_to_new.items():
        if new_indexes:
            lookup[int(old_index)] = int(new_indexes[0])

    copied = 0
    duplicate_old_to_new = {
        int(old_index): [int(new_index) for new_index in new_indexes[1:]]
        for old_index, new_indexes in old_to_new.items()
        if len(new_indexes) > 1
    }
    for batch_index, batch in enumerate(
        parquet.iter_batches(
            batch_size=int(batch_size),
            columns=["activation", "doc_index", "token_index"],
        ),
        start=1,
    ):
        old_doc_indices = batch.column("doc_index").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        in_bounds = old_doc_indices <= max_old_index
        mapped = np.full(len(old_doc_indices), -1, dtype=np.int64)
        mapped[in_bounds] = lookup[old_doc_indices[in_bounds]]
        keep = mapped >= 0
        if keep.any():
            activation_column = batch.column("activation")
            kept_activations = (
                activation_column.values.to_numpy(zero_copy_only=False)
                .reshape(len(old_doc_indices), hidden_dim)[keep]
                .copy()
            )
            kept_doc_indices = old_doc_indices[keep]
            kept_token_indices = batch.column("token_index").to_numpy(zero_copy_only=False)[keep].copy()
            extra_activations: list[np.ndarray] = []
            extra_doc_indices: list[np.ndarray] = []
            extra_token_indices: list[np.ndarray] = []
            if duplicate_old_to_new:
                for old_index, new_indexes in duplicate_old_to_new.items():
                    positions = np.nonzero(kept_doc_indices == old_index)[0]
                    if len(positions) == 0:
                        continue
                    for new_index in new_indexes:
                        extra_activations.append(kept_activations[positions])
                        extra_doc_indices.append(np.full(len(positions), new_index, dtype=np.int64))
                        extra_token_indices.append(kept_token_indices[positions])
                if extra_activations:
                    activations = np.concatenate([kept_activations, *extra_activations], axis=0)
                    doc_indices = np.concatenate([mapped[keep], *extra_doc_indices], axis=0)
                    token_indices = np.concatenate([kept_token_indices, *extra_token_indices], axis=0)
                else:
                    activations = kept_activations
                    doc_indices = mapped[keep]
                    token_indices = kept_token_indices
            else:
                activations = kept_activations
                doc_indices = mapped[keep]
                token_indices = kept_token_indices
            single_keep = doc_indices >= 0
            writer.write(
                activations=torch.from_numpy(activations[single_keep].copy()),
                doc_indices=torch.from_numpy(doc_indices[single_keep].copy()),
                token_indices=torch.from_numpy(token_indices[single_keep].copy()),
            )
            copied += int(single_keep.sum())
        if progress_every > 0 and (batch_index == 1 or batch_index % progress_every == 0):
            print(f"Copied {copied:,} cached activation rows", flush=True)
    return copied


@torch.inference_mode()
def _collect_missing_rows(
    *,
    writer: ActivationCacheParquetStream,
    model: Any,
    tokenizer: Any,
    missing: list[tuple[int, SAEDoc]],
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    if not missing:
        return 0
    collected = 0
    for batch_index, start in enumerate(range(0, len(missing), int(args.collect_batch_size)), start=1):
        batch = missing[start : start + int(args.collect_batch_size)]
        target_indices = [target_index for target_index, _doc in batch]
        batch_docs = [doc for _target_index, doc in batch]
        encoded = _encode_docs(
            tokenizer,
            batch_docs,
            max_length=int(args.max_length),
            device=device,
            pad_to_max_length=bool(args.pad_to_max_length),
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
        hidden_index = -1 if int(args.activation_layer) < 0 else int(args.activation_layer) + 1
        hidden = cast(torch.Tensor, hidden_states[hidden_index])
        token_mask = _activation_mask(encoded)
        positions = torch.nonzero(token_mask, as_tuple=False).detach().cpu()
        if positions.numel() == 0:
            continue
        target_doc_indices = torch.tensor(
            [target_indices[int(local_doc_index)] for local_doc_index in positions[:, 0].tolist()],
            dtype=torch.long,
        )
        token_indices = positions[:, 1].to(dtype=torch.long)
        vectors = hidden[token_mask].detach().to(dtype=torch.float16).cpu()
        writer.write(
            activations=vectors,
            doc_indices=target_doc_indices,
            token_indices=token_indices,
        )
        collected += int(vectors.shape[0])
        if args.progress_every > 0 and (
            batch_index == 1 or batch_index % int(args.progress_every) == 0
        ):
            print(
                f"Collected {collected:,} activation rows from "
                f"{min(start + int(args.collect_batch_size), len(missing)):,}/{len(missing):,} missing docs",
                flush=True,
            )
    return collected


def run(args: argparse.Namespace) -> dict[str, Any]:
    old_dataset = pd.read_parquet(args.old_dataset_parquet)
    new_dataset = pd.read_parquet(args.new_dataset_parquet)
    old_cache_docs = _read_cache_documents(args.old_cache)
    target_dataset, target_dataset_path = _filter_target_dataset(
        args=args,
        old_cache_docs=old_cache_docs,
        old_dataset=old_dataset,
        new_dataset=new_dataset,
    )
    target_docs = _load_docs(_loader_args(args, target_dataset_path))
    old_to_new, missing, plan_summary = _build_reuse_plan(
        args=args,
        old_cache_docs=old_cache_docs,
        old_dataset=old_dataset,
        target_dataset=target_dataset,
        target_docs=target_docs,
    )
    summary: dict[str, Any] = {
        **plan_summary,
        "old_cache": str(args.old_cache),
        "old_dataset_parquet": str(args.old_dataset_parquet),
        "new_dataset_parquet": str(args.new_dataset_parquet),
        "target_dataset_parquet": str(target_dataset_path),
        "output_cache": str(args.output_cache),
        "target_mode": str(args.target_mode),
        "forced_collect_sources": sorted({_norm_model(source) for source in args.new_source}),
        "activation_layer": int(args.activation_layer),
        "max_length": int(args.max_length),
        "pad_to_max_length": bool(args.pad_to_max_length),
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return summary

    parquet = pq.ParquetFile(args.old_cache)
    activation_type = parquet.schema_arrow.field("activation").type
    if not isinstance(activation_type, pa.FixedSizeListType):
        raise ValueError("old cache activation column must be fixed-size-list")
    hidden_dim = int(activation_type.list_size)
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    writer = ActivationCacheParquetStream(
        args.output_cache,
        hidden_dim=hidden_dim,
        metadata={
            **_pangram_activation_cache_metadata(
                _cache_metadata_args(args, target_dataset_path),
                docs=target_docs,
            ),
            "incremental_from_cache": str(args.old_cache),
            "incremental_reused_docs": int(summary["reused_docs"]),
            "incremental_missing_docs": int(summary["missing_docs"]),
        },
        compression_level=int(args.compression_level),
    )
    copied_rows = 0
    collected_rows = 0
    try:
        copied_rows = _copy_cached_rows(
            writer=writer,
            old_cache=args.old_cache,
            old_to_new=old_to_new,
            batch_size=int(args.copy_batch_size),
            progress_every=int(args.progress_every),
        )
        if missing:
            device = _device(args.device)
            if device.type == "cuda":
                torch.set_float32_matmul_precision("high")
            model, tokenizer = load_pangram_model(_model_args(args), device=device)
            collected_rows = _collect_missing_rows(
                writer=writer,
                model=model,
                tokenizer=tokenizer,
                missing=missing,
                args=args,
                device=device,
            )
    finally:
        writer.close(document_rows=_activation_document_rows(target_docs))

    final_summary = {
        **summary,
        "copied_activation_rows": copied_rows,
        "collected_activation_rows": collected_rows,
        "total_activation_rows": copied_rows + collected_rows,
    }
    args.output_cache.with_name(f"{args.output_cache.stem}.incremental_summary.json").write_text(
        json.dumps(final_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_summary, indent=2, sort_keys=True), flush=True)
    return final_summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
