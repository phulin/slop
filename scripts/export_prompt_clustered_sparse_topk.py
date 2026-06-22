from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


MODEL_ORDER = {
    "human": 0,
    "gpt-5.5": 1,
    "gemini-3.5-flash": 2,
    "accounts/fireworks/models/glm-5p2": 3,
    "qwen3.6-35b": 4,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite sparse SAE top-k activations clustered by prompt/model/token order."
    )
    parser.add_argument("--doc-tokens-parquet", type=Path, required=True)
    parser.add_argument("--sparse-topk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser-index", type=Path, default=None)
    parser.add_argument("--browser-sparse-url", default=None)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--row-group-size", type=int, default=8192)
    parser.add_argument("--compression-level", type=int, default=3)
    return parser


def _schema(k: int, metadata: dict[str, Any]) -> pa.Schema:
    return pa.schema(
        [
            ("latent_ids", pa.list_(pa.uint16(), list_size=k)),
            ("activations", pa.list_(pa.float16(), list_size=k)),
            ("doc_index", pa.uint32()),
            ("token_index", pa.uint16()),
        ],
        metadata={b"slop_prompt_clustered_sparse_sae_topk": json.dumps(metadata).encode("utf-8")},
    )


def _ordered_docs(doc_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, dict[str, Any]]]:
    by_turn: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in doc_rows:
        by_turn[str(row["turn_id"])].append(row)
    ordered: list[dict[str, Any]] = []
    turn_ranges: dict[str, dict[str, int]] = {}
    doc_ranges: dict[str, dict[str, Any]] = {}
    cursor = 0
    for turn_id in sorted(by_turn):
        turn_start = cursor
        docs = sorted(
            by_turn[turn_id],
            key=lambda row: (
                MODEL_ORDER.get(str(row["model"]), 999),
                str(row["model"]),
                str(row["doc_id"]),
            ),
        )
        for row in docs:
            sparse_start = row.get("sparse_row_start")
            sparse_end = row.get("sparse_row_end")
            if sparse_start is not None and sparse_end is not None:
                token_count = int(sparse_end) - int(sparse_start)
            else:
                token_count = len(row.get("token_indices") or [])
            start = cursor
            end = start + token_count
            doc_id = str(row["doc_id"])
            doc_ranges[doc_id] = {
                "start": start,
                "end": end,
                "doc_index": int(row["doc_index"]),
                "turn_id": turn_id,
            }
            ordered.append(row)
            cursor = end
        turn_ranges[turn_id] = {"start": turn_start, "end": cursor}
    return ordered, doc_ranges, turn_ranges


def _write_sparse_parquet(
    *,
    output: Path,
    latent_ids: np.ndarray,
    activations: np.ndarray,
    doc_indices: np.ndarray,
    token_indices: np.ndarray,
    metadata: dict[str, Any],
    row_group_size: int,
    compression_level: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    k = int(latent_ids.shape[1])
    schema = _schema(k, metadata)
    writer = pq.ParquetWriter(
        output,
        schema,
        compression="zstd",
        compression_level=max(1, min(22, int(compression_level))),
        use_dictionary=False,
    )
    try:
        for start in range(0, int(latent_ids.shape[0]), int(row_group_size)):
            end = min(int(latent_ids.shape[0]), start + int(row_group_size))
            ids = latent_ids[start:end]
            values = activations[start:end]
            table = pa.Table.from_arrays(
                [
                    pa.FixedSizeListArray.from_arrays(
                        pa.array(ids.reshape(-1), type=pa.uint16()),
                        k,
                    ),
                    pa.FixedSizeListArray.from_arrays(
                        pa.array(values.reshape(-1), type=pa.float16()),
                        k,
                    ),
                    pa.array(doc_indices[start:end], type=pa.uint32()),
                    pa.array(token_indices[start:end], type=pa.uint16()),
                ],
                schema=schema,
            )
            writer.write_table(table, row_group_size=int(row_group_size))
    finally:
        writer.close()


def _patch_browser_index(
    *,
    browser_index: Path,
    browser_sparse_url: str,
    doc_ranges: dict[str, dict[str, Any]],
    turn_ranges: dict[str, dict[str, int]],
) -> None:
    payload = json.loads(browser_index.read_text(encoding="utf-8"))
    documents = payload.get("documents", {})
    for doc_id, row_range in doc_ranges.items():
        if doc_id not in documents:
            continue
        documents[doc_id]["sparse_row_start"] = int(row_range["start"])
        documents[doc_id]["sparse_row_end"] = int(row_range["end"])
    payload["turnRanges"] = {
        turn_id: {"row_start": int(row_range["start"]), "row_end": int(row_range["end"])}
        for turn_id, row_range in turn_ranges.items()
    }
    payload["sparseTopkParquet"] = browser_sparse_url
    browser_index.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def export_prompt_clustered(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    doc_rows = pq.read_table(args.doc_tokens_parquet).to_pylist()
    ordered_docs, doc_ranges, turn_ranges = _ordered_docs(doc_rows)
    total_rows = sum(int(row_range["end"] - row_range["start"]) for row_range in doc_ranges.values())
    sparse = pq.ParquetFile(args.sparse_topk)
    latent_type = sparse.schema_arrow.field("latent_ids").type
    if not isinstance(latent_type, pa.FixedSizeListType):
        raise ValueError("--sparse-topk must have fixed-size-list latent_ids")
    k = int(latent_type.list_size)
    latent_ids_out = np.zeros((total_rows, k), dtype=np.uint16)
    activations_out = np.zeros((total_rows, k), dtype=np.float16)
    doc_indices_out = np.zeros(total_rows, dtype=np.uint32)
    token_indices_out = np.zeros(total_rows, dtype=np.uint16)
    start_by_doc_index = {
        int(row_range["doc_index"]): int(row_range["start"])
        for row_range in doc_ranges.values()
    }
    wanted_doc_indices = np.array(sorted(start_by_doc_index), dtype=np.uint32)
    seen_counts: dict[int, int] = defaultdict(int)
    copied = 0
    for batch_index, batch in enumerate(
        sparse.iter_batches(
            batch_size=int(args.batch_size),
            columns=["latent_ids", "activations", "doc_index", "token_index"],
        ),
        start=1,
    ):
        doc_indices = batch.column("doc_index").to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
        keep = np.isin(doc_indices, wanted_doc_indices)
        if not keep.any():
            continue
        latent_column = batch.column("latent_ids")
        value_column = batch.column("activations")
        rows = len(doc_indices)
        latent_matrix = latent_column.values.to_numpy(zero_copy_only=False).reshape(rows, k)[keep]
        value_matrix = value_column.values.to_numpy(zero_copy_only=False).reshape(rows, k)[keep]
        kept_doc_indices = doc_indices[keep]
        kept_token_indices = batch.column("token_index").to_numpy(zero_copy_only=False)[keep].astype(np.uint16, copy=False)
        output_rows = np.empty(len(kept_doc_indices), dtype=np.int64)
        for index, doc_index in enumerate(kept_doc_indices.tolist()):
            output_rows[index] = start_by_doc_index[int(doc_index)] + seen_counts[int(doc_index)]
            seen_counts[int(doc_index)] += 1
        latent_ids_out[output_rows] = latent_matrix
        activations_out[output_rows] = value_matrix
        doc_indices_out[output_rows] = kept_doc_indices
        token_indices_out[output_rows] = kept_token_indices
        copied += len(kept_doc_indices)
        if batch_index == 1 or batch_index % 25 == 0:
            print(f"Clustered {copied:,}/{total_rows:,} sparse rows", flush=True)
    missing = [
        row
        for row in ordered_docs
        if seen_counts[int(row["doc_index"])] != int(doc_ranges[str(row["doc_id"])]["end"] - doc_ranges[str(row["doc_id"])]["start"])
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} docs had incomplete sparse rows")
    metadata = {
        "format": "prompt_clustered_sparse_sae_topk",
        "source_sparse_topk": str(args.sparse_topk),
        "doc_tokens_parquet": str(args.doc_tokens_parquet),
        "rows": int(total_rows),
        "documents": len(doc_ranges),
        "turns": len(turn_ranges),
        "k": k,
        "cluster_order": "turn_id,model_order,token_index",
    }
    _write_sparse_parquet(
        output=args.output,
        latent_ids=latent_ids_out,
        activations=activations_out,
        doc_indices=doc_indices_out,
        token_indices=token_indices_out,
        metadata=metadata,
        row_group_size=int(args.row_group_size),
        compression_level=int(args.compression_level),
    )
    metadata = {**metadata, "elapsed_seconds": time.time() - started}
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.browser_index is not None:
        _patch_browser_index(
            browser_index=args.browser_index,
            browser_sparse_url=args.browser_sparse_url or f"/data/full/{args.output.name}",
            doc_ranges=doc_ranges,
            turn_ranges=turn_ranges,
        )
    return metadata


def main() -> None:
    summary = export_prompt_clustered(build_parser().parse_args())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
