from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from slop_sftdiv.cli.phase4_batchtopk_sae import _load_sae


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export per-token sparse SAE top-k latent activations to Parquet."
    )
    parser.add_argument("--activation-cache", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--row-group-size", type=int, default=8192)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for smoke tests. Omit for a full export.",
    )
    return parser


def _device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _sidecar(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}.{suffix}")


def _metadata_dict(path: Path) -> dict[str, Any]:
    metadata_path = _sidecar(path, "metadata.json")
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _document_rows(path: Path) -> list[dict[str, Any]]:
    documents_path = _sidecar(path, "documents.parquet")
    if not documents_path.is_file():
        return []
    return pq.read_table(documents_path).to_pylist()


def _schema(k: int, metadata: dict[str, Any]) -> pa.Schema:
    return pa.schema(
        [
            ("latent_ids", pa.list_(pa.uint16(), list_size=k)),
            ("activations", pa.list_(pa.float16(), list_size=k)),
            ("doc_index", pa.uint32()),
            ("token_index", pa.uint16()),
        ],
        metadata={b"slop_sparse_sae_topk": json.dumps(metadata).encode("utf-8")},
    )


@torch.inference_mode()
def export_sparse_topk(args: argparse.Namespace) -> dict[str, Any]:
    if args.k <= 0:
        raise ValueError("--k must be positive")
    device = _device(args.device)
    sae, sae_summary = _load_sae(args.sae, device=device)
    sae.eval()
    if int(args.k) > int(sae.latent_dim):
        raise ValueError(f"--k={args.k} exceeds latent_dim={sae.latent_dim}")

    parquet_file = pq.ParquetFile(args.activation_cache)
    activation_type = parquet_file.schema_arrow.field("activation").type
    if not isinstance(activation_type, pa.FixedSizeListType):
        raise ValueError("activation cache must use fixed-size-list activation rows")
    hidden_dim = int(activation_type.list_size)
    source_row_count = int(parquet_file.metadata.num_rows)
    row_count = (
        min(source_row_count, int(args.max_rows))
        if args.max_rows is not None
        else source_row_count
    )
    if hidden_dim != int(sae.input_dim):
        raise ValueError(f"activation dim {hidden_dim} does not match SAE input dim {sae.input_dim}")

    cache_metadata = _metadata_dict(args.activation_cache)
    metadata = {
        "format": "sparse_sae_topk_parquet",
        "activation_cache": str(args.activation_cache),
        "sae": str(args.sae),
        "source_activation_rows": source_row_count,
        "export_activation_rows": row_count,
        "source_activation_dim": hidden_dim,
        "latent_dim": int(sae.latent_dim),
        "k": int(args.k),
        "latent_id_dtype": "uint16",
        "activation_dtype": "float16",
        "doc_index_dtype": "uint32",
        "token_index_dtype": "uint16",
        "encoder": "dense_relu_topk",
        "compression": "zstd",
        "compression_level": int(args.compression_level),
        "source_cache_metadata": cache_metadata,
        "sae_summary": sae_summary,
    }
    schema = _schema(int(args.k), metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(
        args.output,
        schema,
        compression="zstd",
        compression_level=max(1, min(22, int(args.compression_level))),
        use_dictionary=False,
    )
    started = time.time()
    written = 0
    try:
        for batch_index, batch in enumerate(
            parquet_file.iter_batches(
                batch_size=int(args.batch_size),
                columns=["activation", "doc_index", "token_index"],
            ),
            start=1,
        ):
            if written >= row_count:
                break
            activation_column = batch.column("activation")
            rows = min(len(activation_column), row_count - written)
            hidden = (
                activation_column.values.to_numpy(zero_copy_only=False)
                .reshape(len(activation_column), hidden_dim)[:rows]
                .copy()
            )
            activations = torch.from_numpy(hidden).to(device=device, dtype=torch.float32)
            values, indexes = torch.topk(sae.encode_dense(activations), k=int(args.k), dim=1)
            latent_ids = indexes.detach().cpu().numpy().astype(np.uint16, copy=False)
            latent_values = values.detach().cpu().to(dtype=torch.float16).numpy()
            doc_indices = (
                batch.column("doc_index").to_numpy(zero_copy_only=False)[:rows].astype(np.uint32, copy=False)
            )
            token_indices = (
                batch.column("token_index").to_numpy(zero_copy_only=False)[:rows].astype(np.uint16, copy=False)
            )
            table = pa.Table.from_arrays(
                [
                    pa.FixedSizeListArray.from_arrays(
                        pa.array(latent_ids.reshape(-1), type=pa.uint16()),
                        int(args.k),
                    ),
                    pa.FixedSizeListArray.from_arrays(
                        pa.array(latent_values.reshape(-1), type=pa.float16()),
                        int(args.k),
                    ),
                    pa.array(doc_indices, type=pa.uint32()),
                    pa.array(token_indices, type=pa.uint16()),
                ],
                schema=schema,
            )
            writer.write_table(table, row_group_size=int(args.row_group_size))
            written += rows
            if args.progress_every > 0 and (
                batch_index == 1 or batch_index % int(args.progress_every) == 0
            ):
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"Wrote {written:,}/{row_count:,} token rows "
                    f"({written / elapsed:,.0f} rows/s)",
                    flush=True,
                )
    finally:
        writer.close()

    documents = _document_rows(args.activation_cache)
    if documents:
        pq.write_table(
            pa.Table.from_pylist(documents),
            _sidecar(args.output, "documents.parquet"),
            compression="zstd",
        )
    final_metadata = {**metadata, "rows_written": written, "elapsed_seconds": time.time() - started}
    _sidecar(args.output, "metadata.json").write_text(
        json.dumps(final_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(args.output),
        "rows_written": written,
        "k": int(args.k),
        "elapsed_seconds": final_metadata["elapsed_seconds"],
    }


def main() -> None:
    summary = export_sparse_topk(build_parser().parse_args())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
