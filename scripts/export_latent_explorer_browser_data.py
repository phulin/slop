from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from slop_sftdiv.cli.phase4_batchtopk_sae import _load_docs
from slop_sftdiv.cli.pangram_llama_sae import load_pangram_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export browser index and tokenized document parquet for the latent explorer."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset-parquet", type=Path, required=True)
    parser.add_argument("--assistant-turns-parquet", type=Path, required=True)
    parser.add_argument("--activation-cache", type=Path, required=True)
    parser.add_argument("--sparse-topk", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--top-latents", type=int, default=100)
    parser.add_argument("--examples-per-latent", type=int, default=50)
    return parser


def _string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def _prompt_text(messages_json: str) -> tuple[str, list[dict[str, str]]]:
    try:
        raw_messages = json.loads(messages_json)
    except json.JSONDecodeError:
        return messages_json, []
    messages: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        role = _string(message.get("role")).strip() or "message"
        content = _string(message.get("content")).strip()
        messages.append({"role": role, "content": content})
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"], messages
    rendered = "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    )
    return rendered, messages


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


def _read_latent_examples(
    artifact_dir: Path,
    *,
    top_latents: int,
    examples_per_latent: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latents = pd.read_csv(artifact_dir / "pangram_llama_sae_latents.csv").sort_values("rank")
    latents = latents.head(top_latents).copy()
    examples = pd.read_csv(artifact_dir / "pangram_llama_sae_examples.csv")
    examples = examples[examples["latent_id"].astype(int).isin(latents["latent_id"].astype(int))].copy()
    examples = (
        examples.sort_values(["rank", "example_rank"])
        .groupby("latent_id", as_index=False, sort=False)
        .head(examples_per_latent)
        .reset_index(drop=True)
    )
    return latents, examples


def _row_ranges(cache_path: Path, doc_indices: set[int]) -> dict[int, dict[str, int]]:
    if not doc_indices:
        return {}
    ranges: dict[int, dict[str, int]] = {}
    parquet = pq.ParquetFile(cache_path)
    offset = 0
    for batch in parquet.iter_batches(batch_size=65536, columns=["doc_index"]):
        values = batch.column("doc_index").to_numpy(zero_copy_only=False)
        for local_index, raw_doc_index in enumerate(values.tolist()):
            doc_index = int(raw_doc_index)
            if doc_index in doc_indices:
                row = offset + local_index
                current = ranges.get(doc_index)
                if current is None:
                    ranges[doc_index] = {"start": row, "end": row + 1}
                else:
                    current["end"] = row + 1
        offset += len(values)
    return ranges


def _generation_maxes(
    *,
    sparse_topk_path: Path,
    latent_ids: set[int],
    doc_id_by_doc_index: dict[int, str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    if not latent_ids or not doc_id_by_doc_index:
        return {}
    selected = np.array(sorted(latent_ids), dtype=np.uint16)
    wanted_docs = set(doc_id_by_doc_index)
    wanted_doc_array = np.array(sorted(wanted_docs), dtype=np.uint32)
    maxes: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
    parquet = pq.ParquetFile(sparse_topk_path)
    for batch in parquet.iter_batches(
        batch_size=65536,
        columns=["latent_ids", "activations", "doc_index", "token_index"],
    ):
        doc_indices = batch.column("doc_index").to_numpy(zero_copy_only=False)
        keep_mask = np.isin(doc_indices, wanted_doc_array)
        if not keep_mask.any():
            continue
        latent_column = batch.column("latent_ids")
        value_column = batch.column("activations")
        width = int(latent_column.type.list_size)
        latent_matrix = (
            latent_column.values.to_numpy(zero_copy_only=False)
            .reshape(len(doc_indices), width)[keep_mask]
        )
        value_matrix = (
            value_column.values.to_numpy(zero_copy_only=False)
            .reshape(len(doc_indices), width)[keep_mask]
        )
        kept_doc_indices = doc_indices[keep_mask]
        token_indices = batch.column("token_index").to_numpy(zero_copy_only=False)[keep_mask]
        hit_rows, hit_offsets = np.nonzero(np.isin(latent_matrix, selected))
        for row_offset, latent_offset in zip(hit_rows.tolist(), hit_offsets.tolist()):
            latent_id = int(latent_matrix[row_offset, latent_offset])
            doc_id = doc_id_by_doc_index[int(kept_doc_indices[row_offset])]
            value = float(value_matrix[row_offset, latent_offset])
            token_index = int(token_indices[row_offset])
            current = maxes[str(latent_id)].get(doc_id)
            if current is None or value > float(current["max_activation"]):
                maxes[str(latent_id)][doc_id] = {
                    "max_activation": value,
                    "max_token_index": token_index,
                }
    return maxes


def _tokenize_doc(tokenizer: Any, text: str, *, max_length: int) -> tuple[list[int], list[str]]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
        return_special_tokens_mask=True,
    )
    input_ids = [int(value) for value in encoded.get("input_ids", [])]
    attention_mask = [int(value) for value in encoded.get("attention_mask", [])]
    special_mask = [int(value) for value in encoded.get("special_tokens_mask", [])]
    token_indices = [
        index
        for index, attention in enumerate(attention_mask)
        if attention and (index >= len(special_mask) or not special_mask[index])
    ]
    tokens = [
        str(tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(input_ids[index])]))
        for index in token_indices
    ]
    return token_indices, tokens


def export_browser_data(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = args.artifact_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    latents, examples = _read_latent_examples(
        artifact_dir,
        top_latents=int(args.top_latents),
        examples_per_latent=int(args.examples_per_latent),
    )
    dataset = pd.read_parquet(args.dataset_parquet)
    turns = pd.read_parquet(args.assistant_turns_parquet)
    turns_by_id = {str(row.turn_id): row._asdict() for row in turns.itertuples(index=False)}
    wanted_example_ids = set(examples["doc_id"].fillna("").astype(str))
    example_rows = dataset[dataset["record_id"].fillna("").astype(str).isin(wanted_example_ids)].copy()
    wanted_turns = set(example_rows["turn_id"].fillna("").astype(str))
    sibling_rows = dataset[dataset["turn_id"].fillna("").astype(str).isin(wanted_turns)].copy()

    loader_args = _loader_args(args)
    docs = _load_docs(loader_args)
    doc_index_by_id = {doc.doc_id: index for index, doc in enumerate(docs)}
    tokenizer = load_pangram_tokenizer(loader_args)
    wanted_doc_ids = set(sibling_rows["record_id"].fillna("").astype(str))
    wanted_doc_indices = {
        doc_index_by_id[doc_id] for doc_id in wanted_doc_ids if doc_id in doc_index_by_id
    }
    row_ranges = _row_ranges(args.activation_cache, wanted_doc_indices)
    doc_id_by_doc_index = {
        doc_index_by_id[doc_id]: doc_id
        for doc_id in wanted_doc_ids
        if doc_id in doc_index_by_id and doc_index_by_id[doc_id] in row_ranges
    }
    generation_maxes = _generation_maxes(
        sparse_topk_path=args.sparse_topk,
        latent_ids={int(value) for value in latents["latent_id"].astype(int).tolist()},
        doc_id_by_doc_index=doc_id_by_doc_index,
    )
    row_by_doc_id = {str(row.record_id): row._asdict() for row in sibling_rows.itertuples(index=False)}

    model_order = {
        "human": 0,
        "gpt-5.5": 1,
        "gemini-3.5-flash": 2,
        "accounts/fireworks/models/glm-5p2": 3,
    }
    doc_records: list[dict[str, Any]] = []
    documents: dict[str, dict[str, Any]] = {}
    prompt_groups: dict[str, list[str]] = defaultdict(list)
    for doc_id in sorted(wanted_doc_ids):
        if doc_id not in doc_index_by_id:
            continue
        row = row_by_doc_id[doc_id]
        doc_index = doc_index_by_id[doc_id]
        if doc_index not in row_ranges:
            continue
        turn_id = _string(row.get("turn_id")).strip()
        turn = turns_by_id.get(turn_id, {})
        prompt, _messages = _prompt_text(_string(turn.get("input_messages_json", "")))
        token_indices, tokens = _tokenize_doc(
            tokenizer,
            _string(row.get("text")),
            max_length=int(args.max_length),
        )
        doc_row_index = len(doc_records)
        record = {
            "doc_id": doc_id,
            "turn_id": turn_id,
            "model": _string(row.get("model")),
            "provider": _string(row.get("provider")),
            "role": _string(row.get("role")),
            "category": _string(row.get("category", turn.get("category", ""))),
            "prompt": prompt,
            "text": _string(row.get("text")),
            "word_count": int(row.get("word_count", 0)),
            "token_len": int(row.get("token_len", 0)),
            "doc_index": int(doc_index),
            "sparse_row_start": int(row_ranges[doc_index]["start"]),
            "sparse_row_end": int(row_ranges[doc_index]["end"]),
            "token_indices": token_indices,
            "tokens": tokens,
        }
        doc_records.append(record)
        documents[doc_id] = {
            "row": doc_row_index,
            "turn_id": turn_id,
            "model": record["model"],
            "doc_index": int(doc_index),
            "sparse_row_start": record["sparse_row_start"],
            "sparse_row_end": record["sparse_row_end"],
        }
        prompt_groups[turn_id].append(doc_id)

    for turn_id, doc_ids in prompt_groups.items():
        doc_ids.sort(
            key=lambda doc_id: (
                model_order.get(str(documents[doc_id]["model"]), 999),
                str(documents[doc_id]["model"]),
            )
        )

    latent_docs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in examples.itertuples(index=False):
        doc_id = str(row.doc_id)
        if doc_id not in documents:
            continue
        latent_docs[str(int(row.latent_id))].append(
            {
                "doc_id": doc_id,
                "turn_id": documents[doc_id]["turn_id"],
                "source": str(row.source),
                "example_rank": int(row.example_rank),
                "rank": int(row.rank),
                "peak_activation": float(row.activation),
                "peak_token": str(row.token),
                "peak_token_index": int(row.token_index),
                "context": str(row.context),
            }
        )

    filtered_generation_maxes: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
    for latent_id, rows in latent_docs.items():
        latent_maxes = generation_maxes.get(str(latent_id), {})
        for row in rows:
            turn_id = str(row["turn_id"])
            for doc_id in prompt_groups.get(turn_id, []):
                if doc_id in latent_maxes:
                    filtered_generation_maxes[str(latent_id)][doc_id] = latent_maxes[doc_id]

    doc_table = pa.Table.from_pylist(doc_records)
    pq.write_table(
        doc_table,
        output_dir / "pangram_llama_sae_doc_tokens.parquet",
        compression="zstd",
        compression_level=3,
        use_dictionary=False,
    )
    index_payload = {
        "latents": latents.to_dict(orient="records"),
        "latentDocs": latent_docs,
        "documents": documents,
        "promptGroups": prompt_groups,
        "generationMaxes": filtered_generation_maxes,
        "docTokensParquet": "/data/full/pangram_llama_sae_doc_tokens.parquet",
        "sparseTopkParquet": "/data/full/pangram_llama_sae_sparse_topk_k64.parquet",
    }
    (output_dir / "pangram_llama_sae_browser_index.json").write_text(
        json.dumps(index_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return {
        "latents": len(latents),
        "latent_docs": int(sum(len(rows) for rows in latent_docs.values())),
        "documents": len(doc_records),
        "prompt_groups": len(prompt_groups),
        "output_dir": str(output_dir),
    }


def main() -> None:
    summary = export_browser_data(build_parser().parse_args())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
