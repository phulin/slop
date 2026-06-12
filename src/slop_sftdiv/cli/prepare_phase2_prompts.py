from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from slop_sftdiv.cli.census import _id_fields_with_pair_id, _source_from_input
from slop_sftdiv.features.tier1_matchers import count_tokens
from slop_sftdiv.minhash import (
    estimate_jaccard,
    minhash_signature,
    normalize_for_minhash,
    signature_bands,
)
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


MAX_WANDB_MANIFEST_ROWS = 1000


@dataclass(frozen=True)
class PromptCandidate:
    order: int
    row: dict[str, Any]
    manifest_row: dict[str, Any]
    signature: tuple[int, ...]
    normalized_prompt: str


@dataclass
class PrepStats:
    scanned: int = 0
    missing_prompt: int = 0
    missing_text: int = 0
    duplicate_prompt_id: int = 0
    near_duplicate_prompt: int = 0
    eligible: int = 0
    selected: int = 0
    prompt_tokens: int = 0
    target_tokens: int = 0
    wall_seconds: float = 0.0
    bytes_written: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare held-out Phase 2 prompt/target packages.")
    parser.add_argument("--input", action="append", required=True, help="Local JSONL file or HF dataset id.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--max-scan", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--sampling-strategy",
        default="hash_reservoir",
        choices=["first", "hash_reservoir"],
        help="Select from de-duplicated prompts in stream order or by stable hash.",
    )
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--minhash-size", type=int, default=64)
    parser.add_argument("--minhash-shingle-tokens", type=int, default=5)
    parser.add_argument("--minhash-band-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-data")
    parser.add_argument("--wandb-job-type", default="prepare-prompts")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _load_components():
    from slop_sftdiv.data import SamplingConfig, iter_corpus_records
    from slop_sftdiv.data.corpora import DEFAULT_ID_FIELDS

    return SamplingConfig, iter_corpus_records, DEFAULT_ID_FIELDS


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _selection_key(candidate: PromptCandidate, *, seed: int) -> int:
    key = f"{seed}\0{candidate.row['phase2_prompt_id']}\0{candidate.row['prompt_hash']}"
    return int(hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest(), 16)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _metadata_value(record: Any, *keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = record.metadata.get(key)
        if value:
            return str(value)
    return default


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
            handle.write(payload + "\n")
            bytes_written += len(payload.encode("utf-8")) + 1
    return bytes_written


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _nearest_duplicate(
    *,
    signature: tuple[int, ...],
    normalized_prompt: str,
    exact_index: dict[str, str],
    band_index: dict[tuple[int, tuple[int, ...]], set[int]],
    candidates: list[PromptCandidate],
    threshold: float,
    band_size: int,
) -> tuple[str | None, float]:
    exact_id = exact_index.get(normalized_prompt)
    if exact_id is not None:
        return exact_id, 1.0
    candidate_indexes: set[int] = set()
    for band in signature_bands(signature, band_size=band_size):
        candidate_indexes.update(band_index.get(band, set()))
    best_id = None
    best_similarity = 0.0
    for index in candidate_indexes:
        existing = candidates[index]
        similarity = estimate_jaccard(signature, existing.signature)
        if similarity > best_similarity:
            best_similarity = similarity
            best_id = str(existing.row["phase2_prompt_id"])
    if best_id is not None and best_similarity >= threshold:
        return best_id, best_similarity
    return None, best_similarity


def _candidate_from_record(record: Any, *, source_name: str, order: int) -> PromptCandidate:
    prompt = str(record.prompt or "")
    target_text = str(record.text or "")
    phase2_prompt_id = str(record.record_id)
    prompt_hash = _content_hash(prompt)
    target_hash = _content_hash(target_text)
    prompt_tokens = int(count_tokens(prompt))
    target_tokens = int(count_tokens(target_text))
    row = {
        "phase2_prompt_id": phase2_prompt_id,
        "source": source_name,
        "corpus": _metadata_value(record, "corpus", "corpus_name", default=source_name),
        "ladder": _metadata_value(record, "ladder"),
        "source_dataset": _metadata_value(record, "source_dataset", "hf_name", default=source_name),
        "stratum": _metadata_value(record, "stratum", "inferred_stratum", default="unknown"),
        "provenance": _metadata_value(
            record,
            "provenance",
            "source_dataset",
            "dataset",
            "data_source",
            "source",
            "domain",
            default="unknown",
        ),
        "source_record_id": record.record_id,
        "doc_id": record.metadata.get("doc_id") or record.record_id,
        "prompt_id": record.record_id,
        "text_role": record.metadata.get("text_role") or "target_response",
        "split": record.split,
        "prompt": prompt,
        "text": target_text,
        "prompt_hash": prompt_hash,
        "target_hash": target_hash,
        "prompt_tokens": prompt_tokens,
        "target_tokens": target_tokens,
        "source_row_index": record.row_index,
        "source_sample_key": record.metadata.get("sample_key"),
    }
    manifest_row = {key: value for key, value in row.items() if key not in {"prompt", "text"}}
    manifest_row["prompt_chars"] = len(prompt)
    manifest_row["target_chars"] = len(target_text)
    normalized_prompt = normalize_for_minhash(prompt)
    return PromptCandidate(
        order=order,
        row=row,
        manifest_row=manifest_row,
        signature=tuple(),
        normalized_prompt=normalized_prompt,
    )


def _prepare_candidates(args: argparse.Namespace) -> tuple[list[PromptCandidate], PrepStats]:
    SamplingConfig, iter_corpus_records, DEFAULT_ID_FIELDS = _load_components()
    text_fields = [args.text_field] if args.text_field else None
    sampling = SamplingConfig(cap=None, seed=args.seed, strategy="first", max_scan=args.max_scan)
    id_fields = _id_fields_with_pair_id(DEFAULT_ID_FIELDS)
    candidates: list[PromptCandidate] = []
    exact_index: dict[str, str] = {}
    band_index: dict[tuple[int, tuple[int, ...]], set[int]] = defaultdict(set)
    seen_prompt_ids: set[str] = set()
    stats = PrepStats()
    order = 0

    for raw_source in args.input:
        source = _source_from_input(raw_source, split=args.split, hf_config=args.hf_config)
        records_kwargs: dict[str, Any] = {"sampling": sampling, "id_fields": id_fields}
        if text_fields:
            records_kwargs["text_fields"] = text_fields
        for record in tqdm(
            iter_corpus_records(source, **records_kwargs),
            desc=f"phase2-prompts:{source.name}",
            unit="row",
        ):
            stats.scanned += 1
            if not record.prompt:
                stats.missing_prompt += 1
                continue
            if not record.text:
                stats.missing_text += 1
                continue
            if record.record_id in seen_prompt_ids:
                stats.duplicate_prompt_id += 1
                continue
            seen_prompt_ids.add(record.record_id)
            candidate = _candidate_from_record(record, source_name=source.name, order=order)
            if not candidate.normalized_prompt:
                stats.missing_prompt += 1
                continue
            signature = minhash_signature(
                record.prompt,
                num_hashes=args.minhash_size,
                shingle_size=args.minhash_shingle_tokens,
            )
            near_duplicate_id, similarity = _nearest_duplicate(
                signature=signature,
                normalized_prompt=candidate.normalized_prompt,
                exact_index=exact_index,
                band_index=band_index,
                candidates=candidates,
                threshold=args.near_duplicate_threshold,
                band_size=args.minhash_band_size,
            )
            if near_duplicate_id is not None:
                stats.near_duplicate_prompt += 1
                continue
            candidate = PromptCandidate(
                order=candidate.order,
                row={
                    **candidate.row,
                    "minhash_similarity_to_nearest": similarity,
                },
                manifest_row={
                    **candidate.manifest_row,
                    "minhash_similarity_to_nearest": similarity,
                },
                signature=signature,
                normalized_prompt=candidate.normalized_prompt,
            )
            index = len(candidates)
            candidates.append(candidate)
            exact_index[candidate.normalized_prompt] = str(candidate.row["phase2_prompt_id"])
            for band in signature_bands(signature, band_size=args.minhash_band_size):
                band_index[band].add(index)
            stats.eligible += 1
            order += 1
    return candidates, stats


def _select_candidates(candidates: list[PromptCandidate], args: argparse.Namespace) -> list[PromptCandidate]:
    if args.sampling_strategy == "first":
        return candidates[: args.sample_size]
    return sorted(candidates, key=lambda candidate: _selection_key(candidate, seed=args.seed))[: args.sample_size]


def run_prepare_phase2_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if not 0.0 <= args.near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")
    started_at = time.perf_counter()
    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "prompts", *args.wandb_tag],
        config={
            "inputs": args.input,
            "split": args.split,
            "hf_config": args.hf_config,
            "sample_size": args.sample_size,
            "max_scan": args.max_scan,
            "seed": args.seed,
            "sampling_strategy": args.sampling_strategy,
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "minhash_size": args.minhash_size,
            "minhash_shingle_tokens": args.minhash_shingle_tokens,
            "minhash_band_size": args.minhash_band_size,
            "output": str(args.output),
            "manifest_output": str(args.manifest_output),
        },
    )
    try:
        candidates, stats = _prepare_candidates(args)
        selected = _select_candidates(candidates, args)
        rows = [candidate.row for candidate in selected]
        manifest_rows = [candidate.manifest_row for candidate in selected]
        stats.selected = len(selected)
        stats.prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
        stats.target_tokens = sum(int(row["target_tokens"]) for row in rows)
        stats.wall_seconds = time.perf_counter() - started_at
        stats.bytes_written = _write_jsonl(args.output, rows)
        _write_csv(args.manifest_output, manifest_rows)
        summary = {
            "inputs": args.input,
            "rows_scanned": stats.scanned,
            "missing_prompt": stats.missing_prompt,
            "missing_text": stats.missing_text,
            "duplicate_prompt_id": stats.duplicate_prompt_id,
            "near_duplicate_prompt": stats.near_duplicate_prompt,
            "eligible_prompts": stats.eligible,
            "selected_prompts": stats.selected,
            "prompt_tokens": stats.prompt_tokens,
            "target_tokens": stats.target_tokens,
            "wall_seconds": stats.wall_seconds,
            "prompts_per_sec": stats.selected / stats.wall_seconds if stats.wall_seconds > 0 else 0.0,
            "peak_rss_mb": _peak_rss_mb(),
            "output_size_mb": stats.bytes_written / (1024.0 * 1024.0),
            "output": str(args.output),
            "manifest_output": str(args.manifest_output),
        }
        if args.summary_output is not None:
            _write_summary(args.summary_output, summary)
        run.log(
            {
                "phase2_prompts/rows_scanned": stats.scanned,
                "phase2_prompts/missing_prompt": stats.missing_prompt,
                "phase2_prompts/missing_text": stats.missing_text,
                "phase2_prompts/duplicate_prompt_id": stats.duplicate_prompt_id,
                "phase2_prompts/near_duplicate_prompt": stats.near_duplicate_prompt,
                "phase2_prompts/eligible_prompts": stats.eligible,
                "phase2_prompts/selected_prompts": stats.selected,
                "phase2_prompts/prompt_tokens": stats.prompt_tokens,
                "phase2_prompts/target_tokens": stats.target_tokens,
                "phase2_prompts/wall_seconds": stats.wall_seconds,
                "phase2_prompts/prompts_per_sec": summary["prompts_per_sec"],
                "phase2_prompts/peak_rss_mb": _peak_rss_mb(),
                "phase2_prompts/output_size_mb": summary["output_size_mb"],
            }
        )
        log_summary_table(run, "phase2_prompt_manifest", manifest_rows[:MAX_WANDB_MANIFEST_ROWS])
        return manifest_rows
    finally:
        run.finish()


def main() -> None:
    args = build_parser().parse_args()
    rows = run_prepare_phase2_prompts(args)
    print(f"Wrote {len(rows)} Phase 2 prompt rows to {args.output}")


if __name__ == "__main__":
    main()
