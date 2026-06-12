from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


MAX_WANDB_MANIFEST_ROWS = 1000


@dataclass
class SampleStats:
    scanned: int = 0
    kept: int = 0
    tokens: int = 0
    bytes_written: int = 0
    wall_seconds: float = 0.0

    @property
    def rows_per_sec(self) -> float:
        return self.kept / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def scanned_rows_per_sec(self) -> float:
        return self.scanned / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def output_size_mb(self) -> float:
        return self.bytes_written / (1024.0 * 1024.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a retained normalized Stage 1 corpus sample.")
    parser.add_argument("--input", required=True, help="Local JSONL file or HF dataset id.")
    parser.add_argument("--split", default="train", help="Dataset split for HF inputs.")
    parser.add_argument("--hf-config", default=None, help="Optional Hugging Face dataset config name.")
    parser.add_argument("--text-field", default=None, help="Override text field name.")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--sampling-strategy",
        default="hash_reservoir",
        choices=["first", "hash_reservoir"],
        help="Deterministic row-selection strategy within each stratum.",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=None,
        help="Maximum retained source records or preference pairs overall.",
    )
    parser.add_argument(
        "--target-rows-per-stratum",
        type=int,
        default=None,
        help="Maximum retained source records or preference pairs for each requested stratum.",
    )
    parser.add_argument("--max-scan", type=int, default=None, help="Maximum source rows to scan.")
    parser.add_argument(
        "--stratum",
        action="append",
        default=[],
        help="Retain only this inferred stratum; repeat for multiple strata.",
    )
    parser.add_argument(
        "--exclude-stratum",
        action="append",
        default=[],
        help="Exclude this inferred stratum; repeat for multiple strata.",
    )
    parser.add_argument("--corpus", required=True, help="Canonical corpus name for output rows.")
    parser.add_argument("--ladder", required=True, help="Model/data ladder label, e.g. olmo3.")
    parser.add_argument("--source-dataset", default=None, help="Canonical source dataset id.")
    parser.add_argument("--provenance", default=None, help="Fallback provenance label.")
    parser.add_argument("--text-role", default="text", help="Text role label for output rows.")
    parser.add_argument(
        "--expand-preference-roles",
        action="store_true",
        help="Write one retained row each for chosen and rejected preference texts.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Retained normalized JSONL output.")
    parser.add_argument(
        "--manifest-output",
        type=Path,
        required=True,
        help="Per-record manifest path; .parquet writes Parquet, otherwise CSV.",
    )
    parser.add_argument("--manifest-csv-output", type=Path, default=None)
    parser.add_argument("--manifest-json-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="corpus-sampling")
    parser.add_argument("--wandb-job-type", default="sample")
    parser.add_argument("--wandb-tag", action="append", default=[], help="Additional W&B tag.")
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _load_components():
    from slop_sftdiv.data import CorpusSource, SamplingConfig, iter_corpus_records
    from slop_sftdiv.data.corpora import DEFAULT_ID_FIELDS
    from slop_sftdiv.features.tier1_matchers import SENTENCE_RE, count_tokens

    return CorpusSource, SamplingConfig, iter_corpus_records, SENTENCE_RE, count_tokens, DEFAULT_ID_FIELDS


def _source_from_input(raw_input: str, *, split: str, hf_config: str | None, source_dataset: str):
    CorpusSource, _, _, _, _, _ = _load_components()
    path = Path(raw_input)
    if path.exists():
        return CorpusSource.jsonl(path, name=source_dataset)
    return CorpusSource.hf(raw_input, name=source_dataset, split=split, hf_config=hf_config, streaming=True)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _paragraph_count(text: str) -> int:
    return max(1, sum(1 for part in text.split("\n\n") if part.strip()))


def _sentence_count(text: str, sentence_re: Any) -> int:
    return max(1, len(sentence_re.findall(text)) or 1)


def _sample_key(record: Any) -> int:
    return int(str(record.metadata.get("sample_key", "0")), 16)


def _record_stratum(record: Any) -> str:
    return str(record.metadata.get("stratum") or record.metadata.get("inferred_stratum") or "unknown")


def _record_provenance(record: Any, fallback: str | None) -> str:
    for field_name in (
        "provenance",
        "source_dataset",
        "dataset",
        "data_source",
        "source",
        "domain",
    ):
        value = record.metadata.get(field_name)
        if value:
            return str(value)
    return fallback or "unknown"


def _json_safe_metadata(metadata: Any) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in dict(metadata).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
        else:
            safe[str(key)] = json.dumps(value, sort_keys=True, default=str)
    return safe


def _output_row(
    record: Any,
    *,
    text: str,
    corpus: str,
    ladder: str,
    source_dataset: str,
    provenance: str | None,
    text_role: str,
    sentence_re: Any,
    count_tokens: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    token_count = int(count_tokens(text))
    sentence_count = _sentence_count(text, sentence_re)
    paragraph_count = _paragraph_count(text)
    content_hash = _content_hash(text)
    stratum = _record_stratum(record)
    is_preference_pair = bool(record.chosen and record.rejected)
    pair_id = _unique_record_id(record) if is_preference_pair else None
    doc_id = f"{pair_id}:{text_role}" if pair_id is not None else record.record_id
    row = {
        "corpus": corpus,
        "ladder": ladder,
        "source_dataset": source_dataset,
        "split": record.split,
        "stratum": stratum,
        "provenance": _record_provenance(record, provenance),
        "prompt_id": record.record_id if record.prompt or is_preference_pair else None,
        "doc_id": doc_id,
        "pair_id": pair_id,
        "source_record_id": record.record_id,
        "text_role": text_role,
        "text": text,
        "token_count": token_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "content_hash": content_hash,
        "row_index": record.row_index,
        "sample_key": record.metadata.get("sample_key"),
        "sampling_seed": record.metadata.get("sampling_seed"),
        "sampling_strategy": record.metadata.get("sampling_strategy"),
        "sampling_max_scan": record.metadata.get("sampling_max_scan"),
    }
    metadata = _json_safe_metadata(record.metadata)
    for key, value in metadata.items():
        row[f"metadata.{key}"] = value
    manifest_row = {key: value for key, value in row.items() if key != "text"}
    manifest_row["text_chars"] = len(text)
    return row, manifest_row


def _record_text_roles(record: Any, default_text_role: str, expand_preference_roles: bool) -> list[tuple[str, str]]:
    if expand_preference_roles and record.chosen and record.rejected:
        return [("chosen", record.chosen), ("rejected", record.rejected)]
    if expand_preference_roles:
        return []
    return [(default_text_role, record.text)]


def _unique_record_id(record: Any) -> str:
    return f"{record.record_id}#row{record.row_index}"


def _id_fields_with_pair_id(default_id_fields: tuple[str, ...]) -> tuple[str, ...]:
    fields = ("pair_id", "pair.id", "preference_pair_id", "prompt_id", *default_id_fields)
    return tuple(dict.fromkeys(fields))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
            handle.write(payload)
            handle.write("\n")
            bytes_written += len(payload.encode("utf-8")) + 1
    return bytes_written


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": summary, "records": rows}, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _write_summary(path: Path, summary: dict[str, Any], stratum_counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Retained Corpus Sample",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Strata", "", "| Stratum | Rows |", "|---|---:|"])
    for stratum, count in sorted(stratum_counts.items()):
        lines.append(f"| `{stratum}` | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _select_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], SampleStats]:
    _, SamplingConfig, iter_corpus_records, sentence_re, count_tokens, default_id_fields = _load_components()
    source_dataset = args.source_dataset or args.input
    source = _source_from_input(args.input, split=args.split, hf_config=args.hf_config, source_dataset=source_dataset)
    text_fields = [args.text_field] if args.text_field else None
    sampling = SamplingConfig(cap=None, seed=args.seed, strategy="first", max_scan=args.max_scan)
    records_kwargs: dict[str, Any] = {
        "sampling": sampling,
        "id_fields": _id_fields_with_pair_id(default_id_fields),
    }
    if text_fields:
        records_kwargs["text_fields"] = text_fields
    included = set(args.stratum)
    excluded = set(args.exclude_stratum)
    per_stratum_target = args.target_rows_per_stratum
    total_target = args.target_rows
    heaps: dict[str, list[tuple[tuple[int, int], list[tuple[dict[str, Any], dict[str, Any]]]]]] = {}
    sequence = 0
    stats = SampleStats()
    started_at = time.perf_counter()

    for record in tqdm(iter_corpus_records(source, **records_kwargs), desc=f"sample:{source.name}", unit="doc"):
        stats.scanned += 1
        stratum = _record_stratum(record)
        if included and stratum not in included:
            continue
        if stratum in excluded:
            continue
        role_rows = [
            _output_row(
                record,
                text=text,
                corpus=args.corpus,
                ladder=args.ladder,
                source_dataset=source_dataset,
                provenance=args.provenance,
                text_role=text_role,
                sentence_re=sentence_re,
                count_tokens=count_tokens,
            )
            for text_role, text in _record_text_roles(
                record,
                default_text_role=args.text_role,
                expand_preference_roles=args.expand_preference_roles,
            )
        ]
        if not role_rows:
            continue
        sequence += 1
        if args.sampling_strategy == "first":
            heap = heaps.setdefault(stratum, [])
            if per_stratum_target is not None and len(heap) >= per_stratum_target:
                continue
            if total_target is not None and sum(len(items) for items in heaps.values()) >= total_target:
                break
            heap.append(((sequence, sequence), role_rows))
            continue

        key = _sample_key(record)
        item = ((-key, sequence), role_rows)
        if per_stratum_target is None:
            heap = heaps.setdefault("all", [])
            target = total_target
        else:
            heap = heaps.setdefault(stratum, [])
            target = per_stratum_target
        if target is None or len(heap) < target:
            heap.append(item)
            heap.sort()
        elif item[0] > heap[0][0]:
            heap[0] = item
            heap.sort()

    selected: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for heap in heaps.values():
        for item in heap:
            for row, manifest_row in item[1]:
                selected.append((item[0][1], row, manifest_row))
    selected.sort(key=lambda item: item[0])
    rows = [item[1] for item in selected]
    manifest_rows = [item[2] for item in selected]
    stats.kept = len(rows)
    stats.tokens = sum(int(row["token_count"]) for row in rows)
    stats.wall_seconds = time.perf_counter() - started_at
    return rows, manifest_rows, stats


def run_sample(args: argparse.Namespace) -> list[dict[str, Any]]:
    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        mode=args.wandb_mode,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        tags=["stage1", "corpus", "sample", *args.wandb_tag],
        config={
            "input": args.input,
            "split": args.split,
            "hf_config": args.hf_config,
            "seed": args.seed,
            "sampling_strategy": args.sampling_strategy,
            "target_rows": args.target_rows,
            "target_rows_per_stratum": args.target_rows_per_stratum,
            "max_scan": args.max_scan,
            "strata": args.stratum,
            "exclude_strata": args.exclude_stratum,
            "corpus": args.corpus,
            "ladder": args.ladder,
            "source_dataset": args.source_dataset,
            "text_role": args.text_role,
            "expand_preference_roles": args.expand_preference_roles,
            "output": str(args.output),
            "manifest_output": str(args.manifest_output),
        },
    )
    try:
        rows, manifest_rows, stats = _select_records(args)
        stats.bytes_written = _write_jsonl(args.output, rows)
        _write_manifest(args.manifest_output, manifest_rows)
        if args.manifest_csv_output is not None:
            _write_csv(args.manifest_csv_output, manifest_rows)
        stratum_counts: dict[str, int] = {}
        for row in manifest_rows:
            stratum = str(row["stratum"])
            stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
        summary = {
            "input": args.input,
            "rows_scanned": stats.scanned,
            "rows_retained": stats.kept,
            "tokens": stats.tokens,
            "wall_seconds": stats.wall_seconds,
            "rows_per_sec": stats.rows_per_sec,
            "scanned_rows_per_sec": stats.scanned_rows_per_sec,
            "tokens_per_sec": stats.tokens_per_sec,
            "peak_rss_mb": _peak_rss_mb(),
            "output_size_mb": stats.output_size_mb,
            "manifest_output": str(args.manifest_output),
            "output": str(args.output),
        }
        if args.manifest_json_output is not None:
            _write_json(args.manifest_json_output, manifest_rows, summary)
        if args.summary_output is not None:
            _write_summary(args.summary_output, summary, stratum_counts)
        run.log(
            {
                "corpus/scanned_rows": stats.scanned,
                "corpus/rows": stats.kept,
                "corpus/tokens": stats.tokens,
                "corpus/wall_seconds": stats.wall_seconds,
                "corpus/rows_per_sec": stats.rows_per_sec,
                "corpus/scanned_rows_per_sec": stats.scanned_rows_per_sec,
                "corpus/tokens_per_sec": stats.tokens_per_sec,
                "corpus/peak_rss_mb": _peak_rss_mb(),
                "corpus/output_size_mb": stats.output_size_mb,
                "corpus/strata": len(stratum_counts),
            }
        )
        log_summary_table(
            run,
            "corpus_manifest_sample",
            manifest_rows[:MAX_WANDB_MANIFEST_ROWS],
        )
        log_summary_table(
            run,
            "corpus_stratum_counts",
            [{"stratum": key, "rows": value} for key, value in sorted(stratum_counts.items())],
        )
        return manifest_rows
    finally:
        run.finish()


def main() -> None:
    args = build_parser().parse_args()
    rows = run_sample(args)
    print(f"Wrote {len(rows)} retained rows to {args.output}")


if __name__ == "__main__":
    main()
