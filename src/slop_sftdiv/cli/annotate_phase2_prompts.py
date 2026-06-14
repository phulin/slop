from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from slop_sftdiv.cli.prepare_phase2_prompts import (
    _load_metadata_bucket_maps,
    _metadata_buckets,
)
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


RAW_TEXT_FIELDS = {"prompt", "text"}
MAX_WANDB_MANIFEST_ROWS = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate an existing Phase 2 prompt package without resampling rows."
    )
    parser.add_argument("--input", type=Path, required=True, help="Existing Phase 2 prompt JSONL.")
    parser.add_argument(
        "--metadata-bucket-map",
        action="append",
        type=Path,
        required=True,
        help=(
            "JSON mapping that materializes a normalized metadata field. "
            "Schema: output_field, source_field/source_fields, values, default."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-data")
    parser.add_argument("--wandb-job-type", default="annotate-prompts")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _iter_jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            yield row


def _manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    manifest = {key: value for key, value in row.items() if key not in RAW_TEXT_FIELDS}
    if "prompt" in row:
        manifest["prompt_chars"] = len(str(row.get("prompt") or ""))
    if "text" in row:
        manifest["target_chars"] = len(str(row.get("text") or ""))
    return manifest


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


def _bucket_count_rows(rows: list[dict[str, Any]], output_fields: list[str]) -> list[dict[str, Any]]:
    count_rows: list[dict[str, Any]] = []
    for field in output_fields:
        counts = Counter(str(row.get(field, "")) for row in rows)
        for value, count in sorted(counts.items()):
            count_rows.append({"field": field, "value": value, "count": count})
    return count_rows


def run_annotate_phase2_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    started_at = time.perf_counter()
    bucket_maps = _load_metadata_bucket_maps(args.metadata_bucket_map)
    output_fields = [item.output_field for item in bucket_maps]
    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "prompts", "annotation", *args.wandb_tag],
        config={
            "input": str(args.input),
            "metadata_bucket_maps": [
                {
                    "path": str(item.path),
                    "output_field": item.output_field,
                    "source_fields": item.source_fields,
                    "default": item.default,
                }
                for item in bucket_maps
            ],
            "output": str(args.output),
            "manifest_output": str(args.manifest_output),
        },
    )
    try:
        rows: list[dict[str, Any]] = []
        for row in tqdm(_iter_jsonl_rows(args.input), desc=f"annotate-prompts:{args.input.name}"):
            annotated = {**row, **_metadata_buckets(row, bucket_maps)}
            rows.append(annotated)
        manifest_rows = [_manifest_row(row) for row in rows]
        bytes_written = _write_jsonl(args.output, rows)
        _write_csv(args.manifest_output, manifest_rows)
        bucket_rows = _bucket_count_rows(rows, output_fields)
        wall_seconds = time.perf_counter() - started_at
        summary = {
            "input": str(args.input),
            "output": str(args.output),
            "manifest_output": str(args.manifest_output),
            "rows": len(rows),
            "metadata_fields": output_fields,
            "bucket_counts": bucket_rows,
            "wall_seconds": wall_seconds,
            "rows_per_sec": len(rows) / wall_seconds if wall_seconds > 0 else 0.0,
            "peak_rss_mb": _peak_rss_mb(),
            "output_size_mb": bytes_written / (1024.0 * 1024.0),
        }
        if args.summary_output is not None:
            _write_summary(args.summary_output, summary)
        run.log(
            {
                "phase2_prompt_annotation/rows": len(rows),
                "phase2_prompt_annotation/wall_seconds": wall_seconds,
                "phase2_prompt_annotation/rows_per_sec": summary["rows_per_sec"],
                "phase2_prompt_annotation/peak_rss_mb": summary["peak_rss_mb"],
                "phase2_prompt_annotation/output_size_mb": summary["output_size_mb"],
            }
        )
        log_summary_table(run, "phase2_prompt_annotation_bucket_counts", bucket_rows)
        log_summary_table(
            run,
            "phase2_prompt_annotation_manifest",
            manifest_rows[:MAX_WANDB_MANIFEST_ROWS],
        )
        return manifest_rows
    finally:
        run.finish()


def main() -> None:
    args = build_parser().parse_args()
    rows = run_annotate_phase2_prompts(args)
    print(f"Wrote {len(rows)} annotated Phase 2 prompt rows to {args.output}")


if __name__ == "__main__":
    main()
