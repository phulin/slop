from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


@dataclass(frozen=True)
class ArtifactManifestRow:
    path: str
    bytes: int
    sha256: str
    modified_utc: str
    format: str
    records: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a manifest for retained local artifacts.")
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Manifest CSV path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON manifest path.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional Markdown summary.")
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="artifact-manifest")
    parser.add_argument("--wandb-job-type", default="manifest")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_records(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _format_for_path(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "unknown"


def manifest_path(path: Path) -> ArtifactManifestRow:
    if not path.exists():
        raise FileNotFoundError(path)
    stat = path.stat()
    return ArtifactManifestRow(
        path=str(path),
        bytes=stat.st_size,
        sha256=_sha256(path),
        modified_utc=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        format=_format_for_path(path),
        records=_csv_records(path),
    )


def _write_csv(path: Path, rows: list[ArtifactManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["path", "bytes", "sha256", "modified_utc", "format", "records"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_json(path: Path, rows: list[ArtifactManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_summary(path: Path, rows: list[ArtifactManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(row.bytes for row in rows)
    lines = [
        "# Retained Artifact Manifest",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Artifacts: {len(rows)}",
        f"Total bytes: {total_bytes}",
        "",
        "| Path | Records | Bytes | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        records = "" if row.records is None else str(row.records)
        lines.append(f"| `{row.path}` | {records} | {row.bytes} | `{row.sha256}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [manifest_path(path) for path in args.input]
    _write_csv(args.output, rows)
    if args.json_output is not None:
        _write_json(args.json_output, rows)
    if args.summary_output is not None:
        _write_summary(args.summary_output, rows)

    row_dicts = [asdict(row) for row in rows]
    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage1", "manifest", *args.wandb_tag],
        config={
            "inputs": [str(path) for path in args.input],
            "output": str(args.output),
            "json_output": str(args.json_output) if args.json_output is not None else None,
            "summary_output": str(args.summary_output) if args.summary_output is not None else None,
        },
    )
    try:
        run.log(
            {
                "manifest/artifacts": len(rows),
                "manifest/total_bytes": sum(row.bytes for row in rows),
                "manifest/total_records": sum(row.records or 0 for row in rows),
            }
        )
        log_summary_table(run, "artifact_manifest", row_dicts)
    finally:
        run.finish()
    return row_dicts


def main() -> None:
    args = build_parser().parse_args()
    rows = run_manifest(args)
    print(f"Wrote {len(rows)} manifest rows to {args.output}")


if __name__ == "__main__":
    main()
