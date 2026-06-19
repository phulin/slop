from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.summarize_phase4_human_labels import (
    ALLOWED_PERCEPTION_LABELS,
    ALLOWED_TAXONOMY_LABELS,
)


DEFAULT_BASE = Path("artifacts/phase4/modernbert_detector_combined_v2_clean")
DEFAULT_BLIND_LABELS = DEFAULT_BASE / "phase4_human_perceptibility_blind_pilot_20_each.csv"
DEFAULT_BLIND_MAP = DEFAULT_BASE / "phase4_human_perceptibility_blind_pilot_20_each_map.csv"
DEFAULT_OUTPUT = DEFAULT_BASE / "phase4_human_perceptibility_blind_pilot_labeled.jsonl"
REQUIRED_MAP_FIELDS = (
    "blind_id",
    "annotation_id",
    "feature",
    "candidate_label",
    "cluster",
    "source",
    "record_id",
    "matched_text",
    "snippet",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import blinded Phase 4 human labels into canonical JSONL rows."
    )
    parser.add_argument("--blind-labels", type=Path, default=DEFAULT_BLIND_LABELS)
    parser.add_argument("--blind-map", type=Path, default=DEFAULT_BLIND_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any blind row is unlabeled or has an unknown label.",
    )
    return parser


def run_import_phase4_blind_labels(args: argparse.Namespace) -> list[dict[str, Any]]:
    blind_rows = _read_csv(args.blind_labels)
    map_rows = _read_csv(args.blind_map)
    canonical_rows = import_blind_rows(blind_rows, map_rows, strict=args.strict)
    _write_jsonl(args.output, canonical_rows)
    return canonical_rows


def import_blind_rows(
    blind_rows: list[dict[str, str]],
    map_rows: list[dict[str, str]],
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    map_by_blind_id = _map_by_blind_id(map_rows)
    canonical_rows: list[dict[str, Any]] = []
    seen_blind_ids: set[str] = set()
    for row_number, blind_row in enumerate(blind_rows, start=2):
        blind_id = str(blind_row.get("blind_id", "")).strip()
        if not blind_id:
            raise ValueError(f"blind labels row {row_number}: missing blind_id")
        if blind_id in seen_blind_ids:
            raise ValueError(f"blind labels row {row_number}: duplicate blind_id={blind_id!r}")
        seen_blind_ids.add(blind_id)
        mapped = map_by_blind_id.get(blind_id)
        if mapped is None:
            raise ValueError(f"blind labels row {row_number}: no map row for {blind_id!r}")
        perception = str(blind_row.get("human_perceived_slop", "")).strip().lower()
        taxonomy = str(blind_row.get("shaib_taxonomy_label", "")).strip().lower()
        _validate_label(
            value=perception,
            allowed=ALLOWED_PERCEPTION_LABELS,
            field="human_perceived_slop",
            blind_id=blind_id,
            strict=strict,
        )
        _validate_label(
            value=taxonomy,
            allowed=ALLOWED_TAXONOMY_LABELS,
            field="shaib_taxonomy_label",
            blind_id=blind_id,
            strict=strict,
        )
        _validate_protected_field(
            blind_row=blind_row,
            mapped=mapped,
            field="matched_text",
            blind_id=blind_id,
        )
        _validate_protected_field(
            blind_row=blind_row,
            mapped=mapped,
            field="snippet",
            blind_id=blind_id,
        )
        if strict and not perception:
            raise ValueError(f"blind_id={blind_id!r}: missing human_perceived_slop")
        canonical_rows.append(
            {
                "annotation_id": mapped.get("annotation_id", ""),
                "feature": mapped.get("feature", ""),
                "cluster": mapped.get("cluster", ""),
                "candidate_label": mapped.get("candidate_label", ""),
                "source": mapped.get("source", ""),
                "record_id": mapped.get("record_id", ""),
                "matched_text": blind_row.get("matched_text", mapped.get("matched_text", "")),
                "snippet": blind_row.get("snippet", ""),
                "human_perceived_slop": perception,
                "shaib_taxonomy_label": taxonomy,
                "notes": blind_row.get("notes", ""),
                "blind_id": blind_id,
            }
        )
    missing_blind_ids = sorted(set(map_by_blind_id) - seen_blind_ids)
    if missing_blind_ids and strict:
        preview = ", ".join(missing_blind_ids[:10])
        raise ValueError(f"blind labels missing map IDs: {preview}")
    return canonical_rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _map_by_blind_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    mapped: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        blind_id = str(row.get("blind_id", "")).strip()
        if not blind_id:
            raise ValueError(f"blind map row {row_number}: missing blind_id")
        if blind_id in mapped:
            raise ValueError(f"blind map row {row_number}: duplicate blind_id={blind_id!r}")
        missing = [field for field in REQUIRED_MAP_FIELDS if field not in row]
        if missing:
            raise ValueError(
                f"blind map row {row_number}: missing required fields {', '.join(missing)}"
            )
        mapped[blind_id] = row
    return mapped


def _validate_label(
    *,
    value: str,
    allowed: set[str],
    field: str,
    blind_id: str,
    strict: bool,
) -> None:
    if value in allowed:
        return
    message = f"blind_id={blind_id!r}: unknown {field}={value!r}"
    if strict:
        raise ValueError(message)
    raise ValueError(message)


def _validate_protected_field(
    *,
    blind_row: dict[str, str],
    mapped: dict[str, str],
    field: str,
    blind_id: str,
) -> None:
    if field not in mapped:
        return
    if str(blind_row.get(field, "")) != str(mapped.get(field, "")):
        raise ValueError(f"blind_id={blind_id!r}: protected field {field!r} was edited")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_import_phase4_blind_labels(args)
    labeled = sum(1 for row in rows if str(row.get("human_perceived_slop", "")).strip())
    print(f"Wrote {len(rows)} canonical blind-label rows to {args.output}")
    print(f"Rows with human_perceived_slop labels: {labeled}")


if __name__ == "__main__":
    main()
