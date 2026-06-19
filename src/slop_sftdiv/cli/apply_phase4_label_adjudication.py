from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.import_phase4_blind_labels import import_blind_rows
from slop_sftdiv.cli.summarize_phase4_human_labels import (
    ALLOWED_PERCEPTION_LABELS,
    ALLOWED_TAXONOMY_LABELS,
)


DEFAULT_BASE = Path("artifacts/phase4/modernbert_detector_combined_v2_clean")
DEFAULT_BLIND_MAP = DEFAULT_BASE / "phase4_human_perceptibility_blind_pilot_20_each_map.csv"
DEFAULT_ADJUDICATION = DEFAULT_BASE / "phase4_human_perceptibility_pilot_disagreements.csv"
DEFAULT_OUTPUT = DEFAULT_BASE / "phase4_human_perceptibility_blind_pilot_adjudicated.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply Phase 4 blind-label adjudications to two annotator sheets."
    )
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--blind-map", type=Path, default=DEFAULT_BLIND_MAP)
    parser.add_argument("--adjudications", type=Path, default=DEFAULT_ADJUDICATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require adjudicated labels for every disagreement.",
    )
    return parser


def run_apply_phase4_label_adjudication(args: argparse.Namespace) -> list[dict[str, Any]]:
    map_rows = _read_csv(args.blind_map)
    rows_a = import_blind_rows(_read_csv(args.annotator_a), map_rows, strict=True)
    rows_b = import_blind_rows(_read_csv(args.annotator_b), map_rows, strict=True)
    adjudication_rows = _read_csv(args.adjudications)
    merged = apply_adjudications(rows_a, rows_b, adjudication_rows, strict=args.strict)
    _write_jsonl(args.output, merged)
    return merged


def apply_adjudications(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    adjudication_rows: list[dict[str, str]],
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    by_id_a = _rows_by_blind_id(rows_a, "annotator A")
    by_id_b = _rows_by_blind_id(rows_b, "annotator B")
    if set(by_id_a) != set(by_id_b):
        raise ValueError("annotator files have different blind_id sets")
    adjudications = _adjudications_by_blind_id(adjudication_rows)
    needs_adjudication_by_id = {
        blind_id
        for blind_id, row_a in by_id_a.items()
        if _label(row_a, "human_perceived_slop")
        != _label(by_id_b[blind_id], "human_perceived_slop")
        or _label(row_a, "shaib_taxonomy_label")
        != _label(by_id_b[blind_id], "shaib_taxonomy_label")
    }
    _validate_adjudication_ids(
        adjudication_ids=set(adjudications),
        valid_ids=set(by_id_a),
        needed_ids=needs_adjudication_by_id,
    )
    merged: list[dict[str, Any]] = []
    for blind_id in sorted(by_id_a):
        row_a = by_id_a[blind_id]
        row_b = by_id_b[blind_id]
        perception_a = _label(row_a, "human_perceived_slop")
        perception_b = _label(row_b, "human_perceived_slop")
        taxonomy_a = _label(row_a, "shaib_taxonomy_label")
        taxonomy_b = _label(row_b, "shaib_taxonomy_label")
        needs_adjudication = perception_a != perception_b or taxonomy_a != taxonomy_b
        if needs_adjudication:
            adjudication = adjudications.get(blind_id)
            if adjudication is None:
                if strict:
                    raise ValueError(f"blind_id={blind_id!r}: missing adjudication row")
                final_perception = ""
                final_taxonomy = ""
                notes = "missing adjudication"
            else:
                final_perception = _adjudicated_label(
                    adjudication,
                    "adjudicated_human_perceived_slop",
                    ALLOWED_PERCEPTION_LABELS,
                    blind_id,
                    strict=strict,
                )
                final_taxonomy = _adjudicated_label(
                    adjudication,
                    "adjudicated_shaib_taxonomy_label",
                    ALLOWED_TAXONOMY_LABELS,
                    blind_id,
                    strict=strict,
                )
                notes = adjudication.get("adjudication_notes", "")
        else:
            final_perception = perception_a
            final_taxonomy = taxonomy_a
            notes = "annotator consensus"
        output_row = dict(row_a)
        output_row["human_perceived_slop"] = final_perception
        output_row["shaib_taxonomy_label"] = final_taxonomy
        output_row["notes"] = notes
        output_row["annotator_a_human_perceived_slop"] = perception_a
        output_row["annotator_b_human_perceived_slop"] = perception_b
        output_row["annotator_a_shaib_taxonomy_label"] = taxonomy_a
        output_row["annotator_b_shaib_taxonomy_label"] = taxonomy_b
        output_row["adjudication_status"] = "adjudicated" if needs_adjudication else "consensus"
        merged.append(output_row)
    return merged


def _adjudicated_label(
    row: dict[str, str],
    field: str,
    allowed: set[str],
    blind_id: str,
    *,
    strict: bool,
) -> str:
    value = str(row.get(field, "")).strip().lower()
    if value in allowed and value:
        return value
    if strict:
        raise ValueError(f"blind_id={blind_id!r}: missing or invalid {field}={value!r}")
    return value if value in allowed else ""


def _rows_by_blind_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        blind_id = str(row.get("blind_id", "")).strip()
        if not blind_id:
            raise ValueError(f"{label}: row missing blind_id")
        if blind_id in by_id:
            raise ValueError(f"{label}: duplicate blind_id={blind_id!r}")
        by_id[blind_id] = row
    return by_id


def _adjudications_by_blind_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        blind_id = str(row.get("blind_id", "")).strip()
        if not blind_id:
            continue
        if blind_id in by_id:
            raise ValueError(f"duplicate adjudication blind_id={blind_id!r}")
        by_id[blind_id] = row
    return by_id


def _validate_adjudication_ids(
    *,
    adjudication_ids: set[str],
    valid_ids: set[str],
    needed_ids: set[str],
) -> None:
    unknown_ids = sorted(adjudication_ids - valid_ids)
    if unknown_ids:
        preview = ", ".join(unknown_ids[:10])
        raise ValueError(f"adjudication rows reference unknown blind_id values: {preview}")
    unnecessary_ids = sorted(adjudication_ids - needed_ids)
    if unnecessary_ids:
        preview = ", ".join(unnecessary_ids[:10])
        raise ValueError(f"adjudication rows provided for consensus blind_id values: {preview}")


def _label(row: dict[str, Any], field: str) -> str:
    return str(row.get(field, "")).strip().lower()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_apply_phase4_label_adjudication(args)
    adjudicated = sum(row.get("adjudication_status") == "adjudicated" for row in rows)
    print(f"Wrote {len(rows)} adjudicated Phase 4 label rows to {args.output}")
    print(f"Rows requiring adjudication: {adjudicated}")


if __name__ == "__main__":
    main()
