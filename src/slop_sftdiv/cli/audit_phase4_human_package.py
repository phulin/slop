from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "phase4_human_perceptibility_annotation_package.jsonl"
)
DEFAULT_OUTPUT = Path(
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "phase4_human_annotation_readiness.csv"
)
DEFAULT_SUMMARY = Path("docs/experiments/phase4_human_annotation_readiness.md")
DEFAULT_PILOT_OUTPUT = Path(
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "phase4_human_perceptibility_pilot_20_each.csv"
)
DEFAULT_BLIND_PILOT_OUTPUT = Path(
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "phase4_human_perceptibility_blind_pilot_20_each.csv"
)
DEFAULT_BLIND_PILOT_MAP_OUTPUT = Path(
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "phase4_human_perceptibility_blind_pilot_20_each_map.csv"
)
DEFAULT_BLIND_FULL_OUTPUT = Path(
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "phase4_human_perceptibility_blind_full.csv"
)
DEFAULT_BLIND_FULL_MAP_OUTPUT = Path(
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "phase4_human_perceptibility_blind_full_map.csv"
)
DEFAULT_HANDOFF_DIR = Path(
    "artifacts/phase4/modernbert_detector_combined_v2_clean/human_annotation_handoff"
)

REQUIRED_FIELDS = (
    "annotation_id",
    "feature",
    "cluster",
    "candidate_label",
    "source",
    "record_id",
    "matched_text",
    "snippet",
    "human_perceived_slop",
    "shaib_taxonomy_label",
    "notes",
)

PILOT_FIELDS = (
    "annotation_id",
    "feature",
    "candidate_label",
    "cluster",
    "source",
    "record_id",
    "matched_text",
    "snippet",
    "human_perceived_slop",
    "shaib_taxonomy_label",
    "notes",
)
BLIND_PILOT_FIELDS = (
    "blind_id",
    "matched_text",
    "snippet",
    "human_perceived_slop",
    "shaib_taxonomy_label",
    "notes",
)
BLIND_MAP_FIELDS = (
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
        description="Audit Phase 4 human-perceptibility annotation package readiness."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--pilot-output", type=Path, default=DEFAULT_PILOT_OUTPUT)
    parser.add_argument("--blind-pilot-output", type=Path, default=DEFAULT_BLIND_PILOT_OUTPUT)
    parser.add_argument(
        "--blind-pilot-map-output",
        type=Path,
        default=DEFAULT_BLIND_PILOT_MAP_OUTPUT,
    )
    parser.add_argument("--blind-full-output", type=Path, default=DEFAULT_BLIND_FULL_OUTPUT)
    parser.add_argument(
        "--blind-full-map-output",
        type=Path,
        default=DEFAULT_BLIND_FULL_MAP_OUTPUT,
    )
    parser.add_argument("--pilot-per-feature", type=int, default=20)
    parser.add_argument("--blind-seed", type=int, default=17)
    parser.add_argument("--expected-features", type=int, default=10)
    parser.add_argument("--expected-rows-per-feature", type=int, default=100)
    return parser


def run_audit_phase4_human_package(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows, package_warnings = _load_rows(args.input)
    summary_rows, audit_warnings = _summarize_rows(
        rows,
        expected_features=args.expected_features,
        expected_rows_per_feature=args.expected_rows_per_feature,
    )
    warnings = package_warnings + audit_warnings
    _write_csv(args.output_csv, summary_rows)
    pilot_rows = _pilot_rows(rows, per_feature=args.pilot_per_feature)
    _write_pilot_csv(args.pilot_output, pilot_rows)
    blind_rows, blind_map_rows = _blind_pilot_rows(pilot_rows, seed=args.blind_seed)
    _write_blind_pilot_csv(args.blind_pilot_output, blind_rows)
    _write_blind_map_csv(args.blind_pilot_map_output, blind_map_rows)
    blind_full_rows, blind_full_map_rows = _blind_pilot_rows(rows, seed=args.blind_seed)
    _write_blind_pilot_csv(args.blind_full_output, blind_full_rows)
    _write_blind_map_csv(args.blind_full_map_output, blind_full_map_rows)
    _write_markdown(
        args.output_md,
        summary_rows,
        warnings=warnings,
        input_path=args.input,
        output_csv=args.output_csv,
        pilot_output=args.pilot_output,
        blind_pilot_output=args.blind_pilot_output,
        blind_pilot_map_output=args.blind_pilot_map_output,
        blind_full_output=args.blind_full_output,
        blind_full_map_output=args.blind_full_map_output,
        pilot_per_feature=args.pilot_per_feature,
        blind_seed=args.blind_seed,
    )
    return summary_rows


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                warnings.append(f"{path}:{line_number}: missing fields {', '.join(missing)}")
            rows.append(row)
    return rows, warnings


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    expected_features: int,
    expected_rows_per_feature: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids = Counter()
    warnings: list[str] = []
    for row in rows:
        feature = str(row.get("feature", ""))
        by_feature[feature].append(row)
        ids[str(row.get("annotation_id", ""))] += 1

    duplicate_ids = sorted(annotation_id for annotation_id, count in ids.items() if count > 1)
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:10])
        warnings.append(f"duplicate annotation_id values: {preview}")
    if len(by_feature) != expected_features:
        warnings.append(f"feature count {len(by_feature)} != expected {expected_features}")

    summary_rows: list[dict[str, Any]] = []
    for feature, feature_rows in sorted(by_feature.items()):
        missing_required = sum(
            1
            for row in feature_rows
            if any(field not in row for field in REQUIRED_FIELDS)
        )
        blank_perception = sum(
            not str(row.get("human_perceived_slop", "")).strip() for row in feature_rows
        )
        blank_taxonomy = sum(
            not str(row.get("shaib_taxonomy_label", "")).strip() for row in feature_rows
        )
        labeled = len(feature_rows) - blank_perception
        source_count = len({str(row.get("source", "")) for row in feature_rows})
        record_count = len({str(row.get("record_id", "")) for row in feature_rows})
        candidate_label = str(feature_rows[0].get("candidate_label", "")) if feature_rows else ""
        cluster = str(feature_rows[0].get("cluster", "")) if feature_rows else ""
        row_count_status = (
            "ok" if len(feature_rows) == expected_rows_per_feature else "unexpected_count"
        )
        schema_status = "ok" if missing_required == 0 else "missing_fields"
        label_status = "blank" if labeled == 0 else "partially_labeled"
        summary_rows.append(
            {
                "feature": feature,
                "candidate_label": candidate_label,
                "cluster": cluster,
                "rows": len(feature_rows),
                "expected_rows": expected_rows_per_feature,
                "row_count_status": row_count_status,
                "schema_status": schema_status,
                "labeled": labeled,
                "blank_human_perceived_slop": blank_perception,
                "blank_shaib_taxonomy_label": blank_taxonomy,
                "unique_sources": source_count,
                "unique_record_ids": record_count,
                "label_status": label_status,
            }
        )
    return summary_rows, warnings


def _pilot_rows(rows: list[dict[str, Any]], *, per_feature: int) -> list[dict[str, Any]]:
    if per_feature <= 0:
        return []
    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_feature[str(row.get("feature", ""))].append(row)
    pilot: list[dict[str, Any]] = []
    for feature in sorted(by_feature):
        feature_rows = sorted(
            by_feature[feature],
            key=lambda row: str(row.get("annotation_id", "")),
        )
        for row in feature_rows[:per_feature]:
            pilot.append({field: row.get(field, "") for field in PILOT_FIELDS})
    return pilot


def _blind_pilot_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    randomized = list(rows)
    random.Random(seed).shuffle(randomized)
    blind_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    for index, row in enumerate(randomized, start=1):
        blind_id = f"blind_{index:04d}"
        blind_rows.append(
            {
                "blind_id": blind_id,
                "matched_text": row.get("matched_text", ""),
                "snippet": row.get("snippet", ""),
                "human_perceived_slop": row.get("human_perceived_slop", ""),
                "shaib_taxonomy_label": row.get("shaib_taxonomy_label", ""),
                "notes": row.get("notes", ""),
            }
        )
        map_rows.append(
            {
                "blind_id": blind_id,
                "annotation_id": row.get("annotation_id", ""),
                "feature": row.get("feature", ""),
                "candidate_label": row.get("candidate_label", ""),
                "cluster": row.get("cluster", ""),
                "source": row.get("source", ""),
                "record_id": row.get("record_id", ""),
                "matched_text": row.get("matched_text", ""),
                "snippet": row.get("snippet", ""),
            }
        )
    return blind_rows, map_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "feature",
        "candidate_label",
        "cluster",
        "rows",
        "expected_rows",
        "row_count_status",
        "schema_status",
        "labeled",
        "blank_human_perceived_slop",
        "blank_shaib_taxonomy_label",
        "unique_sources",
        "unique_record_ids",
        "label_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_pilot_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PILOT_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _write_blind_pilot_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BLIND_PILOT_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _write_blind_map_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BLIND_MAP_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    warnings: list[str],
    input_path: Path,
    output_csv: Path,
    pilot_output: Path,
    blind_pilot_output: Path,
    blind_pilot_map_output: Path,
    blind_full_output: Path,
    blind_full_map_output: Path,
    pilot_per_feature: int,
    blind_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = sum(int(row["rows"]) for row in rows)
    schema_ok = all(row["schema_status"] == "ok" for row in rows)
    balanced = all(row["row_count_status"] == "ok" for row in rows)
    all_blank = all(int(row["labeled"]) == 0 for row in rows)
    readiness = "ready_for_labeling" if schema_ok and balanced and all_blank and not warnings else "review"
    lines = [
        "# Phase 4 Human Annotation Readiness",
        "",
        f"Input package: `{input_path}`",
        f"Machine-readable audit: `{output_csv}`",
        f"Pilot labeling sheet: `{pilot_output}`",
        f"Blind pilot labeling sheet: `{blind_pilot_output}`",
        f"Blind pilot ID map: `{blind_pilot_map_output}`",
        f"Blind full-package labeling sheet: `{blind_full_output}`",
        f"Blind full-package ID map: `{blind_full_map_output}`",
        f"Safe handoff bundle: `{DEFAULT_HANDOFF_DIR}`",
        "",
        f"Readiness status: `{readiness}`.",
        f"Package rows: `{total_rows}` across `{len(rows)}` candidate features.",
        f"Pilot sheet size: `{pilot_per_feature}` rows per feature.",
        f"Blind pilot randomization seed: `{blind_seed}`.",
        "",
        "| Feature | Rows | Schema | Labeled | Sources | Records | Candidate label |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {feature} | {rows}/{expected_rows} | {schema_status} | {labeled} | "
            "{unique_sources} | {unique_record_ids} | {candidate_label} |".format(**row)
        )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.extend(
            [
                "",
                "No package-balance or schema warnings were found. The package remains",
                "unlabeled and should still be treated as detector-only evidence until",
                "human labels are collected and summarized. Use the blind pilot sheet",
                "for annotator-facing dry runs, or the blind full-package sheet for",
                "the recommended two-annotator labeling plan; keep ID maps separate",
                "until after adjudication. The handoff bundle materializes this",
                "separation: share only its `annotator/` directory with annotators",
                "and keep `coordinator/private_maps/` private.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_phase4_human_package(args)
    print(f"Wrote {len(rows)} Phase 4 annotation-readiness rows to {args.output_csv}")
    print(f"Wrote Phase 4 annotation-readiness summary to {args.output_md}")
    print(f"Wrote pilot labeling sheet to {args.pilot_output}")
    print(f"Wrote blind pilot labeling sheet to {args.blind_pilot_output}")
    print(f"Wrote blind pilot ID map to {args.blind_pilot_map_output}")
    print(f"Wrote blind full-package labeling sheet to {args.blind_full_output}")
    print(f"Wrote blind full-package ID map to {args.blind_full_map_output}")


if __name__ == "__main__":
    main()
