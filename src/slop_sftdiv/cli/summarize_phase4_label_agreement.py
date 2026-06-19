from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.import_phase4_blind_labels import import_blind_rows


DEFAULT_BASE = Path("artifacts/phase4/modernbert_detector_combined_v2_clean")
DEFAULT_BLIND_MAP = DEFAULT_BASE / "phase4_human_perceptibility_blind_pilot_20_each_map.csv"
DEFAULT_OUTPUT = DEFAULT_BASE / "phase4_human_perceptibility_pilot_agreement.csv"
DEFAULT_DISAGREEMENTS = DEFAULT_BASE / "phase4_human_perceptibility_pilot_disagreements.csv"
DEFAULT_SUMMARY = Path("docs/experiments/phase4_human_perceptibility_pilot_agreement.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize two-annotator agreement for Phase 4 blinded labels."
    )
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--blind-map", type=Path, default=DEFAULT_BLIND_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--disagreements-output", type=Path, default=DEFAULT_DISAGREEMENTS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require both annotators to label every row.",
    )
    return parser


def run_summarize_phase4_label_agreement(args: argparse.Namespace) -> list[dict[str, Any]]:
    map_rows = _read_csv(args.blind_map)
    rows_a = import_blind_rows(_read_csv(args.annotator_a), map_rows, strict=args.strict)
    rows_b = import_blind_rows(_read_csv(args.annotator_b), map_rows, strict=args.strict)
    summary_rows, disagreement_rows = summarize_agreement(rows_a, rows_b)
    _write_csv(args.output, summary_rows, SUMMARY_COLUMNS)
    _write_csv(args.disagreements_output, disagreement_rows, DISAGREEMENT_COLUMNS)
    _write_markdown(args.summary_output, summary_rows, disagreement_rows)
    return summary_rows


SUMMARY_COLUMNS = [
    "feature",
    "candidate_label",
    "rows_compared",
    "both_labeled",
    "perception_agreement_count",
    "perception_agreement_rate",
    "perception_cohen_kappa",
    "yes_no_conflicts",
    "taxonomy_both_labeled",
    "taxonomy_agreement_count",
    "taxonomy_agreement_rate",
]
DISAGREEMENT_COLUMNS = [
    "blind_id",
    "annotation_id",
    "feature",
    "candidate_label",
    "matched_text",
    "human_perceived_slop_a",
    "human_perceived_slop_b",
    "shaib_taxonomy_label_a",
    "shaib_taxonomy_label_b",
    "notes_a",
    "notes_b",
    "adjudicated_human_perceived_slop",
    "adjudicated_shaib_taxonomy_label",
    "adjudication_notes",
]


def summarize_agreement(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id_a = _rows_by_blind_id(rows_a, "annotator A")
    by_id_b = _rows_by_blind_id(rows_b, "annotator B")
    missing_from_b = sorted(set(by_id_a) - set(by_id_b))
    missing_from_a = sorted(set(by_id_b) - set(by_id_a))
    if missing_from_a or missing_from_b:
        raise ValueError(
            "annotator files have different blind_id sets: "
            f"missing_from_a={missing_from_a[:5]}, missing_from_b={missing_from_b[:5]}"
        )

    pairs_by_feature: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    disagreement_rows: list[dict[str, Any]] = []
    for blind_id in sorted(by_id_a):
        row_a = by_id_a[blind_id]
        row_b = by_id_b[blind_id]
        feature = str(row_a.get("feature", ""))
        pairs_by_feature[feature].append((row_a, row_b))
        if _label(row_a, "human_perceived_slop") != _label(row_b, "human_perceived_slop"):
            disagreement_rows.append(_disagreement_row(row_a, row_b))

    summary_rows: list[dict[str, Any]] = []
    for feature, pairs in sorted(pairs_by_feature.items()):
        summary_rows.append(_summarize_feature(feature, pairs))
    summary_rows.append(_summarize_feature("ALL", [pair for pairs in pairs_by_feature.values() for pair in pairs]))
    return summary_rows, disagreement_rows


def _summarize_feature(
    feature: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    both_perception = [
        (row_a, row_b)
        for row_a, row_b in pairs
        if _label(row_a, "human_perceived_slop") and _label(row_b, "human_perceived_slop")
    ]
    perception_agree = sum(
        _label(row_a, "human_perceived_slop") == _label(row_b, "human_perceived_slop")
        for row_a, row_b in both_perception
    )
    yes_no_conflicts = sum(
        {_label(row_a, "human_perceived_slop"), _label(row_b, "human_perceived_slop")}
        == {"yes", "no"}
        for row_a, row_b in both_perception
    )
    both_taxonomy = [
        (row_a, row_b)
        for row_a, row_b in pairs
        if _label(row_a, "shaib_taxonomy_label") and _label(row_b, "shaib_taxonomy_label")
    ]
    taxonomy_agree = sum(
        _label(row_a, "shaib_taxonomy_label") == _label(row_b, "shaib_taxonomy_label")
        for row_a, row_b in both_taxonomy
    )
    first = pairs[0][0] if pairs else {}
    return {
        "feature": feature,
        "candidate_label": "All features" if feature == "ALL" else first.get("candidate_label", ""),
        "rows_compared": len(pairs),
        "both_labeled": len(both_perception),
        "perception_agreement_count": perception_agree,
        "perception_agreement_rate": _rate(perception_agree, len(both_perception)),
        "perception_cohen_kappa": _cohen_kappa(
            [_label(row_a, "human_perceived_slop") for row_a, _ in both_perception],
            [_label(row_b, "human_perceived_slop") for _, row_b in both_perception],
        ),
        "yes_no_conflicts": yes_no_conflicts,
        "taxonomy_both_labeled": len(both_taxonomy),
        "taxonomy_agreement_count": taxonomy_agree,
        "taxonomy_agreement_rate": _rate(taxonomy_agree, len(both_taxonomy)),
    }


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


def _disagreement_row(row_a: dict[str, Any], row_b: dict[str, Any]) -> dict[str, Any]:
    return {
        "blind_id": row_a.get("blind_id", ""),
        "annotation_id": row_a.get("annotation_id", ""),
        "feature": row_a.get("feature", ""),
        "candidate_label": row_a.get("candidate_label", ""),
        "matched_text": row_a.get("matched_text", ""),
        "human_perceived_slop_a": _label(row_a, "human_perceived_slop"),
        "human_perceived_slop_b": _label(row_b, "human_perceived_slop"),
        "shaib_taxonomy_label_a": _label(row_a, "shaib_taxonomy_label"),
        "shaib_taxonomy_label_b": _label(row_b, "shaib_taxonomy_label"),
        "notes_a": row_a.get("notes", ""),
        "notes_b": row_b.get("notes", ""),
        "adjudicated_human_perceived_slop": "",
        "adjudicated_shaib_taxonomy_label": "",
        "adjudication_notes": "",
    }


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> str:
    if not labels_a:
        return ""
    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / len(labels_a)
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    labels = sorted(set(counts_a) | set(counts_b))
    expected = sum((counts_a[label] / len(labels_a)) * (counts_b[label] / len(labels_b)) for label in labels)
    if expected == 1:
        return "1.000" if observed == 1 else "0.000"
    return f"{(observed - expected) / (1 - expected):.3f}"


def _label(row: dict[str, Any], field: str) -> str:
    return str(row.get(field, "")).strip().lower()


def _rate(numerator: int, denominator: int) -> str:
    return "" if denominator == 0 else f"{numerator / denominator:.3f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    disagreement_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 4 Human-Perceptibility Pilot Agreement",
        "",
        "This summary compares two independently labeled blinded pilot sheets.",
        "It supports adjudication and codebook refinement; it is not itself a",
        "human-perceptibility result unless real labels have been collected.",
        "",
        "| Feature | Both labeled | Agreement | Kappa | Yes/no conflicts | Taxonomy agreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {feature} | {both_labeled}/{rows_compared} | {agreement} | {kappa} | {conflicts} | {taxonomy} |".format(
                feature=row["feature"],
                both_labeled=row["both_labeled"],
                rows_compared=row["rows_compared"],
                agreement=row["perception_agreement_rate"] or "",
                kappa=row["perception_cohen_kappa"] or "",
                conflicts=row["yes_no_conflicts"],
                taxonomy=row["taxonomy_agreement_rate"] or "",
            )
        )
    lines.extend(
        [
            "",
            f"Perception disagreement rows: `{len(disagreement_rows)}`.",
            "Adjudicate disagreement rows before promoting detector candidates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_summarize_phase4_label_agreement(args)
    print(f"Wrote {len(rows)} Phase 4 agreement summary rows to {args.output}")
    print(f"Wrote disagreement rows to {args.disagreements_output}")
    print(f"Wrote agreement summary to {args.summary_output}")


if __name__ == "__main__":
    main()
