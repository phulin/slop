from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.score_labels import CORE_FEATURES, DERIVED_FEATURES, VALID_LABELS


EXPLORATORY_FEATURES = {
    "eqbench_slop_score",
    "slop_lexicon_v2_candidate",
}
DEFAULT_FEATURES = tuple(
    sorted(CORE_FEATURES | EXPLORATORY_FEATURES | DERIVED_FEATURES)
)


@dataclass
class FeatureValidation:
    feature: str
    queued: int = 0
    labeled: int = 0
    exact: int = 0
    partial: int = 0
    false_positive: int = 0
    ambiguous: int = 0
    label_files: set[str] = field(default_factory=set)
    queue_files: set[str] = field(default_factory=set)

    @property
    def precision(self) -> float | None:
        return self.exact / self.labeled if self.labeled else None

    @property
    def ambiguous_rate(self) -> float | None:
        return self.ambiguous / self.labeled if self.labeled else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize current precision-validation coverage and gate status."
    )
    parser.add_argument("--label-input", type=Path, action="append", default=[])
    parser.add_argument("--queue-input", type=Path, action="append", default=[])
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--max-ambiguous-rate", type=float, default=0.10)
    parser.add_argument("--min-labeled", type=int, default=50)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def run_summarize_precision_validation(args: argparse.Namespace) -> list[dict[str, Any]]:
    features = set(args.feature or DEFAULT_FEATURES)
    status = {feature: FeatureValidation(feature=feature) for feature in features}
    queue_file_counts: dict[str, int] = {}

    for path in args.queue_input:
        matched_rows = 0
        for row in _read_csv(path):
            feature = row.get("feature", "")
            if feature not in features:
                continue
            item = status.setdefault(feature, FeatureValidation(feature=feature))
            item.queued += 1
            item.queue_files.add(str(path))
            matched_rows += 1
        queue_file_counts[str(path)] = matched_rows

    for path in args.label_input:
        for row in _read_csv(path):
            feature = row.get("feature", "")
            if feature not in features:
                continue
            item = status.setdefault(feature, FeatureValidation(feature=feature))
            label = (row.get("label") or "").strip()
            if not label:
                continue
            if label not in VALID_LABELS:
                raise ValueError(f"{path}: unknown label {label!r} for feature {feature}")
            item.labeled += 1
            setattr(item, label, getattr(item, label) + 1)
            item.label_files.add(str(path))

    rows = [
        _status_row(
            item,
            target_precision=args.target_precision,
            max_ambiguous_rate=args.max_ambiguous_rate,
            min_labeled=args.min_labeled,
        )
        for item in sorted(status.values(), key=lambda item: item.feature)
    ]
    _write_csv(args.output_csv, rows)
    _write_markdown(
        args.output_md,
        rows,
        label_inputs=args.label_input,
        queue_inputs=args.queue_input,
        queue_file_counts=queue_file_counts,
        target_precision=args.target_precision,
        max_ambiguous_rate=args.max_ambiguous_rate,
        min_labeled=args.min_labeled,
    )
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _status_row(
    item: FeatureValidation,
    *,
    target_precision: float,
    max_ambiguous_rate: float,
    min_labeled: int,
) -> dict[str, Any]:
    is_core = item.feature in CORE_FEATURES
    is_exploratory = item.feature in EXPLORATORY_FEATURES
    is_derived = item.feature in DERIVED_FEATURES
    precision = item.precision
    ambiguous_rate = item.ambiguous_rate
    precision_pass = precision is not None and precision >= target_precision
    ambiguity_pass = ambiguous_rate is not None and ambiguous_rate <= max_ambiguous_rate
    sample_size_pass = item.labeled >= min_labeled
    if is_derived:
        gate_status = "derived"
    elif is_core and precision_pass and ambiguity_pass and sample_size_pass:
        gate_status = "pass"
    elif item.labeled == 0:
        gate_status = "unlabeled"
    elif item.labeled < min_labeled:
        gate_status = "incomplete"
    elif not precision_pass or not ambiguity_pass:
        gate_status = "demote"
    elif is_exploratory:
        gate_status = "exploratory"
    else:
        gate_status = "review"
    return {
        "feature": item.feature,
        "is_core": is_core,
        "is_exploratory": is_exploratory,
        "is_derived": is_derived,
        "gate_status": gate_status,
        "queued": item.queued,
        "labeled": item.labeled,
        "exact": item.exact,
        "partial": item.partial,
        "false_positive": item.false_positive,
        "ambiguous": item.ambiguous,
        "precision": "" if precision is None else f"{precision:.6f}",
        "ambiguous_rate": "" if ambiguous_rate is None else f"{ambiguous_rate:.6f}",
        "target_precision": f"{target_precision:.2f}",
        "max_ambiguous_rate": f"{max_ambiguous_rate:.2f}",
        "min_labeled": min_labeled,
        "precision_pass": precision_pass,
        "ambiguity_pass": ambiguity_pass,
        "sample_size_pass": sample_size_pass,
        "label_files": "; ".join(sorted(item.label_files or [])),
        "queue_files": "; ".join(sorted(item.queue_files or [])),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    label_inputs: list[Path],
    queue_inputs: list[Path],
    queue_file_counts: dict[str, int],
    target_precision: float,
    max_ambiguous_rate: float,
    min_labeled: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Precision Validation Status",
        "",
        "Date: 2026-06-18",
        "",
        "This report summarizes the current manual precision-labeling state for",
        "the revised Phase 1 feature surface. It is a status artifact, not a",
        "claim that the precision gate is complete.",
        "",
        "## Inputs",
        "",
    ]
    for label_input in label_inputs:
        lines.append(f"- Label CSV: `{label_input}`")
    for queue_input in queue_inputs:
        lines.append(f"- Hit queue CSV: `{queue_input}`")
    empty_queue_inputs = [
        str(queue_input)
        for queue_input in queue_inputs
        if queue_file_counts.get(str(queue_input), 0) == 0
    ]
    if empty_queue_inputs:
        lines.extend(["", "Queue files with zero retained feature hits:"])
        lines.extend(f"- `{queue_input}`" for queue_input in empty_queue_inputs)
    lines.extend(
        [
            "",
            "## Gate Settings",
            "",
            f"- Target precision: `{target_precision:.2f}`",
            f"- Maximum ambiguous rate: `{max_ambiguous_rate:.2f}`",
            f"- Minimum labeled rows for core features: `{min_labeled}`",
            "",
            "## Feature Status",
            "",
            "| Feature | Core | Derived | Status | Queued | Labeled | Exact | Partial | FP | Ambig | Precision | Ambig rate |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {feature} | {is_core} | {is_derived} | {gate_status} | {queued} | {labeled} | {exact} | "
            "{partial} | {false_positive} | {ambiguous} | {precision} | {ambiguous_rate} |".format(
                **row
            )
        )
    core_rows = [row for row in rows if row["is_core"]]
    passed_core = [row for row in core_rows if row["gate_status"] == "pass"]
    incomplete_core = [row for row in core_rows if row["gate_status"] != "pass"]
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- Core features passing the current gate: `{len(passed_core)}/{len(core_rows)}`.",
            f"- Core features not passing publication-grade Phase 1 regex claims: `{len(incomplete_core)}`.",
            "- Rows marked `incomplete` can still be useful smell-test evidence, but",
            "  they should not be cited as passing the 200-hit precision plan.",
            "- Rows marked `unlabeled` need either a direct label queue or an explicit",
            "  decision to infer precision from component features.",
            "- Rows marked `derived` are pooled convenience views rather than",
            "  independent precision-validation targets.",
            "- A supplied queue file with zero retained hits means the bounded sampling",
            "  attempt did not find labelable examples under the current scan limit.",
            "",
            "## Next Actions",
            "",
            "1. Continue labeling core rows until each retained core feature has at",
            "   least 50 labels for the interim gate and, ideally, 200 labels for the",
            "   original validation plan.",
            "2. Treat `stock_openers_closers` as a derived convenience metric unless",
            "   a broader scan is intentionally budgeted; the bounded direct queue",
            "   attempt supplied here produced zero retained hits.",
            "3. Score the completed label files with `slop-score-labels` before using",
            "   corpus-rate tables for publication-grade claims.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_summarize_precision_validation(args)
    print(f"Wrote {len(rows)} precision-validation status rows")


if __name__ == "__main__":
    main()
