from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


VALID_LABELS = {"exact", "partial", "false_positive", "ambiguous"}
CORE_FEATURES = {
    "contrastive_negation",
    "slop_lexicon",
    "stock_openers",
    "stock_closers",
    "rule_of_three_approx",
}
DERIVED_FEATURES = {
    "stock_openers_closers",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score labeled Tier-1 matcher precision samples.")
    parser.add_argument("--input", type=Path, required=True, help="Labeled hit CSV from slop-sample-hits.")
    parser.add_argument("--output", type=Path, required=True, help="Per-feature precision report CSV.")
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument("--max-ambiguous-rate", type=float, default=0.10)
    parser.add_argument("--min-labeled", type=int, default=50)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="precision-validation")
    parser.add_argument("--wandb-job-type", default="score-labels")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _read_rows(path: Path, *, allow_incomplete: bool) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    invalid_rows: list[str] = []
    for index, row in enumerate(rows, start=2):
        label = (row.get("label") or "").strip()
        if not label and allow_incomplete:
            continue
        if label not in VALID_LABELS:
            invalid_rows.append(f"line {index}: {label!r}")
    if invalid_rows:
        preview = "; ".join(invalid_rows[:10])
        raise ValueError(f"invalid or missing labels ({preview})")
    return rows


def _score_rows(
    rows: list[dict[str, str]],
    *,
    target_precision: float,
    max_ambiguous_rate: float,
    min_labeled: int,
    allow_incomplete: bool,
) -> list[dict[str, Any]]:
    feature_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        label = (row.get("label") or "").strip()
        if not label and allow_incomplete:
            continue
        feature = row.get("feature") or "unknown"
        feature_counts[feature][label] += 1

    scored: list[dict[str, Any]] = []
    for feature, counts in sorted(feature_counts.items()):
        total = sum(counts.values())
        exact = counts["exact"]
        ambiguous = counts["ambiguous"]
        precision = exact / total if total else 0.0
        ambiguous_rate = ambiguous / total if total else 0.0
        is_core = feature in CORE_FEATURES
        precision_pass = precision >= target_precision
        ambiguity_pass = ambiguous_rate <= max_ambiguous_rate
        sample_size_pass = total >= min_labeled or not is_core
        is_derived = feature in DERIVED_FEATURES
        if is_derived:
            status = "derived"
        else:
            status = "pass" if precision_pass and ambiguity_pass and sample_size_pass else "demote"
        scored.append(
            {
                "feature": feature,
                "is_core": is_core,
                "is_derived": is_derived,
                "status": status,
                "labeled": total,
                "exact": exact,
                "partial": counts["partial"],
                "false_positive": counts["false_positive"],
                "ambiguous": ambiguous,
                "precision": precision,
                "ambiguous_rate": ambiguous_rate,
                "target_precision": target_precision,
                "max_ambiguous_rate": max_ambiguous_rate,
                "min_labeled": min_labeled,
                "precision_pass": precision_pass,
                "ambiguity_pass": ambiguity_pass,
                "sample_size_pass": sample_size_pass,
            }
        )
    return scored


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature",
        "is_core",
        "is_derived",
        "status",
        "labeled",
        "exact",
        "partial",
        "false_positive",
        "ambiguous",
        "precision",
        "ambiguous_rate",
        "target_precision",
        "max_ambiguous_rate",
        "min_labeled",
        "precision_pass",
        "ambiguity_pass",
        "sample_size_pass",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_score_labels(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _read_rows(args.input, allow_incomplete=args.allow_incomplete)
    scored = _score_rows(
        rows,
        target_precision=args.target_precision,
        max_ambiguous_rate=args.max_ambiguous_rate,
        min_labeled=args.min_labeled,
        allow_incomplete=args.allow_incomplete,
    )
    _write_report(args.output, scored)

    core_rows = [row for row in scored if row["is_core"]]
    demoted_rows = [row for row in scored if row["status"] != "pass"]
    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage1", "precision-validation", "score-labels", *args.wandb_tag],
        config={
            "input": str(args.input),
            "output": str(args.output),
            "target_precision": args.target_precision,
            "max_ambiguous_rate": args.max_ambiguous_rate,
            "min_labeled": args.min_labeled,
            "allow_incomplete": args.allow_incomplete,
        },
    )
    try:
        run.log(
            {
                "matcher/features_scored": len(scored),
                "matcher/features_demoted": len(demoted_rows),
                "matcher/core_features_scored": len(core_rows),
                "matcher/core_features_demoted": sum(row["status"] != "pass" for row in core_rows),
            }
        )
        log_summary_table(run, "precision_report", scored)
    finally:
        run.finish()
    return scored


def main() -> None:
    args = build_parser().parse_args()
    rows = run_score_labels(args)
    print(f"Wrote {len(rows)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
