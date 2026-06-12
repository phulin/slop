from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


REPORT_COLUMNS = [
    "source",
    "subset",
    "feature",
    "pairs",
    "positive_pairs",
    "negative_pairs",
    "zero_pairs",
    "mean_delta",
    "median_delta",
    "mean_chosen_rate",
    "mean_rejected_rate",
    "mean_chosen_tokens",
    "mean_rejected_tokens",
    "mean_token_delta",
    "sign_test_p",
    "bh_q",
    "direction",
    "fdr_significant",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze chosen/rejected pair deltas.")
    parser.add_argument("--input", type=Path, required=True, help="Pair delta CSV from slop-census.")
    parser.add_argument("--output", type=Path, required=True, help="Feature-level analysis CSV.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="preference-analysis")
    parser.add_argument("--wandb-job-type", default="result-a")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _read_pair_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "feature",
        "pair_id",
        "chosen_rate_per_1k_tokens",
        "rejected_rate_per_1k_tokens",
        "chosen_minus_rejected",
        "chosen_tokens",
        "rejected_tokens",
    }
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"pair delta CSV missing required columns: {sorted(missing)}")
    return rows


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def _binomial_two_sided_p(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    lower_tail = sum(math.comb(trials, k) for k in range(0, successes + 1))
    upper_tail = sum(math.comb(trials, k) for k in range(successes, trials + 1))
    total = 2**trials
    return min(1.0, 2.0 * min(lower_tail, upper_tail) / total)


GroupKey = tuple[str, str, str]


def _benjamini_hochberg(p_values: dict[GroupKey, float]) -> dict[GroupKey, float]:
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1], reverse=True)
    m = len(ordered)
    q_values: dict[GroupKey, float] = {}
    running_min = 1.0
    for rank_from_end, (feature, p_value) in enumerate(ordered, start=1):
        rank = m - rank_from_end + 1
        q_value = min(running_min, p_value * m / rank)
        running_min = q_value
        q_values[feature] = min(q_value, 1.0)
    return q_values


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze_rows(rows: list[dict[str, Any]], *, alpha: float) -> list[dict[str, Any]]:
    by_group: dict[GroupKey, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(str(row.get("source", "")), str(row.get("subset", "")), str(row["feature"]))].append(
            row
        )

    reports: list[dict[str, Any]] = []
    p_values: dict[GroupKey, float] = {}
    for (source, subset, feature), feature_rows in sorted(by_group.items()):
        deltas = [_float(row, "chosen_minus_rejected") for row in feature_rows]
        chosen_rates = [_float(row, "chosen_rate_per_1k_tokens") for row in feature_rows]
        rejected_rates = [_float(row, "rejected_rate_per_1k_tokens") for row in feature_rows]
        chosen_tokens = [_float(row, "chosen_tokens") for row in feature_rows]
        rejected_tokens = [_float(row, "rejected_tokens") for row in feature_rows]
        positive = sum(delta > 0 for delta in deltas)
        negative = sum(delta < 0 for delta in deltas)
        zero = sum(delta == 0 for delta in deltas)
        sign_trials = positive + negative
        sign_successes = max(positive, negative)
        p_value = _binomial_two_sided_p(sign_successes, sign_trials)
        p_values[(source, subset, feature)] = p_value
        direction = "chosen_gt_rejected" if positive > negative else "rejected_gt_chosen"
        if positive == negative:
            direction = "tie"
        reports.append(
            {
                "feature": feature,
                "source": source,
                "subset": subset,
                "pairs": len(feature_rows),
                "positive_pairs": positive,
                "negative_pairs": negative,
                "zero_pairs": zero,
                "mean_delta": _mean(deltas),
                "median_delta": median(deltas) if deltas else 0.0,
                "mean_chosen_rate": _mean(chosen_rates),
                "mean_rejected_rate": _mean(rejected_rates),
                "mean_chosen_tokens": _mean(chosen_tokens),
                "mean_rejected_tokens": _mean(rejected_tokens),
                "mean_token_delta": _mean(
                    [chosen - rejected for chosen, rejected in zip(chosen_tokens, rejected_tokens)]
                ),
                "sign_test_p": p_value,
                "bh_q": 1.0,
                "direction": direction,
                "fdr_significant": False,
            }
        )

    q_values = _benjamini_hochberg(p_values)
    for report in reports:
        q_value = q_values[
            (str(report["source"]), str(report["subset"]), str(report["feature"]))
        ]
        report["bh_q"] = q_value
        report["fdr_significant"] = q_value <= alpha
    return reports


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_analyze_pairs(args: argparse.Namespace) -> list[dict[str, Any]]:
    pair_rows = _read_pair_rows(args.input)
    report_rows = analyze_rows(pair_rows, alpha=args.alpha)
    _write_report(args.output, report_rows)

    significant = [row for row in report_rows if row["fdr_significant"]]
    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage1", "phase1", "preference-analysis", "result-a", *args.wandb_tag],
        config={"input": str(args.input), "output": str(args.output), "alpha": args.alpha},
    )
    try:
        run.log(
            {
                "preference/features": len(report_rows),
                "preference/significant_features": len(significant),
                "preference/pair_delta_rows": len(pair_rows),
            }
        )
        log_summary_table(run, "preference_feature_analysis", report_rows)
    finally:
        run.finish()
    return report_rows


def main() -> None:
    args = build_parser().parse_args()
    rows = run_analyze_pairs(args)
    print(f"Wrote {len(rows)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
