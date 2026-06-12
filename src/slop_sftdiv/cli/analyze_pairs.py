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
    "preference_type",
    "chosen_model",
    "rejected_model",
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
    "logit_feature_coef",
    "logit_feature_se",
    "logit_feature_z",
    "logit_feature_odds_ratio",
    "logit_length_coef",
    "logit_iterations",
    "logit_converged",
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
    tail_successes = min(successes, trials - successes)
    log_terms = [
        math.lgamma(trials + 1)
        - math.lgamma(k + 1)
        - math.lgamma(trials - k + 1)
        - trials * math.log(2.0)
        for k in range(tail_successes + 1)
    ]
    max_log = max(log_terms)
    tail_probability = math.exp(max_log) * math.fsum(
        math.exp(term - max_log) for term in log_terms
    )
    return min(1.0, 2.0 * tail_probability)


GroupKey = tuple[str, str, str, str, str, str]


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


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    n = 3
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for item in range(column, n + 1):
            augmented[column][item] /= pivot_value
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            for item in range(column, n + 1):
                augmented[row][item] -= factor * augmented[column][item]
    return [augmented[row][n] for row in range(n)]


def _invert_3x3(matrix: list[list[float]]) -> list[list[float]] | None:
    columns = []
    for index in range(3):
        unit = [0.0, 0.0, 0.0]
        unit[index] = 1.0
        solved = _solve_3x3(matrix, unit)
        if solved is None:
            return None
        columns.append(solved)
    return [[columns[column][row] for column in range(3)] for row in range(3)]


def _standardize(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = _mean(values)
    variance = _mean([(value - mean) ** 2 for value in values])
    scale = math.sqrt(variance)
    if scale <= 1e-12:
        return [0.0 for _ in values]
    return [(value - mean) / scale for value in values]


def _length_adjusted_logit(
    *,
    chosen_rates: list[float],
    rejected_rates: list[float],
    chosen_tokens: list[float],
    rejected_tokens: list[float],
    ridge: float = 0.1,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    if not chosen_rates:
        return _empty_logit_result()
    observations = [
        (1.0, feature_rate, tokens)
        for feature_rate, tokens in zip(chosen_rates, chosen_tokens)
    ] + [
        (0.0, feature_rate, tokens)
        for feature_rate, tokens in zip(rejected_rates, rejected_tokens)
    ]
    xs_feature = _standardize([feature_rate for _, feature_rate, _ in observations])
    xs_length = _standardize([tokens for _, _, tokens in observations])
    beta = [0.0, 0.0, 0.0]
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        gradient = [0.0, -ridge * beta[1], -ridge * beta[2]]
        hessian = [
            [0.0, 0.0, 0.0],
            [0.0, -ridge, 0.0],
            [0.0, 0.0, -ridge],
        ]
        for (y, _, _), feature_x, length_x in zip(observations, xs_feature, xs_length):
            x = [1.0, feature_x, length_x]
            p = _sigmoid(sum(coef * item for coef, item in zip(beta, x)))
            residual = y - p
            weight = p * (1.0 - p)
            for i in range(3):
                gradient[i] += residual * x[i]
                for j in range(3):
                    hessian[i][j] -= weight * x[i] * x[j]
        neg_hessian = [[-value for value in row] for row in hessian]
        step = _solve_3x3(neg_hessian, gradient)
        if step is None:
            break
        beta = [coef + delta for coef, delta in zip(beta, step)]
        iterations = iteration
        if max(abs(delta) for delta in step) < tolerance:
            converged = True
            break

    neg_hessian = [[ridge if i == j and i > 0 else 0.0 for j in range(3)] for i in range(3)]
    for feature_x, length_x in zip(xs_feature, xs_length):
        x = [1.0, feature_x, length_x]
        p = _sigmoid(sum(coef * item for coef, item in zip(beta, x)))
        weight = p * (1.0 - p)
        for i in range(3):
            for j in range(3):
                neg_hessian[i][j] += weight * x[i] * x[j]
    covariance = _invert_3x3(neg_hessian)
    feature_se = math.sqrt(covariance[1][1]) if covariance is not None else 0.0
    feature_z = beta[1] / feature_se if feature_se > 0 else 0.0
    odds_ratio = math.exp(max(min(beta[1], 50.0), -50.0))
    return {
        "logit_feature_coef": beta[1],
        "logit_feature_se": feature_se,
        "logit_feature_z": feature_z,
        "logit_feature_odds_ratio": odds_ratio,
        "logit_length_coef": beta[2],
        "logit_iterations": iterations,
        "logit_converged": converged,
    }


def _empty_logit_result() -> dict[str, Any]:
    return {
        "logit_feature_coef": 0.0,
        "logit_feature_se": 0.0,
        "logit_feature_z": 0.0,
        "logit_feature_odds_ratio": 1.0,
        "logit_length_coef": 0.0,
        "logit_iterations": 0,
        "logit_converged": False,
    }


def analyze_rows(rows: list[dict[str, Any]], *, alpha: float) -> list[dict[str, Any]]:
    by_group: dict[GroupKey, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[
            (
                str(row.get("source", "")),
                str(row.get("subset", "")),
                str(row.get("preference_type", "")),
                str(row.get("chosen_model", "")),
                str(row.get("rejected_model", "")),
                str(row["feature"]),
            )
        ].append(row)

    reports: list[dict[str, Any]] = []
    p_values: dict[GroupKey, float] = {}
    for (
        source,
        subset,
        preference_type,
        chosen_model,
        rejected_model,
        feature,
    ), feature_rows in sorted(by_group.items()):
        deltas = [_float(row, "chosen_minus_rejected") for row in feature_rows]
        chosen_rates = [_float(row, "chosen_rate_per_1k_tokens") for row in feature_rows]
        rejected_rates = [_float(row, "rejected_rate_per_1k_tokens") for row in feature_rows]
        chosen_tokens = [_float(row, "chosen_tokens") for row in feature_rows]
        rejected_tokens = [_float(row, "rejected_tokens") for row in feature_rows]
        token_deltas = [chosen - rejected for chosen, rejected in zip(chosen_tokens, rejected_tokens)]
        logit = _length_adjusted_logit(
            chosen_rates=chosen_rates,
            rejected_rates=rejected_rates,
            chosen_tokens=chosen_tokens,
            rejected_tokens=rejected_tokens,
        )
        positive = sum(delta > 0 for delta in deltas)
        negative = sum(delta < 0 for delta in deltas)
        zero = sum(delta == 0 for delta in deltas)
        sign_trials = positive + negative
        sign_successes = max(positive, negative)
        p_value = _binomial_two_sided_p(sign_successes, sign_trials)
        p_values[(source, subset, preference_type, chosen_model, rejected_model, feature)] = p_value
        direction = "chosen_gt_rejected" if positive > negative else "rejected_gt_chosen"
        if positive == negative:
            direction = "tie"
        reports.append(
            {
                "feature": feature,
                "source": source,
                "subset": subset,
                "preference_type": preference_type,
                "chosen_model": chosen_model,
                "rejected_model": rejected_model,
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
                "mean_token_delta": _mean(token_deltas),
                "sign_test_p": p_value,
                "bh_q": 1.0,
                "direction": direction,
                "fdr_significant": False,
                **logit,
            }
        )

    q_values = _benjamini_hochberg(p_values)
    for report in reports:
        q_value = q_values[
            (
                str(report["source"]),
                str(report["subset"]),
                str(report["preference_type"]),
                str(report["chosen_model"]),
                str(report["rejected_model"]),
                str(report["feature"]),
            )
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
