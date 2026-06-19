from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.classify_amplification_spectrum import STAGE_ORDER
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


OUTPUT_COLUMNS = [
    "comparison",
    "left_stage",
    "right_stage",
    "feature",
    "paired_units",
    "positive_pairs",
    "negative_pairs",
    "zero_pairs",
    "left_mean_per_1k_tokens",
    "right_mean_per_1k_tokens",
    "mean_delta_per_1k_tokens",
    "median_delta_per_1k_tokens",
    "sign_test_p",
    "bh_q",
    "fdr_significant",
    "direction",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired Phase 3 free-running stage-effect tests over shared prompts."
    )
    parser.add_argument(
        "--generation-cache",
        action="append",
        required=True,
        metavar="STAGE=PATH",
        help="Stage-tagged free-running generation JSONL from slop-free-running-emission.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="LEFT=RIGHT",
        help="Stage comparison to test. Defaults to adjacent ladder stages.",
    )
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase3-analysis")
    parser.add_argument("--wandb-job-type", default="free-run-stage-effects")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _parse_mapping(items: list[str], *, argument_name: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{argument_name} must use stage=path syntax: {item!r}")
        stage, path = item.split("=", 1)
        stage = stage.strip()
        path = path.strip()
        if not stage or not path:
            raise ValueError(f"{argument_name} must use non-empty stage=path syntax: {item!r}")
        if stage in mapping:
            raise ValueError(f"duplicate stage for {argument_name}: {stage}")
        mapping[stage] = Path(path)
    return mapping


def _parse_comparisons(items: list[str], stages: set[str]) -> list[tuple[str, str]]:
    if not items:
        adjacent = [
            (left, right)
            for left, right in zip(STAGE_ORDER, STAGE_ORDER[1:], strict=False)
            if left in stages and right in stages
        ]
        if adjacent:
            return adjacent
        ordered = sorted(stages)
        return list(zip(ordered, ordered[1:], strict=False))
    comparisons = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--comparison must use LEFT=RIGHT syntax: {item!r}")
        left, right = (part.strip() for part in item.split("=", 1))
        if not left or not right:
            raise ValueError(f"--comparison must use non-empty LEFT=RIGHT syntax: {item!r}")
        if left not in stages or right not in stages:
            raise ValueError(f"comparison {item!r} references a missing generation-cache stage")
        comparisons.append((left, right))
    return comparisons


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


def _benjamini_hochberg(p_values: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1], reverse=True)
    m = len(ordered)
    q_values: dict[tuple[str, str], float] = {}
    running_min = 1.0
    for rank_from_end, (key, p_value) in enumerate(ordered, start=1):
        rank = m - rank_from_end + 1
        q_value = min(running_min, p_value * m / rank)
        running_min = q_value
        q_values[key] = min(q_value, 1.0)
    return q_values


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def _record_key(row: dict[str, Any]) -> str:
    for key in ("record_id", "prompt_id", "phase2_prompt_id", "source_row_index", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _record_key(row),
        str(row.get("completion_index", "")),
        str(row.get("temperature", "")),
        str(row.get("top_p", "")),
    )


def _load_stage_cache(path: Path) -> dict[tuple[str, str, str, str], dict[str, float]]:
    values: dict[tuple[str, str, str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            features = json.loads(row.get("features_json") or "{}")
            generated_tokens = max(1.0, float(row.get("generated_tokens") or 0.0))
            values[_pair_key(row)] = {
                str(feature): 1000.0 * float(count) / generated_tokens
                for feature, count in features.items()
            }
    return values


def _available_features(caches: dict[str, dict[tuple[str, str, str, str], dict[str, float]]]) -> list[str]:
    features = set()
    for cache in caches.values():
        for row in cache.values():
            features.update(row)
    return sorted(features)


def analyze_free_run_effects(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.alpha <= 0.0 or args.alpha > 1.0:
        raise ValueError("--alpha must be in (0, 1]")
    cache_paths = _parse_mapping(args.generation_cache, argument_name="--generation-cache")
    caches = {stage: _load_stage_cache(path) for stage, path in cache_paths.items()}
    comparisons = _parse_comparisons(args.comparison, set(caches))
    features = list(dict.fromkeys(args.feature or _available_features(caches)))
    rows: list[dict[str, Any]] = []
    p_values: dict[tuple[str, str], float] = {}
    for left_stage, right_stage in comparisons:
        left_cache = caches[left_stage]
        right_cache = caches[right_stage]
        shared_keys = sorted(set(left_cache) & set(right_cache))
        comparison = f"{left_stage}_vs_{right_stage}"
        for feature in features:
            left_values = [left_cache[key].get(feature, 0.0) for key in shared_keys]
            right_values = [right_cache[key].get(feature, 0.0) for key in shared_keys]
            deltas = [right - left for left, right in zip(left_values, right_values, strict=True)]
            positive = sum(1 for delta in deltas if delta > 0.0)
            negative = sum(1 for delta in deltas if delta < 0.0)
            zero = sum(1 for delta in deltas if delta == 0.0)
            trials = positive + negative
            p_value = _binomial_two_sided_p(positive, trials)
            key = (comparison, feature)
            p_values[key] = p_value
            rows.append(
                {
                    "comparison": comparison,
                    "left_stage": left_stage,
                    "right_stage": right_stage,
                    "feature": feature,
                    "paired_units": len(shared_keys),
                    "positive_pairs": positive,
                    "negative_pairs": negative,
                    "zero_pairs": zero,
                    "left_mean_per_1k_tokens": sum(left_values) / len(left_values)
                    if left_values
                    else 0.0,
                    "right_mean_per_1k_tokens": sum(right_values) / len(right_values)
                    if right_values
                    else 0.0,
                    "mean_delta_per_1k_tokens": sum(deltas) / len(deltas) if deltas else 0.0,
                    "median_delta_per_1k_tokens": _median(deltas),
                    "sign_test_p": p_value,
                    "bh_q": 1.0,
                    "fdr_significant": False,
                    "direction": "right_gt_left"
                    if positive > negative
                    else "left_gt_right"
                    if negative > positive
                    else "tie",
                }
            )
    q_values = _benjamini_hochberg(p_values)
    for row in rows:
        key = (str(row["comparison"]), str(row["feature"]))
        q_value = q_values[key]
        row["bh_q"] = q_value
        row["fdr_significant"] = q_value <= args.alpha
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value != 0.0 and abs(value) < 0.001:
            return f"{value:.3e}"
        return f"{value:.3f}"
    return str(value)


def _write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    significant = [row for row in rows if row["fdr_significant"]]
    lines = [
        "# Phase 3 Free-Running Stage Effects",
        "",
        "Paired sign tests over shared `(record/prompt id, completion_index, temperature, top_p)` "
        "generation units.",
        "",
        f"Rows: `{len(rows)}`",
        f"BH-FDR alpha: `{args.alpha}`",
        f"FDR-significant rows: `{len(significant)}`",
        "",
        "| Comparison | Feature | Pairs | Mean Delta /1k | p | q | Direction | FDR |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda item: (float(item["bh_q"]), item["comparison"], item["feature"])):
        lines.append(
            "| {comparison} | `{feature}` | {pairs} | {delta} | {p} | {q} | {direction} | {fdr} |".format(
                comparison=row["comparison"],
                feature=row["feature"],
                pairs=row["paired_units"],
                delta=_fmt(row["mean_delta_per_1k_tokens"]),
                p=_fmt(row["sign_test_p"]),
                q=_fmt(row["bh_q"]),
                direction=row["direction"],
                fdr="yes" if row["fdr_significant"] else "no",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analyze_phase3_free_run_effects(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = analyze_free_run_effects(args)
    _write_csv(args.output, rows)
    _write_summary(args.summary_output, rows, args)
    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage3", "phase3", "free-run-stage-effects", *args.wandb_tag],
        config={
            "generation_cache": args.generation_cache,
            "comparisons": args.comparison,
            "features": args.feature,
            "alpha": args.alpha,
            "output": str(args.output),
            "summary_output": str(args.summary_output),
        },
    )
    try:
        run.log(
            {
                "free_run_effects/rows": len(rows),
                "free_run_effects/fdr_significant_rows": sum(
                    1 for row in rows if row["fdr_significant"]
                ),
            }
        )
        log_summary_table(run, "phase3_free_run_stage_effects", rows)
    finally:
        run.finish()
    return rows


def main() -> None:
    args = build_parser().parse_args()
    rows = run_analyze_phase3_free_run_effects(args)
    print(f"Wrote {len(rows)} free-running stage-effect rows to {args.output}")


if __name__ == "__main__":
    main()
