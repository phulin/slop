from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


STAGE_ORDER = {
    "base": 0,
    "instruct_final": 1,
    "think_sft": 2,
    "think_dpo": 3,
    "think_final": 4,
    "rl_zero_general": 5,
    "rl_zero_if": 6,
    "rl_zero_math": 7,
    "rl_zero_code": 8,
    "rl_zero_mix": 9,
}
STAGE_FAMILY = {
    "base": "base",
    "instruct_final": "instruct",
    "think_sft": "think",
    "think_dpo": "think",
    "think_final": "think",
    "rl_zero_general": "rl_zero",
    "rl_zero_if": "rl_zero",
    "rl_zero_math": "rl_zero",
    "rl_zero_code": "rl_zero",
    "rl_zero_mix": "rl_zero",
}
STAGE_LABELS = {
    "base": "Base",
    "instruct_final": "Instruct final",
    "think_sft": "Think SFT",
    "think_dpo": "Think DPO",
    "think_final": "Think final",
    "rl_zero_general": "RL-Zero General",
    "rl_zero_if": "RL-Zero IF",
    "rl_zero_math": "RL-Zero Math",
    "rl_zero_code": "RL-Zero Code",
    "rl_zero_mix": "RL-Zero Mix",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble the optional OLMo Instruct/Think/RL-Zero stretch comparison."
    )
    parser.add_argument("--generation-grid", type=Path, required=True)
    parser.add_argument("--primary-feature", default="slop_lexicon")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-summary-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase3-analysis")
    parser.add_argument("--wandb-job-type", default="olmo-stretch-comparison")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _safe_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _safe_ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    if baseline == 0:
        return math.inf if value > 0 else 0.0
    return value / baseline


def _stage_sort_key(stage: str) -> tuple[int, str]:
    return STAGE_ORDER.get(stage, len(STAGE_ORDER)), stage


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{float(value):.3f}"


def _comparison_rows(grid_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rates: dict[tuple[str, str], float | None] = {}
    for row in grid_rows:
        stage = str(row["stage"])
        feature = str(row["feature"])
        rates[(stage, feature)] = _float_or_none(row.get("per_1k_tokens"))

    out: list[dict[str, Any]] = []
    for row in grid_rows:
        stage = str(row["stage"])
        feature = str(row["feature"])
        value = _float_or_none(row.get("per_1k_tokens"))
        base = rates.get(("base", feature))
        instruct_final = rates.get(("instruct_final", feature))
        think_final = rates.get(("think_final", feature))
        out.append(
            {
                "stage": stage,
                "stage_order": STAGE_ORDER.get(stage, len(STAGE_ORDER)),
                "stage_label": STAGE_LABELS.get(stage, stage),
                "stage_family": STAGE_FAMILY.get(stage, "other"),
                "feature": feature,
                "temperature": _float_or_none(row.get("temperature")),
                "top_p": _float_or_none(row.get("top_p")),
                "count": int(float(row.get("count") or 0)),
                "generations": int(float(row.get("generations") or 0)),
                "generated_tokens": int(float(row.get("generated_tokens") or 0)),
                "per_1k_tokens": value,
                "per_generation": _float_or_none(row.get("per_generation")),
                "share_generations_with_feature": _float_or_none(
                    row.get("share_generations_with_feature")
                ),
                "base_per_1k_tokens": base,
                "instruct_final_per_1k_tokens": instruct_final,
                "think_final_per_1k_tokens": think_final,
                "delta_vs_base": _safe_delta(value, base),
                "ratio_vs_base": _safe_ratio(value, base),
                "delta_vs_instruct_final": _safe_delta(value, instruct_final),
                "ratio_vs_instruct_final": _safe_ratio(value, instruct_final),
                "delta_vs_think_final": _safe_delta(value, think_final),
                "ratio_vs_think_final": _safe_ratio(value, think_final),
            }
        )
    return sorted(out, key=lambda row: (str(row["feature"]), row["stage_order"], str(row["stage"])))


def _feature_summary_rows(comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        by_feature[str(row["feature"])].append(row)

    rows: list[dict[str, Any]] = []
    for feature, feature_rows in sorted(by_feature.items()):
        valid = [row for row in feature_rows if row["per_1k_tokens"] is not None]
        if not valid:
            continue
        max_row = max(valid, key=lambda row: float(row["per_1k_tokens"]))
        base_row = next((row for row in valid if row["stage"] == "base"), None)
        instruct_row = next((row for row in valid if row["stage"] == "instruct_final"), None)
        think_row = next((row for row in valid if row["stage"] == "think_final"), None)
        rl_rows = [row for row in valid if row["stage_family"] == "rl_zero"]
        rl_mean = (
            sum(float(row["per_1k_tokens"]) for row in rl_rows) / len(rl_rows)
            if rl_rows
            else None
        )
        rows.append(
            {
                "feature": feature,
                "max_stage": max_row["stage"],
                "max_stage_family": max_row["stage_family"],
                "max_per_1k_tokens": max_row["per_1k_tokens"],
                "base_per_1k_tokens": base_row["per_1k_tokens"] if base_row else None,
                "instruct_final_per_1k_tokens": (
                    instruct_row["per_1k_tokens"] if instruct_row else None
                ),
                "think_final_per_1k_tokens": think_row["per_1k_tokens"] if think_row else None,
                "rl_zero_mean_per_1k_tokens": rl_mean,
                "think_final_minus_instruct_final": _safe_delta(
                    think_row["per_1k_tokens"] if think_row else None,
                    instruct_row["per_1k_tokens"] if instruct_row else None,
                ),
                "rl_zero_mean_minus_instruct_final": _safe_delta(
                    rl_mean,
                    instruct_row["per_1k_tokens"] if instruct_row else None,
                ),
            }
        )
    return rows


def _write_summary(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    primary_feature: str,
) -> None:
    primary_rows = [row for row in rows if row["feature"] == primary_feature]
    primary_rows.sort(key=lambda row: _stage_sort_key(str(row["stage"])))
    primary_summary = next(
        (row for row in feature_rows if row["feature"] == primary_feature),
        None,
    )
    lines = [
        "# OLMo Think/RL-Zero Stretch Comparison",
        "",
        "This report assembles already generated final-answer-mode Tier-1 summaries. It does not run generation.",
        "",
        f"Primary feature: `{primary_feature}`",
        "",
        "## Primary Feature",
        "",
        "| Stage | Family | Per 1k Tokens | Delta vs Instruct Final | Delta vs Think Final |",
        "|---|---|---:|---:|---:|",
    ]
    for row in primary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["stage"]),
                    str(row["stage_family"]),
                    _fmt(row["per_1k_tokens"]),
                    _fmt(row["delta_vs_instruct_final"]),
                    _fmt(row["delta_vs_think_final"]),
                ]
            )
            + " |"
        )
    if primary_summary is not None:
        lines.extend(
            [
                "",
                "Primary-feature maximum: "
                f"`{primary_summary['max_stage']}` "
                f"({_fmt(primary_summary['max_per_1k_tokens'])} per 1k tokens).",
            ]
        )
    lines.extend(
        [
            "",
            "## Feature Maxima",
            "",
            "| Feature | Max Stage | Max Family | Max Per 1k | Think Final - Instruct Final | RL-Zero Mean - Instruct Final |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in feature_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["feature"]),
                    str(row["max_stage"]),
                    str(row["max_stage_family"]),
                    _fmt(row["max_per_1k_tokens"]),
                    _fmt(row["think_final_minus_instruct_final"]),
                    _fmt(row["rl_zero_mean_minus_instruct_final"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Input summaries must be generated with `--feature-text-mode final_answer` for reasoning-model comparability.",
            "- This table is output-side evidence only; it does not add teacher-forced AF cells.",
            "- The reduced stretch plan is intentionally small and should be read as a path comparison, not a full production grid.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_assemble_olmo_stretch_comparison(args: argparse.Namespace) -> dict[str, Any]:
    grid_rows = _read_csv(args.generation_grid)
    if not grid_rows:
        raise ValueError("generation grid is empty")
    rows = _comparison_rows(grid_rows)
    feature_rows = _feature_summary_rows(rows)
    _write_csv(args.output, rows)
    _write_csv(args.feature_summary_output, feature_rows)
    _write_summary(
        path=args.summary_output,
        rows=rows,
        feature_rows=feature_rows,
        primary_feature=args.primary_feature,
    )

    primary_feature_summary = next(
        (row for row in feature_rows if row["feature"] == args.primary_feature),
        {},
    )
    summary = {
        "rows": len(rows),
        "features": len(feature_rows),
        "stages": len({row["stage"] for row in rows}),
        "primary_feature": args.primary_feature,
        "primary_feature_max_stage": primary_feature_summary.get("max_stage"),
        "primary_feature_max_per_1k_tokens": primary_feature_summary.get(
            "max_per_1k_tokens"
        ),
        "output": str(args.output),
        "feature_summary_output": str(args.feature_summary_output),
        "summary_output": str(args.summary_output),
    }

    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["phase3", "olmo", "think-rlzero", *args.wandb_tag],
        config={
            "generation_grid": str(args.generation_grid),
            "primary_feature": args.primary_feature,
            "output": str(args.output),
            "feature_summary_output": str(args.feature_summary_output),
            "summary_output": str(args.summary_output),
        },
    )
    try:
        run.log({f"phase3_olmo_stretch/{key}": value for key, value in summary.items()})
        log_summary_table(run, "phase3_olmo_stretch_comparison", rows)
        log_summary_table(run, "phase3_olmo_stretch_feature_summary", feature_rows)
    finally:
        run.finish()
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_assemble_olmo_stretch_comparison(args)
    print(
        "Wrote OLMo stretch comparison with "
        f"{summary['rows']} rows across {summary['stages']} stages"
    )


if __name__ == "__main__":
    main()
