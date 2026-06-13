from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.assemble_phase2_grid import (
    STAGE_ORDER,
    _float_or_none,
    _fmt,
    _int_or_zero,
    _parse_stage_mapping,
    _stage_sort_key,
    _write_csv,
)
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble Phase 2 free-running generation stage-grid summaries."
    )
    parser.add_argument(
        "--generation-summary",
        action="append",
        required=True,
        help="Stage-tagged free-running summary CSV in the form stage=path.",
    )
    parser.add_argument(
        "--checkpoint-label",
        action="append",
        default=[],
        help="Optional display label in the form stage=label.",
    )
    parser.add_argument("--primary-feature", default="slop_lexicon")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase2/analysis/phase2_generation_stage_grid.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("artifacts/phase2/analysis/phase2_generation_primary_feature_comparison.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/phase2/analysis/phase2_generation_stage_grid_summary.md"),
    )
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-analysis")
    parser.add_argument("--wandb-job-type", default="assemble-phase2-generation-grid")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _read_generation_rows(
    summary_inputs: dict[str, str],
    checkpoint_labels: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in sorted(summary_inputs, key=_stage_sort_key):
        path = Path(summary_inputs[stage])
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "stage": stage,
                        "stage_order": _stage_sort_key(stage)[0],
                        "checkpoint_label": checkpoint_labels.get(stage, stage),
                        "input_path": str(path),
                        "source": row.get("source") or "",
                        "temperature": _float_or_none(row.get("temperature")),
                        "top_p": _float_or_none(row.get("top_p")),
                        "feature": row["feature"],
                        "count": _int_or_zero(row.get("count")),
                        "repeated_count": _int_or_zero(row.get("repeated_count")),
                        "generations": _int_or_zero(row.get("generations")),
                        "generations_with_feature": _int_or_zero(
                            row.get("generations_with_feature")
                        ),
                        "repeat_generations": _int_or_zero(row.get("repeat_generations")),
                        "generated_tokens": _int_or_zero(row.get("generated_tokens")),
                        "per_1k_tokens": _float_or_none(row.get("per_1k_tokens")),
                        "per_generation": _float_or_none(row.get("per_generation")),
                        "repeat_per_generation": _float_or_none(
                            row.get("repeat_per_generation")
                        ),
                        "repeat_share_after_first": _float_or_none(
                            row.get("repeat_share_after_first")
                        ),
                        "share_generations_with_feature": _float_or_none(
                            row.get("share_generations_with_feature")
                        ),
                        "share_repeat_generations": _float_or_none(
                            row.get("share_repeat_generations")
                        ),
                    }
                )
    return rows


def _comparison_sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
    temperature = row["temperature"]
    return (
        float("-inf") if temperature is None else temperature,
        row["stage_order"],
        row["stage"],
    )


def _primary_feature_comparison(
    rows: list[dict[str, Any]],
    *,
    primary_feature: str,
) -> list[dict[str, Any]]:
    primary_rows = [row for row in rows if row["feature"] == primary_feature]
    primary_rows.sort(key=_comparison_sort_key)
    if not primary_rows:
        raise ValueError(f"primary feature not found in summaries: {primary_feature}")

    baseline_by_temperature: dict[float | None, float | None] = {}
    previous_by_temperature: dict[float | None, float | None] = {}
    comparison_rows: list[dict[str, Any]] = []
    for row in primary_rows:
        temperature = row["temperature"]
        baseline = baseline_by_temperature.setdefault(temperature, row["per_1k_tokens"])
        previous = previous_by_temperature.get(temperature)
        per_1k_tokens = row["per_1k_tokens"]
        comparison_rows.append(
            {
                "stage": row["stage"],
                "stage_order": row["stage_order"],
                "checkpoint_label": row["checkpoint_label"],
                "temperature": temperature,
                "top_p": row["top_p"],
                "feature": primary_feature,
                "count": row["count"],
                "repeated_count": row["repeated_count"],
                "generations": row["generations"],
                "generated_tokens": row["generated_tokens"],
                "per_1k_tokens": per_1k_tokens,
                "per_generation": row["per_generation"],
                "repeat_per_generation": row["repeat_per_generation"],
                "repeat_share_after_first": row["repeat_share_after_first"],
                "share_generations_with_feature": row["share_generations_with_feature"],
                "share_repeat_generations": row["share_repeat_generations"],
                "delta_per_1k_vs_baseline_stage": (
                    per_1k_tokens - baseline
                    if per_1k_tokens is not None and baseline is not None
                    else None
                ),
                "delta_per_1k_vs_previous_stage": (
                    per_1k_tokens - previous
                    if per_1k_tokens is not None and previous is not None
                    else None
                ),
            }
        )
        previous_by_temperature[temperature] = per_1k_tokens
    return comparison_rows


def _write_summary(
    *,
    path: Path,
    grid_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    primary_feature: str,
) -> None:
    lines = [
        "# Phase 2 Generation Stage Grid",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        f"Primary feature: `{primary_feature}`",
        "",
        "## Primary Feature Comparison",
        "",
        "| Temperature | Stage | Count | Per 1k Tokens | Per Generation | Repeat/Generation | Repeat Share |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["temperature"]),
                    str(row["stage"]),
                    str(row["count"]),
                    _fmt(row["per_1k_tokens"]),
                    _fmt(row["per_generation"]),
                    _fmt(row["repeat_per_generation"]),
                    _fmt(row["repeat_share_after_first"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            *[
                f"- `{row['stage']}`: `{row['input_path']}`"
                for row in sorted(
                    {row["stage"]: row for row in grid_rows}.values(),
                    key=lambda item: (item["stage_order"], item["stage"]),
                )
            ],
            "",
            "## Caveats",
            "",
            "- This assembles existing free-running summary CSVs; it does not rerun generation.",
            "- Sparse shards are useful throughput and plumbing measurements, not stable rate estimates.",
            "- Repeat columns are the direct inputs for the Phase 2 compounding analysis.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_assemble_phase2_generation_grid(args: argparse.Namespace) -> dict[str, Any]:
    summary_inputs = _parse_stage_mapping(
        args.generation_summary,
        argument_name="--generation-summary",
    )
    checkpoint_labels = _parse_stage_mapping(
        args.checkpoint_label,
        argument_name="--checkpoint-label",
    )
    grid_rows = _read_generation_rows(summary_inputs, checkpoint_labels)
    if not grid_rows:
        raise ValueError("no generation summary rows loaded")
    comparison_rows = _primary_feature_comparison(
        grid_rows,
        primary_feature=args.primary_feature,
    )
    _write_csv(args.output, grid_rows)
    _write_csv(args.comparison_output, comparison_rows)
    _write_summary(
        path=args.summary_output,
        grid_rows=grid_rows,
        comparison_rows=comparison_rows,
        primary_feature=args.primary_feature,
    )

    best_row = max(
        comparison_rows,
        key=lambda row: row["per_1k_tokens"] or float("-inf"),
    )
    summary = {
        "grid_rows": len(grid_rows),
        "comparison_rows": len(comparison_rows),
        "stages": len(summary_inputs),
        "features": len({row["feature"] for row in grid_rows}),
        "temperatures": len({row["temperature"] for row in grid_rows}),
        "primary_feature": args.primary_feature,
        "max_per_1k_stage": best_row["stage"],
        "max_per_1k_temperature": best_row["temperature"],
        "max_per_1k_tokens": best_row["per_1k_tokens"],
        "output": str(args.output),
        "comparison_output": str(args.comparison_output),
        "summary_output": str(args.summary_output),
    }

    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "assemble", "generation_grid", *args.wandb_tag],
        config={
            "generation_summaries": summary_inputs,
            "checkpoint_labels": checkpoint_labels,
            "primary_feature": args.primary_feature,
            "output": str(args.output),
            "comparison_output": str(args.comparison_output),
            "summary_output": str(args.summary_output),
            "stage_order": STAGE_ORDER,
        },
    )
    try:
        run.log({f"phase2_generation_grid/{key}": value for key, value in summary.items()})
        log_summary_table(run, "phase2_generation_grid", grid_rows)
        log_summary_table(run, "phase2_generation_primary_feature_comparison", comparison_rows)
    finally:
        run.finish()
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_assemble_phase2_generation_grid(args)
    print(
        "Wrote Phase 2 generation grid with "
        f"{summary['grid_rows']} rows across {summary['stages']} stages"
    )


if __name__ == "__main__":
    main()
