from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Any

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


STAGE_ORDER = ("base", "sft", "dpo", "final")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble Phase 2 propensity stage-grid summaries.")
    parser.add_argument(
        "--propensity-summary",
        action="append",
        required=True,
        help="Stage-tagged summary CSV in the form stage=path. Repeat for each checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-label",
        action="append",
        default=[],
        help="Optional display label in the form stage=label.",
    )
    parser.add_argument("--primary-feature", default="slop_lexicon")
    parser.add_argument("--normalization-feature", default="neutral_common_controls")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase2/analysis/phase2_propensity_stage_grid.csv"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("artifacts/phase2/analysis/phase2_primary_feature_stage_comparison.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/phase2/analysis/phase2_propensity_stage_grid_summary.md"),
    )
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-analysis")
    parser.add_argument("--wandb-job-type", default="assemble-phase2-grid")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _parse_stage_mapping(items: list[str], *, argument_name: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{argument_name} must use stage=value syntax: {item!r}")
        stage, value = item.split("=", 1)
        stage = stage.strip()
        value = value.strip()
        if not stage or not value:
            raise ValueError(f"{argument_name} must use non-empty stage=value syntax: {item!r}")
        if stage in mapping:
            raise ValueError(f"duplicate stage for {argument_name}: {stage}")
        mapping[stage] = value
    return mapping


def _stage_sort_key(stage: str) -> tuple[int, str]:
    if stage in STAGE_ORDER:
        return (STAGE_ORDER.index(stage), stage)
    return (len(STAGE_ORDER), stage)


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _int_or_zero(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def _read_grid_rows(
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
                        "feature": row["feature"],
                        "opportunities": _int_or_zero(row.get("opportunities")),
                        "reference_initiations": _int_or_zero(row.get("reference_initiations")),
                        "reference_rate": _float_or_none(row.get("reference_rate")),
                        "reference_rate_ci_low": _float_or_none(row.get("reference_rate_ci_low")),
                        "reference_rate_ci_high": _float_or_none(row.get("reference_rate_ci_high")),
                        "mean_prob_mass": _float_or_none(row.get("mean_prob_mass")),
                        "mean_prob_mass_ci_low": _float_or_none(row.get("mean_prob_mass_ci_low")),
                        "mean_prob_mass_ci_high": _float_or_none(row.get("mean_prob_mass_ci_high")),
                        "amplification_factor": _float_or_none(row.get("amplification_factor")),
                        "amplification_factor_ci_low": _float_or_none(
                            row.get("amplification_factor_ci_low")
                        ),
                        "amplification_factor_ci_high": _float_or_none(
                            row.get("amplification_factor_ci_high")
                        ),
                        "normalization_feature": row.get("normalization_feature") or "",
                        "normalized_amplification_factor": _float_or_none(
                            row.get("normalized_amplification_factor")
                        ),
                        "normalized_amplification_factor_ci_low": _float_or_none(
                            row.get("normalized_amplification_factor_ci_low")
                        ),
                        "normalized_amplification_factor_ci_high": _float_or_none(
                            row.get("normalized_amplification_factor_ci_high")
                        ),
                    }
                )
    return rows


def _primary_feature_comparison(
    rows: list[dict[str, Any]],
    *,
    primary_feature: str,
) -> list[dict[str, Any]]:
    primary_rows = [
        row for row in rows if row["feature"] == primary_feature
    ]
    primary_rows.sort(key=lambda row: (row["stage_order"], row["stage"]))
    if not primary_rows:
        raise ValueError(f"primary feature not found in summaries: {primary_feature}")
    base_af = primary_rows[0]["normalized_amplification_factor"]
    previous_af: float | None = None
    comparison_rows: list[dict[str, Any]] = []
    for row in primary_rows:
        normalized_af = row["normalized_amplification_factor"]
        delta_vs_base = (
            normalized_af - base_af
            if normalized_af is not None and base_af is not None
            else None
        )
        delta_vs_previous = (
            normalized_af - previous_af
            if normalized_af is not None and previous_af is not None
            else None
        )
        ratio_vs_base = (
            normalized_af / base_af
            if normalized_af is not None and base_af not in (None, 0.0)
            else None
        )
        comparison_rows.append(
            {
                "stage": row["stage"],
                "stage_order": row["stage_order"],
                "checkpoint_label": row["checkpoint_label"],
                "feature": primary_feature,
                "reference_initiations": row["reference_initiations"],
                "amplification_factor": row["amplification_factor"],
                "amplification_factor_ci_low": row["amplification_factor_ci_low"],
                "amplification_factor_ci_high": row["amplification_factor_ci_high"],
                "normalized_amplification_factor": normalized_af,
                "normalized_amplification_factor_ci_low": row[
                    "normalized_amplification_factor_ci_low"
                ],
                "normalized_amplification_factor_ci_high": row[
                    "normalized_amplification_factor_ci_high"
                ],
                "delta_normalized_af_vs_base": delta_vs_base,
                "delta_normalized_af_vs_previous": delta_vs_previous,
                "ratio_normalized_af_vs_base": ratio_vs_base,
            }
        )
        previous_af = normalized_af
    return comparison_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_summary(
    *,
    path: Path,
    grid_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    primary_feature: str,
    normalization_feature: str,
) -> None:
    lines = [
        "# Phase 2 Propensity Stage Grid",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        f"Primary feature: `{primary_feature}`",
        f"Normalization feature: `{normalization_feature}`",
        "",
        "## Primary Feature Comparison",
        "",
        "| Stage | Raw AF | Raw AF CI | Normalized AF | Normalized AF CI | Delta vs Base |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["stage"]),
                    _fmt(row["amplification_factor"]),
                    f"{_fmt(row['amplification_factor_ci_low'])}-{_fmt(row['amplification_factor_ci_high'])}",
                    _fmt(row["normalized_amplification_factor"]),
                    (
                        f"{_fmt(row['normalized_amplification_factor_ci_low'])}-"
                        f"{_fmt(row['normalized_amplification_factor_ci_high'])}"
                    ),
                    _fmt(row["delta_normalized_af_vs_base"]),
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
            "- This assembles existing teacher-forced summary CSVs; it does not rerun model scoring.",
            "- Normalized AF depends on the current `neutral_common_controls` opportunity contract.",
            "- The narrow grid currently covers the slop-lexicon positive control, not the full Phase 2 feature set.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_assemble_phase2_grid(args: argparse.Namespace) -> dict[str, Any]:
    summary_inputs = _parse_stage_mapping(
        args.propensity_summary,
        argument_name="--propensity-summary",
    )
    checkpoint_labels = _parse_stage_mapping(
        args.checkpoint_label,
        argument_name="--checkpoint-label",
    )
    grid_rows = _read_grid_rows(summary_inputs, checkpoint_labels)
    if not grid_rows:
        raise ValueError("no summary rows loaded")
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
        normalization_feature=args.normalization_feature,
    )

    best_row = max(
        comparison_rows,
        key=lambda row: row["normalized_amplification_factor"] or float("-inf"),
    )
    summary = {
        "grid_rows": len(grid_rows),
        "comparison_rows": len(comparison_rows),
        "stages": len(summary_inputs),
        "features": len({row["feature"] for row in grid_rows}),
        "primary_feature": args.primary_feature,
        "normalization_feature": args.normalization_feature,
        "max_normalized_af_stage": best_row["stage"],
        "max_normalized_af": best_row["normalized_amplification_factor"],
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
        tags=["stage2", "phase2", "assemble", "propensity_grid", *args.wandb_tag],
        config={
            "propensity_summaries": summary_inputs,
            "checkpoint_labels": checkpoint_labels,
            "primary_feature": args.primary_feature,
            "normalization_feature": args.normalization_feature,
            "output": str(args.output),
            "comparison_output": str(args.comparison_output),
            "summary_output": str(args.summary_output),
        },
    )
    try:
        run.log({f"phase2_grid/{key}": value for key, value in summary.items()})
        log_summary_table(run, "phase2_propensity_grid", grid_rows)
        log_summary_table(run, "phase2_primary_feature_comparison", comparison_rows)
    finally:
        run.finish()
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_assemble_phase2_grid(args)
    print(
        "Wrote Phase 2 grid with "
        f"{summary['grid_rows']} rows across {summary['stages']} stages"
    )


if __name__ == "__main__":
    main()
