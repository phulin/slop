from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


STAGE_ORDER = ("base", "sft", "dpo", "final")
STAGE_LABELS = {
    "base": "Base",
    "sft": "SFT",
    "dpo": "DPO",
    "final": "Final/RLVR",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two Phase 3 amplification spectra and report cross-ladder AF rank correlations."
    )
    parser.add_argument("--left-spectrum", type=Path, required=True)
    parser.add_argument("--right-spectrum", type=Path, required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--aligned-output", type=Path, required=True)
    parser.add_argument("--correlation-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase3-analysis")
    parser.add_argument("--wandb-job-type", default="compare-phase3-ladders")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.3f}"


def _stage_order(stage: str) -> int:
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)


def _af_value(row: dict[str, Any]) -> tuple[float | None, str]:
    normalized = _float_or_none(row.get("teacher_forced_normalized_af"))
    if normalized is not None and normalized > 0.0:
        return normalized, "normalized"
    raw = _float_or_none(row.get("teacher_forced_af"))
    return raw, "raw" if raw is not None else ""


def _spectrum_values(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    values = {}
    for row in _read_csv(path):
        feature = str(row["feature"])
        stage = str(row["stage"])
        af, af_source = _af_value(row)
        values[(feature, stage)] = {
            "feature": feature,
            "stage": stage,
            "stage_label": row.get("stage_label") or STAGE_LABELS.get(stage, stage),
            "af": af,
            "af_source": af_source,
            "free_run_per_1k_tokens": _float_or_none(row.get("free_run_per_1k_tokens")),
            "sft_target_per_1k_tokens": _float_or_none(row.get("sft_target_per_1k_tokens")),
            "dpo_chosen_per_1k_tokens": _float_or_none(row.get("dpo_chosen_per_1k_tokens")),
            "dpo_rejected_per_1k_tokens": _float_or_none(row.get("dpo_rejected_per_1k_tokens")),
            "coverage_note": row.get("coverage_note", ""),
        }
    return values


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for offset in range(index, end):
            ranks[ordered[offset][0]] = average_rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_ss = sum((a - left_mean) ** 2 for a in left)
    right_ss = sum((b - right_mean) ** 2 for b in right)
    if left_ss <= 0.0 or right_ss <= 0.0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    return _pearson(_ranks(left), _ranks(right))


def _correlation_row(scope: str, stage: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    af_rows = [
        row for row in rows if row["left_af"] is not None and row["right_af"] is not None
    ]
    left_values = [float(row["left_af"]) for row in af_rows]
    right_values = [float(row["right_af"]) for row in af_rows]
    return {
        "scope": scope,
        "stage": stage,
        "shared_af_values": len(af_rows),
        "spearman_af": _spearman(left_values, right_values),
        "pearson_af": _pearson(left_values, right_values),
    }


def compare_spectra(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left = _spectrum_values(args.left_spectrum)
    right = _spectrum_values(args.right_spectrum)
    aligned = []
    for key in sorted(set(left) & set(right), key=lambda item: (item[0], _stage_order(item[1]))):
        left_row = left[key]
        right_row = right[key]
        aligned.append(
            {
                "feature": key[0],
                "stage": key[1],
                "stage_label": left_row["stage_label"],
                "left_label": args.left_label,
                "right_label": args.right_label,
                "left_af": left_row["af"],
                "right_af": right_row["af"],
                "af_delta_right_minus_left": (
                    right_row["af"] - left_row["af"]
                    if left_row["af"] is not None and right_row["af"] is not None
                    else None
                ),
                "left_af_source": left_row["af_source"],
                "right_af_source": right_row["af_source"],
                "left_free_run_per_1k_tokens": left_row["free_run_per_1k_tokens"],
                "right_free_run_per_1k_tokens": right_row["free_run_per_1k_tokens"],
                "free_run_delta_right_minus_left": (
                    right_row["free_run_per_1k_tokens"] - left_row["free_run_per_1k_tokens"]
                    if left_row["free_run_per_1k_tokens"] is not None
                    and right_row["free_run_per_1k_tokens"] is not None
                    else None
                ),
                "left_sft_target_per_1k_tokens": left_row["sft_target_per_1k_tokens"],
                "right_sft_target_per_1k_tokens": right_row["sft_target_per_1k_tokens"],
                "left_dpo_chosen_per_1k_tokens": left_row["dpo_chosen_per_1k_tokens"],
                "right_dpo_chosen_per_1k_tokens": right_row["dpo_chosen_per_1k_tokens"],
                "left_dpo_rejected_per_1k_tokens": left_row["dpo_rejected_per_1k_tokens"],
                "right_dpo_rejected_per_1k_tokens": right_row["dpo_rejected_per_1k_tokens"],
                "left_coverage_note": left_row["coverage_note"],
                "right_coverage_note": right_row["coverage_note"],
            }
        )

    rows_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aligned:
        rows_by_stage[str(row["stage"])].append(row)
    correlation_rows = [_correlation_row("all", "all", aligned)]
    for stage in STAGE_ORDER:
        correlation_rows.append(_correlation_row("stage", stage, rows_by_stage.get(stage, [])))
    return aligned, correlation_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "feature",
        "stage",
        "stage_label",
        "left_label",
        "right_label",
        "left_af",
        "right_af",
        "af_delta_right_minus_left",
        "left_af_source",
        "right_af_source",
        "left_free_run_per_1k_tokens",
        "right_free_run_per_1k_tokens",
        "free_run_delta_right_minus_left",
    ]
    columns = [column for column in preferred if column in columns] + [
        column for column in columns if column not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    aligned_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 3 Cross-Ladder Amplification Comparison",
        "",
        f"Left ladder: `{args.left_label}`",
        f"Right ladder: `{args.right_label}`",
        f"Left spectrum: `{args.left_spectrum}`",
        f"Right spectrum: `{args.right_spectrum}`",
        "",
        f"Aligned feature-stage rows: {len(aligned_rows)}",
        "",
        "## AF Rank Correlations",
        "",
        "| Scope | Stage | Shared AF Values | Spearman AF | Pearson AF |",
        "|---|---|---:|---:|---:|",
    ]
    for row in correlation_rows:
        lines.append(
            "| {scope} | {stage} | {n} | {spearman} | {pearson} |".format(
                scope=row["scope"],
                stage=row["stage"],
                n=row["shared_af_values"],
                spearman=_fmt(row["spearman_af"]),
                pearson=_fmt(row["pearson_af"]),
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Correlations use only shared feature-stage rows with non-missing AF values.",
            "- Normalized AF is used when present; otherwise raw AF is used.",
            "- This command reports the cross-ladder statistic required by Phase 3, but it does not create the missing ladder spectra.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aligned_rows, correlation_rows = compare_spectra(args)
    _write_csv(args.aligned_output, aligned_rows)
    _write_csv(args.correlation_output, correlation_rows)
    _write_summary(args.summary_output, aligned_rows, correlation_rows, args)
    run_obj = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage3", "phase3", "cross-ladder", *args.wandb_tag],
        config={
            "left_spectrum": str(args.left_spectrum),
            "right_spectrum": str(args.right_spectrum),
            "left_label": args.left_label,
            "right_label": args.right_label,
            "aligned_output": str(args.aligned_output),
            "correlation_output": str(args.correlation_output),
            "summary_output": str(args.summary_output),
        },
    )
    try:
        overall = next(row for row in correlation_rows if row["scope"] == "all")
        run_obj.log(
            {
                "phase3_cross_ladder/aligned_rows": len(aligned_rows),
                "phase3_cross_ladder/shared_af_values": overall["shared_af_values"],
                "phase3_cross_ladder/spearman_af": overall["spearman_af"],
                "phase3_cross_ladder/pearson_af": overall["pearson_af"],
            }
        )
        log_summary_table(run_obj, "phase3_cross_ladder_aligned", aligned_rows)
        log_summary_table(run_obj, "phase3_cross_ladder_correlations", correlation_rows)
    finally:
        run_obj.finish()
    return aligned_rows, correlation_rows


def main() -> None:
    args = build_parser().parse_args()
    aligned_rows, correlation_rows = run(args)
    overall = next(row for row in correlation_rows if row["scope"] == "all")
    print(
        "Wrote {aligned} aligned rows to {path}; overall Spearman AF={spearman}".format(
            aligned=len(aligned_rows),
            path=args.aligned_output,
            spearman=_fmt(overall["spearman_af"]),
        )
    )


if __name__ == "__main__":
    main()
