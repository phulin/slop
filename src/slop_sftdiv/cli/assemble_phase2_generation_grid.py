from __future__ import annotations

import argparse
import csv
import json
import random
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
        "--generation-cache",
        action="append",
        default=[],
        help=(
            "Optional stage-tagged free-running JSONL cache in the form stage=path. "
            "When supplied with --bootstrap-samples > 0, document-cluster bootstrap "
            "CIs are added for per_1k_tokens."
        ),
    )
    parser.add_argument(
        "--checkpoint-label",
        action="append",
        default=[],
        help="Optional display label in the form stage=label.",
    )
    parser.add_argument("--primary-feature", default="slop_lexicon")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
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


def _parse_stage_paths(items: list[str], *, argument_name: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{argument_name} must use stage=value syntax: {item!r}")
        stage, value = item.split("=", 1)
        stage = stage.strip()
        value = value.strip()
        if not stage or not value:
            raise ValueError(f"{argument_name} must use non-empty stage=value syntax: {item!r}")
        mapping.setdefault(stage, []).append(value)
    return mapping


def _read_generation_rows(
    summary_inputs: dict[str, list[str]],
    checkpoint_labels: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in sorted(summary_inputs, key=_stage_sort_key):
        for path_text in summary_inputs[stage]:
            path = Path(path_text)
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
                            "repeat_generations": _int_or_zero(
                                row.get("repeat_generations")
                            ),
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


def _record_key(row: dict[str, Any]) -> str:
    for key in ("record_id", "prompt_id", "phase2_prompt_id", "source_row_index", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return str(row.get("row_index", ""))


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = probability * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap_rate_intervals(
    cache_inputs: dict[str, list[str]],
    *,
    samples: int,
    seed: int,
) -> dict[tuple[str, str, float | None, float | None, str], dict[str, float]]:
    if samples <= 0 or not cache_inputs:
        return {}
    rng = random.Random(seed)
    grouped: dict[
        tuple[str, str, float | None, float | None],
        dict[str, dict[str, float]],
    ] = {}
    for stage in sorted(cache_inputs, key=_stage_sort_key):
        for path_text in cache_inputs[stage]:
            path = Path(path_text)
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    source = str(row.get("source") or "")
                    temperature = _float_or_none(row.get("temperature"))
                    top_p = _float_or_none(row.get("top_p"))
                    generated_tokens = float(row.get("generated_tokens") or 0.0)
                    features = json.loads(row.get("features_json") or "{}")
                    repeated = json.loads(row.get("repeated_features_json") or "{}")
                    cluster = _record_key(row)
                    cell_key = (stage, source, temperature, top_p)
                    group = grouped.setdefault(cell_key, {}).setdefault(cluster, {})
                    group["__tokens__"] = group.get("__tokens__", 0.0) + generated_tokens
                    group["__generations__"] = group.get("__generations__", 0.0) + 1.0
                    for feature, count in features.items():
                        feature = str(feature)
                        value = float(count)
                        count_key = f"{feature}__count"
                        generation_key = f"{feature}__generations_with_feature"
                        group[count_key] = group.get(count_key, 0.0) + value
                        group[generation_key] = group.get(generation_key, 0.0) + (
                            1.0 if value > 0 else 0.0
                        )
                    for feature, count in repeated.items():
                        repeated_key = f"{feature}__repeated_count"
                        group[repeated_key] = group.get(repeated_key, 0.0) + float(count)

    intervals: dict[tuple[str, str, float | None, float | None, str], dict[str, float]] = {}
    for key, clusters in grouped.items():
        cluster_values = list(clusters.values())
        if not cluster_values:
            continue
        features = sorted(
            {
                metric.removesuffix("__count")
                for cluster in cluster_values
                for metric in cluster
                if metric.endswith("__count")
            }
        )
        for feature in features:
            per_1k_samples: list[float] = []
            share_samples: list[float] = []
            for _ in range(samples):
                count = 0.0
                tokens = 0.0
                generations = 0.0
                generations_with_feature = 0.0
                for _cluster_index in range(len(cluster_values)):
                    cluster = rng.choice(cluster_values)
                    count += cluster.get(f"{feature}__count", 0.0)
                    tokens += cluster.get("__tokens__", 0.0)
                    generations += cluster.get("__generations__", 0.0)
                    generations_with_feature += cluster.get(
                        f"{feature}__generations_with_feature", 0.0
                    )
                per_1k_samples.append(1000.0 * count / tokens if tokens > 0 else 0.0)
                share_samples.append(
                    generations_with_feature / generations if generations > 0 else 0.0
                )
            per_1k_samples.sort()
            share_samples.sort()
            intervals[(*key, feature)] = {
                "per_1k_tokens_ci_low": _quantile(per_1k_samples, 0.025),
                "per_1k_tokens_ci_high": _quantile(per_1k_samples, 0.975),
                "share_generations_with_feature_ci_low": _quantile(share_samples, 0.025),
                "share_generations_with_feature_ci_high": _quantile(share_samples, 0.975),
            }
    return intervals


def _apply_bootstrap_intervals(
    rows: list[dict[str, Any]],
    intervals: dict[tuple[str, str, float | None, float | None, str], dict[str, float]],
) -> None:
    if not intervals:
        return
    for row in rows:
        key = (
            row["stage"],
            row["source"],
            row["temperature"],
            row["top_p"],
            row["feature"],
        )
        row.update(intervals.get(key, {}))


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
                "per_1k_tokens_ci_low": row.get("per_1k_tokens_ci_low"),
                "per_1k_tokens_ci_high": row.get("per_1k_tokens_ci_high"),
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
                    (
                        f"{_fmt(row['per_1k_tokens'])} "
                        f"[{_fmt(row.get('per_1k_tokens_ci_low'))}, "
                        f"{_fmt(row.get('per_1k_tokens_ci_high'))}]"
                        if row.get("per_1k_tokens_ci_low") is not None
                        and row.get("per_1k_tokens_ci_high") is not None
                        else _fmt(row["per_1k_tokens"])
                    ),
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
                    {
                        (row["stage"], row["input_path"]): row
                        for row in grid_rows
                    }.values(),
                    key=lambda item: (item["stage_order"], item["stage"], item["input_path"]),
                )
            ],
            "",
            "## Caveats",
            "",
            "- This assembles existing free-running summary CSVs; it does not rerun generation.",
            "- Generation-rate CIs are document/prompt-cluster bootstrap intervals when JSONL caches are supplied.",
            "- Sparse shards are useful throughput and plumbing measurements, not stable rate estimates.",
            "- Repeat columns are the direct inputs for the Phase 2 compounding analysis.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_assemble_phase2_generation_grid(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    summary_inputs = _parse_stage_paths(
        args.generation_summary,
        argument_name="--generation-summary",
    )
    cache_inputs = _parse_stage_paths(
        args.generation_cache,
        argument_name="--generation-cache",
    )
    checkpoint_labels = _parse_stage_mapping(
        args.checkpoint_label,
        argument_name="--checkpoint-label",
    )
    grid_rows = _read_generation_rows(summary_inputs, checkpoint_labels)
    if not grid_rows:
        raise ValueError("no generation summary rows loaded")
    intervals = _bootstrap_rate_intervals(
        cache_inputs,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    _apply_bootstrap_intervals(grid_rows, intervals)
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
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_cells": len(intervals),
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
            "generation_caches": cache_inputs,
            "checkpoint_labels": checkpoint_labels,
            "primary_feature": args.primary_feature,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
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
