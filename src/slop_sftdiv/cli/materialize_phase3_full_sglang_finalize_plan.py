from __future__ import annotations

import argparse
import csv
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Any


STAGE_ORDER = ("base", "sft", "dpo", "final")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the post-generation finalization commands for the full "
            "OLMo SGLang Phase 3 grid."
        )
    )
    parser.add_argument("--generation-plan", type=Path, required=True)
    parser.add_argument("--output-script", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--artifact-prefix",
        default="artifacts/phase3/analysis/olmo3_full_sglang_5000prompt_8comp_3temp",
    )
    parser.add_argument("--primary-temperature", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--feature-rate", action="append", default=[])
    parser.add_argument("--weighted-pretrain-baseline", action="append", default=[])
    parser.add_argument("--propensity-grid", action="append", default=[])
    parser.add_argument(
        "--compounding-propensity-grid",
        type=Path,
        default=None,
        help=(
            "Single propensity grid for expected-vs-observed compounding. Defaults "
            "to the first --propensity-grid when omitted."
        ),
    )
    parser.add_argument("--denominator-support", action="append", default=[])
    parser.add_argument("--preference-analysis", action="append", default=[])
    parser.add_argument("--teacher-forced-stage-effects", action="append", default=[])
    parser.add_argument("--free-run-stage-effects", action="append", default=[])
    parser.add_argument("--right-spectrum", type=Path, default=None)
    parser.add_argument("--wandb-mode", default="disabled", choices=["online", "offline", "disabled"])
    return parser


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _stage_key(stage: str) -> int:
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)


def _temp_tag(temperature: float) -> str:
    return f"t{temperature:g}".replace(".", "p")


def _q(value: Any) -> str:
    return shlex.quote(str(value))


def _command(parts: list[Any]) -> str:
    return " ".join(_q(part) for part in parts)


def _paths_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _missing_paths(paths: list[str | Path]) -> list[Path]:
    return [Path(path) for path in paths if not Path(path).exists()]


def _prerequisite_status_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    compounding_grid = (
        [args.compounding_propensity_grid]
        if args.compounding_propensity_grid is not None
        else ([args.propensity_grid[0]] if args.propensity_grid else [])
    )
    groups: list[tuple[str, list[str | Path]]] = [
        ("feature_rates", list(args.feature_rate)),
        ("weighted_pretrain_baselines", list(args.weighted_pretrain_baseline)),
        ("propensity_grids", list(args.propensity_grid)),
        ("compounding_propensity_grid", compounding_grid),
        ("denominator_support", list(args.denominator_support)),
        ("preference_analysis", list(args.preference_analysis)),
        ("teacher_forced_stage_effects", list(args.teacher_forced_stage_effects)),
        ("free_run_stage_effects", list(args.free_run_stage_effects)),
        ("right_spectrum", [] if args.right_spectrum is None else [args.right_spectrum]),
    ]
    return [
        {
            "name": name,
            "count": len(paths),
            "missing": _missing_paths(paths),
        }
        for name, paths in groups
    ]


def _has_missing_prerequisites(
    prerequisite_rows: list[dict[str, Any]],
    names: set[str],
) -> bool:
    return any(row["name"] in names and bool(row["missing"]) for row in prerequisite_rows)


def _row_status(row: dict[str, Any]) -> dict[str, Any]:
    generations_path = Path(row["generations_output"])
    summary_path = Path(row["summary_output"])
    expected = int(float(row["expected_generations"]))
    existing = _jsonl_records(generations_path)
    completed = summary_path.exists() and existing >= expected
    return {
        **row,
        "temperature_float": float(row["temperature"]),
        "expected_generations_int": expected,
        "existing_generations_int": existing,
        "summary_exists": summary_path.exists(),
        "completed_bool": completed,
    }


def _group_by_temperature(rows: list[dict[str, Any]]) -> dict[float, list[dict[str, Any]]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["temperature"])].append(row)
    for temp_rows in grouped.values():
        temp_rows.sort(key=lambda row: _stage_key(str(row["stage"])))
    return dict(sorted(grouped.items()))


def _temperature_complete(rows: list[dict[str, Any]]) -> bool:
    stages = {str(row["stage"]) for row in rows}
    return set(STAGE_ORDER).issubset(stages) and all(_bool(row["completed_bool"]) for row in rows)


def _generation_grid_paths(prefix: str, temperature: float) -> dict[str, Path]:
    tag = _temp_tag(temperature)
    stem = f"{prefix}_generation_stage_grid_5000prompt_8comp_{tag}_1024"
    return {
        "grid": Path(f"{stem}.csv"),
        "comparison": Path(f"{stem}_primary_feature_comparison.csv"),
        "summary": Path(f"{stem}_summary.md"),
    }


def _compounding_paths(prefix: str, temperature: float) -> dict[str, Path]:
    tag = _temp_tag(temperature)
    stem = f"{prefix}_compounding_analysis_5000prompt_8comp_{tag}_1024"
    return {
        "table": Path(f"{stem}.csv"),
        "summary": Path(f"{stem}_summary.md"),
        "plot": Path(f"{stem}_realized_af.svg"),
        "counter_cache": Path(f"{stem}_cluster_counts.jsonl"),
    }


def _spectrum_paths(prefix: str, temperature: float) -> dict[str, Path]:
    tag = _temp_tag(temperature)
    stem = f"{prefix}_amplification_spectrum_5000prompt_8comp_{tag}_1024"
    return {
        "spectrum": Path(f"{stem}.csv"),
        "spectrum_summary": Path(f"{stem}_summary.md"),
        "classification": Path(f"{prefix}_feature_classification_5000prompt_8comp_{tag}_1024.csv"),
        "classification_summary": Path(
            f"{prefix}_feature_classification_5000prompt_8comp_{tag}_1024_summary.md"
        ),
        "aligned": Path(f"{prefix}_vs_smollm3_fullgrid_{tag}_aligned.csv"),
        "correlations": Path(f"{prefix}_vs_smollm3_fullgrid_{tag}_correlations.csv"),
        "comparison_summary": Path(f"{prefix}_vs_smollm3_fullgrid_{tag}_summary.md"),
    }


def _generation_grid_command(
    rows: list[dict[str, Any]],
    *,
    output_paths: dict[str, Path],
    bootstrap_samples: int,
    bootstrap_seed: int,
    wandb_mode: str,
) -> str:
    parts: list[Any] = ["uv", "run", "slop-assemble-phase2-generation-grid"]
    for row in rows:
        stage = row["stage"]
        parts.extend(["--generation-summary", f"{stage}={row['summary_output']}"])
    for row in rows:
        stage = row["stage"]
        parts.extend(["--generation-cache", f"{stage}={row['generations_output']}"])
    parts.extend(
        [
            "--bootstrap-samples",
            bootstrap_samples,
            "--bootstrap-seed",
            bootstrap_seed,
            "--output",
            output_paths["grid"],
            "--comparison-output",
            output_paths["comparison"],
            "--summary-output",
            output_paths["summary"],
            "--wandb-mode",
            wandb_mode,
        ]
    )
    return _command(parts)


def _compounding_command(
    rows: list[dict[str, Any]],
    *,
    output_paths: dict[str, Path],
    propensity_grid: str | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
    wandb_mode: str,
) -> str:
    parts: list[Any] = ["uv", "run", "slop-analyze-phase2-compounding"]
    for row in rows:
        parts.extend(["--generation-cache", f"{row['stage']}={row['generations_output']}"])
    if propensity_grid is not None:
        parts.extend(["--propensity-grid", propensity_grid])
    parts.extend(
        [
            "--counter-cache-output",
            output_paths["counter_cache"],
            "--window-tokens",
            32,
            "--output",
            output_paths["table"],
            "--summary-output",
            output_paths["summary"],
            "--plot-output",
            output_paths["plot"],
            "--bootstrap-samples",
            bootstrap_samples,
            "--bootstrap-seed",
            bootstrap_seed,
            "--wandb-mode",
            wandb_mode,
        ]
    )
    return _command(parts)


def _spectrum_command(
    *,
    output_paths: dict[str, Path],
    generation_grid: Path,
    compounding: Path,
    feature_rates: list[str],
    weighted_pretrain_baselines: list[str],
    propensity_grids: list[str],
    denominator_support: list[str],
    wandb_mode: str,
) -> str:
    parts: list[Any] = ["uv", "run", "slop-assemble-amplification-spectrum"]
    for path in feature_rates:
        parts.extend(["--feature-rate", path])
    for path in weighted_pretrain_baselines:
        parts.extend(["--weighted-pretrain-baseline", path])
    for path in propensity_grids:
        parts.extend(["--propensity-grid", path])
    for path in denominator_support:
        parts.extend(["--denominator-support", path])
    parts.extend(
        [
            "--generation-grid",
            generation_grid,
            "--compounding",
            compounding,
            "--output",
            output_paths["spectrum"],
            "--summary-output",
            output_paths["spectrum_summary"],
            "--wandb-mode",
            wandb_mode,
        ]
    )
    return _command(parts)


def _classification_command(
    *,
    output_paths: dict[str, Path],
    preference_analysis: list[str],
    teacher_forced_stage_effects: list[str],
    free_run_stage_effects: list[str],
    wandb_mode: str,
) -> str:
    parts: list[Any] = [
        "uv",
        "run",
        "slop-classify-amplification-spectrum",
        "--spectrum",
        output_paths["spectrum"],
    ]
    for path in preference_analysis:
        parts.extend(["--preference-analysis", path])
    for path in teacher_forced_stage_effects:
        parts.extend(["--teacher-forced-stage-effects", path])
    for path in free_run_stage_effects:
        parts.extend(["--free-run-stage-effects", path])
    parts.extend(
        [
            "--output",
            output_paths["classification"],
            "--summary-output",
            output_paths["classification_summary"],
            "--wandb-mode",
            wandb_mode,
        ]
    )
    return _command(parts)


def _cross_ladder_command(
    *,
    output_paths: dict[str, Path],
    right_spectrum: Path,
    wandb_mode: str,
) -> str:
    parts: list[Any] = [
        "uv",
        "run",
        "slop-compare-phase3-ladders",
        "--left-spectrum",
        output_paths["spectrum"],
        "--left-label",
        "OLMo3 full SGLang",
        "--right-spectrum",
        right_spectrum,
        "--right-label",
        "SmolLM3 no_think",
        "--aligned-output",
        output_paths["aligned"],
        "--correlation-output",
        output_paths["correlations"],
        "--summary-output",
        output_paths["comparison_summary"],
        "--wandb-mode",
        wandb_mode,
    ]
    return _command(parts)


def materialize_finalize_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    rows = [_row_status(row) for row in _read_rows(args.generation_plan)]
    by_temperature = _group_by_temperature(rows)
    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by slop-materialize-phase3-full-sglang-finalize-plan.",
        "# Commands for incomplete temperature slices are intentionally commented out.",
        "",
    ]
    summary_rows: list[dict[str, Any]] = []
    stage_status_rows: list[dict[str, Any]] = []
    prerequisite_rows = _prerequisite_status_rows(args)
    enabled_commands = 0
    primary_paths: dict[str, Path] | None = None
    primary_generation_grid: Path | None = None
    primary_compounding: Path | None = None
    compounding_propensity_grid = (
        str(args.compounding_propensity_grid)
        if args.compounding_propensity_grid is not None
        else (args.propensity_grid[0] if args.propensity_grid else None)
    )
    for temperature, temp_rows in by_temperature.items():
        complete = _temperature_complete(temp_rows)
        existing = sum(int(row["existing_generations_int"]) for row in temp_rows)
        expected = sum(int(row["expected_generations_int"]) for row in temp_rows)
        tag = _temp_tag(temperature)
        temperature_enabled_commands = 0
        script_lines.extend([f"# Temperature {temperature:g}: {existing}/{expected} rows", ""])
        for row in temp_rows:
            stage_status_rows.append(
                {
                    "temperature": temperature,
                    "stage": str(row["stage"]),
                    "complete": bool(row["completed_bool"]),
                    "existing_generations": int(row["existing_generations_int"]),
                    "expected_generations": int(row["expected_generations_int"]),
                    "summary_exists": bool(row["summary_exists"]),
                }
            )
        generation_paths = _generation_grid_paths(args.artifact_prefix, temperature)
        compounding_paths = _compounding_paths(args.artifact_prefix, temperature)
        generation_command = _generation_grid_command(
            temp_rows,
            output_paths=generation_paths,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            wandb_mode=args.wandb_mode,
        )
        compounding_command = _compounding_command(
            temp_rows,
            output_paths=compounding_paths,
            propensity_grid=compounding_propensity_grid,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            wandb_mode=args.wandb_mode,
        )
        if complete:
            if _paths_exist(list(generation_paths.values())):
                script_lines.append("# generation grid outputs already exist; command skipped")
            else:
                script_lines.append(generation_command)
                enabled_commands += 1
                temperature_enabled_commands += 1
            if _paths_exist(list(compounding_paths.values())):
                script_lines.append("# compounding outputs already exist; command skipped")
            elif _has_missing_prerequisites(prerequisite_rows, {"compounding_propensity_grid"}):
                script_lines.append("# compounding prerequisites missing; command disabled")
                script_lines.append("# " + compounding_command)
            else:
                script_lines.append(compounding_command)
                enabled_commands += 1
                temperature_enabled_commands += 1
            script_lines.append("")
        else:
            script_lines.extend(
                [
                    f"# incomplete: {existing}/{expected} rows; regenerate after shards finish",
                    "# " + generation_command,
                    "# " + compounding_command,
                    "",
                ]
            )
        if temperature == args.primary_temperature:
            primary_paths = _spectrum_paths(args.artifact_prefix, temperature)
            primary_generation_grid = generation_paths["grid"]
            primary_compounding = compounding_paths["table"]
        summary_rows.append(
            {
                "temperature": temperature,
                "tag": tag,
                "complete": complete,
                "existing_generations": existing,
                "expected_generations": expected,
                "enabled_commands": temperature_enabled_commands,
            }
        )

    primary_complete = any(
        row["temperature"] == args.primary_temperature and row["complete"]
        for row in summary_rows
    )
    script_lines.extend([f"# Primary spectrum temperature: {args.primary_temperature:g}", ""])
    if primary_paths and primary_generation_grid and primary_compounding and primary_complete:
        spectrum_outputs = [primary_paths["spectrum"], primary_paths["spectrum_summary"]]
        classification_outputs = [
            primary_paths["classification"],
            primary_paths["classification_summary"],
        ]
        cross_ladder_outputs = [
            primary_paths["aligned"],
            primary_paths["correlations"],
            primary_paths["comparison_summary"],
        ]
        if _paths_exist(spectrum_outputs):
            script_lines.append("# amplification spectrum outputs already exist; command skipped")
        elif _has_missing_prerequisites(
            prerequisite_rows,
            {
                "feature_rates",
                "weighted_pretrain_baselines",
                "propensity_grids",
                "compounding_propensity_grid",
                "denominator_support",
            },
        ):
            script_lines.append("# amplification spectrum prerequisites missing; command disabled")
            script_lines.append(
                "# "
                + _spectrum_command(
                    output_paths=primary_paths,
                    generation_grid=primary_generation_grid,
                    compounding=primary_compounding,
                    feature_rates=args.feature_rate,
                    weighted_pretrain_baselines=args.weighted_pretrain_baseline,
                    propensity_grids=args.propensity_grid,
                    denominator_support=args.denominator_support,
                    wandb_mode=args.wandb_mode,
                )
            )
        else:
            script_lines.append(
                _spectrum_command(
                    output_paths=primary_paths,
                    generation_grid=primary_generation_grid,
                    compounding=primary_compounding,
                    feature_rates=args.feature_rate,
                    weighted_pretrain_baselines=args.weighted_pretrain_baseline,
                    propensity_grids=args.propensity_grid,
                    denominator_support=args.denominator_support,
                    wandb_mode=args.wandb_mode,
                )
            )
            enabled_commands += 1
        if _paths_exist(classification_outputs):
            script_lines.append("# classification outputs already exist; command skipped")
        elif _has_missing_prerequisites(
            prerequisite_rows,
            {
                "feature_rates",
                "weighted_pretrain_baselines",
                "propensity_grids",
                "compounding_propensity_grid",
                "denominator_support",
                "preference_analysis",
                "teacher_forced_stage_effects",
                "free_run_stage_effects",
            },
        ):
            script_lines.append("# classification prerequisites missing; command disabled")
            script_lines.append(
                "# "
                + _classification_command(
                    output_paths=primary_paths,
                    preference_analysis=args.preference_analysis,
                    teacher_forced_stage_effects=args.teacher_forced_stage_effects,
                    free_run_stage_effects=args.free_run_stage_effects,
                    wandb_mode=args.wandb_mode,
                )
            )
        else:
            script_lines.append(
                _classification_command(
                    output_paths=primary_paths,
                    preference_analysis=args.preference_analysis,
                    teacher_forced_stage_effects=args.teacher_forced_stage_effects,
                    free_run_stage_effects=args.free_run_stage_effects,
                    wandb_mode=args.wandb_mode,
                )
            )
            enabled_commands += 1
        if args.right_spectrum is not None:
            if _paths_exist(cross_ladder_outputs):
                script_lines.append("# cross-ladder outputs already exist; command skipped")
            elif _has_missing_prerequisites(
                prerequisite_rows,
                {
                    "feature_rates",
                    "weighted_pretrain_baselines",
                    "propensity_grids",
                    "compounding_propensity_grid",
                    "denominator_support",
                    "right_spectrum",
                },
            ):
                script_lines.append("# cross-ladder prerequisites missing; command disabled")
                script_lines.append(
                    "# "
                    + _cross_ladder_command(
                        output_paths=primary_paths,
                        right_spectrum=args.right_spectrum,
                        wandb_mode=args.wandb_mode,
                    )
                )
            else:
                script_lines.append(
                    _cross_ladder_command(
                        output_paths=primary_paths,
                        right_spectrum=args.right_spectrum,
                        wandb_mode=args.wandb_mode,
                    )
                )
                enabled_commands += 1
    else:
        script_lines.extend(
            [
                "# incomplete primary temperature; spectrum/classification commands disabled",
                "# Rerun this materializer after the primary temperature has all four completed stages.",
            ]
        )
    script_lines.append("")
    args.output_script.parent.mkdir(parents=True, exist_ok=True)
    args.output_script.write_text("\n".join(script_lines), encoding="utf-8")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    _write_summary(
        args.summary_output,
        summary_rows=summary_rows,
        stage_status_rows=stage_status_rows,
        prerequisite_rows=prerequisite_rows,
        enabled_commands=enabled_commands,
        output_script=args.output_script,
        primary_temperature=args.primary_temperature,
    )
    return {
        "temperatures": len(summary_rows),
        "complete_temperatures": sum(1 for row in summary_rows if row["complete"]),
        "enabled_commands": enabled_commands,
        "missing_prerequisite_groups": sum(1 for row in prerequisite_rows if row["missing"]),
        "output_script": str(args.output_script),
        "summary_output": str(args.summary_output),
    }


def _write_summary(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    stage_status_rows: list[dict[str, Any]],
    prerequisite_rows: list[dict[str, Any]],
    enabled_commands: int,
    output_script: Path,
    primary_temperature: float,
) -> None:
    lines = [
        "# Full OLMo SGLang Finalization Plan",
        "",
        f"Output script: `{output_script}`",
        f"Primary temperature: `{primary_temperature:g}`",
        f"Enabled commands: `{enabled_commands}`",
        "",
        "| Temperature | Complete | Existing generations | Expected generations | Enabled commands |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {temperature:g} | {complete} | {existing} | {expected} | {enabled} |".format(
                temperature=float(row["temperature"]),
                complete="yes" if row["complete"] else "no",
                existing=int(row["existing_generations"]),
                expected=int(row["expected_generations"]),
                enabled=int(row["enabled_commands"]),
            )
        )
    lines.extend(
        [
            "",
            "## Stage Readiness",
            "",
            "| Temperature | Stage | Complete | Existing generations | Expected generations | Summary exists |",
            "|---:|---|---|---:|---:|---|",
        ]
    )
    for row in sorted(
        stage_status_rows,
        key=lambda item: (float(item["temperature"]), _stage_key(str(item["stage"]))),
    ):
        lines.append(
            "| {temperature:g} | {stage} | {complete} | {existing} | {expected} | {summary_exists} |".format(
                temperature=float(row["temperature"]),
                stage=row["stage"],
                complete="yes" if row["complete"] else "no",
                existing=int(row["existing_generations"]),
                expected=int(row["expected_generations"]),
                summary_exists="yes" if row["summary_exists"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Prerequisite Inputs",
            "",
            "| Group | Paths | Missing |",
            "|---|---:|---|",
        ]
    )
    for row in prerequisite_rows:
        missing = ", ".join(f"`{path}`" for path in row["missing"]) if row["missing"] else ""
        lines.append(
            "| {name} | {count} | {missing} |".format(
                name=row["name"],
                count=int(row["count"]),
                missing=missing,
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Incomplete temperature slices are commented out in the script.",
            "- Existing downstream outputs are skipped so rerunning the script only executes pending commands.",
            "- Commands with missing prerequisite inputs are commented out until the materializer is rerun with those inputs present.",
            "- Generation-grid assembly uses document/prompt-cluster bootstrap CIs.",
            "- Compounding commands materialize counter caches so CI reruns can reuse them.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    summary = materialize_finalize_plan(args)
    print(
        "Wrote full-grid finalization plan with "
        f"{summary['enabled_commands']} enabled commands to {summary['output_script']}"
    )


if __name__ == "__main__":
    main()
