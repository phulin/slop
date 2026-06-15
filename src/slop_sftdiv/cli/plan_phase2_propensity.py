from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.plan_phase2_generation import (
    StageSpec,
    _jsonl_records,
    _package_tag,
    _shell_quote,
    _stages,
)
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


OUTPUT_COLUMNS = [
    "stage",
    "model",
    "model_revision",
    "sample_size",
    "features",
    "normalization_feature",
    "max_opportunities",
    "max_token_start_opportunities",
    "max_prefix_tokens",
    "mass_mode",
    "fixed_prefix_tokens",
    "opportunity_batch_size",
    "sequence_cache",
    "cache_branch_batch_size",
    "estimated_opportunities",
    "estimated_seconds",
    "estimated_a100_hours",
    "opportunities_output",
    "summary_output",
    "reference_subset_summary_output",
    "completed",
    "existing_opportunities",
    "command",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan Phase 2 teacher-forced propensity shards.")
    parser.add_argument(
        "--prompt-package",
        type=Path,
        required=True,
        help="Phase 2 prompt/reference package JSONL to pass to slop-teacher-forced-propensity.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase2/propensity"))
    parser.add_argument("--artifact-prefix", default="olmo3")
    parser.add_argument("--package-tag", default=None)
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--normalization-feature", default=None)
    parser.add_argument("--reference-subset", action="append", default=[])
    parser.add_argument("--max-opportunities", type=int, default=512)
    parser.add_argument("--max-token-start-opportunities", type=int, default=128)
    parser.add_argument("--max-prefix-tokens", type=int, default=1024)
    parser.add_argument(
        "--mass-mode",
        default="sequence",
        choices=["sequence", "first_token"],
        help="Teacher-forced mass mode to pass through.",
    )
    parser.add_argument("--fixed-prefix-tokens", type=int, default=256)
    parser.add_argument("--opportunity-batch-size", type=int, default=16)
    parser.add_argument(
        "--sequence-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --sequence-cache/--no-sequence-cache through to the scorer.",
    )
    parser.add_argument("--cache-branch-batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        metavar="STAGE=MODEL[@REVISION]",
        help=(
            "Stage/model pair. Add @REVISION for Hugging Face branches/tags, "
            "for example sft=HuggingFaceTB/SmolLM3-3B-checkpoints@it-SFT. "
            "Defaults to the four OLMo 3 Instruct stages."
        ),
    )
    parser.add_argument(
        "--opportunities-per-sec-estimate",
        type=float,
        default=25.0,
        help="Rough throughput estimate for A100-hour accounting.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-propensity")
    parser.add_argument("--wandb-job-type", default="propensity-plan")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _feature_list(features: list[str]) -> list[str]:
    return list(dict.fromkeys(features or ["slop_lexicon", "neutral_common_controls"]))


def _feature_tag(features: list[str]) -> str:
    if features == ["slop_lexicon", "neutral_common_controls"]:
        return "slop-vs-neutral"
    if len(features) == 1:
        return features[0].replace("_", "-")
    return "features-" + "-".join(feature.replace("_", "-") for feature in features[:3])


def _csv_records(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = sum(1 for _row in csv.reader(handle))
    return max(0, rows - 1)


def _command(
    *,
    model: str,
    model_revision: str | None,
    prompt_package: Path,
    sample_size: int,
    seed: int,
    features: list[str],
    normalization_feature: str | None,
    reference_subsets: list[str],
    max_opportunities: int,
    max_token_start_opportunities: int,
    max_prefix_tokens: int,
    mass_mode: str,
    fixed_prefix_tokens: int,
    opportunity_batch_size: int,
    sequence_cache: bool,
    cache_branch_batch_size: int,
    dtype: str,
    torch_compile: bool,
    bootstrap_samples: int,
    bootstrap_seed: int,
    opportunities_output: Path,
    summary_output: Path,
    reference_subset_summary_output: Path | None,
    wandb_run_name: str,
) -> str:
    parts = [
        "uv",
        "run",
        "slop-teacher-forced-propensity",
        "--model",
        model,
        "--input",
        str(prompt_package),
        "--sample-size",
        str(sample_size),
        "--seed",
        str(seed),
    ]
    if model_revision:
        parts.extend(["--model-revision", model_revision])
    for feature in features:
        parts.extend(["--feature", feature])
    if normalization_feature:
        parts.extend(["--normalization-feature", normalization_feature])
    for subset in reference_subsets:
        parts.extend(["--reference-subset", subset])
    parts.extend(
        [
            "--max-opportunities",
            str(max_opportunities),
            "--max-token-start-opportunities",
            str(max_token_start_opportunities),
            "--max-prefix-tokens",
            str(max_prefix_tokens),
            "--mass-mode",
            mass_mode,
            "--fixed-prefix-tokens",
            str(fixed_prefix_tokens),
            "--opportunity-batch-size",
            str(opportunity_batch_size),
            "--cache-branch-batch-size",
            str(cache_branch_batch_size),
            "--dtype",
            dtype,
            "--bootstrap-samples",
            str(bootstrap_samples),
            "--bootstrap-seed",
            str(bootstrap_seed),
            "--output",
            str(opportunities_output),
            "--summary-output",
            str(summary_output),
            "--wandb-run-name",
            wandb_run_name,
            "--wandb-mode",
            "online",
        ]
    )
    if reference_subset_summary_output is not None:
        parts.extend(["--reference-subset-summary-output", str(reference_subset_summary_output)])
    parts.append("--sequence-cache" if sequence_cache else "--no-sequence-cache")
    parts.append("--torch-compile" if torch_compile else "--no-torch-compile")
    return " ".join(_shell_quote(part) for part in parts)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_hours = sum(float(row["estimated_a100_hours"]) for row in rows if not row["completed"])
    lines = [
        "# Phase 2 Teacher-Forced Propensity Plan",
        "",
        f"Missing estimated A100-hours: `{total_hours:.2f}`",
        "",
        "| Stage | Features | Est opportunities | Est A100 h | Complete |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {stage} | `{features}` | {opps} | {hours:.2f} | {done} |".format(
                stage=row["stage"],
                features=row["features"],
                opps=row["estimated_opportunities"],
                hours=float(row["estimated_a100_hours"]),
                done="yes" if row["completed"] else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_for_stage(
    *,
    stage: str,
    stage_spec: StageSpec,
    args: argparse.Namespace,
    package_tag: str,
    planned_docs: int,
    features: list[str],
) -> dict[str, Any]:
    feature_tag = _feature_tag(features)
    shard_tag = (
        f"{args.artifact_prefix}_{stage}_{package_tag}_{feature_tag}_"
        f"promptpkg{args.sample_size}_cached_branch{args.cache_branch_batch_size}_sequence"
    )
    opportunities_output = args.output_dir / f"{shard_tag}_opportunities.csv"
    summary_output = args.output_dir / f"{shard_tag}_summary.csv"
    reference_subset_summary_output = (
        args.output_dir / f"{shard_tag}_reference_subset_summary.csv"
        if args.reference_subset
        else None
    )
    estimated_opportunities = planned_docs * args.max_opportunities * len(features)
    estimated_seconds = estimated_opportunities / args.opportunities_per_sec_estimate
    existing_opportunities = _csv_records(opportunities_output) or 0
    completed = opportunities_output.exists() and summary_output.exists() and existing_opportunities > 0
    wandb_run_name = (
        f"stage2-phase2-{args.artifact_prefix}-{stage}-{package_tag}-{feature_tag}-"
        f"teacher-forced-{args.sample_size}prompt-branch{args.cache_branch_batch_size}"
    )
    return {
        "stage": stage,
        "model": stage_spec.model,
        "model_revision": stage_spec.revision or "",
        "sample_size": args.sample_size,
        "features": ",".join(features),
        "normalization_feature": args.normalization_feature or "",
        "max_opportunities": args.max_opportunities,
        "max_token_start_opportunities": args.max_token_start_opportunities,
        "max_prefix_tokens": args.max_prefix_tokens,
        "mass_mode": args.mass_mode,
        "fixed_prefix_tokens": args.fixed_prefix_tokens,
        "opportunity_batch_size": args.opportunity_batch_size,
        "sequence_cache": args.sequence_cache,
        "cache_branch_batch_size": args.cache_branch_batch_size,
        "estimated_opportunities": estimated_opportunities,
        "estimated_seconds": estimated_seconds,
        "estimated_a100_hours": estimated_seconds / 3600.0,
        "opportunities_output": str(opportunities_output),
        "summary_output": str(summary_output),
        "reference_subset_summary_output": str(reference_subset_summary_output or ""),
        "completed": completed,
        "existing_opportunities": existing_opportunities,
        "command": _command(
            model=stage_spec.model,
            model_revision=stage_spec.revision,
            prompt_package=args.prompt_package,
            sample_size=args.sample_size,
            seed=args.seed,
            features=features,
            normalization_feature=args.normalization_feature,
            reference_subsets=args.reference_subset,
            max_opportunities=args.max_opportunities,
            max_token_start_opportunities=args.max_token_start_opportunities,
            max_prefix_tokens=args.max_prefix_tokens,
            mass_mode=args.mass_mode,
            fixed_prefix_tokens=args.fixed_prefix_tokens,
            opportunity_batch_size=args.opportunity_batch_size,
            sequence_cache=args.sequence_cache,
            cache_branch_batch_size=args.cache_branch_batch_size,
            dtype=args.dtype,
            torch_compile=args.torch_compile,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            opportunities_output=opportunities_output,
            summary_output=summary_output,
            reference_subset_summary_output=reference_subset_summary_output,
            wandb_run_name=wandb_run_name,
        ),
    }


def run_plan_phase2_propensity(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.max_opportunities <= 0:
        raise ValueError("--max-opportunities must be positive")
    if args.max_token_start_opportunities <= 0:
        raise ValueError("--max-token-start-opportunities must be positive")
    if args.max_prefix_tokens <= 0:
        raise ValueError("--max-prefix-tokens must be positive")
    if args.opportunity_batch_size <= 0:
        raise ValueError("--opportunity-batch-size must be positive")
    if args.cache_branch_batch_size <= 0:
        raise ValueError("--cache-branch-batch-size must be positive")
    if args.opportunities_per_sec_estimate <= 0:
        raise ValueError("--opportunities-per-sec-estimate must be positive")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    features = _feature_list(args.feature)
    stages = _stages(args.stage)
    package_tag = _package_tag(args.prompt_package, args.package_tag)
    package_rows = _jsonl_records(args.prompt_package)
    planned_docs = min(args.sample_size, package_rows) if package_rows is not None else args.sample_size
    rows = [
        _row_for_stage(
            stage=stage,
            stage_spec=stage_spec,
            args=args,
            package_tag=package_tag,
            planned_docs=planned_docs,
            features=features,
        )
        for stage, stage_spec in stages.items()
    ]
    _write_csv(args.output, rows)
    if args.summary_output is not None:
        _write_markdown(args.summary_output, rows)
    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "propensity-plan", *args.wandb_tag],
        config={
            "prompt_package": str(args.prompt_package),
            "package_rows": package_rows,
            "planned_docs": planned_docs,
            "sample_size": args.sample_size,
            "features": features,
            "normalization_feature": args.normalization_feature,
            "reference_subsets": args.reference_subset,
            "stages": {
                stage: {"model": spec.model, "revision": spec.revision}
                for stage, spec in stages.items()
            },
            "max_opportunities": args.max_opportunities,
            "max_token_start_opportunities": args.max_token_start_opportunities,
            "cache_branch_batch_size": args.cache_branch_batch_size,
            "opportunities_per_sec_estimate": args.opportunities_per_sec_estimate,
        },
    )
    try:
        missing_hours = sum(float(row["estimated_a100_hours"]) for row in rows if not row["completed"])
        run.log(
            {
                "propensity_plan/shards": len(rows),
                "propensity_plan/completed_shards": sum(1 for row in rows if row["completed"]),
                "propensity_plan/missing_shards": sum(1 for row in rows if not row["completed"]),
                "propensity_plan/missing_estimated_a100_hours": missing_hours,
            }
        )
        log_summary_table(run, "phase2_propensity_plan", rows)
    finally:
        run.finish()
    return rows


def main() -> None:
    args = build_parser().parse_args()
    rows = run_plan_phase2_propensity(args)
    print(f"Wrote {len(rows)} propensity shard plans to {args.output}")


if __name__ == "__main__":
    main()
