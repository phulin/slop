from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.assemble_phase2_generation_grid import (
    run_assemble_phase2_generation_grid,
)
from slop_sftdiv.cli.analyze_phase2_compounding import run_analyze_phase2_compounding
from slop_sftdiv.cli.phase2_generation_status import (
    _command_option_int,
    _format_hms,
    _latest_log_progress,
    _remaining_seconds,
)


DEFAULT_FEATURES = [
    "contrastive_negation",
    "rule_of_three_approx",
    "slop_lexicon",
    "stock_openers",
    "stock_closers",
    "stock_openers_closers",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2 post-shard DPO scale and compounding analyses."
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--baseline-dpo-summary",
        type=Path,
        default=Path(
            "artifacts/phase2/generations/"
            "olmo3_dpo_promptpkg5000_free_run_512prompt_8comp_t1_batched1024_summary.csv"
        ),
    )
    parser.add_argument(
        "--propensity-grid",
        type=Path,
        default=Path(
            "artifacts/phase2/analysis/"
            "olmo3_promptpkg1024_slop_neutral_common_normalized_stage_grid.csv"
        ),
    )
    parser.add_argument(
        "--scale-grid-output",
        type=Path,
        default=Path(
            "artifacts/phase2/analysis/"
            "olmo3_dpo_generation_scale_grid_target_shape_512_vs_1024prompt_8comp_t1_1024.csv"
        ),
    )
    parser.add_argument(
        "--scale-comparison-output",
        type=Path,
        default=Path(
            "artifacts/phase2/analysis/"
            "olmo3_dpo_generation_scale_comparison_target_shape_512_vs_1024prompt_8comp_t1_1024.csv"
        ),
    )
    parser.add_argument(
        "--scale-summary-output",
        type=Path,
        default=Path(
            "artifacts/phase2/analysis/"
            "olmo3_dpo_generation_scale_grid_target_shape_512_vs_1024prompt_8comp_t1_1024_summary.md"
        ),
    )
    parser.add_argument(
        "--compounding-output",
        type=Path,
        default=Path(
            "artifacts/phase2/analysis/"
            "olmo3_dpo_generation_compounding_target_shape_1024prompt_8comp_t1_1024_tf1024.csv"
        ),
    )
    parser.add_argument(
        "--compounding-summary-output",
        type=Path,
        default=Path(
            "artifacts/phase2/analysis/"
            "olmo3_dpo_generation_compounding_target_shape_1024prompt_8comp_t1_1024_tf1024_summary.md"
        ),
    )
    parser.add_argument(
        "--compounding-plot-output",
        type=Path,
        default=Path(
            "artifacts/phase2/analysis/"
            "olmo3_dpo_generation_compounding_target_shape_1024prompt_8comp_t1_1024_tf1024_realized_af.svg"
        ),
    )
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--primary-feature", default="slop_lexicon")
    parser.add_argument("--window-tokens", type=int, default=32)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run analyses. Without this, only checks readiness and prints planned outputs.",
    )
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return parser


def _jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _process_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _load_selection(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selection_status(selection_path: Path) -> dict[str, Any]:
    selection = _load_selection(selection_path)
    pid = selection.get("pid")
    pid_int = int(pid) if pid not in (None, "") else None
    generations_output = Path(selection["generations_output"])
    summary_output = Path(selection["summary_output"])
    log_output_raw = selection.get("log_output", "")
    log_output = Path(log_output_raw) if log_output_raw else None
    expected_generations = int(selection["expected_generations"])
    existing_generations = _jsonl_records(generations_output)
    latest_log_progress = _latest_log_progress(log_output)
    latest_log_prompts = latest_log_progress["prompts"]
    latest_log_elapsed_seconds = latest_log_progress["elapsed_seconds"]
    avg_seconds_per_prompt = (
        latest_log_elapsed_seconds / latest_log_prompts
        if latest_log_elapsed_seconds is not None and latest_log_prompts
        else None
    )
    completions_per_prompt = _command_option_int(selection.get("command"), "--completions-per-prompt")
    latest_log_generations_estimate = (
        latest_log_prompts * completions_per_prompt
        if latest_log_prompts is not None and completions_per_prompt is not None
        else None
    )
    expected_prompts = (
        expected_generations / completions_per_prompt
        if completions_per_prompt not in (None, 0)
        else None
    )
    eta_seconds = _remaining_seconds(
        expected_prompts=expected_prompts,
        latest_log_prompts=latest_log_prompts,
        seconds_per_prompt=latest_log_progress["seconds_per_prompt"],
    )
    eta_avg_seconds = _remaining_seconds(
        expected_prompts=expected_prompts,
        latest_log_prompts=latest_log_prompts,
        seconds_per_prompt=avg_seconds_per_prompt,
    )
    completed = summary_output.exists() and existing_generations >= expected_generations
    return {
        "selection": selection,
        "pid": pid_int,
        "process_alive": _process_alive(pid_int),
        "generations_output": generations_output,
        "summary_output": summary_output,
        "expected_generations": expected_generations,
        "existing_generations": existing_generations,
        "latest_log_prompts": latest_log_prompts,
        "latest_log_generations_estimate": latest_log_generations_estimate,
        "latest_log_elapsed_seconds": latest_log_elapsed_seconds,
        "latest_log_avg_seconds_per_prompt": avg_seconds_per_prompt,
        "latest_log_seconds_per_prompt": latest_log_progress["seconds_per_prompt"],
        "eta_seconds": eta_seconds,
        "eta_hms": _format_hms(eta_seconds),
        "eta_avg_seconds": eta_avg_seconds,
        "eta_avg_hms": _format_hms(eta_avg_seconds),
        "completed": completed,
    }


def _wait_for_completion(args: argparse.Namespace) -> dict[str, Any]:
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.timeout_seconds < 0:
        raise ValueError("--timeout-seconds cannot be negative")

    started = time.monotonic()
    while True:
        status = _selection_status(args.selection)
        if status["completed"]:
            return status
        if not args.wait:
            raise ValueError(
                "generation shard is not complete: "
                f"{status['existing_generations']}/{status['expected_generations']} rows"
            )
        if not status["process_alive"] and status["existing_generations"] < status["expected_generations"]:
            raise RuntimeError(
                "generation process is not alive and shard is incomplete: "
                f"{status['existing_generations']}/{status['expected_generations']} rows"
            )
        if args.timeout_seconds and time.monotonic() - started >= args.timeout_seconds:
            raise TimeoutError(
                "timed out waiting for generation shard completion: "
                f"{status['existing_generations']}/{status['expected_generations']} rows"
            )
        print(
            "waiting for shard completion: "
            f"{status['existing_generations']}/{status['expected_generations']} rows; "
            f"log_prompts={status['latest_log_prompts'] or 'n/a'}; "
            f"log_generation_estimate={status['latest_log_generations_estimate'] or 'n/a'}; "
            f"eta={status['eta_hms'] or 'n/a'}; "
            f"eta_avg={status['eta_avg_hms'] or 'n/a'}",
            flush=True,
        )
        time.sleep(args.poll_seconds)


def _features(args: argparse.Namespace) -> list[str]:
    return args.feature or DEFAULT_FEATURES


def _scale_args(args: argparse.Namespace, status: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        generation_summary=[
            f"a_dpo512={args.baseline_dpo_summary}",
            f"b_dpo1024={status['summary_output']}",
        ],
        checkpoint_label=[
            "a_dpo512=DPO 512 prompts x 8",
            "b_dpo1024=DPO 1024 prompts x 8",
        ],
        primary_feature=args.primary_feature,
        output=args.scale_grid_output,
        comparison_output=args.scale_comparison_output,
        summary_output=args.scale_summary_output,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name="stage2-phase2-olmo3-dpo-generation-scale-512-vs-1024-t1",
        wandb_group="phase2-analysis",
        wandb_job_type="assemble-phase2-generation-grid",
        wandb_tag=["dpo-scale", "post-shard"],
        wandb_mode=args.wandb_mode,
    )


def _compounding_args(args: argparse.Namespace, status: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        generation_cache=[f"dpo={status['generations_output']}"],
        feature=_features(args),
        window_tokens=args.window_tokens,
        propensity_grid=args.propensity_grid,
        output=args.compounding_output,
        summary_output=args.compounding_summary_output,
        plot_output=args.compounding_plot_output,
        primary_feature=args.primary_feature,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name="stage2-phase2-olmo3-dpo-generation-compounding-target-shape-1024prompt-t1-tf1024",
        wandb_group="phase2-compounding",
        wandb_job_type="compounding-analysis",
        wandb_tag=["dpo-scale", "post-shard"],
        wandb_mode=args.wandb_mode,
    )


def run_phase2_post_shard_analysis(args: argparse.Namespace) -> dict[str, Any]:
    status = _wait_for_completion(args)
    planned = {
        "completed": status["completed"],
        "existing_generations": status["existing_generations"],
        "expected_generations": status["expected_generations"],
        "latest_log_prompts": status["latest_log_prompts"],
        "latest_log_generations_estimate": status["latest_log_generations_estimate"],
        "latest_log_elapsed_seconds": status["latest_log_elapsed_seconds"],
        "latest_log_avg_seconds_per_prompt": status["latest_log_avg_seconds_per_prompt"],
        "latest_log_seconds_per_prompt": status["latest_log_seconds_per_prompt"],
        "eta_seconds": status["eta_seconds"],
        "eta_hms": status["eta_hms"],
        "eta_avg_seconds": status["eta_avg_seconds"],
        "eta_avg_hms": status["eta_avg_hms"],
        "generations_output": str(status["generations_output"]),
        "summary_output": str(status["summary_output"]),
        "scale_grid_output": str(args.scale_grid_output),
        "scale_comparison_output": str(args.scale_comparison_output),
        "scale_summary_output": str(args.scale_summary_output),
        "compounding_output": str(args.compounding_output),
        "compounding_summary_output": str(args.compounding_summary_output),
        "compounding_plot_output": str(args.compounding_plot_output),
    }
    if not args.execute:
        print("post-shard analysis ready; rerun with --execute to write outputs")
        print(json.dumps(planned, indent=2, sort_keys=True))
        return planned

    scale_summary = run_assemble_phase2_generation_grid(_scale_args(args, status))
    compounding_rows = run_analyze_phase2_compounding(_compounding_args(args, status))
    planned.update(
        {
            "scale_rows": scale_summary["grid_rows"],
            "scale_comparison_rows": scale_summary["comparison_rows"],
            "compounding_rows": len(compounding_rows),
        }
    )
    print(json.dumps(planned, indent=2, sort_keys=True))
    return planned


def main() -> None:
    args = build_parser().parse_args()
    run_phase2_post_shard_analysis(args)


if __name__ == "__main__":
    main()
