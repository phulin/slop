from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


DEFAULT_STAGES = {
    "base": "allenai/Olmo-3-1025-7B",
    "sft": "allenai/Olmo-3-7B-Instruct-SFT",
    "dpo": "allenai/Olmo-3-7B-Instruct-DPO",
    "final": "allenai/Olmo-3-7B-Instruct",
}


@dataclass(frozen=True)
class StageSpec:
    model: str
    revision: str | None = None

OUTPUT_COLUMNS = [
    "stage",
    "model",
    "model_revision",
    "temperature",
    "top_p",
    "sample_size",
    "completions_per_prompt",
    "max_new_tokens",
    "apply_chat_template",
    "chat_template_kwargs_json",
    "missing_chat_template",
    "expected_generations",
    "expected_generated_tokens",
    "estimated_seconds",
    "estimated_a100_hours",
    "generations_output",
    "summary_output",
    "completed",
    "existing_generations",
    "command",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan Phase 2 free-running generation shards.")
    parser.add_argument(
        "--prompt-package",
        type=Path,
        required=True,
        help="Phase 2 prompt package JSONL to pass to slop-free-running-emission.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase2/generations"))
    parser.add_argument("--artifact-prefix", default="olmo3")
    parser.add_argument("--package-tag", default=None)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--completions-per-prompt", type=int, default=8)
    parser.add_argument("--temperature", action="append", type=float, default=[])
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--generation-batch-size", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Include --apply-chat-template in planned slop-free-running-emission commands.",
    )
    parser.add_argument(
        "--chat-template-kwargs-json",
        default=None,
        help="Optional JSON object to pass through to --chat-template-kwargs-json.",
    )
    parser.add_argument(
        "--missing-chat-template",
        default="error",
        choices=["error", "plain"],
        help="Pass through to slop-free-running-emission.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=True)
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
        "--tokens-per-sec-estimate",
        type=float,
        default=356.0,
        help="Throughput estimate for rough A100-hour accounting.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-generation")
    parser.add_argument("--wandb-job-type", default="generation-plan")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _parse_stage(raw_value: str) -> tuple[str, StageSpec]:
    if "=" not in raw_value:
        raise ValueError(f"invalid --stage {raw_value!r}; expected STAGE=MODEL[@REVISION]")
    stage, model_spec = raw_value.split("=", 1)
    stage = stage.strip()
    model_spec = model_spec.strip()
    if not stage or not model_spec:
        raise ValueError(f"invalid --stage {raw_value!r}; expected STAGE=MODEL[@REVISION]")
    if "@" in model_spec:
        model, revision = model_spec.rsplit("@", 1)
        model = model.strip()
        revision = revision.strip()
        if not model or not revision:
            raise ValueError(f"invalid --stage {raw_value!r}; expected STAGE=MODEL[@REVISION]")
        return stage, StageSpec(model=model, revision=revision)
    return stage, StageSpec(model=model_spec)


def _stages(raw_stages: list[str]) -> dict[str, StageSpec]:
    if not raw_stages:
        return {stage: StageSpec(model=model) for stage, model in DEFAULT_STAGES.items()}
    stages = [_parse_stage(item) for item in raw_stages]
    return dict(stages)


def _jsonl_records(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _package_tag(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    stem = path.stem
    if "prompt_package_5000" in stem:
        return "promptpkg5000"
    if "prompt_package_512" in stem:
        return "promptpkg512"
    if "prompt_package_128" in stem:
        return "promptpkg128"
    return stem.replace("_", "-")


def _temperature_tag(temperature: float) -> str:
    text = f"{temperature:g}".replace(".", "")
    return f"t{text}"


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:=+-")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _command(
    *,
    model: str,
    model_revision: str | None,
    prompt_package: Path,
    sample_size: int,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    completions_per_prompt: int,
    generation_batch_size: int,
    max_prompt_tokens: int,
    apply_chat_template: bool,
    chat_template_kwargs_json: str | None,
    missing_chat_template: str,
    dtype: str,
    torch_compile: bool,
    generations_output: Path,
    summary_output: Path,
    wandb_run_name: str,
) -> str:
    parts = [
        "uv",
        "run",
        "slop-free-running-emission",
        "--model",
        model,
        "--input",
        str(prompt_package),
        "--sample-size",
        str(sample_size),
        "--seed",
        str(seed),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--max-new-tokens",
        str(max_new_tokens),
        "--completions-per-prompt",
        str(completions_per_prompt),
        "--generation-batch-size",
        str(generation_batch_size),
        "--max-prompt-tokens",
        str(max_prompt_tokens),
        "--dtype",
        dtype,
        "--generations-output",
        str(generations_output),
        "--summary-output",
        str(summary_output),
        "--wandb-run-name",
        wandb_run_name,
        "--wandb-mode",
        "online",
    ]
    if model_revision:
        parts.extend(["--model-revision", model_revision])
    if apply_chat_template:
        parts.append("--apply-chat-template")
    if chat_template_kwargs_json:
        parts.extend(["--chat-template-kwargs-json", chat_template_kwargs_json])
    parts.extend(["--missing-chat-template", missing_chat_template])
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
        "# Phase 2 Generation Plan",
        "",
        f"Missing estimated A100-hours: `{total_hours:.2f}`",
        "",
        "| Stage | Temp | Expected gens | Est A100 h | Complete |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {temp:.3g} | {gens} | {hours:.2f} | {done} |".format(
                stage=row["stage"],
                temp=float(row["temperature"]),
                gens=row["expected_generations"],
                hours=float(row["estimated_a100_hours"]),
                done="yes" if row["completed"] else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_plan_phase2_generation(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.completions_per_prompt <= 0:
        raise ValueError("--completions-per-prompt must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.tokens_per_sec_estimate <= 0:
        raise ValueError("--tokens-per-sec-estimate must be positive")
    temperatures = args.temperature or [0.0, 0.7, 1.0]
    stages = _stages(args.stage)
    package_tag = _package_tag(args.prompt_package, args.package_tag)
    package_rows = _jsonl_records(args.prompt_package)
    planned_prompts = min(args.sample_size, package_rows) if package_rows is not None else args.sample_size
    rows: list[dict[str, Any]] = []
    for stage, stage_spec in stages.items():
        for temperature in temperatures:
            temp_tag = _temperature_tag(temperature)
            shard_tag = (
                f"{args.artifact_prefix}_{stage}_{package_tag}_free_run_"
                f"{args.sample_size}prompt_{args.completions_per_prompt}comp_"
                f"{temp_tag}_batched{args.generation_batch_size}"
            )
            generations_output = args.output_dir / f"{shard_tag}.jsonl"
            summary_output = args.output_dir / f"{shard_tag}_summary.csv"
            expected_generations = planned_prompts * args.completions_per_prompt
            expected_generated_tokens = expected_generations * args.max_new_tokens
            estimated_seconds = expected_generated_tokens / args.tokens_per_sec_estimate
            existing_generations = _jsonl_records(generations_output) or 0
            completed = (
                generations_output.exists()
                and summary_output.exists()
                and existing_generations >= expected_generations
            )
            wandb_run_name = (
                f"stage2-phase2-{args.artifact_prefix}-{stage}-{package_tag}-"
                f"free-running-{args.sample_size}prompt-{args.completions_per_prompt}comp-"
                f"{temp_tag}-bench{args.max_new_tokens}"
            )
            rows.append(
                {
                    "stage": stage,
                    "model": stage_spec.model,
                    "model_revision": stage_spec.revision or "",
                    "temperature": temperature,
                    "top_p": args.top_p,
                    "sample_size": args.sample_size,
                    "completions_per_prompt": args.completions_per_prompt,
                    "max_new_tokens": args.max_new_tokens,
                    "apply_chat_template": args.apply_chat_template,
                    "chat_template_kwargs_json": args.chat_template_kwargs_json or "",
                    "missing_chat_template": args.missing_chat_template,
                    "expected_generations": expected_generations,
                    "expected_generated_tokens": expected_generated_tokens,
                    "estimated_seconds": estimated_seconds,
                    "estimated_a100_hours": estimated_seconds / 3600.0,
                    "generations_output": str(generations_output),
                    "summary_output": str(summary_output),
                    "completed": completed,
                    "existing_generations": existing_generations,
                    "command": _command(
                        model=stage_spec.model,
                        model_revision=stage_spec.revision,
                        prompt_package=args.prompt_package,
                        sample_size=args.sample_size,
                        seed=args.seed,
                        temperature=temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        completions_per_prompt=args.completions_per_prompt,
                        generation_batch_size=args.generation_batch_size,
                        max_prompt_tokens=args.max_prompt_tokens,
                        apply_chat_template=args.apply_chat_template,
                        chat_template_kwargs_json=args.chat_template_kwargs_json,
                        missing_chat_template=args.missing_chat_template,
                        dtype=args.dtype,
                        torch_compile=args.torch_compile,
                        generations_output=generations_output,
                        summary_output=summary_output,
                        wandb_run_name=wandb_run_name,
                    ),
                }
            )
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
        tags=["stage2", "phase2", "generation-plan", *args.wandb_tag],
        config={
            "prompt_package": str(args.prompt_package),
            "package_rows": package_rows,
            "planned_prompts": planned_prompts,
            "sample_size": args.sample_size,
            "completions_per_prompt": args.completions_per_prompt,
            "temperatures": temperatures,
            "stages": {
                stage: {"model": spec.model, "revision": spec.revision}
                for stage, spec in stages.items()
            },
            "max_new_tokens": args.max_new_tokens,
            "generation_batch_size": args.generation_batch_size,
            "apply_chat_template": args.apply_chat_template,
            "chat_template_kwargs_json": args.chat_template_kwargs_json,
            "missing_chat_template": args.missing_chat_template,
            "tokens_per_sec_estimate": args.tokens_per_sec_estimate,
        },
    )
    try:
        missing_hours = sum(float(row["estimated_a100_hours"]) for row in rows if not row["completed"])
        run.log(
            {
                "generation_plan/shards": len(rows),
                "generation_plan/completed_shards": sum(1 for row in rows if row["completed"]),
                "generation_plan/missing_shards": sum(1 for row in rows if not row["completed"]),
                "generation_plan/missing_estimated_a100_hours": missing_hours,
            }
        )
        log_summary_table(run, "phase2_generation_plan", rows)
    finally:
        run.finish()
    return rows


def main() -> None:
    args = build_parser().parse_args()
    rows = run_plan_phase2_generation(args)
    print(f"Wrote {len(rows)} generation shard plans to {args.output}")


if __name__ == "__main__":
    main()
