from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.plan_phase2_generation import OUTPUT_COLUMNS, _shell_quote


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a SGLang-backed generation plan from a standard Phase 2 "
            "generation plan while preserving output paths and expected row counts."
        )
    )
    parser.add_argument("--input-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--sglang-python",
        default=".venvs/sglang-cu128/bin/python",
        help="Python executable in the SGLang sidecar environment.",
    )
    parser.add_argument(
        "--sglang-script",
        type=Path,
        default=Path("scripts/benchmark_sglang_generation.py"),
        help="SGLang generation script to run.",
    )
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=0,
        help=(
            "Number of prompts per SGLang generate call. Use 0 to send all pending "
            "prompts at once."
        ),
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument(
        "--disable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --disable-cuda-graph to the SGLang harness.",
    )
    parser.add_argument(
        "--ld-library-path",
        action="append",
        default=None,
        help=(
            "Library path entry to pass through env LD_LIBRARY_PATH. Defaults to "
            "the extracted local libnuma path plus /usr/lib/x86_64-linux-gnu."
        ),
    )
    parser.add_argument("--torch-cuda-arch-list", default="8.0")
    parser.add_argument(
        "--sampling-strategy",
        default="hash_reservoir",
        choices=["random", "first", "hash_reservoir"],
        help="Prompt sampling strategy passed to the SGLang harness.",
    )
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", default="error")
    parser.add_argument(
        "--feature-text-mode",
        default="raw",
        choices=["raw", "final_answer"],
        help="Feature extraction text mode to pass to the SGLang generation harness.",
    )
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    return parser


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _command_option(command: str, option: str) -> str | None:
    import shlex

    parts = shlex.split(command)
    for index, part in enumerate(parts):
        if part == option and index + 1 < len(parts):
            return parts[index + 1]
        prefix = option + "="
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _base_input(row: dict[str, Any]) -> str:
    value = _command_option(str(row["command"]), "--input")
    if not value:
        raise ValueError("input plan command does not include --input")
    return value


def _sglang_command(
    row: dict[str, Any],
    *,
    sglang_python: str,
    sglang_script: Path,
    context_length: int,
    prompt_batch_size: int,
    tp_size: int,
    mem_fraction_static: float,
    sampling_strategy: str,
    ignore_eos: bool,
    trust_remote_code: bool,
    log_level: str,
    feature_text_mode: str,
    wandb_mode: str,
    attention_backend: str | None,
    disable_cuda_graph: bool,
    ld_library_paths: list[str],
    torch_cuda_arch_list: str | None,
) -> str:
    if row.get("model_revision"):
        raise ValueError(
            "SGLang plan materialization currently supports rows without model_revision; "
            f"got {row['model_revision']!r} for stage {row['stage']}"
        )
    if _bool(row.get("apply_chat_template")):
        raise ValueError(
            "SGLang plan materialization currently supports plain prompt rows only; "
            f"stage {row['stage']} requested apply_chat_template"
        )

    parts = ["env"]
    if ld_library_paths:
        parts.append("LD_LIBRARY_PATH=" + ":".join(ld_library_paths))
    if torch_cuda_arch_list:
        parts.append("TORCH_CUDA_ARCH_LIST=" + torch_cuda_arch_list)
    parts.extend(
        [
        sglang_python,
        str(sglang_script),
        "--model",
        str(row["model"]),
        "--input",
        _base_input(row),
        "--sample-size",
        str(row["sample_size"]),
        "--seed",
        _command_option(str(row["command"]), "--seed") or "1729",
        "--sampling-strategy",
        sampling_strategy,
        "--temperature",
        str(row["temperature"]),
        "--top-p",
        str(row["top_p"]),
        "--max-new-tokens",
        str(row["max_new_tokens"]),
        "--completions-per-prompt",
        str(row["completions_per_prompt"]),
        "--prompt-batch-size",
        str(prompt_batch_size),
        "--max-prompt-tokens",
        _command_option(str(row["command"]), "--max-prompt-tokens") or "1024",
        "--context-length",
        str(context_length),
        "--dtype",
        _command_option(str(row["command"]), "--dtype") or "bfloat16",
        "--tp-size",
        str(tp_size),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--log-level",
        log_level,
        "--feature-text-mode",
        feature_text_mode,
        "--generations-output",
        str(row["generations_output"]),
        "--summary-output",
        str(row["summary_output"]),
        "--wandb-run-name",
        (
            "stage2-phase2-sglang-"
            f"{row['stage']}-promptpkg{row['sample_size']}-"
            f"{row['completions_per_prompt']}comp-t{float(row['temperature']):g}-"
            f"{row['max_new_tokens']}tok"
        ),
        "--wandb-mode",
        wandb_mode,
        ]
    )
    if attention_backend:
        parts.extend(["--attention-backend", attention_backend])
    if disable_cuda_graph:
        parts.append("--disable-cuda-graph")
    else:
        parts.append("--no-disable-cuda-graph")
    if ignore_eos:
        parts.append("--ignore-eos")
    else:
        parts.append("--no-ignore-eos")
    if trust_remote_code:
        parts.append("--trust-remote-code")
    else:
        parts.append("--no-trust-remote-code")
    return " ".join(_shell_quote(part) for part in parts)


def materialize_sglang_generation_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive")
    if args.prompt_batch_size < 0:
        raise ValueError("--prompt-batch-size must be non-negative")
    if args.tp_size <= 0:
        raise ValueError("--tp-size must be positive")
    if not 0 < args.mem_fraction_static <= 1:
        raise ValueError("--mem-fraction-static must be in (0, 1]")

    ld_library_paths = args.ld_library_path or [
        str(
            Path.cwd()
            / ".venvs/sglang-cu128/sysroot/extracted/usr/lib/x86_64-linux-gnu"
        ),
        "/usr/lib/x86_64-linux-gnu",
    ]
    rows: list[dict[str, Any]] = []
    for row in _read_rows(args.input_plan):
        existing_generations = _jsonl_records(Path(row["generations_output"]))
        completed = (
            Path(row["summary_output"]).exists()
            and existing_generations >= int(row["expected_generations"])
        )
        updated = dict(row)
        updated["command"] = _sglang_command(
            row,
            sglang_python=args.sglang_python,
            sglang_script=args.sglang_script,
            context_length=args.context_length,
            prompt_batch_size=args.prompt_batch_size,
            tp_size=args.tp_size,
            mem_fraction_static=args.mem_fraction_static,
            sampling_strategy=args.sampling_strategy,
            ignore_eos=args.ignore_eos,
            trust_remote_code=args.trust_remote_code,
            log_level=args.log_level,
            feature_text_mode=args.feature_text_mode,
            wandb_mode=args.wandb_mode,
            attention_backend=args.attention_backend,
            disable_cuda_graph=args.disable_cuda_graph,
            ld_library_paths=ld_library_paths,
            torch_cuda_arch_list=args.torch_cuda_arch_list,
        )
        updated["existing_generations"] = existing_generations
        updated["completed"] = completed
        rows.append(updated)
    _write_csv(args.output, rows)
    _write_markdown(args.summary_output, rows, args=args)
    print(f"Wrote {len(rows)} SGLang generation plan rows to {args.output}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_hours = sum(float(row["estimated_a100_hours"]) for row in rows if not _bool(row["completed"]))
    lines = [
        "# SGLang Generation Plan",
        "",
        f"SGLang Python: `{args.sglang_python}`",
        f"SGLang script: `{args.sglang_script}`",
        f"Sampling strategy: `{args.sampling_strategy}`",
        f"Ignore EOS: `{args.ignore_eos}`",
        f"Context length: `{args.context_length}`",
        f"Prompt batch size: `{args.prompt_batch_size}`",
        f"Memory fraction static: `{args.mem_fraction_static}`",
        f"Attention backend: `{args.attention_backend}`",
        f"Disable CUDA graph: `{args.disable_cuda_graph}`",
        f"Feature text mode: `{args.feature_text_mode}`",
        "",
        f"Missing estimated A100-hours from source plan: `{total_hours:.2f}`",
        "",
        "| Stage | Temp | Expected gens | Existing gens | Est A100 h | Complete |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {temp:.3g} | {expected} | {existing} | {hours:.2f} | {done} |".format(
                stage=row["stage"],
                temp=float(row["temperature"]),
                expected=row["expected_generations"],
                existing=row["existing_generations"],
                hours=float(row["estimated_a100_hours"]),
                done="yes" if _bool(row["completed"]) else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    materialize_sglang_generation_plan(args)


if __name__ == "__main__":
    main()
