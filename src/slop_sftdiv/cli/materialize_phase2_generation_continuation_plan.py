from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.continue_phase2_generation_plan import (
    _with_generation_batch_size,
    _with_resume,
)
from slop_sftdiv.cli.plan_phase2_generation import OUTPUT_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a continuation generation plan while preserving output paths. "
            "Useful when a generated cache should be resumed with safer command settings."
        )
    )
    parser.add_argument("--input-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=None,
        help="Rewrite --generation-batch-size in each planned command.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append --resume to planned commands that do not already include it.",
    )
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


def materialize_phase2_generation_continuation_plan(
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.generation_batch_size is not None and args.generation_batch_size <= 0:
        raise ValueError("--generation-batch-size must be positive")
    rows = []
    for row in _read_rows(args.input_plan):
        command = str(row["command"])
        command = _with_generation_batch_size(command, args.generation_batch_size)
        if args.resume:
            command = _with_resume(command)
        existing_generations = _jsonl_records(Path(row["generations_output"]))
        completed = (
            Path(row["summary_output"]).exists()
            and existing_generations >= int(row["expected_generations"])
        )
        updated = dict(row)
        updated["command"] = command
        updated["existing_generations"] = existing_generations
        updated["completed"] = completed
        rows.append(updated)
    _write_csv(args.output, rows)
    _write_markdown(args.summary_output, rows, resume=args.resume, batch_size=args.generation_batch_size)
    print(f"Wrote {len(rows)} continuation plan rows to {args.output}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    resume: bool,
    batch_size: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_hours = sum(float(row["estimated_a100_hours"]) for row in rows if not _bool(row["completed"]))
    command_note_parts = ["preserves original output paths"]
    if batch_size is not None:
        command_note_parts.append(f"uses generation batch size {batch_size}")
    if resume:
        command_note_parts.append("adds `--resume`")
    lines = [
        "# Phase 2 Generation Continuation Plan",
        "",
        "Continuation plan: " + ", ".join(command_note_parts) + ".",
        "",
        f"Missing estimated A100-hours: `{total_hours:.2f}`",
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
    materialize_phase2_generation_continuation_plan(args)


if __name__ == "__main__":
    main()
