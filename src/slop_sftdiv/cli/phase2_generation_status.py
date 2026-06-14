from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = [
    "selection_path",
    "stage",
    "temperature",
    "pid",
    "process_alive",
    "completed",
    "existing_generations",
    "expected_generations",
    "latest_log_prompts",
    "latest_log_generations_estimate",
    "generations_output",
    "summary_output",
    "log_output",
]

PROMPT_PROGRESS_RE = re.compile(r":\s+(\d+)prompt\b")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect detached Phase 2 generation launch records.")
    parser.add_argument("--selection", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
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


def _command_option_int(command: str | None, option: str) -> int | None:
    if not command:
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    for index, part in enumerate(parts):
        if part == option and index + 1 < len(parts):
            try:
                return int(parts[index + 1])
            except ValueError:
                return None
        prefix = option + "="
        if part.startswith(prefix):
            try:
                return int(part[len(prefix) :])
            except ValueError:
                return None
    return None


def _latest_log_prompt_count(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = PROMPT_PROGRESS_RE.findall(text.replace("\r", "\n"))
    if not matches:
        return None
    return int(matches[-1])


def _load_selection(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_row(path: Path) -> dict[str, Any]:
    payload = _load_selection(path)
    pid = payload.get("pid")
    pid_int = int(pid) if pid not in (None, "") else None
    generations_output = Path(payload["generations_output"])
    summary_output = Path(payload["summary_output"])
    log_output_raw = payload.get("log_output", "")
    log_output = Path(log_output_raw) if log_output_raw else None
    existing_generations = _jsonl_records(generations_output)
    expected_generations = int(payload["expected_generations"])
    latest_log_prompts = _latest_log_prompt_count(log_output)
    completions_per_prompt = _command_option_int(payload.get("command"), "--completions-per-prompt")
    latest_log_generations_estimate = (
        latest_log_prompts * completions_per_prompt
        if latest_log_prompts is not None and completions_per_prompt is not None
        else None
    )
    completed = summary_output.exists() and existing_generations >= expected_generations
    return {
        "selection_path": str(path),
        "stage": payload["stage"],
        "temperature": payload["temperature"],
        "pid": pid_int or "",
        "process_alive": _process_alive(pid_int),
        "completed": completed,
        "existing_generations": existing_generations,
        "expected_generations": expected_generations,
        "latest_log_prompts": latest_log_prompts if latest_log_prompts is not None else "",
        "latest_log_generations_estimate": (
            latest_log_generations_estimate if latest_log_generations_estimate is not None else ""
        ),
        "generations_output": str(generations_output),
        "summary_output": str(summary_output),
        "log_output": log_output_raw,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_phase2_generation_status(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [_status_row(path) for path in args.selection]
    if args.output is not None:
        _write_csv(args.output, rows)
    for row in rows:
        print(
            "{stage} t={temperature}: alive={alive} completed={completed} "
            "generations={existing}/{expected} log_prompts={log_prompts}".format(
                stage=row["stage"],
                temperature=row["temperature"],
                alive=row["process_alive"],
                completed=row["completed"],
                existing=row["existing_generations"],
                expected=row["expected_generations"],
                log_prompts=row["latest_log_prompts"] or "n/a",
            )
        )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    run_phase2_generation_status(args)


if __name__ == "__main__":
    main()
