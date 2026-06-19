from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
from datetime import datetime, timezone
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
    "latest_log_elapsed_seconds",
    "latest_log_avg_seconds_per_prompt",
    "latest_log_seconds_per_prompt",
    "eta_seconds",
    "eta_hms",
    "eta_avg_seconds",
    "eta_avg_hms",
    "elapsed_since_start_seconds",
    "existing_generations_per_second",
    "existing_eta_seconds",
    "existing_eta_hms",
    "generations_output",
    "summary_output",
    "log_output",
]

PROMPT_PROGRESS_RE = re.compile(r":\s+(\d+)prompt\b")
PROMPT_PROGRESS_DETAIL_RE = re.compile(
    r":\s+(?P<prompts>\d+)prompt\s+\[(?P<elapsed>[^,\]]+),\s+"
    r"(?P<seconds_per_prompt>[\d.]+)s/prompt\]"
)


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


def _duration_seconds(text: str) -> int | None:
    parts = text.strip().split(":")
    if not parts or not all(part.isdigit() for part in parts):
        return None
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return None


def _format_hms(seconds: float | int | None) -> str:
    if seconds is None:
        return ""
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_int = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds_int:02d}"


def _parse_started_at(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        started_at = datetime.fromisoformat(text)
    except ValueError:
        return None
    if started_at.tzinfo is None:
        return started_at.replace(tzinfo=timezone.utc)
    return started_at.astimezone(timezone.utc)


def _remaining_seconds(
    *,
    expected_prompts: float | None,
    latest_log_prompts: int | None,
    seconds_per_prompt: float | None,
) -> float | None:
    if expected_prompts is None or latest_log_prompts is None or seconds_per_prompt is None:
        return None
    return max(0.0, (expected_prompts - latest_log_prompts) * seconds_per_prompt)


def _latest_log_progress(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "prompts": None,
            "elapsed_seconds": None,
            "seconds_per_prompt": None,
        }
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    detail_matches = list(PROMPT_PROGRESS_DETAIL_RE.finditer(text))
    if detail_matches:
        match = detail_matches[-1]
        return {
            "prompts": int(match.group("prompts")),
            "elapsed_seconds": _duration_seconds(match.group("elapsed")),
            "seconds_per_prompt": float(match.group("seconds_per_prompt")),
        }
    matches = PROMPT_PROGRESS_RE.findall(text)
    return {
        "prompts": int(matches[-1]) if matches else None,
        "elapsed_seconds": None,
        "seconds_per_prompt": None,
    }


def _latest_log_prompt_count(path: Path | None) -> int | None:
    return _latest_log_progress(path)["prompts"]


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
    resume_initial_generations = int(payload.get("resume_initial_generations", 0) or 0)
    new_generations_since_start = max(0, existing_generations - resume_initial_generations)
    started_at = _parse_started_at(payload.get("started_at"))
    elapsed_since_start_seconds = None
    existing_generations_per_second = None
    existing_eta_seconds = None
    if started_at is not None:
        elapsed_since_start_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - started_at).total_seconds(),
        )
        if new_generations_since_start > 0 and elapsed_since_start_seconds > 0:
            existing_generations_per_second = new_generations_since_start / elapsed_since_start_seconds
            existing_eta_seconds = max(
                0.0,
                (expected_generations - existing_generations) / existing_generations_per_second,
            )
    latest_log_progress = _latest_log_progress(log_output)
    latest_log_prompts = latest_log_progress["prompts"]
    latest_log_elapsed_seconds = latest_log_progress["elapsed_seconds"]
    avg_seconds_per_prompt = (
        latest_log_elapsed_seconds / latest_log_prompts
        if latest_log_elapsed_seconds is not None and latest_log_prompts
        else None
    )
    completions_per_prompt = _command_option_int(payload.get("command"), "--completions-per-prompt")
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
        "latest_log_elapsed_seconds": (
            latest_log_elapsed_seconds if latest_log_elapsed_seconds is not None else ""
        ),
        "latest_log_avg_seconds_per_prompt": (
            avg_seconds_per_prompt if avg_seconds_per_prompt is not None else ""
        ),
        "latest_log_seconds_per_prompt": (
            latest_log_progress["seconds_per_prompt"]
            if latest_log_progress["seconds_per_prompt"] is not None
            else ""
        ),
        "eta_seconds": eta_seconds if eta_seconds is not None else "",
        "eta_hms": _format_hms(eta_seconds),
        "eta_avg_seconds": eta_avg_seconds if eta_avg_seconds is not None else "",
        "eta_avg_hms": _format_hms(eta_avg_seconds),
        "elapsed_since_start_seconds": (
            elapsed_since_start_seconds if elapsed_since_start_seconds is not None else ""
        ),
        "existing_generations_per_second": (
            existing_generations_per_second if existing_generations_per_second is not None else ""
        ),
        "existing_eta_seconds": existing_eta_seconds if existing_eta_seconds is not None else "",
        "existing_eta_hms": _format_hms(existing_eta_seconds),
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
            "generations={existing}/{expected} log_prompts={log_prompts} "
            "eta={eta} eta_avg={eta_avg} eta_existing={eta_existing}".format(
                stage=row["stage"],
                temperature=row["temperature"],
                alive=row["process_alive"],
                completed=row["completed"],
                existing=row["existing_generations"],
                expected=row["expected_generations"],
                log_prompts=row["latest_log_prompts"] or "n/a",
                eta=row["eta_hms"] or "n/a",
                eta_avg=row["eta_avg_hms"] or "n/a",
                eta_existing=row["existing_eta_hms"] or "n/a",
            )
        )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    run_phase2_generation_status(args)


if __name__ == "__main__":
    main()
