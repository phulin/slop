from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.phase2_generation_status import _process_alive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Conservatively continue a Phase 2 generation plan by launching the "
            "next safe missing shard when no tracked detached shard is active."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-prefix",
        default="phase2_generation_shard",
        help="Prefix for generated selection JSON files and active selection discovery.",
    )
    parser.add_argument(
        "--max-estimated-a100-hours",
        type=float,
        default=8.0,
        help="Refuse to launch shards above this estimate unless --force is set.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch the next safe shard. Without this, only reports the decision.",
    )
    parser.add_argument(
        "--allow-partial-retry",
        action="store_true",
        help=(
            "Allow selecting a missing shard whose generations file already has rows. "
            "The generation command truncates output at startup, so this is off by default."
        ),
    )
    parser.add_argument(
        "--generation-batch-size-override",
        type=int,
        default=None,
        help="Optionally replace --generation-batch-size in the selected command before launch.",
    )
    parser.add_argument(
        "--priority-temperature",
        action="append",
        type=float,
        default=[],
        help=(
            "Prefer missing shards at this temperature before other temperatures. "
            "May be repeated; plan order is preserved within equal priorities."
        ),
    )
    parser.add_argument("--decision-output", type=Path, default=None)
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


def _load_plan(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _selection_paths(selection_dir: Path, prefix: str) -> list[Path]:
    if not selection_dir.exists():
        return []
    return sorted(selection_dir.glob(f"{prefix}*.json"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _active_selection(path: Path) -> dict[str, Any] | None:
    payload = _load_json(path)
    if not payload.get("generations_output") or not payload.get("summary_output"):
        return None
    pid = payload.get("pid")
    pid_int = int(pid) if pid not in (None, "") else None
    if not _process_alive(pid_int):
        return None
    generations = Path(str(payload.get("generations_output", "")))
    summary = Path(str(payload.get("summary_output", "")))
    expected = int(payload.get("expected_generations", 0))
    existing = _jsonl_records(generations)
    completed = summary.exists() and existing >= expected
    if completed:
        return None
    return {
        "selection_path": str(path),
        "stage": payload.get("stage", ""),
        "temperature": payload.get("temperature", ""),
        "pid": pid_int or "",
        "existing_generations": existing,
        "expected_generations": expected,
    }


def _is_complete(row: dict[str, Any]) -> bool:
    expected = int(row["expected_generations"])
    generations = Path(row["generations_output"])
    summary = Path(row["summary_output"])
    return summary.exists() and _jsonl_records(generations) >= expected


def _stage_temp_tag(row: dict[str, Any]) -> str:
    temperature = str(float(row["temperature"])).replace(".", "p")
    return f"{row['stage']}_t{temperature}"


def _selection_payload(row: dict[str, Any], *, executed: bool) -> dict[str, Any]:
    return {
        "stage": row["stage"],
        "temperature": float(row["temperature"]),
        "estimated_a100_hours": float(row["estimated_a100_hours"]),
        "expected_generations": int(row["expected_generations"]),
        "generations_output": row["generations_output"],
        "summary_output": row["summary_output"],
        "completed": _is_complete(row),
        "executed": executed,
        "detached": executed,
        "command": row["command"],
    }


def _with_resume(command: str) -> str:
    parts = shlex.split(command)
    if "--resume" in parts:
        return command
    return command + " --resume"


def _with_generation_batch_size(command: str, batch_size: int | None) -> str:
    if batch_size is None:
        return command
    if batch_size <= 0:
        raise ValueError("--generation-batch-size-override must be positive")
    parts = shlex.split(command)
    rewritten: list[str] = []
    index = 0
    replaced = False
    while index < len(parts):
        part = parts[index]
        if part == "--generation-batch-size" and index + 1 < len(parts):
            rewritten.extend([part, str(batch_size)])
            index += 2
            replaced = True
            continue
        prefix = "--generation-batch-size="
        if part.startswith(prefix):
            rewritten.append(prefix + str(batch_size))
            index += 1
            replaced = True
            continue
        rewritten.append(part)
        index += 1
    if not replaced:
        rewritten.extend(["--generation-batch-size", str(batch_size)])
    return shlex.join(rewritten)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prioritized_rows(
    rows: list[dict[str, Any]],
    *,
    priority_temperatures: list[float],
) -> list[dict[str, Any]]:
    if not priority_temperatures:
        return rows
    priority_by_temperature = {
        float(temperature): index for index, temperature in enumerate(priority_temperatures)
    }
    fallback_priority = len(priority_by_temperature)
    return [
        row
        for _index, row in sorted(
            enumerate(rows),
            key=lambda indexed_row: (
                priority_by_temperature.get(
                    float(indexed_row[1].get("temperature", 0.0)),
                    fallback_priority,
                ),
                indexed_row[0],
            ),
        )
    ]


def _find_next_safe_row(
    rows: list[dict[str, Any]],
    *,
    allow_partial_retry: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for row in rows:
        if _is_complete(row):
            continue
        existing = _jsonl_records(Path(row["generations_output"]))
        if existing > 0:
            if not allow_partial_retry:
                return None, {
                    "action": "blocked_partial_shard",
                    "stage": row["stage"],
                    "temperature": float(row["temperature"]),
                    "existing_generations": existing,
                    "generations_output": row["generations_output"],
                    "message": (
                        "missing shard has partial output and no active tracked process; "
                        "use --allow-partial-retry only if resuming it is intentional"
                    ),
                }
            row = dict(row)
            row["command"] = _with_resume(str(row["command"]))
        return row, None
    return None, {"action": "complete", "message": "no missing shards remain"}


def continue_phase2_generation_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_estimated_a100_hours <= 0:
        raise ValueError("--max-estimated-a100-hours must be positive")
    active = [
        row
        for path in _selection_paths(args.selection_dir, args.selection_prefix)
        if (row := _active_selection(path)) is not None
    ]
    if active:
        decision = {"action": "active_shard_running", "active": active}
        _maybe_write_decision(args.decision_output, decision)
        print(json.dumps(decision, sort_keys=True))
        return decision

    rows = _prioritized_rows(
        _load_plan(args.plan),
        priority_temperatures=args.priority_temperature,
    )
    row, stop_decision = _find_next_safe_row(rows, allow_partial_retry=args.allow_partial_retry)
    if stop_decision is not None:
        _maybe_write_decision(args.decision_output, stop_decision)
        print(json.dumps(stop_decision, sort_keys=True))
        return stop_decision
    if row is None:
        raise AssertionError("unreachable: row and stop decision are both empty")
    if args.generation_batch_size_override is not None:
        row = dict(row)
        row["command"] = _with_generation_batch_size(
            str(row["command"]),
            args.generation_batch_size_override,
        )

    estimated_hours = float(row["estimated_a100_hours"])
    if estimated_hours > args.max_estimated_a100_hours and not args.force:
        raise ValueError(
            f"selected shard is estimated at {estimated_hours:.2f} A100-hours, "
            f"above limit {args.max_estimated_a100_hours:.2f}; pass --force to allow"
        )

    selection_path = args.selection_dir / f"{args.selection_prefix}_{_stage_temp_tag(row)}.json"
    log_path = selection_path.with_suffix(".log")
    payload = _selection_payload(row, executed=args.execute)
    resume_initial_generations = _jsonl_records(Path(row["generations_output"]))
    payload.update({"selection_path": str(selection_path), "log_output": str(log_path)})
    if "--resume" in shlex.split(str(row["command"])):
        payload["resume_initial_generations"] = resume_initial_generations
    if args.execute:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("ab")
        process = subprocess.Popen(
            shlex.split(row["command"]),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        payload.update(
            {
                "pid": process.pid,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "launcher_pid": os.getpid(),
            }
        )
    _write_json(selection_path, payload)
    decision = {
        "action": "launched" if args.execute else "would_launch",
        "selection_path": str(selection_path),
        "stage": row["stage"],
        "temperature": float(row["temperature"]),
        "estimated_a100_hours": estimated_hours,
        "expected_generations": int(row["expected_generations"]),
        "executed": args.execute,
    }
    if args.execute:
        decision["pid"] = payload["pid"]
    _maybe_write_decision(args.decision_output, decision)
    print(json.dumps(decision, sort_keys=True))
    return decision


def _maybe_write_decision(path: Path | None, decision: dict[str, Any]) -> None:
    if path is None:
        return
    _write_json(path, decision)


def main() -> None:
    args = build_parser().parse_args()
    continue_phase2_generation_plan(args)


if __name__ == "__main__":
    main()
