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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select and optionally launch one shard from a Phase 2 generation plan."
    )
    parser.add_argument("--plan", type=Path, required=True, help="CSV from slop-plan-phase2-generation.")
    parser.add_argument("--stage", default=None, help="Optional stage to select, e.g. dpo.")
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature to select.")
    parser.add_argument(
        "--max-estimated-a100-hours",
        type=float,
        default=8.0,
        help="Refuse to launch shards above this estimate unless --force is set.",
    )
    parser.add_argument("--allow-completed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the selected command. Without this, only prints the command.",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="With --execute, launch in the background and return immediately.",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=None,
        help="Log file for --detach stdout/stderr. Defaults next to --selection-output.",
    )
    parser.add_argument("--selection-output", type=Path, default=None)
    return parser


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_plan(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _temperature_matches(row: dict[str, Any], temperature: float | None) -> bool:
    if temperature is None:
        return True
    return abs(float(row["temperature"]) - temperature) < 1e-9


def _select_row(
    rows: list[dict[str, Any]],
    *,
    stage: str | None,
    temperature: float | None,
    allow_completed: bool,
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if (stage is None or row["stage"] == stage)
        and _temperature_matches(row, temperature)
        and (allow_completed or not _bool(row["completed"]))
    ]
    if not candidates:
        filters = {
            "stage": stage,
            "temperature": temperature,
            "allow_completed": allow_completed,
        }
        raise ValueError(f"no matching generation shard in plan for filters {filters}")
    return candidates[0]


def _selection_payload(row: dict[str, Any], *, executed: bool) -> dict[str, Any]:
    return {
        "stage": row["stage"],
        "temperature": float(row["temperature"]),
        "estimated_a100_hours": float(row["estimated_a100_hours"]),
        "expected_generations": int(row["expected_generations"]),
        "generations_output": row["generations_output"],
        "summary_output": row["summary_output"],
        "completed": _bool(row["completed"]),
        "executed": executed,
        "command": row["command"],
    }


def _default_log_output(selection_output: Path | None, row: dict[str, Any]) -> Path:
    if selection_output is not None:
        return selection_output.with_suffix(".log")
    generations_path = Path(row["generations_output"])
    return generations_path.with_suffix(".log")


def _write_selection(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_launch_phase2_generation_shard(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_estimated_a100_hours <= 0:
        raise ValueError("--max-estimated-a100-hours must be positive")
    if args.detach and not args.execute:
        raise ValueError("--detach requires --execute")
    rows = _load_plan(args.plan)
    row = _select_row(
        rows,
        stage=args.stage,
        temperature=args.temperature,
        allow_completed=args.allow_completed,
    )
    estimated_hours = float(row["estimated_a100_hours"])
    if estimated_hours > args.max_estimated_a100_hours and not args.force:
        raise ValueError(
            f"selected shard is estimated at {estimated_hours:.2f} A100-hours, "
            f"above limit {args.max_estimated_a100_hours:.2f}; pass --force to allow"
        )
    print(row["command"])
    if args.execute:
        if args.detach:
            log_output = args.log_output or _default_log_output(args.selection_output, row)
            log_output.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_output.open("ab")
            process = subprocess.Popen(
                shlex.split(row["command"]),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            payload = _selection_payload(row, executed=True)
            payload.update(
                {
                    "detached": True,
                    "pid": process.pid,
                    "log_output": str(log_output),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "launcher_pid": os.getpid(),
                }
            )
            if args.selection_output is not None:
                _write_selection(args.selection_output, payload)
            return payload
        subprocess.run(shlex.split(row["command"]), check=True)
    payload = _selection_payload(row, executed=args.execute)
    payload["detached"] = False
    if args.selection_output is not None:
        _write_selection(args.selection_output, payload)
    return payload


def main() -> None:
    args = build_parser().parse_args()
    run_launch_phase2_generation_shard(args)


if __name__ == "__main__":
    main()
