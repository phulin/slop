from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
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


def _write_selection(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_launch_phase2_generation_shard(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_estimated_a100_hours <= 0:
        raise ValueError("--max-estimated-a100-hours must be positive")
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
    payload = _selection_payload(row, executed=args.execute)
    if args.selection_output is not None:
        _write_selection(args.selection_output, payload)
    print(row["command"])
    if args.execute:
        subprocess.run(shlex.split(row["command"]), check=True)
    return payload


def main() -> None:
    args = build_parser().parse_args()
    run_launch_phase2_generation_shard(args)


if __name__ == "__main__":
    main()
