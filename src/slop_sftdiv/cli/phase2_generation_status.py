from __future__ import annotations

import argparse
import csv
import json
import os
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
    "generations_output",
    "summary_output",
    "log_output",
]


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


def _load_selection(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_row(path: Path) -> dict[str, Any]:
    payload = _load_selection(path)
    pid = payload.get("pid")
    pid_int = int(pid) if pid not in (None, "") else None
    generations_output = Path(payload["generations_output"])
    summary_output = Path(payload["summary_output"])
    existing_generations = _jsonl_records(generations_output)
    expected_generations = int(payload["expected_generations"])
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
        "generations_output": str(generations_output),
        "summary_output": str(summary_output),
        "log_output": payload.get("log_output", ""),
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
            "generations={existing}/{expected}".format(
                stage=row["stage"],
                temperature=row["temperature"],
                alive=row["process_alive"],
                completed=row["completed"],
                existing=row["existing_generations"],
                expected=row["expected_generations"],
            )
        )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    run_phase2_generation_status(args)


if __name__ == "__main__":
    main()
