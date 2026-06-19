from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = [
    "plan_name",
    "plan_path",
    "total_shards",
    "completed_shards",
    "in_flight_shards",
    "missing_shards",
    "expected_generations",
    "existing_generations",
    "status_existing_generations",
    "active_eta_hms",
    "missing_estimated_a100_hours",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize completion status across Phase 3 generation plan CSVs."
    )
    parser.add_argument(
        "--plan",
        action="append",
        required=True,
        help="Named generation plan in the form name=path.",
    )
    parser.add_argument(
        "--selection-status",
        action="append",
        default=[],
        help=(
            "Optional named live status CSV from slop-phase2-generation-status, "
            "in the form plan_name=path. May be repeated."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int_value(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _float_value(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _parse_named_paths(items: list[str], *, flag_name: str) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{flag_name} must use name=path syntax: {item!r}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise ValueError(f"{flag_name} must use non-empty name=path syntax: {item!r}")
        if name in specs:
            raise ValueError(f"duplicate name for {flag_name}: {name}")
        specs[name] = Path(raw_path)
    return specs


def _parse_plan_specs(items: list[str]) -> dict[str, Path]:
    return _parse_named_paths(items, flag_name="--plan")


def _parse_selection_status_specs(items: list[str]) -> dict[str, Path]:
    return _parse_named_paths(items, flag_name="--selection-status")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _stage_temperature_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("stage", "")), str(float(row.get("temperature", 0.0))))


def _summarize_plan(
    name: str,
    path: Path,
    *,
    selection_status_path: Path | None = None,
) -> dict[str, Any]:
    rows = _read_csv(path)
    status_rows = _read_csv(selection_status_path) if selection_status_path is not None else []
    status_by_key = {_stage_temperature_key(row): row for row in status_rows}
    total_shards = len(rows)
    completed_shards = sum(1 for row in rows if _bool(row.get("completed")))
    in_flight_keys = {
        _stage_temperature_key(row)
        for row in status_rows
        if _bool(row.get("process_alive")) and not _bool(row.get("completed"))
    }
    in_flight_shards = len(in_flight_keys)
    missing_rows = [
        row
        for row in rows
        if not _bool(row.get("completed")) and _stage_temperature_key(row) not in in_flight_keys
    ]
    existing_generations = 0
    status_existing_generations = 0
    for row in rows:
        plan_existing = _int_value(row.get("existing_generations"))
        status_row = status_by_key.get(_stage_temperature_key(row), {})
        status_existing = _int_value(status_row.get("existing_generations"))
        existing_generations += max(plan_existing, status_existing)
        status_existing_generations += status_existing
    active_etas = [
        str(row.get("eta_hms") or row.get("eta_avg_hms") or row.get("existing_eta_hms") or "")
        for row in status_rows
        if _bool(row.get("process_alive")) and not _bool(row.get("completed"))
    ]
    active_etas = [value for value in active_etas if value]
    return {
        "plan_name": name,
        "plan_path": str(path),
        "total_shards": total_shards,
        "completed_shards": completed_shards,
        "in_flight_shards": in_flight_shards,
        "missing_shards": len(missing_rows),
        "expected_generations": sum(_int_value(row.get("expected_generations")) for row in rows),
        "existing_generations": existing_generations,
        "status_existing_generations": status_existing_generations,
        "active_eta_hms": "; ".join(active_etas),
        "missing_estimated_a100_hours": sum(
            _float_value(row.get("estimated_a100_hours")) for row in missing_rows
        ),
    }


def summarize_phase3_generation_plans(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = _parse_plan_specs(args.plan)
    status_specs = _parse_selection_status_specs(args.selection_status)
    unknown_status_names = sorted(set(status_specs) - set(specs))
    if unknown_status_names:
        raise ValueError(
            "selection status names must match a --plan name; unknown: "
            + ", ".join(unknown_status_names)
        )
    return [
        _summarize_plan(name, path, selection_status_path=status_specs.get(name))
        for name, path in specs.items()
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 3 Generation Plan Status Summary",
        "",
        "| Plan | Completed shards | In-flight shards | Missing shards | Expected generations | Existing generations | Active ETA | Missing A100 h |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| `{plan}` | {completed} | {in_flight} | {missing} | {expected} | {existing} | {eta} | {hours:.2f} |".format(
                plan=row["plan_name"],
                completed=row["completed_shards"],
                in_flight=row["in_flight_shards"],
                missing=row["missing_shards"],
                expected=row["expected_generations"],
                existing=row["existing_generations"],
                eta=row["active_eta_hms"] or "",
                hours=float(row["missing_estimated_a100_hours"]),
            )
        )
    lines.extend(
        [
            "",
            "This summary is generated from the current generation plan CSVs; rerun after launching or completing shards.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = summarize_phase3_generation_plans(args)
    _write_csv(args.output, rows)
    _write_markdown(args.summary_output, rows)
    print(f"Wrote {len(rows)} Phase 3 generation plan summaries to {args.output}")
    return rows


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
