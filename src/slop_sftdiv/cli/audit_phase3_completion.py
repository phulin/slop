from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


STAGE_ORDER = ("base", "sft", "dpo", "final")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a machine-readable Phase 3 completion audit snapshot."
    )
    parser.add_argument("--generation-plan", type=Path, required=True)
    parser.add_argument("--finalization-summary", type=Path, required=True)
    parser.add_argument(
        "--integrity-audit",
        type=Path,
        help="Optional SGLang generation-cache integrity audit CSV to require as a passing gate.",
    )
    parser.add_argument(
        "--primary-temperature",
        type=float,
        default=1.0,
        help="Temperature slice used for primary spectrum/classification readiness.",
    )
    parser.add_argument(
        "--full-grid-scope",
        choices=("required", "optional"),
        default="required",
        help=(
            "Whether full original-scope OLMo generation/finalization gates are "
            "required for the completion headline or tracked as optional addendum evidence."
        ),
    )
    parser.add_argument(
        "--required-artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional required artifact to check for existence.",
    )
    parser.add_argument(
        "--optional-artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional optional/addendum artifact to track without blocking completion.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _stage_key(stage: str) -> int:
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)


def _parse_required_artifact(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--required-artifact must be NAME=PATH, got {spec!r}")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"--required-artifact has empty NAME in {spec!r}")
    return name, Path(raw_path)


def _enabled_finalization_commands(path: Path) -> int | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Enabled commands:"):
            value = line.split("`", 2)[1] if "`" in line else line.rsplit(":", 1)[-1]
            return int(value.strip(" `"))
    return None


def _missing_finalization_prerequisite_groups(path: Path) -> int | None:
    if not path.exists():
        return None
    in_section = False
    missing_groups = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if in_section:
                break
            in_section = line.strip() == "## Prerequisite Inputs"
            continue
        if not in_section or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in {"Group", "---"}:
            continue
        if cells[2]:
            missing_groups += 1
    return missing_groups if in_section else None


def _path_has_content(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _artifact_complete(path: Path) -> bool:
    if not _path_has_content(path):
        return False
    if path.suffix == ".jsonl":
        return _jsonl_records(path) > 0
    if path.suffix == ".csv":
        return bool(_read_csv_rows(path))
    return True


def _integrity_audit_status(path: Path) -> dict[str, Any]:
    if not _path_has_content(path):
        return {
            "requirement": "full_grid_sglang_cache_integrity",
            "status": "missing",
            "complete": False,
            "evidence": str(path),
            "existing": 0,
            "expected": 1,
            "remaining": 1,
        }
    rows = _read_csv_rows(path)
    if not rows:
        return {
            "requirement": "full_grid_sglang_cache_integrity",
            "status": "empty",
            "complete": False,
            "evidence": str(path),
            "existing": 0,
            "expected": 1,
            "remaining": 1,
        }
    passed = sum(1 for row in rows if _bool(row.get("passed", False)))
    complete = passed == len(rows)
    return {
        "requirement": "full_grid_sglang_cache_integrity",
        "status": "complete" if complete else "failed",
        "complete": complete,
        "evidence": str(path),
        "existing": passed,
        "expected": len(rows),
        "remaining": max(0, len(rows) - passed),
    }


def _severity_and_blocker(row: dict[str, Any]) -> tuple[str, str]:
    if _bool(row.get("optional", False)):
        if _bool(row["complete"]):
            return "none", ""
        return "optional", "optional/addendum evidence is incomplete"
    if _bool(row["complete"]):
        return "none", ""
    requirement = str(row["requirement"])
    if requirement == "full_olmo_sglang_generation_grid":
        return "critical", "full original-scope OLMo SGLang generation grid is incomplete"
    if requirement.startswith("primary_temperature_"):
        return "critical", "primary temperature four-stage generation slice is incomplete"
    if requirement == "full_grid_pending_finalization_commands":
        return "critical", "enabled finalization commands still need to run"
    if requirement == "full_grid_finalization_prerequisites":
        return "critical", "finalization prerequisite inputs are missing"
    if requirement == "full_grid_sglang_cache_integrity":
        return "critical", "SGLang generation cache integrity audit is missing or failing"
    if requirement.startswith("full_olmo_primary_"):
        return "critical", "full-grid primary downstream artifact has not been produced"
    return "supporting", "supporting artifact is missing or incomplete"


def _annotate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        severity, blocker = _severity_and_blocker(row)
        scope = "optional" if _bool(row.get("optional", False)) else "required"
        annotated.append({**row, "scope": scope, "severity": severity, "blocker": blocker})
    return annotated


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "requirement",
        "status",
        "scope",
        "severity",
        "complete",
        "evidence",
        "existing",
        "expected",
        "remaining",
        "blocker",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    required_rows = [row for row in rows if row.get("scope") != "optional"]
    optional_rows = [row for row in rows if row.get("scope") == "optional"]
    complete_required = sum(1 for row in required_rows if _bool(row["complete"]))
    complete_optional = sum(1 for row in optional_rows if _bool(row["complete"]))
    complete_all = sum(1 for row in rows if _bool(row["complete"]))
    lines = [
        "# Phase 3 Completion Audit Snapshot",
        "",
        f"Required requirements complete: `{complete_required}/{len(required_rows)}`",
        f"All tracked rows complete: `{complete_all}/{len(rows)}`",
    ]
    if optional_rows:
        lines.append(
            f"Optional/addendum rows complete: `{complete_optional}/{len(optional_rows)}`"
        )
    lines.extend(
        [
            "",
            "| Requirement | Status | Scope | Severity | Complete | Existing | Expected | Remaining | Evidence | Blocker |",
            "|---|---|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {requirement} | {status} | {scope} | {severity} | {complete} | {existing} | {expected} | {remaining} | {evidence} | {blocker} |".format(
                requirement=row["requirement"],
                status=row["status"],
                scope=row["scope"],
                severity=row["severity"],
                complete="yes" if _bool(row["complete"]) else "no",
                existing=row["existing"],
                expected=row["expected"],
                remaining=row["remaining"],
                evidence=row["evidence"],
                blocker=row["blocker"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generation_row_status(row: dict[str, str]) -> dict[str, Any]:
    expected = int(float(row["expected_generations"]))
    existing = _jsonl_records(Path(row["generations_output"]))
    summary_exists = Path(row["summary_output"]).exists()
    completed = summary_exists and existing >= expected
    return {
        **row,
        "temperature_float": float(row["temperature"]),
        "expected_generations_int": expected,
        "existing_generations_int": existing,
        "summary_exists": summary_exists,
        "completed_bool": completed,
    }


def audit_phase3_completion(args: argparse.Namespace) -> list[dict[str, Any]]:
    generation_rows = [_generation_row_status(row) for row in _read_csv_rows(args.generation_plan)]
    if not generation_rows:
        raise ValueError("generation plan has no rows")

    rows: list[dict[str, Any]] = []
    full_grid_optional = getattr(args, "full_grid_scope", "required") == "optional"
    expected_total = sum(int(row["expected_generations_int"]) for row in generation_rows)
    existing_total = sum(int(row["existing_generations_int"]) for row in generation_rows)
    complete_rows = [row for row in generation_rows if _bool(row["completed_bool"])]
    full_grid_complete = len(complete_rows) == len(generation_rows)
    rows.append(
        {
            "requirement": "full_olmo_sglang_generation_grid",
            "status": "complete" if full_grid_complete else "in_progress",
            "complete": full_grid_complete,
            "evidence": str(args.generation_plan),
            "existing": existing_total,
            "expected": expected_total,
            "remaining": max(0, expected_total - existing_total),
            "optional": full_grid_optional,
        }
    )

    by_temperature: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        by_temperature[float(row["temperature_float"])].append(row)
    for temp_rows in by_temperature.values():
        temp_rows.sort(key=lambda row: _stage_key(str(row["stage"])))

    primary_rows = by_temperature.get(args.primary_temperature, [])
    primary_expected = sum(int(row["expected_generations_int"]) for row in primary_rows)
    primary_existing = sum(int(row["existing_generations_int"]) for row in primary_rows)
    primary_stages = {str(row["stage"]) for row in primary_rows}
    primary_complete = set(STAGE_ORDER).issubset(primary_stages) and all(
        _bool(row["completed_bool"]) for row in primary_rows
    )
    rows.append(
        {
            "requirement": f"primary_temperature_{args.primary_temperature:g}_four_stage_slice",
            "status": "complete" if primary_complete else "in_progress",
            "complete": primary_complete,
            "evidence": str(args.generation_plan),
            "existing": primary_existing,
            "expected": primary_expected,
            "remaining": max(0, primary_expected - primary_existing),
            "optional": full_grid_optional,
        }
    )

    finalization_exists = args.finalization_summary.exists()
    enabled_finalization_commands = _enabled_finalization_commands(args.finalization_summary)
    missing_finalization_prerequisites = _missing_finalization_prerequisite_groups(
        args.finalization_summary
    )
    rows.append(
        {
            "requirement": "full_grid_finalization_readiness_summary",
            "status": "complete" if finalization_exists else "missing",
            "complete": finalization_exists,
            "evidence": str(args.finalization_summary),
            "existing": 1 if finalization_exists else 0,
            "expected": 1,
            "remaining": 0 if finalization_exists else 1,
            "optional": full_grid_optional,
        }
    )
    no_pending_finalization = enabled_finalization_commands == 0
    rows.append(
        {
            "requirement": "full_grid_pending_finalization_commands",
            "status": (
                "complete"
                if no_pending_finalization
                else ("pending" if enabled_finalization_commands is not None else "missing")
            ),
            "complete": no_pending_finalization,
            "evidence": str(args.finalization_summary),
            "existing": 0 if enabled_finalization_commands is None else enabled_finalization_commands,
            "expected": 0,
            "remaining": 0 if no_pending_finalization else (1 if enabled_finalization_commands is None else enabled_finalization_commands),
            "optional": full_grid_optional,
        }
    )
    finalization_prerequisites_complete = missing_finalization_prerequisites == 0
    rows.append(
        {
            "requirement": "full_grid_finalization_prerequisites",
            "status": (
                "complete"
                if finalization_prerequisites_complete
                else ("missing" if missing_finalization_prerequisites is None else "missing_inputs")
            ),
            "complete": finalization_prerequisites_complete,
            "evidence": str(args.finalization_summary),
            "existing": 0
            if missing_finalization_prerequisites is None
            else missing_finalization_prerequisites,
            "expected": 0,
            "remaining": 0 if finalization_prerequisites_complete else 1,
            "optional": full_grid_optional,
        }
    )

    if args.integrity_audit is not None:
        rows.append({**_integrity_audit_status(args.integrity_audit), "optional": full_grid_optional})

    artifact_specs = [
        *((spec, False) for spec in args.required_artifact),
        *((spec, True) for spec in getattr(args, "optional_artifact", [])),
    ]
    for spec, optional in artifact_specs:
        name, path = _parse_required_artifact(spec)
        exists = _artifact_complete(path)
        rows.append(
            {
                "requirement": name,
                "status": "complete" if exists else "missing_or_empty_or_header_only",
                "complete": exists,
                "evidence": str(path),
                "existing": 1 if exists else 0,
                "expected": 1,
                "remaining": 0 if exists else 1,
                "optional": optional,
            }
        )

    annotated_rows = _annotate_rows(rows)
    _write_csv(args.output, annotated_rows)
    _write_summary(args.summary_output, annotated_rows)
    return annotated_rows


def main() -> None:
    args = build_parser().parse_args()
    rows = audit_phase3_completion(args)
    required_rows = [row for row in rows if row.get("scope") != "optional"]
    complete = sum(1 for row in required_rows if _bool(row["complete"]))
    print(
        f"Wrote Phase 3 completion audit snapshot with {complete}/{len(required_rows)} required requirements complete "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
