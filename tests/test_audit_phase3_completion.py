from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest

from slop_sftdiv.cli.audit_phase3_completion import audit_phase3_completion


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plan_row(
    tmp_path: Path,
    *,
    stage: str,
    temperature: float,
    expected: int = 2,
    existing: int = 2,
    summary_exists: bool = True,
) -> dict[str, object]:
    generations = tmp_path / f"{stage}_t{temperature:g}.jsonl"
    summary = tmp_path / f"{stage}_t{temperature:g}_summary.csv"
    generations.write_text("".join("{}\n" for _ in range(existing)), encoding="utf-8")
    if summary_exists:
        summary.write_text("feature,count\nslop_lexicon,1\n", encoding="utf-8")
    return {
        "stage": stage,
        "temperature": temperature,
        "expected_generations": expected,
        "generations_output": generations,
        "summary_output": summary,
    }


def _args(tmp_path: Path, plan: Path, finalization: Path) -> argparse.Namespace:
    return argparse.Namespace(
        generation_plan=plan,
        finalization_summary=finalization,
        integrity_audit=None,
        primary_temperature=1.0,
        full_grid_scope="required",
        required_artifact=[f"existing_required={tmp_path / 'existing.txt'}"],
        optional_artifact=[],
        output=tmp_path / "audit.csv",
        summary_output=tmp_path / "audit.md",
    )


def _finalization_summary_text(*, enabled_commands: int = 0, missing: str = "") -> str:
    return "\n".join(
        [
            "# ready",
            "",
            f"Enabled commands: `{enabled_commands}`",
            "",
            "## Prerequisite Inputs",
            "",
            "| Group | Paths | Missing |",
            "|---|---:|---|",
            f"| feature_rates | 1 | {missing} |",
            "",
        ]
    )


def test_audit_phase3_completion_reports_partial_generation(tmp_path: Path) -> None:
    rows = [
        _plan_row(tmp_path, stage="base", temperature=1.0),
        _plan_row(tmp_path, stage="sft", temperature=1.0),
        _plan_row(tmp_path, stage="dpo", temperature=1.0),
        _plan_row(tmp_path, stage="final", temperature=1.0, existing=1),
    ]
    plan = tmp_path / "plan.csv"
    finalization = tmp_path / "finalize.md"
    (tmp_path / "existing.txt").write_text("ok\n", encoding="utf-8")
    finalization.write_text(_finalization_summary_text(), encoding="utf-8")
    _write_csv(plan, rows)

    audit_rows = audit_phase3_completion(_args(tmp_path, plan, finalization))

    by_requirement = {row["requirement"]: row for row in audit_rows}
    assert by_requirement["full_olmo_sglang_generation_grid"]["complete"] is False
    assert by_requirement["full_olmo_sglang_generation_grid"]["severity"] == "critical"
    assert "OLMo SGLang generation grid" in by_requirement[
        "full_olmo_sglang_generation_grid"
    ]["blocker"]
    assert by_requirement["full_olmo_sglang_generation_grid"]["existing"] == 7
    assert by_requirement["full_olmo_sglang_generation_grid"]["expected"] == 8
    assert by_requirement["primary_temperature_1_four_stage_slice"]["complete"] is False
    assert by_requirement["full_grid_finalization_readiness_summary"]["complete"] is True
    assert by_requirement["full_grid_pending_finalization_commands"]["complete"] is True
    assert by_requirement["full_grid_finalization_prerequisites"]["complete"] is True
    assert by_requirement["existing_required"]["complete"] is True
    report = (tmp_path / "audit.md").read_text(encoding="utf-8")
    assert "Required requirements complete: `4/6`" in report


def test_audit_phase3_completion_reports_complete_generation(tmp_path: Path) -> None:
    rows = [
        _plan_row(tmp_path, stage=stage, temperature=1.0)
        for stage in ("base", "sft", "dpo", "final")
    ]
    plan = tmp_path / "plan.csv"
    finalization = tmp_path / "finalize.md"
    (tmp_path / "existing.txt").write_text("ok\n", encoding="utf-8")
    finalization.write_text(_finalization_summary_text(), encoding="utf-8")
    _write_csv(plan, rows)

    audit_rows = audit_phase3_completion(_args(tmp_path, plan, finalization))

    by_requirement = {row["requirement"]: row for row in audit_rows}
    assert by_requirement["full_olmo_sglang_generation_grid"]["complete"] is True
    assert by_requirement["primary_temperature_1_four_stage_slice"]["complete"] is True
    assert by_requirement["full_grid_finalization_readiness_summary"]["complete"] is True
    assert by_requirement["full_grid_pending_finalization_commands"]["complete"] is True
    assert by_requirement["full_grid_finalization_prerequisites"]["complete"] is True


def test_audit_phase3_completion_can_track_full_grid_as_optional(
    tmp_path: Path,
) -> None:
    rows = [
        _plan_row(tmp_path, stage="base", temperature=1.0),
        _plan_row(tmp_path, stage="sft", temperature=1.0),
        _plan_row(tmp_path, stage="dpo", temperature=1.0),
        _plan_row(tmp_path, stage="final", temperature=1.0, existing=1),
    ]
    plan = tmp_path / "plan.csv"
    finalization = tmp_path / "finalize.md"
    required = tmp_path / "required.txt"
    optional = tmp_path / "optional_missing.csv"
    required.write_text("ok\n", encoding="utf-8")
    finalization.write_text(_finalization_summary_text(), encoding="utf-8")
    _write_csv(plan, rows)
    args = _args(tmp_path, plan, finalization)
    args.full_grid_scope = "optional"
    args.required_artifact = [f"required={required}"]
    args.optional_artifact = [f"optional_missing={optional}"]

    audit_rows = audit_phase3_completion(args)

    by_requirement = {row["requirement"]: row for row in audit_rows}
    assert by_requirement["full_olmo_sglang_generation_grid"]["complete"] is False
    assert by_requirement["full_olmo_sglang_generation_grid"]["scope"] == "optional"
    assert by_requirement["full_olmo_sglang_generation_grid"]["severity"] == "optional"
    assert by_requirement["primary_temperature_1_four_stage_slice"]["scope"] == "optional"
    assert by_requirement["optional_missing"]["scope"] == "optional"
    assert by_requirement["optional_missing"]["severity"] == "optional"
    report = (tmp_path / "audit.md").read_text(encoding="utf-8")
    assert "Required requirements complete: `1/1`" in report
    assert "Optional/addendum rows complete: `3/6`" in report


def test_audit_phase3_completion_reports_pending_finalization_commands(
    tmp_path: Path,
) -> None:
    rows = [
        _plan_row(tmp_path, stage=stage, temperature=1.0)
        for stage in ("base", "sft", "dpo", "final")
    ]
    plan = tmp_path / "plan.csv"
    finalization = tmp_path / "finalize.md"
    (tmp_path / "existing.txt").write_text("ok\n", encoding="utf-8")
    finalization.write_text(
        _finalization_summary_text(enabled_commands=3), encoding="utf-8"
    )
    _write_csv(plan, rows)

    audit_rows = audit_phase3_completion(_args(tmp_path, plan, finalization))

    by_requirement = {row["requirement"]: row for row in audit_rows}
    assert by_requirement["full_grid_pending_finalization_commands"]["complete"] is False
    assert by_requirement["full_grid_pending_finalization_commands"]["existing"] == 3
    assert by_requirement["full_grid_pending_finalization_commands"]["remaining"] == 3


def test_audit_phase3_completion_reports_missing_finalization_prerequisites(
    tmp_path: Path,
) -> None:
    rows = [
        _plan_row(tmp_path, stage=stage, temperature=1.0)
        for stage in ("base", "sft", "dpo", "final")
    ]
    plan = tmp_path / "plan.csv"
    finalization = tmp_path / "finalize.md"
    (tmp_path / "existing.txt").write_text("ok\n", encoding="utf-8")
    finalization.write_text(
        _finalization_summary_text(missing="`missing.csv`"), encoding="utf-8"
    )
    _write_csv(plan, rows)

    audit_rows = audit_phase3_completion(_args(tmp_path, plan, finalization))

    by_requirement = {row["requirement"]: row for row in audit_rows}
    assert by_requirement["full_grid_finalization_prerequisites"]["complete"] is False
    assert by_requirement["full_grid_finalization_prerequisites"]["status"] == "missing_inputs"
    assert by_requirement["full_grid_finalization_prerequisites"]["existing"] == 1
    assert by_requirement["full_grid_finalization_prerequisites"]["severity"] == "critical"


def test_audit_phase3_completion_accepts_passing_integrity_audit(tmp_path: Path) -> None:
    rows = [
        _plan_row(tmp_path, stage=stage, temperature=1.0)
        for stage in ("base", "sft", "dpo", "final")
    ]
    plan = tmp_path / "plan.csv"
    finalization = tmp_path / "finalize.md"
    integrity = tmp_path / "integrity.csv"
    (tmp_path / "existing.txt").write_text("ok\n", encoding="utf-8")
    finalization.write_text(_finalization_summary_text(), encoding="utf-8")
    _write_csv(plan, rows)
    _write_csv(
        integrity,
        [
            {"generation_cache": "base.jsonl", "records": 2, "passed": True},
            {"generation_cache": "sft.jsonl", "records": 2, "passed": "yes"},
        ],
    )
    args = _args(tmp_path, plan, finalization)
    args.integrity_audit = integrity

    audit_rows = audit_phase3_completion(args)

    by_requirement = {row["requirement"]: row for row in audit_rows}
    assert by_requirement["full_grid_sglang_cache_integrity"]["complete"] is True
    assert by_requirement["full_grid_sglang_cache_integrity"]["existing"] == 2
    assert by_requirement["full_grid_sglang_cache_integrity"]["expected"] == 2


def test_audit_phase3_completion_rejects_failing_integrity_audit(tmp_path: Path) -> None:
    rows = [
        _plan_row(tmp_path, stage=stage, temperature=1.0)
        for stage in ("base", "sft", "dpo", "final")
    ]
    plan = tmp_path / "plan.csv"
    finalization = tmp_path / "finalize.md"
    integrity = tmp_path / "integrity.csv"
    (tmp_path / "existing.txt").write_text("ok\n", encoding="utf-8")
    finalization.write_text(_finalization_summary_text(), encoding="utf-8")
    _write_csv(plan, rows)
    _write_csv(
        integrity,
        [
            {"generation_cache": "base.jsonl", "records": 2, "passed": True},
            {"generation_cache": "sft.jsonl", "records": 2, "passed": False},
        ],
    )
    args = _args(tmp_path, plan, finalization)
    args.integrity_audit = integrity

    audit_rows = audit_phase3_completion(args)

    by_requirement = {row["requirement"]: row for row in audit_rows}
    assert by_requirement["full_grid_sglang_cache_integrity"]["complete"] is False
    assert by_requirement["full_grid_sglang_cache_integrity"]["severity"] == "critical"
    assert by_requirement["full_grid_sglang_cache_integrity"]["existing"] == 1
    assert by_requirement["full_grid_sglang_cache_integrity"]["expected"] == 2


def test_audit_phase3_completion_rejects_missing_integrity_audit(tmp_path: Path) -> None:
    rows = [
        _plan_row(tmp_path, stage=stage, temperature=1.0)
        for stage in ("base", "sft", "dpo", "final")
    ]
    plan = tmp_path / "plan.csv"
    finalization = tmp_path / "finalize.md"
    (tmp_path / "existing.txt").write_text("ok\n", encoding="utf-8")
    finalization.write_text(_finalization_summary_text(), encoding="utf-8")
    _write_csv(plan, rows)
    args = _args(tmp_path, plan, finalization)
    args.integrity_audit = tmp_path / "missing_integrity.csv"

    audit_rows = audit_phase3_completion(args)

    by_requirement = {row["requirement"]: row for row in audit_rows}
    assert by_requirement["full_grid_sglang_cache_integrity"]["complete"] is False
    assert by_requirement["full_grid_sglang_cache_integrity"]["status"] == "missing"


def test_audit_phase3_completion_rejects_empty_required_artifact(tmp_path: Path) -> None:
    rows = [
        _plan_row(tmp_path, stage=stage, temperature=1.0)
        for stage in ("base", "sft", "dpo", "final")
    ]
    plan = tmp_path / "plan.csv"
    finalization = tmp_path / "finalize.md"
    (tmp_path / "existing.txt").write_text("", encoding="utf-8")
    finalization.write_text(_finalization_summary_text(), encoding="utf-8")
    _write_csv(plan, rows)

    audit_rows = audit_phase3_completion(_args(tmp_path, plan, finalization))

    by_requirement = {row["requirement"]: row for row in audit_rows}
    assert by_requirement["existing_required"]["complete"] is False
    assert by_requirement["existing_required"]["status"] == "missing_or_empty_or_header_only"


def test_audit_phase3_completion_rejects_header_only_csv_required_artifact(
    tmp_path: Path,
) -> None:
    rows = [
        _plan_row(tmp_path, stage=stage, temperature=1.0)
        for stage in ("base", "sft", "dpo", "final")
    ]
    plan = tmp_path / "plan.csv"
    finalization = tmp_path / "finalize.md"
    required = tmp_path / "existing.txt"
    args = _args(tmp_path, plan, finalization)
    args.required_artifact = [f"existing_required={required}"]
    required.write_text("header\n", encoding="utf-8")
    finalization.write_text(_finalization_summary_text(), encoding="utf-8")
    _write_csv(plan, rows)

    audit_rows = audit_phase3_completion(args)

    by_requirement = {row["requirement"]: row for row in audit_rows}
    assert by_requirement["existing_required"]["complete"] is True

    required = tmp_path / "header_only.csv"
    args.required_artifact = [f"header_only={required}"]
    required.write_text("feature,value\n", encoding="utf-8")
    audit_rows = audit_phase3_completion(args)
    by_requirement = {row["requirement"]: row for row in audit_rows}
    assert by_requirement["header_only"]["complete"] is False
    assert by_requirement["header_only"]["status"] == "missing_or_empty_or_header_only"


def test_audit_phase3_completion_rejects_bad_required_artifact(tmp_path: Path) -> None:
    plan = tmp_path / "plan.csv"
    finalization = tmp_path / "finalize.md"
    _write_csv(plan, [_plan_row(tmp_path, stage="base", temperature=1.0)])
    args = _args(tmp_path, plan, finalization)
    args.required_artifact = ["not-a-pair"]

    with pytest.raises(ValueError, match="NAME=PATH"):
        audit_phase3_completion(args)
