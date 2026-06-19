from __future__ import annotations

import csv
from pathlib import Path

from slop_sftdiv.cli.audit_paper_terminology import (
    TERMINOLOGY_CHECKS,
    audit_terminology,
    build_parser,
    run_audit_paper_terminology,
)


def _complete_text() -> str:
    return "\n".join(
        phrase
        for check in TERMINOLOGY_CHECKS
        for phrase in check.required_phrases
    )


def test_audit_terminology_marks_complete_text_ready() -> None:
    rows = audit_terminology(_complete_text())

    assert len(rows) == len(TERMINOLOGY_CHECKS)
    assert {row["status"] for row in rows} == {"ready"}


def test_audit_terminology_normalizes_whitespace() -> None:
    text = _complete_text().replace(
        "Teacher-forced propensity asks what a checkpoint wants",
        "Teacher-forced propensity asks what a\ncheckpoint wants",
    )

    rows = audit_terminology(text)

    mechanism = next(row for row in rows if row["check_id"] == "mechanism_separation")
    assert mechanism["status"] == "ready"


def test_audit_terminology_flags_missing_definition() -> None:
    text = _complete_text().replace("not as factual error or answer quality", "")

    rows = audit_terminology(text)

    scope = next(row for row in rows if row["check_id"] == "slop_style_scope")
    assert scope["status"] == "review"
    assert "not as factual error or answer quality" in scope["missing_required"]


def test_audit_paper_terminology_cli_writes_outputs(tmp_path: Path) -> None:
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text(_complete_text(), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--manuscript",
            str(manuscript),
            "--output-csv",
            str(tmp_path / "terminology.csv"),
            "--output-md",
            str(tmp_path / "terminology.md"),
        ]
    )

    rows = run_audit_paper_terminology(args)

    assert {row["status"] for row in rows} == {"ready"}
    with (tmp_path / "terminology.csv").open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[0]["check_id"] == "slop_style_scope"
    assert "Readiness status: `ready`." in (tmp_path / "terminology.md").read_text(
        encoding="utf-8"
    )
