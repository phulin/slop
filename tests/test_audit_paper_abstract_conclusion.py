from __future__ import annotations

import csv
from pathlib import Path

from slop_sftdiv.cli.audit_paper_abstract_conclusion import (
    PACKAGE_INDEX_SPEC,
    SECTION_SPECS,
    build_parser,
    run_audit_paper_abstract_conclusion,
)


def _clean_manuscript() -> str:
    abstract = "\n".join(SECTION_SPECS[0].required_phrases)
    conclusion = "\n".join(SECTION_SPECS[1].required_phrases)
    return (
        "# Title\n\n"
        "## Abstract\n\n"
        f"{abstract}\n\n"
        "## 1. Introduction\n\n"
        "Body.\n\n"
        "## 7. Conclusion\n\n"
        f"{conclusion}\n"
    )


def _clean_index() -> str:
    return "# Paper Package Index\n\n" + "\n".join(PACKAGE_INDEX_SPEC.required_phrases) + "\n"


def _run(tmp_path: Path, manuscript: str, package_index: str | None = None):
    manuscript_path = tmp_path / "manuscript.md"
    index_path = tmp_path / "paper_package_index.md"
    output_csv = tmp_path / "abstract_conclusion.csv"
    output_md = tmp_path / "abstract_conclusion.md"
    manuscript_path.write_text(manuscript, encoding="utf-8")
    index_path.write_text(package_index or _clean_index(), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--manuscript",
            str(manuscript_path),
            "--package-index",
            str(index_path),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ]
    )
    return run_audit_paper_abstract_conclusion(args), output_csv, output_md


def test_audit_paper_abstract_conclusion_marks_complete_sections_ready(tmp_path):
    rows, output_csv, output_md = _run(tmp_path, _clean_manuscript())

    assert len(rows) == 3
    assert {row["status"] for row in rows} == {"ready"}
    assert "Readiness status: `ready`." in output_md.read_text(encoding="utf-8")
    with output_csv.open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[0]["check_id"] == "abstract_alignment"
    assert persisted[1]["check_id"] == "conclusion_alignment"
    assert persisted[2]["check_id"] == "package_index_alignment"


def test_audit_paper_abstract_conclusion_flags_missing_required_phrase(tmp_path):
    manuscript = _clean_manuscript().replace("EQ-Bench Slop Score diagnostics", "")

    rows, _, _ = _run(tmp_path, manuscript)

    abstract = next(row for row in rows if row["check_id"] == "abstract_alignment")
    assert abstract["status"] == "review"
    assert "EQ-Bench Slop Score diagnostics" in abstract["missing_required"]


def test_audit_paper_abstract_conclusion_flags_forbidden_overclaim(tmp_path):
    manuscript = _clean_manuscript().replace(
        "narrower claim than \"alignment creates slop\"",
        "alignment creates slop.",
    )

    rows, _, _ = _run(tmp_path, manuscript)

    conclusion = next(row for row in rows if row["check_id"] == "conclusion_alignment")
    assert conclusion["status"] == "review"
    assert "alignment creates slop." in conclusion["forbidden_hits"]


def test_audit_paper_abstract_conclusion_flags_stale_package_index(tmp_path):
    rows, _, _ = _run(tmp_path, _clean_manuscript(), package_index="# Package\n\nPlaceholder.\n")

    index = next(row for row in rows if row["check_id"] == "package_index_alignment")
    assert index["status"] == "review"
    assert "paper_claim_matrix.md" in index["missing_required"]
