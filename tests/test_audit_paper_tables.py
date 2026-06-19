from __future__ import annotations

import csv
from pathlib import Path

from slop_sftdiv.cli.audit_paper_tables import (
    TABLE_SPECS,
    build_parser,
    run_audit_paper_tables,
)


def _markdown_table(heading: str, rows: int) -> str:
    body = "\n".join(f"| row {index} | value {index} |" for index in range(rows))
    return (
        f"{heading}\n\n"
        "| Column | Value |\n"
        "|---|---:|\n"
        f"{body}\n"
    )


def _clean_markdown() -> str:
    return "# Tables\n\n" + "\n\n".join(
        _markdown_table(spec.markdown_heading, spec.min_markdown_data_rows)
        for spec in TABLE_SPECS
    )


def _clean_latex() -> str:
    return "\n".join(
        f"{spec.latex_caption}\n{spec.latex_label}" for spec in TABLE_SPECS
    )


def _run(tmp_path: Path, markdown: str, latex: str):
    markdown_path = tmp_path / "tables.md"
    latex_path = tmp_path / "tables.tex"
    output_csv = tmp_path / "table_audit.csv"
    output_md = tmp_path / "table_audit.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    latex_path.write_text(latex, encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--markdown",
            str(markdown_path),
            "--latex",
            str(latex_path),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ]
    )
    return run_audit_paper_tables(args), output_csv, output_md


def test_audit_paper_tables_marks_clean_package_ready(tmp_path):
    rows, output_csv, output_md = _run(tmp_path, _clean_markdown(), _clean_latex())

    assert len(rows) == 6
    assert {row["status"] for row in rows} == {"ready_for_venue_styling"}
    assert "Readiness status: `ready_for_venue_styling`." in output_md.read_text(
        encoding="utf-8"
    )
    with output_csv.open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[-1]["table_id"] == "Appendix Table A3"
    assert persisted[-1]["latex_caption_present"] == "True"


def test_audit_paper_tables_marks_missing_label_and_short_table_for_review(tmp_path):
    short_markdown = _clean_markdown().replace("| row 7 | value 7 |\n", "", 1)
    latex = _clean_latex().replace(r"\label{tab:pybiber-means}", "")

    rows, _, _ = _run(tmp_path, short_markdown, latex)

    table1 = next(row for row in rows if row["table_id"] == "Table 1")
    table2 = next(row for row in rows if row["table_id"] == "Table 2")
    assert table1["status"] == "review"
    assert "too_few_markdown_rows" in table1["warnings"]
    assert table2["status"] == "review"
    assert "missing_latex_label" in table2["warnings"]


def test_audit_paper_tables_marks_forbidden_provenance_for_review(tmp_path):
    markdown = _clean_markdown().replace("row 0", "docs/experiments/source", 1)
    latex = _clean_latex() + "\n% artifacts/path\n"

    rows, _, _ = _run(tmp_path, markdown, latex)

    assert any("forbidden_markdown_phrase" in row["warnings"] for row in rows)
    assert all("forbidden_latex_phrase" in row["warnings"] for row in rows)
