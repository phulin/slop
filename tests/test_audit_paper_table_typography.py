from __future__ import annotations

import csv
from pathlib import Path

from slop_sftdiv.cli.audit_paper_table_typography import (
    TYPOGRAPHY_SPECS,
    build_parser,
    run_audit_paper_table_typography,
)


def _table_block(
    *,
    label: str,
    environment: str = "table*",
    width: str = r"\textwidth",
    threeparttable: bool = True,
    notes: bool = True,
    columns: str = r"@{}p{0.25\textwidth}X@{}",
) -> str:
    before_tabular = r"\begin{threeparttable}" + "\n" if threeparttable else ""
    after_tabular = ""
    if notes:
        after_tabular += (
            "\n"
            r"\begin{tablenotes}"
            "\n"
            r"\footnotesize"
            "\n"
            r"\item Note."
            "\n"
            r"\end{tablenotes}"
        )
    after_tabular += "\n" + (r"\end{threeparttable}" if threeparttable else "")
    return (
        rf"\begin{{{environment}}}[t]"
        "\n"
        r"\centering"
        "\n"
        r"\caption{Caption.}"
        "\n"
        f"{label}\n"
        f"{before_tabular}"
        r"\small"
        "\n"
        rf"\begin{{tabularx}}{{{width}}}{{{columns}}}"
        "\n"
        r"\toprule"
        "\n"
        r"A & B \\"
        "\n"
        r"\midrule"
        "\n"
        r"a & b \\"
        "\n"
        r"\bottomrule"
        "\n"
        r"\end{tabularx}"
        f"{after_tabular}\n"
        rf"\end{{{environment}}}"
        "\n"
    )


def _clean_latex() -> str:
    blocks: list[str] = []
    for spec in TYPOGRAPHY_SPECS:
        blocks.append(
            _table_block(
                label=spec.latex_label,
                environment=spec.environment,
                width=spec.width_target,
                threeparttable=spec.expected_threeparttable,
                notes=spec.expected_table_notes,
            )
        )
    return "\n".join(blocks)


def _run(tmp_path: Path, latex: str):
    latex_path = tmp_path / "tables.tex"
    output_csv = tmp_path / "table_typography.csv"
    output_md = tmp_path / "table_typography.md"
    latex_path.write_text(latex, encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--latex",
            str(latex_path),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ]
    )
    return run_audit_paper_table_typography(args), output_csv, output_md


def test_audit_paper_table_typography_marks_clean_package_passed(tmp_path):
    rows, output_csv, output_md = _run(tmp_path, _clean_latex())

    assert len(rows) == 6
    assert {row["review_status"] for row in rows} == {"typography_review_passed"}
    assert "Review status: `typography_review_passed`." in output_md.read_text(
        encoding="utf-8"
    )
    with output_csv.open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[-1]["table_id"] == "Appendix Table A3"
    assert persisted[-1]["no_vertical_rules"] == "True"


def test_audit_paper_table_typography_flags_unexpected_table_shape(tmp_path):
    latex = _clean_latex().replace(r"\begin{tabularx}{\textwidth}{@{}", r"\begin{tabularx}{\textwidth}{|@{}", 1)

    rows, _, _ = _run(tmp_path, latex)

    table1 = next(row for row in rows if row["table_id"] == "Table 1")
    assert table1["review_status"] == "review"
    assert "vertical_rules_in_column_spec" in table1["warnings"]


def test_audit_paper_table_typography_flags_missing_notes(tmp_path):
    latex = _clean_latex().replace(
        r"\begin{tablenotes}"
        "\n"
        r"\footnotesize"
        "\n"
        r"\item Note."
        "\n"
        r"\end{tablenotes}",
        "",
        1,
    )

    rows, _, _ = _run(tmp_path, latex)

    table1 = next(row for row in rows if row["table_id"] == "Table 1")
    assert table1["review_status"] == "review"
    assert "unexpected_table_notes_usage" in table1["warnings"]
