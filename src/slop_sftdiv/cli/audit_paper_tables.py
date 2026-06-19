from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MARKDOWN = Path("docs/experiments/paper_camera_ready_tables.md")
DEFAULT_LATEX = Path("docs/experiments/paper_latex_table_drafts.tex")
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/tables/paper_table_readiness_audit.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_table_readiness_audit.md")

FORBIDDEN_TABLE_PHRASES = ("docs/", "artifacts/", "src/", "uv run", "CLI")


@dataclass(frozen=True)
class TableSpec:
    table_id: str
    markdown_heading: str
    latex_label: str
    latex_caption: str
    min_markdown_data_rows: int


TABLE_SPECS = (
    TableSpec(
        "Table 1",
        "## Table 1. Data And Measurement Scope",
        r"\label{tab:data-scope}",
        r"\caption{Data and measurement scope.}",
        8,
    ),
    TableSpec(
        "Table 2",
        "## Table 2. Selected Full-Pybiber Register Means",
        r"\label{tab:pybiber-means}",
        r"\caption{Selected full-pybiber register means.}",
        11,
    ),
    TableSpec(
        "Table 4",
        "## Table 4. Tier-1 Precision-Validation Status",
        r"\label{tab:tier1-precision}",
        r"\caption{Tier-1 precision-validation status.}",
        8,
    ),
    TableSpec(
        "Appendix Table A1",
        "## Appendix Table A1. Claim Strength Bands",
        r"\label{tab:claim-bands}",
        r"\caption{Claim strength bands.}",
        3,
    ),
    TableSpec(
        "Appendix Table A2",
        "## Appendix Table A2. Selected Pybiber Bootstrap Intervals",
        r"\label{tab:pybiber-intervals}",
        r"\caption{Selected pybiber bootstrap intervals.}",
        8,
    ),
    TableSpec(
        "Appendix Table A3",
        "## Appendix Table A3. Paper-Safe Negative Claims",
        r"\label{tab:negative-claims}",
        r"\caption{Paper-safe negative claims.}",
        6,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit paper table package readiness.")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--latex", type=Path, default=DEFAULT_LATEX)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def run_audit_paper_tables(args: argparse.Namespace) -> list[dict[str, Any]]:
    markdown = args.markdown.read_text(encoding="utf-8") if args.markdown.exists() else ""
    latex = args.latex.read_text(encoding="utf-8") if args.latex.exists() else ""
    rows = [_audit_one_table(spec, markdown=markdown, latex=latex) for spec in TABLE_SPECS]
    _write_csv(args.output_csv, rows)
    _write_markdown(args.output_md, rows, output_csv=args.output_csv)
    return rows


def _audit_one_table(spec: TableSpec, *, markdown: str, latex: str) -> dict[str, Any]:
    warnings: list[str] = []
    heading_present = spec.markdown_heading in markdown
    markdown_table_present = False
    markdown_data_rows = 0
    if not heading_present:
        warnings.append("missing_markdown_heading")
    else:
        table_lines = _extract_markdown_table(markdown, spec.markdown_heading)
        markdown_table_present = bool(table_lines)
        markdown_data_rows = _count_markdown_data_rows(table_lines)
        if not markdown_table_present:
            warnings.append("missing_markdown_table")
        elif markdown_data_rows < spec.min_markdown_data_rows:
            warnings.append("too_few_markdown_rows")

    latex_label_present = spec.latex_label in latex
    latex_caption_present = spec.latex_caption in latex
    if not latex_label_present:
        warnings.append("missing_latex_label")
    if not latex_caption_present:
        warnings.append("missing_latex_caption")

    markdown_section = _extract_section(markdown, spec.markdown_heading) if heading_present else ""
    forbidden_markdown = [phrase for phrase in FORBIDDEN_TABLE_PHRASES if phrase in markdown_section]
    forbidden_latex = [phrase for phrase in FORBIDDEN_TABLE_PHRASES if phrase in latex]
    if forbidden_markdown:
        warnings.append("forbidden_markdown_phrase")
    if forbidden_latex:
        warnings.append("forbidden_latex_phrase")

    status = "ready_for_venue_styling" if not warnings else "review"
    return {
        "table_id": spec.table_id,
        "markdown_heading": spec.markdown_heading,
        "latex_label": spec.latex_label,
        "latex_caption": spec.latex_caption,
        "status": status,
        "markdown_heading_present": str(heading_present),
        "markdown_table_present": str(markdown_table_present),
        "markdown_data_rows": str(markdown_data_rows),
        "min_markdown_data_rows": str(spec.min_markdown_data_rows),
        "latex_label_present": str(latex_label_present),
        "latex_caption_present": str(latex_caption_present),
        "forbidden_markdown_phrases": ";".join(forbidden_markdown),
        "forbidden_latex_phrases": ";".join(forbidden_latex),
        "warnings": ";".join(warnings),
    }


def _extract_section(text: str, heading: str) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    next_heading = text.find("\n## ", start + len(heading))
    if next_heading < 0:
        return text[start:]
    return text[start:next_heading]


def _extract_markdown_table(text: str, heading: str) -> list[str]:
    section = _extract_section(text, heading)
    lines = section.splitlines()
    table_lines: list[str] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
            in_table = True
        elif in_table:
            break
    return table_lines


def _count_markdown_data_rows(lines: list[str]) -> int:
    count = 0
    for line in lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and all(cell and set(cell) <= {":", "-"} for cell in cells):
            continue
        if line.startswith("|"):
            count += 1
    return max(0, count - 1)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "table_id",
        "markdown_heading",
        "latex_label",
        "latex_caption",
        "status",
        "markdown_heading_present",
        "markdown_table_present",
        "markdown_data_rows",
        "min_markdown_data_rows",
        "latex_label_present",
        "latex_caption_present",
        "forbidden_markdown_phrases",
        "forbidden_latex_phrases",
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_ready = all(row["status"] == "ready_for_venue_styling" for row in rows)
    readiness = "ready_for_venue_styling" if all_ready else "review"
    lines = [
        "# Paper Table Readiness Audit",
        "",
        f"Machine-readable audit: `{output_csv}`",
        "",
        f"Readiness status: `{readiness}`.",
        "This audit checks Markdown/LaTeX table package hygiene; it does not replace final venue-specific typography.",
        "",
        "| Table | Status | Markdown rows | LaTeX label | LaTeX caption | Warnings |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        warning_text = row["warnings"] or "none"
        lines.append(
            "| {table_id} | {status} | {markdown_data_rows} | "
            "{latex_label_present} | {latex_caption_present} | {warning_text} |".format(
                **row,
                warning_text=warning_text,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_tables(args)
    print(f"Wrote {len(rows)} paper table-readiness rows to {args.output_csv}")
    print(f"Wrote paper table-readiness summary to {args.output_md}")


if __name__ == "__main__":
    main()
