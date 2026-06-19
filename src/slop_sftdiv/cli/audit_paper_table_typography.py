from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LATEX = Path("docs/experiments/paper_latex_table_drafts.tex")
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/tables/paper_table_typography_review.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_table_typography_review.md")


@dataclass(frozen=True)
class TypographySpec:
    table_id: str
    latex_label: str
    environment: str
    width_target: str
    expected_threeparttable: bool
    expected_table_notes: bool


TYPOGRAPHY_SPECS = (
    TypographySpec("Table 1", r"\label{tab:data-scope}", "table*", r"\textwidth", True, True),
    TypographySpec("Table 2", r"\label{tab:pybiber-means}", "table*", r"\textwidth", True, True),
    TypographySpec("Table 4", r"\label{tab:tier1-precision}", "table*", r"\textwidth", True, True),
    TypographySpec("Appendix Table A1", r"\label{tab:claim-bands}", "table", r"\linewidth", False, False),
    TypographySpec(
        "Appendix Table A2",
        r"\label{tab:pybiber-intervals}",
        "table*",
        r"\textwidth",
        False,
        False,
    ),
    TypographySpec(
        "Appendix Table A3",
        r"\label{tab:negative-claims}",
        "table*",
        r"\textwidth",
        False,
        False,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review paper table typography conventions.")
    parser.add_argument("--latex", type=Path, default=DEFAULT_LATEX)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def run_audit_paper_table_typography(args: argparse.Namespace) -> list[dict[str, Any]]:
    latex = args.latex.read_text(encoding="utf-8") if args.latex.exists() else ""
    rows = [_review_one_table(spec, latex=latex) for spec in TYPOGRAPHY_SPECS]
    _write_csv(args.output_csv, rows)
    _write_markdown(args.output_md, rows, output_csv=args.output_csv)
    return rows


def _review_one_table(spec: TypographySpec, *, latex: str) -> dict[str, Any]:
    block = _extract_table_block(latex, spec.latex_label)
    warnings: list[str] = []
    if not block:
        warnings.append("missing_table_block")

    environment = _table_environment(block)
    width_target = _tabularx_width(block)
    uses_booktabs = all(command in block for command in (r"\toprule", r"\midrule", r"\bottomrule"))
    uses_tabularx = r"\begin{tabularx}" in block and r"\end{tabularx}" in block
    uses_threeparttable = r"\begin{threeparttable}" in block
    has_table_notes = r"\begin{tablenotes}" in block
    uses_compact_size = bool(re.search(r"\\(?:small|footnotesize)\b", block))
    no_vertical_rules = "|" not in _tabularx_column_spec(block)

    if environment != spec.environment:
        warnings.append("unexpected_table_environment")
    if width_target != spec.width_target:
        warnings.append("unexpected_width_target")
    if not uses_booktabs:
        warnings.append("missing_booktabs_rules")
    if not uses_tabularx:
        warnings.append("missing_tabularx")
    if uses_threeparttable != spec.expected_threeparttable:
        warnings.append("unexpected_threeparttable_usage")
    if has_table_notes != spec.expected_table_notes:
        warnings.append("unexpected_table_notes_usage")
    if not uses_compact_size:
        warnings.append("missing_compact_size")
    if not no_vertical_rules:
        warnings.append("vertical_rules_in_column_spec")

    status = "typography_review_passed" if not warnings else "review"
    return {
        "table_id": spec.table_id,
        "latex_label": spec.latex_label,
        "review_status": status,
        "table_environment": environment,
        "width_target": width_target,
        "uses_booktabs": str(uses_booktabs),
        "uses_tabularx": str(uses_tabularx),
        "uses_threeparttable": str(uses_threeparttable),
        "has_table_notes": str(has_table_notes),
        "uses_compact_size": str(uses_compact_size),
        "no_vertical_rules": str(no_vertical_rules),
        "warnings": ";".join(warnings),
        "next_action": "Adapt spacing and placement to the target venue template.",
    }


def _extract_table_block(latex: str, label: str) -> str:
    label_index = latex.find(label)
    if label_index < 0:
        return ""
    begin_candidates = [
        latex.rfind(r"\begin{table*}", 0, label_index),
        latex.rfind(r"\begin{table}", 0, label_index),
    ]
    begin = max(begin_candidates)
    if begin < 0:
        return ""
    end_star = latex.find(r"\end{table*}", label_index)
    end_regular = latex.find(r"\end{table}", label_index)
    end_candidates = [candidate for candidate in (end_star, end_regular) if candidate >= 0]
    if not end_candidates:
        return latex[begin:]
    end = min(end_candidates)
    end_token = r"\end{table*}" if end == end_star else r"\end{table}"
    return latex[begin : end + len(end_token)]


def _table_environment(block: str) -> str:
    match = re.search(r"\\begin\{(table\*?)\}", block)
    return match.group(1) if match else ""


def _tabularx_width(block: str) -> str:
    groups = _tabularx_groups(block)
    return groups[0] if len(groups) >= 1 else ""


def _tabularx_column_spec(block: str) -> str:
    groups = _tabularx_groups(block)
    return groups[1] if len(groups) >= 2 else ""


def _tabularx_groups(block: str) -> list[str]:
    start = block.find(r"\begin{tabularx}")
    if start < 0:
        return []
    index = start + len(r"\begin{tabularx}")
    groups: list[str] = []
    while index < len(block) and len(groups) < 2:
        while index < len(block) and block[index].isspace():
            index += 1
        if index >= len(block) or block[index] != "{":
            break
        group, index = _read_brace_group(block, index)
        groups.append(group)
    return groups


def _read_brace_group(text: str, start: int) -> tuple[str, int]:
    depth = 0
    chars: list[str] = []
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
            if depth > 1:
                chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars), index + 1
            chars.append(char)
        else:
            chars.append(char)
    return "".join(chars), len(text)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "table_id",
        "latex_label",
        "review_status",
        "table_environment",
        "width_target",
        "uses_booktabs",
        "uses_tabularx",
        "uses_threeparttable",
        "has_table_notes",
        "uses_compact_size",
        "no_vertical_rules",
        "warnings",
        "next_action",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_ready = all(row["review_status"] == "typography_review_passed" for row in rows)
    readiness = "typography_review_passed" if all_ready else "review"
    lines = [
        "# Paper Table Typography Review",
        "",
        f"Machine-readable review: `{output_csv}`",
        "",
        f"Review status: `{readiness}`.",
        "This review checks generic booktabs/tabularx typography; it does not replace "
        "target venue template adaptation.",
        "",
        "| Table | Status | Environment | Width | Notes | Warnings |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        notes = "yes" if row["has_table_notes"] == "True" else "no"
        warning_text = row["warnings"] or "none"
        lines.append(
            "| {table_id} | {review_status} | {table_environment} | {width_target} | "
            "{notes} | {warning_text} |".format(
                **row,
                notes=notes,
                warning_text=warning_text,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_table_typography(args)
    print(f"Wrote {len(rows)} paper table-typography rows to {args.output_csv}")
    print(f"Wrote paper table-typography summary to {args.output_md}")


if __name__ == "__main__":
    main()
