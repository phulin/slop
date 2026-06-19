from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


DEFAULT_MANUSCRIPT = Path("docs/experiments/paper_manuscript_draft.md")
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/manuscript/paper_manuscript_structure_audit.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_manuscript_structure_audit.md")

EXPECTED_HEADINGS = (
    '# Localizing "Slop Style" In Open Post-Training Ladders',
    "## Abstract",
    "## 1. Introduction",
    "## 2. Related Work",
    "## 3. Methods",
    "## 4. Results",
    "## 5. Discussion",
    "## 6. Limitations",
    "## 7. Conclusion",
    "## Figure And Table Captions",
    "## Reproducibility Appendices",
)

EXPECTED_RESULT_SUBSECTIONS = (
    "### 4.1 Post-Training Data Shift Toward Answer-Register Prose",
    "### 4.2 Aggregate EQ-Bench Scores Are Useful But Insufficient",
    "### 4.3 OLMo Slop-Lexicon Propensity Peaks At DPO",
    "### 4.4 SmolLM3 Replicates Style Pressure With Different Stage Localization",
    "### 4.5 Detector Discovery Produces Candidate Tier-3 Style Families",
)

MIN_TOTAL_WORDS = 4500
MAX_TOTAL_WORDS = 9000
MIN_ABSTRACT_WORDS = 100
MAX_ABSTRACT_WORDS = 250
MIN_CONCLUSION_WORDS = 80
MAX_CONCLUSION_WORDS = 250


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit integrated paper manuscript structure.")
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def run_audit_paper_manuscript_structure(args: argparse.Namespace) -> list[dict[str, Any]]:
    text = args.manuscript.read_text(encoding="utf-8") if args.manuscript.exists() else ""
    rows = audit_manuscript_structure(text)
    _write_csv(args.output_csv, rows)
    _write_markdown(args.output_md, rows, output_csv=args.output_csv)
    return rows


def audit_manuscript_structure(text: str) -> list[dict[str, Any]]:
    headings = _headings(text)
    rows = [
        _row(
            "main_heading_order",
            _headings_in_order(text, EXPECTED_HEADINGS),
            value=f"{sum(heading in text for heading in EXPECTED_HEADINGS)}/{len(EXPECTED_HEADINGS)}",
            expected="all expected main headings present in order",
        ),
        _row(
            "result_subsection_order",
            _headings_in_order(text, EXPECTED_RESULT_SUBSECTIONS),
            value=f"{sum(heading in text for heading in EXPECTED_RESULT_SUBSECTIONS)}/{len(EXPECTED_RESULT_SUBSECTIONS)}",
            expected="all expected Result 4.x subsections present in order",
        ),
    ]

    missing_headings = [heading for heading in EXPECTED_HEADINGS if heading not in text]
    if missing_headings:
        rows[0]["warnings"] = "; ".join(f"missing:{heading}" for heading in missing_headings)
        rows[0]["status"] = "review"

    missing_results = [heading for heading in EXPECTED_RESULT_SUBSECTIONS if heading not in text]
    if missing_results:
        rows[1]["warnings"] = "; ".join(f"missing:{heading}" for heading in missing_results)
        rows[1]["status"] = "review"

    total_words = _word_count(text)
    abstract_words = _word_count(_section(text, "## Abstract"))
    conclusion_words = _word_count(_section(text, "## 7. Conclusion"))
    figure_caption_count = len(_caption_ids(text, "Figure"))
    table_caption_count = len(_caption_ids(text, "Table"))
    appendix_words = _word_count(_section(text, "## Reproducibility Appendices"))

    rows.extend(
        [
            _bounded_count_row(
                "total_word_count",
                total_words,
                minimum=MIN_TOTAL_WORDS,
                maximum=MAX_TOTAL_WORDS,
                unit="words",
            ),
            _bounded_count_row(
                "abstract_word_count",
                abstract_words,
                minimum=MIN_ABSTRACT_WORDS,
                maximum=MAX_ABSTRACT_WORDS,
                unit="words",
            ),
            _bounded_count_row(
                "conclusion_word_count",
                conclusion_words,
                minimum=MIN_CONCLUSION_WORDS,
                maximum=MAX_CONCLUSION_WORDS,
                unit="words",
            ),
            _row(
                "figure_caption_count",
                figure_caption_count == 5,
                value=str(figure_caption_count),
                expected="5 figure captions",
            ),
            _row(
                "table_caption_count",
                table_caption_count == 5,
                value=str(table_caption_count),
                expected="5 table captions",
            ),
            _row(
                "reproducibility_appendix_present",
                appendix_words >= 20,
                value=f"{appendix_words} words",
                expected="appendix heading with at least 20 words",
            ),
            _row(
                "heading_count",
                len(headings) >= len(EXPECTED_HEADINGS) + len(EXPECTED_RESULT_SUBSECTIONS),
                value=str(len(headings)),
                expected="main headings plus result subsections present",
            ),
        ]
    )
    return rows


def _headings(text: str) -> list[str]:
    return re.findall(r"(?m)^#{1,3} .+$", text)


def _headings_in_order(text: str, expected: tuple[str, ...]) -> bool:
    position = -1
    for heading in expected:
        next_position = text.find(heading, position + 1)
        if next_position < 0:
            return False
        position = next_position
    return True


def _section(text: str, heading: str) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    next_heading = re.search(r"\n## ", text[start + len(heading) :])
    if next_heading:
        end = start + len(heading) + next_heading.start()
        return text[start:end]
    return text[start:]


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def _caption_ids(text: str, prefix: str) -> set[str]:
    return {
        f"{prefix} {match}"
        for match in re.findall(rf"(?m)^\*\*{prefix} ([0-9]+)\.", text)
    }


def _bounded_count_row(
    check_id: str,
    value: int,
    *,
    minimum: int,
    maximum: int,
    unit: str,
) -> dict[str, str]:
    return _row(
        check_id,
        minimum <= value <= maximum,
        value=f"{value} {unit}",
        expected=f"{minimum}-{maximum} {unit}",
    )


def _row(check_id: str, passed: bool, *, value: str, expected: str) -> dict[str, str]:
    return {
        "check_id": check_id,
        "status": "ready" if passed else "review",
        "value": value,
        "expected": expected,
        "warnings": "" if passed else "outside_expected_bounds",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["check_id", "status", "value", "expected", "warnings"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness = "ready" if all(row["status"] == "ready" for row in rows) else "review"
    lines = [
        "# Paper Manuscript Structure Audit",
        "",
        f"Machine-readable audit: `{output_csv}`",
        "",
        f"Readiness status: `{readiness}`.",
        "This audit checks the integrated manuscript's section order, length bounds, result-section coverage, captions, and appendix presence.",
        "",
        "| Check | Status | Value | Expected | Warnings |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        warning_text = row["warnings"] or "none"
        lines.append(
            "| {check_id} | {status} | {value} | {expected} | {warning_text} |".format(
                **row,
                warning_text=warning_text,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_manuscript_structure(args)
    print(f"Wrote {len(rows)} manuscript-structure rows to {args.output_csv}")
    print(f"Wrote manuscript-structure summary to {args.output_md}")


if __name__ == "__main__":
    main()
