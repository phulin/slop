from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MANUSCRIPT = Path("docs/experiments/paper_manuscript_draft.md")
DEFAULT_PACKAGE_INDEX = Path("docs/experiments/paper_package_index.md")
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/manuscript/paper_abstract_conclusion_audit.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_abstract_conclusion_audit.md")


@dataclass(frozen=True)
class SectionSpec:
    check_id: str
    section_heading: str
    required_phrases: tuple[str, ...]
    forbidden_phrases: tuple[str, ...] = ()


SECTION_SPECS = (
    SectionSpec(
        "abstract_alignment",
        "## Abstract",
        (
            "full pybiber register measurement",
            "compounding tests",
            "EQ-Bench Slop Score diagnostics",
            "post-training corpora move away from narrative and personal web prose",
            "not simply an aggregate increase in hedging",
            "preference data do not show chosen-response enrichment",
            "human perceptibility validation remains open",
            "rather than a single global score",
        ),
        (
            "alignment creates slop",
            "DPO universally creates slop",
            "human-perceived slop",
        ),
    ),
    SectionSpec(
        "conclusion_alignment",
        "## 7. Conclusion",
        (
            "Slop style is best understood as a spectrum rather than a single defect",
            "broad register movement can be separated from feature-specific propensity amplification",
            "generation-time self-conditioning",
            "not explained by measured chosen-response enrichment",
            "promotion requires human validation",
            "narrower claim than \"alignment creates slop\"",
        ),
        (
            "is a single defect",
            "alignment creates slop.",
            "DPO universally creates slop",
        ),
    ),
)

PACKAGE_INDEX_SPEC = SectionSpec(
    "package_index_alignment",
    "docs/experiments/paper_package_index.md",
    (
        "Use `docs/experiments/paper_claim_matrix.md` as the authority",
        "slop style is feature-specific, stage-specific",
        "Full pybiber is a Phase 1 corpus-side register result",
        "paper_measurement_synthesis.md",
        "EQ-Bench Slop Score is a benchmark bridge",
        "Phase 4 detector-derived families remain machine-detectable candidates",
        "The package is paper-draft ready but not submission ready.",
        "uv run slop-check-paper-package --root /home/user/slop",
    ),
    (
        "alignment creates slop.",
        "DPO universally creates slop",
        "human-perceived slop families are complete",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit paper abstract and conclusion claim alignment."
    )
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--package-index", type=Path, default=DEFAULT_PACKAGE_INDEX)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def run_audit_paper_abstract_conclusion(args: argparse.Namespace) -> list[dict[str, Any]]:
    text = args.manuscript.read_text(encoding="utf-8") if args.manuscript.exists() else ""
    index_text = (
        args.package_index.read_text(encoding="utf-8") if args.package_index.exists() else ""
    )
    rows = [_audit_section(spec, text) for spec in SECTION_SPECS]
    rows.append(_audit_text(PACKAGE_INDEX_SPEC, index_text))
    _write_csv(args.output_csv, rows)
    _write_markdown(args.output_md, rows, output_csv=args.output_csv)
    return rows


def _audit_section(spec: SectionSpec, text: str) -> dict[str, str]:
    return _audit_text(spec, _section(text, spec.section_heading))


def _audit_text(spec: SectionSpec, text: str) -> dict[str, str]:
    section = _normalize_text(text)
    missing_required = [
        phrase for phrase in spec.required_phrases if _normalize_text(phrase) not in section
    ]
    forbidden_hits = [
        phrase for phrase in spec.forbidden_phrases if _normalize_text(phrase) in section
    ]
    status = "ready" if not missing_required and not forbidden_hits else "review"
    return {
        "check_id": spec.check_id,
        "section_heading": spec.section_heading,
        "status": status,
        "required_phrases": str(len(spec.required_phrases)),
        "required_present": str(len(spec.required_phrases) - len(missing_required)),
        "forbidden_phrases": str(len(spec.forbidden_phrases)),
        "forbidden_hits": "; ".join(forbidden_hits),
        "missing_required": "; ".join(missing_required),
    }


def _section(text: str, heading: str) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    remainder_start = start + len(heading)
    next_heading = re.search(r"\n## ", text[remainder_start:])
    if next_heading:
        return text[start : remainder_start + next_heading.start()]
    return text[start:]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "check_id",
        "section_heading",
        "status",
        "required_phrases",
        "required_present",
        "forbidden_phrases",
        "forbidden_hits",
        "missing_required",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_ready = all(row["status"] == "ready" for row in rows)
    readiness = "ready" if all_ready else "review"
    lines = [
        "# Paper Abstract And Conclusion Audit",
        "",
        f"Machine-readable audit: `{output_csv}`",
        "",
        f"Readiness status: `{readiness}`.",
        "This audit checks that the abstract and conclusion preserve the bounded "
        "slop-style thesis, pybiber/EQ-Bench framing, and human-validation caveat.",
        "",
        "| Check | Status | Required | Forbidden hits | Missing |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        missing = row["missing_required"] or "none"
        forbidden = row["forbidden_hits"] or "none"
        lines.append(
            "| {check_id} | {status} | {required_present}/{required_phrases} | "
            "{forbidden} | {missing} |".format(
                **row,
                forbidden=forbidden,
                missing=missing,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_abstract_conclusion(args)
    print(f"Wrote {len(rows)} abstract/conclusion audit rows to {args.output_csv}")
    print(f"Wrote abstract/conclusion audit summary to {args.output_md}")


if __name__ == "__main__":
    main()
