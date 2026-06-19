from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path.cwd()
DEFAULT_MANUSCRIPT = Path("docs/experiments/paper_manuscript_draft.md")
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/manuscript/paper_terminology_audit.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_terminology_audit.md")


@dataclass(frozen=True)
class TerminologyCheck:
    check_id: str
    required_phrases: tuple[str, ...]
    purpose: str


TERMINOLOGY_CHECKS: tuple[TerminologyCheck, ...] = (
    TerminologyCheck(
        "slop_style_scope",
        (
            "The study measures slop style as a set of observable stylistic markers and register features",
            "not as factual error or answer quality",
        ),
        "Defines slop style as a measurable stylistic surface rather than answer correctness.",
    ),
    TerminologyCheck(
        "mechanism_separation",
        (
            "Corpus rates ask what the data contain.",
            "Teacher-forced propensity asks what a checkpoint wants to do in the same context.",
            "Free-running generation asks what appears in sampled answers.",
            "Compounding asks whether one marker makes later markers more likely.",
        ),
        "Explains the four measurement layers in plain language.",
    ),
    TerminologyCheck(
        "localization_surface_boundary",
        (
            "We separate four localization layers",
            "Two additional measurement surfaces are kept separate from those localization layers.",
            "The EQ-Bench Slop Score is an aggregate public benchmark bridge",
            "Detector-derived Tier-3 features are a discovery surface",
        ),
        "Keeps the manuscript's methods hierarchy aligned with the measurement synthesis.",
    ),
    TerminologyCheck(
        "amplification_spectrum",
        (
            "The main estimand is an amplification spectrum",
            "data-side frequency with model-side propensity and generated-output behavior",
        ),
        "Defines the paper's main estimand.",
    ),
    TerminologyCheck(
        "tier1_policy",
        (
            "Formatting-only features such as list/header/bold lead-ins are retired",
            "rule_of_three_approx` is demoted",
            "stock_openers_closers` produced zero retained hits",
        ),
        "Preserves the scoped Tier-1 feature policy.",
    ),
    TerminologyCheck(
        "pybiber_boundary",
        (
            "Generated-output full pybiber was not run, so generated-output register claims are excluded",
            "does not make generated-output full-register claims",
        ),
        "Keeps pybiber claims corpus-side.",
    ),
    TerminologyCheck(
        "eqbench_boundary",
        (
            "EQ-Bench Slop Score is implemented as an exploratory benchmark bridge",
            "not as the paper's causal outcome variable",
        ),
        "Keeps EQ-Bench framed as an aggregate bridge, not mechanism evidence.",
    ),
    TerminologyCheck(
        "phase4_boundary",
        (
            "These features are candidates until human perceptibility labels",
            "matcher precision validation are complete",
        ),
        "Keeps detector-derived Tier-3 features provisional.",
    ),
    TerminologyCheck(
        "negative_thesis",
        (
            "not that post-training uniformly creates slop",
            "slop style is not one object",
        ),
        "Prevents the manuscript from collapsing into a single-cause thesis.",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit reader-facing paper terminology.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def run_audit_paper_terminology(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.root.resolve()
    manuscript_path = _resolve_path(root, args.manuscript)
    text = manuscript_path.read_text(encoding="utf-8") if manuscript_path.exists() else ""
    rows = audit_terminology(text)
    output_csv = _resolve_path(root, args.output_csv)
    output_md = _resolve_path(root, args.output_md)
    _write_csv(output_csv, rows)
    _write_markdown(output_md, rows, output_csv=args.output_csv)
    return rows


def audit_terminology(text: str) -> list[dict[str, Any]]:
    normalized_text = _normalize(text)
    rows = []
    for check in TERMINOLOGY_CHECKS:
        missing = [
            phrase for phrase in check.required_phrases if _normalize(phrase) not in normalized_text
        ]
        rows.append(
            {
                "check_id": check.check_id,
                "status": "ready" if not missing else "review",
                "required_count": str(len(check.required_phrases)),
                "matched_count": str(len(check.required_phrases) - len(missing)),
                "missing_required": "; ".join(missing),
                "purpose": check.purpose,
            }
        )
    return rows


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "check_id",
        "status",
        "required_count",
        "matched_count",
        "missing_required",
        "purpose",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness = "ready" if all(row["status"] == "ready" for row in rows) else "review"
    lines = [
        "# Paper Terminology Audit",
        "",
        f"Machine-readable audit: `{output_csv}`",
        "",
        f"Readiness status: `{readiness}`.",
        (
            "This audit checks that the integrated manuscript defines the core "
            "slop-style measurement terms and preserves key interpretation boundaries."
        ),
        "",
        "| Check | Status | Matched | Missing | Purpose |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        missing = row["missing_required"] or "none"
        lines.append(
            f"| {row['check_id']} | {row['status']} | "
            f"{row['matched_count']}/{row['required_count']} | {missing} | "
            f"{row['purpose']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_terminology(args)
    print(f"Wrote {len(rows)} terminology audit rows to {args.output_csv}")
    print(f"Wrote terminology audit summary to {args.output_md}")


if __name__ == "__main__":
    main()
