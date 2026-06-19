from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path.cwd()
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/submission/paper_submission_exit_audit.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_submission_exit_audit.md")

READY = "ready"
VENUE_REVIEW_PENDING = "venue_review_pending"
HUMAN_VALIDATION_PENDING = "human_validation_pending"
EDITORIAL_POLISH_PENDING = "editorial_polish_pending"
BLOCKED = "blocked"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit paper submission exit criteria.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def run_audit_paper_submission_exit(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.root.resolve()
    rows = audit_submission_exit(root)
    output_csv = _resolve_output(root, args.output_csv)
    output_md = _resolve_output(root, args.output_md)
    _write_csv(output_csv, rows)
    _write_markdown(output_md, rows, output_csv=_display_path(args.output_csv))
    return rows


def audit_submission_exit(root: Path) -> list[dict[str, str]]:
    rows = [
        _row(
            "figure_table_finalization",
            _status_for_figure_tables(root),
            "Figure/table audits, rendered-layout review, PDF/PNG export candidates, table typography review, venue-decision checklist, sizing inventory, and draft artifacts present",
            "Final venue-specific figure dimensions, export choice, and table spacing/placement",
            "Adapt Figure 1-Figure 5 dimensions and table spacing/placement to the target venue template; use the existing PDF/PNG exports when they match venue requirements.",
        ),
        _row(
            "manuscript_body",
            _status_for_manuscript_body(root),
            "Manuscript structure, limitations, terminology, and numeric-claims audits",
            "No repo-path prose, expected sections, bounded length, captions, appendix surface, limitations boundary coverage, reader-facing measurement definitions, and source-backed headline numbers",
            "Perform final editorial polish without breaking the structure, limitations, terminology, or numeric-claims audits.",
        ),
        _row(
            "claim_discipline",
            _status_for_claim_discipline(root),
            "Claim language and claim-evidence audits",
            "C1-C14 supported wording present, forbidden overclaims absent, and mapped evidence paths intact",
            "Keep manuscript edits inside the claim matrix boundaries and preserve source-backed evidence mapping.",
        ),
        _row(
            "tier1_boundaries",
            _status_for_tier1(root),
            "Tier-1 precision validation status",
            "Pass/demote/derived boundaries preserved in final text and captions",
            "Keep rule_of_three_approx exploratory and stock_openers_closers derived.",
        ),
        _row(
            "phase4_human_perceptibility",
            _status_for_phase4_human_labels(root),
            "Phase 4 human perceptibility package and summary",
            "Human labels collected and summarized before detector clusters are called human-perceived slop",
            "Label the blinded pilot or full package, then rerun slop-summarize-phase4-human-labels.",
        ),
        _row(
            "bibliography_and_package",
            _status_for_required_paths(root),
            "Required paper package artifacts, citation/source audit, and reproducibility manifest",
            "Citation/source inventories, citation audit, required artifacts, and checksum manifest present",
            "Run slop-check-paper-package after any manuscript, citation, figure, or table edits.",
        ),
    ]
    rows.append(_overall_row(rows))
    return rows


def _status_for_figure_tables(root: Path) -> str:
    figure_ready = _csv_all_status(
        root / "artifacts/paper/figures/paper_figure_readiness_audit.csv",
        "ready_for_visual_review",
    )
    visual_review_ready = _csv_all_status(
        root / "artifacts/paper/figures/paper_figure_visual_review.csv",
        "visual_review_passed",
        status_column="review_status",
    )
    table_ready = _csv_all_status(
        root / "artifacts/paper/tables/paper_table_readiness_audit.csv",
        "ready_for_venue_styling",
    )
    table_typography_ready = _csv_all_status(
        root / "artifacts/paper/tables/paper_table_typography_review.csv",
        "typography_review_passed",
        status_column="review_status",
    )
    venue_controls_ready = _venue_controls_ready(root)
    if (
        figure_ready
        and visual_review_ready
        and table_ready
        and table_typography_ready
        and venue_controls_ready
    ):
        return VENUE_REVIEW_PENDING
    return BLOCKED


def _venue_controls_ready(root: Path) -> bool:
    checklist_path = root / "artifacts/paper/submission/paper_venue_decision_checklist.csv"
    required_docs = (
        root / "docs/experiments/paper_venue_decision_checklist.md",
        root / "docs/experiments/paper_venue_adaptation_runbook.md",
        root / "docs/experiments/paper_venue_sizing_inventory.md",
    )
    if not checklist_path.exists() or not all(path.exists() for path in required_docs):
        return False
    with checklist_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_items = {
        "target_venue",
        "figure_dimensions",
        "figure_export_format",
        "table_placement",
        "appendix_table_inclusion",
        "manuscript_template",
        "human_labeling_requirement",
    }
    observed_items = {row.get("item") or row.get("decision_item", "") for row in rows}
    if expected_items - observed_items:
        return False
    return all(row.get("status") == "decision_pending" for row in rows)


def _status_for_tier1(root: Path) -> str:
    path = root / "artifacts/stage1/validation/precision_validation_status.csv"
    if not path.exists():
        return BLOCKED
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {row.get("feature", ""): row for row in csv.DictReader(handle)}
    expected = {
        "contrastive_negation": "pass",
        "slop_lexicon": "pass",
        "stock_openers": "pass",
        "stock_closers": "pass",
        "rule_of_three_approx": "demote",
        "stock_openers_closers": "derived",
    }
    all_expected = all(
        rows.get(feature, {}).get("gate_status") == status
        for feature, status in expected.items()
    )
    return READY if all_expected else BLOCKED


def _status_for_claim_discipline(root: Path) -> str:
    language_ready = _status_from_ready_csv(
        root / "artifacts/paper/claims/paper_claim_language_audit.csv"
    )
    evidence_ready = _status_from_ready_csv(
        root / "artifacts/paper/claims/paper_claim_evidence_audit.csv"
    )
    return READY if language_ready == READY and evidence_ready == READY else BLOCKED


def _status_for_manuscript_body(root: Path) -> str:
    structure_ready = _status_from_ready_csv(
        root / "artifacts/paper/manuscript/paper_manuscript_structure_audit.csv"
    )
    limitations_ready = _status_from_ready_csv(
        root / "artifacts/paper/manuscript/paper_limitations_audit.csv"
    )
    terminology_ready = _status_from_ready_csv(
        root / "artifacts/paper/manuscript/paper_terminology_audit.csv"
    )
    numeric_ready = _status_from_ready_csv(
        root / "artifacts/paper/manuscript/paper_numeric_claims_audit.csv"
    )
    return (
        READY
        if structure_ready == READY
        and limitations_ready == READY
        and terminology_ready == READY
        and numeric_ready == READY
        else BLOCKED
    )


def _status_for_phase4_human_labels(root: Path) -> str:
    path = (
        root
        / "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_perceptibility_summary.csv"
    )
    if not path.exists():
        return BLOCKED
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return BLOCKED
    labeled = sum(int(row.get("labeled", "0") or 0) for row in rows)
    if labeled == 0 and all(row.get("venn_region") == "unlabeled" for row in rows):
        return HUMAN_VALIDATION_PENDING
    if all(row.get("venn_region") != "unlabeled" for row in rows):
        return READY
    return HUMAN_VALIDATION_PENDING


def _status_for_required_paths(root: Path) -> str:
    required = (
        "docs/experiments/paper_references.bib",
        "docs/experiments/paper_reference_sources.md",
        "docs/experiments/paper_citation_audit.md",
        "artifacts/paper/references/paper_citation_audit.csv",
        "docs/experiments/paper_claim_evidence_audit.md",
        "artifacts/paper/claims/paper_claim_evidence_audit.csv",
        "docs/experiments/paper_source_provenance_audit.md",
        "artifacts/paper/submission/paper_source_provenance_audit.csv",
        "docs/experiments/paper_readiness_audit.md",
        "docs/experiments/paper_manuscript_draft.md",
        "docs/experiments/paper_terminology_audit.md",
        "artifacts/paper/manuscript/paper_terminology_audit.csv",
        "docs/experiments/paper_numeric_claims_audit.md",
        "artifacts/paper/manuscript/paper_numeric_claims_audit.csv",
        "docs/experiments/paper_reproducibility_manifest.md",
        "artifacts/paper/submission/paper_reproducibility_manifest.csv",
    )
    if not all((root / relative).exists() for relative in required):
        return BLOCKED
    audit_paths = (
        root / "artifacts/paper/references/paper_citation_audit.csv",
        root / "artifacts/paper/claims/paper_claim_evidence_audit.csv",
        root / "artifacts/paper/submission/paper_source_provenance_audit.csv",
    )
    if any(_status_from_ready_csv(path) != READY for path in audit_paths):
        return BLOCKED
    return _status_for_reproducibility_manifest(
        root / "artifacts/paper/submission/paper_reproducibility_manifest.csv"
    )


def _status_for_reproducibility_manifest(path: Path) -> str:
    if not path.exists():
        return BLOCKED
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return BLOCKED
    for row in rows:
        relative_path = row.get("path", "")
        size = row.get("size_bytes", "")
        sha256 = row.get("sha256", "")
        if row.get("status") != READY:
            return BLOCKED
        if not size.isdigit() or int(size) <= 0:
            return BLOCKED
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            return BLOCKED
        target = path.parents[3] / relative_path
        if not target.exists():
            return BLOCKED
        if str(target.stat().st_size) != size:
            return BLOCKED
        if hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
            return BLOCKED
    return READY


def _status_from_ready_csv(path: Path) -> str:
    if not path.exists():
        return BLOCKED
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return READY if rows and all(row.get("status") == READY for row in rows) else BLOCKED


def _csv_all_status(path: Path, expected: str, *, status_column: str = "status") -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return bool(rows) and all(row.get(status_column) == expected for row in rows)


def _overall_row(rows: list[dict[str, str]]) -> dict[str, str]:
    blockers = [
        row["exit_item"]
        for row in rows
        if row["status"]
        in {
            VENUE_REVIEW_PENDING,
            HUMAN_VALIDATION_PENDING,
            EDITORIAL_POLISH_PENDING,
            BLOCKED,
        }
    ]
    status = READY if not blockers else BLOCKED
    next_action = (
        "Submission-ready under this audit."
        if not blockers
        else "Resolve pending items: " + ", ".join(blockers)
    )
    return _row(
        "overall_submission_status",
        status,
        "Synthesis of submission exit rows",
        "All exit rows ready",
        next_action,
    )


def _row(
    exit_item: str,
    status: str,
    evidence: str,
    required_for_submission: str,
    next_action: str,
) -> dict[str, str]:
    return {
        "exit_item": exit_item,
        "status": status,
        "evidence": evidence,
        "required_for_submission": required_for_submission,
        "next_action": next_action,
    }


def _resolve_output(root: Path, output: Path) -> Path:
    return output if output.is_absolute() else root / output


def _display_path(output: Path) -> Path:
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "exit_item",
        "status",
        "evidence",
        "required_for_submission",
        "next_action",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overall = next(row for row in rows if row["exit_item"] == "overall_submission_status")
    lines = [
        "# Paper Submission Exit Audit",
        "",
        f"Machine-readable audit: `{output_csv}`",
        "",
        f"Submission status: `{overall['status']}`.",
        "This audit separates paper-draft readiness from submission readiness by making human-validation, venue-review, and final-polish blockers explicit.",
        "",
        "| Exit item | Status | Evidence | Required for submission | Next action |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {exit_item} | {status} | {evidence} | {required_for_submission} | {next_action} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_submission_exit(args)
    print(f"Wrote {len(rows)} paper submission-exit rows to {args.output_csv}")
    print(f"Wrote paper submission-exit summary to {args.output_md}")


if __name__ == "__main__":
    main()
