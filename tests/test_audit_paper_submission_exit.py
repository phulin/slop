from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from slop_sftdiv.cli.audit_paper_submission_exit import run_audit_paper_submission_exit


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ready_status_csv() -> str:
    return "id,status\none,ready\n"


def _reproducibility_manifest_csv(root: Path) -> str:
    target = root / "docs/experiments/paper_references.bib"
    return (
        "category,path,status,size_bytes,sha256\n"
        "paper_control,docs/experiments/paper_references.bib,ready,"
        f"{target.stat().st_size},{hashlib.sha256(target.read_bytes()).hexdigest()}\n"
    )


def _write_venue_controls(root: Path) -> None:
    _write(
        root / "artifacts/paper/submission/paper_venue_decision_checklist.csv",
        "item,status,current_evidence,decision_needed,next_action\n"
        "target_venue,decision_pending,Evidence,Decision,Next\n"
        "figure_dimensions,decision_pending,Evidence,Decision,Next\n"
        "figure_export_format,decision_pending,Evidence,Decision,Next\n"
        "table_placement,decision_pending,Evidence,Decision,Next\n"
        "appendix_table_inclusion,decision_pending,Evidence,Decision,Next\n"
        "manuscript_template,decision_pending,Evidence,Decision,Next\n"
        "human_labeling_requirement,decision_pending,Evidence,Decision,Next\n",
    )
    for relative in (
        "docs/experiments/paper_venue_decision_checklist.md",
        "docs/experiments/paper_venue_adaptation_runbook.md",
        "docs/experiments/paper_venue_sizing_inventory.md",
    ):
        _write(root / relative, "venue control\n")


def _write_known_submission_blocker_fixture(root: Path) -> None:
    _write(
        root / "artifacts/paper/figures/paper_figure_readiness_audit.csv",
        "figure_id,status\nFigure 1,ready_for_visual_review\n",
    )
    _write(
        root / "artifacts/paper/figures/paper_figure_visual_review.csv",
        "figure_id,review_status\nFigure 1,visual_review_passed\n",
    )
    _write(
        root / "artifacts/paper/tables/paper_table_readiness_audit.csv",
        "table_id,status\nTable 1,ready_for_venue_styling\n",
    )
    _write(
        root / "artifacts/paper/tables/paper_table_typography_review.csv",
        "table_id,review_status\nTable 1,typography_review_passed\n",
    )
    _write_venue_controls(root)
    _write(
        root / "artifacts/paper/manuscript/paper_manuscript_structure_audit.csv",
        _ready_status_csv(),
    )
    _write(
        root / "artifacts/paper/manuscript/paper_limitations_audit.csv",
        _ready_status_csv(),
    )
    _write(
        root / "artifacts/paper/manuscript/paper_terminology_audit.csv",
        _ready_status_csv(),
    )
    _write(
        root / "artifacts/paper/manuscript/paper_numeric_claims_audit.csv",
        _ready_status_csv(),
    )
    _write(
        root / "artifacts/paper/claims/paper_claim_language_audit.csv",
        _ready_status_csv(),
    )
    _write(
        root / "artifacts/paper/claims/paper_claim_evidence_audit.csv",
        _ready_status_csv(),
    )
    _write(
        root / "artifacts/stage1/validation/precision_validation_status.csv",
        "feature,gate_status\n"
        "contrastive_negation,pass\n"
        "slop_lexicon,pass\n"
        "stock_openers,pass\n"
        "stock_closers,pass\n"
        "rule_of_three_approx,demote\n"
        "stock_openers_closers,derived\n",
    )
    _write(
        root
        / "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_perceptibility_summary.csv",
        "feature,labeled,venn_region\nphase4_ig_process_framing,0,unlabeled\n",
    )
    for relative in (
        "docs/experiments/paper_references.bib",
        "docs/experiments/paper_reference_sources.md",
        "docs/experiments/paper_citation_audit.md",
        "docs/experiments/paper_claim_evidence_audit.md",
        "docs/experiments/paper_readiness_audit.md",
        "docs/experiments/paper_manuscript_draft.md",
        "docs/experiments/paper_terminology_audit.md",
        "docs/experiments/paper_numeric_claims_audit.md",
        "docs/experiments/paper_reproducibility_manifest.md",
        "docs/experiments/paper_source_provenance_audit.md",
    ):
        _write(root / relative, "x\n")
    for relative in (
        "artifacts/paper/references/paper_citation_audit.csv",
        "artifacts/paper/claims/paper_claim_evidence_audit.csv",
        "artifacts/paper/submission/paper_source_provenance_audit.csv",
    ):
        _write(root / relative, _ready_status_csv())
    _write(
        root / "artifacts/paper/submission/paper_reproducibility_manifest.csv",
        _reproducibility_manifest_csv(root),
    )


def test_submission_exit_audit_marks_known_submission_blockers(tmp_path: Path) -> None:
    _write_known_submission_blocker_fixture(tmp_path)

    rows = run_audit_paper_submission_exit(
        argparse.Namespace(
            root=tmp_path,
            output_csv=Path("artifacts/paper/submission/paper_submission_exit_audit.csv"),
            output_md=Path("docs/experiments/paper_submission_exit_audit.md"),
        )
    )

    by_item = {row["exit_item"]: row for row in rows}
    assert by_item["figure_table_finalization"]["status"] == "venue_review_pending"
    assert by_item["manuscript_body"]["status"] == "ready"
    assert by_item["phase4_human_perceptibility"]["status"] == "human_validation_pending"
    assert by_item["overall_submission_status"]["status"] == "blocked"
    assert "phase4_human_perceptibility" in by_item["overall_submission_status"]["next_action"]
    assert (tmp_path / "docs/experiments/paper_submission_exit_audit.md").exists()


def test_submission_exit_audit_blocks_figure_tables_without_venue_controls(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "artifacts/paper/figures/paper_figure_readiness_audit.csv",
        "figure_id,status\nFigure 1,ready_for_visual_review\n",
    )
    _write(
        tmp_path / "artifacts/paper/figures/paper_figure_visual_review.csv",
        "figure_id,review_status\nFigure 1,visual_review_passed\n",
    )
    _write(
        tmp_path / "artifacts/paper/tables/paper_table_readiness_audit.csv",
        "table_id,status\nTable 1,ready_for_venue_styling\n",
    )
    _write(
        tmp_path / "artifacts/paper/tables/paper_table_typography_review.csv",
        "table_id,review_status\nTable 1,typography_review_passed\n",
    )

    rows = run_audit_paper_submission_exit(
        argparse.Namespace(
            root=tmp_path,
            output_csv=Path("artifacts/paper/submission/paper_submission_exit_audit.csv"),
            output_md=Path("docs/experiments/paper_submission_exit_audit.md"),
        )
    )

    by_item = {row["exit_item"]: row for row in rows}
    assert by_item["figure_table_finalization"]["status"] == "blocked"
    assert "figure_table_finalization" in by_item["overall_submission_status"]["next_action"]


def test_submission_exit_audit_blocks_package_row_for_nonready_citation_audit(
    tmp_path: Path,
) -> None:
    _write_known_submission_blocker_fixture(tmp_path)
    _write(
        tmp_path / "artifacts/paper/references/paper_citation_audit.csv",
        "check_id,status\ncited_keys_have_bib_entries,review\n",
    )

    rows = run_audit_paper_submission_exit(
        argparse.Namespace(
            root=tmp_path,
            output_csv=Path("artifacts/paper/submission/paper_submission_exit_audit.csv"),
            output_md=Path("docs/experiments/paper_submission_exit_audit.md"),
        )
    )

    by_item = {row["exit_item"]: row for row in rows}
    assert by_item["bibliography_and_package"]["status"] == "blocked"
    assert "bibliography_and_package" in by_item["overall_submission_status"]["next_action"]


def test_submission_exit_audit_blocks_package_row_for_manifest_hash_drift(
    tmp_path: Path,
) -> None:
    _write_known_submission_blocker_fixture(tmp_path)
    _write(tmp_path / "docs/experiments/paper_references.bib", "changed\n")

    rows = run_audit_paper_submission_exit(
        argparse.Namespace(
            root=tmp_path,
            output_csv=Path("artifacts/paper/submission/paper_submission_exit_audit.csv"),
            output_md=Path("docs/experiments/paper_submission_exit_audit.md"),
        )
    )

    by_item = {row["exit_item"]: row for row in rows}
    assert by_item["bibliography_and_package"]["status"] == "blocked"
    assert "bibliography_and_package" in by_item["overall_submission_status"]["next_action"]
