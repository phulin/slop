from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


REQUIRED_PATHS = (
    "EXPERIMENTS.md",
    "docs/experiments/paper_package_index.md",
    "docs/experiments/paper_claim_matrix.md",
    "docs/experiments/paper_claim_evidence_map.md",
    "docs/experiments/paper_reader_glossary.md",
    "docs/experiments/paper_measurement_synthesis.md",
    "docs/experiments/paper_claim_evidence_audit.md",
    "docs/experiments/paper_claim_language_audit.md",
    "docs/experiments/paper_abstract_conclusion_audit.md",
    "docs/experiments/paper_manuscript_structure_audit.md",
    "docs/experiments/paper_limitations_audit.md",
    "docs/experiments/paper_terminology_audit.md",
    "docs/experiments/paper_numeric_claims_audit.md",
    "docs/experiments/paper_review_hygiene_audit.md",
    "docs/experiments/paper_source_provenance_audit.md",
    "docs/experiments/paper_submission_exit_audit.md",
    "docs/experiments/paper_venue_decision_checklist.md",
    "docs/experiments/paper_venue_adaptation_runbook.md",
    "docs/experiments/paper_venue_sizing_inventory.md",
    "docs/experiments/paper_reproduction_runbook.md",
    "docs/experiments/paper_reviewer_faq.md",
    "docs/experiments/paper_release_inventory.md",
    "docs/experiments/paper_reproducibility_manifest.md",
    "docs/experiments/paper_tier1_publication_policy.md",
    "docs/experiments/precision_validation_status.md",
    "docs/experiments/paper_readiness_audit.md",
    "docs/experiments/paper_scaffold.md",
    "docs/experiments/paper_tables.md",
    "docs/experiments/paper_camera_ready_tables.md",
    "docs/experiments/paper_latex_table_drafts.tex",
    "docs/experiments/paper_methods_draft.md",
    "docs/experiments/paper_results_draft.md",
    "docs/experiments/paper_intro_discussion_draft.md",
    "docs/experiments/paper_manuscript_draft.md",
    "docs/experiments/paper_figure_table_manifest.md",
    "docs/experiments/paper_figure_readiness_audit.md",
    "docs/experiments/paper_figure_visual_review.md",
    "docs/experiments/paper_table_readiness_audit.md",
    "docs/experiments/paper_table_typography_review.md",
    "docs/experiments/paper_caption_appendix_draft.md",
    "docs/experiments/paper_citation_audit.md",
    "docs/experiments/paper_reference_sources.md",
    "docs/experiments/paper_citation_plan.md",
    "docs/experiments/paper_references.bib",
    "docs/experiments/source_card_notes.md",
    "docs/experiments/phase4_human_annotation_readiness.md",
    "docs/experiments/phase4_human_annotation_handoff.md",
    "docs/experiments/phase4_human_annotation_codebook.md",
    "docs/experiments/phase4_human_labeling_execution_checklist.md",
    "docs/experiments/phase4_human_perceptibility_protocol.md",
    "docs/experiments/phase4_human_perceptibility_summary.md",
    "artifacts/stage1/census/phase1_pybiber_register_intervals.csv",
    "artifacts/paper/claims/paper_claim_evidence_audit.csv",
    "artifacts/paper/claims/paper_claim_language_audit.csv",
    "artifacts/paper/manuscript/paper_abstract_conclusion_audit.csv",
    "artifacts/paper/manuscript/paper_manuscript_structure_audit.csv",
    "artifacts/paper/manuscript/paper_limitations_audit.csv",
    "artifacts/paper/manuscript/paper_terminology_audit.csv",
    "artifacts/paper/manuscript/paper_numeric_claims_audit.csv",
    "artifacts/paper/manuscript/paper_review_hygiene_audit.csv",
    "artifacts/paper/submission/paper_source_provenance_audit.csv",
    "artifacts/paper/submission/paper_submission_exit_audit.csv",
    "artifacts/paper/submission/paper_venue_decision_checklist.csv",
    "artifacts/paper/submission/paper_release_inventory.csv",
    "artifacts/paper/submission/paper_reproducibility_manifest.csv",
    "artifacts/paper/submission/draft_bundle/README.md",
    "artifacts/paper/submission/draft_bundle/MANIFEST.csv",
    "artifacts/stage1/validation/precision_validation_status.csv",
    "artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv",
    "artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.md",
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "phase4_human_annotation_readiness.csv",
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "phase4_human_perceptibility_pilot_20_each.csv",
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "phase4_human_perceptibility_blind_pilot_20_each.csv",
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "phase4_human_perceptibility_blind_pilot_20_each_map.csv",
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "phase4_human_perceptibility_blind_full.csv",
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "phase4_human_perceptibility_blind_full_map.csv",
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "phase4_human_perceptibility_summary.csv",
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "human_annotation_handoff/README.md",
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "human_annotation_handoff/MANIFEST.csv",
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "human_annotation_handoff/annotator/README.md",
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "human_annotation_handoff/coordinator/phase4_human_labeling_execution_checklist.md",
    "artifacts/paper/figures/paper_figure_readiness_audit.csv",
    "artifacts/paper/figures/paper_figure_visual_review.csv",
    "artifacts/paper/tables/paper_table_readiness_audit.csv",
    "artifacts/paper/tables/paper_table_typography_review.csv",
    "artifacts/paper/references/paper_citation_audit.csv",
    "artifacts/paper/figures/figure1_pybiber_register_selected.svg",
    "artifacts/paper/figures/figure2_eqbench_stage_scores.svg",
    "artifacts/paper/figures/figure3_olmo_slop_lexicon_views.svg",
    "artifacts/paper/figures/figure4_cross_ladder_af_scatter.svg",
    "artifacts/paper/figures/figure5_phase4_tier3_raw_af.svg",
)

TEXT_PATHS_TO_CHECK = (
    "EXPERIMENTS.md",
    "docs/experiments/paper_readiness_audit.md",
    "docs/experiments/paper_scaffold.md",
    "docs/experiments/paper_claim_evidence_map.md",
    "docs/experiments/paper_claim_evidence_audit.md",
    "docs/experiments/paper_figure_table_manifest.md",
    "docs/experiments/paper_figure_visual_review.md",
    "docs/experiments/paper_table_typography_review.md",
    "docs/experiments/paper_caption_appendix_draft.md",
    "docs/experiments/paper_camera_ready_tables.md",
    "docs/experiments/paper_methods_draft.md",
    "docs/experiments/paper_results_draft.md",
    "docs/experiments/paper_intro_discussion_draft.md",
    "docs/experiments/paper_citation_plan.md",
    "docs/experiments/paper_citation_audit.md",
    "docs/experiments/paper_submission_exit_audit.md",
    "docs/experiments/paper_venue_decision_checklist.md",
)

CITATION_TEXT_PATHS = (
    "docs/experiments/paper_manuscript_draft.md",
    "docs/experiments/paper_methods_draft.md",
    "docs/experiments/paper_results_draft.md",
    "docs/experiments/paper_intro_discussion_draft.md",
)

MANUSCRIPT_FORBIDDEN = (
    "docs/experiments",
    "artifacts/",
    "src/slop",
    "uv run",
    "CLI",
    "Appendix Pointers",
    "Status: integrated manuscript draft",
    "This is a writing draft",
    "As of this draft",
    "local retained artifacts",
    "local checkpoint artifacts",
    "project generation/reference",
    "project-specific generation slices",
    "project vocabulary",
    "The project uses",
    "this project",
    "In the terminology of this project",
    "pipeline ran",
    "current paper package",
    "This table is mainly a writing guardrail",
    "2/6 core features",
    "3/6 core features",
    "4/6 core features",
    "2 of six core features",
    "four of six core features",
    "three of six core features",
    "two of six core features",
    "two still blocking",
    "3 block",
    "4 block",
    "three still blocking",
    "four still blocking",
    "`stock_closers` | Core | Incomplete",
    "`stock_closers` | yes | incomplete",
    "`slop_lexicon` | Core | Incomplete",
    "`slop_lexicon` | yes | incomplete",
    "stock_closers`, and\n`stock_openers_closers` remain blocking",
    "`slop_lexicon`, and\n`stock_openers_closers` remain blocking",
    "`rule_of_three_approx` and `stock_openers_closers` remain blocking",
    "`rule_of_three_approx` and the derived pooled `stock_openers_closers` view remain blocking",
)

CAMERA_TABLE_FORBIDDEN = (
    "docs/",
    "artifacts/",
    "src/",
    "uv run",
    "CLI",
)

LATEX_TABLE_LABELS = (
    r"\label{tab:data-scope}",
    r"\label{tab:pybiber-means}",
    r"\label{tab:tier1-precision}",
    r"\label{tab:claim-bands}",
    r"\label{tab:pybiber-intervals}",
    r"\label{tab:negative-claims}",
)

LATEX_TABLE_CAPTIONS = (
    r"\caption{Data and measurement scope.}",
    r"\caption{Selected full-pybiber register means.}",
    r"\caption{Tier-1 precision-validation status.}",
    r"\caption{Claim strength bands.}",
    r"\caption{Selected pybiber bootstrap intervals.}",
    r"\caption{Paper-safe negative claims.}",
)

STALE_PHRASES = (
    "Tables are prose-oriented",
    "paper_tables.md is useful but not camera-ready",
    "Manuscript still has repo-path appendix pointers",
    "Appendix Pointers",
    "replace repo-path appendix",
    "lacks uncertainty",
    "does not include uncertainty intervals",
    "currently lacks uncertainty",
    "should use those intervals in the plotted panel",
    "reserve them for caption/appendix detail",
    "Manual precision validation for Tier-1 matchers remains incomplete",
    "Regenerate Figure 2 intervals",
    "current paper package",
    "current paper is bounded",
    "current interim gate",
    "current scorecard",
    "current Phase 1 sample",
    "project-specific generation slices",
    "In the terminology of this project",
    "This table is mainly a writing guardrail",
    "more purple words",
    "pile of slop words",
    "global certificate",
    "broader direct scan is explicitly budgeted",
    "should be described as derived unless",
    "scan is deliberately run",
    "manuscript assembly",
    "Figure 1 carries",
    "Figure 2 carries",
    "Figure 3 carries",
    "Figure 4 carries",
    "Figure 5 carries",
)
EXPECTED_CLAIM_IDS = tuple(f"C{index}" for index in range(1, 15))


@dataclass(frozen=True)
class CheckReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check paper-package consistency.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def run_check_paper_package(root: Path) -> CheckReport:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    _check_required_paths(root, errors)
    _check_experiments_scope(root, errors)
    _check_manuscript_clean(root, errors)
    _check_camera_tables_clean(root, errors)
    _check_camera_table_appendices(root, errors)
    _check_latex_table_pack(root, errors)
    _check_precision_validation_status(root, errors)
    _check_phase4_human_annotation_readiness(root, errors)
    _check_phase4_human_perceptibility_summary(root, errors)
    _check_phase4_human_perceptibility_protocol(root, errors)
    _check_phase4_human_annotation_handoff(root, errors)
    _check_phase4_blind_annotation_packets(root, errors)
    _check_phase4_human_annotation_codebook(root, errors)
    _check_phase4_human_labeling_execution_checklist(root, errors)
    _check_phase3_production_runbook(root, errors)
    _check_phase4_production_runbook(root, errors)
    _check_paper_figure_readiness(root, errors)
    _check_paper_figure_visual_review(root, errors)
    _check_paper_table_readiness(root, errors)
    _check_paper_table_typography_review(root, errors)
    _check_paper_claim_evidence_audit(root, errors)
    _check_paper_claim_language_audit(root, errors)
    _check_paper_abstract_conclusion_audit(root, errors)
    _check_paper_manuscript_structure_audit(root, errors)
    _check_paper_limitations_audit(root, errors)
    _check_paper_terminology_audit(root, errors)
    _check_paper_numeric_claims_audit(root, errors)
    _check_paper_review_hygiene_audit(root, errors)
    _check_paper_source_provenance_audit(root, errors)
    _check_paper_readiness_audit(root, errors)
    _check_paper_submission_exit_audit(root, errors)
    _check_paper_submission_draft_bundle(root, errors)
    _check_paper_venue_decision_checklist(root, errors)
    _check_paper_venue_adaptation_runbook(root, errors)
    _check_paper_venue_sizing_inventory(root, errors)
    _check_paper_reproduction_runbook(root, errors)
    _check_paper_package_index(root, errors)
    _check_paper_measurement_synthesis(root, errors)
    _check_paper_reader_glossary(root, errors)
    _check_paper_reviewer_faq(root, errors)
    _check_paper_release_inventory(root, errors)
    _check_paper_reproducibility_manifest(root, errors)
    _check_retired_feature_language(root, errors)
    _check_stale_phrases(root, errors)
    _check_backtick_paths(root, errors)
    _check_figure_table_references(root, errors)
    _check_interval_reference_coverage(root, errors)
    _check_caption_coverage(root, errors)
    _check_claim_coverage(root, errors)
    _check_reference_source_coverage(root, errors)
    _check_paper_citation_audit(root, errors)
    _check_citations(root, errors, warnings)

    return CheckReport(errors=tuple(errors), warnings=tuple(warnings))


def _read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def _check_required_paths(root: Path, errors: list[str]) -> None:
    for relative in REQUIRED_PATHS:
        if not (root / relative).exists():
            errors.append(f"missing required path: {relative}")


def _check_experiments_scope(root: Path, errors: list[str]) -> None:
    relative = "EXPERIMENTS.md"
    path = root / relative
    if not path.exists():
        return
    text = _read(root, relative)
    normalized = " ".join(text.split())
    required = (
        "Full pybiber is currently a corpus-side Phase 1 register surface",
        "generated-output full pybiber is not part of the active paper claim surface",
        "Run full pybiber register features over the retained Phase 1 corpus-side samples only",
    )
    for phrase in required:
        if phrase not in normalized:
            errors.append(f"{relative} missing pybiber scope boundary: {phrase}")

    forbidden = (
        "run the revised Tier-1 matchers and full pybiber register features over: "
        "pretrain strata, SFT targets"
    )
    if forbidden in text and "later generations from each checkpoint" in text:
        errors.append(
            f"{relative} blurs corpus-side full pybiber with generated-output "
            "checkpoint measurements"
        )


def _check_manuscript_clean(root: Path, errors: list[str]) -> None:
    relative = "docs/experiments/paper_manuscript_draft.md"
    path = root / relative
    if not path.exists():
        return
    text = _read(root, relative)
    for pattern in MANUSCRIPT_FORBIDDEN:
        if pattern in text:
            errors.append(f"{relative} contains forbidden main-body marker: {pattern}")


def _check_camera_tables_clean(root: Path, errors: list[str]) -> None:
    relative = "docs/experiments/paper_camera_ready_tables.md"
    path = root / relative
    if not path.exists():
        return
    text = _read(root, relative)
    for pattern in CAMERA_TABLE_FORBIDDEN:
        if pattern in text:
            errors.append(f"{relative} contains non-submission table marker: {pattern}")


def _check_camera_table_appendices(root: Path, errors: list[str]) -> None:
    relative = "docs/experiments/paper_camera_ready_tables.md"
    path = root / relative
    if not path.exists():
        return
    text = _read(root, relative)
    main_headings = (
        "## Table 1. Data And Measurement Scope",
        "## Table 2. Selected Full-Pybiber Register Means",
        "## Table 4. Tier-1 Precision-Validation Status",
    )
    for heading in main_headings:
        if heading not in text:
            errors.append(f"{relative} missing main table heading: {heading}")
    expected = (
        "## Appendix Table A1. Claim Strength Bands",
        "## Appendix Table A2. Selected Pybiber Bootstrap Intervals",
        "## Appendix Table A3. Paper-Safe Negative Claims",
    )
    for heading in expected:
        if heading not in text:
            errors.append(f"{relative} missing appendix table heading: {heading}")


def _check_latex_table_pack(root: Path, errors: list[str]) -> None:
    relative = "docs/experiments/paper_latex_table_drafts.tex"
    path = root / relative
    if not path.exists():
        return
    text = _read(root, relative)
    for pattern in CAMERA_TABLE_FORBIDDEN:
        if pattern in text:
            errors.append(f"{relative} contains non-submission table marker: {pattern}")
    for label in LATEX_TABLE_LABELS:
        if label not in text:
            errors.append(f"{relative} missing required table label: {label}")
    for caption in LATEX_TABLE_CAPTIONS:
        if caption not in text:
            errors.append(f"{relative} missing required table caption: {caption}")


def _check_precision_validation_status(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/stage1/validation/precision_validation_status.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_feature = {row.get("feature", ""): row for row in rows}
    expected_feature_states = {
        "stock_closers": {
            "gate_status": "pass",
            "labeled": "50",
            "exact": "48",
            "false_positive": "2",
            "ambiguous": "0",
            "precision": "0.960000",
        },
        "slop_lexicon": {
            "gate_status": "pass",
            "labeled": "50",
            "exact": "50",
            "false_positive": "0",
            "ambiguous": "0",
            "precision": "1.000000",
        },
        "rule_of_three_approx": {
            "gate_status": "demote",
            "labeled": "50",
            "exact": "28",
            "partial": "1",
            "false_positive": "19",
            "ambiguous": "2",
            "precision": "0.560000",
        },
        "stock_openers_closers": {
            "is_core": "False",
            "is_derived": "True",
            "gate_status": "derived",
            "labeled": "0",
        },
    }
    for feature, expected in expected_feature_states.items():
        row = by_feature.get(feature)
        if not row:
            errors.append(f"{csv_relative} missing {feature} row")
            continue
        for field, value in expected.items():
            if row.get(field) != value:
                errors.append(
                    f"{csv_relative} {feature} {field}="
                    f"{row.get(field)!r}, expected {value!r}"
                )
    core_rows = [row for row in rows if row.get("is_core") == "True"]
    passing_core = [row for row in core_rows if row.get("gate_status") == "pass"]
    if len(core_rows) == 5 and len(passing_core) != 4:
        errors.append(
            f"{csv_relative} expected 4/5 passing core features, "
            f"found {len(passing_core)}/5"
        )

    md_relative = "docs/experiments/precision_validation_status.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "| stock_closers | True | False | pass | 420 | 50 | 48 | 0 | 2 | 0 | 0.960000 | 0.000000 |",
        "| slop_lexicon | True | False | pass | 420 | 50 | 50 | 0 | 0 | 0 | 1.000000 | 0.000000 |",
        "| rule_of_three_approx | True | False | demote | 420 | 50 | 28 | 1 | 19 | 2 | 0.560000 | 0.040000 |",
        "| stock_openers_closers | False | True | derived | 0 | 0 | 0 | 0 | 0 | 0 |  |  |",
        "Core features passing the current gate: `4/5`.",
        "Core features not passing publication-grade Phase 1 regex claims: `1`.",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing precision-status phrase: {phrase}")


def _check_phase4_human_annotation_readiness(root: Path, errors: list[str]) -> None:
    csv_relative = (
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_annotation_readiness.csv"
    )
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10:
        errors.append(f"{csv_relative} expected 10 candidate rows, found {len(rows)}")
    for row in rows:
        feature = row.get("feature", "<missing feature>")
        expected = {
            "rows": "100",
            "expected_rows": "100",
            "row_count_status": "ok",
            "schema_status": "ok",
            "labeled": "0",
            "blank_human_perceived_slop": "100",
            "blank_shaib_taxonomy_label": "100",
            "label_status": "blank",
        }
        for field, value in expected.items():
            if row.get(field) != value:
                errors.append(
                    f"{csv_relative} {feature} {field}="
                    f"{row.get(field)!r}, expected {value!r}"
                )

    md_relative = "docs/experiments/phase4_human_annotation_readiness.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready_for_labeling`.",
        "Package rows: `1000` across `10` candidate features.",
        "Pilot sheet size: `20` rows per feature.",
        "Blind full-package labeling sheet:",
        "The package remains\nunlabeled and should still be treated as detector-only evidence",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing readiness phrase: {phrase}")


def _check_phase4_human_perceptibility_summary(root: Path, errors: list[str]) -> None:
    csv_relative = (
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_perceptibility_summary.csv"
    )
    csv_path = root / csv_relative
    if csv_path.exists():
        rows = _read_csv_rows(csv_path)
        expected_columns = [
            "feature",
            "candidate_label",
            "cluster",
            "total",
            "labeled",
            "blank",
            "yes",
            "context_dependent",
            "no",
            "unclear",
            "human_perceived_or_context_count",
            "human_perceived_or_context_rate",
            "yes_rate",
            "top_shaib_taxonomy_label",
            "venn_region",
        ]
        _check_csv_columns(csv_relative, rows, expected_columns, errors)
        if len(rows) != 10:
            errors.append(f"{csv_relative} expected 10 candidate summary rows, found {len(rows)}")
        for row in rows:
            feature = row.get("feature", "<missing feature>")
            expected = {
                "total": "100",
                "labeled": "0",
                "blank": "100",
                "yes": "0",
                "context_dependent": "0",
                "no": "0",
                "unclear": "0",
                "human_perceived_or_context_count": "0",
                "human_perceived_or_context_rate": "",
                "yes_rate": "",
                "top_shaib_taxonomy_label": "",
                "venn_region": "unlabeled",
            }
            for field, value in expected.items():
                if row.get(field) != value:
                    errors.append(
                        f"{csv_relative} {feature} {field}="
                        f"{row.get(field)!r}, expected {value!r}"
                    )

    md_relative = "docs/experiments/phase4_human_perceptibility_summary.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Blank rows\nmean human annotation has not been completed.",
        "A candidate should remain\ndetector-only until enough rows are labeled",
        "`detector_and_human_perceived`.",
        "| phase4_ig_followup_offer | 0/100 | 0 | 0 | 0 | 0 |  | unlabeled |",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing human-summary phrase: {phrase}")


def _check_phase4_human_perceptibility_protocol(root: Path, errors: list[str]) -> None:
    md_relative = "docs/experiments/phase4_human_perceptibility_protocol.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "phase4_human_perceptibility_blind_full_labeled.jsonl",
        "phase4_human_perceptibility_blind_full_adjudicated.jsonl",
        "phase4_human_perceptibility_summary.csv",
        "Typo IDs and stale rows for consensus examples are rejected",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing human-protocol phrase: {phrase}")
    forbidden_phrases = (
        "--input artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_annotation_package.jsonl",
    )
    for phrase in forbidden_phrases:
        if phrase in markdown:
            errors.append(f"{md_relative} contains stale human-protocol phrase: {phrase}")


def _check_phase4_human_annotation_handoff(root: Path, errors: list[str]) -> None:
    base = Path("artifacts/phase4/modernbert_detector_combined_v2_clean/human_annotation_handoff")
    manifest_relative = base / "MANIFEST.csv"
    manifest_path = root / manifest_relative
    if not manifest_path.exists():
        return
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 14:
        errors.append(f"{manifest_relative} expected 14 handoff rows, found {len(rows)}")
    required_bundle_paths = {
        "annotator/README.md",
        "annotator/phase4_human_annotation_codebook.md",
        "annotator/phase4_human_perceptibility_blind_pilot_20_each.csv",
        "annotator/phase4_human_perceptibility_blind_full.csv",
        "annotator/annotator_a/phase4_human_perceptibility_blind_pilot_20_each_annotator_a.csv",
        "annotator/annotator_b/phase4_human_perceptibility_blind_pilot_20_each_annotator_b.csv",
        "annotator/annotator_a/phase4_human_perceptibility_blind_full_annotator_a.csv",
        "annotator/annotator_b/phase4_human_perceptibility_blind_full_annotator_b.csv",
        "coordinator/phase4_human_perceptibility_protocol.md",
        "coordinator/phase4_human_annotation_readiness.md",
        "coordinator/phase4_human_labeling_execution_checklist.md",
        "coordinator/private_maps/phase4_human_perceptibility_blind_pilot_20_each_map.csv",
        "coordinator/private_maps/phase4_human_perceptibility_blind_full_map.csv",
        "coordinator/private_maps/phase4_human_perceptibility_annotation_package.jsonl",
    }
    observed_bundle_paths = {row.get("bundle_path", "") for row in rows}
    missing_bundle_paths = sorted(required_bundle_paths - observed_bundle_paths)
    if missing_bundle_paths:
        errors.append(
            f"{manifest_relative} missing handoff bundle paths: "
            f"{', '.join(missing_bundle_paths)}"
        )
    for row in rows:
        bundle_path = row.get("bundle_path", "<missing bundle path>")
        if row.get("status") != "ready":
            errors.append(
                f"{manifest_relative} {bundle_path} status={row.get('status')!r}, "
                "expected 'ready'"
            )
        if row.get("audience") == "annotator" and bundle_path.endswith(".csv"):
            if row.get("redaction_status") != "redacted":
                errors.append(
                    f"{manifest_relative} {bundle_path} redaction_status="
                    f"{row.get('redaction_status')!r}, expected 'redacted'"
                )
        sha256 = row.get("sha256", "")
        if len(sha256) != 64 or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            errors.append(f"{manifest_relative} {bundle_path} has invalid sha256")
        size = row.get("size_bytes", "")
        if not size.isdigit() or int(size) <= 0:
            errors.append(f"{manifest_relative} {bundle_path} has invalid size_bytes: {size!r}")
        bundle_file = root / base / bundle_path
        if row.get("status") == "ready" and not bundle_file.exists():
            errors.append(f"{manifest_relative} {bundle_path} points to missing bundle file")
            continue
        if row.get("status") == "ready" and bundle_file.exists():
            if row.get("audience") == "annotator" and bundle_path.endswith(".csv"):
                leaked = sorted(
                    _phase4_forbidden_annotator_columns()
                    & set(_csv_fieldnames(bundle_file))
                )
                if leaked:
                    errors.append(
                        f"{manifest_relative} {bundle_path} leaks unblinded columns: "
                        f"{', '.join(leaked)}"
                    )
            actual_size = str(bundle_file.stat().st_size)
            actual_sha = _sha256(bundle_file)
            if size != actual_size:
                errors.append(
                    f"{manifest_relative} {bundle_path} size_bytes={size!r}, "
                    f"expected {actual_size!r}"
                )
            if sha256 != actual_sha:
                errors.append(
                    f"{manifest_relative} {bundle_path} sha256={sha256!r}, "
                    f"expected {actual_sha!r}"
                )
            source_relative = row.get("source_path", "")
            if not source_relative:
                errors.append(f"{manifest_relative} {bundle_path} missing source_path")
            elif source_relative != "generated":
                source_file = root / source_relative
                if not source_file.exists():
                    errors.append(
                        f"{manifest_relative} {bundle_path} source_path={source_relative!r} "
                        "points to missing source file"
                    )
                else:
                    source_size = str(source_file.stat().st_size)
                    source_sha = _sha256(source_file)
                    if size != source_size:
                        errors.append(
                            f"{manifest_relative} {bundle_path} size_bytes={size!r}, "
                            f"expected source size {source_size!r}"
                        )
                    if sha256 != source_sha:
                        errors.append(
                            f"{manifest_relative} {bundle_path} sha256={sha256!r}, "
                            f"expected source sha256 {source_sha!r}"
                        )

    readme_relative = base / "README.md"
    readme_path = root / readme_relative
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        required_readme_phrases = (
            "Share only `annotator/`",
            "Ready files: `14`.",
            "Missing files: `0`.",
            "Redacted annotator CSVs: `6`.",
            "`annotator/`: quickstart, codebook, generic blinded CSVs, and annotator-specific copies.",
            "`coordinator/`: protocol, readiness summary, labeling checklist, private maps, and source package.",
            "Annotator fill rules: edit only `human_perceived_slop`,",
            "Do not edit `blind_id`,",
            "Protected-field edits are",
            "`uv run slop-summarize-phase4-label-agreement`",
            "`adjudicated_human_perceived_slop`,",
            "`uv run slop-apply-phase4-label-adjudication`",
            "`uv run slop-summarize-phase4-human-labels`",
        )
        for phrase in required_readme_phrases:
            if phrase not in readme:
                errors.append(f"{readme_relative} missing handoff phrase: {phrase}")

    md_relative = "docs/experiments/phase4_human_annotation_handoff.md"
    md_path = root / md_relative
    if md_path.exists():
        markdown = _read(root, md_relative)
        required_md_phrases = (
            "Readiness status: `ready`.",
            "coordinator-only files preserve maps and source identifiers",
            "| annotator | annotator_guidance | `annotator/README.md` | ready | not_applicable |",
            "| annotator | annotator_sheet |",
            "| coordinator | coordinator_guidance | `coordinator/phase4_human_labeling_execution_checklist.md` | ready | not_applicable |",
        )
        for phrase in required_md_phrases:
            if phrase not in markdown:
                errors.append(f"{md_relative} missing handoff phrase: {phrase}")


def _check_phase4_human_labeling_execution_checklist(
    root: Path, errors: list[str]
) -> None:
    md_relative = "docs/experiments/phase4_human_labeling_execution_checklist.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Execution status: `ready_for_external_labels`.",
        "This checklist does not report completed labels.",
        "Share only `human_annotation_handoff/annotator/`",
        "Do not share\n`coordinator/private_maps/`",
        "uv run slop-summarize-phase4-label-agreement",
        "uv run slop-apply-phase4-label-adjudication",
        "uv run slop-summarize-phase4-human-labels",
        "Keep detector claims machine-detectable only until",
        "Update C11/C12/C13 in `paper_claim_matrix.md`",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing labeling-checklist phrase: {phrase}")


def _check_phase4_blind_annotation_packets(root: Path, errors: list[str]) -> None:
    base = Path("artifacts/phase4/modernbert_detector_combined_v2_clean")
    packet_specs = (
        (
            "pilot",
            base / "phase4_human_perceptibility_blind_pilot_20_each.csv",
            base / "phase4_human_perceptibility_blind_pilot_20_each_map.csv",
            200,
        ),
        (
            "full",
            base / "phase4_human_perceptibility_blind_full.csv",
            base / "phase4_human_perceptibility_blind_full_map.csv",
            1000,
        ),
    )
    blind_columns = [
        "blind_id",
        "matched_text",
        "snippet",
        "human_perceived_slop",
        "shaib_taxonomy_label",
        "notes",
    ]
    map_columns = [
        "blind_id",
        "annotation_id",
        "feature",
        "candidate_label",
        "cluster",
        "source",
        "record_id",
        "matched_text",
        "snippet",
    ]
    for label, blind_relative, map_relative, expected_rows in packet_specs:
        blind_path = root / blind_relative
        map_path = root / map_relative
        if not blind_path.exists() or not map_path.exists():
            continue
        blind_rows = _read_csv_rows(blind_path)
        map_rows = _read_csv_rows(map_path)
        blind_relative_text = str(blind_relative)
        map_relative_text = str(map_relative)
        _check_csv_columns(blind_relative_text, blind_rows, blind_columns, errors)
        _check_csv_columns(map_relative_text, map_rows, map_columns, errors)
        if len(blind_rows) != expected_rows:
            errors.append(
                f"{blind_relative_text} expected {expected_rows} {label} blind rows, "
                f"found {len(blind_rows)}"
            )
        if len(map_rows) != expected_rows:
            errors.append(
                f"{map_relative_text} expected {expected_rows} {label} map rows, "
                f"found {len(map_rows)}"
            )
        blind_fieldnames = set(blind_rows[0]) if blind_rows else set()
        leaked = sorted(_phase4_forbidden_annotator_columns() & blind_fieldnames)
        if leaked:
            errors.append(f"{blind_relative_text} leaks unblinded columns: {', '.join(leaked)}")
        blind_ids = [row.get("blind_id", "") for row in blind_rows]
        map_ids = [row.get("blind_id", "") for row in map_rows]
        if len(set(blind_ids)) != len(blind_ids):
            errors.append(f"{blind_relative_text} contains duplicate blind_id values")
        if len(set(map_ids)) != len(map_ids):
            errors.append(f"{map_relative_text} contains duplicate blind_id values")
        if set(blind_ids) != set(map_ids):
            errors.append(f"{blind_relative_text} blind_id set does not match {map_relative_text}")
        if any(row.get("feature", "") == "" for row in map_rows):
            errors.append(f"{map_relative_text} contains blank feature values")
        if any(row.get("annotation_id", "") == "" for row in map_rows):
            errors.append(f"{map_relative_text} contains blank annotation_id values")


def _check_csv_columns(
    relative: str,
    rows: list[dict[str, str]],
    expected_columns: list[str],
    errors: list[str],
) -> None:
    actual_columns = list(rows[0]) if rows else []
    if actual_columns != expected_columns:
        errors.append(
            f"{relative} columns={actual_columns!r}, expected {expected_columns!r}"
        )


def _phase4_forbidden_annotator_columns() -> set[str]:
    return {
        "annotation_id",
        "feature",
        "candidate_label",
        "cluster",
        "source",
        "record_id",
    }


def _csv_fieldnames(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _check_phase4_human_annotation_codebook(root: Path, errors: list[str]) -> None:
    md_relative = "docs/experiments/phase4_human_annotation_codebook.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Does this matched span contribute to a recognizable AI slop-style effect",
        "Use `human_perceived_slop=yes` only when",
        "Annotators should edit only `human_perceived_slop`, `shaib_taxonomy_label`,",
        "Do not edit `blind_id`, `matched_text`, or `snippet`",
        "`phase4_ig_process_framing`",
        "`phase4_ig_followup_offer`",
        "Compare `human_perceived_slop` before taxonomy labels.",
        "The pilot should be reported as a pilot if the full package remains unlabeled.",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing annotation-codebook phrase: {phrase}")


def _check_phase4_production_runbook(root: Path, errors: list[str]) -> None:
    md_relative = "docs/experiments/phase4_production_runbook.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "512-document exact teacher-forced continuation already exist",
        "The completed compute target is the larger 512-document exact sequence-mass",
        "The 512 exact continuation is complete.",
        "The 128-document results remain a historical",
        "scoping artifact.",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing production-runbook phrase: {phrase}")
    stale_phrases = (
        "128-document exact teacher-forced rerun already exist",
        "128-document exact teacher-forced rerun already exists",
    )
    for phrase in stale_phrases:
        if phrase in markdown:
            errors.append(f"{md_relative} contains stale production-runbook phrase: {phrase}")


def _check_phase3_production_runbook(root: Path, errors: list[str]) -> None:
    md_relative = "docs/experiments/phase3_production_runbook.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    normalized = " ".join(markdown.split())
    required_phrases = (
        "Do not treat generated-output full pybiber as part of the active Phase 3 route.",
        "The paper-facing full-pybiber result is the Phase 1 corpus-side register",
        "Older generated-output register-proxy/style-signature artifacts are archived diagnostics only",
    )
    for phrase in required_phrases:
        if phrase not in normalized:
            errors.append(f"{md_relative} missing generated-output pybiber boundary: {phrase}")
    stale_phrases = (
        "After the new `t=1.0` generated caches complete, extract full pybiber",
        "## Rebuild Biber-Lite And Style Signature",
        "--register-comparison artifacts/phase2/analysis/smollm3_pybiber_register_generation_vs_corpus",
    )
    for phrase in stale_phrases:
        if phrase in markdown:
            errors.append(
                f"{md_relative} contains stale generated-output pybiber instruction: {phrase}"
            )


def _check_paper_figure_readiness(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/figures/paper_figure_readiness_audit.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        errors.append(f"{csv_relative} expected 5 figure rows, found {len(rows)}")
    for row in rows:
        figure_id = row.get("figure_id", "<missing figure>")
        expected = {
            "status": "ready_for_visual_review",
            "has_viewbox": "True",
            "title_present": "True",
            "forbidden_svg_phrases": "",
            "pdf_export_present": "True",
            "pdf_export_valid": "True",
            "png_export_present": "True",
            "png_export_valid": "True",
            "warnings": "",
        }
        for field, value in expected.items():
            if row.get(field) != value:
                errors.append(
                    f"{csv_relative} {figure_id} {field}="
                    f"{row.get(field)!r}, expected {value!r}"
                )
        if int(row.get("editable_text_elements", "0") or 0) < 8:
            errors.append(f"{csv_relative} {figure_id} has too few editable text elements")

    md_relative = "docs/experiments/paper_figure_readiness_audit.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready_for_visual_review`.",
        "complements the rendered-layout review",
        "remaining figure work is final venue sizing/export",
        "PDF+PNG",
        "| Figure 5 | ready_for_visual_review",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing figure-readiness phrase: {phrase}")


def _check_paper_figure_visual_review(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/figures/paper_figure_visual_review.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        errors.append(f"{csv_relative} expected 5 figure rows, found {len(rows)}")
    for row in rows:
        figure_id = row.get("figure_id", "<missing figure>")
        if row.get("review_status") != "visual_review_passed":
            errors.append(
                f"{csv_relative} {figure_id} review_status={row.get('review_status')!r}, "
                "expected 'visual_review_passed'"
            )
        for field in ("svg_path", "png_review_path"):
            relative = row.get(field, "")
            if not relative or not (root / relative).exists():
                errors.append(f"{csv_relative} {figure_id} missing existing {field}: {relative}")
        if not row.get("findings"):
            errors.append(f"{csv_relative} {figure_id} missing visual-review findings")

    md_relative = "docs/experiments/paper_figure_visual_review.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Review status: `visual_review_passed`.",
        "does not replace final venue sizing",
        "one label per feature cluster",
        "| Figure 5 | visual_review_passed",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing figure-visual-review phrase: {phrase}")


def _check_paper_table_readiness(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/tables/paper_table_readiness_audit.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        errors.append(f"{csv_relative} expected 6 table rows, found {len(rows)}")
    for row in rows:
        table_id = row.get("table_id", "<missing table>")
        expected = {
            "status": "ready_for_venue_styling",
            "markdown_heading_present": "True",
            "markdown_table_present": "True",
            "latex_label_present": "True",
            "latex_caption_present": "True",
            "forbidden_markdown_phrases": "",
            "forbidden_latex_phrases": "",
            "warnings": "",
        }
        for field, value in expected.items():
            if row.get(field) != value:
                errors.append(
                    f"{csv_relative} {table_id} {field}="
                    f"{row.get(field)!r}, expected {value!r}"
                )
        data_rows = int(row.get("markdown_data_rows", "0") or 0)
        min_rows = int(row.get("min_markdown_data_rows", "0") or 0)
        if data_rows < min_rows:
            errors.append(
                f"{csv_relative} {table_id} has {data_rows} markdown data rows, "
                f"expected at least {min_rows}"
            )

    md_relative = "docs/experiments/paper_table_readiness_audit.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready_for_venue_styling`.",
        "does not replace final venue-specific typography",
        "| Appendix Table A3 | ready_for_venue_styling",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing table-readiness phrase: {phrase}")


def _check_paper_table_typography_review(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/tables/paper_table_typography_review.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        errors.append(f"{csv_relative} expected 6 table rows, found {len(rows)}")
    for row in rows:
        table_id = row.get("table_id", "<missing table>")
        expected = {
            "review_status": "typography_review_passed",
            "uses_booktabs": "True",
            "uses_tabularx": "True",
            "uses_compact_size": "True",
            "no_vertical_rules": "True",
            "warnings": "",
        }
        for field, value in expected.items():
            if row.get(field) != value:
                errors.append(
                    f"{csv_relative} {table_id} {field}="
                    f"{row.get(field)!r}, expected {value!r}"
                )

    md_relative = "docs/experiments/paper_table_typography_review.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Review status: `typography_review_passed`.",
        "generic booktabs/tabularx typography",
        "does not replace target venue template adaptation",
        "| Appendix Table A3 | typography_review_passed",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing table-typography phrase: {phrase}")


def _check_paper_claim_evidence_audit(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/claims/paper_claim_evidence_audit.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_claim = {row.get("claim_id", ""): row for row in rows}
    expected_claims = set(EXPECTED_CLAIM_IDS)
    missing = sorted(expected_claims - set(by_claim))
    if missing:
        errors.append(f"{csv_relative} missing claim IDs: {', '.join(missing)}")
    if len(rows) != len(expected_claims):
        errors.append(f"{csv_relative} expected {len(expected_claims)} rows, found {len(rows)}")
    for claim_id in sorted(expected_claims & set(by_claim)):
        row = by_claim[claim_id]
        if row.get("status") != "ready":
            errors.append(
                f"{csv_relative} {claim_id} status={row.get('status')!r}, expected 'ready'"
            )
        if row.get("missing_paths"):
            errors.append(f"{csv_relative} {claim_id} has missing_paths: {row.get('missing_paths')}")
        if row.get("missing_fields"):
            errors.append(
                f"{csv_relative} {claim_id} has missing_fields: {row.get('missing_fields')}"
            )
        if not row.get("existing_paths") and not row.get("referenced_claims"):
            errors.append(f"{csv_relative} {claim_id} has no evidence path or referenced claim")

    md_relative = "docs/experiments/paper_claim_evidence_audit.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready`.",
        "claim evidence map covers C1-C14",
        "| C14 | ready |",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing claim-evidence phrase: {phrase}")


def _check_paper_claim_language_audit(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/claims/paper_claim_language_audit.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    claim_rows = [row for row in rows if row.get("check_type") == "claim"]
    forbidden_rows = [row for row in rows if row.get("check_type") == "forbidden"]
    if len(claim_rows) != 14:
        errors.append(f"{csv_relative} expected 14 claim rows, found {len(claim_rows)}")
    if len(forbidden_rows) != 27:
        errors.append(
            f"{csv_relative} expected 27 forbidden-overclaim rows, found {len(forbidden_rows)}"
        )
    for row in rows:
        claim_id = row.get("claim_id", "<missing claim>")
        if row.get("status") != "ready":
            errors.append(
                f"{csv_relative} {claim_id} status={row.get('status')!r}, expected 'ready'"
            )
        for field in ("missing_required", "missing_caveats", "forbidden_hits"):
            if row.get(field):
                errors.append(f"{csv_relative} {claim_id} has {field}: {row.get(field)}")

    md_relative = "docs/experiments/paper_claim_language_audit.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready`.",
        "Forbidden overclaim checks: `27/27` passed.",
        "| C14 | ready | 2/2 | 0/0 | none |",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing claim-language phrase: {phrase}")


def _check_paper_abstract_conclusion_audit(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/manuscript/paper_abstract_conclusion_audit.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_checks = {
        "abstract_alignment",
        "conclusion_alignment",
        "package_index_alignment",
    }
    observed_checks = {row.get("check_id", "") for row in rows}
    missing = sorted(expected_checks - observed_checks)
    if missing:
        errors.append(f"{csv_relative} missing checks: {', '.join(missing)}")
    if len(rows) != len(expected_checks):
        errors.append(f"{csv_relative} expected {len(expected_checks)} rows, found {len(rows)}")
    for row in rows:
        check_id = row.get("check_id", "<missing check>")
        if row.get("status") != "ready":
            errors.append(
                f"{csv_relative} {check_id} status={row.get('status')!r}, expected 'ready'"
            )
        for field in ("forbidden_hits", "missing_required"):
            if row.get(field):
                errors.append(f"{csv_relative} {check_id} has {field}: {row.get(field)}")

    md_relative = "docs/experiments/paper_abstract_conclusion_audit.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready`.",
        "bounded slop-style thesis",
        "| abstract_alignment | ready | 8/8 | none | none |",
        "| conclusion_alignment | ready | 6/6 | none | none |",
        "| package_index_alignment | ready | 8/8 | none | none |",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing abstract/conclusion phrase: {phrase}")


def _check_paper_manuscript_structure_audit(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/manuscript/paper_manuscript_structure_audit.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_checks = {
        "main_heading_order",
        "result_subsection_order",
        "total_word_count",
        "abstract_word_count",
        "conclusion_word_count",
        "figure_caption_count",
        "table_caption_count",
        "reproducibility_appendix_present",
        "heading_count",
    }
    observed_checks = {row.get("check_id", "") for row in rows}
    missing = sorted(expected_checks - observed_checks)
    if missing:
        errors.append(f"{csv_relative} missing checks: {', '.join(missing)}")
    for row in rows:
        check_id = row.get("check_id", "<missing check>")
        if row.get("status") != "ready":
            errors.append(
                f"{csv_relative} {check_id} status={row.get('status')!r}, expected 'ready'"
            )
        if row.get("warnings"):
            errors.append(f"{csv_relative} {check_id} has warnings: {row.get('warnings')}")

    md_relative = "docs/experiments/paper_manuscript_structure_audit.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready`.",
        "section order, length bounds, result-section coverage, captions, and appendix presence",
        "| result_subsection_order | ready | 5/5 | all expected Result 4.x subsections present in order | none |",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing manuscript-structure phrase: {phrase}")


def _check_paper_limitations_audit(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/manuscript/paper_limitations_audit.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_checks = {
        "bounded_open_ladders",
        "tier1_validation_boundary",
        "sparse_denominators",
        "dpo_preference_caveat",
        "pybiber_generation_boundary",
        "eqbench_aggregate_boundary",
        "phase4_human_validation_boundary",
        "reduced_grid_boundary",
        "forbidden_overclaims",
    }
    observed_checks = {row.get("check_id", "") for row in rows}
    missing = sorted(expected_checks - observed_checks)
    if missing:
        errors.append(f"{csv_relative} missing checks: {', '.join(missing)}")
    for row in rows:
        check_id = row.get("check_id", "<missing check>")
        if row.get("status") != "ready":
            errors.append(
                f"{csv_relative} {check_id} status={row.get('status')!r}, "
                "expected 'ready'"
            )
        for field in ("missing_required", "forbidden_hits"):
            if row.get(field):
                errors.append(f"{csv_relative} {check_id} has {field}: {row.get(field)}")

    md_relative = "docs/experiments/paper_limitations_audit.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready`.",
        "bounded-scope, pybiber, EQ-Bench, Tier-1, and Phase 4 human-validation boundaries",
        "| pybiber_generation_boundary | ready | 3/3 | none | none |",
        "| reduced_grid_boundary | ready | 3/3 | none | none |",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing limitations phrase: {phrase}")


def _check_paper_submission_exit_audit(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/submission/paper_submission_exit_audit.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_item = {row.get("exit_item", ""): row for row in rows}
    expected_statuses = {
        "figure_table_finalization": "venue_review_pending",
        "manuscript_body": "ready",
        "claim_discipline": "ready",
        "tier1_boundaries": "ready",
        "phase4_human_perceptibility": "human_validation_pending",
        "bibliography_and_package": "ready",
        "overall_submission_status": "blocked",
    }
    missing = sorted(set(expected_statuses) - set(by_item))
    if missing:
        errors.append(f"{csv_relative} missing exit items: {', '.join(missing)}")
    for exit_item, status in expected_statuses.items():
        row = by_item.get(exit_item)
        if not row:
            continue
        if row.get("status") != status:
            errors.append(
                f"{csv_relative} {exit_item} status={row.get('status')!r}, "
                f"expected {status!r}"
            )
    overall_action = by_item.get("overall_submission_status", {}).get("next_action", "")
    for blocker in ("figure_table_finalization", "phase4_human_perceptibility"):
        if blocker not in overall_action:
            errors.append(f"{csv_relative} overall row does not name blocker: {blocker}")

    md_relative = "docs/experiments/paper_submission_exit_audit.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Submission status: `blocked`.",
        "paper-draft readiness from submission readiness",
        "| phase4_human_perceptibility | human_validation_pending |",
        "| overall_submission_status | blocked |",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing submission-exit phrase: {phrase}")


def _check_paper_submission_draft_bundle(root: Path, errors: list[str]) -> None:
    manifest_relative = "artifacts/paper/submission/draft_bundle/MANIFEST.csv"
    manifest_path = root / manifest_relative
    if not manifest_path.exists():
        return
    rows = _read_csv_rows(manifest_path)
    if len(rows) != 31:
        errors.append(f"{manifest_relative} expected 31 bundle rows, found {len(rows)}")
    expected_categories = {
        "manuscript",
        "tables",
        "references",
        "claim_control",
        "readiness",
        "figure_svg",
        "figure_pdf",
        "figure_png",
    }
    categories = {row.get("category", "") for row in rows}
    missing_categories = sorted(expected_categories - categories)
    if missing_categories:
        errors.append(f"{manifest_relative} missing categories: {', '.join(missing_categories)}")
    required_bundle_paths = {
        "manuscript/paper_manuscript_draft.md",
        "tables/paper_latex_table_drafts.tex",
        "references/paper_references.bib",
        "figures/svg/figure1_pybiber_register_selected.svg",
        "figures/pdf/figure1_pybiber_register_selected.pdf",
        "figures/png/figure1_pybiber_register_selected.png",
        "review_controls/paper_package_index.md",
        "review_controls/paper_claim_matrix.md",
        "review_controls/paper_reader_glossary.md",
        "review_controls/paper_measurement_synthesis.md",
        "review_controls/paper_submission_exit_audit.md",
        "review_controls/paper_review_hygiene_audit.md",
        "review_controls/paper_release_inventory.md",
        "review_controls/paper_venue_sizing_inventory.md",
    }
    bundle_paths = {row.get("bundle_path", "") for row in rows}
    missing_bundle_paths = sorted(required_bundle_paths - bundle_paths)
    if missing_bundle_paths:
        errors.append(
            f"{manifest_relative} missing bundle paths: {', '.join(missing_bundle_paths)}"
        )
    for row in rows:
        bundle_path = row.get("bundle_path", "<missing bundle_path>")
        if row.get("status") != "ready":
            errors.append(
                f"{manifest_relative} {bundle_path} status={row.get('status')!r}, "
                "expected 'ready'"
            )
        sha256 = row.get("sha256", "")
        if len(sha256) != 64 or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            errors.append(f"{manifest_relative} {bundle_path} has invalid sha256: {sha256!r}")
        size = row.get("size_bytes", "")
        if not size.isdigit() or int(size) <= 0:
            errors.append(f"{manifest_relative} {bundle_path} has invalid size_bytes: {size!r}")
        file_path = root / "artifacts/paper/submission/draft_bundle" / bundle_path
        if row.get("status") == "ready" and not file_path.exists():
            errors.append(f"{manifest_relative} {bundle_path} points to missing bundle file")
            continue
        if row.get("status") == "ready" and file_path.exists():
            actual_size = str(file_path.stat().st_size)
            actual_sha = _sha256(file_path)
            if size != actual_size:
                errors.append(
                    f"{manifest_relative} {bundle_path} size_bytes={size!r}, "
                    f"expected {actual_size!r}"
                )
            if sha256 != actual_sha:
                errors.append(
                    f"{manifest_relative} {bundle_path} sha256={sha256!r}, "
                    f"expected {actual_sha!r}"
                )
            source_relative = row.get("source_path", "")
            source_path = root / source_relative
            if not source_relative:
                errors.append(f"{manifest_relative} {bundle_path} missing source_path")
            elif not source_path.exists():
                errors.append(
                    f"{manifest_relative} {bundle_path} source_path={source_relative!r} "
                    "points to missing source file"
                )
            else:
                source_size = str(source_path.stat().st_size)
                source_sha = _sha256(source_path)
                if size != source_size:
                    errors.append(
                        f"{manifest_relative} {bundle_path} size_bytes={size!r}, "
                        f"expected source size {source_size!r}"
                    )
                if sha256 != source_sha:
                    errors.append(
                        f"{manifest_relative} {bundle_path} sha256={sha256!r}, "
                        f"expected source sha256 {source_sha!r}"
                    )

    readme_relative = "artifacts/paper/submission/draft_bundle/README.md"
    readme_path = root / readme_relative
    if not readme_path.exists():
        return
    readme = _read(root, readme_relative)
    required_phrases = (
        "venue-neutral draft bundle",
        "not a submission certificate",
        "Ready files: `31`.",
        "Missing files: `0`.",
        "package index plus claim/readiness controls",
        "`MANIFEST.csv`",
    )
    for phrase in required_phrases:
        if phrase not in readme:
            errors.append(f"{readme_relative} missing draft-bundle phrase: {phrase}")


def _check_paper_venue_decision_checklist(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/submission/paper_venue_decision_checklist.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
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
    by_item = {row.get("item", ""): row for row in rows}
    missing = sorted(expected_items - set(by_item))
    if missing:
        errors.append(f"{csv_relative} missing venue-decision items: {', '.join(missing)}")
    if len(rows) != len(expected_items):
        errors.append(f"{csv_relative} expected {len(expected_items)} rows, found {len(rows)}")
    for item in sorted(expected_items & set(by_item)):
        row = by_item[item]
        if row.get("status") != "decision_pending":
            errors.append(
                f"{csv_relative} {item} status={row.get('status')!r}, "
                "expected 'decision_pending'"
            )
        for field in ("current_evidence", "decision_needed", "next_action"):
            if not row.get(field):
                errors.append(f"{csv_relative} {item} missing {field}")

    md_relative = "docs/experiments/paper_venue_decision_checklist.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Venue decision status: `venue_decision_pending`.",
        "target venue is selected",
        "| target_venue | decision_pending |",
        "`paper_venue_sizing_inventory.md` records current dimensions",
        "`paper_venue_sizing_inventory.md` records current table environments and widths",
        "| human_labeling_requirement | decision_pending |",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing venue-decision phrase: {phrase}")


def _check_paper_venue_adaptation_runbook(root: Path, errors: list[str]) -> None:
    md_relative = "docs/experiments/paper_venue_adaptation_runbook.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Current status: venue decisions are `decision_pending`.",
        "Procedure after venue selection:",
        "uv run slop-render-paper-figures",
        "uv run slop-audit-paper-table-typography",
        "## Per-Figure Decision Matrix",
        "| Figure 5 | Phase 4 Tier-3 candidate AF |",
        "## Per-Table Decision Matrix",
        "| Appendix Table A2 | Selected pybiber bootstrap intervals |",
        "## Venue-Neutral Packing Default",
        "Main text figures: Figure 1, Figure 2, Figure 3, and Figure 4.",
        "Figure 5 and the appendix guardrail tables should move out of the main text",
        "uv run slop-audit-paper-submission-exit --root /home/user/slop",
        "uv run slop-materialize-paper-submission-bundle --root /home/user/slop",
        "paper_submission_exit_audit.csv` reports `overall_submission_status=ready`",
        "draft_bundle/MANIFEST.csv` has been refreshed",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing venue-runbook phrase: {phrase}")


def _check_paper_venue_sizing_inventory(root: Path, errors: list[str]) -> None:
    md_relative = "docs/experiments/paper_venue_sizing_inventory.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Purpose: make the remaining venue-formatting blocker concrete.",
        "## Figure Sizing Snapshot",
        "Each figure has:",
        "## Table Sizing Snapshot",
        "Current blocker summary: figure/table assets are draft-ready",
        "Use the per-figure and per-table decision matrices",
        "per-artifact decisions needed to close those rows",
        "## Default Packing Assumption",
        "| Figure 5 | Appendix or supplement by default |",
        "| Table 4 | Main text |",
        "move Figure 5, Figure 4, and appendix guardrail tables before removing caveats",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing venue-sizing phrase: {phrase}")

    figure_rows = _read_csv_rows(root / "artifacts/paper/figures/paper_figure_readiness_audit.csv")
    for row in figure_rows:
        figure_id = row.get("figure_id", "<missing figure>")
        path = row.get("path", "")
        width = row.get("width_pt", "")
        height = row.get("height_pt", "")
        try:
            inches = f"{float(width) / 72:.2f} x {float(height) / 72:.2f} in"
        except ValueError:
            inches = ""
        expected_parts = (
            f"{figure_id} | `{path}`",
            f"{figure_id} | `{path}` | {width} x {height} pt",
            f"{width} x {height} pt",
            inches,
        )
        for part in expected_parts:
            if part and part not in markdown:
                errors.append(f"{md_relative} missing venue-sizing figure detail: {part}")

    table_rows = _read_csv_rows(root / "artifacts/paper/tables/paper_table_typography_review.csv")
    for row in table_rows:
        table_id = row.get("table_id", "<missing table>")
        environment = row.get("table_environment", "")
        width = row.get("width_target", "")
        expected = f"{table_id} | `{environment}` | `{width}`"
        if expected not in markdown:
            errors.append(f"{md_relative} missing venue-sizing table detail: {expected}")


def _check_paper_reproduction_runbook(root: Path, errors: list[str]) -> None:
    md_relative = "docs/experiments/paper_reproduction_runbook.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "This runbook is for paper-package refreshes",
        "uv run slop-summarize-eqbench-intervals --bootstrap-samples 500 --seed 1729",
        "uv run slop-audit-paper-claim-language",
        "uv run slop-audit-paper-review-hygiene",
        "uv run slop-audit-paper-release-inventory",
        "uv run slop-audit-phase4-human-package",
        "uv run slop-materialize-phase4-human-handoff --root /home/user/slop",
        "uv run slop-summarize-phase4-human-labels",
        "The current package is intentionally unlabeled.",
        "phase4_human_perceptibility_summary.csv` remains unlabeled unless real",
        "uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop",
        "release_code` rows for the source",
        "draft_bundle/MANIFEST.csv` records 31 ready",
        "This runbook does not relaunch:",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing reproduction-runbook phrase: {phrase}")


def _check_paper_package_index(root: Path, errors: list[str]) -> None:
    md_relative = "docs/experiments/paper_package_index.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Use `docs/experiments/paper_claim_matrix.md` as the authority",
        "paper_measurement_synthesis.md",
        "The bounded paper argues that slop style is feature-specific, stage-specific",
        "Full pybiber is a Phase 1 corpus-side register result",
        "EQ-Bench Slop Score is a benchmark bridge",
        "The pybiber DPO contrast is register-specific",
        "chosen-responses-are-more-slop-like",
        "mostly lexical-density evidence rather than a mechanism claim",
        "they do not turn the aggregate score into AF or mechanism evidence",
        "retained English-filtered Dolma sample",
        "Phase 4 detector-derived families remain machine-detectable candidates",
        "`phase4_production_runbook.md`",
        "completed 512-document exact Tier-3 rerun",
        "release-code and artifact checksums",
        "The package is paper-draft ready but not submission ready.",
        "uv run slop-materialize-paper-submission-bundle --root /home/user/slop",
        "uv run slop-check-paper-package --root /home/user/slop",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing package-index phrase: {phrase}")


def _check_paper_measurement_synthesis(root: Path, errors: list[str]) -> None:
    md_relative = "docs/experiments/paper_measurement_synthesis.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "No single measurement layer is treated as the whole phenomenon.",
        "Full pybiber register",
        "EQ-Bench Slop Score",
        "Teacher-forced propensity",
        "Free-running emission and compounding",
        "Detector-derived Tier 3",
        "The pybiber result supplies the broad register backbone.",
        "increase in hedging or intensification",
        "chosen responses look more descriptive and expository",
        "chosen-equals-slop interpretation",
        "aggregate lexical-density scores and feature-specific propensity measure different things",
        "Keep propensity, sampled emission, and compounding separate.",
        "candidate-only until human labels are available",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing measurement-synthesis phrase: {phrase}")


def _check_paper_reader_glossary(root: Path, errors: list[str]) -> None:
    md_relative = "docs/experiments/paper_reader_glossary.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Purpose: give new readers a compact map",
        "Slop style",
        "Teacher-forced propensity",
        "Free-running emission",
        "Amplification factor (AF)",
        "Neutral-normalized AF",
        "Pybiber",
        "EQ-Bench Slop Score",
        "its stage path can disagree with feature-specific propensity",
        "Tier 1",
        "Tier 2",
        "Tier 3",
        "OLMo ladder",
        "SmolLM3 ladder",
        "Dolci DPO",
        "pybiber reads chosen as more descriptive/expository and rejected as more personal/narrative",
        "Candidate human-perceived slop",
        "Use corpus rates and full pybiber",
        "Use teacher-forced propensity and AF",
        "Use EQ-Bench as an aggregate public bridge",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing reader-glossary phrase: {phrase}")


def _check_paper_reviewer_faq(root: Path, errors: list[str]) -> None:
    md_relative = "docs/experiments/paper_reviewer_faq.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "In this paper, \"slop style\" means measurable stylistic markers",
        "The paper separates four localization layers:",
        "Self-conditioning: whether later markers become more likely",
        "EQ-Bench is an aggregate public benchmark bridge",
        "Detector-derived Tier 3 is a candidate-feature discovery surface",
        "The supported thesis is narrower:",
        "The pybiber result is corpus-side.",
        "they should not be described as full generated-output pybiber evidence.",
        "descriptive/expository on this register layer",
        "uniformly more slop-like",
        "EQ-Bench Slop Score is included as an aggregate benchmark bridge.",
        "aggregate EQ-Bench movement can disagree with a feature-specific propensity path",
        "Dolci DPO includes synthetic and Delta Learning-style construction regimes.",
        "The OLMo pretraining reference is an English-filtered retained Dolma",
        "not required for the current claim matrix",
        "`rule_of_three_approx` is demoted for independent publication-grade claims",
        "`stock_openers_closers` is a derived pooled convenience view",
        "The reduced grid is the production estimand.",
        "a second ladder shows related style pressure",
        "Follow-up offers have a very large AF point estimate",
        "Claims about detector-discovered human-perceived slop families remain provisional",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing reviewer-faq phrase: {phrase}")


def _check_paper_release_inventory(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/submission/paper_release_inventory.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_categories = {
        "feature_definitions",
        "opportunity_definitions",
        "harness_code",
        "runbook",
        "cached_statistics",
    }
    categories = {row.get("category", "") for row in rows}
    missing_categories = sorted(expected_categories - categories)
    if missing_categories:
        errors.append(f"{csv_relative} missing categories: {', '.join(missing_categories)}")
    if len(rows) < 25:
        errors.append(f"{csv_relative} expected at least 25 release rows, found {len(rows)}")
    required_paths = {
        "src/slop_sftdiv/features/tier1_matchers.py",
        "src/slop_sftdiv/features/eqbench_slop.py",
        "src/slop_sftdiv/features/pybiber_full.py",
        "src/slop_sftdiv/propensity.py",
        "src/slop_sftdiv/cli/teacher_forced_propensity.py",
        "src/slop_sftdiv/cli/free_running_emission.py",
        "src/slop_sftdiv/cli/import_phase4_blind_labels.py",
        "src/slop_sftdiv/cli/summarize_phase4_label_agreement.py",
        "src/slop_sftdiv/cli/apply_phase4_label_adjudication.py",
        "src/slop_sftdiv/cli/check_paper_package.py",
        "src/slop_sftdiv/cli/materialize_paper_submission_bundle.py",
        "src/slop_sftdiv/cli/audit_paper_reproducibility_manifest.py",
        "docs/experiments/paper_reproduction_runbook.md",
        "docs/experiments/phase3_production_runbook.md",
        "docs/experiments/phase4_production_runbook.md",
        "artifacts/stage1/census/feature_rates_by_corpus.parquet",
        "artifacts/stage1/census/preference_pair_deltas.parquet",
        "artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv",
        "artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv",
        (
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_detector_tier3_matcher_specs.json"
        ),
    }
    row_paths = {row.get("path", "") for row in rows}
    missing_paths = sorted(required_paths - row_paths)
    if missing_paths:
        errors.append(f"{csv_relative} missing release paths: {', '.join(missing_paths)}")
    for row in rows:
        path = row.get("path", "<missing path>")
        if row.get("status") != "ready":
            errors.append(
                f"{csv_relative} {path} status={row.get('status')!r}, expected 'ready'"
            )
        size = row.get("size_bytes", "")
        if not size.isdigit() or int(size) <= 0:
            errors.append(f"{csv_relative} {path} has invalid size_bytes: {size!r}")
        release_path = root / path
        if row.get("status") == "ready":
            if not release_path.exists():
                errors.append(f"{csv_relative} {path} points to missing release file")
            elif size.isdigit() and str(release_path.stat().st_size) != size:
                errors.append(
                    f"{csv_relative} {path} size_bytes={size!r}, "
                    f"expected {release_path.stat().st_size!r}"
                )
        for field in ("release_role", "caveat"):
            if not row.get(field):
                errors.append(f"{csv_relative} {path} missing {field}")

    md_relative = "docs/experiments/paper_release_inventory.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready`.",
        "feature definitions, opportunity definitions, harness code, runbooks, and cached paper statistics",
        "| feature_definitions |",
        "| cached_statistics |",
        "not a license review or final archival deposit",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing release-inventory phrase: {phrase}")


def _check_paper_numeric_claims_audit(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/manuscript/paper_numeric_claims_audit.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = {
        "pybiber_narrative_drop",
        "pybiber_expository_shift",
        "pybiber_not_hedging",
        "pybiber_dpo_contrast",
        "eqbench_olmo_stage_scores",
        "eqbench_smollm3_stage_scores",
        "olmo_slop_lexicon_af",
        "olmo_slop_lexicon_preference_data",
        "olmo_slop_lexicon_free_run",
        "olmo_slop_lexicon_compounding",
        "smollm3_slop_lexicon_free_run",
        "cross_ladder_correlation",
        "phase4_attribution_counts",
        "phase4_tier3_selected_af",
        "phase4_tier3_sparse_af",
    }
    by_id = {row.get("check_id", ""): row for row in rows}
    missing = sorted(expected_ids - set(by_id))
    if missing:
        errors.append(f"{csv_relative} missing numeric claim checks: {', '.join(missing)}")
    if len(rows) != len(expected_ids):
        errors.append(f"{csv_relative} expected {len(expected_ids)} rows, found {len(rows)}")
    for check_id in sorted(expected_ids & set(by_id)):
        row = by_id[check_id]
        if row.get("status") != "ready":
            errors.append(
                f"{csv_relative} {check_id} status={row.get('status')!r}, "
                "expected 'ready'"
            )
        if row.get("missing_text"):
            errors.append(f"{csv_relative} {check_id} has non-empty missing_text")
        if not row.get("source_path"):
            errors.append(f"{csv_relative} {check_id} missing source_path")
        if not row.get("expected_text"):
            errors.append(f"{csv_relative} {check_id} missing expected_text")

    md_relative = "docs/experiments/paper_numeric_claims_audit.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready`.",
        "headline manuscript numbers match their source CSV/JSON artifacts",
        "| pybiber_narrative_drop | ready |",
        "| phase4_tier3_sparse_af | ready |",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing numeric-claims phrase: {phrase}")


def _check_paper_review_hygiene_audit(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/manuscript/paper_review_hygiene_audit.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_checks = {
        "manuscript_main_body_hygiene",
        "author_placeholder_hygiene",
        "bundle_manuscript_matches_source",
        "paper_facing_bundle_hygiene",
    }
    observed_checks = {row.get("check_id", "") for row in rows}
    missing = sorted(expected_checks - observed_checks)
    if missing:
        errors.append(f"{csv_relative} missing checks: {', '.join(missing)}")
    if len(rows) != len(expected_checks):
        errors.append(f"{csv_relative} expected {len(expected_checks)} rows, found {len(rows)}")
    for row in rows:
        check_id = row.get("check_id", "<missing check>")
        if row.get("status") != "ready":
            errors.append(
                f"{csv_relative} {check_id} status={row.get('status')!r}, "
                "expected 'ready'"
            )
        for field in ("forbidden_hits", "missing_required"):
            if row.get(field):
                errors.append(f"{csv_relative} {check_id} has {field}: {row.get(field)}")

    md_relative = "docs/experiments/paper_review_hygiene_audit.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready`.",
        "avoid local paths, internal tool commands, draft placeholders",
        "| manuscript_main_body_hygiene | ready |",
        "| paper_facing_bundle_hygiene | ready |",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing review-hygiene phrase: {phrase}")


def _check_paper_source_provenance_audit(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/submission/paper_source_provenance_audit.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_checks = {
        "dolci_dpo_source_card_mixture",
        "dolci_dpo_not_pure_human_preference",
        "dolma_sample_boundary",
        "pybiber_generated_output_boundary",
        "smollm3_apo_source_boundary",
    }
    observed_checks = {row.get("check_id", "") for row in rows}
    missing = sorted(expected_checks - observed_checks)
    if missing:
        errors.append(f"{csv_relative} missing checks: {', '.join(missing)}")
    if len(rows) != len(expected_checks):
        errors.append(f"{csv_relative} expected {len(expected_checks)} rows, found {len(rows)}")
    for row in rows:
        check_id = row.get("check_id", "<missing check>")
        if row.get("status") != "ready":
            errors.append(
                f"{csv_relative} {check_id} status={row.get('status')!r}, "
                "expected 'ready'"
            )
        for field in ("missing_required", "forbidden_hits"):
            if row.get(field):
                errors.append(f"{csv_relative} {check_id} has {field}: {row.get(field)}")
        required_hits = row.get("required_hits", "")
        if not re.fullmatch(r"\d+/\d+", required_hits):
            errors.append(f"{csv_relative} {check_id} has invalid required_hits")
            continue
        hit_count, total_count = (int(part) for part in required_hits.split("/"))
        if hit_count != total_count:
            errors.append(
                f"{csv_relative} {check_id} required_hits={required_hits}, "
                "expected all required phrases present"
            )

    md_relative = "docs/experiments/paper_source_provenance_audit.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready`.",
        "source-card and manuscript-facing interpretation boundaries",
        "| dolci_dpo_source_card_mixture | ready |",
        "| pybiber_generated_output_boundary | ready |",
        "not a new empirical measurement",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing source-provenance phrase: {phrase}")


def _check_paper_readiness_audit(root: Path, errors: list[str]) -> None:
    md_relative = "docs/experiments/paper_readiness_audit.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "package index, measurement synthesis, manuscript draft, caption appendix",
        "measurement synthesis for the pybiber/EQ-Bench/propensity/generation/compounding/detector hierarchy",
        "paper-package materializers",
        "release-code source files",
        "annotator quickstart",
        "submission-blocked by venue finalization and Phase 4 human-validation",
        "Submission Blocker Burn-Down Order",
        "Collect two independent full-package blind label",
        "Done means the summary contains real labeled rows",
        "resolve every",
        "`decision_pending` row in the venue-decision checklist",
        "overall_submission_status` as\n   `ready`",
        "detector clusters as human-perceived slop",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing readiness-audit phrase: {phrase}")


def _check_paper_terminology_audit(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/manuscript/paper_terminology_audit.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_checks = {
        "slop_style_scope",
        "mechanism_separation",
        "localization_surface_boundary",
        "amplification_spectrum",
        "tier1_policy",
        "pybiber_boundary",
        "eqbench_boundary",
        "phase4_boundary",
        "negative_thesis",
    }
    observed_checks = {row.get("check_id", "") for row in rows}
    missing = sorted(expected_checks - observed_checks)
    if missing:
        errors.append(f"{csv_relative} missing checks: {', '.join(missing)}")
    if len(rows) != len(expected_checks):
        errors.append(f"{csv_relative} expected {len(expected_checks)} rows, found {len(rows)}")
    for row in rows:
        check_id = row.get("check_id", "<missing check>")
        if row.get("status") != "ready":
            errors.append(
                f"{csv_relative} {check_id} status={row.get('status')!r}, expected 'ready'"
            )
        if row.get("missing_required"):
            errors.append(
                f"{csv_relative} {check_id} has missing_required: "
                f"{row.get('missing_required')}"
            )

    md_relative = "docs/experiments/paper_terminology_audit.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready`.",
        "defines the core slop-style measurement terms",
        "| mechanism_separation | ready | 4/4 | none |",
        "| localization_surface_boundary | ready | 4/4 | none |",
        "| phase4_boundary | ready | 2/2 | none |",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing terminology phrase: {phrase}")


def _check_paper_reproducibility_manifest(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/submission/paper_reproducibility_manifest.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        errors.append(f"{csv_relative} is empty")
        return
    by_path = {row.get("path", ""): row for row in rows}
    categories = {row.get("category", "") for row in rows}
    if "release_code" not in categories:
        errors.append(f"{csv_relative} missing release_code category")
    expected_release_code_paths = {
        "src/slop_sftdiv/features/tier1_matchers.py",
        "src/slop_sftdiv/features/eqbench_slop.py",
        "src/slop_sftdiv/features/pybiber_full.py",
        "src/slop_sftdiv/propensity.py",
        "src/slop_sftdiv/cli/teacher_forced_propensity.py",
        "src/slop_sftdiv/cli/free_running_emission.py",
        "src/slop_sftdiv/generation_text.py",
        "src/slop_sftdiv/cli/import_phase4_blind_labels.py",
        "src/slop_sftdiv/cli/summarize_phase4_label_agreement.py",
        "src/slop_sftdiv/cli/apply_phase4_label_adjudication.py",
        "src/slop_sftdiv/cli/check_paper_package.py",
        "src/slop_sftdiv/cli/materialize_paper_submission_bundle.py",
        "src/slop_sftdiv/cli/audit_paper_reproducibility_manifest.py",
    }
    expected_paths = {
        *expected_release_code_paths,
        "docs/experiments/paper_manuscript_draft.md",
        "docs/experiments/paper_numeric_claims_audit.md",
        "docs/experiments/paper_terminology_audit.md",
        "docs/experiments/paper_review_hygiene_audit.md",
        "docs/experiments/paper_source_provenance_audit.md",
        "docs/experiments/paper_release_inventory.md",
        "docs/experiments/source_card_notes.md",
        "docs/experiments/phase4_human_annotation_codebook.md",
        "docs/experiments/phase4_human_annotation_handoff.md",
        "docs/experiments/phase4_human_labeling_execution_checklist.md",
        "docs/experiments/paper_limitations_audit.md",
        "docs/experiments/paper_references.bib",
        "artifacts/paper/figures/figure1_pybiber_register_selected.svg",
        "artifacts/paper/figures/submission_exports/figure1_pybiber_register_selected.pdf",
        "artifacts/paper/figures/submission_exports/figure1_pybiber_register_selected.png",
        "artifacts/stage1/census/phase1_pybiber_register_intervals.csv",
        "artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv",
        "artifacts/paper/manuscript/paper_numeric_claims_audit.csv",
        "artifacts/paper/manuscript/paper_terminology_audit.csv",
        "artifacts/paper/manuscript/paper_review_hygiene_audit.csv",
        "artifacts/paper/submission/paper_source_provenance_audit.csv",
        "artifacts/paper/submission/paper_submission_exit_audit.csv",
        "artifacts/paper/submission/paper_release_inventory.csv",
        (
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_summary.csv"
        ),
        (
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/README.md"
        ),
        (
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/MANIFEST.csv"
        ),
        (
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/annotator/README.md"
        ),
        (
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/coordinator/"
            "phase4_human_labeling_execution_checklist.md"
        ),
        (
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/annotator/annotator_a/"
            "phase4_human_perceptibility_blind_full_annotator_a.csv"
        ),
        (
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "human_annotation_handoff/annotator/annotator_b/"
            "phase4_human_perceptibility_blind_full_annotator_b.csv"
        ),
        (
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_full.csv"
        ),
        (
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_full_map.csv"
        ),
    }
    missing_paths = sorted(expected_paths - set(by_path))
    if missing_paths:
        errors.append(
            f"{csv_relative} missing reproducibility paths: {', '.join(missing_paths)}"
        )
    for relative in sorted(expected_release_code_paths & set(by_path)):
        if by_path[relative].get("category") != "release_code":
            errors.append(
                f"{csv_relative} {relative} category={by_path[relative].get('category')!r}, "
                "expected 'release_code'"
            )
    for row in rows:
        relative = row.get("path", "<missing path>")
        if row.get("status") != "ready":
            errors.append(
                f"{csv_relative} {relative} status={row.get('status')!r}, "
                "expected 'ready'"
            )
        sha256 = row.get("sha256", "")
        if len(sha256) != 64 or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            errors.append(f"{csv_relative} {relative} has invalid sha256: {sha256!r}")
        size = row.get("size_bytes", "")
        if not size.isdigit() or int(size) <= 0:
            errors.append(f"{csv_relative} {relative} has invalid size_bytes: {size!r}")
        file_path = root / relative
        if row.get("status") == "ready" and not file_path.exists():
            errors.append(f"{csv_relative} {relative} points to missing file")
            continue
        if row.get("status") == "ready" and file_path.exists():
            actual_size = str(file_path.stat().st_size)
            actual_sha = _sha256(file_path)
            if size != actual_size:
                errors.append(
                    f"{csv_relative} {relative} size_bytes={size!r}, "
                    f"expected {actual_size!r}"
                )
            if sha256 != actual_sha:
                errors.append(
                    f"{csv_relative} {relative} sha256={sha256!r}, "
                    f"expected {actual_sha!r}"
                )

    md_relative = "docs/experiments/paper_reproducibility_manifest.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready`.",
        "Machine-readable manifest:",
        "SHA-256 checksums",
        "| release_code |",
        "not a venue submission certificate",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing reproducibility-manifest phrase: {phrase}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_retired_feature_language(root: Path, errors: list[str]) -> None:
    for path in sorted((root / "docs/experiments").glob("paper_*.md")):
        text = path.read_text(encoding="utf-8")
        for phrase in ("Biber-lite", "Biber lite", "biber_lite", "biber lite"):
            if phrase in text:
                errors.append(
                    f"{path.relative_to(root)} uses retired feature language: {phrase}"
                )


def _check_stale_phrases(root: Path, errors: list[str]) -> None:
    for relative in TEXT_PATHS_TO_CHECK + ("docs/experiments/paper_manuscript_draft.md",):
        path = root / relative
        if not path.exists():
            continue
        text = _read(root, relative)
        for phrase in STALE_PHRASES:
            if phrase in text:
                errors.append(f"{relative} contains stale phrase: {phrase}")


def _check_backtick_paths(root: Path, errors: list[str]) -> None:
    for relative in TEXT_PATHS_TO_CHECK:
        path = root / relative
        if not path.exists():
            continue
        text = _read(root, relative)
        for candidate in _extract_backtick_paths(text):
            if "*" in candidate:
                continue
            if not (root / candidate).exists():
                errors.append(f"{relative} references missing path: {candidate}")


def _extract_backtick_paths(text: str) -> set[str]:
    paths: set[str] = set()
    for match in re.finditer(r"`([^`]+)`", text):
        candidate = match.group(1).strip()
        if candidate.startswith(("docs/", "artifacts/", "src/", "EXPERIMENTS.md")):
            paths.add(candidate)
    return paths


def _check_figure_table_references(root: Path, errors: list[str]) -> None:
    manifest_path = root / "docs/experiments/paper_figure_table_manifest.md"
    manuscript_path = root / "docs/experiments/paper_manuscript_draft.md"
    if not manifest_path.exists() or not manuscript_path.exists():
        return

    manifest = manifest_path.read_text(encoding="utf-8")
    manuscript = manuscript_path.read_text(encoding="utf-8")
    manifest_figures = _ids(manifest, "Figure")
    manuscript_figures = _ids(manuscript, "Figure")
    manifest_tables = _ids(manifest, "Table")
    manuscript_tables = _ids(manuscript, "Table")

    for figure in sorted(manuscript_figures - manifest_figures):
        errors.append(f"manuscript references {figure}, absent from figure/table manifest")
    for figure in sorted(manifest_figures - manuscript_figures):
        errors.append(f"figure/table manifest defines {figure}, absent from manuscript")
    for table in sorted(manuscript_tables - manifest_tables):
        errors.append(f"manuscript references {table}, absent from figure/table manifest")
    for table in sorted({"Table 1", "Table 2", "Table 4"} - manuscript_tables):
        errors.append(f"manuscript does not reference required main table: {table}")


def _ids(text: str, prefix: str) -> set[str]:
    return {f"{prefix} {match}" for match in re.findall(rf"\b{prefix} ([0-9]+)\b", text)}


def _check_interval_reference_coverage(root: Path, errors: list[str]) -> None:
    manifest_path = root / "docs/experiments/paper_figure_table_manifest.md"
    if not manifest_path.exists():
        return
    manifest = manifest_path.read_text(encoding="utf-8")
    required_interval_paths = (
        "artifacts/stage1/census/phase1_pybiber_register_intervals.csv",
        "artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv",
    )
    for relative in required_interval_paths:
        if relative not in manifest:
            errors.append(f"figure/table manifest missing interval artifact reference: {relative}")
    required_caption_phrases = (
        "document-bootstrap intervals over retained rows",
        "bootstrap intervals over valid generated outputs",
    )
    for phrase in required_caption_phrases:
        if phrase not in manifest:
            errors.append(f"figure/table manifest missing interval caption phrase: {phrase}")


def _check_caption_coverage(root: Path, errors: list[str]) -> None:
    manuscript_path = root / "docs/experiments/paper_manuscript_draft.md"
    if not manuscript_path.exists():
        return
    manuscript = manuscript_path.read_text(encoding="utf-8")
    expected_figures = {f"Figure {index}" for index in range(1, 6)}
    expected_tables = {f"Table {index}" for index in range(1, 6)}

    manuscript_figure_captions = _caption_ids(manuscript, "Figure")
    manuscript_table_captions = _caption_ids(manuscript, "Table")

    for figure in sorted(expected_figures - manuscript_figure_captions):
        errors.append(f"manuscript is missing caption for {figure}")
    for table in sorted(expected_tables - manuscript_table_captions):
        errors.append(f"manuscript is missing caption for {table}")


def _caption_ids(text: str, prefix: str) -> set[str]:
    return {
        f"{prefix} {match}"
        for match in re.findall(rf"(?m)^\*\*{prefix} ([0-9]+)\.", text)
    }


def _check_claim_coverage(root: Path, errors: list[str]) -> None:
    expected = set(EXPECTED_CLAIM_IDS)
    matrix_path = root / "docs/experiments/paper_claim_matrix.md"
    map_path = root / "docs/experiments/paper_claim_evidence_map.md"
    readiness_path = root / "docs/experiments/paper_readiness_audit.md"
    results_path = root / "docs/experiments/paper_results_draft.md"

    if matrix_path.exists():
        matrix_ids = _claim_table_ids(matrix_path.read_text(encoding="utf-8"))
        _compare_claim_sets(errors, "claim matrix", matrix_ids, expected)
    if map_path.exists():
        map_text = map_path.read_text(encoding="utf-8")
        map_ids = _claim_table_ids(map_text)
        _compare_claim_sets(errors, "claim evidence map", map_ids, expected)
        coverage_ids = _claim_ids_in_section(map_text, "## Manuscript Coverage By Section")
        _compare_claim_sets(errors, "claim evidence map section coverage", coverage_ids, expected)
        placement_ids = _claim_ids_in_section(map_text, "## Claim Strength For Main Text")
        _compare_claim_sets(errors, "claim evidence map placement table", placement_ids, expected)
    if readiness_path.exists():
        readiness_ids = _claim_ids_in_section(
            readiness_path.read_text(encoding="utf-8"),
            "## Claim Readiness Summary",
        )
        _compare_claim_sets(errors, "readiness claim bands", readiness_ids, expected)
    if results_path.exists():
        results_ids = _claim_ids_in_claim_id_lines(results_path.read_text(encoding="utf-8"))
        expected_results = expected - {"C14"}
        _compare_claim_sets(errors, "results draft claim-id lines", results_ids, expected_results)


def _claim_table_ids(text: str) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(r"^\|\s*(C[0-9]+)\s*\|", text, flags=re.MULTILINE)
    }


def _claim_ids_in_section(text: str, heading: str) -> set[str]:
    try:
        start = text.index(heading)
    except ValueError:
        return set()
    remainder = text[start + len(heading) :]
    next_heading = re.search(r"\n##\s+", remainder)
    section = remainder[: next_heading.start()] if next_heading else remainder
    return set(re.findall(r"\bC[0-9]+\b", section))


def _claim_ids_in_claim_id_lines(text: str) -> set[str]:
    ids: set[str] = set()
    for line in text.splitlines():
        if line.startswith("Claim ID"):
            ids.update(re.findall(r"\bC[0-9]+\b", line))
    return ids


def _compare_claim_sets(
    errors: list[str],
    label: str,
    actual: set[str],
    expected: set[str],
) -> None:
    missing = sorted(expected - actual, key=_claim_sort_key)
    extra = sorted(actual - expected, key=_claim_sort_key)
    if missing:
        errors.append(f"{label} missing claim IDs: {', '.join(missing)}")
    if extra:
        errors.append(f"{label} has unexpected claim IDs: {', '.join(extra)}")


def _claim_sort_key(claim_id: str) -> int:
    return int(claim_id.removeprefix("C"))


def _check_citations(root: Path, errors: list[str], warnings: list[str]) -> None:
    bib_path = root / "docs/experiments/paper_references.bib"
    if not bib_path.exists():
        return
    bib = bib_path.read_text(encoding="utf-8")
    cited_by_path: dict[str, set[str]] = {}
    for relative in CITATION_TEXT_PATHS:
        path = root / relative
        if path.exists():
            cited_by_path[relative] = set(
                re.findall(r"@([A-Za-z0-9_:-]+)", path.read_text(encoding="utf-8"))
            )
    cited = set().union(*cited_by_path.values()) if cited_by_path else set()
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+)", bib))
    for relative, path_citations in cited_by_path.items():
        for key in sorted(path_citations - bib_keys):
            errors.append(f"{relative} cites missing BibTeX key: {key}")
    for key in sorted(bib_keys - cited):
        warnings.append(f"BibTeX key is not cited in manuscript-facing drafts: {key}")


def _check_reference_source_coverage(root: Path, errors: list[str]) -> None:
    bib_path = root / "docs/experiments/paper_references.bib"
    source_path = root / "docs/experiments/paper_reference_sources.md"
    if not bib_path.exists() or not source_path.exists():
        return
    bib = bib_path.read_text(encoding="utf-8")
    sources = source_path.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+)", bib))
    source_keys = set(re.findall(r"^\|\s*`([^`]+)`\s*\|", sources, flags=re.MULTILINE))
    for key in sorted(bib_keys - source_keys):
        errors.append(f"BibTeX key missing verified source row: {key}")
    for key in sorted(source_keys - bib_keys):
        errors.append(f"verified source row missing BibTeX entry: {key}")


def _check_paper_citation_audit(root: Path, errors: list[str]) -> None:
    csv_relative = "artifacts/paper/references/paper_citation_audit.csv"
    csv_path = root / csv_relative
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_checks = {
        "cited_keys_have_bib_entries",
        "bib_keys_have_source_rows",
        "source_rows_have_bib_entries",
        "source_rows_are_populated",
        "source_rows_have_urls",
        "uncited_bib_keys_recorded",
    }
    observed_checks = {row.get("check_id", "") for row in rows}
    missing = sorted(expected_checks - observed_checks)
    if missing:
        errors.append(f"{csv_relative} missing checks: {', '.join(missing)}")
    if len(rows) != len(expected_checks):
        errors.append(f"{csv_relative} expected {len(expected_checks)} rows, found {len(rows)}")
    for row in rows:
        check_id = row.get("check_id", "<missing check>")
        if row.get("status") != "ready":
            errors.append(
                f"{csv_relative} {check_id} status={row.get('status')!r}, expected 'ready'"
            )
        if row.get("missing_or_invalid"):
            errors.append(
                f"{csv_relative} {check_id} has missing_or_invalid: "
                f"{row.get('missing_or_invalid')}"
            )

    md_relative = "docs/experiments/paper_citation_audit.md"
    md_path = root / md_relative
    if not md_path.exists():
        return
    markdown = _read(root, md_relative)
    required_phrases = (
        "Readiness status: `ready`.",
        "cited manuscript keys, BibTeX entries, and verified source-inventory rows",
        "| uncited_bib_keys_recorded | ready |",
    )
    for phrase in required_phrases:
        if phrase not in markdown:
            errors.append(f"{md_relative} missing citation-audit phrase: {phrase}")


def main() -> None:
    args = build_parser().parse_args()
    report = run_check_paper_package(args.root)
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    if report.ok:
        print("paper package check passed")
        return
    for error in report.errors:
        print(f"ERROR: {error}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
