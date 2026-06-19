from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from slop_sftdiv.cli.check_paper_package import run_check_paper_package


def _write(path: Path, text: str | bytes = "placeholder\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")


def _rewrite_handoff_manifest_row(
    root: Path,
    *,
    bundle_path: str,
    source_path: str | None = None,
    redaction_status: str | None = None,
) -> None:
    manifest_path = (
        root
        / "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "human_annotation_handoff/MANIFEST.csv"
    )
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    bundle_file = manifest_path.parent / bundle_path
    for row in rows:
        if row["bundle_path"] == bundle_path:
            if source_path is not None:
                row["source_path"] = source_path
            if redaction_status is not None:
                row["redaction_status"] = redaction_status
            row["size_bytes"] = str(bundle_file.stat().st_size)
            row["sha256"] = hashlib.sha256(bundle_file.read_bytes()).hexdigest()
            break
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _experiments_md() -> str:
    return (
        "# Experiments\n\n"
        "Full pybiber is currently a corpus-side Phase 1 register surface "
        "for retained pretrain/SFT/DPO samples; generated-output full pybiber is\n"
        "not part of the active paper claim surface.\n\n"
        "Run full pybiber register features over\n"
        "the retained Phase 1 corpus-side samples only.\n"
    )


def _claim_ids() -> list[str]:
    return [f"C{index}" for index in range(1, 15)]


def _claim_table() -> str:
    lines = [
        "| Claim ID | Paper claim | Evidence | Strength | Caveat | Allowed wording |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(f"| {claim_id} | Claim | Evidence | Strong | Caveat | Wording |" for claim_id in _claim_ids())
    return "\n".join(lines) + "\n"


def _claim_evidence_map() -> str:
    coverage = [
        "| Manuscript section | Claim IDs | Evidence objects |",
        "|---|---|---|",
        "| Section 1 | C1, C2, C3, C4, C5, C6, C7 | Evidence |",
        "| Section 2 | C8, C9, C10, C11, C12, C13, C14 | Evidence |",
    ]
    placement = [
        "| Placement | Claim IDs | Treatment |",
        "|---|---|---|",
        "| Main | C1, C2, C7, C8, C9 | Use |",
        "| Caveated | C3, C4, C5, C6, C10, C12, C13 | Use |",
        "| Status | C11, C14 | Use |",
    ]
    return (
        "# Claim Evidence Map\n\n"
        "## Claim-To-Manuscript Map\n\n"
        + _claim_table()
        + "\n## Manuscript Coverage By Section\n\n"
        + "\n".join(coverage)
        + "\n\n## Claim Strength For Main Text\n\n"
        + "\n".join(placement)
        + "\n"
    )


def _readiness_audit() -> str:
    return (
        "# Readiness\n\n"
        "The current package is therefore best described as submission-blocked by venue "
        "finalization and Phase 4 human-validation.\n\n"
        "| Area | Status | Evidence | Paper treatment |\n"
        "|---|---|---|---|\n"
        "| Venue-neutral submission draft bundle | Ready | Evidence | A single staging directory now contains the package index, measurement synthesis, manuscript draft, caption appendix, table drafts, references, SVG/PDF/PNG figure candidates, and claim/readiness controls. |\n"
        "| Reader orientation | Ready | Evidence | Use the measurement synthesis for the pybiber/EQ-Bench/propensity/generation/compounding/detector hierarchy. |\n"
        "| Open-release inventory | Ready | Evidence | Maps the open-release deliverable to concrete feature definitions, opportunity definitions, harness code, paper-package materializers, runbooks, and cached paper statistics. |\n"
        "| Reproducibility manifest | Ready | Evidence | Paper-facing docs, figures, audits, release-code source files, Phase 1 pybiber evidence, EQ-Bench interval evidence, and Phase 4 human-labeling materials are indexed with file sizes and SHA-256 checksums. |\n\n"
        "Phase 4 includes an annotator quickstart.\n\n"
        "## Submission Blocker Burn-Down Order\n\n"
        "Collect two independent full-package blind label sheets.\n"
        "Done means the summary contains real labeled rows.\n"
        "resolve every `decision_pending` row in the venue-decision checklist.\n"
        "overall_submission_status` as\n"
        "   `ready`.\n"
        "must not describe detector clusters as human-perceived slop.\n\n"
        "## Claim Readiness Summary\n\n"
        "| Claim band | Claim IDs | Readiness |\n"
        "|---|---|---|\n"
        "| Main | C1, C2, C7, C8, C9 | Ready |\n"
        "| Caveated | C3, C4, C5, C6, C10, C12, C13 | Caveat |\n"
        "| Status | C11, C14 | Status |\n"
    )


def _results_draft() -> str:
    return (
        "# Results\n\n"
        "Claim IDs: C1, C2, C3.\n"
        "Claim IDs: C4, C5, C6.\n"
        "Claim IDs: C7, C8, C9.\n"
        "Claim ID: C10.\n"
        "Claim IDs: C11, C12, C13.\n"
    )


def _caption_block() -> str:
    captions = ["## Figure And Table Captions"]
    captions.extend(
        f"**Figure {index}. Caption.**\nCaption text."
        for index in range(1, 6)
    )
    captions.extend(
        f"**Table {index}. Caption.**\nCaption text."
        for index in range(1, 6)
    )
    return "\n\n".join(captions)


def _precision_status_csv() -> str:
    return (
        "feature,is_core,is_exploratory,is_derived,gate_status,queued,labeled,exact,partial,"
        "false_positive,ambiguous,precision,ambiguous_rate\n"
        "contrastive_negation,True,False,False,pass,220,61,57,1,3,0,0.934426,0.000000\n"
        "rule_of_three_approx,True,False,False,demote,420,50,28,1,19,2,0.560000,0.040000\n"
        "slop_lexicon,True,False,False,pass,420,50,50,0,0,0,1.000000,0.000000\n"
        "stock_closers,True,False,False,pass,420,50,48,0,2,0,0.960000,0.000000\n"
        "stock_openers,True,False,False,pass,420,68,67,0,1,0,0.985294,0.000000\n"
        "stock_openers_closers,False,False,True,derived,0,0,0,0,0,0,,\n"
    )


def _precision_status_md() -> str:
    return (
        "# Precision Validation Status\n\n"
        "| Feature | Core | Derived | Status | Queued | Labeled | Exact | Partial | FP | Ambig | Precision | Ambig rate |\n"
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        "| slop_lexicon | True | False | pass | 420 | 50 | 50 | 0 | 0 | 0 | 1.000000 | 0.000000 |\n"
        "| rule_of_three_approx | True | False | demote | 420 | 50 | 28 | 1 | 19 | 2 | 0.560000 | 0.040000 |\n"
        "| stock_closers | True | False | pass | 420 | 50 | 48 | 0 | 2 | 0 | 0.960000 | 0.000000 |\n"
        "| stock_openers_closers | False | True | derived | 0 | 0 | 0 | 0 | 0 | 0 |  |  |\n\n"
        "Core features passing the current gate: `4/5`.\n"
        "Core features not passing publication-grade Phase 1 regex claims: `1`.\n"
    )


def _phase4_readiness_csv() -> str:
    header = (
        "feature,candidate_label,cluster,rows,expected_rows,row_count_status,"
        "schema_status,labeled,blank_human_perceived_slop,"
        "blank_shaib_taxonomy_label,unique_sources,unique_record_ids,label_status\n"
    )
    rows = [
        (
            f"phase4_feature_{index},Candidate,cluster,{100},{100},ok,ok,0,"
            "100,100,1,10,blank\n"
        )
        for index in range(10)
    ]
    return header + "".join(rows)


def _phase4_readiness_md() -> str:
    return (
        "# Phase 4 Human Annotation Readiness\n\n"
        "Readiness status: `ready_for_labeling`.\n"
        "Package rows: `1000` across `10` candidate features.\n"
        "Pilot sheet size: `20` rows per feature.\n\n"
        "Blind full-package labeling sheet: `blind_full.csv`\n"
        "The package remains\n"
        "unlabeled and should still be treated as detector-only evidence until\n"
        "human labels are collected and summarized.\n"
    )


def _phase4_human_summary_csv() -> str:
    header = (
        "feature,candidate_label,cluster,total,labeled,blank,yes,context_dependent,"
        "no,unclear,human_perceived_or_context_count,human_perceived_or_context_rate,"
        "yes_rate,top_shaib_taxonomy_label,venn_region\n"
    )
    features = [
        "phase4_ig_additive_transition",
        "phase4_ig_benefit_intensifier",
        "phase4_ig_careful_reasoning",
        "phase4_ig_code_boilerplate",
        "phase4_ig_conversation_greeting",
        "phase4_ig_followup_offer",
        "phase4_ig_prescriptive_instruction",
        "phase4_ig_process_framing",
        "phase4_ig_quantity_time_recipe",
        "phase4_ig_response_constraint",
    ]
    return header + "".join(
        f"{feature},Candidate,cluster,100,0,100,0,0,0,0,0,,,,unlabeled\n"
        for feature in features
    )


def _phase4_human_summary_md() -> str:
    return (
        "# Phase 4 Human-Perceptibility Summary\n\n"
        "Blank rows\n"
        "mean human annotation has not been completed. A candidate should remain\n"
        "detector-only until enough rows are labeled and the Venn region is\n"
        "`detector_and_human_perceived`.\n\n"
        "| Feature | Labeled / total | Yes | Context | No | Unclear | Top taxonomy | Venn region |\n"
        "|---|---:|---:|---:|---:|---:|---|---|\n"
        "| phase4_ig_followup_offer | 0/100 | 0 | 0 | 0 | 0 |  | unlabeled |\n"
    )


def _phase4_blind_csv(row_count: int) -> str:
    rows = ["blind_id,matched_text,snippet,human_perceived_slop,shaib_taxonomy_label,notes\n"]
    for index in range(1, row_count + 1):
        rows.append(
            f"blind_{index:04d},matched text {index},snippet {index},,,\n"
        )
    return "".join(rows)


def _phase4_blind_map_csv(row_count: int) -> str:
    rows = [
        "blind_id,annotation_id,feature,candidate_label,cluster,source,record_id,matched_text,snippet\n"
    ]
    for index in range(1, row_count + 1):
        rows.append(
            "blind_{index:04d},annotation_{index:04d},phase4_feature_{feature},"
            "Candidate,cluster,source,record-{index},matched text {index},snippet {index}\n".format(
                index=index,
                feature=(index - 1) % 10,
            )
        )
    return "".join(rows)


def _phase4_codebook_md() -> str:
    return (
        "# Phase 4 Human Annotation Codebook\n\n"
        "Does this matched span contribute to a recognizable AI slop-style effect\n"
        "in this local snippet?\n\n"
        "Use `human_perceived_slop=yes` only when the matched text is locally\n"
        "perceptible as stylistic excess.\n\n"
        "Annotators should edit only `human_perceived_slop`, `shaib_taxonomy_label`,\n"
        "and `notes`.\n"
        "Do not edit `blind_id`, `matched_text`, or `snippet`.\n"
        "| Feature | Yes when |\n"
        "|---|---|\n"
        "| `phase4_ig_process_framing` | The model narrates its own process. |\n"
        "| `phase4_ig_followup_offer` | The answer adds unsolicited assistance. |\n\n"
        "Compare `human_perceived_slop` before taxonomy labels.\n"
        "The pilot should be reported as a pilot if the full package remains unlabeled.\n"
    )


def _phase4_handoff_md() -> str:
    return (
        "# Phase 4 Human Annotation Handoff\n\n"
        "Readiness status: `ready`.\n"
        "Annotator-facing files contain only blinded IDs, matched text, snippets, "
        "label columns, and notes; coordinator-only files preserve maps and source "
        "identifiers.\n\n"
        "| Audience | Category | Bundle path | Status | Redaction status |\n"
        "|---|---|---|---|---|\n"
        "| annotator | annotator_guidance | `annotator/README.md` | ready | not_applicable |\n"
        "| annotator | annotator_sheet | `annotator/phase4_human_perceptibility_blind_full.csv` | ready | redacted |\n"
        "| annotator | annotator_assignment | `annotator/annotator_a/phase4_human_perceptibility_blind_full_annotator_a.csv` | ready | redacted |\n"
        "| coordinator | coordinator_guidance | `coordinator/phase4_human_labeling_execution_checklist.md` | ready | not_applicable |\n"
    )


def _phase4_labeling_checklist_md() -> str:
    return (
        "# Phase 4 Human Labeling Execution Checklist\n\n"
        "This checklist does not report completed labels.\n"
        "Execution status: `ready_for_external_labels`.\n"
        "Share only `human_annotation_handoff/annotator/`.\n"
        "Do not share\n"
        "`coordinator/private_maps/`.\n"
        "`uv run slop-summarize-phase4-label-agreement`\n"
        "`uv run slop-apply-phase4-label-adjudication`\n"
        "`uv run slop-summarize-phase4-human-labels`\n"
        "Keep detector claims machine-detectable only until labels exist.\n"
        "Update C11/C12/C13 in `paper_claim_matrix.md`.\n"
    )


def _phase4_protocol_md() -> str:
    return (
        "# Phase 4 Human-Perceptibility Annotation Protocol\n\n"
        "Full package import writes "
        "`phase4_human_perceptibility_blind_full_labeled.jsonl`.\n"
        "Typo IDs and stale rows for consensus examples are rejected.\n"
        "After agreement and adjudication, summarize "
        "`phase4_human_perceptibility_blind_full_adjudicated.jsonl` into "
        "`phase4_human_perceptibility_summary.csv`.\n"
    )


def _phase4_handoff_readme() -> str:
    return (
        "# Phase 4 Human Annotation Handoff\n\n"
        "Share only `annotator/` with independent annotators.\n"
        "Ready files: `14`.\n"
        "Missing files: `0`.\n"
        "Redacted annotator CSVs: `6`.\n"
        "`annotator/`: quickstart, codebook, generic blinded CSVs, and annotator-specific copies.\n"
        "`coordinator/`: protocol, readiness summary, labeling checklist, private maps, and source package.\n"
        "Annotator fill rules: edit only `human_perceived_slop`,\n"
        "Do not edit `blind_id`,\n"
        "Protected-field edits are rejected during import.\n"
        "`uv run slop-summarize-phase4-label-agreement`\n"
        "`adjudicated_human_perceived_slop`,\n"
        "`uv run slop-apply-phase4-label-adjudication`\n"
        "`uv run slop-summarize-phase4-human-labels`\n"
    )


def _phase4_handoff_manifest_csv(root: Path | None = None) -> str:
    rows = [
        ("annotator_guidance", "annotator", "generated", "annotator/README.md", "not_applicable"),
        (
            "annotator_guidance",
            "annotator",
            "docs/experiments/phase4_human_annotation_codebook.md",
            "annotator/phase4_human_annotation_codebook.md",
            "not_applicable",
        ),
        (
            "annotator_sheet",
            "annotator",
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_pilot_20_each.csv",
            "annotator/phase4_human_perceptibility_blind_pilot_20_each.csv",
            "redacted",
        ),
        (
            "annotator_sheet",
            "annotator",
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_full.csv",
            "annotator/phase4_human_perceptibility_blind_full.csv",
            "redacted",
        ),
        (
            "annotator_assignment",
            "annotator",
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_pilot_20_each.csv",
            "annotator/annotator_a/phase4_human_perceptibility_blind_pilot_20_each_annotator_a.csv",
            "redacted",
        ),
        (
            "annotator_assignment",
            "annotator",
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_pilot_20_each.csv",
            "annotator/annotator_b/phase4_human_perceptibility_blind_pilot_20_each_annotator_b.csv",
            "redacted",
        ),
        (
            "annotator_assignment",
            "annotator",
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_full.csv",
            "annotator/annotator_a/phase4_human_perceptibility_blind_full_annotator_a.csv",
            "redacted",
        ),
        (
            "annotator_assignment",
            "annotator",
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_full.csv",
            "annotator/annotator_b/phase4_human_perceptibility_blind_full_annotator_b.csv",
            "redacted",
        ),
        (
            "coordinator_guidance",
            "coordinator",
            "docs/experiments/phase4_human_perceptibility_protocol.md",
            "coordinator/phase4_human_perceptibility_protocol.md",
            "not_applicable",
        ),
        (
            "coordinator_guidance",
            "coordinator",
            "docs/experiments/phase4_human_annotation_readiness.md",
            "coordinator/phase4_human_annotation_readiness.md",
            "not_applicable",
        ),
        (
            "coordinator_guidance",
            "coordinator",
            "docs/experiments/phase4_human_labeling_execution_checklist.md",
            "coordinator/phase4_human_labeling_execution_checklist.md",
            "not_applicable",
        ),
        (
            "coordinator_map",
            "coordinator",
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_pilot_20_each_map.csv",
            "coordinator/private_maps/phase4_human_perceptibility_blind_pilot_20_each_map.csv",
            "not_applicable",
        ),
        (
            "coordinator_map",
            "coordinator",
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_blind_full_map.csv",
            "coordinator/private_maps/phase4_human_perceptibility_blind_full_map.csv",
            "not_applicable",
        ),
        (
            "coordinator_source",
            "coordinator",
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_annotation_package.jsonl",
            "coordinator/private_maps/phase4_human_perceptibility_annotation_package.jsonl",
            "not_applicable",
        ),
    ]
    lines = [
        "category,audience,source_path,bundle_path,status,redaction_status,size_bytes,sha256\n"
    ]
    for category, audience, source_path, bundle_path, redaction in rows:
        if root is None:
            size = "12"
            sha = "a" * 64
        else:
            path = (
                root
                / "artifacts/phase4/modernbert_detector_combined_v2_clean/"
                "human_annotation_handoff"
                / bundle_path
            )
            size = str(path.stat().st_size)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(
            f"{category},{audience},{source_path},{bundle_path},ready,{redaction},{size},{sha}\n"
        )
    return "".join(lines)


def _figure_readiness_csv() -> str:
    header = (
        "figure_id,path,status,width_pt,height_pt,has_viewbox,editable_text_elements,"
        "title_fragment,title_present,forbidden_svg_phrases,pdf_export_path,"
        "pdf_export_present,pdf_export_valid,png_export_path,png_export_present,"
        "png_export_valid,warnings\n"
    )
    rows = [
        (
            f"Figure {index},artifacts/paper/figures/figure{index}.svg,"
            "ready_for_visual_review,500.000,320.000,True,10,Title,True,,"
            f"artifacts/paper/figures/submission_exports/figure{index}.pdf,True,True,"
            f"artifacts/paper/figures/submission_exports/figure{index}.png,True,True,\n"
        )
        for index in range(1, 6)
    ]
    return header + "".join(rows)


def _figure_readiness_md() -> str:
    return (
        "# Paper Figure Readiness Audit\n\n"
        "Readiness status: `ready_for_visual_review`.\n"
        "This audit checks SVG package hygiene and complements the rendered-layout review; "
        "remaining figure work is final venue sizing/export.\n\n"
        "| Figure | Status |\n"
        "|---|---|\n"
        "| Figure 5 | ready_for_visual_review | PDF+PNG |\n"
    )


def _figure_visual_review_csv() -> str:
    header = "figure_id,svg_path,png_review_path,review_status,findings,next_action\n"
    rows = [
        (
            f"Figure {index},artifacts/paper/figures/figure{index}.svg,"
            f"artifacts/paper/figures/visual_review_png/figure{index}.png,"
            "visual_review_passed,one label per feature cluster,Final venue sizing.\n"
        )
        for index in range(1, 6)
    ]
    return header + "".join(rows)


def _figure_visual_review_md() -> str:
    return (
        "# Paper Figure Rendered-Layout Review\n\n"
        "Review status: `visual_review_passed`.\n"
        "This review does not replace final venue sizing.\n"
        "Figure 4 uses one label per feature cluster.\n\n"
        "| Figure | Status | Finding | Remaining action |\n"
        "|---|---|---|---|\n"
        "| Figure 5 | visual_review_passed | Readable. | Final venue sizing. |\n"
    )


def _table_readiness_csv() -> str:
    header = (
        "table_id,markdown_heading,latex_label,latex_caption,status,"
        "markdown_heading_present,markdown_table_present,markdown_data_rows,"
        "min_markdown_data_rows,latex_label_present,latex_caption_present,"
        "forbidden_markdown_phrases,forbidden_latex_phrases,warnings\n"
    )
    rows = [
        (
            r"Table 1,## Table 1. Data And Measurement Scope,\label{tab:data-scope},"
            r"\caption{Data and measurement scope.},ready_for_venue_styling,"
            "True,True,8,8,True,True,,,\n"
        ),
        (
            r"Table 2,## Table 2. Selected Full-Pybiber Register Means,"
            r"\label{tab:pybiber-means},"
            r"\caption{Selected full-pybiber register means.},ready_for_venue_styling,"
            "True,True,11,11,True,True,,,\n"
        ),
        (
            r"Table 4,## Table 4. Tier-1 Precision-Validation Status,"
            r"\label{tab:tier1-precision},"
            r"\caption{Tier-1 precision-validation status.},ready_for_venue_styling,"
            "True,True,8,8,True,True,,,\n"
        ),
        (
            r"Appendix Table A1,## Appendix Table A1. Claim Strength Bands,"
            r"\label{tab:claim-bands},"
            r"\caption{Claim strength bands.},ready_for_venue_styling,"
            "True,True,3,3,True,True,,,\n"
        ),
        (
            r"Appendix Table A2,## Appendix Table A2. Selected Pybiber Bootstrap Intervals,"
            r"\label{tab:pybiber-intervals},"
            r"\caption{Selected pybiber bootstrap intervals.},ready_for_venue_styling,"
            "True,True,8,8,True,True,,,\n"
        ),
        (
            r"Appendix Table A3,## Appendix Table A3. Paper-Safe Negative Claims,"
            r"\label{tab:negative-claims},"
            r"\caption{Paper-safe negative claims.},ready_for_venue_styling,"
            "True,True,6,6,True,True,,,\n"
        ),
    ]
    return header + "".join(rows)


def _table_readiness_md() -> str:
    return (
        "# Paper Table Readiness Audit\n\n"
        "Readiness status: `ready_for_venue_styling`.\n"
        "This audit checks Markdown/LaTeX table package hygiene; it does not replace final venue-specific typography.\n\n"
        "| Table | Status |\n"
        "|---|---|\n"
        "| Appendix Table A3 | ready_for_venue_styling |\n"
    )


def _table_typography_csv() -> str:
    header = (
        "table_id,latex_label,review_status,table_environment,width_target,uses_booktabs,"
        "uses_tabularx,uses_threeparttable,has_table_notes,uses_compact_size,"
        "no_vertical_rules,warnings,next_action\n"
    )
    rows = [
        r"Table 1,\label{tab:data-scope},typography_review_passed,table*,\textwidth,"
        "True,True,True,True,True,True,,Adapt spacing.\n",
        r"Table 2,\label{tab:pybiber-means},typography_review_passed,table*,\textwidth,"
        "True,True,True,True,True,True,,Adapt spacing.\n",
        r"Table 4,\label{tab:tier1-precision},typography_review_passed,table*,\textwidth,"
        "True,True,True,True,True,True,,Adapt spacing.\n",
        r"Appendix Table A1,\label{tab:claim-bands},typography_review_passed,table,\linewidth,"
        "True,True,False,False,True,True,,Adapt spacing.\n",
        r"Appendix Table A2,\label{tab:pybiber-intervals},typography_review_passed,table*,"
        r"\textwidth,"
        "True,True,False,False,True,True,,Adapt spacing.\n",
        r"Appendix Table A3,\label{tab:negative-claims},typography_review_passed,table*,"
        r"\textwidth,"
        "True,True,False,False,True,True,,Adapt spacing.\n",
    ]
    return header + "".join(rows)


def _table_typography_md() -> str:
    return (
        "# Paper Table Typography Review\n\n"
        "Review status: `typography_review_passed`.\n"
        "This review checks generic booktabs/tabularx typography; it does not replace "
        "target venue template adaptation.\n\n"
        "| Table | Status |\n"
        "|---|---|\n"
        "| Appendix Table A3 | typography_review_passed |\n"
    )


def _claim_language_csv() -> str:
    header = (
        "claim_id,check_type,status,required_phrases,required_present,"
        "caveat_phrases,caveats_present,missing_required,missing_caveats,"
        "forbidden_hits\n"
    )
    claim_rows = "".join(
        f"C{index},claim,ready,1,1,0,0,,,\n" for index in range(1, 15)
    )
    forbidden_rows = "".join(
        f"forbidden_{index},forbidden,ready,0,0,0,0,,,\n" for index in range(1, 28)
    )
    return header + claim_rows + forbidden_rows


def _claim_language_md() -> str:
    return (
        "# Paper Claim Language Audit\n\n"
        "Readiness status: `ready`.\n"
        "Forbidden overclaim checks: `27/27` passed.\n\n"
        "| Claim | Status | Required phrases | Caveat phrases | Missing |\n"
        "|---|---|---:|---:|---|\n"
        "| C14 | ready | 2/2 | 0/0 | none |\n"
    )


def _abstract_conclusion_csv() -> str:
    return (
        "check_id,section_heading,status,required_phrases,required_present,"
        "forbidden_phrases,forbidden_hits,missing_required\n"
        "abstract_alignment,## Abstract,ready,8,8,3,,\n"
        "conclusion_alignment,## 7. Conclusion,ready,6,6,3,,\n"
        "package_index_alignment,docs/experiments/paper_package_index.md,ready,8,8,3,,\n"
    )


def _abstract_conclusion_md() -> str:
    return (
        "# Paper Abstract And Conclusion Audit\n\n"
        "Readiness status: `ready`.\n"
        "This audit checks the bounded slop-style thesis.\n\n"
        "| Check | Status | Required | Forbidden hits | Missing |\n"
        "|---|---|---:|---|---|\n"
        "| abstract_alignment | ready | 8/8 | none | none |\n"
        "| conclusion_alignment | ready | 6/6 | none | none |\n"
        "| package_index_alignment | ready | 8/8 | none | none |\n"
    )


def _manuscript_structure_csv() -> str:
    return (
        "check_id,status,value,expected,warnings\n"
        "main_heading_order,ready,11/11,all expected main headings present in order,\n"
        "result_subsection_order,ready,5/5,all expected Result 4.x subsections present in order,\n"
        "total_word_count,ready,5375 words,4500-9000 words,\n"
        "abstract_word_count,ready,178 words,100-250 words,\n"
        "conclusion_word_count,ready,157 words,80-250 words,\n"
        "figure_caption_count,ready,5,5 figure captions,\n"
        "table_caption_count,ready,5,5 table captions,\n"
        "reproducibility_appendix_present,ready,102 words,appendix heading with at least 20 words,\n"
        "heading_count,ready,23,main headings plus result subsections present,\n"
    )


def _manuscript_structure_md() -> str:
    return (
        "# Paper Manuscript Structure Audit\n\n"
        "Readiness status: `ready`.\n"
        "This audit checks the integrated manuscript's section order, length bounds, "
        "result-section coverage, captions, and appendix presence.\n\n"
        "| Check | Status | Value | Expected | Warnings |\n"
        "|---|---|---:|---|---|\n"
        "| result_subsection_order | ready | 5/5 | all expected Result 4.x subsections present in order | none |\n"
    )


def _limitations_csv() -> str:
    return (
        "check_id,status,required_phrases,required_present,missing_required,forbidden_hits\n"
        "bounded_open_ladders,ready,4,4,,\n"
        "tier1_validation_boundary,ready,4,4,,\n"
        "sparse_denominators,ready,3,3,,\n"
        "dpo_preference_caveat,ready,3,3,,\n"
        "pybiber_generation_boundary,ready,3,3,,\n"
        "eqbench_aggregate_boundary,ready,3,3,,\n"
        "phase4_human_validation_boundary,ready,2,2,,\n"
        "reduced_grid_boundary,ready,3,3,,\n"
        "forbidden_overclaims,ready,0,0,,\n"
    )


def _limitations_md() -> str:
    return (
        "# Paper Limitations Audit\n\n"
        "Readiness status: `ready`.\n"
        "This audit checks the bounded-scope, pybiber, EQ-Bench, Tier-1, "
        "and Phase 4 human-validation boundaries.\n\n"
        "| Check | Status | Required | Forbidden hits | Missing |\n"
        "|---|---|---:|---|---|\n"
        "| pybiber_generation_boundary | ready | 3/3 | none | none |\n"
        "| reduced_grid_boundary | ready | 3/3 | none | none |\n"
    )


def _submission_exit_csv() -> str:
    return (
        "exit_item,status,evidence,required_for_submission,next_action\n"
        "figure_table_finalization,venue_review_pending,evidence,required,next\n"
        "manuscript_body,ready,evidence,required,next\n"
        "claim_discipline,ready,evidence,required,next\n"
        "tier1_boundaries,ready,evidence,required,next\n"
        "phase4_human_perceptibility,human_validation_pending,evidence,required,next\n"
        "bibliography_and_package,ready,evidence,required,next\n"
        "overall_submission_status,blocked,evidence,required,"
        "Resolve pending items: figure_table_finalization; phase4_human_perceptibility\n"
    )


def _submission_exit_md() -> str:
    return (
        "# Paper Submission Exit Audit\n\n"
        "Submission status: `blocked`.\n"
        "This audit separates paper-draft readiness from submission readiness by making "
        "human-validation, venue-review, and final-polish blockers explicit.\n\n"
        "| Exit item | Status | Evidence | Required for submission | Next action |\n"
        "|---|---|---|---|---|\n"
        "| phase4_human_perceptibility | human_validation_pending | evidence | required | next |\n"
        "| overall_submission_status | blocked | evidence | required | next |\n"
    )


def _venue_decision_csv() -> str:
    header = "item,status,current_evidence,decision_needed,next_action\n"
    rows = [
        "target_venue,decision_pending,Evidence,Decision,Next\n",
        "figure_dimensions,decision_pending,Evidence,Decision,Next\n",
        "figure_export_format,decision_pending,Evidence,Decision,Next\n",
        "table_placement,decision_pending,Evidence,Decision,Next\n",
        "appendix_table_inclusion,decision_pending,Evidence,Decision,Next\n",
        "manuscript_template,decision_pending,Evidence,Decision,Next\n",
        "human_labeling_requirement,decision_pending,Evidence,Decision,Next\n",
    ]
    return header + "".join(rows)


def _venue_decision_md() -> str:
    return (
        "# Paper Venue Decision Checklist\n\n"
        "Venue decision status: `venue_decision_pending`.\n"
        "These decisions cannot be finalized until a target venue is selected.\n\n"
        "| Item | Status |\n"
        "|---|---|\n"
        "| target_venue | decision_pending |\n"
        "| figure_dimensions | decision_pending | `paper_venue_sizing_inventory.md` records current dimensions |\n"
        "| table_placement | decision_pending | `paper_venue_sizing_inventory.md` records current table environments and widths |\n"
        "| human_labeling_requirement | decision_pending |\n"
    )


def _venue_runbook_md() -> str:
    return (
        "# Paper Venue Adaptation Runbook\n\n"
        "Current status: venue decisions are `decision_pending`.\n\n"
        "## Figure Adaptation\n\n"
        "Procedure after venue selection:\n\n"
        "Run `uv run slop-render-paper-figures`.\n\n"
        "## Per-Figure Decision Matrix\n\n"
        "| Figure | Current role | Likely placement | Required venue decision | Verification after decision |\n"
        "|---|---|---|---|---|\n"
        "| Figure 5 | Phase 4 Tier-3 candidate AF | Two-column | Decision | Verify |\n\n"
        "## Venue-Neutral Packing Default\n\n"
        "Main text figures: Figure 1, Figure 2, Figure 3, and Figure 4.\n"
        "Figure 5 and the appendix guardrail tables should move out of the main text.\n\n"
        "## Table Adaptation\n\n"
        "Run `uv run slop-audit-paper-table-typography`.\n\n"
        "## Per-Table Decision Matrix\n\n"
        "| Table | Current role | Likely placement | Required venue decision | Verification after decision |\n"
        "|---|---|---|---|---|\n"
        "| Appendix Table A2 | Selected pybiber bootstrap intervals | Appendix or supplement | Decision | Verify |\n\n"
        "## Final Gate\n\n"
        "Run `uv run slop-audit-paper-submission-exit --root /home/user/slop`.\n"
        "Run `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`.\n"
        "`paper_submission_exit_audit.csv` reports `overall_submission_status=ready`.\n"
        "`draft_bundle/MANIFEST.csv` has been refreshed.\n"
    )


def _venue_sizing_inventory_md() -> str:
    figure_rows = "".join(
        f"| Figure {index} | `artifacts/paper/figures/figure{index}.svg` | "
        "500.000 x 320.000 pt | 6.94 x 4.44 in |\n"
        for index in range(1, 6)
    )
    return (
        "# Paper Venue Sizing Inventory\n\n"
        "Purpose: make the remaining venue-formatting blocker concrete.\n\n"
        "Use the per-figure and per-table decision matrices.\n\n"
        "## Figure Sizing Snapshot\n\n"
        "| Figure | SVG artifact | Current size | Approx. inches |\n"
        "|---|---|---:|---:|\n"
        f"{figure_rows}\n"
        "Each figure has:\n\n"
        "## Table Sizing Snapshot\n\n"
        "| Table | Current environment | Current width |\n"
        "|---|---|---|\n"
        "| Table 1 | `table*` | `\\textwidth` |\n\n"
        "| Table 2 | `table*` | `\\textwidth` |\n"
        "| Table 4 | `table*` | `\\textwidth` |\n"
        "| Appendix Table A1 | `table` | `\\linewidth` |\n"
        "| Appendix Table A2 | `table*` | `\\textwidth` |\n"
        "| Appendix Table A3 | `table*` | `\\textwidth` |\n\n"
        "## Default Packing Assumption\n\n"
        "| Figure 5 | Appendix or supplement by default |\n"
        "| Table 4 | Main text |\n"
        "move Figure 5, Figure 4, and appendix guardrail tables before removing caveats.\n\n"
        "Current blocker summary: figure/table assets are draft-ready; "
        "per-artifact decisions needed to close those rows.\n"
    )


def _reproduction_runbook_md() -> str:
    return (
        "# Paper Reproduction Runbook\n\n"
        "This runbook is for paper-package refreshes, not large production runs.\n\n"
        "`uv run slop-summarize-eqbench-intervals --bootstrap-samples 500 --seed 1729`\n"
        "`uv run slop-audit-paper-claim-language`\n"
        "`uv run slop-audit-paper-review-hygiene`\n"
        "`uv run slop-audit-paper-release-inventory`\n"
        "`uv run slop-audit-phase4-human-package`\n"
        "`uv run slop-materialize-phase4-human-handoff --root /home/user/slop`\n"
        "`uv run slop-summarize-phase4-human-labels`\n"
        "The current package is intentionally unlabeled.\n"
        "`phase4_human_perceptibility_summary.csv` remains unlabeled unless real\n"
        "`uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`\n\n"
        "`release_code` rows for the source files named in the release inventory.\n"
        "`draft_bundle/MANIFEST.csv` records 31 ready staged files.\n"
        "This runbook does not relaunch:\n"
    )


def _reviewer_faq_md() -> str:
    return (
        "# Paper Reviewer FAQ\n\n"
        "In this paper, \"slop style\" means measurable stylistic markers.\n"
        "The paper separates four localization layers:\n"
        "Self-conditioning: whether later markers become more likely.\n"
        "EQ-Bench is an aggregate public benchmark bridge.\n"
        "Detector-derived Tier 3 is a candidate-feature discovery surface.\n"
        "The supported thesis is narrower:\n"
        "The pybiber result is corpus-side.\n"
        "they should not be described as full generated-output pybiber evidence.\n"
        "Chosen responses are more descriptive/expository on this register layer.\n"
        "different from a claim that chosen responses are uniformly more slop-like.\n"
        "EQ-Bench Slop Score is included as an aggregate benchmark bridge.\n"
        "aggregate EQ-Bench movement can disagree with a feature-specific propensity path.\n"
        "Dolci DPO includes synthetic and Delta Learning-style construction regimes.\n"
        "The OLMo pretraining reference is an English-filtered retained Dolma.\n"
        "not required for the current claim matrix.\n"
        "`rule_of_three_approx` is demoted for independent publication-grade claims.\n"
        "`stock_openers_closers` is a derived pooled convenience view.\n"
        "The reduced grid is the production estimand.\n"
        "a second ladder shows related style pressure.\n"
        "Follow-up offers have a very large AF point estimate.\n"
        "Claims about detector-discovered human-perceived slop families remain provisional.\n"
    )


def _paper_package_index_md() -> str:
    return (
        "# Paper Package Index\n\n"
        "Use `docs/experiments/paper_claim_matrix.md` as the authority.\n"
        "Read `docs/experiments/paper_measurement_synthesis.md` for the measurement hierarchy.\n"
        "The bounded paper argues that slop style is feature-specific, stage-specific.\n"
        "Full pybiber is a Phase 1 corpus-side register result.\n"
        "EQ-Bench Slop Score is a benchmark bridge.\n"
        "The pybiber DPO contrast is register-specific.\n"
        "chosen-responses-are-more-slop-like.\n"
        "mostly lexical-density evidence rather than a mechanism claim.\n"
        "they do not turn the aggregate score into AF or mechanism evidence.\n"
        "retained English-filtered Dolma sample.\n"
        "Phase 4 detector-derived families remain machine-detectable candidates.\n"
        "`phase4_production_runbook.md`.\n"
        "completed 512-document exact Tier-3 rerun.\n"
        "release-code and artifact checksums.\n"
        "The package is paper-draft ready but not submission ready.\n"
        "uv run slop-materialize-paper-submission-bundle --root /home/user/slop\n"
        "uv run slop-check-paper-package --root /home/user/slop\n"
    )


def _paper_measurement_synthesis_md() -> str:
    return (
        "# Paper Measurement Synthesis\n\n"
        "No single measurement layer is treated as the whole phenomenon.\n"
        "Full pybiber register.\n"
        "EQ-Bench Slop Score.\n"
        "Teacher-forced propensity.\n"
        "Free-running emission and compounding.\n"
        "Detector-derived Tier 3.\n"
        "The pybiber result supplies the broad register backbone.\n"
        "not simply an increase in hedging or intensification.\n"
        "chosen responses look more descriptive and expository.\n"
        "not a pure human-preference or chosen-equals-slop interpretation.\n"
        "aggregate lexical-density scores and feature-specific propensity measure different things.\n"
        "Keep propensity, sampled emission, and compounding separate.\n"
        "candidate-only until human labels are available.\n"
    )


def _paper_reader_glossary_md() -> str:
    return (
        "# Paper Reader Glossary\n\n"
        "Purpose: give new readers a compact map of the paper's recurring terms.\n\n"
        "| Term | Meaning in this paper | Boundary |\n"
        "|---|---|---|\n"
        "| Slop style | Measurable stylistic markers and register features. | Not factual error. |\n"
        "| Pybiber | Full 67-feature Biber-style extractor. | Corpus-side use. |\n"
        "| Tier 1 | Hand-defined regex features. | Precision-scoped. |\n"
        "| Tier 2 | Full pybiber register features. | Corpus-side use. |\n"
        "| Tier 3 | Detector-discovered candidate style families. | Candidate-only. |\n"
        "| EQ-Bench Slop Score | Public aggregate slop-score diagnostic. | Benchmark bridge; its stage path can disagree with feature-specific propensity. |\n"
        "| Teacher-forced propensity | Fixed-context probability measurement. | Decoding-free. |\n"
        "| Free-running emission | Sampled generation feature rates. | Decoding-dependent. |\n"
        "| Amplification factor (AF) | Model probability divided by reference rate. | Denominator-sensitive. |\n"
        "| Neutral-normalized AF | Feature AF divided by matched neutral controls. | Calibration control. |\n"
        "| OLMo ladder | Base, SFT, DPO, and final/RLVR checkpoints. | Main ladder. |\n"
        "| SmolLM3 ladder | Base, SFT, DPO/APO-labeled, and final checkpoints. | Replication. |\n"
        "| Dolci DPO | Chosen/rejected preference-pair sample. | Not pure human preference; pybiber reads chosen as more descriptive/expository and rejected as more personal/narrative. |\n"
        "| Candidate human-perceived slop | Detector-derived style family. | Needs labels. |\n\n"
        "## How To Read The Evidence\n\n"
        "- Use corpus rates and full pybiber for data-side inheritance and register shift.\n"
        "- Use teacher-forced propensity and AF for fixed-context model-side localization.\n"
        "- Use EQ-Bench as an aggregate public bridge, not as the mechanism estimate.\n"
    )


def _reproducibility_manifest_csv(root: Path) -> str:
    paths = [
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
        "docs/experiments/paper_package_index.md",
        "docs/experiments/paper_reader_glossary.md",
        "docs/experiments/paper_measurement_synthesis.md",
        "docs/experiments/paper_manuscript_draft.md",
        "docs/experiments/paper_numeric_claims_audit.md",
        "docs/experiments/paper_terminology_audit.md",
        "docs/experiments/paper_review_hygiene_audit.md",
        "docs/experiments/paper_source_provenance_audit.md",
        "docs/experiments/paper_release_inventory.md",
        "docs/experiments/paper_venue_adaptation_runbook.md",
        "docs/experiments/paper_venue_sizing_inventory.md",
        "docs/experiments/paper_reproduction_runbook.md",
        "docs/experiments/paper_reviewer_faq.md",
        "docs/experiments/paper_claim_evidence_audit.md",
        "docs/experiments/paper_limitations_audit.md",
        "docs/experiments/paper_citation_audit.md",
        "docs/experiments/phase4_human_annotation_codebook.md",
        "docs/experiments/phase4_human_annotation_handoff.md",
        "docs/experiments/phase4_human_labeling_execution_checklist.md",
        "docs/experiments/paper_references.bib",
        "docs/experiments/source_card_notes.md",
        "artifacts/paper/submission/draft_bundle/README.md",
        "artifacts/paper/submission/draft_bundle/MANIFEST.csv",
        "artifacts/paper/figures/figure1_pybiber_register_selected.svg",
        "artifacts/paper/figures/submission_exports/figure1_pybiber_register_selected.pdf",
        "artifacts/paper/figures/submission_exports/figure1_pybiber_register_selected.png",
        "artifacts/paper/figures/submission_exports/figure2_eqbench_stage_scores.pdf",
        "artifacts/paper/figures/submission_exports/figure2_eqbench_stage_scores.png",
        "artifacts/paper/figures/submission_exports/figure3_olmo_slop_lexicon_views.pdf",
        "artifacts/paper/figures/submission_exports/figure3_olmo_slop_lexicon_views.png",
        "artifacts/paper/figures/submission_exports/figure4_cross_ladder_af_scatter.pdf",
        "artifacts/paper/figures/submission_exports/figure4_cross_ladder_af_scatter.png",
        "artifacts/paper/figures/submission_exports/figure5_phase4_tier3_raw_af.pdf",
        "artifacts/paper/figures/submission_exports/figure5_phase4_tier3_raw_af.png",
        "artifacts/stage1/census/phase1_pybiber_register_intervals.csv",
        "artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv",
        "artifacts/paper/manuscript/paper_numeric_claims_audit.csv",
        "artifacts/paper/manuscript/paper_terminology_audit.csv",
        "artifacts/paper/manuscript/paper_review_hygiene_audit.csv",
        "artifacts/paper/submission/paper_source_provenance_audit.csv",
        "artifacts/paper/submission/paper_release_inventory.csv",
        "artifacts/paper/claims/paper_claim_evidence_audit.csv",
        "artifacts/paper/references/paper_citation_audit.csv",
        "artifacts/paper/submission/paper_submission_exit_audit.csv",
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_perceptibility_summary.csv",
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "human_annotation_handoff/README.md",
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "human_annotation_handoff/MANIFEST.csv",
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "human_annotation_handoff/annotator/README.md",
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "human_annotation_handoff/coordinator/"
        "phase4_human_labeling_execution_checklist.md",
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "human_annotation_handoff/annotator/annotator_a/"
        "phase4_human_perceptibility_blind_full_annotator_a.csv",
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "human_annotation_handoff/annotator/annotator_b/"
        "phase4_human_perceptibility_blind_full_annotator_b.csv",
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_perceptibility_blind_pilot_20_each.csv",
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_perceptibility_blind_pilot_20_each_map.csv",
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_perceptibility_blind_full.csv",
        "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_perceptibility_blind_full_map.csv",
    ]
    rows = ["category,path,status,size_bytes,sha256\n"]
    for path in paths:
        resolved = root / path
        category = "release_code" if path.startswith("src/") else "test"
        rows.append(
            f"{category},{path},ready,{resolved.stat().st_size},"
            f"{hashlib.sha256(resolved.read_bytes()).hexdigest()}\n"
        )
    return "".join(rows)


def _draft_bundle_rows() -> list[tuple[str, str, str]]:
    rows = [
        (
            "readiness",
            "docs/experiments/paper_package_index.md",
            "review_controls/paper_package_index.md",
        ),
        (
            "claim_control",
            "docs/experiments/paper_measurement_synthesis.md",
            "review_controls/paper_measurement_synthesis.md",
        ),
        ("manuscript", "docs/experiments/paper_manuscript_draft.md", "manuscript/paper_manuscript_draft.md"),
        ("manuscript", "docs/experiments/paper_caption_appendix_draft.md", "manuscript/paper_caption_appendix_draft.md"),
        ("tables", "docs/experiments/paper_camera_ready_tables.md", "tables/paper_camera_ready_tables.md"),
        ("tables", "docs/experiments/paper_latex_table_drafts.tex", "tables/paper_latex_table_drafts.tex"),
        ("references", "docs/experiments/paper_references.bib", "references/paper_references.bib"),
        ("claim_control", "docs/experiments/paper_claim_matrix.md", "review_controls/paper_claim_matrix.md"),
        ("claim_control", "docs/experiments/paper_claim_evidence_map.md", "review_controls/paper_claim_evidence_map.md"),
        ("claim_control", "docs/experiments/paper_reader_glossary.md", "review_controls/paper_reader_glossary.md"),
        ("readiness", "docs/experiments/paper_submission_exit_audit.md", "review_controls/paper_submission_exit_audit.md"),
        ("readiness", "docs/experiments/paper_review_hygiene_audit.md", "review_controls/paper_review_hygiene_audit.md"),
        ("readiness", "docs/experiments/paper_release_inventory.md", "review_controls/paper_release_inventory.md"),
        ("readiness", "docs/experiments/paper_source_provenance_audit.md", "review_controls/paper_source_provenance_audit.md"),
        ("readiness", "docs/experiments/paper_venue_adaptation_runbook.md", "review_controls/paper_venue_adaptation_runbook.md"),
        ("readiness", "docs/experiments/paper_venue_sizing_inventory.md", "review_controls/paper_venue_sizing_inventory.md"),
    ]
    for stem in [
        "figure1_pybiber_register_selected",
        "figure2_eqbench_stage_scores",
        "figure3_olmo_slop_lexicon_views",
        "figure4_cross_ladder_af_scatter",
        "figure5_phase4_tier3_raw_af",
    ]:
        rows.extend(
            [
                ("figure_svg", f"artifacts/paper/figures/{stem}.svg", f"figures/svg/{stem}.svg"),
                ("figure_pdf", f"artifacts/paper/figures/submission_exports/{stem}.pdf", f"figures/pdf/{stem}.pdf"),
                ("figure_png", f"artifacts/paper/figures/submission_exports/{stem}.png", f"figures/png/{stem}.png"),
            ]
        )
    return rows


def _write_draft_bundle(root: Path) -> None:
    bundle_root = root / "artifacts/paper/submission/draft_bundle"
    for category, source, bundle_path in _draft_bundle_rows():
        suffix = Path(bundle_path).suffix
        source_path = root / source
        if source_path.exists():
            content: str | bytes = source_path.read_bytes()
        elif suffix == ".pdf":
            content = b"%PDF\n"
        elif suffix == ".png":
            content = b"\x89PNG\r\n\x1a\n"
        else:
            content = f"{category} {bundle_path}\n"
        _write(bundle_root / bundle_path, content)
    manifest_rows = ["category,source_path,bundle_path,status,size_bytes,sha256\n"]
    for category, source, bundle_path in _draft_bundle_rows():
        file_path = bundle_root / bundle_path
        manifest_rows.append(
            f"{category},{source},{bundle_path},ready,{file_path.stat().st_size},"
            f"{hashlib.sha256(file_path.read_bytes()).hexdigest()}\n"
        )
    _write(bundle_root / "MANIFEST.csv", "".join(manifest_rows))
    _write(
        bundle_root / "README.md",
        "# Paper Submission Draft Bundle\n\n"
        "This is a venue-neutral draft bundle and not a submission certificate.\n"
        "Ready files: `31`.\n"
        "Missing files: `0`.\n"
        "package index plus claim/readiness controls.\n"
        "`MANIFEST.csv`\n",
    )


def _reproducibility_manifest_md() -> str:
    return (
        "# Paper Reproducibility Manifest\n\n"
        "Machine-readable manifest: "
        "`artifacts/paper/submission/paper_reproducibility_manifest.csv`\n\n"
        "Readiness status: `ready`.\n"
        "| release_code | 13 | 0 |\n"
        "This manifest records SHA-256 checksums and is not a venue submission "
        "certificate.\n"
    )


def _numeric_claims_csv() -> str:
    check_ids = [
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
    ]
    rows = ["check_id,status,source_path,expected_text,missing_text\n"]
    for check_id in check_ids:
        rows.append(f"{check_id},ready,source.csv,expected,\n")
    return "".join(rows)


def _numeric_claims_md() -> str:
    return (
        "# Paper Numeric Claims Audit\n\n"
        "Machine-readable audit: "
        "`artifacts/paper/manuscript/paper_numeric_claims_audit.csv`\n\n"
        "Readiness status: `ready`.\n"
        "This audit checks that headline manuscript numbers match their source "
        "CSV/JSON artifacts after manuscript rounding.\n\n"
        "| Check | Status | Source | Missing |\n"
        "|---|---|---|---|\n"
        "| pybiber_narrative_drop | ready | `source.csv` | none |\n"
        "| phase4_tier3_sparse_af | ready | `source.csv` | none |\n"
    )


def _review_hygiene_csv() -> str:
    return (
        "check_id,status,scope,forbidden_hits,missing_required,purpose\n"
        "manuscript_main_body_hygiene,ready,docs/experiments/paper_manuscript_draft.md,,,"
        "Keeps local paths out of the paper main body.\n"
        "author_placeholder_hygiene,ready,docs/experiments/paper_manuscript_draft.md,,,"
        "Prevents author placeholders.\n"
        "bundle_manuscript_matches_source,ready,"
        "docs/experiments/paper_manuscript_draft.md; "
        "artifacts/paper/submission/draft_bundle/manuscript/paper_manuscript_draft.md,,,"
        "Ensures staged manuscript matches source.\n"
        "paper_facing_bundle_hygiene,ready,draft_bundle manuscript/table/reference files,,,"
        "Keeps paper-facing staged files clean.\n"
    )


def _review_hygiene_md() -> str:
    return (
        "# Paper Review Hygiene Audit\n\n"
        "Machine-readable audit: "
        "`artifacts/paper/manuscript/paper_review_hygiene_audit.csv`\n\n"
        "Readiness status: `ready`.\n"
        "This audit checks that paper-facing files avoid local paths, internal tool "
        "commands, draft placeholders, and author placeholder text.\n\n"
        "| Check | Status | Scope | Forbidden hits | Missing required | Purpose |\n"
        "|---|---|---|---|---|---|\n"
        "| manuscript_main_body_hygiene | ready | docs/experiments/paper_manuscript_draft.md | none | none | Purpose |\n"
        "| paper_facing_bundle_hygiene | ready | draft_bundle manuscript/table/reference files | none | none | Purpose |\n"
    )


def _phase4_production_runbook_md() -> str:
    return (
        "# Phase 4 Production Runbook\n\n"
        "The detector, annotation package, and 512-document exact teacher-forced "
        "continuation already exist under the production directory.\n\n"
        "The completed compute target is the larger 512-document exact sequence-mass "
        "teacher-forced Tier-3 OLMo checkpoint grid.\n\n"
        "The 512 exact continuation is complete. The 128-document results remain "
        "a historical scoping artifact.\n"
    )


def _phase3_production_runbook_md() -> str:
    return (
        "# Phase 3 Production Runbook\n\n"
        "Do not treat generated-output full pybiber as part of the active Phase 3 route.\n"
        "The paper-facing full-pybiber result is the Phase 1 corpus-side register "
        "analysis over retained pretrain/SFT/DPO samples.\n"
        "Older generated-output register-proxy/style-signature artifacts are archived "
        "diagnostics only.\n"
    )


def _release_inventory_csv(root: Path | None = None) -> str:
    rows = [
        ("feature_definitions", "src/slop_sftdiv/features/tier1_matchers.py"),
        ("feature_definitions", "src/slop_sftdiv/features/eqbench_slop.py"),
        ("feature_definitions", "src/slop_sftdiv/features/pybiber_full.py"),
        ("opportunity_definitions", "src/slop_sftdiv/propensity.py"),
        (
            "opportunity_definitions",
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_detector_tier3_matcher_specs.json",
        ),
        ("harness_code", "src/slop_sftdiv/cli/teacher_forced_propensity.py"),
        ("harness_code", "src/slop_sftdiv/cli/free_running_emission.py"),
        ("harness_code", "src/slop_sftdiv/generation_text.py"),
        ("harness_code", "src/slop_sftdiv/cli/import_phase4_blind_labels.py"),
        ("harness_code", "src/slop_sftdiv/cli/summarize_phase4_label_agreement.py"),
        ("harness_code", "src/slop_sftdiv/cli/apply_phase4_label_adjudication.py"),
        ("harness_code", "src/slop_sftdiv/cli/check_paper_package.py"),
        ("harness_code", "src/slop_sftdiv/cli/materialize_paper_submission_bundle.py"),
        ("harness_code", "src/slop_sftdiv/cli/audit_paper_reproducibility_manifest.py"),
        ("runbook", "docs/experiments/paper_reproduction_runbook.md"),
        ("runbook", "docs/experiments/paper_venue_sizing_inventory.md"),
        ("runbook", "docs/experiments/phase3_production_runbook.md"),
        ("runbook", "docs/experiments/phase4_production_runbook.md"),
        ("cached_statistics", "artifacts/stage1/census/feature_rates_by_corpus.parquet"),
        ("cached_statistics", "artifacts/stage1/census/preference_pair_deltas.parquet"),
        ("cached_statistics", "artifacts/stage1/census/phase1_pybiber_register_intervals.csv"),
        (
            "cached_statistics",
            "artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv",
        ),
        (
            "cached_statistics",
            "artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv",
        ),
        (
            "cached_statistics",
            "artifacts/phase3/analysis/"
            "smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_"
            "baselines_data_rates_slop_neutral_rule3_production.csv",
        ),
        (
            "cached_statistics",
            "artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/"
            "olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv",
        ),
    ]
    lines = ["category,path,status,size_bytes,release_role,caveat\n"]
    for category, path in rows:
        size = "12" if root is None else str((root / path).stat().st_size)
        lines.append(f"{category},{path},ready,{size},Release role,Caveat\n")
    return "".join(lines)


def _release_inventory_md() -> str:
    return (
        "# Paper Release Inventory\n\n"
        "Machine-readable inventory: `artifacts/paper/submission/paper_release_inventory.csv`\n\n"
        "Readiness status: `ready`.\n"
        "This inventory maps the EXPERIMENTS.md open-release deliverable to concrete "
        "repository paths: feature definitions, opportunity definitions, harness code, "
        "runbooks, and cached paper statistics. It is an inventory, not a license "
        "review or final archival deposit.\n\n"
        "| Category | Items |\n"
        "|---|---:|\n"
        "| feature_definitions | 3 |\n"
        "| harness_code | 9 |\n"
        "| cached_statistics | 7 |\n"
    )


def _source_provenance_csv() -> str:
    rows = [
        ("dolci_dpo_source_card_mixture", "6/6"),
        ("dolci_dpo_not_pure_human_preference", "3/3"),
        ("dolma_sample_boundary", "3/3"),
        ("pybiber_generated_output_boundary", "2/2"),
        ("smollm3_apo_source_boundary", "2/2"),
    ]
    lines = [
        "check_id,status,scope,required_hits,missing_required,forbidden_hits,purpose\n"
    ]
    lines.extend(
        f"{check_id},ready,scope,{required_hits},,,Purpose\n"
        for check_id, required_hits in rows
    )
    return "".join(lines)


def _source_provenance_md() -> str:
    return (
        "# Paper Source Provenance Audit\n\n"
        "Machine-readable audit: "
        "`artifacts/paper/submission/paper_source_provenance_audit.csv`\n\n"
        "Readiness status: `ready`.\n"
        "This audit checks source-card and manuscript-facing interpretation boundaries. "
        "It is not a new empirical measurement.\n\n"
        "| Check | Status | Required hits | Missing required | Forbidden hits | Purpose |\n"
        "|---|---|---:|---|---|---|\n"
        "| dolci_dpo_source_card_mixture | ready | 6/6 | none | none | Purpose |\n"
        "| pybiber_generated_output_boundary | ready | 2/2 | none | none | Purpose |\n"
    )


def _terminology_csv() -> str:
    check_ids = [
        ("slop_style_scope", "2", "2"),
        ("mechanism_separation", "4", "4"),
        ("localization_surface_boundary", "4", "4"),
        ("amplification_spectrum", "2", "2"),
        ("tier1_policy", "3", "3"),
        ("pybiber_boundary", "2", "2"),
        ("eqbench_boundary", "2", "2"),
        ("phase4_boundary", "2", "2"),
        ("negative_thesis", "2", "2"),
    ]
    rows = ["check_id,status,required_count,matched_count,missing_required,purpose\n"]
    for check_id, required, matched in check_ids:
        rows.append(f"{check_id},ready,{required},{matched},,purpose\n")
    return "".join(rows)


def _terminology_md() -> str:
    return (
        "# Paper Terminology Audit\n\n"
        "Machine-readable audit: "
        "`artifacts/paper/manuscript/paper_terminology_audit.csv`\n\n"
        "Readiness status: `ready`.\n"
        "This audit checks that the integrated manuscript defines the core "
        "slop-style measurement terms and preserves key interpretation boundaries.\n\n"
        "| Check | Status | Matched | Missing | Purpose |\n"
        "|---|---|---:|---|---|\n"
        "| mechanism_separation | ready | 4/4 | none | purpose |\n"
        "| localization_surface_boundary | ready | 4/4 | none | purpose |\n"
        "| phase4_boundary | ready | 2/2 | none | purpose |\n"
    )


def _claim_evidence_csv() -> str:
    rows = [
        "claim_id,status,existing_paths,missing_paths,referenced_claims,missing_fields,evidence_cell,residual_caveat\n"
    ]
    for index in range(1, 15):
        rows.append(
            f"C{index},ready,docs/experiments/paper_claim_matrix.md,,,,"
            "`docs/experiments/paper_claim_matrix.md`,Caveat\n"
        )
    return "".join(rows)


def _claim_evidence_md() -> str:
    return (
        "# Paper Claim Evidence Audit\n\n"
        "Machine-readable audit: `artifacts/paper/claims/paper_claim_evidence_audit.csv`\n\n"
        "Readiness status: `ready`.\n"
        "This audit checks that the claim evidence map covers C1-C14 and that "
        "mapped source paths exist.\n\n"
        "| Claim | Status | Existing paths | Referenced claims | Missing paths | Missing fields |\n"
        "|---|---|---|---|---|---|\n"
        "| C14 | ready | docs/experiments/paper_claim_matrix.md | none | none | none |\n"
    )


def _citation_audit_csv() -> str:
    check_ids = [
        "cited_keys_have_bib_entries",
        "bib_keys_have_source_rows",
        "source_rows_have_bib_entries",
        "source_rows_are_populated",
        "source_rows_have_urls",
        "uncited_bib_keys_recorded",
    ]
    rows = ["check_id,status,observed,missing_or_invalid,notes\n"]
    for check_id in check_ids:
        rows.append(f"{check_id},ready,observed,,notes\n")
    return "".join(rows)


def _citation_audit_md() -> str:
    return (
        "# Paper Citation Audit\n\n"
        "Machine-readable audit: `artifacts/paper/references/paper_citation_audit.csv`\n\n"
        "Readiness status: `ready`.\n"
        "This audit checks cited manuscript keys, BibTeX entries, and verified "
        "source-inventory rows for the paper reference package.\n\n"
        "| Check | Status | Observed | Missing or invalid | Notes |\n"
        "|---|---|---|---|---|\n"
        "| uncited_bib_keys_recorded | ready | observed | none | notes |\n"
    )


def _latex_table_pack() -> str:
    labels = [
        r"\label{tab:data-scope}",
        r"\label{tab:pybiber-means}",
        r"\label{tab:tier1-precision}",
        r"\label{tab:claim-bands}",
        r"\label{tab:pybiber-intervals}",
        r"\label{tab:negative-claims}",
    ]
    captions = [
        r"\caption{Data and measurement scope.}",
        r"\caption{Selected full-pybiber register means.}",
        r"\caption{Tier-1 precision-validation status.}",
        r"\caption{Claim strength bands.}",
        r"\caption{Selected pybiber bootstrap intervals.}",
        r"\caption{Paper-safe negative claims.}",
    ]
    return "\n".join(captions + labels) + "\n"


def _write_minimal_package(root: Path) -> None:
    required_plain = [
        "docs/experiments/paper_tier1_publication_policy.md",
        "docs/experiments/paper_package_index.md",
        "docs/experiments/paper_reader_glossary.md",
        "docs/experiments/paper_measurement_synthesis.md",
        "docs/experiments/precision_validation_status.md",
        "docs/experiments/paper_scaffold.md",
        "docs/experiments/paper_tables.md",
        "docs/experiments/paper_methods_draft.md",
        "docs/experiments/paper_intro_discussion_draft.md",
        "docs/experiments/paper_citation_plan.md",
        "docs/experiments/paper_citation_audit.md",
        "docs/experiments/paper_claim_language_audit.md",
        "docs/experiments/paper_claim_evidence_audit.md",
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
        "docs/experiments/paper_figure_readiness_audit.md",
        "docs/experiments/paper_figure_visual_review.md",
        "docs/experiments/paper_table_readiness_audit.md",
        "docs/experiments/paper_table_typography_review.md",
        "docs/experiments/phase4_human_annotation_readiness.md",
        "docs/experiments/phase4_human_annotation_codebook.md",
        "docs/experiments/phase4_human_annotation_handoff.md",
        "docs/experiments/phase4_human_labeling_execution_checklist.md",
        "docs/experiments/phase4_human_perceptibility_protocol.md",
        "docs/experiments/phase4_human_perceptibility_summary.md",
        "docs/experiments/phase3_production_runbook.md",
        "docs/experiments/phase4_production_runbook.md",
        "docs/experiments/source_card_notes.md",
    ]
    for relative in required_plain:
        _write(root / relative)
    for relative in [
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
    ]:
        _write(root / relative, "# placeholder\n")
    _write(root / "EXPERIMENTS.md", _experiments_md())
    _write(root / "docs/experiments/paper_claim_matrix.md", _claim_table())
    _write(root / "docs/experiments/paper_claim_evidence_map.md", _claim_evidence_map())
    _write(root / "docs/experiments/paper_reader_glossary.md", _paper_reader_glossary_md())
    _write(
        root / "docs/experiments/paper_measurement_synthesis.md",
        _paper_measurement_synthesis_md(),
    )
    _write(root / "docs/experiments/paper_readiness_audit.md", _readiness_audit())
    _write(root / "docs/experiments/paper_results_draft.md", _results_draft())
    _write(root / "docs/experiments/precision_validation_status.md", _precision_status_md())
    _write(
        root / "artifacts/stage1/validation/precision_validation_status.csv",
        _precision_status_csv(),
    )

    for index in range(1, 6):
        _write(root / f"artifacts/paper/figures/figure{index}.svg", "<svg></svg>\n")
    for relative in [
        "artifacts/paper/figures/figure1_pybiber_register_selected.svg",
        "artifacts/paper/figures/figure2_eqbench_stage_scores.svg",
        "artifacts/paper/figures/figure3_olmo_slop_lexicon_views.svg",
        "artifacts/paper/figures/figure4_cross_ladder_af_scatter.svg",
        "artifacts/paper/figures/figure5_phase4_tier3_raw_af.svg",
        "artifacts/paper/figures/submission_exports/figure1_pybiber_register_selected.pdf",
        "artifacts/paper/figures/submission_exports/figure1_pybiber_register_selected.png",
        "artifacts/paper/figures/submission_exports/figure2_eqbench_stage_scores.pdf",
        "artifacts/paper/figures/submission_exports/figure2_eqbench_stage_scores.png",
        "artifacts/paper/figures/submission_exports/figure3_olmo_slop_lexicon_views.pdf",
        "artifacts/paper/figures/submission_exports/figure3_olmo_slop_lexicon_views.png",
        "artifacts/paper/figures/submission_exports/figure4_cross_ladder_af_scatter.pdf",
        "artifacts/paper/figures/submission_exports/figure4_cross_ladder_af_scatter.png",
        "artifacts/paper/figures/submission_exports/figure5_phase4_tier3_raw_af.pdf",
        "artifacts/paper/figures/submission_exports/figure5_phase4_tier3_raw_af.png",
        "artifacts/stage1/census/phase1_pybiber_register_intervals.csv",
        "artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv",
        "artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.md",
        "artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv",
        (
            "artifacts/phase3/analysis/"
            "smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_"
            "baselines_data_rates_slop_neutral_rule3_production.csv"
        ),
        (
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_detector_tier3_matcher_specs.json"
        ),
        (
            "artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/"
            "olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv"
        ),
        "artifacts/stage1/census/feature_rates_by_corpus.parquet",
        "artifacts/stage1/census/preference_pair_deltas.parquet",
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
        "artifacts/paper/figures/paper_figure_readiness_audit.csv",
        "artifacts/paper/figures/paper_figure_visual_review.csv",
        "artifacts/paper/tables/paper_table_readiness_audit.csv",
        "artifacts/paper/tables/paper_table_typography_review.csv",
        "artifacts/paper/references/paper_citation_audit.csv",
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
        "artifacts/paper/submission/paper_release_inventory.csv",
        "artifacts/paper/submission/paper_venue_decision_checklist.csv",
        "artifacts/paper/submission/paper_reproducibility_manifest.csv",
    ]:
        _write(root / relative, "<svg></svg>\n")
    _write(
        root
        / "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_annotation_readiness.csv",
        _phase4_readiness_csv(),
    )
    _write(
        root / "docs/experiments/phase4_human_annotation_readiness.md",
        _phase4_readiness_md(),
    )
    _write(
        root / "docs/experiments/phase4_human_annotation_codebook.md",
        _phase4_codebook_md(),
    )
    _write(
        root / "docs/experiments/phase4_human_annotation_handoff.md",
        _phase4_handoff_md(),
    )
    _write(
        root / "docs/experiments/phase4_human_labeling_execution_checklist.md",
        _phase4_labeling_checklist_md(),
    )
    _write(
        root / "docs/experiments/phase4_human_perceptibility_protocol.md",
        _phase4_protocol_md(),
    )
    _write(
        root / "docs/experiments/phase4_human_perceptibility_summary.md",
        _phase4_human_summary_md(),
    )
    _write(
        root
        / "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_perceptibility_summary.csv",
        _phase4_human_summary_csv(),
    )
    _write(
        root / "docs/experiments/phase3_production_runbook.md",
        _phase3_production_runbook_md(),
    )
    _write(
        root / "docs/experiments/phase4_production_runbook.md",
        _phase4_production_runbook_md(),
    )
    phase4_base = root / "artifacts/phase4/modernbert_detector_combined_v2_clean"
    _write(
        phase4_base / "phase4_human_perceptibility_blind_pilot_20_each.csv",
        _phase4_blind_csv(200),
    )
    _write(
        phase4_base / "phase4_human_perceptibility_blind_pilot_20_each_map.csv",
        _phase4_blind_map_csv(200),
    )
    _write(
        phase4_base / "phase4_human_perceptibility_blind_full.csv",
        _phase4_blind_csv(1000),
    )
    _write(
        phase4_base / "phase4_human_perceptibility_blind_full_map.csv",
        _phase4_blind_map_csv(1000),
    )
    _write(
        phase4_base / "phase4_human_perceptibility_annotation_package.jsonl",
        '{"annotation_id": "phase4_feature::000", "feature": "phase4_feature"}\n',
    )
    handoff_base = phase4_base / "human_annotation_handoff"
    _write(handoff_base / "README.md", _phase4_handoff_readme())
    _write(
        handoff_base / "annotator/README.md",
        "# Phase 4 Annotator Quickstart\n\n"
        "Edit only `human_perceived_slop`, `shaib_taxonomy_label`, and `notes`.\n"
        "Return the filled assigned CSV only.\n",
    )
    _write(
        handoff_base
        / "annotator/annotator_a/phase4_human_perceptibility_blind_full_annotator_a.csv",
        _phase4_blind_csv(1000),
    )
    _write(
        handoff_base
        / "annotator/annotator_b/phase4_human_perceptibility_blind_full_annotator_b.csv",
        _phase4_blind_csv(1000),
    )
    handoff_copies = [
        (
            "docs/experiments/phase4_human_annotation_codebook.md",
            "annotator/phase4_human_annotation_codebook.md"
        ),
        (
            str(phase4_base.relative_to(root) / "phase4_human_perceptibility_blind_pilot_20_each.csv"),
            "annotator/phase4_human_perceptibility_blind_pilot_20_each.csv"
        ),
        (
            str(phase4_base.relative_to(root) / "phase4_human_perceptibility_blind_full.csv"),
            "annotator/phase4_human_perceptibility_blind_full.csv"
        ),
        (
            str(phase4_base.relative_to(root) / "phase4_human_perceptibility_blind_pilot_20_each.csv"),
            "annotator/annotator_a/phase4_human_perceptibility_blind_pilot_20_each_annotator_a.csv"
        ),
        (
            str(phase4_base.relative_to(root) / "phase4_human_perceptibility_blind_pilot_20_each.csv"),
            "annotator/annotator_b/phase4_human_perceptibility_blind_pilot_20_each_annotator_b.csv"
        ),
        (
            str(phase4_base.relative_to(root) / "phase4_human_perceptibility_blind_full.csv"),
            "annotator/annotator_a/phase4_human_perceptibility_blind_full_annotator_a.csv"
        ),
        (
            str(phase4_base.relative_to(root) / "phase4_human_perceptibility_blind_full.csv"),
            "annotator/annotator_b/phase4_human_perceptibility_blind_full_annotator_b.csv"
        ),
        (
            "docs/experiments/phase4_human_perceptibility_protocol.md",
            "coordinator/phase4_human_perceptibility_protocol.md"
        ),
        (
            "docs/experiments/phase4_human_annotation_readiness.md",
            "coordinator/phase4_human_annotation_readiness.md"
        ),
        (
            "docs/experiments/phase4_human_labeling_execution_checklist.md",
            "coordinator/phase4_human_labeling_execution_checklist.md"
        ),
        (
            str(phase4_base.relative_to(root) / "phase4_human_perceptibility_blind_pilot_20_each_map.csv"),
            "coordinator/private_maps/phase4_human_perceptibility_blind_pilot_20_each_map.csv"
        ),
        (
            str(phase4_base.relative_to(root) / "phase4_human_perceptibility_blind_full_map.csv"),
            "coordinator/private_maps/phase4_human_perceptibility_blind_full_map.csv"
        ),
        (
            str(phase4_base.relative_to(root) / "phase4_human_perceptibility_annotation_package.jsonl"),
            "coordinator/private_maps/phase4_human_perceptibility_annotation_package.jsonl"
        ),
    ]
    for source_relative, bundle_relative in handoff_copies:
        _write(handoff_base / bundle_relative, (root / source_relative).read_bytes())
    _write(
        handoff_base / "MANIFEST.csv",
        _phase4_handoff_manifest_csv(root),
    )
    _write(root / "artifacts/paper/figures/paper_figure_readiness_audit.csv", _figure_readiness_csv())
    _write(root / "docs/experiments/paper_figure_readiness_audit.md", _figure_readiness_md())
    _write(root / "artifacts/paper/figures/paper_figure_visual_review.csv", _figure_visual_review_csv())
    _write(root / "docs/experiments/paper_figure_visual_review.md", _figure_visual_review_md())
    for index in range(1, 6):
        _write(root / f"artifacts/paper/figures/visual_review_png/figure{index}.png", "png\n")
    _write(root / "artifacts/paper/tables/paper_table_readiness_audit.csv", _table_readiness_csv())
    _write(root / "docs/experiments/paper_table_readiness_audit.md", _table_readiness_md())
    _write(root / "artifacts/paper/tables/paper_table_typography_review.csv", _table_typography_csv())
    _write(root / "docs/experiments/paper_table_typography_review.md", _table_typography_md())
    _write(root / "artifacts/paper/claims/paper_claim_language_audit.csv", _claim_language_csv())
    _write(root / "docs/experiments/paper_claim_language_audit.md", _claim_language_md())
    _write(root / "artifacts/paper/claims/paper_claim_evidence_audit.csv", _claim_evidence_csv())
    _write(root / "docs/experiments/paper_claim_evidence_audit.md", _claim_evidence_md())
    _write(root / "artifacts/paper/references/paper_citation_audit.csv", _citation_audit_csv())
    _write(root / "docs/experiments/paper_citation_audit.md", _citation_audit_md())
    _write(
        root / "artifacts/paper/manuscript/paper_abstract_conclusion_audit.csv",
        _abstract_conclusion_csv(),
    )
    _write(
        root / "docs/experiments/paper_abstract_conclusion_audit.md",
        _abstract_conclusion_md(),
    )
    _write(
        root / "artifacts/paper/manuscript/paper_manuscript_structure_audit.csv",
        _manuscript_structure_csv(),
    )
    _write(
        root / "docs/experiments/paper_manuscript_structure_audit.md",
        _manuscript_structure_md(),
    )
    _write(
        root / "artifacts/paper/manuscript/paper_limitations_audit.csv",
        _limitations_csv(),
    )
    _write(root / "docs/experiments/paper_limitations_audit.md", _limitations_md())
    _write(
        root / "artifacts/paper/manuscript/paper_terminology_audit.csv",
        _terminology_csv(),
    )
    _write(root / "docs/experiments/paper_terminology_audit.md", _terminology_md())
    _write(
        root / "artifacts/paper/manuscript/paper_numeric_claims_audit.csv",
        _numeric_claims_csv(),
    )
    _write(root / "docs/experiments/paper_numeric_claims_audit.md", _numeric_claims_md())
    _write(
        root / "artifacts/paper/manuscript/paper_review_hygiene_audit.csv",
        _review_hygiene_csv(),
    )
    _write(root / "docs/experiments/paper_review_hygiene_audit.md", _review_hygiene_md())
    _write(root / "docs/experiments/paper_release_inventory.md", _release_inventory_md())
    _write(
        root / "artifacts/paper/submission/paper_source_provenance_audit.csv",
        _source_provenance_csv(),
    )
    _write(
        root / "docs/experiments/paper_source_provenance_audit.md",
        _source_provenance_md(),
    )
    _write(
        root / "artifacts/paper/submission/paper_submission_exit_audit.csv",
        _submission_exit_csv(),
    )
    _write(root / "docs/experiments/paper_submission_exit_audit.md", _submission_exit_md())
    _write(
        root / "artifacts/paper/submission/paper_venue_decision_checklist.csv",
        _venue_decision_csv(),
    )
    _write(root / "docs/experiments/paper_venue_decision_checklist.md", _venue_decision_md())
    _write(root / "docs/experiments/paper_venue_adaptation_runbook.md", _venue_runbook_md())
    _write(root / "docs/experiments/paper_venue_sizing_inventory.md", _venue_sizing_inventory_md())
    _write(root / "docs/experiments/paper_reproduction_runbook.md", _reproduction_runbook_md())
    _write(root / "docs/experiments/paper_package_index.md", _paper_package_index_md())
    _write(root / "docs/experiments/paper_reviewer_faq.md", _reviewer_faq_md())
    _write(
        root / "docs/experiments/paper_manuscript_draft.md",
        "\n".join(
            [
                "# Manuscript",
                "Figure 1, Figure 2, Figure 3, Figure 4, and Figure 5 show results.",
                "Table 1, Table 2, and Table 4 summarize scope and validation.",
                "The related work cites [@known_key].",
                _caption_block(),
            ]
        )
        + "\n",
    )
    _write(
        root / "docs/experiments/paper_figure_table_manifest.md",
        "\n".join(
            [
                "# Manifest",
                "Figure 1 Figure 2 Figure 3 Figure 4 Figure 5",
                "Table 1 Table 2 Table 3 Table 4 Table 5",
                "`artifacts/stage1/census/phase1_pybiber_register_intervals.csv`",
                "`artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv`",
                "document-bootstrap intervals over retained rows",
                "bootstrap intervals over valid generated outputs",
                "`docs/experiments/paper_camera_ready_tables.md`",
            ]
        )
        + "\n",
    )
    _write(
        root / "docs/experiments/paper_caption_appendix_draft.md",
        "`docs/experiments/paper_camera_ready_tables.md`\n"
        "`docs/experiments/paper_latex_table_drafts.tex`\n",
    )
    _write(root / "docs/experiments/paper_latex_table_drafts.tex", _latex_table_pack())
    _write(
        root / "docs/experiments/paper_camera_ready_tables.md",
        "# Tables\n\n"
        "## Table 1. Data And Measurement Scope\n\n"
        "| Measurement layer | Scope | Size | Paper role |\n"
        "|---|---|---:|---|\n"
        "| Layer | Scope | 1 | Role |\n\n"
        "## Table 2. Selected Full-Pybiber Register Means\n\n"
        "| Feature | Dolma | SFT | DPO chosen | DPO rejected | Interpretation |\n"
        "|---|---:|---:|---:|---:|---|\n"
        "| Past tense | 1 | 2 | 3 | 4 | Example |\n\n"
        "## Table 4. Tier-1 Precision-Validation Status\n\n"
        "| Feature | Role | Status | Labeled | Exact | False positive | Ambiguous | Precision | Paper treatment |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---|\n"
        "| `contrastive_negation` | Core | Pass | 1 | 1 | 0 | 0 | 1.000 | Use |\n\n"
        "No repository paths here.\n\n"
        "## Appendix Table A1. Claim Strength Bands\n\n"
        "| Band | Claim IDs | Allowed use |\n"
        "|---|---|---|\n"
        "| Main | C1 | Use |\n\n"
        "## Appendix Table A2. Selected Pybiber Bootstrap Intervals\n\n"
        "| Feature | Dolma |\n"
        "|---|---:|\n"
        "| Past tense | 1.000 [0.900, 1.100] |\n\n"
        "## Appendix Table A3. Paper-Safe Negative Claims\n\n"
        "| Avoided claim | Replacement |\n"
        "|---|---|\n"
        "| Overclaim | Bounded claim |\n",
    )
    _write(
        root / "docs/experiments/paper_references.bib",
        "@article{known_key,\n  title={Known}\n}\n",
    )
    _write(
        root / "docs/experiments/paper_reference_sources.md",
        "| Key | Source | Verified from | Notes for use |\n"
        "|---|---|---|---|\n"
        "| `known_key` | Known | source | notes |\n",
    )
    _write_draft_bundle(root)
    _write(
        root / "artifacts/paper/submission/paper_release_inventory.csv",
        _release_inventory_csv(root),
    )
    _write(
        root / "artifacts/paper/submission/paper_reproducibility_manifest.csv",
        _reproducibility_manifest_csv(root),
    )
    _write(
        root / "docs/experiments/paper_reproducibility_manifest.md",
        _reproducibility_manifest_md(),
    )


def test_check_paper_package_passes_minimal_package(tmp_path):
    _write_minimal_package(tmp_path)

    report = run_check_paper_package(tmp_path)

    assert report.errors == ()


def test_check_paper_package_flags_main_body_paths(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_manuscript_draft.md",
        "Figure 1 Figure 2 Figure 3 Figure 4 Figure 5 Table 1 Table 2 Table 4\n"
        f"{_caption_block()}\n"
        "See `docs/experiments/paper_tables.md` and [@known_key].\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any("forbidden main-body marker" in error for error in report.errors)


def test_check_paper_package_flags_review_hygiene_audit_findings(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "artifacts/paper/manuscript/paper_review_hygiene_audit.csv",
        _review_hygiene_csv().replace(
            "manuscript_main_body_hygiene,ready,docs/experiments/paper_manuscript_draft.md,,,",
            "manuscript_main_body_hygiene,review,docs/experiments/paper_manuscript_draft.md,docs/,,",
        ),
    )
    _write(
        tmp_path / "docs/experiments/paper_review_hygiene_audit.md",
        "# Paper Review Hygiene Audit\n\nReadiness status: `review`.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_review_hygiene_audit.csv manuscript_main_body_hygiene status='review'"
        in error
        for error in report.errors
    )
    assert any(
        "paper_review_hygiene_audit.csv manuscript_main_body_hygiene has forbidden_hits"
        in error
        for error in report.errors
    )
    assert any(
        "paper_review_hygiene_audit.md missing review-hygiene phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_release_inventory_findings(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "artifacts/paper/submission/paper_release_inventory.csv",
        _release_inventory_csv().replace(
            "feature_definitions,src/slop_sftdiv/features/tier1_matchers.py,ready,12,Release role,Caveat",
            "feature_definitions,src/slop_sftdiv/features/tier1_matchers.py,missing,,Release role,Caveat",
        ),
    )
    _write(
        tmp_path / "docs/experiments/paper_release_inventory.md",
        "# Paper Release Inventory\n\nReadiness status: `review`.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_release_inventory.csv src/slop_sftdiv/features/tier1_matchers.py "
        "status='missing'" in error
        for error in report.errors
    )
    assert any(
        "paper_release_inventory.csv src/slop_sftdiv/features/tier1_matchers.py "
        "has invalid size_bytes" in error
        for error in report.errors
    )
    assert any(
        "paper_release_inventory.md missing release-inventory phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_release_inventory_size_drift(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "src/slop_sftdiv/features/tier1_matchers.py",
        "# changed after release inventory\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_release_inventory.csv src/slop_sftdiv/features/tier1_matchers.py "
        "size_bytes=" in error
        for error in report.errors
    )


def test_check_paper_package_flags_source_provenance_findings(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "artifacts/paper/submission/paper_source_provenance_audit.csv",
        _source_provenance_csv().replace(
            "dolma_sample_boundary,ready,scope,3/3,,,Purpose",
            "dolma_sample_boundary,review,scope,2/3,not full Dolma,,Purpose",
        ),
    )
    _write(
        tmp_path / "docs/experiments/paper_source_provenance_audit.md",
        "# Paper Source Provenance Audit\n\nReadiness status: `review`.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_source_provenance_audit.csv dolma_sample_boundary status='review'"
        in error
        for error in report.errors
    )
    assert any(
        "paper_source_provenance_audit.csv dolma_sample_boundary has missing_required"
        in error
        for error in report.errors
    )
    assert any(
        "paper_source_provenance_audit.md missing source-provenance phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_phase4_readiness(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/phase4_human_annotation_readiness.md",
        "# Phase 4 Human Annotation Readiness\n\nReadiness status: `review`.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "phase4_human_annotation_readiness.md missing readiness phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_unblinded_phase4_sheet(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path
        / "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_perceptibility_blind_full.csv",
        "blind_id,matched_text,snippet,human_perceived_slop,shaib_taxonomy_label,notes,feature\n"
        "blind_0001,text,snippet,,,,phase4_feature_0\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "phase4_human_perceptibility_blind_full.csv expected 1000 full blind rows"
        in error
        for error in report.errors
    )
    assert any(
        "phase4_human_perceptibility_blind_full.csv leaks unblinded columns: feature"
        in error
        for error in report.errors
    )


def test_check_paper_package_flags_phase4_handoff_redaction_findings(tmp_path):
    _write_minimal_package(tmp_path)
    manifest_path = (
        tmp_path
        / "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "human_annotation_handoff/MANIFEST.csv"
    )
    manifest_path.write_text(
        _phase4_handoff_manifest_csv().replace(
            "annotator/phase4_human_perceptibility_blind_full.csv,ready,redacted",
            "annotator/phase4_human_perceptibility_blind_full.csv,ready,leaks_columns:feature",
        ),
        encoding="utf-8",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "human_annotation_handoff/MANIFEST.csv "
        "annotator/phase4_human_perceptibility_blind_full.csv "
        "redaction_status='leaks_columns:feature'" in error
        for error in report.errors
    )


def test_check_paper_package_flags_phase4_handoff_actual_annotator_leak(tmp_path):
    _write_minimal_package(tmp_path)
    bundle_path = "annotator/phase4_human_perceptibility_blind_full.csv"
    handoff_file = (
        tmp_path
        / "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "human_annotation_handoff"
        / bundle_path
    )
    handoff_file.write_text(
        "blind_id,matched_text,snippet,human_perceived_slop,shaib_taxonomy_label,notes,feature\n"
        "blind_0001,text,snippet,,,,phase4_feature_0\n",
        encoding="utf-8",
    )
    _rewrite_handoff_manifest_row(
        tmp_path,
        bundle_path=bundle_path,
        source_path="generated",
        redaction_status="redacted",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "human_annotation_handoff/MANIFEST.csv "
        "annotator/phase4_human_perceptibility_blind_full.csv "
        "leaks unblinded columns: feature" in error
        for error in report.errors
    )


def test_check_paper_package_flags_phase4_handoff_hash_drift(tmp_path):
    _write_minimal_package(tmp_path)
    handoff_file = (
        tmp_path
        / "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "human_annotation_handoff/annotator/phase4_human_annotation_codebook.md"
    )
    handoff_file.write_text("changed after manifest\n", encoding="utf-8")

    report = run_check_paper_package(tmp_path)

    assert any(
        "human_annotation_handoff/MANIFEST.csv "
        "annotator/phase4_human_annotation_codebook.md sha256=" in error
        for error in report.errors
    )


def test_check_paper_package_flags_phase4_handoff_missing_bundle_file(tmp_path):
    _write_minimal_package(tmp_path)
    handoff_file = (
        tmp_path
        / "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "human_annotation_handoff/coordinator/private_maps/"
        "phase4_human_perceptibility_blind_full_map.csv"
    )
    handoff_file.unlink()

    report = run_check_paper_package(tmp_path)

    assert any(
        "human_annotation_handoff/MANIFEST.csv "
        "coordinator/private_maps/phase4_human_perceptibility_blind_full_map.csv "
        "points to missing bundle file" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_phase4_codebook(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/phase4_human_annotation_codebook.md",
        "# Phase 4 Human Annotation Codebook\n\nPlaceholder.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "phase4_human_annotation_codebook.md missing annotation-codebook phrase"
        in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_phase4_labeling_checklist(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/phase4_human_labeling_execution_checklist.md",
        "# Phase 4 Human Labeling Execution Checklist\n\nPlaceholder.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "phase4_human_labeling_execution_checklist.md missing labeling-checklist phrase"
        in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_phase4_protocol_full_summary_input(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/phase4_human_perceptibility_protocol.md",
        "# Phase 4 Human-Perceptibility Annotation Protocol\n\n"
        "After full-package annotation, run:\n\n"
        "```bash\n"
        "uv run slop-summarize-phase4-human-labels \\\n"
        "  --input artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_perceptibility_annotation_package.jsonl \\\n"
        "  --output artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_perceptibility_summary.csv\n"
        "```\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "phase4_human_perceptibility_protocol.md contains stale human-protocol phrase"
        in error
        for error in report.errors
    )


def test_check_paper_package_flags_labeled_phase4_human_summary(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path
        / "artifacts/phase4/modernbert_detector_combined_v2_clean/"
        "phase4_human_perceptibility_summary.csv",
        _phase4_human_summary_csv().replace(
            "phase4_ig_followup_offer,Candidate,cluster,100,0,100,0,0,0,0,0,,,,unlabeled",
            "phase4_ig_followup_offer,Candidate,cluster,100,64,36,40,24,0,0,64,0.64,0.40,verbosity,detector_and_human_perceived",
        ),
    )
    _write(
        tmp_path / "docs/experiments/phase4_human_perceptibility_summary.md",
        "# Phase 4 Human-Perceptibility Summary\n\n"
        "| phase4_ig_followup_offer | 64/100 | 40 | 24 | 0 | 0 | verbosity | detector_and_human_perceived |\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "phase4_human_perceptibility_summary.csv phase4_ig_followup_offer labeled="
        in error
        for error in report.errors
    )
    assert any(
        "phase4_human_perceptibility_summary.md missing human-summary phrase"
        in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_phase4_production_runbook(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/phase4_production_runbook.md",
        "# Phase 4 Production Runbook\n\n"
        "The detector, annotation package, and 128-document exact teacher-forced "
        "rerun already exist under the production directory.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "phase4_production_runbook.md contains stale production-runbook phrase"
        in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_phase3_generated_pybiber_runbook(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/phase3_production_runbook.md",
        "# Phase 3 Production Runbook\n\n"
        "## Rebuild Biber-Lite And Style Signature\n\n"
        "After the new `t=1.0` generated caches complete, extract full pybiber "
        "register features for each generated cache.\n"
        "--register-comparison artifacts/phase2/analysis/"
        "smollm3_pybiber_register_generation_vs_corpus.csv\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "phase3_production_runbook.md missing generated-output pybiber boundary"
        in error
        for error in report.errors
    )
    assert any(
        "phase3_production_runbook.md contains stale generated-output pybiber instruction"
        in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_figure_readiness(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_figure_readiness_audit.md",
        "# Paper Figure Readiness Audit\n\nReadiness status: `review`.\n",
    )
    csv_text = _figure_readiness_csv().replace("ready_for_visual_review", "review", 1)
    _write(tmp_path / "artifacts/paper/figures/paper_figure_readiness_audit.csv", csv_text)

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_figure_readiness_audit.csv Figure 1 status='review'" in error
        for error in report.errors
    )
    assert any(
        "paper_figure_readiness_audit.md missing figure-readiness phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_figure_visual_review(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_figure_visual_review.md",
        "# Paper Figure Rendered-Layout Review\n\nReview status: `review`.\n",
    )
    csv_text = _figure_visual_review_csv().replace(
        "Figure 4,artifacts/paper/figures/figure4.svg,"
        "artifacts/paper/figures/visual_review_png/figure4.png,visual_review_passed",
        "Figure 4,artifacts/paper/figures/figure4.svg,"
        "artifacts/paper/figures/visual_review_png/figure4.png,review",
    )
    _write(tmp_path / "artifacts/paper/figures/paper_figure_visual_review.csv", csv_text)

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_figure_visual_review.csv Figure 4 review_status='review'" in error
        for error in report.errors
    )
    assert any(
        "paper_figure_visual_review.md missing figure-visual-review phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_table_readiness(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_table_readiness_audit.md",
        "# Paper Table Readiness Audit\n\nReadiness status: `review`.\n",
    )
    csv_text = _table_readiness_csv().replace("ready_for_venue_styling", "review", 1)
    _write(tmp_path / "artifacts/paper/tables/paper_table_readiness_audit.csv", csv_text)

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_table_readiness_audit.csv Table 1 status='review'" in error
        for error in report.errors
    )
    assert any(
        "paper_table_readiness_audit.md missing table-readiness phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_table_typography_review(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_table_typography_review.md",
        "# Paper Table Typography Review\n\nReview status: `review`.\n",
    )
    csv_text = _table_typography_csv().replace("typography_review_passed", "review", 1)
    _write(tmp_path / "artifacts/paper/tables/paper_table_typography_review.csv", csv_text)

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_table_typography_review.csv Table 1 review_status='review'" in error
        for error in report.errors
    )
    assert any(
        "paper_table_typography_review.md missing table-typography phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_claim_language_audit(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_claim_language_audit.md",
        "# Paper Claim Language Audit\n\nReadiness status: `review`.\n",
    )
    csv_text = _claim_language_csv().replace("C8,claim,ready", "C8,claim,review")
    _write(tmp_path / "artifacts/paper/claims/paper_claim_language_audit.csv", csv_text)

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_claim_language_audit.csv C8 status='review'" in error
        for error in report.errors
    )
    assert any(
        "paper_claim_language_audit.md missing claim-language phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_abstract_conclusion_audit(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_abstract_conclusion_audit.md",
        "# Paper Abstract And Conclusion Audit\n\nReadiness status: `review`.\n",
    )
    csv_text = _abstract_conclusion_csv()
    csv_text = csv_text.replace(
        "conclusion_alignment,## 7. Conclusion,ready",
        "conclusion_alignment,## 7. Conclusion,review",
    )
    csv_text = csv_text.replace(
        "conclusion_alignment,## 7. Conclusion,review,6,6,3,,",
        "conclusion_alignment,## 7. Conclusion,review,6,6,3,alignment creates slop.,",
    )
    _write(
        tmp_path / "artifacts/paper/manuscript/paper_abstract_conclusion_audit.csv",
        csv_text,
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_abstract_conclusion_audit.csv conclusion_alignment status='review'"
        in error
        for error in report.errors
    )
    assert any(
        "paper_abstract_conclusion_audit.md missing abstract/conclusion phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_manuscript_structure_audit(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_manuscript_structure_audit.md",
        "# Paper Manuscript Structure Audit\n\nReadiness status: `review`.\n",
    )
    csv_text = _manuscript_structure_csv().replace(
        "result_subsection_order,ready,5/5",
        "result_subsection_order,review,4/5",
    )
    _write(
        tmp_path / "artifacts/paper/manuscript/paper_manuscript_structure_audit.csv",
        csv_text,
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_manuscript_structure_audit.csv result_subsection_order status='review'"
        in error
        for error in report.errors
    )
    assert any(
        "paper_manuscript_structure_audit.md missing manuscript-structure phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_limitations_audit(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_limitations_audit.md",
        "# Paper Limitations Audit\n\nReadiness status: `review`.\n",
    )
    csv_text = _limitations_csv().replace(
        "pybiber_generation_boundary,ready,3,3,,",
        "pybiber_generation_boundary,review,3,2,not over the generation caches,",
    )
    _write(
        tmp_path / "artifacts/paper/manuscript/paper_limitations_audit.csv",
        csv_text,
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_limitations_audit.csv pybiber_generation_boundary status='review'"
        in error
        for error in report.errors
    )
    assert any(
        "paper_limitations_audit.md missing limitations phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_submission_exit_audit(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_submission_exit_audit.md",
        "# Paper Submission Exit Audit\n\nSubmission status: `ready`.\n",
    )
    csv_text = _submission_exit_csv().replace(
        "phase4_human_perceptibility,human_validation_pending",
        "phase4_human_perceptibility,ready",
    )
    _write(tmp_path / "artifacts/paper/submission/paper_submission_exit_audit.csv", csv_text)

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_submission_exit_audit.csv phase4_human_perceptibility status='ready'"
        in error
        for error in report.errors
    )
    assert any(
        "paper_submission_exit_audit.md missing submission-exit phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_venue_decision_checklist(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_venue_decision_checklist.md",
        "# Paper Venue Decision Checklist\n\nVenue decision status: `ready`.\n",
    )
    csv_text = _venue_decision_csv().replace(
        "figure_dimensions,decision_pending",
        "figure_dimensions,ready",
    )
    _write(
        tmp_path / "artifacts/paper/submission/paper_venue_decision_checklist.csv",
        csv_text,
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_venue_decision_checklist.csv figure_dimensions status='ready'" in error
        for error in report.errors
    )
    assert any(
        "paper_venue_decision_checklist.md missing venue-decision phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_submission_draft_bundle(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path
        / "artifacts/paper/submission/draft_bundle/manuscript/paper_manuscript_draft.md",
        "changed\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "draft_bundle/MANIFEST.csv manuscript/paper_manuscript_draft.md sha256="
        in error
        for error in report.errors
    )


def test_check_paper_package_flags_submission_draft_bundle_source_drift(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_manuscript_draft.md",
        "changed source after bundle materialization\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "draft_bundle/MANIFEST.csv manuscript/paper_manuscript_draft.md sha256="
        in error
        and "expected source sha256" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_venue_adaptation_runbook(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_venue_adaptation_runbook.md",
        "# Paper Venue Adaptation Runbook\n\nPlaceholder.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_venue_adaptation_runbook.md missing venue-runbook phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_venue_sizing_inventory(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_venue_sizing_inventory.md",
        "# Paper Venue Sizing Inventory\n\nPlaceholder.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_venue_sizing_inventory.md missing venue-sizing phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_venue_sizing_dimensions(tmp_path):
    _write_minimal_package(tmp_path)
    sizing = (tmp_path / "docs/experiments/paper_venue_sizing_inventory.md").read_text(
        encoding="utf-8"
    )
    _write(
        tmp_path / "docs/experiments/paper_venue_sizing_inventory.md",
        sizing.replace("500.000 x 320.000 pt", "501.000 x 320.000 pt", 1),
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_venue_sizing_inventory.md missing venue-sizing figure detail: "
        "Figure 1 | `artifacts/paper/figures/figure1.svg` | 500.000 x 320.000 pt"
        in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_reproduction_runbook(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_reproduction_runbook.md",
        "# Paper Reproduction Runbook\n\nPlaceholder.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_reproduction_runbook.md missing reproduction-runbook phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_reviewer_faq(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_reviewer_faq.md",
        "# Paper Reviewer FAQ\n\nPlaceholder.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_reviewer_faq.md missing reviewer-faq phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_readiness_audit(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_readiness_audit.md",
        "# Readiness\n\n"
        "## Claim Readiness Summary\n\n"
        "| Claim band | Claim IDs | Readiness |\n"
        "|---|---|---|\n"
        "| Main | C1, C2, C7, C8, C9 | Ready |\n"
        "| Caveated | C3, C4, C5, C6, C10, C12, C13 | Caveat |\n"
        "| Status | C11, C14 | Status |\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_readiness_audit.md missing readiness-audit phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_package_index(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_package_index.md",
        "# Paper Package Index\n\nPlaceholder.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_package_index.md missing package-index phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_numeric_claims_audit(tmp_path):
    _write_minimal_package(tmp_path)
    csv_text = _numeric_claims_csv().replace(
        "pybiber_narrative_drop,ready,source.csv,expected,",
        "pybiber_narrative_drop,review,source.csv,expected,missing claim",
    )
    _write(
        tmp_path / "artifacts/paper/manuscript/paper_numeric_claims_audit.csv",
        csv_text,
    )
    _write(
        tmp_path / "docs/experiments/paper_numeric_claims_audit.md",
        "# Paper Numeric Claims Audit\n\nReadiness status: `review`.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_numeric_claims_audit.csv pybiber_narrative_drop status='review'"
        in error
        for error in report.errors
    )
    assert any(
        "paper_numeric_claims_audit.csv pybiber_narrative_drop has non-empty missing_text"
        in error
        for error in report.errors
    )
    assert any(
        "paper_numeric_claims_audit.md missing numeric-claims phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_terminology_audit(tmp_path):
    _write_minimal_package(tmp_path)
    csv_text = _terminology_csv().replace(
        "mechanism_separation,ready,4,4,,purpose",
        "mechanism_separation,review,4,3,missing phrase,purpose",
    )
    _write(
        tmp_path / "artifacts/paper/manuscript/paper_terminology_audit.csv",
        csv_text,
    )
    _write(
        tmp_path / "docs/experiments/paper_terminology_audit.md",
        "# Paper Terminology Audit\n\nReadiness status: `review`.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_terminology_audit.csv mechanism_separation status='review'"
        in error
        for error in report.errors
    )
    assert any(
        "paper_terminology_audit.csv mechanism_separation has missing_required"
        in error
        for error in report.errors
    )
    assert any(
        "paper_terminology_audit.md missing terminology phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_claim_evidence_audit(tmp_path):
    _write_minimal_package(tmp_path)
    csv_text = _claim_evidence_csv().replace(
        "C14,ready,docs/experiments/paper_claim_matrix.md,,,,",
        "C14,review,,docs/missing.md,,evidence_cell,",
    )
    _write(
        tmp_path / "artifacts/paper/claims/paper_claim_evidence_audit.csv",
        csv_text,
    )
    _write(
        tmp_path / "docs/experiments/paper_claim_evidence_audit.md",
        "# Paper Claim Evidence Audit\n\nReadiness status: `review`.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_claim_evidence_audit.csv C14 status='review'" in error
        for error in report.errors
    )
    assert any(
        "paper_claim_evidence_audit.csv C14 has missing_paths" in error
        for error in report.errors
    )
    assert any(
        "paper_claim_evidence_audit.md missing claim-evidence phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_citation_audit(tmp_path):
    _write_minimal_package(tmp_path)
    csv_text = _citation_audit_csv().replace(
        "cited_keys_have_bib_entries,ready,observed,,notes",
        "cited_keys_have_bib_entries,review,observed,missing_key,notes",
    )
    _write(
        tmp_path / "artifacts/paper/references/paper_citation_audit.csv",
        csv_text,
    )
    _write(
        tmp_path / "docs/experiments/paper_citation_audit.md",
        "# Paper Citation Audit\n\nReadiness status: `review`.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_citation_audit.csv cited_keys_have_bib_entries status='review'"
        in error
        for error in report.errors
    )
    assert any(
        "paper_citation_audit.csv cited_keys_have_bib_entries has missing_or_invalid"
        in error
        for error in report.errors
    )
    assert any(
        "paper_citation_audit.md missing citation-audit phrase" in error
        for error in report.errors
    )


def test_check_paper_package_flags_stale_reproducibility_manifest(tmp_path):
    _write_minimal_package(tmp_path)
    rows = _reproducibility_manifest_csv(tmp_path).splitlines()
    for index, row in enumerate(rows):
        if "docs/experiments/paper_manuscript_draft.md" in row:
            rows[index] = (
                "test,docs/experiments/paper_manuscript_draft.md,missing,0,"
                + "b" * 64
            )
            break
    csv_text = "\n".join(rows) + "\n"
    _write(
        tmp_path / "artifacts/paper/submission/paper_reproducibility_manifest.csv",
        csv_text,
    )
    _write(
        tmp_path / "docs/experiments/paper_reproducibility_manifest.md",
        "# Paper Reproducibility Manifest\n\nReadiness status: `review`.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_reproducibility_manifest.csv docs/experiments/paper_manuscript_draft.md "
        "status='missing'" in error
        for error in report.errors
    )
    assert any(
        "paper_reproducibility_manifest.csv docs/experiments/paper_manuscript_draft.md "
        "has invalid size_bytes" in error
        for error in report.errors
    )
    assert any(
        "paper_reproducibility_manifest.md missing reproducibility-manifest phrase"
        in error
        for error in report.errors
    )


def test_check_paper_package_flags_missing_reproducibility_release_code(tmp_path):
    _write_minimal_package(tmp_path)
    rows = [
        row
        for row in _reproducibility_manifest_csv(tmp_path).splitlines()
        if ",src/" not in row
    ]
    _write(
        tmp_path / "artifacts/paper/submission/paper_reproducibility_manifest.csv",
        "\n".join(rows) + "\n",
    )
    _write(
        tmp_path / "docs/experiments/paper_reproducibility_manifest.md",
        "# Paper Reproducibility Manifest\n\n"
        "Machine-readable manifest: `artifacts/paper/submission/paper_reproducibility_manifest.csv`\n\n"
        "Readiness status: `ready`.\n"
        "This manifest records SHA-256 checksums and is not a venue submission certificate.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_reproducibility_manifest.csv missing release_code category" in error
        for error in report.errors
    )
    assert any(
        "paper_reproducibility_manifest.csv missing reproducibility paths: "
        "src/slop_sftdiv/cli/apply_phase4_label_adjudication.py" in error
        for error in report.errors
    )
    assert any(
        "paper_reproducibility_manifest.md missing reproducibility-manifest phrase: "
        "| release_code |" in error
        for error in report.errors
    )


def test_check_paper_package_flags_mislabeled_reproducibility_release_code(tmp_path):
    _write_minimal_package(tmp_path)
    rows = _reproducibility_manifest_csv(tmp_path).replace(
        "release_code,src/slop_sftdiv/features/eqbench_slop.py",
        "test,src/slop_sftdiv/features/eqbench_slop.py",
    )
    _write(
        tmp_path / "artifacts/paper/submission/paper_reproducibility_manifest.csv",
        rows,
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_reproducibility_manifest.csv src/slop_sftdiv/features/eqbench_slop.py "
        "category='test', expected 'release_code'" in error
        for error in report.errors
    )


def test_check_paper_package_flags_reproducibility_manifest_hash_mismatch(tmp_path):
    _write_minimal_package(tmp_path)
    _write(tmp_path / "docs/experiments/paper_manuscript_draft.md", "changed\n")

    report = run_check_paper_package(tmp_path)

    assert any(
        "paper_reproducibility_manifest.csv docs/experiments/paper_manuscript_draft.md "
        "sha256=" in error
        for error in report.errors
    )


def test_check_paper_package_flags_manuscript_draft_status_prose(tmp_path):
    _write_minimal_package(tmp_path)
    manuscript = (tmp_path / "docs/experiments/paper_manuscript_draft.md").read_text(
        encoding="utf-8"
    )
    _write(
        tmp_path / "docs/experiments/paper_manuscript_draft.md",
        "Status: integrated manuscript draft\n"
        "The project uses local retained artifacts and project vocabulary.\n"
        "This table is mainly a writing guardrail.\n"
        + manuscript,
    )

    report = run_check_paper_package(tmp_path)

    assert any("forbidden main-body marker" in error for error in report.errors)


def test_check_paper_package_flags_missing_bib_key(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_manuscript_draft.md",
        "Figure 1 Figure 2 Figure 3 Figure 4 Figure 5 Table 1 Table 2 Table 4 [@missing].\n"
        f"{_caption_block()}\n",
    )

    report = run_check_paper_package(tmp_path)

    assert (
        "docs/experiments/paper_manuscript_draft.md cites missing BibTeX key: missing"
        in report.errors
    )


def test_check_paper_package_flags_methods_citation_missing_bib_key(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_methods_draft.md",
        "Methods background cites [@missing_methods_key].\n",
    )

    report = run_check_paper_package(tmp_path)

    assert (
        "docs/experiments/paper_methods_draft.md cites missing BibTeX key: missing_methods_key"
        in report.errors
    )


def test_check_paper_package_flags_bib_key_missing_source_row(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_reference_sources.md",
        "| Key | Source | Verified from | Notes for use |\n"
        "|---|---|---|---|\n",
    )

    report = run_check_paper_package(tmp_path)

    assert "BibTeX key missing verified source row: known_key" in report.errors


def test_check_paper_package_flags_missing_backtick_path(tmp_path):
    _write_minimal_package(tmp_path)
    with (tmp_path / "docs/experiments/paper_scaffold.md").open("a", encoding="utf-8") as handle:
        handle.write("`docs/experiments/does_not_exist.md`\n")

    report = run_check_paper_package(tmp_path)

    assert any("references missing path" in error for error in report.errors)


def test_check_paper_package_flags_missing_claim_ids(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_claim_matrix.md",
        "| Claim ID | Paper claim | Evidence | Strength | Caveat | Allowed wording |\n"
        "|---|---|---|---|---|---|\n"
        "| C1 | Claim | Evidence | Strong | Caveat | Wording |\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any("claim matrix missing claim IDs" in error for error in report.errors)


def test_check_paper_package_flags_missing_manuscript_caption(tmp_path):
    _write_minimal_package(tmp_path)
    manuscript = (tmp_path / "docs/experiments/paper_manuscript_draft.md").read_text(
        encoding="utf-8"
    )
    manuscript = manuscript.replace("**Figure 3. Caption.**", "**Missing Figure Caption.**")
    _write(tmp_path / "docs/experiments/paper_manuscript_draft.md", manuscript)

    report = run_check_paper_package(tmp_path)

    assert "manuscript is missing caption for Figure 3" in report.errors


def test_check_paper_package_flags_retired_biber_lite_language(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_methods_draft.md",
        "Generated-output Biber-lite remains available.\n"
        "Generated-output Biber lite remains available.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any("uses retired feature language: Biber-lite" in error for error in report.errors)
    assert any("uses retired feature language: Biber lite" in error for error in report.errors)


def test_check_paper_package_flags_stale_paper_status_phrases(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_scaffold.md",
        "Manual precision validation for Tier-1 matchers remains incomplete.\n"
        "Regenerate Figure 2 intervals before final figure rendering.\n",
    )
    _write(
        tmp_path / "docs/experiments/paper_caption_appendix_draft.md",
        "Bounded corpus sizes used in the current paper package.\n"
        "This table is mainly a writing guardrail.\n",
    )
    _write(
        tmp_path / "docs/experiments/paper_methods_draft.md",
        "Evaluate transfer to held-out HAP-E and project-specific generation slices.\n"
        "The pooled feature should be described as derived unless a broader direct "
        "scan is deliberately run.\n",
    )
    _write(
        tmp_path / "docs/experiments/paper_intro_discussion_draft.md",
        'This is not just more purple words or a pile of slop words.\n'
        "Tier-1 precision validation is not a global certificate.\n",
    )
    _write(
        tmp_path / "docs/experiments/paper_results_draft.md",
        "For manuscript assembly, Figure 3 carries the headline feature view.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any(
        "contains stale phrase: Manual precision validation for Tier-1 matchers remains incomplete"
        in error
        for error in report.errors
    )
    assert any(
        "contains stale phrase: Regenerate Figure 2 intervals" in error
        for error in report.errors
    )
    assert any(
        "contains stale phrase: current paper package" in error
        for error in report.errors
    )
    assert any(
        "contains stale phrase: This table is mainly a writing guardrail" in error
        for error in report.errors
    )
    assert any(
        "contains stale phrase: project-specific generation slices" in error
        for error in report.errors
    )
    assert any("contains stale phrase: more purple words" in error for error in report.errors)
    assert any("contains stale phrase: pile of slop words" in error for error in report.errors)
    assert any("contains stale phrase: global certificate" in error for error in report.errors)
    assert any("contains stale phrase: should be described as derived unless" in error for error in report.errors)
    assert any("contains stale phrase: manuscript assembly" in error for error in report.errors)
    assert any("contains stale phrase: Figure 3 carries" in error for error in report.errors)


def test_check_paper_package_flags_missing_interval_manifest_reference(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_figure_table_manifest.md",
        "\n".join(
            [
                "# Manifest",
                "Figure 1 Figure 2 Figure 3 Figure 4 Figure 5",
                "Table 1 Table 2 Table 3 Table 4 Table 5",
                "`docs/experiments/paper_camera_ready_tables.md`",
            ]
        )
        + "\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any("missing interval artifact reference" in error for error in report.errors)


def test_check_paper_package_flags_missing_camera_appendix_table(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_camera_ready_tables.md",
        "# Tables\n\n"
        "## Appendix Table A1. Claim Strength Bands\n\n"
        "## Appendix Table A3. Paper-Safe Negative Claims\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any("missing appendix table heading" in error for error in report.errors)


def test_check_paper_package_flags_missing_camera_main_table(tmp_path):
    _write_minimal_package(tmp_path)
    camera_tables = (tmp_path / "docs/experiments/paper_camera_ready_tables.md").read_text(
        encoding="utf-8"
    )
    _write(
        tmp_path / "docs/experiments/paper_camera_ready_tables.md",
        camera_tables.replace("## Table 4. Tier-1 Precision-Validation Status", ""),
    )

    report = run_check_paper_package(tmp_path)

    assert any("missing main table heading" in error for error in report.errors)


def test_check_paper_package_flags_missing_latex_table_label(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "docs/experiments/paper_latex_table_drafts.tex",
        _latex_table_pack().replace(r"\label{tab:pybiber-means}", ""),
    )

    report = run_check_paper_package(tmp_path)

    assert (
        r"docs/experiments/paper_latex_table_drafts.tex missing required table label: \label{tab:pybiber-means}"
        in report.errors
    )


def test_check_paper_package_flags_stale_precision_status(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "artifacts/stage1/validation/precision_validation_status.csv",
        _precision_status_csv().replace(
            "stock_closers,True,False,False,pass,420,50,48,0,2,0,0.960000,0.000000",
            "stock_closers,True,False,False,incomplete,420,20,18,0,2,0,0.900000,0.000000",
        ),
    )

    report = run_check_paper_package(tmp_path)

    assert any("stock_closers gate_status" in error for error in report.errors)


def test_check_paper_package_flags_stale_slop_lexicon_status(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "artifacts/stage1/validation/precision_validation_status.csv",
        _precision_status_csv().replace(
            "slop_lexicon,True,False,False,pass,420,50,50,0,0,0,1.000000,0.000000",
            "slop_lexicon,True,False,False,incomplete,420,8,8,0,0,0,1.000000,0.000000",
        ),
    )

    report = run_check_paper_package(tmp_path)

    assert any("slop_lexicon gate_status" in error for error in report.errors)


def test_check_paper_package_flags_experiments_pybiber_generation_boundary(tmp_path):
    _write_minimal_package(tmp_path)
    _write(
        tmp_path / "EXPERIMENTS.md",
        "# Experiments\n\n"
        "What: run the revised Tier-1 matchers and full pybiber register features over: "
        "pretrain strata, SFT targets, DPO-chosen, DPO-rejected, and later generations "
        "from each checkpoint.\n",
    )

    report = run_check_paper_package(tmp_path)

    assert any("missing pybiber scope boundary" in error for error in report.errors)
    assert any("blurs corpus-side full pybiber" in error for error in report.errors)
