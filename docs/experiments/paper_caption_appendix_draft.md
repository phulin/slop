# Paper Caption And Reproducibility Appendix Draft

Date: 2026-06-18

Purpose: provide submission-facing figure/table captions and a reproducibility
appendix draft without putting repo paths in the main manuscript body. The
planning source remains `docs/experiments/paper_figure_table_manifest.md`.

## Main Figure Captions

**Figure 1. Phase 1 register shift in selected pybiber features.**
Token-weighted pybiber rates for retained English-filtered Dolma, Dolci SFT,
and Dolci DPO chosen/rejected samples. Post-training data move sharply away
from narrative/personal prose markers such as past tense and personal
pronouns, and toward nominalized, adjectival, answer-register exposition. This
is a corpus-side register result; generated-output full pybiber is not claimed
here. Error bars show nonparametric document-bootstrap intervals over retained
rows for the plotted features.

**Figure 2. EQ-Bench Slop Score by checkpoint.**
Aggregate EQ-Bench Slop Score for bounded OLMo and SmolLM3 generation panels.
The score provides a public benchmark bridge and visible lexical slop
diagnostic, but it is not the causal measurement layer. Error bars show
nonparametric 95% bootstrap intervals over valid generated outputs.

**Figure 3. OLMo `slop_lexicon` amplification views.**
Teacher-forced normalized amplification factor, generated-output rate, and
compounding evidence for the original `slop_lexicon` in the bounded OLMo
ladder. The teacher-forced propensity peak occurs at DPO, while the measured
Dolci chosen/rejected contrast does not show chosen-response enrichment for
the same feature.

**Figure 4. Cross-ladder AF comparison.**
Matched OLMo and SmolLM3 amplification-factor coordinates where both ladders
have non-missing support. The positive rank correlation is suggestive evidence
for related style pressure, but the small overlap and different checkpoint
recipes make this a replication-pressure result rather than a universal
stage-localization claim.

**Figure 5. Phase 4 Tier-3 exact AF with denominators.**
Exact sequence-mass teacher-forced amplification factors for
detector-discovered candidate features over the 512-reference OLMo rerun.
Denominator labels are part of the figure because sparse candidates with large
point estimates remain provisional until human perceptibility and precision
validation are complete.

## Main Table Captions

**Table 1. Data and measurement scope.**
Bounded corpus, generation, detector, and teacher-forced rerun sizes used in
the study. The scope is the reduced production estimand, not the originally
proposed exhaustive OLMo generation grid.

**Table 2. Selected full-pybiber register means.**
Token-weighted means for selected pybiber features across retained Dolma,
Dolci SFT, and Dolci DPO chosen/rejected samples. The selected features
summarize the main register movement; the full 67-feature outputs are reported
in the reproducibility appendix.

**Table 3. Current claim strength bands.**
Paper claim IDs grouped by evidence strength and required caveat language.
This table can appear in the appendix or remain a review-control table if the
main text already carries the caveats locally.

**Table 4. Tier-1 precision-validation status.**
Current manual-validation state for the Tier-1 regex features and exploratory
addenda. Passing, demoted, derived, and unlabeled rows define the paper's
claim boundaries rather than serving as headline empirical results.

**Table 5. Paper-safe negative claims.**
Claims the paper should avoid and the bounded replacement wording supported by
the evidence. This table functions as an appendix or review-control
scope-control table; main-text claims should still carry their caveats locally.

## Reproducibility Appendix Draft

### A. Claim And Scope Controls

The claim matrix defines allowed wording and caveats for C1-C14. The claim
evidence map links each claim to manuscript sections, figures/tables, primary
artifacts, and residual caveats. The readiness audit distinguishes paper-grade
evidence from caveated evidence and submission blockers.

Key artifacts:

- `docs/experiments/paper_claim_matrix.md`
- `docs/experiments/paper_reviewer_faq.md`
- `docs/experiments/paper_claim_evidence_map.md`
- `docs/experiments/paper_claim_language_audit.md`
- `docs/experiments/paper_claim_evidence_audit.md`
- `docs/experiments/paper_manuscript_structure_audit.md`
- `docs/experiments/paper_limitations_audit.md`
- `docs/experiments/paper_terminology_audit.md`
- `docs/experiments/paper_numeric_claims_audit.md`
- `docs/experiments/paper_tier1_publication_policy.md`
- `docs/experiments/paper_readiness_audit.md`
- `docs/experiments/paper_submission_exit_audit.md`
- `docs/experiments/paper_reproducibility_manifest.md`
- `docs/experiments/paper_reproduction_runbook.md`
- `docs/experiments/paper_citation_audit.md`
- `artifacts/paper/manuscript/paper_manuscript_structure_audit.csv`
- `artifacts/paper/manuscript/paper_limitations_audit.csv`
- `artifacts/paper/manuscript/paper_terminology_audit.csv`
- `artifacts/paper/manuscript/paper_numeric_claims_audit.csv`
- `artifacts/paper/claims/paper_claim_evidence_audit.csv`
- `artifacts/paper/submission/paper_submission_exit_audit.csv`
- `artifacts/paper/submission/paper_reproducibility_manifest.csv`
- `artifacts/paper/references/paper_citation_audit.csv`
- `docs/experiments/paper_figure_visual_review.md`
- `artifacts/paper/figures/paper_figure_visual_review.csv`

### B. Phase 1 Corpus And Pybiber Artifacts

Phase 1 uses fixed English-filtered OLMo/Dolci samples plus a bounded Dolma
pretraining reference. Full pybiber extraction completed with retained-row
coverage over all Phase 1 samples.

Key artifacts:

- `docs/experiments/phase1_pybiber_register_analysis.md`
- `docs/experiments/phase1_conclusions.md`
- `docs/experiments/phase1_gap_audit.md`
- `artifacts/stage1/census/census_summary.md`
- `artifacts/stage1/census/phase1_pybiber_register_means.csv`
- `artifacts/stage1/census/phase1_pybiber_register_deltas.csv`
- `artifacts/stage1/census/phase1_pybiber_register_intervals.csv`
- `artifacts/stage1/census/olmo3_dolma3_80k_scan_pybiber_full.csv`
- `artifacts/stage1/census/olmo3_dolci_sft_40k_pybiber_full.csv`
- `artifacts/stage1/census/olmo3_dolci_dpo_40k_retained_pybiber_full.csv`

### C. Tier-1 Precision Validation

Tier-1 regex features have scoped publication status. The interim gate passes
`contrastive_negation`, `slop_lexicon`, `stock_openers`, and
`stock_closers`; `rule_of_three_approx` is demoted for independent claims; and
`stock_openers_closers` is a derived pooled view.

Key artifacts:

- `docs/experiments/precision_validation_status.md`
- `artifacts/stage1/validation/precision_validation_status.csv`
- `artifacts/stage1/validation/labels/olmo3_dolci_sft_dpo_10k_labels.csv`
- `artifacts/stage1/validation/labels/revised_phase1_remaining_core_labels.csv`
- `artifacts/stage1/validation/labels/revised_phase1_stock_closers_additional_labels_2026-06-18.csv`
- `artifacts/stage1/validation/labels/revised_phase1_slop_lexicon_additional_labels_2026-06-18.csv`
- `artifacts/stage1/validation/labels/revised_phase1_rule_of_three_additional_labels_2026-06-18.csv`

### D. EQ-Bench Slop Score Bridge

EQ-Bench scoring is included for public benchmark comparability and aggregate
lexical diagnostics. Bootstrap intervals quantify output-sample variability
for the bounded generation panels, but the score remains an aggregate
diagnostic rather than the causal propensity estimand.

Key artifacts:

- `src/slop_sftdiv/features/eqbench_slop.py`
- `src/slop_sftdiv/cli/eqbench_slop_score.py`
- `src/slop_sftdiv/cli/summarize_eqbench_intervals.py`
- `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_stage_comparison.md`
- `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.md`
- `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv`
- `artifacts/phase2/analysis/eqbench_slop/phase1_phase2_eqbench_slop_comparison.csv`

### E. Phase 3 Amplification Spectrum And Generation Panels

The bounded Phase 3 evidence combines teacher-forced propensity,
free-running generation rates, compounding, and cross-ladder comparison. The
reduced SGLang panels are the active production estimand.

Key artifacts:

- `docs/experiments/phase3_integrated_conclusion_report.md`
- `docs/experiments/phase3_completion_audit.md`
- `docs/experiments/phase3_dpo_vs_apo_variation_report.md`
- `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv`
- `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_feature_classification_t07_long1024.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_production.csv`

### F. Phase 4 Detector Discovery And Tier-3 Rerun

Phase 4 produced detector-discovered candidate features and a bounded exact
teacher-forced OLMo rerun. These candidates remain candidate style families
until human perceptibility labels and matcher precision validation are
complete.

Key artifacts:

- `docs/experiments/phase4_detector_discovery_report.md`
- `docs/experiments/phase4_completion_audit.md`
- `docs/experiments/phase4_tier3_teacher_forced_addendum.md`
- `docs/experiments/phase4_human_perceptibility_protocol.md`
- `docs/experiments/phase4_human_annotation_codebook.md`
- `docs/experiments/phase4_human_annotation_readiness.md`
- `docs/experiments/phase4_human_labeling_execution_checklist.md`
- `docs/experiments/phase4_human_perceptibility_summary.md`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_metrics.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_tier3_matcher_specs.json`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_annotation_package.jsonl`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_annotation_readiness.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_pilot_20_each.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each_map.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full_map.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_summary.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_stage_effects.csv`

### G. Paper Figures, Tables, And References

The figure/table manifest tracks draft figure artifacts, captions, source
artifacts, intended claims, and caveats. The reference source inventory and
BibTeX file track checked external provenance.

Key artifacts:

- `docs/experiments/paper_figure_table_manifest.md`
- `docs/experiments/paper_figure_readiness_audit.md`
- `docs/experiments/paper_table_readiness_audit.md`
- `docs/experiments/paper_table_typography_review.md`
- `docs/experiments/paper_venue_decision_checklist.md`
- `docs/experiments/paper_venue_adaptation_runbook.md`
- `docs/experiments/paper_reproducibility_manifest.md`
- `docs/experiments/paper_reproduction_runbook.md`
- `docs/experiments/paper_abstract_conclusion_audit.md`
- `docs/experiments/paper_terminology_audit.md`
- `docs/experiments/paper_numeric_claims_audit.md`
- `docs/experiments/paper_tables.md`
- `docs/experiments/paper_camera_ready_tables.md`
- `docs/experiments/paper_latex_table_drafts.tex`
- `artifacts/paper/figures/figure1_pybiber_register_selected.svg`
- `artifacts/paper/figures/figure2_eqbench_stage_scores.svg`
- `artifacts/paper/figures/figure3_olmo_slop_lexicon_views.svg`
- `artifacts/paper/figures/figure4_cross_ladder_af_scatter.svg`
- `artifacts/paper/figures/figure5_phase4_tier3_raw_af.svg`
- `artifacts/paper/figures/paper_figure_readiness_audit.csv`
- `artifacts/paper/tables/paper_table_readiness_audit.csv`
- `artifacts/paper/tables/paper_table_typography_review.csv`
- `artifacts/paper/submission/paper_venue_decision_checklist.csv`
- `artifacts/paper/submission/paper_reproducibility_manifest.csv`
- `artifacts/paper/claims/paper_claim_language_audit.csv`
- `artifacts/paper/claims/paper_claim_evidence_audit.csv`
- `artifacts/paper/manuscript/paper_abstract_conclusion_audit.csv`
- `artifacts/paper/manuscript/paper_terminology_audit.csv`
- `artifacts/paper/manuscript/paper_numeric_claims_audit.csv`
- `docs/experiments/paper_reference_sources.md`
- `docs/experiments/paper_references.bib`
- `docs/experiments/paper_citation_audit.md`
- `artifacts/paper/references/paper_citation_audit.csv`
