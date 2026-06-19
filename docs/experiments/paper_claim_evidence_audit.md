# Paper Claim Evidence Audit

Machine-readable audit: `artifacts/paper/claims/paper_claim_evidence_audit.csv`

Readiness status: `ready`.
This audit checks that the claim evidence map covers C1-C14, that mapped source paths exist, and that claim-to-claim evidence inheritance is explicit.

| Claim | Status | Existing paths | Referenced claims | Missing paths | Missing fields |
|---|---|---|---|---|---|
| C1 | ready | artifacts/stage1/census/phase1_pybiber_register_deltas.csv; artifacts/stage1/census/phase1_pybiber_register_intervals.csv; artifacts/stage1/census/phase1_pybiber_register_means.csv; docs/experiments/phase1_pybiber_register_analysis.md | none | none | none |
| C2 | ready | none | C1 | none | none |
| C3 | ready | docs/experiments/phase1_pybiber_register_analysis.md | none | none | none |
| C4 | ready | artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.md; artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_stage_comparison.md; docs/experiments/paper_reference_sources.md; src/slop_sftdiv/cli/eqbench_slop_score.py; src/slop_sftdiv/cli/summarize_eqbench_intervals.py; src/slop_sftdiv/features/eqbench_slop.py | none | none | none |
| C5 | ready | artifacts/phase2/analysis/eqbench_slop/; artifacts/phase2/analysis/eqbench_slop/phase1_phase2_eqbench_slop_comparison.csv; artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv | none | none | none |
| C6 | ready | artifacts/phase2/analysis/eqbench_slop/; artifacts/phase2/analysis/eqbench_slop/phase1_phase2_eqbench_slop_comparison.csv; artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv | none | none | none |
| C7 | ready | artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv; artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_feature_classification_t07_long1024.csv; docs/experiments/phase3_integrated_conclusion_report.md | none | none | none |
| C8 | ready | docs/experiments/phase1_conclusions.md; docs/experiments/phase3_integrated_conclusion_report.md | none | none | none |
| C9 | ready | docs/experiments/paper_results_draft.md; docs/experiments/phase3_integrated_conclusion_report.md | none | none | none |
| C10 | ready | artifacts/phase3/analysis/; docs/experiments/phase3_dpo_vs_apo_variation_report.md; docs/experiments/phase3_integrated_conclusion_report.md | none | none | none |
| C11 | ready | artifacts/phase4/modernbert_detector_combined_v2_clean/; docs/experiments/phase4_detector_discovery_report.md; docs/experiments/phase4_human_annotation_codebook.md; docs/experiments/phase4_human_annotation_readiness.md; docs/experiments/phase4_human_perceptibility_protocol.md; docs/experiments/phase4_human_perceptibility_summary.md | none | none | none |
| C12 | ready | docs/experiments/phase4_completion_audit.md; docs/experiments/phase4_tier3_teacher_forced_addendum.md | none | none | none |
| C13 | ready | docs/experiments/phase4_tier3_teacher_forced_addendum.md | none | none | none |
| C14 | ready | EXPERIMENTS.md; docs/experiments/manual_feature_regex_audit_2026-06-17.md; docs/experiments/phase1_gap_audit.md | none | none | none |
