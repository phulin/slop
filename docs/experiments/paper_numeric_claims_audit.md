# Paper Numeric Claims Audit

Machine-readable audit: `artifacts/paper/manuscript/paper_numeric_claims_audit.csv`

Readiness status: `ready`.
This audit checks that headline manuscript numbers match their source CSV/JSON artifacts after manuscript rounding.

| Check | Status | Source | Missing |
|---|---|---|---|
| pybiber_narrative_drop | ready | `artifacts/stage1/census/phase1_pybiber_register_intervals.csv` | none |
| pybiber_expository_shift | ready | `artifacts/stage1/census/phase1_pybiber_register_means.csv` | none |
| pybiber_not_hedging | ready | `artifacts/stage1/census/phase1_pybiber_register_means.csv` | none |
| pybiber_dpo_contrast | ready | `artifacts/stage1/census/phase1_pybiber_register_means.csv` | none |
| eqbench_olmo_stage_scores | ready | `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv` | none |
| eqbench_smollm3_stage_scores | ready | `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv` | none |
| olmo_slop_lexicon_af | ready | `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv` | none |
| olmo_slop_lexicon_preference_data | ready | `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_feature_classification_t07_long1024.csv` | none |
| olmo_slop_lexicon_free_run | ready | `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv` | none |
| olmo_slop_lexicon_compounding | ready | `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv` | none |
| smollm3_slop_lexicon_free_run | ready | `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_production.csv` | none |
| cross_ladder_correlation | ready | `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_production_baselines_data_rates_tf_generation_compounding_slop_neutral_rule3_correlations.csv` | none |
| phase4_attribution_counts | ready | `artifacts/phase4/modernbert_detector_combined_v2_clean/ig_10000doc/phase4_detector_ig_summary.json` | none |
| phase4_tier3_selected_af | ready | `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv` | none |
| phase4_tier3_sparse_af | ready | `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv` | none |
