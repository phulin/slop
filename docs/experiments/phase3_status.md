# Phase 3 Status

Date: 2026-06-15

## Scope

Phase 3 in `EXPERIMENTS.md` asks for the main amplification spectrum:
features x Phase 1 data rates x Phase 2 teacher-forced AF/free-running rates,
then feature-level classifications:

- inherited
- SFT-amplified
- preference-amplified, with signal-driven vs. dynamics-driven cause
- compounding-dominant

It also asks for statistics across the full feature set and SmolLM3
replication with cross-ladder AF rank correlation.

This status file records the first bounded Phase 3 implementation over the
retained OLMo 3/Dolci single-temperature Phase 2 artifacts, the first SmolLM3
no_think calibration ladder, and a 512-prompt SmolLM3 no_think measured slice
with teacher-forced `slop_lexicon`, `neutral_common_controls`,
`rule_of_three_approx`, four-stage free-running generation, compounding, and
an OLMo-vs-SmolLM3 cross-ladder comparison. The SmolLM3 slice now also has a
bounded SmolTalk2 no_think SFT/preference data-rate layer plus a bounded
source-stratified pretraining/Mid baseline. It is real Phase 3 progress, but
it is not the full EXPERIMENTS.md Phase 3 completion because the original
5,000-prompt x 8-completion x 3-temperature production grid, a production
weighted SmolLM3 pretraining feature-rate baseline, broader teacher-forced
feature coverage, and stretch Instruct/Think/RL Zero comparison are not
present. The exact SmolLM3 recipe source weights have now been extracted from
the published configs, but most weighted sources do not yet have local feature
rate samples.

## Implemented Phase 3 Layer

Added CLI:

- `slop-classify-amplification-spectrum`
- `slop-compare-phase3-ladders`
- `slop-plan-phase2-propensity`
- `slop-analyze-phase3-free-run-effects`
- `slop-analyze-phase3-teacher-forced-effects`
- `slop-extract-smollm3-config-weights`

Updated Phase 2 harness support needed for the Phase 3 SmolLM3 replication:

- `slop-free-running-emission` now accepts `--model-revision` and passes it to
  tokenizer/model `from_pretrained`.
- `slop-teacher-forced-propensity` now accepts the same `--model-revision`
  flag.
- `slop-plan-phase2-generation` now accepts stage specs in
  `STAGE=MODEL@REVISION` form and writes a separate `model_revision` column in
  the plan CSV.
- `slop-plan-phase2-propensity` now emits the matching revision-aware
  teacher-forced AF shard checklist for the same stage spec format.
- `slop-free-running-emission` now accepts `--apply-chat-template` and
  `--chat-template-kwargs-json`, allowing SmolLM3 no_think generation commands
  to render prompts through the tokenizer chat template with
  `{"enable_thinking": false}`.
- `slop-plan-phase2-generation` passes those chat-template options through to
  planned generation commands.
- `slop-free-running-emission` and `slop-plan-phase2-generation` also accept
  `--missing-chat-template {error,plain}`. The canonical SmolLM3 generation
  plan uses `plain` fallback so the base checkpoint, which lacks a chat
  template, can remain in the same launch grid while SFT/APO/final use their
  chat templates.
- `slop-free-running-emission` forces `use_cache=True` on model and generation
  configs when those attributes exist. This is needed for inference on the
  SmolLM3 SFT/APO branch configs, which advertise `use_cache=False` and made
  long no_think generation impractically slow without the override.
- `slop-assemble-amplification-spectrum` now writes a model-neutral summary
  title because the same assembler is used for OLMo and SmolLM3 spectra.

Inputs:

- `artifacts/phase2/analysis/olmo3_amplification_spectrum_single_temp_t1_v6.csv`
- `artifacts/stage1/census/olmo3_dolci_dpo_10k_pair_analysis.csv`

Outputs:

- `artifacts/phase3/analysis/olmo3_phase3_bounded_feature_classification_t1.csv`
- `artifacts/phase3/analysis/olmo3_phase3_bounded_feature_classification_t1_summary.md`
- `artifacts/phase3/analysis/olmo3_phase3_teacher_forced_stage_effects_t1.csv`
- `artifacts/phase3/analysis/olmo3_phase3_teacher_forced_stage_effects_t1_summary.md`
- `artifacts/phase3/analysis/olmo3_phase3_free_run_stage_effects_t1.csv`
- `artifacts/phase3/analysis/olmo3_phase3_free_run_stage_effects_t1_summary.md`
- `artifacts/phase3/analysis/olmo3_phase3_bounded_artifact_manifest.csv`
- `artifacts/phase3/analysis/olmo3_phase3_bounded_artifact_manifest.json`
- `artifacts/phase3/analysis/olmo3_phase3_bounded_artifact_manifest.md`
- `artifacts/phase3/analysis/olmo3_phase3_cross_ladder_selfcheck_aligned.csv`
- `artifacts/phase3/analysis/olmo3_phase3_cross_ladder_selfcheck_correlations.csv`
- `artifacts/phase3/analysis/olmo3_phase3_cross_ladder_selfcheck_summary.md`
- `artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_512.jsonl`
- `artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_512_manifest.csv`
- `artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_512_summary.json`
- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512_t1.csv`
- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512_t1.md`
- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512_t1_chat.csv`
- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512_t1_chat.md`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_plan_512_slop_neutral.csv`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_plan_512_slop_neutral.md`
- `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_2prompt_1comp_t1_64tok_smoke.jsonl`
- `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_2prompt_1comp_t1_64tok_smoke_summary.csv`
- `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_2prompt_1comp_t1_64tok_chat_smoke.jsonl`
- `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_2prompt_1comp_t1_64tok_chat_smoke_summary.csv`
- `artifacts/phase2/propensity/smollm3_final_smoltalk2_everyday_no_think_2prompt_slop_neutral_smoke_opportunities.csv`
- `artifacts/phase2/propensity/smollm3_final_smoltalk2_everyday_no_think_2prompt_slop_neutral_smoke_summary.csv`
- `artifacts/phase2/propensity/smollm3_final_smoltalk2_everyday_no_think_2prompt_slop_neutral_smoke_reference_subset_summary.csv`
- `artifacts/phase2/generations/smollm3_base_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration.jsonl`
- `artifacts/phase2/generations/smollm3_base_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration_summary.csv`
- `artifacts/phase2/generations/smollm3_sft_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration.jsonl`
- `artifacts/phase2/generations/smollm3_sft_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration_summary.csv`
- `artifacts/phase2/generations/smollm3_dpo_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration.jsonl`
- `artifacts/phase2/generations/smollm3_dpo_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration_summary.csv`
- `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration.jsonl`
- `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration_summary.csv`
- `artifacts/phase2/propensity/smollm3_base_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_opportunities.csv`
- `artifacts/phase2/propensity/smollm3_base_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_summary.csv`
- `artifacts/phase2/propensity/smollm3_sft_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_opportunities.csv`
- `artifacts/phase2/propensity/smollm3_sft_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_summary.csv`
- `artifacts/phase2/propensity/smollm3_dpo_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_opportunities.csv`
- `artifacts/phase2/propensity/smollm3_dpo_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_summary.csv`
- `artifacts/phase2/propensity/smollm3_final_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_opportunities.csv`
- `artifacts/phase2/propensity/smollm3_final_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_summary.csv`
- `artifacts/phase3/analysis/smollm3_no_think_generation_stage_grid_4prompt_1comp_t1_64tok_chat_calibration.csv`
- `artifacts/phase3/analysis/smollm3_no_think_generation_stage_grid_4prompt_1comp_t1_64tok_chat_calibration_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_4prompt_slop_neutral_calibration.csv`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_4prompt_slop_neutral_calibration_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_4prompt_calibration.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_4prompt_calibration_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_4prompt_calibration.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_4prompt_calibration_summary.md`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_4prompt_calibration_aligned.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_4prompt_calibration_correlations.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_4prompt_calibration_summary.md`
- `artifacts/phase2/propensity/smollm3_base_smoltalk2-everyday-no-think-promptpkg512_slop-vs-neutral_promptpkg512_cached_branch8_sequence_summary.csv`
- `artifacts/phase2/propensity/smollm3_sft_smoltalk2-everyday-no-think-promptpkg512_slop-vs-neutral_promptpkg512_cached_branch8_sequence_summary.csv`
- `artifacts/phase2/propensity/smollm3_dpo_smoltalk2-everyday-no-think-promptpkg512_slop-vs-neutral_promptpkg512_cached_branch8_sequence_summary.csv`
- `artifacts/phase2/propensity/smollm3_final_smoltalk2-everyday-no-think-promptpkg512_slop-vs-neutral_promptpkg512_cached_branch8_sequence_summary.csv`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_512prompt_slop_neutral.csv`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_512prompt_slop_neutral_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_comparison_512prompt_slop_neutral.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_slop_neutral.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_slop_neutral_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_slop_neutral.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_slop_neutral_summary.md`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_slop_neutral_aligned.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_slop_neutral_correlations.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_slop_neutral_summary.md`
- `artifacts/phase2/propensity/smollm3_base_smoltalk2-everyday-no-think-promptpkg512_rule-of-three_promptpkg512_cached_branch8_sequence_summary.csv`
- `artifacts/phase2/propensity/smollm3_sft_smoltalk2-everyday-no-think-promptpkg512_rule-of-three_promptpkg512_cached_branch8_sequence_summary.csv`
- `artifacts/phase2/propensity/smollm3_dpo_smoltalk2-everyday-no-think-promptpkg512_rule-of-three_promptpkg512_cached_branch8_sequence_summary.csv`
- `artifacts/phase2/propensity/smollm3_final_smoltalk2-everyday-no-think-promptpkg512_rule-of-three_promptpkg512_cached_branch8_sequence_summary.csv`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_512prompt_rule_of_three.csv`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_512prompt_rule_of_three_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_comparison_512prompt_rule_of_three.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_teacher_forced_stage_effects_512prompt_tf_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_teacher_forced_stage_effects_512prompt_tf_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_slop_neutral_rule3_aligned.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_slop_neutral_rule3_correlations.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_slop_neutral_rule3_summary.md`
- `artifacts/phase2/generations/smollm3_base_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl`
- `artifacts/phase2/generations/smollm3_base_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256_summary.csv`
- `artifacts/phase2/generations/smollm3_sft_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl`
- `artifacts/phase2/generations/smollm3_sft_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256_summary.csv`
- `artifacts/phase2/generations/smollm3_dpo_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl`
- `artifacts/phase2/generations/smollm3_dpo_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256_summary.csv`
- `artifacts/phase2/generations/smollm3_final_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl`
- `artifacts/phase2/generations/smollm3_final_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256_summary.csv`
- `artifacts/phase3/analysis/smollm3_no_think_generation_stage_grid_512prompt_8comp_t1_chat.csv`
- `artifacts/phase3/analysis/smollm3_no_think_generation_stage_grid_512prompt_8comp_t1_chat_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_generation_stage_comparison_512prompt_8comp_t1_chat.csv`
- `artifacts/phase3/analysis/smollm3_no_think_free_run_stage_effects_512prompt_8comp_t1_chat.csv`
- `artifacts/phase3/analysis/smollm3_no_think_free_run_stage_effects_512prompt_8comp_t1_chat_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_512prompt_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_generation_compounding_512prompt_8comp_t1_chat_tf_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_generation_compounding_512prompt_8comp_t1_chat_tf_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_generation_compounding_512prompt_8comp_t1_chat_tf_slop_neutral_rule3_realized_af.svg`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_generation_compounding_slop_neutral_rule3_aligned.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_generation_compounding_slop_neutral_rule3_correlations.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_generation_compounding_slop_neutral_rule3_summary.md`
- `artifacts/stage1/corpora/smollm3_smoltalk2_sft_everyday_no_think_2260.jsonl`
- `artifacts/stage1/corpora/smollm3_smoltalk2_sft_everyday_no_think_2260_summary.md`
- `artifacts/stage1/corpora/smollm3_smoltalk2_preference_tulu_no_think_10k_pairs.jsonl`
- `artifacts/stage1/corpora/smollm3_smoltalk2_preference_tulu_no_think_10k_pairs_summary.md`
- `artifacts/stage1/census/smollm3_smoltalk2_sft2260_pref10k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_smoltalk2_pref10k_tier1_pair_deltas.csv`
- `artifacts/stage1/census/smollm3_smoltalk2_pref10k_tier1_pair_analysis.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_data_rates_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_data_rates_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_data_rates_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_data_rates_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_data_rates_tf_generation_compounding_slop_neutral_rule3_aligned.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_data_rates_tf_generation_compounding_slop_neutral_rule3_correlations.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_data_rates_tf_generation_compounding_slop_neutral_rule3_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_fineweb_edu_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fineweb_edu_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_stackexchange_apple_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_stackexchange_apple_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_smoltalk2_mid_llama_nemotron_reasoning_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_smoltalk2_mid_llama_nemotron_reasoning_2k_summary.md`
- `artifacts/stage1/census/smollm3_pretrain_mid_baselines_2k_tier1_feature_rates.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_baselines_data_rates_tf_generation_compounding_slop_neutral_rule3_aligned.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_baselines_data_rates_tf_generation_compounding_slop_neutral_rule3_correlations.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_baselines_data_rates_tf_generation_compounding_slop_neutral_rule3_summary.md`

W&B:

- `stage3-phase3-olmo3-bounded-feature-classification-t1-v4-pref-fdr`
  (`4oabv7j8`)
- `stage3-phase3-olmo3-bounded-artifact-manifest-v3` (`nlh83sd9`)

The classifier emits one row per feature with:

- Phase 3 primary class.
- preference cause (`signal-driven`, `dynamics-driven`, or
  `not-preference-amplified`).
- Phase 1 data rates.
- base/SFT/DPO/final AF values.
- AF stage jumps.
- free-running stage rates.
- compounding summary fields where available.
- paired Result A sign-test p-values and BH-FDR q-values where available.
- explicit AF-stage FDR caveat via
  `fdr_status=preference_pair_fdr_available_af_fdr_missing`.

The cross-ladder comparator aligns two amplification-spectrum CSVs by
feature-stage pair and reports AF Spearman/Pearson correlations overall and by
stage. It uses normalized AF where present, otherwise raw AF. The current
OLMo-vs-OLMo self-check aligned 24 feature-stage rows and returned overall
Spearman AF `1.000`, which verifies the command path. The first
OLMo-vs-SmolLM3 calibration comparison aligned 24 rows but had only four
shared AF values, all constant on the SmolLM3 side, so Spearman/Pearson AF are
blank. That verified the real cross-ladder command path but was not a
statistically interpretable replication result. The generation-inclusive
512-prompt SmolLM3 comparison now aligns 24 feature-stage rows across the
shared measured feature set, with 8 shared AF values from `slop_lexicon` and
the `rule_of_three_approx` comma-pair proxy. It reports overall Spearman AF
`0.762` and Pearson AF `0.978`. This is interpretable for the bounded
teacher-forced/free-running/compounding slice. The current data-rate version
uses the same shared AF support and adds SmolTalk2 SFT/preference rates for
SmolLM3 classification, but it is still not the full production cross-ladder
comparison because the original production grid, exact weighted SmolLM3
recipe-weighted pretraining feature rates, and broader teacher-forced feature
support are missing.

The free-running stage-effect analyzer runs paired sign tests over shared
`(record_id, completion_index, temperature, top_p)` generation units and
applies BH-FDR across the requested feature/comparison family. On the retained
OLMo target-shape `t=1.0` caches it produced 18 feature-comparison rows and 15
BH-FDR-significant rows.

Key free-running FDR reads:

- SFT -> DPO increases all six retained feature views after BH-FDR:
  `contrastive_negation`, `rule_of_three_approx`, `slop_lexicon`,
  `stock_closers`, `stock_openers`, and pooled `stock_openers_closers`.
- DPO -> final/RLVR significantly attenuates `slop_lexicon`,
  `stock_closers`, and pooled `stock_openers_closers`.
- DPO -> final/RLVR changes for `rule_of_three_approx`, `stock_openers`, and
  `contrastive_negation` are not significant under this sign-test/BH-FDR
  family.

The teacher-forced stage-effect analyzer runs paired sign tests over shared
teacher-forced opportunity rows and applies BH-FDR across the retained
feature/comparison family. On the retained OLMo teacher-forced opportunity
grids it produced 18 feature-comparison rows and 17 BH-FDR-significant rows.
The regenerated classifier joins both teacher-forced and free-running
stage-effect q-values into the feature-level Phase 3 table.

Key teacher-forced FDR reads:

- SFT -> DPO has BH-FDR-significant teacher-forced probability-mass changes for
  all retained measured feature views, including `slop_lexicon`.
- DPO -> final/RLVR has BH-FDR-significant teacher-forced changes for every
  measured feature view except `stock_openers`, although the sign-test
  direction can differ from the mean AF-delta direction when many small pairs
  move one way and fewer larger pairs move the other way.
- `contrastive_negation` remains absent from teacher-forced stage-effect rows
  because its retained Phase 2 teacher-forced support is missing.

## Classification Rules

Default rule settings:

| Rule | Threshold |
|---|---:|
| High SFT-data rate | `>= 0.25` per 1k tokens |
| AF approximately 1 | within `+/- 0.25` |
| Amplified AF | `>= 1.25` |
| Preference-stage absolute AF jump | `>= 0.25` from max(base, SFT) to max(DPO, final) |
| Preference-stage relative AF jump | `>= 0.10` |
| Preference-data complicity | chosen - rejected `> 0.0` per 1k tokens |
| Compounding-dominant | max AF `<= 1.25`, max compounding excess `>= 0.1`, max realized AF `>= 1.1` |

The relative-jump rule is important for high-magnitude AF features such as
`stock_closers`, where a tiny relative change should not be labeled as a
preference-stage jump.

## Current Bounded OLMo Classifications

| Feature | Class | Cause | Key read |
|---|---|---|---|
| `slop_lexicon` | preference-amplified | dynamics-driven | DPO AF rises from SFT (`1.695`) to DPO (`1.999`), but paired Result A evidence is nonsignificant and rejected-greater-chosen (`BH q=0.963`), so the bounded label is dynamics-driven rather than signal-driven. |
| `stock_closers` | SFT-amplified | not preference-amplified | AF is already very high at base/SFT and does not clear the relative preference-jump rule. |
| `stock_openers_closers` | SFT-amplified | not preference-amplified | Pooled AF is high from the start; this remains mostly a closer-side effect from Phase 2. |
| `rule_of_three_approx` | measured-no-phase3-class | not preference-amplified | Teacher-forced comma-pair extension proxy is below 1 across stages and peaks at SFT, not DPO. |
| `stock_openers` | measured-no-phase3-class | not preference-amplified | Teacher-forced AF is far below 1 and declines through DPO/final. |
| `contrastive_negation` | observed-output-only | not preference-amplified | Free-running output exists, but teacher-forced support is missing and held-out references are sparse. |

Class counts:

| Class | Count |
|---|---:|
| preference-amplified | 1 |
| SFT-amplified | 2 |
| measured-no-phase3-class | 2 |
| observed-output-only | 1 |

## Phase 3 Requirement Audit

| EXPERIMENTS.md requirement | Current evidence | Status |
|---|---|---|
| Assemble features x data rates x AF/free-running rates | `olmo3_amplification_spectrum_single_temp_t1_v6.csv` has 24 feature-stage rows over six retained feature views and four OLMo stages. | Bounded OLMo done |
| Classify each feature as inherited/SFT/preference/compounding | `olmo3_phase3_bounded_feature_classification_t1.csv` classifies six retained feature views. | Bounded OLMo done |
| Cross-reference preference-amplified features with chosen-vs-rejected complicity | Classifier uses paired Result A sign-test BH-FDR to label `slop_lexicon` dynamics-driven. | Bounded OLMo done |
| Compounding-dominant classification | Rule implemented; no current retained feature qualifies under default thresholds. | Implemented, no positive class |
| Cluster bootstrap CIs | Existing Phase 2 AF table includes bootstrap CIs where teacher-forced measurements exist. | Partially done from Phase 2 |
| Benjamini-Hochberg FDR across full feature set | Result A chosen-vs-rejected sign-test BH-FDR is joined from `olmo3_dolci_dpo_10k_pair_analysis.csv`; target-shape OLMo free-running stage effects have paired sign-test BH-FDR in `olmo3_phase3_free_run_stage_effects_t1.csv`; retained OLMo teacher-forced stage effects have paired opportunity-level sign-test BH-FDR in `olmo3_phase3_teacher_forced_stage_effects_t1.csv`; both stage-effect families are joined into the regenerated classifier table. | Bounded OLMo done |
| Paired designs wherever corpora share prompts | Preference-complicity uses paired Phase 1 sign-test BH-FDR. Free-running stage effects use paired shared-prompt/completion sign tests. Teacher-forced stage effects use paired shared-opportunity sign tests. | Bounded OLMo done |
| Replicate full spectrum on SmolLM3-3B no_think ladder | A 512-row no_think SmolTalk2 prompt package, canonical chat-template generation plan, propensity plan, all-stage 4-prompt calibration, full 512-prompt four-stage SmolLM3 teacher-forced spectrum for `slop_lexicon`, `neutral_common_controls`, and `rule_of_three_approx`, full 512-prompt x 8-completion free-running caches at `t=1.0`, paired free-running FDR, compounding, SmolTalk2 SFT/preference data rates, bounded FineWeb-Edu/DCLM/FineWeb2 German/FineWeb2 Spanish/PES2O/StackExchange pretraining-source baselines, bounded SmolTalk2 Mid rates, extracted SmolLM3 recipe weights, a coverage-aware weighted pretrain baseline proxy, and a generation-inclusive SmolLM3 spectrum/classification now exist. Missing pieces are the original 5,000-prompt x 8 x 3-temperature production grid, production weighted pretraining feature-rate coverage beyond the current 72.895% recipe share for retained Tier-1 features, and broader teacher-forced support for sparse features. | Bounded generation-inclusive SmolLM3 slice with baseline data rates done; full production spectrum incomplete |
| Report cross-ladder AF rank correlation | `slop-compare-phase3-ladders` implements the statistic, passes an OLMo self-check, aligns OLMo vs. the SmolLM3 4-prompt calibration spectrum, reports a 512-prompt teacher-forced slop-lexicon-only comparison with Spearman AF `0.400`, reports a two-feature teacher-forced comparison with Spearman AF `0.762`, and reports the generation-inclusive bounded comparison with 24 aligned rows and 8 shared AF values, Spearman AF `0.762`, Pearson AF `0.978`. The data-rate comparison is the current best bounded comparison, but full production correlation remains missing until the shared feature scope has broader teacher-forced coverage where support allows and the intended production prompt/temperature grid is complete. | Bounded generation-inclusive data-rate comparison done; full production correlation incomplete |
| Stretch Instruct vs. Think vs. RL Zero comparison | Not started. | Stretch missing |

## SmolLM3 Replication Readiness

The Phase 3 replication ladder needs Hugging Face revisions because several
SmolLM3 training stages are branches of `HuggingFaceTB/SmolLM3-3B-checkpoints`
rather than separate model repo IDs. Current branch tips verified on
2026-06-15:

| Stage candidate | Model repo | Revision | Current SHA |
|---|---|---|---|
| Base | `HuggingFaceTB/SmolLM3-3B-Base` | `main` | `d78a42f79198603e614095753484a04c10c2b940` |
| Mid-training reference | `HuggingFaceTB/SmolLM3-3B-checkpoints` | `it-mid-training` | `0485ec16b618f88f6dbb27b23dbdecca1d6a1cdd` |
| SFT | `HuggingFaceTB/SmolLM3-3B-checkpoints` | `it-SFT` | `f6ddaa5f2e99f24ea507596c214595769fb06387` |
| Preference/APO soup | `HuggingFaceTB/SmolLM3-3B-checkpoints` | `it-soup-APO` | `cfb32d505f5025ec9be4e704f70cfbf5bdf8da94` |
| Long-context expert candidate | `HuggingFaceTB/SmolLM3-3B-checkpoints` | `it-LC-expert` | `99d07b0a954f07f1237176440b0a61149c7d03d3` |
| Final | `HuggingFaceTB/SmolLM3-3B` | `main` | `a07cc9a04f16550a088caea529712d1d335b0ac1` |

The likely no_think Phase 3 ladder is base -> SFT -> preference/APO soup ->
final. The `it-mid-training` and `it-LC-expert` branches are useful audit
points but are not direct replacements for the SFT/APO/final cells in the
OLMo-style spectrum.

Example generation-plan stage specs now supported:

```bash
--stage base=HuggingFaceTB/SmolLM3-3B-Base \
--stage sft=HuggingFaceTB/SmolLM3-3B-checkpoints@it-SFT \
--stage dpo=HuggingFaceTB/SmolLM3-3B-checkpoints@it-soup-APO \
--stage final=HuggingFaceTB/SmolLM3-3B
```

This removed the first harness blocker for SmolLM3 Phase 2/3 replication. The
prompt package, calibration ladder, 512-prompt teacher-forced slop/rule
summaries, 512-prompt x 8 free-running ladder, compounding layer, and
generation-inclusive assembled spectrum now exist. The bounded SmolTalk2
SFT/preference data-rate layer and bounded pretraining/Mid baseline also
exist. Remaining gaps are the original production-scale grid, production
recipe-weighted SmolLM3 pretraining feature rates, and broader teacher-forced
feature coverage where denominator support allows it.

Local W&B-disabled planner smoke outputs:

- `artifacts/phase3/analysis/smollm3_revision_generation_plan_smoke.csv`
- `artifacts/phase3/analysis/smollm3_revision_generation_plan_smoke.md`
- `artifacts/phase3/analysis/smollm3_revision_propensity_plan_smoke.csv`
- `artifacts/phase3/analysis/smollm3_revision_propensity_plan_smoke.md`

The smoke plans use the existing no_think SmolTalk2 smoke file and verify that
SFT/APO rows emit `--model-revision it-SFT` and
`--model-revision it-soup-APO` in both planned free-running and teacher-forced
commands. They are planner contract checks only; they do not run model
generation or propensity scoring.

## SmolLM3 Replication Inputs Started

The first real SmolLM3 no_think Phase 2 input artifact now exists locally:

- Prompt package:
  `artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_512.jsonl`
- Manifest:
  `artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_512_manifest.csv`
- Summary:
  `artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_512_summary.json`

It was streamed from `HuggingFaceTB/smoltalk2`, config `SFT`, split
`smoltalk_smollm3_everyday_conversations_no_think`, with
`sample_size=512`, `max_scan=5000`, `sampling_strategy=hash_reservoir`, and
seed `1729`. The split yielded 2,260 scanned rows before exhaustion, 2,258
eligible prompts after near-duplicate filtering, and 512 selected prompts. The
selected package contains 20,081 prompt tokens and 43,390 target tokens.

Three launch plans were generated against this prompt package:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512_t1.csv`
  and `.md`: the first plain-prompt base/SFT/APO/final generation plan. This
  is retained for audit only, because a live smoke showed plain prompts can
  cause SmolLM3 to continue the dialogue as user text rather than answer as an
  assistant.
- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512_t1_chat.csv`
  and `.md`: the current launch plan to use for SmolLM3 no_think generation.
  It adds `--apply-chat-template --chat-template-kwargs-json
  '{"enable_thinking": false}'` to each base/SFT/APO/final command. It keeps
  the same `512` prompts, `8` completions, `t=1.0`, `top_p=0.95`, and
  `max_new_tokens=1024`. Estimated missing A100-hours: `13.09`.
- `artifacts/phase3/analysis/smollm3_no_think_propensity_plan_512_slop_neutral.csv`
  and `.md`: base/SFT/APO/final teacher-forced `slop_lexicon` plus
  `neutral_common_controls`, sequence mass, neutral normalization. Estimated
  missing A100-hours: `23.30`.

A tiny final-checkpoint generation smoke also completed on the A100:

```bash
uv run slop-free-running-emission \
  --model HuggingFaceTB/SmolLM3-3B \
  --input artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_512.jsonl \
  --sample-size 2 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-new-tokens 64 \
  --completions-per-prompt 1 \
  --generation-batch-size 2 \
  --dtype bfloat16 \
  --wandb-mode disabled \
  --no-torch-compile
```

Outputs:

- `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_2prompt_1comp_t1_64tok_smoke.jsonl`
- `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_2prompt_1comp_t1_64tok_smoke_summary.csv`

The smoke produced two 64-token final-checkpoint generations with valid
feature-count JSON and no obvious `<think>`/`</think>`/`reasoning` markup in
the sampled generations. This verifies model loading, CUDA execution, the new
prompt package, and the generation output schema. It is not a Phase 3
replication result because it covers only two prompts from the final
checkpoint.

That first smoke used plain prompts and surfaced a launch-quality issue: the
final SmolLM3 checkpoint continued the dialogue in user style on the sampled
prompts. The generation harness and planner were updated to support tokenizer
chat-template rendering, and a replacement chat-template smoke completed:

```bash
uv run slop-free-running-emission \
  --model HuggingFaceTB/SmolLM3-3B \
  --input artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_512.jsonl \
  --sample-size 2 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-new-tokens 64 \
  --completions-per-prompt 1 \
  --generation-batch-size 2 \
  --apply-chat-template \
  --chat-template-kwargs-json '{"enable_thinking": false}' \
  --dtype bfloat16 \
  --wandb-mode disabled \
  --no-torch-compile
```

Outputs:

- `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_2prompt_1comp_t1_64tok_chat_smoke.jsonl`
- `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_2prompt_1comp_t1_64tok_chat_smoke_summary.csv`

The replacement smoke produced assistant-style completions, valid
feature-count JSON, and no obvious generated reasoning markup. This
chat-template path supersedes the plain-prompt path for SmolLM3 no_think
free-running generation.

A matching tiny final-checkpoint teacher-forced smoke also completed:

```bash
uv run slop-teacher-forced-propensity \
  --model HuggingFaceTB/SmolLM3-3B \
  --input artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_512.jsonl \
  --sample-size 2 \
  --feature slop_lexicon \
  --feature neutral_common_controls \
  --normalization-feature neutral_common_controls \
  --max-opportunities 32 \
  --max-token-start-opportunities 32 \
  --mass-mode sequence \
  --dtype bfloat16 \
  --wandb-mode disabled \
  --sequence-cache \
  --no-torch-compile
```

Outputs:

- `artifacts/phase2/propensity/smollm3_final_smoltalk2_everyday_no_think_2prompt_slop_neutral_smoke_opportunities.csv`
- `artifacts/phase2/propensity/smollm3_final_smoltalk2_everyday_no_think_2prompt_slop_neutral_smoke_summary.csv`
- `artifacts/phase2/propensity/smollm3_final_smoltalk2_everyday_no_think_2prompt_slop_neutral_smoke_reference_subset_summary.csv`

The smoke wrote 64 opportunity rows and two feature summaries
(`slop_lexicon` and `neutral_common_controls`). Both features had zero
reference initiations in the 2-prompt smoke sample, so this verifies model
loading, exact sequence-mass scoring, CUDA execution, and output schema only;
it is not an interpretable AF result.

## SmolLM3 Four-Stage Calibration Ladder

The first all-stage SmolLM3 no_think calibration ladder now exists locally.
This is still a plumbing/calibration shard, not a Phase 3 replication result:
it uses only 4 prompts, 1 completion per prompt, and 64 generated tokens per
completion. It is useful because it verifies all four model stages, the
chat-template no_think path, the base-checkpoint plain fallback, the Phase 2
summary schema, the Phase 3 spectrum assembler, the classifier, and the real
OLMo-vs-SmolLM3 comparator path.

Generation calibration shape:

- stages: base, SFT, APO-soup, final
- prompts per stage: 4
- completions per prompt: 1
- max new tokens: 64
- temperature: 1.0
- top-p: 0.95
- chat-template handling:
  `--apply-chat-template --chat-template-kwargs-json '{"enable_thinking": false}' --missing-chat-template plain`

Generation outputs:

- `artifacts/phase2/generations/smollm3_base_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration.jsonl`
- `artifacts/phase2/generations/smollm3_base_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration_summary.csv`
- `artifacts/phase2/generations/smollm3_sft_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration.jsonl`
- `artifacts/phase2/generations/smollm3_sft_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration_summary.csv`
- `artifacts/phase2/generations/smollm3_dpo_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration.jsonl`
- `artifacts/phase2/generations/smollm3_dpo_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration_summary.csv`
- `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration.jsonl`
- `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration_summary.csv`

Free-running feature rates per 1k generated tokens in the calibration shard:

| Feature | Base | SFT | APO | Final |
|---|---:|---:|---:|---:|
| `rule_of_three_approx` | 0.000 | 7.812 | 11.719 | 15.625 |
| `slop_lexicon` | 3.906 | 0.000 | 0.000 | 0.000 |
| `contrastive_negation` | 0.000 | 0.000 | 0.000 | 0.000 |
| `stock_openers` | 0.000 | 0.000 | 0.000 | 0.000 |
| `stock_closers` | 0.000 | 0.000 | 0.000 | 0.000 |
| `stock_openers_closers` | 0.000 | 0.000 | 0.000 | 0.000 |

Teacher-forced calibration shape:

- stages: base, SFT, APO-soup, final
- prompts per stage: 4
- features: `slop_lexicon`, `neutral_common_controls`
- max opportunities per feature: 32
- max token-start opportunities: 32
- mass mode: sequence
- normalization feature: `neutral_common_controls`

Teacher-forced outputs:

- `artifacts/phase2/propensity/smollm3_base_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_opportunities.csv`
- `artifacts/phase2/propensity/smollm3_base_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_summary.csv`
- `artifacts/phase2/propensity/smollm3_sft_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_opportunities.csv`
- `artifacts/phase2/propensity/smollm3_sft_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_summary.csv`
- `artifacts/phase2/propensity/smollm3_dpo_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_opportunities.csv`
- `artifacts/phase2/propensity/smollm3_dpo_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_summary.csv`
- `artifacts/phase2/propensity/smollm3_final_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_opportunities.csv`
- `artifacts/phase2/propensity/smollm3_final_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_summary.csv`

All four teacher-forced calibration summaries have zero reference initiations
for both `slop_lexicon` and `neutral_common_controls`. The summaries still
verify that mean probability mass is produced for every stage, but raw AF and
normalized AF are `0.0` because the denominator is zero. Therefore this shard
must not be interpreted as a SmolLM3 propensity result.

Assembled Phase 3 calibration outputs:

- `artifacts/phase3/analysis/smollm3_no_think_generation_stage_grid_4prompt_1comp_t1_64tok_chat_calibration.csv`
- `artifacts/phase3/analysis/smollm3_no_think_generation_stage_grid_4prompt_1comp_t1_64tok_chat_calibration_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_4prompt_slop_neutral_calibration.csv`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_4prompt_slop_neutral_calibration_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_4prompt_calibration.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_4prompt_calibration_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_4prompt_calibration.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_4prompt_calibration_summary.md`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_4prompt_calibration_aligned.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_4prompt_calibration_correlations.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_4prompt_calibration_summary.md`

The OLMo-vs-SmolLM3 calibration comparison aligned 24 feature-stage rows. It
had 4 shared AF values, all from `slop_lexicon` stage rows, but Spearman and
Pearson AF were blank because the SmolLM3 AF vector is constant zero. This is
expected from the zero-reference calibration shard and confirms that a larger
teacher-forced run is required before reporting a cross-ladder AF rank
correlation.

## SmolLM3 512-Prompt Teacher-Forced Ladder

The first full-size SmolLM3 no_think teacher-forced ladder now exists for
`slop_lexicon`, `neutral_common_controls`, and the `rule_of_three_approx`
comma-pair extension proxy. It uses the 512-prompt SmolTalk2 no_think prompt
package, sequence mass, prefix cache branch 8, fixed prefix 256,
`max_opportunities=512`, `max_token_start_opportunities=128`,
`max_prefix_tokens=1024`, bfloat16, and torch compile.

Slop/neutral W&B runs:

| Stage | W&B run | Wall seconds | Opportunities/sec |
|---|---|---:|---:|
| Base | `q7xrq9x6` | 1570.651 | 55.129 |
| SFT | `rmwh8p2z` | 1334.807 | 64.869 |
| APO | `e4xukin8` | 1334.306 | 64.894 |
| Final | `cy8si8df` | 1335.347 | 64.843 |

Rule-of-three W&B runs:

| Stage | W&B run | Wall seconds | Opportunities/sec |
|---|---|---:|---:|
| Base | `xud6jfor` | 102.165 | 7.380 |
| SFT | `9alxeeh7` | 57.359 | 13.145 |
| APO | `6j3vuu0e` | 57.042 | 13.218 |
| Final | `jdruuqay` | 57.821 | 13.040 |

All four stages scored the same opportunities per feature. The reference
target package has 19 `slop_lexicon` initiations
(`reference_rate=0.00043886`) and 2,915 `neutral_common_controls` initiations
(`reference_rate=0.06733035`) over 43,294 token-start opportunities per
feature. For `rule_of_three_approx`, it has 628 references
(`reference_rate=0.83289125`) over 754 comma-pair extension opportunities.

Teacher-forced `slop_lexicon` AF:

| Stage | Raw AF | Raw AF 95% CI | Neutral-normalized AF | Normalized AF 95% CI |
|---|---:|---:|---:|---:|
| Base | 0.444 | 0.263-0.844 | 6.902 | 4.027-13.168 |
| SFT | 2.980 | 1.906-5.303 | 7.647 | 4.940-13.786 |
| APO | 3.323 | 2.154-5.795 | 8.141 | 5.232-14.610 |
| Final | 3.338 | 2.174-5.847 | 8.345 | 5.357-15.008 |

Teacher-forced `rule_of_three_approx` comma-pair extension AF:

| Stage | Raw AF | Raw AF 95% CI |
|---|---:|---:|
| Base | 0.673 | 0.647-0.697 |
| SFT | 0.741 | 0.715-0.764 |
| APO | 0.755 | 0.729-0.779 |
| Final | 0.755 | 0.728-0.778 |

Stage deltas in normalized AF:

| Comparison | Delta |
|---|---:|
| SFT - Base | +0.745 |
| APO - SFT | +0.494 |
| Final - APO | +0.204 |
| Final - Base | +1.443 |

Stage deltas in `rule_of_three_approx` raw AF:

| Comparison | Delta |
|---|---:|
| SFT - Base | +0.068 |
| APO - SFT | +0.014 |
| Final - APO | -0.00008 |
| Final - Base | +0.082 |

Paired teacher-forced stage-effect tests over the SmolLM3 opportunity rows
produced 9 rows, with 8 significant after BH-FDR at alpha `0.05`. SFT vs. APO
and APO vs. final are statistically significant for `slop_lexicon` and
`rule_of_three_approx`, but the sign-test direction can differ from the mean
AF delta when many small probability changes oppose fewer larger changes.

Assembled outputs:

- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_512prompt_slop_neutral.csv`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_512prompt_slop_neutral_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_comparison_512prompt_slop_neutral.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_slop_neutral.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_slop_neutral_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_slop_neutral.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_slop_neutral_summary.md`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_slop_neutral_aligned.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_slop_neutral_correlations.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_slop_neutral_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_512prompt_rule_of_three.csv`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_512prompt_rule_of_three_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_comparison_512prompt_rule_of_three.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_teacher_forced_stage_effects_512prompt_tf_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_teacher_forced_stage_effects_512prompt_tf_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_slop_neutral_rule3_aligned.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_slop_neutral_rule3_correlations.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_slop_neutral_rule3_summary.md`

The OLMo-vs-SmolLM3 512-prompt teacher-forced comparison aligns four
`slop_lexicon` stage rows. It reports overall Spearman AF `0.400` and Pearson
AF `0.685`. The expanded comparison aligns eight rows across `slop_lexicon`
and `rule_of_three_approx`, reporting overall Spearman AF `0.762` and Pearson
AF `0.978`. This is interpretable for the current teacher-forced slice. It is
now superseded for Phase 3 status by the generation-inclusive comparison below,
which adds SmolLM3 free-running and compounding evidence but keeps the same
two-feature AF support.

The TF-only SmolLM3 classification table labels `slop_lexicon` as
`sft-amplified`, because AF is already far above the default amplified
threshold at the SFT stage and the APO/final relative increase over SFT is
small. It labels `rule_of_three_approx` as `measured-no-phase3-class`, because
the raw AF remains below 1 across all stages. This table is now superseded for
Phase 3 status by the generation-inclusive classifier below.

## SmolLM3 512-Prompt Free-Running And Compounding Ladder

The bounded SmolLM3 no_think free-running ladder now exists for the same
512-prompt SmolTalk2 prompt package. It uses `8` completions per prompt,
temperature `1.0`, `top_p=0.95`, `max_new_tokens=1024`, bfloat16,
`torch.compile`, chat-template rendering with `{"enable_thinking": false}`,
and plain fallback for the base checkpoint.

Generation W&B runs:

| Stage | W&B run | Generated tokens | Wall seconds | Tokens/sec |
|---|---|---:|---:|---:|
| Base | `stcaaobk` | 4,194,304 | 2126.854 | 1972.069 |
| SFT | `prb8bw1b` | 3,113,728 | 1412.498 | 2204.413 |
| APO | `mewe19y4` | 3,401,984 | 1549.398 | 2195.682 |
| Final | `4oxiq1xr` | 3,164,416 | 1385.823 | 2283.421 |

The APO generation initially stalled under the checkpoint's advertised
`use_cache=False` config. The generation harness now enables cache for
inference when the model exposes `config.use_cache` or
`generation_config.use_cache`; after that fix the full APO shard completed at
normal throughput. Failed pre-fix APO attempts were discarded and are not part
of the retained artifacts.

Free-running rates per 1,000 generated tokens:

| Feature | Base | SFT | APO | Final | Max stage |
|---|---:|---:|---:|---:|---|
| `slop_lexicon` | 0.118 | 0.269 | 0.376 | 0.391 | Final |
| `rule_of_three_approx` | 1.248 | 4.081 | 5.516 | 5.513 | APO |
| `contrastive_negation` | 0.092 | 0.094 | 0.096 | 0.105 | Final |
| `stock_openers` | 0.137 | 0.069 | 0.159 | 0.137 | APO |
| `stock_closers` | 0.056 | 0.106 | 0.124 | 0.133 | Final |
| `stock_openers_closers` | 0.193 | 0.174 | 0.284 | 0.271 | APO |

Paired free-running stage-effect tests over shared
`(record_id, completion_index, temperature, top_p)` generation units produced
18 rows, with 8 BH-FDR-significant at alpha `0.05`. The strongest reads are:
base -> SFT sharply increases `rule_of_three_approx` and `slop_lexicon`;
SFT -> APO significantly increases `rule_of_three_approx`, `stock_openers`,
pooled stock phrases, and `slop_lexicon`; APO -> final changes are not
significant for the retained feature family, including `slop_lexicon`
(`q=0.135`) and `rule_of_three_approx` (`q=0.924`).

The generation-inclusive compounding join uses the combined SmolLM3
slop/neutral/rule3 teacher-forced propensity grid. `slop_lexicon` direct
prior/no-prior windows remain positive at every stage:

| Stage | Obs/1k Opp | Exp/1k Opp | Excess/1k | Realized AF | P(hit after prior) | P(hit no prior) | Repeat gens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.285 | 0.195 | +0.090 | 0.649 | 0.0883 | 0.0040 | 77 |
| SFT | 0.900 | 1.308 | -0.408 | 2.051 | 0.0632 | 0.0231 | 153 |
| APO | 0.983 | 1.458 | -0.475 | 2.240 | 0.0633 | 0.0246 | 235 |
| Final | 0.995 | 1.465 | -0.470 | 2.267 | 0.0634 | 0.0252 | 234 |

The observed-vs-expected excess is positive only at base for `slop_lexicon`;
SFT/APO/final have high direct prior/no-prior risk ratios but lower observed
opportunity rates than the teacher-forced expectation. For
`rule_of_three_approx`, the comma-pair extension proxy has negative base
excess and positive SFT/APO/final excess: base `-229.840`, SFT `+187.735`, APO
`+149.541`, final `+151.376` per 1,000 opportunities.

The generation-inclusive SmolLM3 spectrum/classifier outputs are:

- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_slop_neutral_rule3_summary.md`

The classifier labels `slop_lexicon` as `sft-amplified`, `rule_of_three_approx`
and `neutral_common_controls` as `measured-no-phase3-class`, and the remaining
features as `observed-output-only` because teacher-forced support is missing
or sparse. The free-running maximum is Final/RLVR for `slop_lexicon`,
`contrastive_negation`, and `stock_closers`; APO for `rule_of_three_approx`,
`stock_openers`, and pooled stock phrases.

The generation-inclusive OLMo-vs-SmolLM3 comparison outputs are:

- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_generation_compounding_slop_neutral_rule3_aligned.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_generation_compounding_slop_neutral_rule3_correlations.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_generation_compounding_slop_neutral_rule3_summary.md`

It aligns 24 feature-stage rows and has 8 shared AF values from
`slop_lexicon` and the `rule_of_three_approx` teacher-forced proxy. Overall
Spearman AF is `0.762`; Pearson AF is `0.978`. Per-stage correlations are
`1.000` because each stage has only two shared AF values. This was the best
bounded cross-ladder result before adding SmolLM3 data rates; the data-rate
version below is now the preferred bounded comparison. Neither is a full
production Phase 3 completion.

## SmolLM3 Baseline And Preference Data-Rate Layer

The bounded SmolLM3 no_think spectrum now has source-stratified baseline data
rates: six pretraining-collection source samples, one SmolTalk2 Mid sample,
the no_think SFT target sample, and the candidate no_think Tulu preference
sample. This closes the bounded data-rate gap for Phase 3 classification. It
does not yet reconstruct a full production weighted SmolLM3 pretraining
feature-rate baseline.

Sampled inputs:

| Sample | Rows | Tokens | Notes |
|---|---:|---:|---|
| FineWeb-Edu pretraining source | 2,000 | 1,610,774 | Web source from the SmolLM3 pretraining collection. |
| DCLM pretraining source | 2,000 | 1,780,498 | Largest extracted recipe source; sampled from `mlfoundations/dclm-baseline-1.0`, config `default`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 German pretraining source | 2,000 | 986,326 | Next largest unsampled web source; sampled from `HuggingFaceFW/fineweb-2`, config `deu_Latn`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 Spanish pretraining source | 2,000 | 1,271,133 | Next FineWeb2 web source; sampled from `HuggingFaceFW/fineweb-2`, config `spa_Latn`, split `train`, hash reservoir over 20,000 scanned rows. |
| PES2O pretraining source | 2,000 | 399,454 | Academic/scientific source; sampled from `allenai/dolmino-mix-1124`, config `pes2o`, split `train`, hash reservoir over 20,000 scanned rows. |
| StackExchange Apple pretraining source | 2,000 | 1,013,267 | Q&A/forum-like source from the SmolLM3 pretraining collection, using `ThreadText`. |
| SmolTalk2 Mid Llama-Nemotron reasoning | 2,000 | 6,041,904 | Bounded Mid config sample; reasoning-heavy, not no_think everyday data. |
| SmolTalk2 SFT everyday no_think | 2,260 | 188,621 | Full retained split scan for `smoltalk_smollm3_everyday_conversations_no_think`. |
| SmolTalk2 Preference Tulu no_think | 20,000 | 4,469,915 | 10,000 chosen/rejected pairs from `llama_3.1_tulu_3_8b_preference_mixture_no_think`. |

Published config source weights:

- Added `slop-extract-smollm3-config-weights`, which parses published
  SmolLM3 Nanotron YAML configs, derives segment spans from
  `start_training_step`, computes tokens per step from the config batch
  geometry, normalizes each segment's `dataset_weights`, and writes detailed
  source rows plus aggregate source/group weights.
- Ran it on `HuggingFaceTB/smollm3-configs` files `stage3_9T_11T.yaml`,
  `long_context_4k_to_32k.yaml`, and `long_context_32k_to_64.yaml`.
  The output contains 256 detail rows, 115 aggregate source rows, 8 heuristic
  source-group rows, and `11.234968T` config-implied tokens including the
  two long-context extension stages. The 4k pretraining portion comes from
  `stage3_9T_11T.yaml` and accounts for `11.135877T` tokens; the two
  long-context stages add about `99.091B`.
- Top aggregate recipe sources are `dclm` (`35.486%`), `fineweb-edu`
  (`31.140%`), `fw2-deu` (`2.209%`), `fw2-spa` (`2.003%`),
  `stack-edu-Python` (`1.811%`), and `pes2o` (`1.724%`). Heuristic group
  shares are web `78.607%`, code `14.046%`, math `4.935%`,
  academic/synthetic web `1.770%`, Q&A/forum `0.333%`, wiki `0.200%`,
  reasoning `0.105%`, and other `0.004%`.
- The currently sampled Phase 3 pretraining sources cover `dclm` (`35.486%`),
  `fineweb-edu` (`31.140%`), `fw2-deu` (`2.209%`), `fw2-spa` (`2.003%`), and
  `pes2o` (`1.724%`), plus `stackexchange` (`0.333%`) of the extracted
  recipe. This is enough for a majority-coverage weighted proxy, but not
  enough to compute a production weighted feature-rate baseline across all
  recipe sources.
- `stack-edu-Python` (`1.811%`) is the largest remaining recipe source by
  weight, but `HuggingFaceTB/stack-edu` config `Python` exposes blob metadata
  rather than code text in the HF dataset schema. Measuring it needs a
  separate blob-hydration path or another text-bearing mirror.

Coverage-aware weighted baseline proxy:

- Added `slop-assemble-weighted-pretrain-baseline`, which joins sampled
  feature rates to exact recipe source weights through explicit source maps
  and reports both a covered-only weighted rate and a missing-as-zero
  lower-bound diagnostic.
- Ran it with source maps
  `smollm3_pretrain_fineweb_edu_2k=fineweb-edu` and
  `smollm3_pretrain_stackexchange_apple_2k=stackexchange`, regenerated it with
  `smollm3_pretrain_dclm_2k=dclm`, then added
  `smollm3_pretrain_fw2_deu_2k=fw2-deu` and
  `smollm3_pretrain_fw2_spa_2k=fw2-spa`, and
  `smollm3_pretrain_pes2o_2k=pes2o`. The current retained Tier-1 proxy covers
  `72.895%` of the extracted recipe and leaves `27.105%` unsampled.
- Covered-only weighted rates per 1k tokens are: `slop_lexicon` `0.422`,
  `rule_of_three_approx` `4.621`, `contrastive_negation` `0.410`,
  `stock_closers` `0.037`, `stock_openers` `0.029`, and pooled stock phrases
  `0.067`. These are coverage-normalized diagnostics, not full-mixture
  estimates, because the missing recipe share is still large.
- `list_header_bold_lead_in` and `punctuation_rhythm` remain at `31.473%`
  coverage in the proxy because the DCLM and FineWeb2 language censuses were
  run only for the retained Tier-1 features.

Coverage proxy artifacts:

- `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
- `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy_summary.md`

Config-weight artifacts:

- `artifacts/phase3/analysis/smollm3_config_source_weights_detail.csv`
- `artifacts/phase3/analysis/smollm3_config_source_weights_aggregate.csv`
- `artifacts/phase3/analysis/smollm3_config_source_weights_groups.csv`
- `artifacts/phase3/analysis/smollm3_config_source_weights_summary.md`

Data-rate and preference artifacts:

- `artifacts/stage1/corpora/smollm3_pretrain_fineweb_edu_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fineweb_edu_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_dclm_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_dclm_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_deu_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_deu_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_spa_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_spa_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_pes2o_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_pes2o_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_stackexchange_apple_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_stackexchange_apple_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_smoltalk2_mid_llama_nemotron_reasoning_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_smoltalk2_mid_llama_nemotron_reasoning_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_smoltalk2_sft_everyday_no_think_2260.jsonl`
- `artifacts/stage1/corpora/smollm3_smoltalk2_sft_everyday_no_think_2260_summary.md`
- `artifacts/stage1/corpora/smollm3_smoltalk2_preference_tulu_no_think_10k_pairs.jsonl`
- `artifacts/stage1/corpora/smollm3_smoltalk2_preference_tulu_no_think_10k_pairs_summary.md`
- `artifacts/stage1/census/smollm3_pretrain_mid_baselines_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_dclm_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_fw2_deu_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_fw2_spa_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_pes2o_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_smoltalk2_sft2260_pref10k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_smoltalk2_pref10k_tier1_pair_deltas.csv`
- `artifacts/stage1/census/smollm3_smoltalk2_pref10k_tier1_pair_analysis.csv`

Tier-1 rates, per 1,000 simple tokens:

| Feature | Weighted proxy covered-only | FineWeb-Edu | DCLM | FineWeb2 German | FineWeb2 Spanish | PES2O | StackExchange | Mid | SFT target | Chosen | Rejected | Chosen - Rejected | Pair BH q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `contrastive_negation` | 0.410 | 0.385 | 0.487 | 0.006 | 0.002 | 0.238 | 0.509 | 0.177 | 0.095 | 0.379 | 0.352 | +0.027 | 0.039 |
| `rule_of_three_approx` | 4.621 | 6.211 | 3.746 | 0.184 | 0.038 | 5.520 | 1.583 | 1.273 | 17.045 | 5.443 | 4.877 | +0.566 | 9.437e-13 |
| `slop_lexicon` | 0.422 | 0.536 | 0.354 | 0.047 | 0.005 | 0.716 | 0.327 | 0.050 | 0.419 | 1.609 | 1.293 | +0.316 | 3.367e-15 |
| `stock_closers` | 0.037 | 0.042 | 0.037 | 0.002 | 0.000 | 0.060 | 0.026 | 0.027 | 0.270 | 0.217 | 0.167 | +0.050 | 3.016e-04 |
| `stock_openers` | 0.029 | 0.027 | 0.037 | 0.000 | 0.000 | 0.000 | 0.069 | 0.000 | 0.106 | 0.571 | 0.334 | +0.237 | 0.161 |
| `stock_openers_closers` | 0.067 | 0.068 | 0.074 | 0.002 | 0.000 | 0.060 | 0.095 | 0.027 | 0.376 | 0.788 | 0.502 | +0.287 | 0.003 |

Reassembled data-rate spectrum outputs:

- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_data_rates_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_data_rates_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_data_rates_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_data_rates_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_data_rates_tf_generation_compounding_slop_neutral_rule3_aligned.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_data_rates_tf_generation_compounding_slop_neutral_rule3_correlations.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_data_rates_tf_generation_compounding_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_summary.md`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_baselines_data_rates_tf_generation_compounding_slop_neutral_rule3_aligned.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_baselines_data_rates_tf_generation_compounding_slop_neutral_rule3_correlations.csv`
- `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_baselines_data_rates_tf_generation_compounding_slop_neutral_rule3_summary.md`

The data-rate classifier still labels `slop_lexicon` as `sft-amplified`, not
preference-amplified. The no_think Tulu preference data are complicit for
`slop_lexicon` under paired BH-FDR (`chosen > rejected`, `q=3.367e-15`), but
the model-side AF is already far above the amplified threshold at SFT and the
APO/final relative increase over SFT is modest under the current rule. This is
the main contrast with the bounded OLMo result, where `slop_lexicon` is
classified as preference-amplified and dynamics-driven.

`rule_of_three_approx` is also chosen-greater-rejected in the SmolTalk2
preference sample, but the retained teacher-forced comma-pair proxy remains
below 1 at every SmolLM3 stage. Stock phrases and `contrastive_negation` have
data-rate and free-running evidence, but their SmolLM3 teacher-forced
opportunity support is currently missing or zero-reference in the 512-prompt
denominator audit, so they remain `observed-output-only`.

Caveat: the normalized SmolTalk2 sample carries the no_think split/source
identity, but per-row `preference_type`, `chosen_model`, and `rejected_model`
fields are not exposed by this sample path and appear as `None` in the paired
analysis table. The source split identifies the candidate Tulu no_think
preference mixture, but the current SmolLM3 paired analysis is not yet
stratified by per-row preference-construction metadata.

Second caveat: the current SmolLM3 feature-rate pretrain aggregate is
token-weighted over the two bounded source samples above. It is useful as a
pretraining-source baseline for Phase 3 classification, but it is not the
full SmolLM3 pretraining mixture. The exact published recipe weights are now
resolved, so the remaining baseline gap is feature-rate coverage for the
additional recipe sources, not source-weight discovery.

## Current Interpretation

The bounded OLMo Phase 3 classification strengthens the final Phase 2
conclusion:

- The only retained feature view classified as preference-amplified is
  `slop_lexicon`.
- That preference-stage amplification is labeled dynamics-driven under the
  current rule because paired Result A evidence for `slop_lexicon` is
  nonsignificant and rejected-greater-chosen.
- Pooled stock phrase behavior and stock closers are better described as
  already-amplified/high-AF features than as preference-stage jumps.
- `rule_of_three_approx` still does not support a DPO amplification claim.

The bounded Phase 3 result is therefore not "DPO makes everything sloppier."
It is: DPO is the clearest stage-localized propensity peak for the retained
`slop_lexicon` feature, while most other retained feature views are inherited,
already high, output-only, or unsupported by the teacher-forced proxy.

The bounded SmolLM3 slice gives a different but compatible cross-ladder read:
`slop_lexicon` is already strongly amplified by SFT under the SmolLM3
teacher-forced normalization and continues rising modestly through APO/final;
`rule_of_three_approx` remains below reference rate in the comma-pair
teacher-forced proxy but is much more visible in sampled SFT/APO/final output
than in base. The shared AF ordering across the two measured features is
similar enough to yield a bounded OLMo-vs-SmolLM3 Spearman AF of `0.762`, but
the support is still only two teacher-forced feature views.

With the SmolTalk2 no_think data-rate layer added, SmolLM3 `slop_lexicon`
looks unlike the bounded OLMo `slop_lexicon` preference story. In OLMo/Dolci,
paired DPO data do not show chosen-greater-rejected `slop_lexicon`, so the
preference-stage model peak is labeled dynamics-driven. In SmolLM3/SmolTalk2,
the no_think Tulu preference sample does show chosen-greater-rejected
`slop_lexicon` (`q=3.367e-15`), but the model-side classification remains
`sft-amplified` because SFT already has very high normalized AF and APO/final
only add a modest relative increase. The bounded cross-ladder conclusion is
therefore not "preference tuning always creates the same slop path"; it is
that comparable surface markers can enter through different measured
stage/data combinations across ladders.

## Next Work Toward Full Phase 3

The next concrete work needed to complete Phase 3 from `EXPERIMENTS.md` is:

1. Decide whether full Phase 3 completion should use the bounded artifact
   shape or launch the original full production shape from `EXPERIMENTS.md`.
   The bounded OLMo and SmolLM3 slices are now generation-inclusive; the
   original production shape is still substantially larger.
2. Add broader SmolLM3 teacher-forced feature coverage only if new opportunity
   contracts or prompt subsets provide denominator support; the current
   512-prompt denominator audit has zero references for stock phrases and
   contrastive negation.
3. Expand SmolLM3 pretraining feature-rate coverage beyond the current
   `72.895%` retained Tier-1 recipe share covered by DCLM, FineWeb-Edu,
   FineWeb2 German, FineWeb2 Spanish, PES2O, and StackExchange.
   `stack-edu-Python` (`1.811%`) is the largest unresolved source but needs
   code-text hydration because the HF dataset exposes blob metadata only. The
   next directly sampleable high-impact source is `fw2-fra` (`1.607%`),
   followed by `finemath` (`1.410%`) and `stack-edu-Cpp` (`1.304%`, same
   blob-hydration caveat as Python).
4. If production completion is required, run the original 5,000-prompt x
   8-completion x 3-temperature generation grids for the retained ladders and
   rebuild the compounding and cross-ladder artifacts at that scope.
5. Summarize DPO-vs-APO variation across the shared feature set in writeup
   form, clearly separating bounded measured evidence from the still-missing
   full production grid.
