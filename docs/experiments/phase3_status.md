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
no_think calibration ladder, and the first full-size SmolLM3 512-prompt
teacher-forced shard for `slop_lexicon`, `neutral_common_controls`, and the
`rule_of_three_approx` comma-pair proxy. It is real Phase 3 progress, but it
is not the full EXPERIMENTS.md Phase 3 completion because the full SmolLM3
free-running ladder, compounding layer, and original production grid are not
present.

## Implemented Phase 3 Layer

Added CLI:

- `slop-classify-amplification-spectrum`
- `slop-compare-phase3-ladders`
- `slop-plan-phase2-propensity`
- `slop-analyze-phase3-free-run-effects`
- `slop-analyze-phase3-teacher-forced-effects`

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
blank. That verifies the real cross-ladder command path but is not a
statistically interpretable replication result. The first full-size
SmolLM3 teacher-forced comparisons now include a slop-lexicon-only comparison
and a two-feature comparison with `slop_lexicon` plus the
`rule_of_three_approx` comma-pair proxy. The two-feature comparison aligns 8
feature-stage AF rows and reports overall Spearman AF `0.762` and Pearson AF
`0.978`; this is interpretable for the measured teacher-forced slice, but it
still lacks free-running and compounding evidence.

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
| Replicate full spectrum on SmolLM3-3B no_think ladder | A 512-row no_think SmolTalk2 prompt package, canonical chat-template generation plan, propensity plan, all-stage 4-prompt generation calibration, all-stage 4-prompt slop/neutral propensity calibration, a schema-compatible 4-prompt SmolLM3 calibration spectrum, and a full 512-prompt four-stage SmolLM3 teacher-forced spectrum for `slop_lexicon`, `neutral_common_controls`, and `rule_of_three_approx` now exist. Full 512-prompt SmolLM3 free-running summaries and compounding are still missing. | Teacher-forced slop/neutral/rule3 done; full spectrum incomplete |
| Report cross-ladder AF rank correlation | `slop-compare-phase3-ladders` now implements the statistic, passes an OLMo self-check, aligns OLMo vs. the SmolLM3 4-prompt calibration spectrum, reports a 512-prompt teacher-forced slop-lexicon-only comparison with Spearman AF `0.400`, and reports a two-feature teacher-forced comparison with Spearman AF `0.762` over eight shared stage rows. Full production correlation remains missing until SmolLM3 has free-running/compounding layers and the final shared feature scope is locked. | Partial interpretable TF comparison done; full production correlation missing |
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

This removes a harness blocker for SmolLM3 Phase 2/3 replication. The prompt
package and tiny calibration ladder now exist, but the full 512-prompt
teacher-forced summaries, free-running summaries, and interpretable assembled
amplification spectrum are still missing.

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
AF `0.978`. This is interpretable for the current teacher-forced slice, but it
does not satisfy the full Phase 3 requirement for a production cross-ladder
comparison because SmolLM3 free-running output and compounding are still
missing.

The generated SmolLM3 classification table labels `slop_lexicon` as
`sft-amplified`, because AF is already far above the default amplified
threshold at the SFT stage and the APO/final relative increase over SFT is
small. It labels `rule_of_three_approx` as `measured-no-phase3-class`, because
the raw AF remains below 1 across all stages. These classifications are
provisional because SmolLM3 corpus rates, free-running emissions, and
compounding measurements are not yet supplied to the classifier.

## Current Interpretation

The bounded Phase 3 classification strengthens the final Phase 2 conclusion:

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

## Next Work Toward Full Phase 3

The next concrete work needed to complete Phase 3 from `EXPERIMENTS.md` is:

1. Decide whether full Phase 3 completion should use the bounded OLMo artifact
   shape or launch the original full production shape.
2. Run the SmolLM3 no_think free-running generation ladder from the generated
   512-prompt chat-template plan. The teacher-forced slop/neutral/rule3 ladder
   is complete, but sampled-output rates and compounding are still missing.
3. Add broader SmolLM3 teacher-forced feature coverage only if new opportunity
   contracts or prompt subsets provide denominator support; the current
   512-prompt denominator audit has zero references for stock phrases and
   contrastive negation.
4. Assemble the full SmolLM3 amplification spectrum with the same schema used
   by the OLMo bounded spectrum, including free-running and compounding
   columns where available.
5. Run `slop-compare-phase3-ladders` on OLMo vs. the completed SmolLM3
   spectrum and summarize DPO-vs-APO variation across the shared feature set.
