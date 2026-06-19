# Phase 3 Completion Audit

Date: 2026-06-17

This audit checks the Phase 3 requirements in `EXPERIMENTS.md` against the
current repository state and the production-scope decision made after the
full-grid SGLang trial. Bounded production evidence is treated as the active
Phase 3 completion target; the original 5,000-prompt x 8-completion x
3-temperature OLMo grid is retained as an optional addendum, not a blocker.

Operational runbook:
`docs/experiments/phase3_production_runbook.md`.

## Scope Decision

Phase 3 requires the main amplification spectrum:

> features x Phase 1 data rates x Phase 2 teacher-forced AF/free-running rates,
> with feature classifications, statistics, full SmolLM3 no_think replication,
> and cross-ladder AF rank correlation.

The current worktree contains the completed production-scope Phase 3 evidence:
reduced OLMo SGLang panels, SmolLM3 no_think production-shape replication,
style-signature layers, full-pybiber Phase 1 corpus-side register evidence,
cross-ladder comparison, and the Think/RL-Zero stretch panel. The remaining
blocker is no longer pretraining source coverage. The current operational
decision is to use SGLang for pure generation tasks when new generation is
needed. The original OLMo
5,000-prompt x 8-completion x 3-temperature grid has a resumable SGLang plan
and partial caches, but it is an optional addendum rather than the basis for
the production conclusion.
Teacher-forced probability scoring remains on the scorer path and is kept
bounded by measured denominator support rather than by broad prompt-count caps.

## Requirement Status

| Requirement from `EXPERIMENTS.md` | Current evidence | Status | Remaining work |
|---|---|---|---|
| Assemble amplification spectrum over features, Phase 1 data rates, teacher-forced AF, and free-running rates. | Reduced OLMo integrated spectrum: `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv`. SmolLM3 production-shape bounded spectrum: `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_production.csv`. SmolLM3 sparse teacher-forced addendum: `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_2258prompt_sparse_supported.csv`. Both main spectra now carry free-running per-1k and compounding document/prompt-cluster bootstrap CI columns from the cached generation JSONL. Reduced OLMo generation panels are complete and assembled as main, temperature, and long-output grids. Full OLMo SGLang plan: `artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.csv`. | Complete for production scope. | Current spectra cover retained Tier-1 features and measured AF cells. SmolLM3 512-prompt production-shape generation, reduced OLMo SGLang generation, and the sparse SmolLM3 teacher-forced addendum are complete. The original full OLMo grid is paused as an optional addendum; it is not needed for the production conclusion. |
| Classify features as inherited, SFT-amplified, preference-amplified, compounding-dominant. | Reduced OLMo classification: `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_feature_classification_t07_long1024.csv`. SmolLM3 production classification: `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_production.csv`. SmolLM3 sparse teacher-forced addendum classification: `artifacts/phase3/analysis/smollm3_no_think_feature_classification_2258prompt_sparse_supported.csv`. | Complete for production scope. | Classification is rule-based over measured cells. The sparse addendum adds teacher-forced AF cells for `contrastive_negation`, `stock_closers`, and `stock_openers_closers`, but reference counts are only 2/3/3, so these are coverage-caveated addendum claims rather than a replacement for the main production spectrum. |
| Assign preference-amplified cause as signal-driven vs. dynamics-driven using chosen-vs-rejected evidence. | OLMo uses `artifacts/stage1/census/olmo3_dolci_dpo_10k_pair_analysis.csv`; SmolLM3 uses `artifacts/stage1/census/smollm3_smoltalk2_pref10k_tier1_pair_analysis.csv`. | Complete for measured Tier-1 features. | Extend only if new features enter the Phase 3 spectrum. |
| Include cluster-bootstrap CIs for all CIs. | Teacher-forced AF artifacts include AF confidence columns where available; classification records whether CI excludes 1. Reduced OLMo long-output and SmolLM3 production `t=1.0` generation grids include document/prompt-cluster bootstrap CIs for generated `per_1k_tokens`. Reduced OLMo long-output, reduced OLMo temperature, SmolLM3 production `t=1.0`, and supplemental SmolLM3 three-temperature `slop_lexicon` compounding tables now include bootstrap CIs for observed/excess per-1k opportunities, realized AF, and prior/no-prior risk metrics; current spectra propagate the primary compounding CI columns. Archived generated-output register-proxy comparison tables include generated-side document/prompt-cluster bootstrap CIs, and style signatures preserve them as `value_ci_low/high` where available. | Complete for production-scope primary statistics. | Corpus-side full-pybiber baselines have selected-feature bootstrap intervals, sparse addendum CIs are coverage-caveated, and auxiliary-feature cells in the large SmolLM3 three-temperature compounding table remain point estimates unless rerun through the cached-counter bootstrap path. |
| Apply Benjamini-Hochberg FDR across the full feature set. | Pair analyses and stage-effect artifacts provide BH-FDR columns for measured Tier-1 comparisons; classifier joins those q-values. | Complete for measured Tier-1 comparisons. | FDR is present for supplied pair/stage-effect artifacts and is not claimed for unmeasured features. |
| Replicate the full spectrum on SmolLM3-3B in no_think mode. | SmolLM3 no_think completed all 12 production-shape 512-prompt shards, 49,152 generated records, production `t=1.0` generation grid, teacher-forced `slop_lexicon`/neutral/rule-of-three grids, compounding, data-rate layer, archived register-proxy style signature, spectrum, classifier, and cross-ladder comparison. A full eligible 2,258-prompt SmolTalk2 no_think package was also materialized for sparse teacher-forced coverage, and all four base/SFT/APO/final sparse AF shards completed for `contrastive_negation`, `stock_closers`, and `stock_openers_closers`. | Complete for the available 512-prompt SmolLM3 production-shape scope, with a sparse 2,258-prompt teacher-forced addendum. | `stock_openers` remains zero-reference unsupported in the 2,258-prompt target package; sparse addendum CIs are wide because held-out references are only 2 or 3. |
| Report feature-level rank correlation of AFs between ladders. | `artifacts/phase3/analysis/olmo3_reduced_sglang_vs_smollm3_no_think_production_summary.md` reports Spearman `0.762`, Pearson `0.978`, with 8 shared AF values. | Complete for bounded measured AF overlap. | Broader correlation awaits broader shared teacher-forced feature coverage. |
| Decompose propensity shift vs. self-conditioning compounding. | Reduced OLMo compounding with bootstrap CIs: `artifacts/phase3/analysis/olmo3_phase3_reduced_long_compounding_analysis_256prompt_2comp_t07_1024_sglang.csv`; primary realized-AF plot: `artifacts/phase3/analysis/olmo3_phase3_reduced_long_compounding_analysis_256prompt_2comp_t07_1024_sglang_realized_af.svg`. Reduced OLMo temperature compounding with bootstrap CIs: `artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_compounding_analysis_512prompt_1comp_3temp_512_sglang.csv`. SmolLM3 production `t=1.0` compounding with bootstrap CIs: `artifacts/phase3/analysis/smollm3_no_think_generation_compounding_512prompt_8comp_t1_chat_production_tf_slop_neutral_rule3.csv`; SmolLM3 temperature compounding point table: `artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_analysis_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3.csv`; supplemental SmolLM3 temperature `slop_lexicon` CI table: `artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_slop_lexicon_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3.csv`. | Complete for supported bounded OLMo and SmolLM3 primary features. | The sparse 2,258-prompt teacher-forced addendum does not add matching compounding expectation joins; auxiliary-feature SmolLM3 temperature-table CIs are deferred, and the primary `slop_lexicon` curve now uses a cached-counter bootstrap. |
| Report temperature dependence of realized AF. | Reduced OLMo temperature realized-AF table and SVG: `artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_compounding_analysis_512prompt_1comp_3temp_512_sglang.csv` and `artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_compounding_analysis_512prompt_1comp_3temp_512_sglang_realized_af.svg`. SmolLM3 no_think temperature realized-AF table and SVG: `artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_analysis_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3.csv` and `artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_analysis_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3_realized_af.svg`. | Complete for bounded OLMo and SmolLM3 supported features. | Extend only if new teacher-forced-supported features enter the spectrum. |
| Use SmolLM3 no_think mode and verify mode compliance. | SmolLM3 generation planning and executed caches use chat-template kwargs `{"enable_thinking": false}` where applicable. Completed production-grid audits through final/RLVR t1 report zero explicit thinking-marker hits. | Complete for completed SmolLM3 generated slices. | Re-run only if new SmolLM3 generation caches are added. |
| Include Biber/linguistic feature layer. | Phase 1 has full 67-feature corpus-side pybiber outputs, token-weighted means/deltas, and selected-feature bootstrap intervals: `artifacts/stage1/census/phase1_pybiber_register_means.csv`, `artifacts/stage1/census/phase1_pybiber_register_deltas.csv`, and `artifacts/stage1/census/phase1_pybiber_register_intervals.csv`. Earlier generated-output register-proxy artifacts exist but are archived diagnostics rather than the active Tier-2 surface. | Complete for the active paper scope as full-pybiber Phase 1 corpus-side register analysis. | Generated-output full pybiber was not run and is not claimed. Older generated-output proxy artifacts should not be promoted as full Biber/register evidence. |
| Stretch: Instruct vs. Think vs. RL Zero path comparison. | Reduced SGLang plan completed: `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_256prompt_2comp_t07_1024_sglang.csv`, with summary `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_256prompt_2comp_t07_1024_sglang.md`. It covers base, Instruct final, Think SFT/DPO/final, and RL-Zero General/IF/Math/Code/Mix: 10/10 shards, 5,120/5,120 records. All JSONL rows use `feature_text_mode=final_answer`. CI-enriched generation grid: `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_generation_grid_256prompt_2comp_t07_1024_sglang.csv`. Path comparison: `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_comparison_256prompt_2comp_t07_1024_sglang.csv`. For `slop_lexicon`, Instruct final is the maximum at `0.807` per 1k final-answer tokens; Think final is lower by `0.384`, and the RL-Zero mean is lower by `0.244`. | Complete for reduced stretch scope. | This is output-side evidence only and does not add teacher-forced AF cells. `RL-Zero-Mix` required a local Hugging Face cache config compatibility patch from `olmo2-retrofit` to `olmo3` for SGLang/Transformers 4.57.6 loading. |

## Reduced OLMo Generation Status

The failed Torch full OLMo production grid is no longer the active operational
target. The original full-grid shape now has a separate SGLang execution plan,
but it is not yet complete and is not the basis for the bounded conclusion.
The reduced SGLang panels remain the completed evidence used by current
reports.

Active reduced SGLang plans:

- Main paired style signature:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_main_signature_1024prompt_2comp_t07_512_sglang.csv`
- Temperature sensitivity:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_512prompt_1comp_3temp_512_sglang.csv`
- Long-output compounding:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_long_compounding_256prompt_2comp_t07_1024_sglang.csv`
- Combined status:
  `artifacts/phase3/analysis/phase3_reduced_sglang_generation_plan_status_summary.md`

These plans total 16,384 planned OLMo generations and use SGLang with
`--ignore-eos` for fixed-budget pure generation.

Current reduced OLMo status:

- Main paired style-signature panel: 4/4 shards, 8,192/8,192 generations.
- Temperature-sensitivity panel: 12/12 shards, 6,144/6,144 generations.
- Long-output compounding panel: 4/4 shards, 2,048/2,048 generations.
- Integrated reduced spectrum:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv`
- Integrated reduced classifier:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_feature_classification_t07_long1024.csv`
- Reduced OLMo-vs-SmolLM3 comparison:
  `artifacts/phase3/analysis/olmo3_reduced_sglang_vs_smollm3_no_think_production_summary.md`
- Combined status:
  `artifacts/phase3/analysis/phase3_reduced_sglang_generation_plan_status_summary.md`

## Legacy Full Grid Status

The original Torch full OLMo production generation attempts are legacy. The
effective Torch continuation plan preserved the original `batched64` output
paths but used `generation-batch-size=32`, `--resume`, and no `torch.compile`.
The original `batched1024` launch, compiled `batched256` launch, no-compile
`batched128` launch, and no-compile batch-64 command shape all failed or
proved too large for the A100 memory budget.

The active full-grid recovery path is now SGLang:

- Full SGLang plan:
  `artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.csv`
- Full SGLang plan summary:
  `artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.md`
- First launched chunked shard selection:
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_generation_shard_base_t0p0.json`
- First launched chunked shard log:
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_generation_shard_base_t0p0.log`

The full SGLang plan keeps the original 12 shard shape: 4 stages x 3
temperatures x 40,000 generations each. Commands use the SGLang sidecar,
`--ignore-eos`, and `--prompt-batch-size 128` so each long shard can write
incremental JSONL chunks and later resume partial output.

- Current effective continuation plan:
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched64_outputs_batched32_resume.csv`
- Current effective continuation plan summary:
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched64_outputs_batched32_resume.md`
- Failed/too-large no-compile batch-64 plan retained for audit and
  output-path provenance:
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched64_no_compile.csv`
- Failed/too-large no-compile batch-64 plan summary:
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched64_no_compile.md`
- Failed no-compile lower-batch plan:
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched128_no_compile.csv`
- Failed compiled lower-batch plan:
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched256.csv`
- Original audit plan:
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp.csv`

Machine-readable status summary:

- `artifacts/phase3/analysis/phase3_generation_plan_status_summary.csv`
- `artifacts/phase3/analysis/phase3_generation_plan_status_summary.md`

It planned 12 shards: 4 stages x 3 temperatures x 40,000 generations each.
The resumed OLMo `base`/`t=0.0` shard was stopped at `1472/40000` rows after
the compute strategy changed. The partial output is retained here:
`artifacts/phase2/generations/olmo3_base_promptpkg5000_free_run_5000prompt_8comp_t0_batched64.jsonl`.

The SmolLM3 production-shape plan for the available no_think prompt package
is complete:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`
- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.md`

It plans 12 shards: 4 stages x 3 temperatures x 4,096 generations each. This
is smaller than the original 5,000-prompt shape because the current SmolLM3
no_think prompt package has 512 held-out prompts.

A next-shard selection has been recorded here:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_next_shard.json`

All 12 SmolLM3 shards completed on 2026-06-16 with 4,096 expected generations
and a summary file each, for 49,152 total generations.

Completed base/t0 record:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_next_shard.json`
- Log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_next_shard.log`
- Generation output:
  `artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_base_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t0_batched256.jsonl`
- Summary output:
  `artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_base_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t0_batched256_summary.csv`

Completed base/t0.7 record:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t07_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t07_shard_status.csv`
- Log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t07_shard.log`

Completed base/t1 detached launch record:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t1_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t1_shard_status.csv`
- Log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t1_shard.log`

Completed SFT/t0 detached launch record:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t0_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t0_shard_status.csv`
- Log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t0_shard.log`

Completed SFT/t0.7 detached launch record:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t07_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t07_shard_status.csv`
- Log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t07_shard.log`

Completed SFT/t1 detached launch record:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t1_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t1_shard_status.csv`
- Log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t1_shard.log`

Completed APO/DPO t0 detached launch record:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t0_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t0_shard_status.csv`
- Log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t0_shard.log`

Completed APO/DPO t0.7 detached launch record:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t07_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t07_shard_status.csv`
- Log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t07_shard.log`

Completed APO/DPO t1 detached launch record:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t1_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t1_shard_status.csv`
- Log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t1_shard.log`

Completed final/RLVR t0 detached launch record:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t0_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t0_shard_status.csv`
- Log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t0_shard.log`

Completed final/RLVR t0.7 detached launch record:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t07_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t07_shard_status.csv`
- Log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t07_shard.log`

Completed final/RLVR t1 detached launch record:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t1_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t1_shard_status.csv`
- Log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t1_shard.log`

To relaunch that shard intentionally, use the guarded launcher with explicit
execution:

```bash
uv run slop-launch-phase2-generation-shard \
  --plan artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv \
  --stage base \
  --temperature 0.0 \
  --max-estimated-a100-hours 4 \
  --selection-output artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_next_shard.json \
  --execute \
  --detach
```

After a detached launch, inspect it with:

```bash
uv run slop-phase2-generation-status \
  --selection artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_next_shard.json
```

## Teacher-Forced Support Status

Current SmolLM3 support audit:

- `artifacts/phase3/analysis/smollm3_no_think_phase3_feature_support_audit_512prompt_and_full_sft.csv`
- `artifacts/phase3/analysis/smollm3_no_think_phase3_feature_support_audit_512prompt_and_full_sft.md`

On the actual 512-prompt SmolLM3 package:

- `slop_lexicon`: teacher-forced supported.
- `rule_of_three_approx`: teacher-forced supported through the comma-pair
  extension proxy.
- `neutral_controls` / `neutral_common_controls`: teacher-forced supported.
- `contrastive_negation`: raw target occurrences exist, but zero aligned
  reference initiations under the current 512-prompt opportunity contract.
- `stock_openers`, `stock_closers`, `stock_openers_closers`: raw target
  occurrences exist, but zero aligned reference initiations under the current
  512-prompt opportunity contract.

These zero-reference features should remain generation-side evidence unless a
new opportunity contract or prompt package creates denominator support.

## Current Bounded Conclusions

- OLMo/Dolci `slop_lexicon` is the cleanest preference-amplified measured
  feature and is labeled dynamics-driven: DPO-stage AF rises even though the
  paired preference data do not show chosen-greater-rejected `slop_lexicon`.
- SmolLM3/SmolTalk2 `slop_lexicon` is classified as SFT-amplified under the
  bounded no_think slice. The no_think preference sample is complicit, but
  SFT already has high normalized AF and APO/final add a smaller relative
  increase.
- `rule_of_three_approx` is visible in generated output but stays below the
  teacher-forced reference rate in both bounded ladders.
- Archived register-proxy style signatures show a broad output-style
  progression. For SmolLM3, Final/RLVR is closest to APO/DPO, then SFT, and far
  from Base under the current 43-coordinate output-style signature. These are
  not active full-pybiber generated-output claims.

## Completion Verdict

Phase 3 is not complete against the full `EXPERIMENTS.md` scope.

It is complete only as a bounded Tier-1 Phase 3 slice with:

- OLMo bounded spectrum and classification.
- SmolLM3 no_think bounded spectrum and classification.
- paired preference-data complicity analysis for both ladders.
- bounded compounding decomposition.
- bounded cross-ladder AF rank correlation.
- archived register-proxy output-style signatures for both ladders.
- bounded DPO-vs-APO variation report:
  `docs/experiments/phase3_dpo_vs_apo_variation_report.md`.
- reduced OLMo Instruct/Think/RL-Zero stretch comparison:
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_comparison_256prompt_2comp_t07_1024_sglang_summary.md`.

The next work that actually moves Phase 3 toward full completion is:

1. Add broader teacher-forced support only where the opportunity contract has
   nonzero references and the initiation contract is defensible.
2. Recompute the spectrum, classifications, FDR joins, and cross-ladder
   correlations at any expanded teacher-forced scope.
3. Propagate any expanded feature set through the Biber/style-signature layer
   only if those features are promoted into the main Phase 3 spectrum.

The remaining pretraining source coverage gap should not be prioritized unless
the study scope changes again.
