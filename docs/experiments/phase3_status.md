# Phase 3 Status

Date: 2026-06-16

Supersession note, 2026-06-18: the active paper-facing Tier-2 register surface
is corpus-side full pybiber from Phase 1. Generated-output Biber-lite/register
proxy artifacts in this status file are historical diagnostics only and should
not be read as generated-output full pybiber evidence.

Authoritative completion audit:
`docs/experiments/phase3_completion_audit.md`.
Operational production runbook:
`docs/experiments/phase3_production_runbook.md`.
Machine-readable completion snapshot:
`artifacts/phase3/analysis/phase3_completion_audit_snapshot.md`.
Current integrated conclusion report:
`docs/experiments/phase3_integrated_conclusion_report.md`.
Current DPO-vs-APO variation report:
`docs/experiments/phase3_dpo_vs_apo_variation_report.md`.

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
bounded SmolTalk2 no_think SFT/preference data-rate layer, a bounded
source-stratified pretraining/Mid baseline, archived generated-output
register-proxy diagnostics, and a combined output-style signature. It is real
Phase 3 progress, but it is not the full EXPERIMENTS.md Phase 3 completion
because the original 5,000-prompt x 8-completion x 3-temperature production
grid and full production-scope teacher-forced coverage are not present. The
exact SmolLM3 recipe source weights have now been
extracted from the published configs. Retained Tier-1 direct feature-rate
coverage reaches `91.969%` of the extracted recipe through 47 mapped source
samples; the current weighted baseline also uses 22 explicit same-family
source proxies covering another `4.601%`, for `96.571%` total covered recipe
share and `3.429%` still unsampled. That remaining recipe-source gap is not
currently the active blocker. The 3-temperature SmolLM3 no_think
production-shape plan is now complete for the available 512-prompt package:
all 12 stage/temperature shards finished with 4,096/4,096 generations each
for 49,152 total generations. The production `t=1.0` generation grid,
free-running stage effects, compounding, archived register-proxy comparison,
style signature, amplification spectrum, classifier, and OLMo-vs-SmolLM3
cross-ladder comparison have been rebuilt from the completed caches. Phase 3
still remains incomplete against the full original scope because the original
full-grid parity target is not complete and sparse teacher-forced features
remain coverage-caveated. A 2,258-prompt SmolTalk2 no_think sparse
teacher-forced addendum now exists for `contrastive_negation`, `stock_closers`,
and `stock_openers_closers`; `stock_openers` still has zero held-out target
references in that package. The reduced OLMo SGLang generation plan is now
complete, including the reduced Instruct/Think/RL-Zero stretch comparison.

The Torch/Transformers OLMo 5,000-prompt x 8-completion x 3-temperature
execution path has been abandoned as the active route because it would take
weeks on the available A100 and repeatedly hit CUDA OOM. The original compiled
`batched1024` plan, compiled `batched256` plan, and no-compile `batched128`
plan all failed before writing usable rows. The no-compile `batched64` base/t0
attempt wrote 512 rows, then hit CUDA OOM during KV-cache growth. A resumed
batch-32 attempt reached `1472/40000` rows before it was deliberately stopped;
that partial JSONL is retained as pilot/cache data, not as an active
production shard. Original-scope OLMo parity work is now active only through
the resumable SGLang full-grid plan described below.

The completed bounded OLMo generation path is a reduced three-panel SGLang
plan: 8,192 main paired style-signature generations, 6,144
temperature-sensitivity generations, and 2,048 long-output compounding
generations, for 16,384 OLMo generations total. The plan files are:

- `artifacts/phase3/analysis/olmo3_phase3_reduced_main_signature_1024prompt_2comp_t07_512_sglang.csv`
- `artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_512prompt_1comp_3temp_512_sglang.csv`
- `artifacts/phase3/analysis/olmo3_phase3_reduced_long_compounding_256prompt_2comp_t07_1024_sglang.csv`
- `artifacts/phase3/analysis/phase3_reduced_sglang_generation_plan_status_summary.md`

SGLang is now the default backend for pure generation tasks. These plans use
the repo-local SGLang sidecar path and `--ignore-eos` fixed-budget generation
contract. Torch/Transformers is no longer the route for pure generation
recovery unless SGLang is blocked by a concrete loader/runtime failure; the
scorer path remains separate for teacher-forced probability measurements. The
SGLang harness now also has JSONL-cache resume support and
`--prompt-batch-size`, so long full-grid shards can checkpoint after each
prompt chunk without falling back to tiny Transformers batches. The sidecar has
now been created and smoke-tested. The working SGLang command shape uses local
`libnuma`, `/usr/lib/x86_64-linux-gnu` for `libcuda.so.1`, triton attention,
and disabled CUDA graph capture.

The current compute policy is therefore:

- use SGLang for pure free-running generation, including OLMo reduced panels
  optional path-comparison generations, and the original OLMo
  5,000-prompt full grid;
- use the resumable chunked SGLang full-grid plan for original-scope OLMo
  parity work rather than Torch batch-size recovery;
- keep teacher-forced probability scoring on the scorer path, but only run it
  where denominator audits show actual opportunity/reference support;
- treat sparse AF measurements with 2-3 held-out references as addendum
  evidence with wide uncertainty, not as a replacement for dense production
  support.

The original OLMo full-grid shape now has a resumable SGLang plan:

- `artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.csv`
- `artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.md`

The first full-grid SGLang shard has been launched with chunk size 128:

- selection:
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_generation_shard_base_t0p0.json`
- log:
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_generation_shard_base_t0p0.log`
- output:
  `artifacts/phase2/generations/olmo3_base_promptpkg5000_free_run_5000prompt_8comp_t0_batched1024.jsonl`

Live check at `2026-06-17T00:05:42Z`: the base/t0 shard was checkpointed and
paused after eighteen chunks, with `18432/40000` rows corresponding to `2304`
prompts x `8` completions and `18,874,368` generated tokens. All checked rows
finished by the fixed length
cap (`1024` tokens), as expected for the full-grid `--ignore-eos` contract.
The primary-slice base/t1 shard was also paused after eight `128`-prompt
checkpoints, so the primary t=1 slice has `8192/160000` rows. The detached
full-grid supervisor and base/t1 worker have both been stopped to avoid
continuing the original `480000`-generation grid while the lightweight Phase 3
scope is being clarified; the partial JSONL caches remain resumable.
This makes the SGLang recovery path an hours-scale shard, not a week-scale
single-shard run, while preserving JSONL resume points after each prompt
chunk.

A detached guarded supervisor script exists to keep the full-grid recovery
moving after each shard exits, but it is intentionally not running while the
smaller Phase 3 grid scope is being clarified:

- Script: `scripts/phase3_full_sglang_supervisor.sh`
- PID file:
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_supervisor.pid`
- Supervisor log:
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_supervisor.log`

When enabled, the supervisor calls `slop-continue-phase2-generation-plan --execute
--allow-partial-retry` every 600 seconds, refreshes the SGLang plan summary
from current JSONL caches before each decision, relies on the launcher's
active-shard detection to avoid duplicate GPU jobs, and stops if `/home/user`
falls below `12` GiB free. It now passes `--priority-temperature 1.0`, so
the current active shard is base/t1, the first stage in the primary t=1.0
four-stage slice; base/t0 remains resumable from its `18432`-row checkpoint
and will return later in the full-grid schedule. The continuation launcher
treats current JSONL and summary
outputs as authoritative for completion, so a stale `completed=True` value in
the sidecar plan cannot silently skip a required shard. It also runs the
generated finalization script when the
readiness summary reports enabled commands, then refreshes the summary and
completion audit before continuing generation. It now also refreshes an
existing-cache SGLang integrity audit from the full-grid plan; the audit checks
duplicate generation keys, malformed JSON, metadata drift, short fixed-budget
rows, empty generations, malformed feature JSON, and prompt groups with
missing or extra completion indices for caches that already exist. The last
detached supervisor PID was `25515` and the last base/t1 worker PID was
`25449`; both are now stopped.

The current lightweight route also separates generation from repeated
analysis. `slop-analyze-phase2-compounding` can materialize and reuse
prompt-cluster counter caches, so expensive bootstrap reruns no longer need to
rescan every generated string. This is now the preferred route for large
compounding CI refreshes. The SmolLM3 three-temperature primary
`slop_lexicon` CI table was regenerated from the counter cache at:
`artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_slop_lexicon_cluster_counts_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3.jsonl`.

The full OLMo SGLang post-generation finalization plan is now materialized at
`artifacts/phase3/analysis/olmo3_full_sglang_5000prompt_8comp_3temp_finalize.sh`
with readiness summary
`artifacts/phase3/analysis/olmo3_full_sglang_5000prompt_8comp_3temp_finalize.md`.
As of the `18432/40000` base/t0 checkpoint and the paused `8192/40000` base/t1
shard, the finalization script has zero enabled commands because no complete
four-stage temperature slice exists yet; commands are commented out until the
required JSONL and summary outputs are complete. The readiness summary now
includes a per-stage table, so the next
safe downstream step is explicit for every stage/temperature cell. The
supervisor refreshes this finalization report after every generation-plan
refresh. The materializer skips downstream commands whose declared output
artifacts already exist, verifies prerequisite input paths, and comments out
commands whose prerequisites are missing, so the generated script is a
pending-work script rather than a rerun-everything script. The supervisor also
checks the refreshed SGLang cache-integrity CSV before running any enabled
finalization commands and exits with a logged block if any existing cache
fails integrity.

The current full-grid SGLang existing-cache integrity audit is
`artifacts/phase3/analysis/olmo3_full_sglang_existing_cache_integrity.md`.
As of the eighth base/t1 checkpoint it reports `2/2` existing caches passed:
base/t0 has `18432` records, `18432` unique generation keys, and `2304`
prompt groups; base/t1 has `8192` records, `8192` unique generation keys, and
`1024` prompt groups. Both caches have zero bad prompt groups and zero
duplicate, invalid-JSON, missing-field, metadata-drift, token-field,
empty-generation, or bad feature-JSON rows. This is an integrity check, not a
completion claim; each row remains incomplete until it reaches `40000/40000`
and has its summary output.

The machine-readable completion audit snapshot is
`artifacts/phase3/analysis/phase3_completion_audit_snapshot.md`. It currently
reports `30/30` required production-scope requirements complete. The expanded
gate set now checks the bounded OLMo and SmolLM3 data-rate inputs, reduced
generation grids, temperature/long-output compounding, archived register-proxy
style-signature diagnostics, SmolLM3 sparse-support addendum, cross-ladder
correlation artifacts, and the OLMo Think/RL-Zero stretch comparison. The original full
OLMo SGLang grid, primary temperature-1 four-stage slice, full-grid
finalization checks, cache-integrity snapshot, and full-grid primary
generation-grid/compounding/spectrum/classification/cross-ladder outputs are
tracked as optional/addendum rows; the current optional/addendum tally is
`4/11` complete.
Required CSV/JSONL artifacts must contain data rows, so empty or header-only
files do not satisfy these gates.

Current reduced OLMo execution status:

- Main paired style-signature panel: complete, 4/4 shards and
  8,192/8,192 generations. The reduced main Tier-1 generation grid is
  `artifacts/phase3/analysis/olmo3_phase3_reduced_main_generation_stage_grid_1024prompt_2comp_t07_sglang.csv`.
- Temperature-sensitivity panel: complete, 12/12 shards and
  6,144/6,144 generations. The reduced temperature grid is
  `artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_generation_stage_grid_512prompt_1comp_3temp_sglang.csv`.
  The reduced temperature realized-AF decomposition and plot are
  `artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_compounding_analysis_512prompt_1comp_3temp_512_sglang.csv`
  and
  `artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_compounding_analysis_512prompt_1comp_3temp_512_sglang_realized_af.svg`.
- Long-output compounding panel: complete, 4/4 shards and
  2,048/2,048 generations. The reduced long-output grid is
  `artifacts/phase3/analysis/olmo3_phase3_reduced_long_generation_stage_grid_256prompt_2comp_t07_1024_sglang.csv`,
  and the window-level compounding analysis is
  `artifacts/phase3/analysis/olmo3_phase3_reduced_long_compounding_analysis_256prompt_2comp_t07_1024_sglang.csv`.
- Integrated reduced OLMo spectrum and classifier:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv`
  and
  `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_feature_classification_t07_long1024.csv`.
- Reduced OLMo-vs-SmolLM3 production comparison:
  `artifacts/phase3/analysis/olmo3_reduced_sglang_vs_smollm3_no_think_production_summary.md`.
- Current bounded Phase 3 conclusion:
  `docs/experiments/phase3_integrated_conclusion_report.md`.
- Supplemental SmolLM3 three-temperature `slop_lexicon` compounding CIs:
  `artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_slop_lexicon_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3.csv`.
- DPO-vs-APO variation report:
  `docs/experiments/phase3_dpo_vs_apo_variation_report.md`.
- Optional OLMo Think/RL-Zero reduced SGLang stretch comparison:
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_256prompt_2comp_t07_1024_sglang.csv`.
  The run is now complete for 10/10 stages and 5,120/5,120 generated records;
  all JSONL rows use `feature_text_mode=final_answer`, so explicit reasoning
  traces are excluded from Tier-1 feature counts while raw generations remain
  cached. Downstream comparison:
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_comparison_256prompt_2comp_t07_1024_sglang_summary.md`.

## Implemented Phase 3 Layer

Added CLI:

- `slop-classify-amplification-spectrum`
- `slop-compare-phase3-ladders`
- `slop-plan-phase2-propensity`
- `slop-analyze-phase3-free-run-effects`
- `slop-analyze-phase3-teacher-forced-effects`
- `slop-extract-smollm3-config-weights`
- `slop-summarize-phase3-generation-plans`
- `slop-continue-phase2-generation-plan`
- `slop-materialize-phase2-generation-continuation-plan`
- `slop-materialize-sglang-generation-plan`
- `slop-materialize-phase3-full-sglang-finalize-plan`
- `slop-audit-sglang-generation-cache`
- `slop-assemble-olmo-stretch-comparison`

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
- `slop-free-running-emission` now accepts `--resume`, preserving existing
  generation-cache rows and skipping already written
  source/record/completion/temperature/top-p cells. This is required for long
  OLMo shards that can OOM after flushing partial JSONL output.
- `slop-assemble-amplification-spectrum` now writes a model-neutral summary
  title because the same assembler is used for OLMo and SmolLM3 spectra.
- `slop-assemble-amplification-spectrum` now accepts
  `--weighted-pretrain-baseline`, so coverage-aware weighted pretraining
  baselines can override the aggregate `pretrain_per_1k_tokens` field while
  raw source-specific pretrain columns remain available for audit. It also
  token-weights repeated rows from the same source before writing
  source-specific pretrain columns, which matters for multi-stratum sources
  such as `infiwebmath-4plus`.
- `slop-assemble-phase2-generation-grid` now accepts
  `--generation-cache stage=path`, `--bootstrap-samples`, and
  `--bootstrap-seed`, so generated-output per-1k rates can receive
  document/prompt-cluster bootstrap CIs from already-completed JSONL caches.
- `slop-assemble-amplification-spectrum` propagates those generated-output
  CI columns as `free_run_per_1k_tokens_ci_low` and
  `free_run_per_1k_tokens_ci_high`. It also infers the intended data-rate role
  for raw OLMo census rows where Dolma pretraining and Dolci SFT rows are
  labeled as generic `text`.

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
- generated-output document/prompt-cluster bootstrap CIs where the generation
  grid supplies them.
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
| Cluster bootstrap CIs | Existing Phase 2 AF table includes bootstrap CIs where teacher-forced measurements exist. Reduced OLMo long-output and SmolLM3 production generated-output grids include document/prompt-cluster bootstrap CIs for per-1k rates, and the current spectra propagate them. Reduced OLMo long-output, reduced OLMo temperature, and SmolLM3 production `t=1.0` compounding tables now include document/prompt-cluster bootstrap CIs for observed/excess rates, realized AF, and prior/no-prior risk metrics; the current spectra propagate the primary compounding CI fields. OLMo and SmolLM3 archived generated-output register-proxy comparison tables include document/prompt-cluster bootstrap CIs and style signatures preserve them as `value_ci_low/high` where available. | Partially done for teacher-forced, generated-output, primary compounding, and archived generated-side register-proxy layers; active full-pybiber uncertainty is corpus-side Phase 1 only and large SmolLM3 temperature-table CI coverage remains incomplete |
| Benjamini-Hochberg FDR across full feature set | Result A chosen-vs-rejected sign-test BH-FDR is joined from `olmo3_dolci_dpo_10k_pair_analysis.csv`; target-shape OLMo free-running stage effects have paired sign-test BH-FDR in `olmo3_phase3_free_run_stage_effects_t1.csv`; retained OLMo teacher-forced stage effects have paired opportunity-level sign-test BH-FDR in `olmo3_phase3_teacher_forced_stage_effects_t1.csv`; both stage-effect families are joined into the regenerated classifier table. | Bounded OLMo done |
| Paired designs wherever corpora share prompts | Preference-complicity uses paired Phase 1 sign-test BH-FDR. Free-running stage effects use paired shared-prompt/completion sign tests. Teacher-forced stage effects use paired shared-opportunity sign tests. | Bounded OLMo done |
| Replicate full spectrum on SmolLM3-3B no_think ladder | A 512-row no_think SmolTalk2 prompt package, canonical chat-template generation plan, propensity plan, all-stage calibration, full 512-prompt four-stage SmolLM3 teacher-forced spectrum for `slop_lexicon`, `neutral_common_controls`, and `rule_of_three_approx`, completed 12-shard 512-prompt x 8-completion x 3-temperature free-running grid, paired free-running FDR, compounding, temperature-dependent realized-AF decomposition, SmolTalk2 SFT/preference data rates, pretraining/Mid baselines, extracted recipe weights, coverage-aware weighted pretrain baseline proxy, and production generation-inclusive SmolLM3 spectrum/classification now exist. A 2,258-prompt sparse teacher-forced addendum now covers `contrastive_negation`, `stock_closers`, and `stock_openers_closers`; `stock_openers` remains unsupported. | SmolLM3 production-shape replication done for available 512-prompt package, with sparse teacher-forced addendum; full Phase 3 still incomplete |
| Report cross-ladder AF rank correlation | `slop-compare-phase3-ladders` implements the statistic, passes an OLMo self-check, and the production SmolLM3 comparison `olmo3_vs_smollm3_no_think_512prompt_production_baselines_data_rates_tf_generation_compounding_slop_neutral_rule3_summary.md` reports 24 aligned rows, 8 shared AF values, Spearman AF `0.762`, and Pearson AF `0.978`. | Production SmolLM3 bounded comparison done. The sparse 2,258-prompt teacher-forced addendum is not merged into this cross-ladder statistic because it has a different sample scope and only 2-3 held-out references per sparse feature. |
| Stretch Instruct vs. Think vs. RL Zero comparison | Reduced final-answer-mode SGLang panel completed: 10/10 stages and 5,120/5,120 records. CI-enriched generation grid: `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_generation_grid_256prompt_2comp_t07_1024_sglang.csv`; comparison summary: `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_comparison_256prompt_2comp_t07_1024_sglang_summary.md`. Instruct final is the maximum for all tracked Tier-1 output features; for `slop_lexicon`, Instruct final is `0.807` per 1k final-answer tokens. | Complete for reduced stretch scope |

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
rates: forty-seven mapped pretraining source samples, one SmolTalk2 Mid sample,
the no_think SFT target sample, and the candidate no_think Tulu preference
sample. This closes a much larger bounded data-rate gap for Phase 3
classification. It still does not reconstruct a full production weighted
SmolLM3 pretraining feature-rate baseline, because `3.429%` of the extracted
recipe remains unsampled after explicit same-family proxies and some sampled
Stack-Edu rows come from hydrated public mirrors rather than the metadata-only
canonical HF config.

Sampled inputs:

| Sample | Rows | Tokens | Notes |
|---|---:|---:|---|
| FineWeb-Edu pretraining source | 2,000 | 1,610,774 | Web source from the SmolLM3 pretraining collection. |
| DCLM pretraining source | 2,000 | 1,780,498 | Largest extracted recipe source; sampled from `mlfoundations/dclm-baseline-1.0`, config `default`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 German pretraining source | 2,000 | 986,326 | Next largest unsampled web source; sampled from `HuggingFaceFW/fineweb-2`, config `deu_Latn`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 Spanish pretraining source | 2,000 | 1,271,133 | Next FineWeb2 web source; sampled from `HuggingFaceFW/fineweb-2`, config `spa_Latn`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 French pretraining source | 2,000 | 1,152,313 | Next directly sampleable FineWeb2 web source; sampled from `HuggingFaceFW/fineweb-2`, config `fra_Latn`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 Italian pretraining source | 2,000 | 1,053,366 | Next directly sampleable FineWeb2 web source; sampled from `HuggingFaceFW/fineweb-2`, config `ita_Latn`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 Chinese pretraining source | 2,000 | 269,299 | Next directly sampleable FineWeb2 web source; sampled from `HuggingFaceFW/fineweb-2`, config `cmn_Hani`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 Russian pretraining source | 2,000 | 1,587,847 | Next directly sampleable FineWeb2 web source; sampled from `HuggingFaceFW/fineweb-2`, config `rus_Cyrl`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 Portuguese pretraining source | 2,000 | 1,344,559 | Next directly sampleable FineWeb2 web source; sampled from `HuggingFaceFW/fineweb-2`, config `por_Latn`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 Persian pretraining source | 2,000 | 1,674,442 | Next directly sampleable FineWeb2 web source; sampled from `HuggingFaceFW/fineweb-2`, config `fas_Arab`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 Hindi pretraining source | 2,000 | 1,781,688 | Next directly sampleable FineWeb2 web source; sampled from `HuggingFaceFW/fineweb-2`, config `hin_Deva`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 Japanese pretraining source | 2,000 | 258,131 | Next directly sampleable FineWeb2 web source; sampled from `HuggingFaceFW/fineweb-2`, config `jpn_Jpan`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 Korean pretraining source | 2,000 | 709,095 | Next directly sampleable FineWeb2 web source; sampled from `HuggingFaceFW/fineweb-2`, config `kor_Hang`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 Thai pretraining source | 2,000 | 1,180,666 | Next directly sampleable FineWeb2 web source; sampled from `HuggingFaceFW/fineweb-2`, config `tha_Thai`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 Vietnamese pretraining source | 2,000 | 1,984,302 | Next directly sampleable FineWeb2 web source; sampled from `HuggingFaceFW/fineweb-2`, config `vie_Latn`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineWeb2 Greek pretraining source | 2,000 | 1,054,624 | Next directly sampleable FineWeb2 web source; sampled from `HuggingFaceFW/fineweb-2`, config `ell_Grek`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineMath pretraining source | 2,000 | 2,302,447 | Math web source; sampled from `HuggingFaceTB/finemath`, config `finemath-3plus`, split `train`, hash reservoir over 20,000 scanned rows. |
| FineMath-4plus pretraining source | 2,000 | 1,774,598 | Exact next directly sampleable math source; sampled from `HuggingFaceTB/finemath`, config `finemath-4plus`, split `train`, hash reservoir over 20,000 scanned rows. |
| InfiWebMath-4plus pretraining source | 2,000 | 1,868,874 | Exact public math source; sampled from `HuggingFaceTB/finemath`, config `infiwebmath-4plus`, split `train`, hash reservoir over 20,000 scanned rows. Retained strata are `web_cc`, `forums_qa`, `wiki`, `code`, and `scientific`; source rates are token-weighted across those strata. |
| InfiWebMath-3plus pretraining source | 2,000 | 1,990,392 | Exact public long-context math source row; sampled from `HuggingFaceTB/finemath`, config `infiwebmath-3plus`, split `train`, hash reservoir over 20,000 scanned rows. Retained strata are `web_cc`, `forums_qa`, `wiki`, `code`, and `scientific`; source rates are token-weighted across those strata. |
| Wikipedia pretraining source | 2,000 | 2,183,945 | Exact public English Wikipedia source mapped to `wiki`; sampled from `wikimedia/wikipedia`, config `20231101.en`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
| OpenMathInstruct-2 solution-text pretraining source | 2,000 | 319,125 | Bounded exact source sample mapped to `openmathinstruct-2`; sampled from `nvidia/OpenMathInstruct-2`, split `train`, text field `generated_solution`, hash reservoir over 20,000 scanned rows. This measures generated solution text rather than reconstructing a combined problem-plus-solution training record. |
| OpenMathReasoning-4k answer-text pretraining source | 2,000 | 4,401,574 | Bounded exact source sample mapped to `openmathreasoning-4k`; sampled from `LLMcompe-Team-Watanabe/math_OpenMathReasoning_preprocess_4k-8k`, split `train`, text field `answer`, hash reservoir over 20,000 scanned rows. This measures answer text as published, including `<think>` reasoning markup where present. |
| Natural Reasoning joined-response pretraining source | 2,000 | 1,000,263 | Bounded exact source sample mapped to `natural_reasoning`; sampled from `facebook/natural_reasoning`, split `train`, text field `responses`, hash reservoir over 20,000 scanned rows. The reader joins `responses[].response` strings per row, so this is a joined-response measurement rather than a reconstruction of any tokenizer packing. |
| MegaMath text-code-block pretraining source | 2,000 | 844,480 | Exact public synthetic math/code source; sampled from `LLM360/MegaMath` parquet data files under `megamath-text-code-block/*.parquet` through the generic `parquet` loader, split `train`, hash reservoir over 19,150 scanned rows. |
| MegaMath web-pro pretraining source | 2,000 | 1,182,841 | Exact public math web source; sampled from `LLM360/MegaMath` parquet data files under `megamath-web-pro/*.parquet` through the generic `parquet` loader, split `train`, hash reservoir over 20,000 scanned rows. Source rates are token-weighted across inferred strata. |
| MegaMath QA Qwen pretraining source | 2,000 | 298,474 | Bounded exact public source mapped to `megamath-qa-qwen`; sampled from `LLM360/MegaMath` parquet data files under `megamath-qa/qwen-2.5/*.parquet` through the generic `parquet` loader, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. Source rates are token-weighted across inferred strata. |
| Cosmopedia2 public 20k pretraining proxy | 2,000 | 1,057,350 | Bounded public proxy mapped to `cosmopedia2`; sampled from `HuggingFaceTB/cosmopedia-20k`, config `default`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. This is a public bounded sample, not a full Cosmopedia2 reconstruction. |
| OpenCodeReasoning-4k output-text pretraining source | 2,000 | 9,993,355 | Bounded source sample mapped to `open-codereasoning-4k`; sampled from `nvidia/OpenCodeReasoning`, config/split `split_0`, text field `output`, hash reservoir over 20,000 scanned rows. This measures output text rather than exact 4k tokenizer packing. |
| Multilingual Wikipedia pretraining source | 3,000 | 7,734,154 | Bounded public multilingual wiki sample mapped to `multilingual_wiki`; combined German, Spanish, and French `wikimedia/wikipedia` configs (`20231101.de`, `20231101.es`, `20231101.fr`), 1,000 rows each, text field `text`, hash reservoir over 10,000 scanned rows per language. This is a three-language proxy for the recipe source, not a full multilingual mixture reconstruction. |
| PES2O pretraining source | 2,000 | 399,454 | Academic/scientific source; sampled from `allenai/dolmino-mix-1124`, config `pes2o`, split `train`, hash reservoir over 20,000 scanned rows. |
| StackExchange Apple pretraining source | 2,000 | 1,013,267 | Q&A/forum-like source from the SmolLM3 pretraining collection, using `ThreadText`. |
| GitHub issues pretraining source | 2,000 | 565,111 | Exact public source mapped to `github-issues`; sampled from `HuggingFaceTB/issues-kaggle-notebooks`, config `issues`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
| Kaggle pretraining source | 2,000 | 2,259,555 | Exact public source mapped to `kaggle`; sampled from `HuggingFaceTB/issues-kaggle-notebooks`, config `kaggle`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
| Stack-Edu Python pretraining source | 2,000 | 611,029 | Code source mapped to `stack-edu-Python`; sampled from hydrated mirror `hongliu9903/stack_edu_python`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. The canonical `HuggingFaceTB/stack-edu` config exposes blob metadata only. |
| Stack-Edu Cpp pretraining source | 2,000 | 621,130 | Code source mapped to `stack-edu-Cpp`; sampled from hydrated mirror `hongliu9903/stack_edu_cpp`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
| Stack-Edu Java pretraining source | 2,000 | 761,122 | Code source mapped to `stack-edu-Java`; sampled from hydrated mirror `hongliu9903/stack_edu_java`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
| Stack-Edu JavaScript pretraining source | 2,000 | 543,739 | Code source mapped to `stack-edu-JavaScript`; sampled from hydrated mirror `hongliu9903/stack_edu_javascript`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
| Stack-Edu C pretraining source | 2,000 | 607,403 | Code source mapped to `stack-edu-C`; sampled from hydrated mirror `hongliu9903/stack_edu_c`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
| Stack-Edu PHP pretraining source | 2,000 | 548,717 | Code source mapped to `stack-edu-PHP`; sampled from hydrated mirror `hongliu9903/stack_edu_php`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
| Stack-Edu TypeScript pretraining source | 2,000 | 528,196 | Code source mapped to `stack-edu-TypeScript`; sampled from hydrated mirror `hongliu9903/stack_edu_typescript`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
| Stack-Edu SQL pretraining source | 2,000 | 1,328,095 | Code source mapped to `stack-edu-SQL`; sampled from hydrated mirror `hongliu9903/stack_edu_sql`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
| Stack-Edu Go pretraining source | 2,000 | 630,770 | Code source mapped to `stack-edu-Go`; sampled from hydrated mirror `hongliu9903/stack_edu_go`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
| Stack-Edu Ruby pretraining source | 2,000 | 405,272 | Code source mapped to `stack-edu-Ruby`; sampled from hydrated mirror `hongliu9903/stack_edu_ruby`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
| Stack-Edu Rust pretraining source | 2,000 | 1,117,169 | Code source mapped to `stack-edu-Rust`; sampled from hydrated mirror `hongliu9903/stack_edu_rust`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
| Stack-Edu Shell pretraining source | 2,000 | 571,723 | Code source mapped to `stack-edu-Shell`; sampled from hydrated mirror `hongliu9903/stack_edu_shell`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
| Stack-Edu Swift pretraining source | 2,000 | 552,201 | Code source mapped to `stack-edu-Swift`; sampled from hydrated mirror `hongliu9903/stack_edu_swift`, split `train`, text field `text`, hash reservoir over 20,000 scanned rows. |
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
  `fineweb-edu` (`31.140%`), `fw2-deu` (`2.209%`), `fw2-spa` (`2.003%`),
  `pes2o` (`1.724%`), `fw2-fra` (`1.607%`), `finemath` (`1.410%`),
  `stack-edu-Python` (`1.811%`), `stack-edu-Cpp` (`1.304%`),
  `fw2-ita` (`1.062%`), `fw2-cmn` (`0.991%`), `fw2-rus` (`0.991%`),
  `fw2-por` (`0.931%`), `megamath-text-code-block` (`0.907%`),
  `stack-edu-Java` (`0.942%`), `stack-edu-JavaScript` (`0.942%`),
  `finemath-4plus` (`0.606%`), `stack-edu-C` (`0.507%`),
  `megamath-web-pro` (`0.477%`), `stack-edu-PHP` (`0.435%`),
  `infiwebmath-4plus` (`0.383%`), `fw2-fas` (`0.347%`), `stackexchange`
  (`0.333%`), `fw2-hin` (`0.321%`), `fw2-jpn` (`0.321%`), `fw2-kor`
  (`0.321%`), `fw2-tha` (`0.321%`), `stack-edu-SQL` (`0.290%`),
  `stack-edu-TypeScript` (`0.264%`), `fw2-vie` (`0.237%`), `fw2-ell`
  (`0.222%`), `github-issues` (`0.340%`), `kaggle` (`0.051%`),
  `stack-edu-Swift` (`0.088%`), `stack-edu-Rust` (`0.071%`),
  `stack-edu-Ruby` (`0.061%`), `stack-edu-Shell` (`0.052%`), and
  `stack-edu-Go` (`0.044%`) of the extracted recipe. This is enough for a
  high-coverage weighted proxy, but not enough to compute a production
  weighted feature-rate baseline across all recipe sources.
- The sampled Stack-Edu language rows use public hydrated mirrors from
  `hongliu9903/*`. The canonical `HuggingFaceTB/stack-edu` configs expose blob
  metadata rather than code text through the HF dataset schema. Python and Cpp
  mirror probes matched the first canonical blob IDs, so these are reasonable
  source-mapped samples, but they should be described as mirror-hydrated
  samples rather than direct canonical dataset text reads.

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
  `smollm3_pretrain_pes2o_2k=pes2o`, and
  `smollm3_pretrain_fw2_fra_2k=fw2-fra`, and
  `smollm3_pretrain_finemath_2k=finemath`, and
  `smollm3_pretrain_fw2_ita_2k=fw2-ita`,
  `smollm3_pretrain_fw2_cmn_2k=fw2-cmn`,
  `smollm3_pretrain_fw2_rus_2k=fw2-rus`,
  `smollm3_pretrain_fw2_por_2k=fw2-por`, and
  `smollm3_pretrain_finemath_4plus_2k=finemath-4plus`, and
  `smollm3_pretrain_infiwebmath_4plus_2k=infiwebmath-4plus`,
  `smollm3_pretrain_megamath_text_code_block_2k=megamath-text-code-block`,
  and `smollm3_pretrain_megamath_web_pro_2k=megamath-web-pro`, then added
  `smollm3_pretrain_fw2_fas_2k=fw2-fas`,
  `smollm3_pretrain_fw2_hin_2k=fw2-hin`,
  `smollm3_pretrain_fw2_jpn_2k=fw2-jpn`,
  `smollm3_pretrain_fw2_kor_2k=fw2-kor`,
  `smollm3_pretrain_fw2_tha_2k=fw2-tha`,
  `smollm3_pretrain_fw2_vie_2k=fw2-vie`, and
  `smollm3_pretrain_fw2_ell_2k=fw2-ell`, then added source maps for
  `stack-edu-Python`, `stack-edu-Cpp`, `stack-edu-Java`,
  `stack-edu-JavaScript`, `stack-edu-C`, `stack-edu-PHP`,
  `stack-edu-TypeScript`, `stack-edu-SQL`, `stack-edu-Go`, `stack-edu-Ruby`,
  `stack-edu-Rust`, `stack-edu-Shell`, and `stack-edu-Swift`, then added
  exact source maps for `github-issues` and `kaggle` from
  `HuggingFaceTB/issues-kaggle-notebooks`, then added exact source maps for
  `wiki`, `infiwebmath-3plus`, and the bounded
  `openmathinstruct-2` solution-text sample, then added
  `openmathreasoning-4k` answer text and `natural_reasoning` joined responses,
  then added `megamath-qa-qwen`, a public `cosmopedia2` proxy,
  `open-codereasoning-4k` output text, and a bounded three-language
  `multilingual_wiki` proxy. The retained Tier-1 direct coverage is
  `91.969%` of the extracted recipe through 47 mapped source samples.
  The current baseline also applies explicit `--source-proxy` mappings for
  inaccessible same-family rows: `infiwebmath` from `infiwebmath-4plus` and
  same-language Stack-Edu `real`/`real-shuffled` recipe rows from the measured
  Stack-Edu language samples. These 22 proxy rows add `4.601%` coverage, so
  the retained Tier-1 weighted baseline covers `96.571%` and leaves `3.429%`
  unsampled.
- Covered-only weighted rates per 1k tokens are now: `slop_lexicon` `0.347`,
  `rule_of_three_approx` `3.638`, `contrastive_negation` `0.320`,
  `stock_closers` `0.031`, `stock_openers` `0.025`, and pooled stock phrases
  `0.056`. These are coverage-normalized diagnostics over direct mapped
  samples plus explicit source proxies, not exact full-mixture estimates,
  because the remaining unsampled recipe share is still nonzero.
- `list_header_bold_lead_in` and `punctuation_rhythm` are at `31.722%`
  coverage in the proxy because most source censuses were run only for the
  retained Tier-1 features; the currently covered structural/punctuation rows
  come from sources whose censi included all current Tier-1 matcher outputs.

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
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_fra_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_fra_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_ita_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_ita_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_cmn_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_cmn_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_rus_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_rus_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_por_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_por_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_fas_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_fas_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_hin_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_hin_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_jpn_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_jpn_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_kor_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_kor_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_tha_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_tha_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_vie_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_vie_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_ell_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_fw2_ell_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_finemath_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_finemath_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_finemath_4plus_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_finemath_4plus_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_infiwebmath_4plus_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_infiwebmath_4plus_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_megamath_text_code_block_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_megamath_text_code_block_2k_summary.md`
- `artifacts/stage1/corpora/smollm3_pretrain_megamath_web_pro_2k.jsonl`
- `artifacts/stage1/corpora/smollm3_pretrain_megamath_web_pro_2k_summary.md`
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
- `artifacts/stage1/census/smollm3_pretrain_fw2_fra_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_fw2_ita_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_fw2_cmn_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_fw2_rus_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_fw2_por_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_fw2_fas_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_fw2_hin_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_fw2_jpn_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_fw2_kor_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_fw2_tha_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_fw2_vie_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_fw2_ell_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_finemath_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_finemath_4plus_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_infiwebmath_4plus_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_megamath_text_code_block_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_megamath_web_pro_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_pretrain_pes2o_2k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_smoltalk2_sft2260_pref10k_tier1_feature_rates.csv`
- `artifacts/stage1/census/smollm3_smoltalk2_pref10k_tier1_pair_deltas.csv`
- `artifacts/stage1/census/smollm3_smoltalk2_pref10k_tier1_pair_analysis.csv`

Tier-1 rates, per 1,000 simple tokens:

| Feature | Weighted proxy covered-only | FineWeb-Edu | DCLM | FineWeb2 German | FineWeb2 Spanish | FineWeb2 French | FineWeb2 Italian | FineWeb2 Chinese | FineWeb2 Russian | FineWeb2 Portuguese | FineWeb2 Persian | FineWeb2 Hindi | FineWeb2 Japanese | FineWeb2 Korean | FineWeb2 Thai | FineWeb2 Vietnamese | FineWeb2 Greek | FineMath | FineMath-4plus | InfiWebMath-4plus | MegaMath text-code | MegaMath web-pro | PES2O | StackExchange | Mid | SFT target | Chosen | Rejected | Chosen - Rejected | Pair BH q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `contrastive_negation` | 0.320 | 0.385 | 0.487 | 0.006 | 0.002 | 0.002 | 0.003 | 0.004 | 0.003 | 0.013 | 0.003 | 0.004 | 0.004 | 0.004 | 0.003 | 0.002 | 0.011 | 0.181 | 0.159 | 0.158 | 0.109 | 0.099 | 0.238 | 0.509 | 0.177 | 0.095 | 0.379 | 0.352 | +0.027 | 0.039 |
| `rule_of_three_approx` | 3.638 | 6.211 | 3.746 | 0.184 | 0.038 | 0.100 | 0.066 | 0.152 | 0.052 | 0.162 | 0.045 | 0.053 | 0.256 | 0.118 | 0.042 | 0.033 | 0.117 | 1.817 | 1.909 | 1.896 | 0.817 | 4.451 | 5.520 | 1.583 | 1.273 | 17.045 | 5.443 | 4.877 | +0.566 | 9.437e-13 |
| `slop_lexicon` | 0.347 | 0.536 | 0.354 | 0.047 | 0.005 | 0.088 | 0.036 | 0.022 | 0.005 | 0.027 | 0.002 | 0.004 | 0.008 | 0.024 | 0.023 | 0.005 | 0.018 | 0.150 | 0.170 | 0.177 | 0.124 | 0.332 | 0.716 | 0.327 | 0.050 | 0.419 | 1.609 | 1.293 | +0.316 | 3.367e-15 |
| `stock_closers` | 0.031 | 0.042 | 0.037 | 0.002 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.036 | 0.025 | 0.029 | 0.033 | 0.085 | 0.060 | 0.026 | 0.027 | 0.270 | 0.217 | 0.167 | +0.050 | 3.016e-04 |
| `stock_openers` | 0.025 | 0.027 | 0.037 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.065 | 0.078 | 0.073 | 0.000 | 0.008 | 0.000 | 0.069 | 0.000 | 0.106 | 0.571 | 0.334 | +0.237 | 0.161 |
| `stock_openers_closers` | 0.056 | 0.068 | 0.074 | 0.002 | 0.000 | 0.001 | 0.000 | 0.007 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.001 | 0.001 | 0.000 | 0.000 | 0.100 | 0.104 | 0.102 | 0.033 | 0.093 | 0.060 | 0.095 | 0.027 | 0.376 | 0.788 | 0.502 | +0.287 | 0.003 |

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

The `baselines_data_rates` spectrum is the current preferred bounded SmolLM3
data-rate spectrum. It now uses
`artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
as the aggregate pretraining baseline override, with `91.969%` direct covered
recipe share from source maps, `4.601%` explicit source-proxy share, `96.571%`
total covered recipe share, and source-specific columns retained for all
forty-seven mapped pretraining source samples. The regenerated
OLMo-vs-SmolLM3 comparison still
aligns 24 feature-stage rows and 8 shared AF values, with overall Spearman AF
`0.762` and Pearson AF `0.978`; the correlation is unchanged because this pass
updated the data-rate layer rather than the teacher-forced AF layer.

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

The explicit SmolLM3 support audit is:

- `artifacts/phase3/analysis/smollm3_no_think_phase3_feature_support_audit_512prompt_and_full_sft.csv`
- `artifacts/phase3/analysis/smollm3_no_think_phase3_feature_support_audit_512prompt_and_full_sft.md`

On the actual 512-prompt package, `slop_lexicon` has 19 reference
initiations over 43,390 token starts, `rule_of_three_approx` has 628
reference initiations over 754 comma-pair extension opportunities, and neutral
controls have denominator support. `contrastive_negation` and the stock
opener/closer families occur in raw target text, but have zero aligned
reference initiations under the current 512-prompt opportunity contract, so
they should not be promoted to SmolLM3 teacher-forced AF claims without a new
contract or a different prompt package.

The archived generated-output register-proxy layer includes SmolLM3 diagnostic
artifacts:

- `artifacts/stage1/census/smollm3_smoltalk2_sft2260_pref10k_biber_lite_feature_rates.csv`
- `artifacts/stage1/census/smollm3_smoltalk2_pref10k_biber_lite_pair_deltas.csv`
- `artifacts/phase2/analysis/smollm3_biber_lite_generation_vs_corpus_512prompt_8comp_t1_chat_production.csv`
- `artifacts/phase2/analysis/smollm3_biber_lite_generation_vs_corpus_512prompt_8comp_t1_chat_production_summary.md`
- `artifacts/phase3/analysis/smollm3_no_think_style_signature_512prompt_8comp_t1_chat_production.csv`
- `artifacts/phase3/analysis/smollm3_no_think_style_signature_512prompt_8comp_t1_chat_production_distances.csv`
- `artifacts/phase3/analysis/smollm3_no_think_style_signature_512prompt_8comp_t1_chat_production_summary.md`

The historical style-signature read is simple: Base is the outlier, while SFT,
APO/DPO, and Final/RLVR cluster tightly. On the production raw mixed-scale
distance table, DPO to Final distance is `8.765`, SFT to Final is `26.157`,
and Base to Final is `459.745` across 43 shared signature coordinates.
Feature-level shifts still
matter more than the aggregate distance: Final/RLVR is far below Base on
first-person pronouns and wh-question style, far above Base on rule-of-three
compounding, and very close to DPO overall.

The concrete production-grid plan artifact is:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`
- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.md`

This plan covers the available 512-prompt SmolLM3 no_think package across four
stages, 8 completions, and temperatures `0.0`, `0.7`, and `1.0` for 12 shards.
All 12 shards are complete.

Caveat: the normalized SmolTalk2 sample carries the no_think split/source
identity, but per-row `preference_type`, `chosen_model`, and `rejected_model`
fields are not exposed by this sample path and appear as `None` in the paired
analysis table. The source split identifies the candidate Tulu no_think
preference mixture, but the current SmolLM3 paired analysis is not yet
stratified by per-row preference-construction metadata.

Second caveat: the current SmolLM3 feature-rate pretrain baseline is a
coverage-aware weighted proxy over forty-seven mapped recipe source samples.
It is useful as a pretraining-source baseline for Phase 3 classification, but
it is not the full SmolLM3 pretraining mixture. The exact published recipe
weights are now resolved, and the current proxy-aware coverage is high enough
for bounded Phase 3 interpretation. The remaining few percent of recipe-source
coverage is intentionally not treated as the active blocker.

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

## SmolLM3 Sparse Teacher-Forced Addendum

A full eligible SmolTalk2 everyday no_think SFT prompt package was materialized
from `HuggingFaceTB/smoltalk2` with 2,258 prompts:

- `artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_2258.jsonl`
- `artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_2258_manifest.csv`
- `artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_2258_summary.json`

The denominator audit shows sparse but nonzero target-reference support for
three features that were missing from the 512-prompt teacher-forced slice:

| Feature | Opportunities | References | Reference rate |
|---|---:|---:|---:|
| `contrastive_negation` | 16,526 | 2 | 0.000121 |
| `stock_closers` | 4,139 | 3 | 0.000725 |
| `stock_openers_closers` | 6,397 | 3 | 0.000469 |
| `stock_openers` | 2,258 | 0 | 0 |

All four corrected base/SFT/APO/final teacher-forced sparse shards completed.
The practical runtime was about 11.5-11.75 minutes per stage on the A100,
because actual sparse opportunities were about 27k rows per stage rather than
the planner's pessimistic `prompt_count * feature_count * cap` estimate.

Sparse addendum outputs:

- `artifacts/phase3/analysis/smollm3_no_think_propensity_plan_2258prompt_sparse_supported.csv`
- `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_2258prompt_sparse_supported.csv`
- `artifacts/phase3/analysis/smollm3_no_think_teacher_forced_stage_effects_2258prompt_sparse_supported.csv`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_2258prompt_sparse_supported.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_2258prompt_sparse_supported.csv`

Sparse teacher-forced raw AFs:

| Feature | Base | SFT | APO | Final |
|---|---:|---:|---:|---:|
| `contrastive_negation` | 1.967 | 3.124 | 3.026 | 3.254 |
| `stock_closers` | 42.206 | 91.178 | 100.983 | 96.294 |
| `stock_openers_closers` | 53.770 | 105.893 | 128.318 | 123.391 |

The paired opportunity-level stage-effect test produced 9 rows and 8
BH-FDR-significant rows. The sparse addendum classifier labels
`contrastive_negation` as `sft-amplified`, and labels `stock_closers` and
`stock_openers_closers` as `preference-amplified`/`dynamics-driven`. These are
addendum labels only: the held-out reference counts are 2 or 3, every raw-AF
CI overlaps 1, and there is no matching generated-output compounding
expectation join for this 2,258-prompt sparse package.

## Next Work Toward Full Phase 3

The next concrete work needed to complete Phase 3 from `EXPERIMENTS.md` is:

1. Promote sparse teacher-forced addendum rows into the main spectrum only if
   the report explicitly accepts their low-reference support and sample-scope
   mismatch.
2. Add any further teacher-forced feature coverage only if new opportunity
   contracts or prompt subsets provide denominator support; `stock_openers`
   remains unsupported in the 2,258-prompt target package.
3. Propagate any expanded feature set through the Biber/style-signature layer
   only if those features are promoted into the main Phase 3 spectrum.
