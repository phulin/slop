# Phase 3 Production Runbook

Date: 2026-06-16

This runbook covers the reduced-compute operational path for finishing the
bounded Phase 3 comparisons in `EXPERIMENTS.md`, plus the current SGLang-only
recovery path for the original 5,000-prompt x 8-completion x 3-temperature
OLMo grid. It does not reprioritize the remaining pretraining source-coverage
gap.

## Current Plan Status

Generated status summary for the active reduced plan:

- `artifacts/phase3/analysis/phase3_reduced_sglang_generation_plan_status_summary.csv`
- `artifacts/phase3/analysis/phase3_reduced_sglang_generation_plan_status_summary.md`

Refresh it with:

```bash
uv run slop-summarize-phase3-generation-plans \
  --plan olmo3_reduced_main_sglang=artifacts/phase3/analysis/olmo3_phase3_reduced_main_signature_1024prompt_2comp_t07_512_sglang.csv \
  --plan olmo3_reduced_temperature_sglang=artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_512prompt_1comp_3temp_512_sglang.csv \
  --plan olmo3_reduced_long_compounding_sglang=artifacts/phase3/analysis/olmo3_phase3_reduced_long_compounding_256prompt_2comp_t07_1024_sglang.csv \
  --plan smollm3_no_think_512prompt_8comp_3temp=artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv \
  --output artifacts/phase3/analysis/phase3_reduced_sglang_generation_plan_status_summary.csv \
  --summary-output artifacts/phase3/analysis/phase3_reduced_sglang_generation_plan_status_summary.md
```

Current active generation plans:

- OLMo main paired style-signature panel:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_main_signature_1024prompt_2comp_t07_512_sglang.csv`
- OLMo temperature-sensitivity panel:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_512prompt_1comp_3temp_512_sglang.csv`
- OLMo long-output compounding panel:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_long_compounding_256prompt_2comp_t07_1024_sglang.csv`
- SmolLM3 no_think completed 512-prompt production-shape plan:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`

The active reduced OLMo plan totals 16,384 planned generations:

- 8,192 main paired generations: 1,024 prompts x 2 completions x 4 stages at
  temperature 0.7 and 512 max new tokens.
- 6,144 temperature generations: 512 prompts x 1 completion x 3 temperatures
  x 4 stages at 512 max new tokens.
- 2,048 long compounding generations: 256 prompts x 2 completions x 4 stages
  at temperature 0.7 and 1,024 max new tokens.

The conservative source-plan estimate is `21.84` A100-hours before SGLang
speedup. The abandoned full OLMo grid planned 480,000 generations and about
`1137.78` conservative A100-hours. The previous full-grid OLMo base/t0 shard
was deliberately stopped at `1472/40000` rows and retained as pilot/cache data:
`artifacts/phase2/generations/olmo3_base_promptpkg5000_free_run_5000prompt_8comp_t0_batched64.jsonl`.

SGLang is the default backend for these pure generation tasks. The SGLang
plans use `scripts/benchmark_sglang_generation.py`,
`.venvs/sglang-cu128/bin/python`, `--ignore-eos`, and a fixed token budget.
This is a deliberate backend-specific generation contract, not a drop-in
replacement for the earlier Torch stop-token contract. The repo-local SGLang
sidecar has now been created and smoke-tested on OLMo 3. The working runtime
uses the extracted local `libnuma.so.1`, `/usr/lib/x86_64-linux-gnu` for the
driver `libcuda.so.1`, `--attention-backend triton`, and
`--disable-cuda-graph`, which avoids the FlashInfer `nvcc` JIT path on this
host.

Create or refresh the SGLang sidecar with:

```bash
uv venv --python 3.12 .venvs/sglang-cu128
uv pip install --python .venvs/sglang-cu128/bin/python -e .
uv pip install --python .venvs/sglang-cu128/bin/python \
  'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' 'torchaudio==2.8.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venvs/sglang-cu128/bin/python \
  'sglang[srt]==0.5.2' 'transformers==4.57.6' \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
```

For this host, the materialized plans set the required runtime environment
through `env` directly in each command:

```bash
LD_LIBRARY_PATH=/home/user/slop/.venvs/sglang-cu128/sysroot/extracted/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu
TORCH_CUDA_ARCH_LIST=8.0
```

The local `libnuma` shim was created without root by downloading and extracting
Ubuntu's `libnuma1_2.0.18-1build1_amd64.deb` into
`.venvs/sglang-cu128/sysroot/extracted`.

Current execution status:

- OLMo main paired style-signature panel: complete, 4/4 shards,
  8,192/8,192 generations.
- OLMo temperature-sensitivity panel: complete, 12/12 shards,
  6,144/6,144 generations.
- OLMo long-output compounding panel: complete, 4/4 shards,
  2,048/2,048 generations.

## Lightweight Strategy

The current bounded-report route is intentionally not blocked on the original
480,000-generation OLMo grid. The practical strategy is:

- Use SGLang for pure generation. Do not keep searching for a workable
  Transformers batch size for generation-only shards unless SGLang is blocked
  by a concrete loader/runtime failure.
- Treat Transformers batch size 32 as a legacy recovery setting, not a
  production strategy. For generation-only shards, the useful knobs are SGLang
  prompt chunk size, resume checkpoints, and slice ordering.
- Use reduced, paired panels that answer specific questions: main style
  progression, temperature sensitivity, and long-output compounding.
- Keep `t=1.0` production-shape grids as the default source for generated
  style signatures and pybiber register comparisons.
- Use three-temperature grids only for realized-AF temperature dependence, not
  as the default source for every downstream style statistic.
- Materialize compounding counter caches before expensive bootstraps. Reusing
  prompt-cluster counters makes bootstrap reruns seconds-scale and avoids
  rescanning the same generated text.
- Treat remaining pretraining-source coverage gaps as audit caveats, not as a
  blocker for the bounded Phase 3 conclusion.

For original-scope parity work, use the full SGLang plan, not the legacy
Torch continuation plans:

- Full plan:
  `artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.csv`
- Summary:
  `artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.md`
- Current shard selection:
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_generation_shard_base_t0p0.json`
- Current supervisor PID file:
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_supervisor.pid`
- Current supervisor log:
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_supervisor.log`
- Existing-cache integrity audit:
  `artifacts/phase3/analysis/olmo3_full_sglang_existing_cache_integrity.md`

The full-grid SGLang policy is:

- Keep one shard per model stage and temperature: 40,000 generations per
  shard, 12 shards total.
- Use `--prompt-batch-size` as checkpoint granularity, not as the equivalent
  of a Transformers generation batch size. A chunk of 128 prompts means 1,024
  completions before each JSONL flush for the current 8-completion plan, while
  SGLang's scheduler still packs decoding work internally.
- Prefer chunk sizes that write rows every few minutes. If a first chunk takes
  too long to flush, stop only after confirming active GPU work, then relaunch
  the same shard with a lower prompt chunk size and `--resume`.
- Keep SGLang's internal scheduler responsible for packing work on the GPU;
  do not fall back to the old batch-32 Transformers path just to get smaller
  checkpoints.
- Measure realized tokens/sec from the first completed chunk before projecting
  total wall time. The source-plan A100-hour estimate is deliberately
  conservative and predates the SGLang backend.
- Refresh the SGLang cache integrity audit from the plan and existing JSONL
  caches before trusting downstream assembly. A partial cache can pass
  integrity while still being incomplete; completion remains governed by the
  generation plan and finalization readiness.

Live status at `2026-06-17T00:05:42Z`: base/t0 was checkpointed and paused at
`18432/40000` rows, or eighteen 128-prompt chunks. Base/t1 is now the active
primary-slice shard and has flushed its first 128-prompt checkpoints to
`8192/40000` rows. The full-grid supervisor and base/t1 worker were then
stopped deliberately because the intended lightweight Phase 3 scope is smaller
than the original `480000`-generation grid. The partial caches are intact and
resumable.
When a SGLang shard is running, the scheduler is expected to stay at high GPU utilization while a
chunk is decoding, and JSONL output appears only after each `engine.generate`
chunk returns.

Start or restart the guarded supervisor with `setsid -f`. Plain `nohup ... &`
from this execution environment did not keep the monitor alive after the
launcher shell exited. The supervisor writes its own PID file on startup and
exits immediately if that PID file already points to a live
`phase3_full_sglang_supervisor.sh` process.
It rotates its own monitor log at startup when `LOG_OUTPUT` exceeds
`MAX_LOG_BYTES`, which defaults to `5000000`.

```bash
setsid -f scripts/phase3_full_sglang_supervisor.sh >/dev/null 2>&1 < /dev/null
```

The supervisor invokes:

```bash
uv run slop-continue-phase2-generation-plan \
  --plan artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.csv \
  --selection-dir artifacts/phase3/analysis \
  --selection-prefix olmo3_full_sglang_chunk128_generation_shard \
  --max-estimated-a100-hours 40 \
  --execute \
  --allow-partial-retry \
  --priority-temperature 1.0 \
  --decision-output artifacts/phase3/analysis/olmo3_full_sglang_chunk128_generation_next_decision.json
```

It checks every 600 seconds, refuses to run below `12` GiB free on
`/home/user`, and relies on the launcher's active-selection detection to avoid
starting another shard while one is still alive. The `PRIORITY_TEMPERATURES`
environment variable defaults to `1.0`, which prioritizes a complete primary
t=1.0 four-stage slice for earlier spectrum/classification assembly while
preserving the full 12-shard grid. The launcher treats current JSONL row
counts plus summary outputs as authoritative for completion; the plan CSV's
`completed` column is refreshed for observability but is not enough by itself
to skip a shard. `REFRESH_FINALIZATION` defaults to `1`, so
each loop also regenerates the guarded finalization shell script and readiness
summary from the current SGLang plan. `REFRESH_INTEGRITY_AUDIT` defaults to
`1`, so each loop also audits all existing SGLang generation caches for
duplicate generation keys, malformed JSON, metadata drift, short fixed-budget
rows, empty generations, malformed feature JSON, and incomplete or overfull
prompt groups. If the finalization summary reports enabled commands, the
supervisor now requires the refreshed integrity audit to exist and have every
cache marked passing before it runs the generated finalization script.
`REFRESH_COMPLETION_AUDIT` also defaults to `1`, so each loop writes a
machine-readable completion snapshot after the finalization refresh.

For SmolLM3 three-temperature primary `slop_lexicon` compounding, the current
cached-counter artifacts are:

- Counter cache:
  `artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_slop_lexicon_cluster_counts_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3.jsonl`
- CI table:
  `artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_slop_lexicon_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3.csv`
- Realized-AF plot:
  `artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_slop_lexicon_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3_realized_af.svg`

Regenerate that CI table from the counter cache with:

```bash
uv run slop-analyze-phase2-compounding \
  --counter-cache-input artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_slop_lexicon_cluster_counts_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3.jsonl \
  --feature slop_lexicon \
  --window-tokens 32 \
  --propensity-grid artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_512prompt_slop_neutral_rule3.csv \
  --output artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_slop_lexicon_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3.csv \
  --summary-output artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_slop_lexicon_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3_summary.md \
  --plot-output artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_slop_lexicon_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3_realized_af.svg \
  --bootstrap-samples 1000 \
  --bootstrap-seed 1729 \
  --wandb-mode disabled
```

Reduced OLMo analysis outputs:

- Main style-signature grid:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_main_generation_stage_grid_1024prompt_2comp_t07_sglang.csv`
- Temperature grid:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_generation_stage_grid_512prompt_1comp_3temp_sglang.csv`
- Long-output grid:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_long_generation_stage_grid_256prompt_2comp_t07_1024_sglang.csv`
- Long-output compounding analysis:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_long_compounding_analysis_256prompt_2comp_t07_1024_sglang.csv`
- Long-output compounding realized-AF plot:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_long_compounding_analysis_256prompt_2comp_t07_1024_sglang_realized_af.svg`
- Integrated reduced OLMo spectrum:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv`
- Integrated reduced OLMo classifier:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_feature_classification_t07_long1024.csv`
- Reduced OLMo-vs-SmolLM3 production comparison:
  `artifacts/phase3/analysis/olmo3_reduced_sglang_vs_smollm3_no_think_production_summary.md`

## Launch Policy

Use the guarded launcher. It is dry-run by default, refuses completed shards
unless told otherwise, and refuses expensive shards unless the configured cap
allows them. Start with the OLMo main panel, because it directly supports the
base/SFT/DPO/final style-signature comparison.

Dry-run the first main-panel shard:

```bash
uv run slop-launch-phase2-generation-shard \
  --plan artifacts/phase3/analysis/olmo3_phase3_reduced_main_signature_1024prompt_2comp_t07_512_sglang.csv \
  --stage base \
  --temperature 0.7 \
  --max-estimated-a100-hours 3 \
  --selection-output artifacts/phase3/analysis/olmo3_phase3_reduced_main_signature_base_t07_sglang_shard.json
```

Launch it detached after inspecting the selection payload:

```bash
uv run slop-launch-phase2-generation-shard \
  --plan artifacts/phase3/analysis/olmo3_phase3_reduced_main_signature_1024prompt_2comp_t07_512_sglang.csv \
  --stage base \
  --temperature 0.7 \
  --max-estimated-a100-hours 3 \
  --selection-output artifacts/phase3/analysis/olmo3_phase3_reduced_main_signature_base_t07_sglang_shard.json \
  --execute \
  --detach
```

Regenerate the SGLang plans from their source reduced Torch plans with
`slop-materialize-sglang-generation-plan` if the SGLang sidecar path, context
length, or fixed-generation contract changes.

The SmolLM3 512-prompt shards are small enough to launch one at a time under a
4-hour cap. The completed base/t0 selection is recorded here:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_next_shard.json`

The completed base/t0.7 launch selection is recorded here:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t07_shard.json`

The completed base/t1 launch selection is recorded here:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t1_shard.json`

The completed SFT/t0 launch selection is recorded here:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t0_shard.json`

The completed SFT/t0.7 launch selection is recorded here:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t07_shard.json`

The completed SFT/t1 launch selection is recorded here:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t1_shard.json`

The completed APO/DPO t0 launch selection is recorded here:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t0_shard.json`

The completed APO/DPO t0.7 launch selection is recorded here:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t07_shard.json`

The completed APO/DPO t1 launch selection is recorded here:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t1_shard.json`

The completed final/RLVR t0 launch selection is recorded here:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t0_shard.json`

The completed final/RLVR t0.7 launch selection is recorded here:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t07_shard.json`

The completed final/RLVR t1 launch selection is recorded here:

- `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t1_shard.json`

## SmolLM3 Launch Sequence

Dry-run a shard selection:

```bash
uv run slop-launch-phase2-generation-shard \
  --plan artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv \
  --stage base \
  --temperature 0.0 \
  --max-estimated-a100-hours 4 \
  --selection-output artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_next_shard.json
```

Launch the selected shard intentionally:

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

Current base/t0 launch files:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_next_shard.json`
- Detached log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_next_shard.log`

Completed base/t0.7 launch files:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t07_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t07_shard_status.csv`
- Detached log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t07_shard.log`

Completed base/t1 launch files:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t1_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t1_shard_status.csv`
- Detached log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t1_shard.log`

Completed SFT/t0 launch files:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t0_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t0_shard_status.csv`
- Detached log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t0_shard.log`

Completed SFT/t0.7 launch files:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t07_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t07_shard_status.csv`
- Detached log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t07_shard.log`

Completed SFT/t1 launch files:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t1_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t1_shard_status.csv`
- Detached log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t1_shard.log`

Completed APO/DPO t0 launch files:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t0_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t0_shard_status.csv`
- Detached log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t0_shard.log`

Completed APO/DPO t0.7 launch files:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t07_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t07_shard_status.csv`
- Detached log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t07_shard.log`

Completed APO/DPO t1 launch files:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t1_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t1_shard_status.csv`
- Detached log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t1_shard.log`

Completed final/RLVR t0 launch files:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t0_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t0_shard_status.csv`
- Detached log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t0_shard.log`

Completed final/RLVR t0.7 launch files:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t07_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t07_shard_status.csv`
- Detached log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t07_shard.log`

Completed final/RLVR t1 launch files:

- Selection payload:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t1_shard.json`
- Status CSV:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t1_shard_status.csv`
- Detached log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t1_shard.log`

Monitor it:

```bash
uv run slop-phase2-generation-status \
  --selection artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t1_shard.json \
  --output artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t1_shard_status.csv
```

Audit explicit no_think mode markup on the generated cache:

```bash
uv run slop-audit-no-think-generations \
  --generation-cache artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_base_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t0_batched256.jsonl \
  --output artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_base_t0_no_think_audit.csv \
  --summary-output artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_base_t0_no_think_audit.md
```

Run this audit after each SmolLM3 shard completes. It flags explicit
thinking/reasoning markup only; ordinary words such as `think` are not treated
as failures.

After a shard completes, regenerate the plan so completed/existing counts are
refreshed:

```bash
uv run slop-plan-phase2-generation \
  --prompt-package artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_512.jsonl \
  --output-dir artifacts/phase2/generations \
  --artifact-prefix smollm3_no_think_promptpkg512_chat_production_grid \
  --package-tag smoltalk2-everyday-no-think-promptpkg512-chat \
  --sample-size 512 \
  --completions-per-prompt 8 \
  --temperature 0.0 \
  --temperature 0.7 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-new-tokens 1024 \
  --generation-batch-size 256 \
  --max-prompt-tokens 1024 \
  --apply-chat-template \
  --chat-template-kwargs-json '{"enable_thinking": false}' \
  --missing-chat-template plain \
  --stage base=HuggingFaceTB/SmolLM3-3B-Base \
  --stage sft=HuggingFaceTB/SmolLM3-3B-checkpoints@it-SFT \
  --stage dpo=HuggingFaceTB/SmolLM3-3B-checkpoints@it-soup-APO \
  --stage final=HuggingFaceTB/SmolLM3-3B \
  --output artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv \
  --summary-output artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.md \
  --wandb-mode disabled
```

Repeat for each stage/temperature until all 12 shards are complete.

## OLMo Launch Gate

The OLMo full grid is no longer the active Phase 3 route. The active route is
the completed reduced SGLang plan in
`artifacts/phase3/analysis/phase3_reduced_sglang_generation_plan_status_summary.md`:
paired main style signatures, a reduced temperature panel, and a reduced
long-output compounding panel. Use SGLang for any additional pure-generation
extensions, and keep the unit of work to paired prompt panels that can finish
in hours rather than week-scale exhaustive grids.

The legacy full-grid continuation command is retained only for audit or for an
explicit opt-in compute decision:

```bash
uv run slop-continue-phase2-generation-plan \
  --plan artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched64_outputs_batched32_resume.csv \
  --selection-dir artifacts/phase2/analysis \
  --selection-prefix olmo3_generation_plan_full_5000prompt_8comp_3temp_batched64_no_compile \
  --max-estimated-a100-hours 100 \
  --decision-output artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched64_outputs_batched32_resume_continuation_decision.json
```

Launching requires an explicit compute decision because each shard is still
conservatively estimated at about `94.81` A100-hours. Do not add `--execute`
as part of the default Phase 3 route.

## Optional OLMo Think/RL-Zero Stretch Plan

The optional `EXPERIMENTS.md` Instruct-vs-Think-vs-RL-Zero stretch comparison
is complete as a reduced SGLang plan:

- Source plan:
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_256prompt_2comp_t07_1024.csv`
- SGLang plan:
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_256prompt_2comp_t07_1024_sglang.csv`
- SGLang summary:
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_256prompt_2comp_t07_1024_sglang.md`
- CI-enriched generation grid:
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_generation_grid_256prompt_2comp_t07_1024_sglang.csv`
- Path comparison:
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_comparison_256prompt_2comp_t07_1024_sglang.csv`
- Comparison summary:
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_comparison_256prompt_2comp_t07_1024_sglang_summary.md`

Scope:

- 10 stages: base, Instruct final, Think SFT, Think DPO, Think final, and
  RL-Zero General/IF/Math/Code/Mix.
- 256 prompts x 2 completions per stage at temperature `0.7`, top-p `0.95`,
  max 1,024 new tokens.
- 5,120/5,120 generated records complete.
- Source-plan estimate: `4.09` A100-hours total before SGLang speedup.
- Feature extraction mode: `final_answer`. The command plan passes
  `--feature-text-mode final_answer`, which keeps raw generations in JSONL but
  excludes explicit `<think>`/`<reasoning>` and fenced thinking/reasoning
  traces from Tier-1 feature counts and per-1k denominators.

All ten JSONL caches were verified to contain only `feature_text_mode=
final_answer` rows. For `slop_lexicon`, Instruct final is the highest measured
stage at `0.807` per 1k final-answer tokens. Think final is lower by `0.384`
per 1k, and the RL-Zero mean is lower than Instruct final by `0.244` per 1k.
Instruct final is also the maximum for all tracked Tier-1 output features in
this reduced stretch panel.

Runtime note: `allenai/Olmo-3-7B-RL-Zero-Mix` shipped a local config with
`model_type=olmo2-retrofit`, which SGLang with Transformers `4.57.6` did not
recognize. The completed local run patched the downloaded Hugging Face cache
config to the same loader identifiers used by the other RL-Zero checkpoints:
`architectures=["Olmo3ForCausalLM"]`, `model_type="olmo3"`,
`dtype="bfloat16"`, and `use_cache=true`.

To reproduce the assembled stretch generation grid with document-cluster
bootstrap CIs:

```bash
uv run slop-assemble-phase2-generation-grid \
  --generation-summary base=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_base_promptpkg256_free_run_256prompt_2comp_t07_batched32_summary.csv \
  --generation-summary instruct_final=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_instruct_final_promptpkg256_free_run_256prompt_2comp_t07_batched32_summary.csv \
  --generation-summary think_sft=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_think_sft_promptpkg256_free_run_256prompt_2comp_t07_batched32_summary.csv \
  --generation-summary think_dpo=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_think_dpo_promptpkg256_free_run_256prompt_2comp_t07_batched32_summary.csv \
  --generation-summary think_final=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_think_final_promptpkg256_free_run_256prompt_2comp_t07_batched32_summary.csv \
  --generation-summary rl_zero_general=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_rl_zero_general_promptpkg256_free_run_256prompt_2comp_t07_batched32_summary.csv \
  --generation-summary rl_zero_if=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_rl_zero_if_promptpkg256_free_run_256prompt_2comp_t07_batched32_summary.csv \
  --generation-summary rl_zero_math=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_rl_zero_math_promptpkg256_free_run_256prompt_2comp_t07_batched32_summary.csv \
  --generation-summary rl_zero_code=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_rl_zero_code_promptpkg256_free_run_256prompt_2comp_t07_batched32_summary.csv \
  --generation-summary rl_zero_mix=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_rl_zero_mix_promptpkg256_free_run_256prompt_2comp_t07_batched32_summary.csv \
  --generation-cache base=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_base_promptpkg256_free_run_256prompt_2comp_t07_batched32.jsonl \
  --generation-cache instruct_final=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_instruct_final_promptpkg256_free_run_256prompt_2comp_t07_batched32.jsonl \
  --generation-cache think_sft=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_think_sft_promptpkg256_free_run_256prompt_2comp_t07_batched32.jsonl \
  --generation-cache think_dpo=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_think_dpo_promptpkg256_free_run_256prompt_2comp_t07_batched32.jsonl \
  --generation-cache think_final=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_think_final_promptpkg256_free_run_256prompt_2comp_t07_batched32.jsonl \
  --generation-cache rl_zero_general=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_rl_zero_general_promptpkg256_free_run_256prompt_2comp_t07_batched32.jsonl \
  --generation-cache rl_zero_if=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_rl_zero_if_promptpkg256_free_run_256prompt_2comp_t07_batched32.jsonl \
  --generation-cache rl_zero_math=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_rl_zero_math_promptpkg256_free_run_256prompt_2comp_t07_batched32.jsonl \
  --generation-cache rl_zero_code=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_rl_zero_code_promptpkg256_free_run_256prompt_2comp_t07_batched32.jsonl \
  --generation-cache rl_zero_mix=artifacts/phase2/generations/olmo3_phase3_stretch_think_rlzero_rl_zero_mix_promptpkg256_free_run_256prompt_2comp_t07_batched32.jsonl \
  --bootstrap-samples 2000 \
  --bootstrap-seed 1729 \
  --primary-feature slop_lexicon \
  --output artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_generation_grid_256prompt_2comp_t07_1024_sglang.csv \
  --comparison-output artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_generation_primary_feature_comparison_256prompt_2comp_t07_1024_sglang.csv \
  --summary-output artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_generation_grid_256prompt_2comp_t07_1024_sglang_summary.md \
  --wandb-mode disabled
```

Then assemble the path comparison:

```bash
uv run slop-assemble-olmo-stretch-comparison \
  --generation-grid artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_generation_grid_256prompt_2comp_t07_1024_sglang.csv \
  --primary-feature slop_lexicon \
  --output artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_comparison_256prompt_2comp_t07_1024_sglang.csv \
  --feature-summary-output artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_feature_summary_256prompt_2comp_t07_1024_sglang.csv \
  --summary-output artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_comparison_256prompt_2comp_t07_1024_sglang_summary.md \
  --wandb-mode disabled
```

## Rebuild After SmolLM3 Generation Completion

The generation-grid assembler is stage-keyed, so rebuild one grid per
temperature. Example for the SmolLM3 `t=1.0` production-grid file names:

```bash
uv run slop-assemble-phase2-generation-grid \
  --generation-summary base=artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_base_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256_summary.csv \
  --generation-summary sft=artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_sft_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256_summary.csv \
  --generation-summary dpo=artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_dpo_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256_summary.csv \
  --generation-summary final=artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_final_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256_summary.csv \
  --generation-cache base=artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_base_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl \
  --generation-cache sft=artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_sft_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl \
  --generation-cache dpo=artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_dpo_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl \
  --generation-cache final=artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_final_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl \
  --checkpoint-label base=Base \
  --checkpoint-label sft=SFT \
  --checkpoint-label dpo=APO \
  --checkpoint-label final=Final/RLVR \
  --primary-feature slop_lexicon \
  --bootstrap-samples 1000 \
  --bootstrap-seed 1729 \
  --output artifacts/phase3/analysis/smollm3_no_think_generation_stage_grid_512prompt_8comp_t1_chat_production.csv \
  --comparison-output artifacts/phase3/analysis/smollm3_no_think_generation_stage_comparison_512prompt_8comp_t1_chat_production.csv \
  --summary-output artifacts/phase3/analysis/smollm3_no_think_generation_stage_grid_512prompt_8comp_t1_chat_production_summary.md \
  --wandb-mode disabled
```

Then rebuild compounding against the existing teacher-forced grid:

```bash
uv run slop-analyze-phase2-compounding \
  --generation-cache base=artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_base_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl \
  --generation-cache sft=artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_sft_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl \
  --generation-cache dpo=artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_dpo_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl \
  --generation-cache final=artifacts/phase2/generations/smollm3_no_think_promptpkg512_chat_production_grid_final_smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl \
  --feature contrastive_negation \
  --feature rule_of_three_approx \
  --feature slop_lexicon \
  --feature stock_openers \
  --feature stock_closers \
  --feature stock_openers_closers \
  --window-tokens 32 \
  --propensity-grid artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_512prompt_slop_neutral_rule3.csv \
  --output artifacts/phase3/analysis/smollm3_no_think_generation_compounding_512prompt_8comp_t1_chat_production_tf_slop_neutral_rule3.csv \
  --summary-output artifacts/phase3/analysis/smollm3_no_think_generation_compounding_512prompt_8comp_t1_chat_production_tf_slop_neutral_rule3_summary.md \
  --plot-output artifacts/phase3/analysis/smollm3_no_think_generation_compounding_512prompt_8comp_t1_chat_production_tf_slop_neutral_rule3_realized_af.svg \
  --primary-feature slop_lexicon \
  --bootstrap-samples 1000 \
  --bootstrap-seed 1729 \
  --wandb-mode disabled
```

Rebuild the SmolLM3 spectrum and classification using the regenerated
generation/compounding artifacts plus the existing data-rate and pretraining
baseline layers. Use the current preferred `baselines_data_rates` commands as
the template, replacing only the generation-grid and compounding paths.

## Register And Style-Signature Boundary

Do not treat generated-output full pybiber as part of the active Phase 3 route.
The paper-facing full-pybiber result is the Phase 1 corpus-side register
analysis over retained pretrain/SFT/DPO samples. Older generated-output
register-proxy/style-signature artifacts are archived diagnostics only; they
should not be rebuilt or promoted as full Biber/register evidence for the
current paper claim surface.

If a future addendum decides that generated-output full pybiber is worth the
additional runtime, define it as a separate addendum with explicit generation
cache inputs, corpus comparison targets, uncertainty intervals, and manuscript
claim boundaries before running `uv run slop-pybiber-full` over generation
caches.

## Completion Gate

## Compute Strategy Going Forward

Use SGLang for pure generation tasks. The repo-local SGLang sidecar path is
the active route for OLMo reduced generation panels, optional path comparisons,
and any original-scope OLMo full-grid recovery. Use fixed-budget generation
with `--ignore-eos` when the analysis needs comparable generated-token
exposure. Do not return to Torch/Transformers batch-size recovery for pure
generation unless SGLang is blocked.

The SGLang harness supports `--resume` and `--prompt-batch-size`; full-grid
plans should use prompt chunks so a long shard writes incremental JSONL rows
rather than waiting for a monolithic 5,000-prompt generate call.

Do not use SGLang for teacher-forced AF scoring. Those jobs require
probability mass over controlled candidate initiator sequences, so they remain
on `slop-teacher-forced-propensity`. Keep them bounded by denominator audits:
run sparse features only when a prompt/reference package shows nonzero
opportunities and references, and report low-reference measurements as
coverage-caveated addenda.

The current sparse SmolLM3 teacher-forced addendum follows that policy:

- prompt package:
  `artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_2258.jsonl`
- denominator support:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg2258_denominator_support.csv`
- completed plan:
  `artifacts/phase3/analysis/smollm3_no_think_propensity_plan_2258prompt_sparse_supported.csv`
- assembled sparse stage grid:
  `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_2258prompt_sparse_supported.csv`
- sparse stage effects:
  `artifacts/phase3/analysis/smollm3_no_think_teacher_forced_stage_effects_2258prompt_sparse_supported.csv`
- sparse spectrum/classifier:
  `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_2258prompt_sparse_supported.csv`
  and
  `artifacts/phase3/analysis/smollm3_no_think_feature_classification_2258prompt_sparse_supported.csv`

The full sparse package gives nonzero support for `contrastive_negation`,
`stock_closers`, and `stock_openers_closers`, but `stock_openers` remains
unsupported with zero target references. Treat the resulting sparse
teacher-forced AFs as addendum evidence because reference counts are only 2 or
3 per measured sparse feature.

Current full OLMo SGLang plan:

```bash
uv run slop-materialize-sglang-generation-plan \
  --input-plan artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp.csv \
  --output artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.csv \
  --summary-output artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.md \
  --wandb-mode disabled \
  --context-length 4096 \
  --prompt-batch-size 128 \
  --mem-fraction-static 0.85 \
  --attention-backend triton \
  --disable-cuda-graph \
  --ignore-eos \
  --feature-text-mode raw
```

Continue one full-grid SGLang shard at a time:

```bash
uv run slop-continue-phase2-generation-plan \
  --plan artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.csv \
  --selection-dir artifacts/phase3/analysis \
  --selection-prefix olmo3_full_sglang_chunk128_generation_shard \
  --max-estimated-a100-hours 40 \
  --execute \
  --decision-output artifacts/phase3/analysis/olmo3_full_sglang_chunk128_generation_next_decision.json
```

If a shard has partial output and no active process, rerun the same command
with `--allow-partial-retry`; the launcher appends `--resume`, and the SGLang
harness skips already written source/prompt/completion/temperature/top-p rows.

Current launched full-grid shard:

- selection:
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_generation_shard_base_t0p0.json`
- log:
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_generation_shard_base_t0p0.log`
- output:
  `artifacts/phase2/generations/olmo3_base_promptpkg5000_free_run_5000prompt_8comp_t0_batched1024.jsonl`

Do not mark Phase 3 complete until the completion audit proves all in-scope
requirements. At minimum, for the non-stretch core, the audit must show:

- production or explicitly downscoped generation grids complete;
- compounding rebuilt from those grids;
- temperature-dependent realized-AF decompositions rebuilt from the completed
  three-temperature grids where teacher-forced support exists;
- Pybiber/style-signature outputs rebuilt from those grids;
- amplification spectrum and classifications regenerated at the chosen scope;
- cross-ladder AF correlation regenerated at the chosen scope;
- CI/FDR caveats updated to match the regenerated artifacts.

The full-grid post-generation finalization commands are materialized here:

- Script:
  `artifacts/phase3/analysis/olmo3_full_sglang_5000prompt_8comp_3temp_finalize.sh`
- Readiness summary:
  `artifacts/phase3/analysis/olmo3_full_sglang_5000prompt_8comp_3temp_finalize.md`
- Machine-readable completion audit snapshot:
  `artifacts/phase3/analysis/phase3_completion_audit_snapshot.md`
- Existing-cache integrity audit:
  `artifacts/phase3/analysis/olmo3_full_sglang_existing_cache_integrity.md`

The completion audit snapshot checks artifact existence, whether the
finalization summary still has enabled commands waiting to run, and whether
the SGLang existing-cache integrity audit is present and passing. Required
CSV/JSONL artifacts must contain data rows; empty or header-only files do not
satisfy completion gates.

Refresh the SGLang cache integrity audit manually with:

```bash
uv run slop-audit-sglang-generation-cache \
  --generation-plan artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.csv \
  --audit-existing-only \
  --require-max-new-tokens \
  --output artifacts/phase3/analysis/olmo3_full_sglang_existing_cache_integrity.csv \
  --summary-output artifacts/phase3/analysis/olmo3_full_sglang_existing_cache_integrity.md
```

The audit uses plan metadata for expected model, temperature, top-p,
completion-index range, completions per prompt, and max-new-token budget. As
of the eighth base/t1 checkpoint, the existing-cache audit passes `2/2` caches:
base/t0 has `18432` unique generation keys and `2304` prompt groups; base/t1
has `8192` unique generation keys and `1024` prompt groups. Both have zero bad
prompt groups and zero duplicate, malformed, metadata-drift, short-token,
empty-text, or bad-feature-JSON rows.

Regenerate the finalization plan after new generation chunks or completed
shards appear:

```bash
uv run slop-materialize-phase3-full-sglang-finalize-plan \
  --generation-plan artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.csv \
  --output-script artifacts/phase3/analysis/olmo3_full_sglang_5000prompt_8comp_3temp_finalize.sh \
  --summary-output artifacts/phase3/analysis/olmo3_full_sglang_5000prompt_8comp_3temp_finalize.md \
  --feature-rate artifacts/stage1/census/olmo3_dolci_sft_dpo_10k_feature_rates.csv \
  --feature-rate artifacts/stage1/census/olmo3_dolma3_20k_scan_feature_rates.csv \
  --propensity-grid artifacts/phase2/analysis/olmo3_promptpkg1024_slop_neutral_common_normalized_stage_grid.csv \
  --propensity-grid artifacts/phase2/analysis/olmo3_promptpkg5000_rule_of_three_completion_stage_grid.csv \
  --propensity-grid artifacts/phase2/analysis/olmo3_promptpkg5000_stock_openers_stage_grid.csv \
  --propensity-grid artifacts/phase2/analysis/olmo3_promptpkg5000_stock_closers_stage_grid.csv \
  --propensity-grid artifacts/phase2/analysis/olmo3_promptpkg5000_stock_openers_closers_stage_grid.csv \
  --compounding-propensity-grid artifacts/phase2/analysis/olmo3_promptpkg1024_slop_neutral_common_normalized_stage_grid.csv \
  --denominator-support artifacts/phase2/analysis/olmo3_promptpkg5000_denominator_support.csv \
  --denominator-support artifacts/phase2/analysis/olmo3_promptpkg5000_rule_of_three_denominator_support_v2.csv \
  --preference-analysis artifacts/stage1/census/olmo3_dolci_dpo_10k_pair_analysis.csv \
  --teacher-forced-stage-effects artifacts/phase3/analysis/olmo3_phase3_teacher_forced_stage_effects_t1.csv \
  --right-spectrum artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_production.csv \
  --bootstrap-samples 1000 \
  --bootstrap-seed 1729 \
  --wandb-mode disabled
```

The script comments out incomplete temperature slices and enables assembly
commands only when all four stage shards for a temperature have complete JSONL
and summary outputs. The readiness summary includes both temperature-level and
per-stage tables, so the exact missing stage/temperature cells are visible
before running downstream assembly. Commands whose declared downstream outputs
already exist are skipped, which makes the generated script safe to rerun as a
pending-work script after partial finalization. The current readiness summary
records `base/t0` progress as `18432/160000` rows for the temperature-0 slice
and `base/t1` progress as `8192/160000` for the primary temperature-1 slice.
The paused full-grid supervisor can refresh this artifact automatically when
re-enabled. When the refreshed
summary reports one or more enabled commands, `RUN_FINALIZATION_WHEN_READY=1`
causes the supervisor to run the pending-work finalization script only after
the refreshed SGLang cache-integrity CSV reports all existing caches passing.
It then refreshes the finalization summary again and the completion audit.
Manual regeneration is still useful after local command edits or before publishing a
status snapshot.

Refresh the completion audit snapshot manually with:

```bash
scripts/phase3_refresh_completion_audit.sh
```

The helper accepts `PLAN`, `FINALIZE_SUMMARY`, `COMPLETION_AUDIT`, and
`COMPLETION_AUDIT_SUMMARY` environment overrides; it also accepts
`INTEGRITY_AUDIT` for the SGLang cache-integrity CSV. The default paths match
the paused full-grid addendum artifacts, but `scripts/phase3_refresh_completion_audit.sh`
now treats those rows as optional. The current snapshot reports `30/30`
required production-scope rows complete and `4/11` optional/addendum rows
complete. The incomplete optional rows are the original full OLMo generation
grid, primary t=1 four-stage slice, and five full-grid primary downstream
artifacts.
