# Phase 2 Post-Shard Analysis Runbook

Date: 2026-06-14

This runbook covers the analysis to run after the active OLMo3 DPO
1,024-prompt target-shape generation shard completes.

## Active Shard

- Selection:
  `artifacts/phase2/analysis/olmo3_generation_launch_selection_dpo_1024prompt_8comp_t1_batched16.json`
- Generations:
  `artifacts/phase2/generations/olmo3_dpo_promptpkg5000_free_run_1024prompt_8comp_t1_batched16.jsonl`
- Summary:
  `artifacts/phase2/generations/olmo3_dpo_promptpkg5000_free_run_1024prompt_8comp_t1_batched16_summary.csv`
- W&B run: `p4v56vda`
- Shape: DPO, 1,024 prompts, 8 completions per prompt, temperature `1.0`,
  top-p `0.95`, 1,024 max new tokens.

The process was launched before the streaming-writer fix, so the JSONL and
summary may not appear until the process completes. Use
`slop-phase2-generation-status` to monitor log-derived prompt progress.

## CUDA Environment Recovery

After a shell or host reset, restore the CUDA wheel library path before running
Phase 2 GPU commands or status helpers:

```bash
source scripts/phase2_cuda_env.sh
```

The helper reconstructs `LD_LIBRARY_PATH` from the NVIDIA CUDA 12.8 libraries
installed in `.venv`, sets `UV_CACHE_DIR` to the repository cache if unset, and
keeps `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless already set.

## Completion Check

```bash
source scripts/phase2_cuda_env.sh
uv run slop-phase2-generation-status \
  --selection artifacts/phase2/analysis/olmo3_generation_launch_selection_dpo_1024prompt_8comp_t1_batched16.json \
  --output artifacts/phase2/analysis/olmo3_generation_launch_status_dpo_1024prompt_8comp_t1_batched16.csv
```

Expected completion evidence:

- status reports `completed=True`;
- generation JSONL has `8192` non-empty rows;
- summary CSV exists and includes the retained Tier-1 features.

## DPO 512-vs-1024 Scale Comparison

Run this first. It checks whether the larger DPO cell materially changes the
free-running feature-rate read before spending more A100 time on matched
1,024-prompt base/SFT/final shards.

The combined runner can perform the scale comparison and compounding join after
the shard completes:

```bash
source scripts/phase2_cuda_env.sh
uv run slop-run-phase2-post-shard-analysis \
  --selection artifacts/phase2/analysis/olmo3_generation_launch_selection_dpo_1024prompt_8comp_t1_batched16.json \
  --wait \
  --poll-seconds 300 \
  --timeout-seconds 43200 \
  --execute \
  --wandb-mode online
```

The lower-level commands are kept below for transparency and manual reruns.

```bash
source scripts/phase2_cuda_env.sh
uv run slop-assemble-phase2-generation-grid \
  --generation-summary a_dpo512=artifacts/phase2/generations/olmo3_dpo_promptpkg5000_free_run_512prompt_8comp_t1_batched1024_summary.csv \
  --generation-summary b_dpo1024=artifacts/phase2/generations/olmo3_dpo_promptpkg5000_free_run_1024prompt_8comp_t1_batched16_summary.csv \
  --checkpoint-label "a_dpo512=DPO 512 prompts x 8" \
  --checkpoint-label "b_dpo1024=DPO 1024 prompts x 8" \
  --primary-feature slop_lexicon \
  --output artifacts/phase2/analysis/olmo3_dpo_generation_scale_grid_target_shape_512_vs_1024prompt_8comp_t1_1024.csv \
  --comparison-output artifacts/phase2/analysis/olmo3_dpo_generation_scale_comparison_target_shape_512_vs_1024prompt_8comp_t1_1024.csv \
  --summary-output artifacts/phase2/analysis/olmo3_dpo_generation_scale_grid_target_shape_512_vs_1024prompt_8comp_t1_1024_summary.md \
  --wandb-run-name stage2-phase2-olmo3-dpo-generation-scale-512-vs-1024-t1 \
  --wandb-mode online
```

## DPO 1,024-Prompt Compounding Join

Run this after the scale comparison. The propensity grid is the existing
1,024-prompt teacher-forced `slop_lexicon` vs `neutral_common_controls` grid.

```bash
source scripts/phase2_cuda_env.sh
uv run slop-analyze-phase2-compounding \
  --generation-cache dpo=artifacts/phase2/generations/olmo3_dpo_promptpkg5000_free_run_1024prompt_8comp_t1_batched16.jsonl \
  --propensity-grid artifacts/phase2/analysis/olmo3_promptpkg1024_slop_neutral_common_normalized_stage_grid.csv \
  --feature contrastive_negation \
  --feature rule_of_three_approx \
  --feature slop_lexicon \
  --feature stock_openers \
  --feature stock_closers \
  --feature stock_openers_closers \
  --window-tokens 32 \
  --primary-feature slop_lexicon \
  --output artifacts/phase2/analysis/olmo3_dpo_generation_compounding_target_shape_1024prompt_8comp_t1_1024_tf1024.csv \
  --summary-output artifacts/phase2/analysis/olmo3_dpo_generation_compounding_target_shape_1024prompt_8comp_t1_1024_tf1024_summary.md \
  --plot-output artifacts/phase2/analysis/olmo3_dpo_generation_compounding_target_shape_1024prompt_8comp_t1_1024_tf1024_realized_af.svg \
  --wandb-run-name stage2-phase2-olmo3-dpo-generation-compounding-1024prompt-8comp-t1-tf1024 \
  --wandb-mode online
```

## Decision Rule

Analyze before launching more GPU work.

- If the 1,024-prompt DPO rates are close to the existing 512-prompt DPO
  target-shape rates, update the bounded Phase 2 notes and avoid launching
  matched base/SFT/final 1,024-prompt shards.
- If the larger DPO cell materially changes the DPO free-running or compounding
  read, plan matched 1,024-prompt base/SFT/final shards with
  `slop-plan-phase2-generation`, then launch one guarded shard at a time with
  `slop-launch-phase2-generation-shard`.
- Do not launch the full 5,000-prompt x 8-completion x 3-temperature OLMo grid
  on the current Torch backend without a new explicit compute decision.
