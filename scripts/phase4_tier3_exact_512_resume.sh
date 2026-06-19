#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512"
mkdir -p "$OUT_DIR"

FEATURES=(
  phase4_ig_conversation_greeting
  phase4_ig_process_framing
  phase4_ig_additive_transition
  phase4_ig_prescriptive_instruction
  phase4_ig_followup_offer
  phase4_ig_response_constraint
  phase4_ig_code_boilerplate
  phase4_ig_quantity_time_recipe
  phase4_ig_benefit_intensifier
  phase4_ig_careful_reasoning
  neutral_common_controls
)

STAGES=(base sft dpo final)
MODELS=(
  allenai/Olmo-3-1025-7B
  allenai/Olmo-3-7B-Instruct-SFT
  allenai/Olmo-3-7B-Instruct-DPO
  allenai/Olmo-3-7B-Instruct
)

run_stage() {
  local stage="$1"
  local model="$2"
  local output="$OUT_DIR/olmo3_${stage}_phase4_tier3_512_exact_sequence_opportunities.csv"
  local summary="$OUT_DIR/olmo3_${stage}_phase4_tier3_512_exact_sequence_summary.csv"

  if [[ -s "$output" && -s "$summary" ]]; then
    echo "Skipping completed stage: $stage"
    return
  fi

  args=(
    uv run slop-teacher-forced-propensity
    --model "$model"
    --input artifacts/phase2/prompts/olmo3_dolci_sft_phase2_prompt_package_512.jsonl
    --sample-size 512
    --seed 1729
    --max-opportunities 512
    --max-token-start-opportunities 32
    --max-prefix-tokens 384
    --fixed-prefix-tokens 192
    --mass-mode sequence
    --no-sequence-cache
    --opportunity-batch-size 8
    --dtype bfloat16
    --device cuda
    --no-torch-compile
    --bootstrap-samples 500
    --normalization-feature neutral_common_controls
    --output "$output"
    --summary-output "$summary"
    --wandb-run-name "phase4-tier3-${stage}-512-exact-sequence"
    --wandb-mode disabled
  )
  for feature in "${FEATURES[@]}"; do
    args+=(--feature "$feature")
  done
  "${args[@]}"
}

for i in "${!STAGES[@]}"; do
  run_stage "${STAGES[$i]}" "${MODELS[$i]}"
done

uv run slop-assemble-phase2-grid \
  --propensity-summary base="$OUT_DIR/olmo3_base_phase4_tier3_512_exact_sequence_summary.csv" \
  --propensity-summary sft="$OUT_DIR/olmo3_sft_phase4_tier3_512_exact_sequence_summary.csv" \
  --propensity-summary dpo="$OUT_DIR/olmo3_dpo_phase4_tier3_512_exact_sequence_summary.csv" \
  --propensity-summary final="$OUT_DIR/olmo3_final_phase4_tier3_512_exact_sequence_summary.csv" \
  --checkpoint-label base=base \
  --checkpoint-label sft=sft \
  --checkpoint-label dpo=dpo \
  --checkpoint-label final=final \
  --primary-feature phase4_ig_process_framing \
  --normalization-feature neutral_common_controls \
  --comparison-metric normalized_af \
  --output "$OUT_DIR/olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv" \
  --comparison-output "$OUT_DIR/olmo3_phase4_tier3_512_exact_sequence_primary_comparison.csv" \
  --summary-output "$OUT_DIR/olmo3_phase4_tier3_512_exact_sequence_stage_grid_summary.md" \
  --wandb-mode disabled

uv run python -m slop_sftdiv.cli.analyze_phase3_teacher_forced_effects \
  --opportunity-cache base="$OUT_DIR/olmo3_base_phase4_tier3_512_exact_sequence_opportunities.csv" \
  --opportunity-cache sft="$OUT_DIR/olmo3_sft_phase4_tier3_512_exact_sequence_opportunities.csv" \
  --opportunity-cache dpo="$OUT_DIR/olmo3_dpo_phase4_tier3_512_exact_sequence_opportunities.csv" \
  --opportunity-cache final="$OUT_DIR/olmo3_final_phase4_tier3_512_exact_sequence_opportunities.csv" \
  --feature phase4_ig_conversation_greeting \
  --feature phase4_ig_process_framing \
  --feature phase4_ig_additive_transition \
  --feature phase4_ig_prescriptive_instruction \
  --feature phase4_ig_followup_offer \
  --feature phase4_ig_response_constraint \
  --feature phase4_ig_code_boilerplate \
  --feature phase4_ig_quantity_time_recipe \
  --feature phase4_ig_benefit_intensifier \
  --feature phase4_ig_careful_reasoning \
  --feature neutral_common_controls \
  --output "$OUT_DIR/olmo3_phase4_tier3_512_exact_sequence_stage_effects.csv" \
  --summary-output "$OUT_DIR/olmo3_phase4_tier3_512_exact_sequence_stage_effects_summary.md" \
  --wandb-mode disabled
