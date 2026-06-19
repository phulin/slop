#!/usr/bin/env bash
set -euo pipefail

SOURCE_PLAN=${SOURCE_PLAN:-artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp.csv}
PLAN=${PLAN:-artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.csv}
PLAN_SUMMARY=${PLAN_SUMMARY:-artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.md}
SELECTION_DIR=${SELECTION_DIR:-artifacts/phase3/analysis}
SELECTION_PREFIX=${SELECTION_PREFIX:-olmo3_full_sglang_chunk128_generation_shard}
DECISION_OUTPUT=${DECISION_OUTPUT:-artifacts/phase3/analysis/olmo3_full_sglang_chunk128_generation_next_decision.json}
LOG_OUTPUT=${LOG_OUTPUT:-artifacts/phase3/analysis/olmo3_full_sglang_chunk128_supervisor.log}
PID_FILE=${PID_FILE:-artifacts/phase3/analysis/olmo3_full_sglang_chunk128_supervisor.pid}
FINALIZE_SCRIPT=${FINALIZE_SCRIPT:-artifacts/phase3/analysis/olmo3_full_sglang_5000prompt_8comp_3temp_finalize.sh}
FINALIZE_SUMMARY=${FINALIZE_SUMMARY:-artifacts/phase3/analysis/olmo3_full_sglang_5000prompt_8comp_3temp_finalize.md}
COMPLETION_AUDIT=${COMPLETION_AUDIT:-artifacts/phase3/analysis/phase3_completion_audit_snapshot.csv}
COMPLETION_AUDIT_SUMMARY=${COMPLETION_AUDIT_SUMMARY:-artifacts/phase3/analysis/phase3_completion_audit_snapshot.md}
INTEGRITY_AUDIT=${INTEGRITY_AUDIT:-artifacts/phase3/analysis/olmo3_full_sglang_existing_cache_integrity.csv}
INTEGRITY_AUDIT_SUMMARY=${INTEGRITY_AUDIT_SUMMARY:-artifacts/phase3/analysis/olmo3_full_sglang_existing_cache_integrity.md}
MAX_LOG_BYTES=${MAX_LOG_BYTES:-5000000}
MAX_ESTIMATED_A100_HOURS=${MAX_ESTIMATED_A100_HOURS:-40}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-600}
MIN_FREE_GB=${MIN_FREE_GB:-12}
WATCH_PATH=${WATCH_PATH:-/home/user}
PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE:-128}
REFRESH_PLAN=${REFRESH_PLAN:-1}
REFRESH_INTEGRITY_AUDIT=${REFRESH_INTEGRITY_AUDIT:-1}
REFRESH_FINALIZATION=${REFRESH_FINALIZATION:-1}
REFRESH_COMPLETION_AUDIT=${REFRESH_COMPLETION_AUDIT:-1}
RUN_FINALIZATION_WHEN_READY=${RUN_FINALIZATION_WHEN_READY:-1}
PRIORITY_TEMPERATURES=${PRIORITY_TEMPERATURES:-1.0}

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

free_gb() {
  df -Pk "$WATCH_PATH" | awk 'NR == 2 { printf "%.2f", $4 / 1024 / 1024 }'
}

free_kb() {
  df -Pk "$WATCH_PATH" | awk 'NR == 2 { print $4 }'
}

refresh_finalization_plan() {
  uv run slop-materialize-phase3-full-sglang-finalize-plan \
    --generation-plan "$PLAN" \
    --output-script "$FINALIZE_SCRIPT" \
    --summary-output "$FINALIZE_SUMMARY" \
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
  chmod +x "$FINALIZE_SCRIPT"
}

enabled_finalization_commands() {
  if [[ ! -f "$FINALIZE_SUMMARY" ]]; then
    echo 0
    return
  fi
  awk -F'`' '/^Enabled commands:/ { print $2; found=1; exit } END { if (!found) print 0 }' "$FINALIZE_SUMMARY"
}

integrity_audit_failures() {
  if [[ ! -s "$INTEGRITY_AUDIT" ]]; then
    echo 1
    return
  fi
  uv run python - "$INTEGRITY_AUDIT" <<'PY'
import csv
import sys

path = sys.argv[1]
truthy = {"1", "true", "yes", "y"}

try:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "passed" not in reader.fieldnames:
            print(1)
            raise SystemExit(0)
        rows = list(reader)
except Exception:
    print(1)
    raise SystemExit(0)

if not rows:
    print(1)
else:
    print(sum(1 for row in rows if row.get("passed", "").strip().lower() not in truthy))
PY
}

supervisor_process_alive() {
  local pid=$1
  if [[ -z "$pid" ]] || ! [[ "$pid" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  local command
  command=$(ps -p "$pid" -o command= 2>/dev/null || true)
  [[ "$command" == *"phase3_full_sglang_supervisor.sh"* ]]
}

rotate_log_if_needed() {
  if [[ ! -f "$LOG_OUTPUT" ]]; then
    return
  fi
  local bytes
  bytes=$(wc -c < "$LOG_OUTPUT")
  if (( bytes <= MAX_LOG_BYTES )); then
    return
  fi
  local rotated="${LOG_OUTPUT}.$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$LOG_OUTPUT" "$rotated"
  printf "%s rotated previous supervisor log to %s after %s bytes\n" "$(timestamp)" "$rotated" "$bytes" > "$LOG_OUTPUT"
}

mkdir -p "$(dirname "$LOG_OUTPUT")"
rotate_log_if_needed
touch "$LOG_OUTPUT"
mkdir -p "$(dirname "$PID_FILE")"
if [[ -f "$PID_FILE" ]]; then
  existing_pid=$(cat "$PID_FILE")
  if supervisor_process_alive "$existing_pid"; then
    echo "$(timestamp) supervisor already running pid=${existing_pid}; exiting" >> "$LOG_OUTPUT"
    exit 0
  fi
fi
echo "$$" > "$PID_FILE"
echo "$(timestamp) supervisor started pid=$$ interval_seconds=${INTERVAL_SECONDS}" >> "$LOG_OUTPUT"

while true; do
  available_kb=$(free_kb)
  minimum_kb=$((MIN_FREE_GB * 1024 * 1024))
  if (( available_kb < minimum_kb )); then
    {
      echo "$(timestamp) disk guard stopped supervisor: free_gb=$(free_gb) min_free_gb=${MIN_FREE_GB}"
    } >> "$LOG_OUTPUT"
    exit 2
  fi

  if [[ "$REFRESH_PLAN" == "1" ]]; then
    {
      echo "$(timestamp) refreshing plan=${PLAN} from source_plan=${SOURCE_PLAN}"
      uv run slop-materialize-sglang-generation-plan \
        --input-plan "$SOURCE_PLAN" \
        --output "$PLAN" \
        --summary-output "$PLAN_SUMMARY" \
        --prompt-batch-size "$PROMPT_BATCH_SIZE" \
        --wandb-mode disabled
    } >> "$LOG_OUTPUT" 2>&1
  fi

  if [[ "$REFRESH_FINALIZATION" == "1" ]]; then
    {
      echo "$(timestamp) refreshing finalization script=${FINALIZE_SCRIPT} summary=${FINALIZE_SUMMARY}"
      refresh_finalization_plan
    } >> "$LOG_OUTPUT" 2>&1
  fi

  if [[ "$REFRESH_INTEGRITY_AUDIT" == "1" ]]; then
    {
      echo "$(timestamp) refreshing integrity audit=${INTEGRITY_AUDIT} summary=${INTEGRITY_AUDIT_SUMMARY}"
      uv run slop-audit-sglang-generation-cache \
        --generation-plan "$PLAN" \
        --audit-existing-only \
        --require-max-new-tokens \
        --output "$INTEGRITY_AUDIT" \
        --summary-output "$INTEGRITY_AUDIT_SUMMARY"
    } >> "$LOG_OUTPUT" 2>&1
  fi

  if [[ "$RUN_FINALIZATION_WHEN_READY" == "1" ]]; then
    {
      enabled_commands=$(enabled_finalization_commands)
      if [[ "$enabled_commands" =~ ^[0-9]+$ ]] && (( enabled_commands > 0 )); then
        integrity_failures=$(integrity_audit_failures || echo 1)
        if ! [[ "$integrity_failures" =~ ^[0-9]+$ ]]; then
          integrity_failures=1
        fi
        if (( integrity_failures > 0 )); then
          echo "$(timestamp) finalization blocked by integrity audit failures=${integrity_failures} audit=${INTEGRITY_AUDIT}"
          exit 3
        fi
        echo "$(timestamp) running finalization script=${FINALIZE_SCRIPT} enabled_commands=${enabled_commands}"
        "$FINALIZE_SCRIPT"
        echo "$(timestamp) finalization script completed; refreshing summary"
        refresh_finalization_plan
      fi
    } >> "$LOG_OUTPUT" 2>&1
  fi

  if [[ "$REFRESH_COMPLETION_AUDIT" == "1" ]]; then
    {
      echo "$(timestamp) refreshing completion audit=${COMPLETION_AUDIT} summary=${COMPLETION_AUDIT_SUMMARY}"
      PLAN="$PLAN" \
        FINALIZE_SUMMARY="$FINALIZE_SUMMARY" \
        INTEGRITY_AUDIT="$INTEGRITY_AUDIT" \
        COMPLETION_AUDIT="$COMPLETION_AUDIT" \
        COMPLETION_AUDIT_SUMMARY="$COMPLETION_AUDIT_SUMMARY" \
        scripts/phase3_refresh_completion_audit.sh
    } >> "$LOG_OUTPUT" 2>&1
  fi

  {
    echo "$(timestamp) checking plan=${PLAN} free_gb=$(free_gb)"
    priority_args=()
    read -r -a priority_temperatures <<< "$PRIORITY_TEMPERATURES"
    for priority_temperature in "${priority_temperatures[@]}"; do
      if [[ -n "$priority_temperature" ]]; then
        priority_args+=(--priority-temperature "$priority_temperature")
      fi
    done
    uv run slop-continue-phase2-generation-plan \
      --plan "$PLAN" \
      --selection-dir "$SELECTION_DIR" \
      --selection-prefix "$SELECTION_PREFIX" \
      --max-estimated-a100-hours "$MAX_ESTIMATED_A100_HOURS" \
      --execute \
      --allow-partial-retry \
      "${priority_args[@]}" \
      --decision-output "$DECISION_OUTPUT"
  } >> "$LOG_OUTPUT" 2>&1

  if grep -q '"action": "complete"' "$DECISION_OUTPUT"; then
    echo "$(timestamp) plan complete; supervisor exiting" >> "$LOG_OUTPUT"
    exit 0
  fi

  echo "$(timestamp) sleeping interval_seconds=${INTERVAL_SECONDS}" >> "$LOG_OUTPUT"
  sleep "$INTERVAL_SECONDS"
done
