# Phase 4 Production Runbook

Date: 2026-06-18

This runbook records the production-grade Phase 4 continuation path after the
initial detector-discovery pass. The detector, IG attribution, GTE/HDBSCAN
clustering, provisional Tier-3 matcher specs, 10k-document attribution pass,
annotation package, and 512-document exact teacher-forced continuation already exist
under `artifacts/phase4/modernbert_detector_combined_v2_clean/`.

## Completed Production Continuation

The completed compute target is the larger 512-document exact sequence-mass
teacher-forced Tier-3 OLMo checkpoint grid:

`artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/`

Scope:

- Model ladder: OLMo 3 base, SFT, DPO, final.
- Reference package:
  `artifacts/phase2/prompts/olmo3_dolci_sft_phase2_prompt_package_512.jsonl`.
- Features: 10 detector-discovered Tier-3 candidates plus
  `neutral_common_controls`.
- Scoring: exact sequence probability mass.
- Opportunity cap: 32 token-start opportunities per text, 512 total
  opportunities per text.
- Runtime settings: `--no-sequence-cache`, `--opportunity-batch-size 8`,
  `--dtype bfloat16`, `--device cuda`, `--no-torch-compile`,
  `--bootstrap-samples 500`.

Status at this write:

- Base complete:
  `olmo3_base_phase4_tier3_512_exact_sequence_opportunities.csv`
  and `olmo3_base_phase4_tier3_512_exact_sequence_summary.csv`.
- SFT complete:
  `olmo3_sft_phase4_tier3_512_exact_sequence_opportunities.csv`
  and `olmo3_sft_phase4_tier3_512_exact_sequence_summary.csv`.
- DPO complete:
  `olmo3_dpo_phase4_tier3_512_exact_sequence_opportunities.csv`
  and `olmo3_dpo_phase4_tier3_512_exact_sequence_summary.csv`.
- Final complete:
  `olmo3_final_phase4_tier3_512_exact_sequence_opportunities.csv`
  and `olmo3_final_phase4_tier3_512_exact_sequence_summary.csv`.
- Stage-grid, primary-comparison, and paired-effect assembly complete.

Each checkpoint opportunity file has approximately 145k CSV lines including
the header, consistent with the 512-document exact run shape.

## Resume Command

Use the resume script only if any completed artifact needs to be regenerated:

```bash
scripts/phase4_tier3_exact_512_resume.sh
```

The script skips stages when both the opportunity CSV and summary CSV already
exist and are non-empty. It then assembles:

- `olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv`
- `olmo3_phase4_tier3_512_exact_sequence_primary_comparison.csv`
- `olmo3_phase4_tier3_512_exact_sequence_stage_grid_summary.md`
- `olmo3_phase4_tier3_512_exact_sequence_stage_effects.csv`
- `olmo3_phase4_tier3_512_exact_sequence_stage_effects_summary.md`

Do not start the resume script while another
`slop-teacher-forced-propensity` process is active for the same output
directory.

## Completion Criteria

The 512 exact continuation is complete. All four stage summaries exist, the
stage grid and paired effects are assembled, and
`docs/experiments/phase4_tier3_teacher_forced_addendum.md` has been refreshed
with the 512-document results. The 128-document results remain a historical
scoping artifact.
