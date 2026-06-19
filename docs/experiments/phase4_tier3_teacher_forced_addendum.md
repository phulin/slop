# Phase 4 Tier-3 Teacher-Forced Addendum

Date: 2026-06-18

This addendum records the production-grade 512-document exact sequence-mass
teacher-forced rerun for the 10 Phase 4 detector-discovered candidate features.
It uses the existing OLMo teacher-forced propensity harness, not a separate ad
hoc scoring script.

## Scope

- Model ladder: OLMo 3 base / SFT / DPO / final.
- Reference text:
  `artifacts/phase2/prompts/olmo3_dolci_sft_phase2_prompt_package_512.jsonl`.
- Features: the 10 Phase 4 IG/HDBSCAN candidate matchers plus
  `neutral_common_controls`.
- Scoring mode: exact sequence probability mass.
- Opportunity cap: 32 token-start opportunities per text, 512 total
  opportunities per text.
- Runtime settings: `--no-sequence-cache`, `--opportunity-batch-size 8`,
  `--dtype bfloat16`, `--device cuda`, `--no-torch-compile`,
  `--bootstrap-samples 500`.

The earlier 128-document exact run remains useful as a scoping artifact. The
512-document run is the current Phase 4 Tier-3 teacher-forced result.

## Artifacts

- Stage grid:
  `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv`
- Primary comparison:
  `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_primary_comparison.csv`
- Paired stage effects:
  `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_stage_effects.csv`
- Stage-grid summary:
  `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_stage_grid_summary.md`
- Paired-effect summary:
  `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_stage_effects_summary.md`

All four checkpoint opportunity and summary CSVs exist for base, SFT, DPO, and
final. The paired stage-effects table has 33 comparison-feature rows; 31 pass
BH-FDR at `alpha=0.05`.

## Results

Raw AF compares model probability mass to the reference initiation rate.
Normalized AF divides by the `neutral_common_controls` AF for the same stage.

| Feature | Ref inits | Raw AF base | Raw AF SFT | Raw AF DPO | Raw AF final | Norm AF base | Norm AF SFT | Norm AF DPO | Norm AF final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| additive transition | 218 | 0.565 | 0.717 | 0.853 | 1.012 | 3.103 | 2.873 | 3.322 | 3.132 |
| code boilerplate | 293 | 0.438 | 0.442 | 0.433 | 0.467 | 2.405 | 1.770 | 1.687 | 1.444 |
| conversation greeting | 10 | 0.138 | 0.103 | 0.093 | 0.088 | 0.760 | 0.412 | 0.363 | 0.273 |
| followup offer | 11 | 6.171 | 38.393 | 39.880 | 39.126 | 33.917 | 153.924 | 155.369 | 121.053 |
| prescriptive instruction | 74 | 0.301 | 0.641 | 0.654 | 0.763 | 1.652 | 2.571 | 2.549 | 2.361 |
| process framing | 126 | 1.401 | 1.629 | 1.726 | 1.957 | 7.701 | 6.529 | 6.722 | 6.056 |
| response constraint | 11 | 0.740 | 1.602 | 1.847 | 2.017 | 4.069 | 6.422 | 7.197 | 6.240 |
| benefit intensifier | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| careful reasoning | 1 | 89.071 | 76.253 | 146.950 | 169.682 | 489.564 | 305.707 | 572.505 | 524.985 |
| quantity time recipe | 11 | 0.158 | 0.261 | 0.311 | 0.342 | 0.868 | 1.047 | 1.213 | 1.059 |
| neutral common controls | 1148 | 0.182 | 0.249 | 0.257 | 0.323 | 1.000 | 1.000 | 1.000 | 1.000 |

Primary feature comparison for `phase4_ig_process_framing`:

| Stage | Raw AF | Raw AF CI | Normalized AF | Normalized AF CI | Normalized AF Delta vs Base |
|---|---:|---:|---:|---:|---:|
| base | 1.401 | 1.197-1.655 | 7.701 | 6.593-9.002 | 0.000 |
| sft | 1.629 | 1.402-1.928 | 6.529 | 5.518-7.661 | -1.172 |
| dpo | 1.726 | 1.471-2.065 | 6.722 | 5.632-7.911 | -0.979 |
| final | 1.957 | 1.674-2.323 | 6.056 | 5.100-7.137 | -1.645 |

## Interpretation

Process framing is the most stable denominator-supported Tier-3 signal. It has
126 reference initiations, raw AF above 1 at every stage, and raw AF rises from
`1.401` at base to `1.957` at final. Neutral-normalized AF declines because
the neutral common-control AF rises faster than process framing, not because
process framing probability falls.

Follow-up offers are still the largest teacher-forced Tier-3 signal, with raw
AF jumping from `6.171` at base to roughly `39` after SFT. The denominator is
sparse at 11 reference initiations, so this is a strong provisional signal but
not a standalone headline claim.

Additive transitions are now well-denominated and move from below the reference
rate at base (`0.565`) to reference-matched by final (`1.012`). This supports a
real alignment-stage movement, but the normalized AF stays around `3`, so it is
not uniquely an alignment-only artifact.

Prescriptive instruction and response constraint both rise materially after
alignment. Prescriptive instruction has moderate denominator support with 74
reference initiations. Response constraint remains sparse with 11 initiations
but has a consistent raw-AF rise from `0.740` to `2.017`.

Conversation greetings do not show OLMo teacher-forced amplification. This
matches the earlier free-running interpretation that greetings were more
SmolLM3-like than OLMo-like in this project.

Careful reasoning has only one reference initiation, so its huge AF values are
not interpretable as a stable rate estimate. Benefit intensifier has no
reference initiations in this package.

## Conclusion

The production-grade 512-document exact run confirms that Phase 4 detector
candidates are not merely free-generation counting artifacts. The strongest
supported OLMo-side Tier-3 candidates are:

- stable and denominator-supported: process framing;
- large but sparse: follow-up offer;
- alignment-rising with better denominator support: additive transition and
  prescriptive instruction;
- alignment-rising but sparse: response constraint.

The remaining blocker for publication-grade Phase 4 claims is not more model
scoring. It is validation: manual precision/perceptibility labels and, for
sparse features, targeted denominator expansion.
