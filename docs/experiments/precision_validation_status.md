# Precision Validation Status

Date: 2026-06-18

This report summarizes the current manual precision-labeling state for
the revised Phase 1 feature surface. It is a status artifact, not a
claim that the precision gate is complete.

## Inputs

- Label CSV: `artifacts/stage1/validation/labels/olmo3_dolci_sft_dpo_10k_labels.csv`
- Label CSV: `artifacts/stage1/validation/labels/revised_phase1_remaining_core_labels.csv`
- Label CSV: `artifacts/stage1/validation/labels/revised_phase1_stock_closers_additional_labels_2026-06-18.csv`
- Label CSV: `artifacts/stage1/validation/labels/revised_phase1_slop_lexicon_additional_labels_2026-06-18.csv`
- Label CSV: `artifacts/stage1/validation/labels/revised_phase1_rule_of_three_additional_labels_2026-06-18.csv`
- Hit queue CSV: `artifacts/stage1/validation/hit_samples/olmo3_dolci_sft_dpo_10k_hits_for_labeling.csv`
- Hit queue CSV: `artifacts/stage1/validation/hit_samples/revised_phase1_remaining_core_hits.csv`
- Hit queue CSV: `artifacts/stage1/validation/hit_samples/manual_feature_smell_test_20_each_updated.csv`
- Hit queue CSV: `artifacts/stage1/validation/hit_samples/revised_phase1_stock_openers_closers_hits.csv`

Queue files with zero retained feature hits:
- `artifacts/stage1/validation/hit_samples/revised_phase1_stock_openers_closers_hits.csv`

## Gate Settings

- Target precision: `0.90`
- Maximum ambiguous rate: `0.10`
- Minimum labeled rows for core features: `50`

## Feature Status

| Feature | Core | Derived | Status | Queued | Labeled | Exact | Partial | FP | Ambig | Precision | Ambig rate |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| contrastive_negation | True | False | pass | 220 | 61 | 57 | 1 | 3 | 0 | 0.934426 | 0.000000 |
| eqbench_slop_score | False | False | unlabeled | 20 | 0 | 0 | 0 | 0 | 0 |  |  |
| rule_of_three_approx | True | False | demote | 420 | 50 | 28 | 1 | 19 | 2 | 0.560000 | 0.040000 |
| slop_lexicon | True | False | pass | 420 | 50 | 50 | 0 | 0 | 0 | 1.000000 | 0.000000 |
| slop_lexicon_v2_candidate | False | False | unlabeled | 20 | 0 | 0 | 0 | 0 | 0 |  |  |
| stock_closers | True | False | pass | 420 | 50 | 48 | 0 | 2 | 0 | 0.960000 | 0.000000 |
| stock_openers | True | False | pass | 420 | 68 | 67 | 0 | 1 | 0 | 0.985294 | 0.000000 |
| stock_openers_closers | False | True | derived | 0 | 0 | 0 | 0 | 0 | 0 |  |  |

## Readout

- Core features passing the current gate: `4/5`.
- Core features not passing publication-grade Phase 1 regex claims: `1`.
- Rows marked `incomplete` can still be useful smell-test evidence, but
  they should not be cited as passing the 200-hit precision plan.
- Rows marked `unlabeled` need either a direct label queue or an explicit
  decision to infer precision from component features.
- Rows marked `derived` are pooled convenience views rather than
  independent precision-validation targets.
- A supplied queue file with zero retained hits means the bounded sampling
  attempt did not find labelable examples under the current scan limit.

## Next Actions

1. Continue labeling core rows until each retained core feature has at
   least 50 labels for the interim gate and, ideally, 200 labels for the
   original validation plan.
2. Treat `stock_openers_closers` as a derived convenience metric unless
   a broader scan is intentionally budgeted; the bounded direct queue
   attempt supplied here produced zero retained hits.
3. Score the completed label files with `slop-score-labels` before using
   corpus-rate tables for publication-grade claims.
