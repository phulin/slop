# Phase 1 Gap Audit

Date: 2026-06-18

Scope: revised Phase 1 close-out. The core feature scope is limited to
contrastive negation, rule-of-three approximation, slop lexicon, and stock
openers/closers. List/header/bold lead-ins are retired from the active feature
surface. Punctuation/rhythm and generic hedging are excluded from Phase 1 core
claims. Full `pybiber` extraction is now available and replaces the sampled
`biber_lite` surface proxies as the supported Tier-2 register surface.

## Current Status

Phase 1 is closeable as a fixed-sample measurement package for OLMo 3 Dolci
SFT, Dolci DPO, and a bounded Dolma 3 pretraining reference. The current
samples require English text using source language metadata when available and
Lingua language detection otherwise. Phase 1 is not yet closeable as a fully
precision-validated claim package: the current interim precision table has
four of five independent core features passing, and `rule_of_three_approx` is
demoted for publication-grade independent regex claims after the 50-label
interim check. The pooled opener/closer feature is treated as a derived
convenience metric rather than an independent core matcher.

## Completed

- Revised plan/config scope now limits independent core features to
  `contrastive_negation`, `rule_of_three_approx`, `slop_lexicon`,
  `stock_openers`, and `stock_closers`; `stock_openers_closers` is retained
  as a derived pooled view, while `slop_lexicon_v2_candidate` and
  `eqbench_slop_score` are exploratory addenda.
- Full `pybiber` extraction covers 67 register features through
  `uv run slop-pybiber-full`.
- Retained Dolci SFT package:
  `artifacts/stage1/corpora/olmo3_dolci_sft_40k_retained_sample.jsonl`,
  with 40,000 target rows after filtering 13,125 non-English source records.
- Retained Dolci DPO package:
  `artifacts/stage1/corpora/olmo3_dolci_dpo_40k_retained_sample.jsonl`, with
  40,000 unique package pair IDs and exactly one `chosen` and one `rejected`
  row per pair after filtering 7,541 source pairs where either side failed the
  English requirement.
- Retained Dolma 3 bounded pretrain sample:
  `artifacts/stage1/corpora/olmo3_dolma3_80k_scan_retained_sample.jsonl`,
  with 5,740 retained rows across four populated non-code strata after
  filtering 3,868 non-English records.
- Revised fixed-sample Tier-1 census runs completed locally with W&B disabled:
  - Dolci SFT retained:
    `artifacts/stage1/census/olmo3_dolci_sft_40k_revised_tier1_feature_rates.csv`
  - Dolci DPO retained expanded:
    `artifacts/stage1/census/olmo3_dolci_dpo_40k_retained_revised_tier1_feature_rates.csv`
  - Dolci DPO pair deltas:
    `artifacts/stage1/census/olmo3_dolci_dpo_40k_retained_revised_tier1_pair_deltas.csv`
  - Dolma 3 retained sample:
    `artifacts/stage1/census/olmo3_dolma3_80k_scan_revised_tier1_feature_rates.csv`
- The retained DPO package run wrote 36 feature-rate rows and 360,000
  pair-feature delta rows over 80,000 retained rows, 27,448,016 simple tokens,
  and 40,000 unique pair IDs.
- Full pybiber runs completed:
  - Dolma: 5,740 returned rows x 67 features
  - Dolci SFT: 40,000 returned rows x 67 features
  - Dolci DPO: 80,000 returned rows x 67 features
- Existing historical all-feature analyses remain useful diagnostics, but any
  list/header results in those files are retired, and punctuation results are
  exploratory and excluded from revised Phase 1 core claims.

## Validation Status

- Current status artifact:
  `docs/experiments/precision_validation_status.md`.
- Machine-readable status:
  `artifacts/stage1/validation/precision_validation_status.csv`.
- Manual label files:
  `artifacts/stage1/validation/labels/olmo3_dolci_sft_dpo_10k_labels.csv`.
  `artifacts/stage1/validation/labels/revised_phase1_remaining_core_labels.csv`.
- Interim passing features: `contrastive_negation`, `slop_lexicon`,
  `stock_openers`, and `stock_closers`.
- Demoted independent core feature: `rule_of_three_approx` (50 labels, 28
  exact, 1 partial, 19 false positives, 2 ambiguous; precision 0.560).
- The direct bounded queue attempt for `stock_openers_closers` produced zero
  retained hits, so the pooled feature should be treated as a derived
  convenience metric unless a broader scan is explicitly budgeted.
- This is enough to document validation progress and feature-specific caveats,
  not enough to claim all revised core features pass the original 200-hit
  precision gate.

Recommended status command:

```bash
uv run slop-summarize-precision-validation \
  --label-input artifacts/stage1/validation/labels/olmo3_dolci_sft_dpo_10k_labels.csv \
  --label-input artifacts/stage1/validation/labels/revised_phase1_remaining_core_labels.csv \
  --queue-input artifacts/stage1/validation/hit_samples/olmo3_dolci_sft_dpo_10k_hits_for_labeling.csv \
  --queue-input artifacts/stage1/validation/hit_samples/revised_phase1_remaining_core_hits.csv \
  --queue-input artifacts/stage1/validation/hit_samples/manual_feature_smell_test_20_each_updated.csv \
  --queue-input artifacts/stage1/validation/hit_samples/revised_phase1_stock_openers_closers_hits.csv \
  --output-csv artifacts/stage1/validation/precision_validation_status.csv \
  --output-md docs/experiments/precision_validation_status.md
```

## Formal Artifacts

Current close-out artifacts:

- `artifacts/stage1/census/feature_rates_by_corpus.parquet`
- `artifacts/stage1/census/feature_rates_by_stratum.parquet`
- `artifacts/stage1/census/preference_pair_deltas.parquet`
- `artifacts/stage1/census/census_summary.md`

These should be assembled from the revised scoped CSVs, not the older
all-feature census outputs.

## Caveats

- The previous pybiber row shortfall is resolved for the English-filtered Phase
  1 samples. Treat language filtering as part of the sample definition when
  comparing against the older 2026-06-17 artifacts.
- Dolci DPO is a mixed preference-construction dataset. Chosen/rejected
  differences should be interpreted by `preference_type`, `chosen_model`, and
  `rejected_model`, not as a pure human-preference effect.
- Dolma 3 is represented by a bounded retained English sample from an 80k scan.
  Rare strata remain underfilled relative to the nominal cap.
- Full SmolLM/SmolTalk replication remains future work for this revised Phase 1
  close-out.

## Exit Status

Revised Phase 1 fixed-sample measurement has exited for OLMo 3 data-side
corpora: formal parquet/summary artifacts exist, Tier-1 verification passed,
full pybiber outputs exist with complete retained-row coverage, and the
Tier-1 validation surface is dispositioned for the current paper scope. Four
independent rows pass the interim precision gate, `rule_of_three_approx` is
demoted, and `stock_openers_closers` is derived.
