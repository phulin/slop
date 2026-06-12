# Phase 1 Gap Audit

Date: 2026-06-12

Scope: revised Phase 1 close-out. The core feature scope is limited to
contrastive negation, rule-of-three approximation, slop lexicon, and stock
openers/closers. Punctuation/rhythm, list/header/bold lead-ins, and generic
hedging are excluded from Phase 1 core claims. Biber is represented here by
sampled `biber_lite` surface proxies; full `pybiber` extraction is deferred
until a compatible Python/spaCy environment is available.

## Current Status

Phase 1 is closeable as a sampled measurement package for OLMo 3 Dolci SFT,
Dolci DPO, and a bounded Dolma 3 pretraining reference, with caveats. It is not
yet closeable as a fully precision-validated claim package because manual labels
currently cover only an interim contrastive-negation subset.

## Completed

- Revised plan/config scope now limits required core features to
  `contrastive_negation`, `rule_of_three_approx`, `slop_lexicon`,
  `stock_openers`, `stock_closers`, and `stock_openers_closers`.
- `biber_lite` deterministic surface proxies cover pronouns, modals, hedges,
  amplifiers, verb classes, nominalizations, complements, subordination,
  wh-questions, and passive-voice approximations.
- Retained Dolci SFT package:
  `artifacts/stage1/corpora/olmo3_dolci_sft_10k_retained_sample.jsonl`.
- Retained Dolci DPO package:
  `artifacts/stage1/corpora/olmo3_dolci_dpo_10k_retained_sample.jsonl`, with
  10,000 unique package pair IDs and exactly one `chosen` and one `rejected`
  row per pair.
- Retained Dolma 3 bounded pretrain sample:
  `artifacts/stage1/corpora/olmo3_dolma3_20k_scan_retained_sample.jsonl`,
  with four populated non-code strata.
- Revised sampled Biber-lite census runs completed and logged to W&B:
  - Dolci SFT retained:
    `https://wandb.ai/phulin-self/slop-stage1/runs/05zob6cx`
  - Dolci DPO retained expanded:
    `https://wandb.ai/phulin-self/slop-stage1/runs/67ebmwe8`
  - Dolma 3 retained sample:
    `https://wandb.ai/phulin-self/slop-stage1/runs/zrpkl0t7`
  - Dolci DPO retained package with 10,000 unique pair IDs:
    `https://wandb.ai/phulin-self/slop-stage1/runs/tmgvdy4t`
- The retained DPO package run wrote 100 feature-rate rows and 250,000
  pair-feature delta rows over 20,000 retained rows, 6,637,271 simple tokens,
  and 10,000 unique pair IDs.
- Existing historical all-feature analyses remain useful diagnostics, but any
  punctuation/list/header results in those files are exploratory and excluded
  from revised Phase 1 core claims.

## Validation Status

- Manual label file:
  `artifacts/stage1/validation/labels/olmo3_dolci_sft_dpo_10k_labels.csv`.
- Current coverage: 56 labeled `contrastive_negation` rows out of a 1,400-row
  historical queue; 52 exact, 3 false positives, and 1 partial.
- This is enough for an interim contrastive-negation check, not enough to claim
  all revised core features pass the 200-hit precision gate.
- Future queues should use `slop-sample-hits --feature ...` to sample only the
  revised core features.

Recommended scoring command:

```bash
uv run slop-score-labels \
  --input artifacts/stage1/validation/labels/olmo3_dolci_sft_dpo_10k_labels.csv \
  --output artifacts/stage1/validation/precision_report.csv \
  --allow-incomplete \
  --wandb-mode online
```

## Formal Artifacts

Expected close-out artifacts:

- `artifacts/stage1/census/feature_rates_by_corpus.parquet`
- `artifacts/stage1/census/feature_rates_by_stratum.parquet`
- `artifacts/stage1/census/preference_pair_deltas.parquet`
- `artifacts/stage1/census/census_summary.md`

These should be assembled from the revised scoped CSVs, not the older
all-feature census outputs.

## Caveats

- `pybiber` could not be installed in the current CPython 3.14 environment
  because spaCy does not provide a compatible wheel here. Phase 1 therefore
  reports `biber_lite` proxies only.
- Dolci DPO is a mixed preference-construction dataset. Chosen/rejected
  differences should be interpreted by `preference_type`, `chosen_model`, and
  `rejected_model`, not as a pure human-preference effect.
- Dolma 3 is represented by a bounded 20k-scan retained sample. Rare strata are
  prefix-skewed and underfilled relative to the nominal cap.
- Full SmolLM/SmolTalk replication remains future work for this revised Phase 1
  close-out.

## Exit Status

Revised Phase 1 sampled measurement can exit once the formal parquet/summary
artifacts exist and verification passes. Precision validation remains caveated
until all revised core matchers have sufficient labels or are explicitly
demoted.
