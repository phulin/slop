# Phase 1 Conclusions

Date: 2026-06-12

Scope: revised Phase 1 sampled measurement. Core features are contrastive
negation, rule-of-three approximation, slop lexicon, and stock
openers/closers. Punctuation/rhythm, list/header/bold lead-ins, generic
hedging, full pybiber extraction, and full SmolLM replication are deferred.

## Artifacts

- Formal rates:
  `artifacts/stage1/census/feature_rates_by_corpus.parquet`
- Formal DPO pair deltas:
  `artifacts/stage1/census/preference_pair_deltas.parquet`
- Summary:
  `artifacts/stage1/census/census_summary.md`
- Assembly W&B run:
  `https://wandb.ai/phulin-self/slop-stage1/runs/oobso9zd`
- Revised DPO retained package W&B run:
  `https://wandb.ai/phulin-self/slop-stage1/runs/tmgvdy4t`

## Core Feature Rates

Rates are counts per 1k simple tokens.

| corpus / role | contrastive negation | rule-of-three approx | slop lexicon | stock openers | stock closers | openers + closers |
|---|---:|---:|---:|---:|---:|---:|
| Dolci DPO chosen | 0.459 | 3.696 | 0.990 | 0.235 | 0.413 | 0.648 |
| Dolci DPO rejected | 0.369 | 3.391 | 1.022 | 0.277 | 0.292 | 0.569 |
| Dolci SFT target | 0.139 | 1.937 | 0.553 | 0.262 | 0.089 | 0.351 |
| Dolma pretrain sample | 0.389 | 3.946 | 0.238 | 0.027 | 0.008 | 0.035 |

## DPO Pair Deltas

Mean chosen-minus-rejected rates, measured within 10,000 retained Dolci DPO
pairs.

| feature | mean delta |
|---|---:|
| contrastive negation | +0.030 |
| rule-of-three approx | +0.494 |
| slop lexicon | -0.134 |
| stock openers | -0.045 |
| stock closers | +0.096 |
| stock openers + closers | +0.051 |

## Biber-Lite Snapshot

Top rates per corpus/role:

| corpus / role | highest Biber-lite rates |
|---|---|
| Dolci DPO chosen | nominalizations 31.70, infinitives 14.29, demonstratives 12.14, first-person pronouns 9.88 |
| Dolci DPO rejected | nominalizations 28.79, infinitives 15.20, demonstratives 11.49, first-person pronouns 9.43 |
| Dolci SFT target | nominalizations 27.95, infinitives 13.93, first-person pronouns 10.38, demonstratives 9.65 |
| Dolma pretrain sample | third-person pronouns 47.35, first-person pronouns 45.57, infinitives 26.14, second-person pronouns 19.00 |

## Conclusions

- Dolci DPO chosen responses have higher rates than rejected responses for
  contrastive negation, rule-of-three approximation, stock closers, and pooled
  stock openers/closers.
- Dolci DPO rejected responses have slightly higher rates for the pooled slop
  lexicon and stock openers.
- Dolci SFT target responses sit below Dolci DPO chosen/rejected on
  contrastive negation and rule-of-three approximation, but above the Dolma
  pretrain sample for slop lexicon and stock opener/closer phrases.
- The Dolma pretrain sample has high contrastive-negation and rule-of-three
  rates relative to Dolci SFT, but stock opener/closer rates are near zero.
- Biber-lite rates are large enough to keep as register/context covariates in
  Phase 2, but they should be described as surface proxies rather than full
  Biber categories.

## Caveats

- The current DPO conclusion is descriptive. It should not be presented as a
  pure human-preference result because Dolci DPO mixes Delta Learning,
  LLM-judged, and multiturn construction regimes.
- The rule-of-three feature is a regex approximation, not a full phrase/clause
  parser.
- The user accepted the current remaining core hit samples as good enough for
  this stage, but deeper publication claims should still retain the precision
  report and label artifacts.
- Dolma rates come from a bounded 20k-scan retained sample and should not be
  interpreted as full-mixture source prevalence.
