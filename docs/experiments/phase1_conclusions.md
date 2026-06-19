# Phase 1 Conclusions

Date: 2026-06-18

Scope: revised Phase 1 fixed-sample measurement. Core Tier-1 features are
contrastive negation, rule-of-three approximation, slop lexicon, expanded slop
lexicon candidate, EQ-Bench slop score, and stock openers/closers. Full
`pybiber` extraction is now available as the Tier-2 register surface. Retired
or non-core surfaces remain list/header/bold lead-ins, punctuation/rhythm, and
generic hedging.

## Inputs

- Dolci SFT retained sample:
  `artifacts/stage1/corpora/olmo3_dolci_sft_40k_retained_sample.jsonl`
- Dolci DPO retained expanded sample:
  `artifacts/stage1/corpora/olmo3_dolci_dpo_40k_retained_sample.jsonl`
- Dolma 3 retained pretraining sample:
  `artifacts/stage1/corpora/olmo3_dolma3_80k_scan_retained_sample.jsonl`

The fixed samples contain 40,000 SFT target rows, 40,000 DPO pairs represented
as 40,000 chosen and 40,000 rejected rows, and 5,740 retained Dolma documents
from an 80k scan. Sampling now requires English text using source language
metadata when available and Lingua language detection otherwise. The rerun
filtered 13,125 Dolci SFT source records, 7,541 Dolci DPO source pairs, and
3,868 Dolma records before retention.

## Artifacts

- Formal rates:
  `artifacts/stage1/census/feature_rates_by_corpus.parquet`
- Stratum rates:
  `artifacts/stage1/census/feature_rates_by_stratum.parquet`
- Formal DPO pair deltas:
  `artifacts/stage1/census/preference_pair_deltas.parquet`
- Summary:
  `artifacts/stage1/census/census_summary.md`
- Full pybiber wide outputs:
  `artifacts/stage1/census/olmo3_dolma3_80k_scan_pybiber_full.csv`,
  `artifacts/stage1/census/olmo3_dolci_sft_40k_pybiber_full.csv`,
  `artifacts/stage1/census/olmo3_dolci_dpo_40k_retained_pybiber_full.csv`
- Full pybiber long outputs:
  `artifacts/stage1/census/olmo3_dolma3_80k_scan_pybiber_full_long.csv`,
  `artifacts/stage1/census/olmo3_dolci_sft_40k_pybiber_full_long.csv`,
  `artifacts/stage1/census/olmo3_dolci_dpo_40k_retained_pybiber_full_long.csv`
- Token-weighted pybiber register analysis:
  `artifacts/stage1/census/phase1_pybiber_register_means.csv`,
  `artifacts/stage1/census/phase1_pybiber_register_deltas.csv`,
  `artifacts/stage1/census/phase1_pybiber_register_intervals.csv`, and
  `docs/experiments/phase1_pybiber_register_analysis.md`

## Core Feature Rates

Rates are counts per 1k simple tokens. `eqbench_slop_score` is the summed
EQ-Bench-style score normalized per 1k tokens, so its magnitude is not directly
comparable to binary/regex counts.

| corpus / role | docs | tokens | contrastive_negation | eqbench_slop_score | rule_of_three_approx | slop_lexicon | slop_lexicon_v2_candidate | stock_openers | stock_closers | stock_openers_closers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dolci DPO chosen | 40,000 | 14,976,916 | 0.465 | 25.869 | 3.085 | 0.971 | 0.174 | 0.252 | 0.343 | 0.596 |
| Dolci DPO rejected | 40,000 | 12,471,100 | 0.465 | 30.430 | 3.131 | 1.150 | 0.186 | 0.322 | 0.242 | 0.564 |
| Dolci SFT target | 40,000 | 8,279,115 | 0.136 | 31.021 | 1.670 | 0.577 | 0.093 | 0.262 | 0.016 | 0.278 |
| Dolma retained pretrain | 5,740 | 8,689,472 | 0.389 | 2.829 | 2.500 | 0.232 | 0.051 | 0.020 | 0.002 | 0.022 |

## DPO Pair Deltas

Mean chosen-minus-rejected rates are measured across 40,000 retained Dolci DPO
pairs. Positive means the chosen response has a higher per-1k-token rate than
the rejected response.

| feature | chosen hits | rejected hits | mean chosen-minus-rejected per-1k delta |
|---|---:|---:|---:|
| contrastive_negation | 6,965.000 | 5,798.000 | +0.005 |
| eqbench_slop_score | 387,437.857 | 379,494.508 | -16.205 |
| rule_of_three_approx | 46,207.000 | 39,050.000 | +0.445 |
| slop_lexicon | 14,541.000 | 14,342.000 | -0.055 |
| slop_lexicon_v2_candidate | 2,613.000 | 2,320.000 | +0.006 |
| stock_openers | 3,776.000 | 4,014.000 | -0.072 |
| stock_closers | 5,143.000 | 3,022.000 | +0.102 |
| stock_openers_closers | 8,919.000 | 7,036.000 | +0.030 |

## Full Pybiber Snapshot

The full pybiber path emits 67 register features. Wide output row counts:

| corpus | source rows | pybiber rows returned | pybiber features |
|---|---:|---:|---:|
| Dolma retained pretrain | 5,740 | 5,740 | 67 |
| Dolci SFT target | 40,000 | 40,000 | 67 |
| Dolci DPO expanded | 80,000 | 80,000 | 67 |

The earlier pybiber shortfall was resolved by making language filtering a real
metadata-first Lingua language-ID step rather than a character-composition
screen. On the rerun, pybiber returned one row for every retained document. The
CLI still row-qualifies duplicate source IDs before calling pybiber, which
fixes duplicate-ID collisions in pybiber input.

## Full Pybiber Register Read

The pybiber register layer changes the interpretation of "slop style." It does
not support a simple story where alignment merely adds hedges or intensifiers.
Instead, the main Phase 1 register movement is from narrative/personal web
prose toward polished answer-like exposition.

Selected token-weighted pybiber means:

| feature | Dolma | SFT | DPO chosen | DPO rejected |
|---|---:|---:|---:|---:|
| past tense | 57.065 | 4.315 | 6.421 | 8.805 |
| first-person pronouns | 44.732 | 9.360 | 9.223 | 11.934 |
| third-person pronouns | 51.809 | 4.154 | 6.138 | 8.671 |
| nominalizations | 8.944 | 24.805 | 27.373 | 28.913 |
| attributive adjectives | 40.742 | 52.458 | 63.965 | 56.861 |
| adverbs | 73.226 | 31.683 | 42.427 | 37.954 |
| hedges | 0.742 | 0.057 | 0.163 | 0.174 |
| amplifiers | 2.292 | 0.416 | 0.818 | 1.454 |
| emphatics | 8.843 | 1.490 | 3.019 | 3.030 |
| analytic negation | 8.831 | 3.702 | 5.459 | 5.688 |

Substantive register findings:

- SFT is much less narrative/personal than Dolma: past tense, first-person
  pronouns, third-person pronouns, adverbs, emphatics, contractions, and
  clausal coordination all drop sharply.
- SFT is more nominalized and expository: nominalizations, attributive
  adjectives, present tense, prepositions, and broad noun density rise.
- DPO chosen and rejected both move away from SFT on broader register
  dimensions, including more coordination, contractions, nominalizations, and
  analytic negation.
- DPO chosen is more descriptive/expository than rejected on this register
  layer: chosen is higher on attributive adjectives, adverbs, prepositions, and
  broad noun density.
- DPO rejected is more personal/narrative/mental-state framed: rejected is
  higher on first-person pronouns, third-person pronouns, past tense, private
  verbs, infinitives, nominalizations, and analytic negation.
- Hedges, amplifiers, and emphatics are higher in Dolma than in SFT. Therefore
  the paper should avoid a generic claim that alignment simply adds hedging.
  The better claim is narrower: alignment data shift toward polished,
  nominalized, answer-like exposition while specific slop lexemes and
  rhetorical forms move on top of that register shift.

## Conclusions

- DPO chosen responses have more raw hits for contrastive negation,
  rule-of-three approximation, slop lexicon, the expanded slop-lexicon
  candidate, stock closers, and pooled stock openers/closers. The
  rule-of-three row should be read only as a demoted exploratory
  coordinated-triple surface, not as a validated rhetorical-triad rate.
- In per-token paired rates, rejected responses are still higher for the
  original slop lexicon, stock openers, and EQ-Bench slop score because the
  rejected responses are shorter on average.
- SFT has the highest aggregate EQ-Bench slop score rate among the three data
  sources, while Dolma is much lower on that score but not low on
  contrastive-negation or rule-of-three approximations.
- Stock opener/closer phrases remain strongly alignment-data-associated:
  Dolma rates are near zero, while SFT and DPO rates are clearly nonzero.
- Full pybiber is now a real Tier-2 register output with complete coverage on
  the English-filtered Phase 1 samples, replacing the old Biber-lite proxy as
  the supported register feature surface.
- Substantively, full pybiber reframes the data-side story: alignment data are
  register-shifted relative to pretraining, but the shift is not reducible to
  hedging. It is an answer-register shift toward nominalized, adjectival,
  expository prose.

## Caveats

- The DPO conclusion is descriptive. Dolci DPO mixes Delta Learning,
  LLM-judged, and multiturn construction regimes, so chosen/rejected
  differences should not be presented as a pure human-preference effect.
- The rule-of-three feature is demoted for independent publication-grade claims:
  the 50-label interim check found 28 exact hits, 1 partial, 19 false positives,
  and 2 ambiguous cases, for precision 0.560.
- Manual precision validation is scoped rather than exhaustive. Four
  independent Tier-1 rows pass the interim gate, `rule_of_three_approx` is
  demoted, and `stock_openers_closers` is derived; none has completed the
  original 200-hit-per-feature gate.
- Dolma rates come from a bounded retained English sample, not the full Dolma
  mixture. Rare strata remain underfilled relative to the nominal per-stratum
  cap.
