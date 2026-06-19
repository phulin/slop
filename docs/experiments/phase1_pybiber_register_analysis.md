# Phase 1 Full Pybiber Register Analysis

This analysis uses the full 67-feature `pybiber` outputs for the
English-filtered OLMo Phase 1 samples. Means are token-weighted over
retained documents, using sample manifest token counts.

## Inputs

- `dolma_pretrain`: 5,740 rows, 8,689,472 tokens, `artifacts/stage1/census/olmo3_dolma3_80k_scan_pybiber_full.csv`
- `sft_target`: 40,000 rows, 8,279,115 tokens, `artifacts/stage1/census/olmo3_dolci_sft_40k_pybiber_full.csv`
- `dpo_chosen`: 40,000 rows, 14,976,916 tokens, `artifacts/stage1/census/olmo3_dolci_dpo_40k_retained_pybiber_full.csv`
- `dpo_rejected`: 40,000 rows, 12,471,100 tokens, `artifacts/stage1/census/olmo3_dolci_dpo_40k_retained_pybiber_full.csv`

## Selected Token-Weighted Means

| feature | dolma_pretrain | sft_target | dpo_chosen | dpo_rejected |
| --- | --- | --- | --- | --- |
| f_01_past_tense | 57.065 | 4.315 | 6.421 | 8.805 |
| f_03_present_tense | 45.773 | 51.300 | 53.126 | 53.432 |
| f_06_first_person_pronouns | 44.732 | 9.360 | 9.223 | 11.934 |
| f_07_second_person_pronouns | 18.849 | 3.456 | 7.208 | 6.786 |
| f_08_third_person_pronouns | 51.809 | 4.154 | 6.138 | 8.671 |
| f_14_nominalizations | 8.944 | 24.805 | 27.373 | 28.913 |
| f_24_infinitives | 18.905 | 11.059 | 10.815 | 12.705 |
| f_39_prepositions | 88.294 | 96.815 | 94.630 | 91.819 |
| f_40_adj_attr | 40.742 | 52.458 | 63.965 | 56.861 |
| f_42_adverbs | 73.226 | 31.683 | 42.427 | 37.954 |
| f_43_type_token | 0.342 | 0.501 | 0.436 | 0.377 |
| f_44_mean_word_length | 4.304 | 4.864 | 4.951 | 4.898 |
| f_47_hedges | 0.742 | 0.057 | 0.163 | 0.174 |
| f_48_amplifiers | 2.292 | 0.416 | 0.818 | 1.454 |
| f_49_emphatics | 8.843 | 1.490 | 3.019 | 3.030 |
| f_51_demonstratives | 5.037 | 4.728 | 4.011 | 4.498 |
| f_52_modal_possibility | 6.653 | 4.911 | 6.083 | 6.737 |
| f_53_modal_necessity | 1.014 | 1.664 | 1.297 | 1.279 |
| f_54_modal_predictive | 5.983 | 2.402 | 2.572 | 2.554 |
| f_64_phrasal_coordination | 6.737 | 6.039 | 8.113 | 9.001 |
| f_65_clausal_coordination | 12.089 | 3.004 | 4.912 | 5.189 |
| f_66_neg_synthetic | 1.327 | 0.666 | 1.030 | 0.738 |
| f_67_neg_analytic | 8.831 | 3.702 | 5.459 | 5.688 |

## Largest Register Deltas

### sft_minus_dolma

| feature | left_value | right_value | delta |
| --- | --- | --- | --- |
| f_16_other_nouns | 460.121 | 222.923 | 237.198 |
| f_01_past_tense | 4.315 | 57.065 | -52.750 |
| f_08_third_person_pronouns | 4.154 | 51.809 | -47.655 |
| f_42_adverbs | 31.683 | 73.226 | -41.543 |
| f_06_first_person_pronouns | 9.360 | 44.732 | -35.372 |
| f_14_nominalizations | 24.805 | 8.944 | 15.861 |
| f_07_second_person_pronouns | 3.456 | 18.849 | -15.393 |
| f_40_adj_attr | 52.458 | 40.742 | 11.716 |
| f_59_contractions | 2.640 | 12.116 | -9.476 |
| f_65_clausal_coordination | 3.004 | 12.089 | -9.085 |
| f_39_prepositions | 96.815 | 88.294 | 8.521 |
| f_24_infinitives | 11.059 | 18.905 | -7.845 |
| f_49_emphatics | 1.490 | 8.843 | -7.352 |
| f_09_pronoun_it | 5.188 | 12.120 | -6.932 |
| f_38_other_adv_sub | 3.490 | 9.369 | -5.879 |

### dpo_chosen_minus_rejected

| feature | left_value | right_value | delta |
| --- | --- | --- | --- |
| f_16_other_nouns | 421.637 | 409.725 | 11.912 |
| f_40_adj_attr | 63.965 | 56.861 | 7.104 |
| f_42_adverbs | 42.427 | 37.954 | 4.473 |
| f_39_prepositions | 94.630 | 91.819 | 2.812 |
| f_06_first_person_pronouns | 9.223 | 11.934 | -2.710 |
| f_08_third_person_pronouns | 6.138 | 8.671 | -2.532 |
| f_01_past_tense | 6.421 | 8.805 | -2.384 |
| f_56_verb_private | 11.109 | 13.088 | -1.979 |
| f_24_infinitives | 10.815 | 12.705 | -1.890 |
| f_14_nominalizations | 27.373 | 28.913 | -1.540 |
| f_19_be_main_verb | 11.091 | 12.214 | -1.122 |
| f_37_if | 5.196 | 4.142 | 1.054 |
| f_45_conjuncts | 4.647 | 3.658 | 0.989 |
| f_61_stranded_preposition | 3.171 | 2.211 | 0.960 |
| f_64_phrasal_coordination | 8.113 | 9.001 | -0.888 |

### dpo_chosen_minus_sft

| feature | left_value | right_value | delta |
| --- | --- | --- | --- |
| f_16_other_nouns | 421.637 | 460.121 | -38.484 |
| f_40_adj_attr | 63.965 | 52.458 | 11.507 |
| f_42_adverbs | 42.427 | 31.683 | 10.744 |
| f_07_second_person_pronouns | 7.208 | 3.456 | 3.752 |
| f_14_nominalizations | 27.373 | 24.805 | 2.568 |
| f_56_verb_private | 11.109 | 13.673 | -2.564 |
| f_59_contractions | 5.125 | 2.640 | 2.485 |
| f_39_prepositions | 94.630 | 96.815 | -2.185 |
| f_61_stranded_preposition | 3.171 | 1.036 | 2.135 |
| f_01_past_tense | 6.421 | 4.315 | 2.106 |
| f_64_phrasal_coordination | 8.113 | 6.039 | 2.074 |
| f_08_third_person_pronouns | 6.138 | 4.154 | 1.984 |
| f_65_clausal_coordination | 4.912 | 3.004 | 1.908 |
| f_03_present_tense | 53.126 | 51.300 | 1.825 |
| f_67_neg_analytic | 5.459 | 3.702 | 1.757 |

### dpo_rejected_minus_sft

| feature | left_value | right_value | delta |
| --- | --- | --- | --- |
| f_16_other_nouns | 409.725 | 460.121 | -50.396 |
| f_42_adverbs | 37.954 | 31.683 | 6.271 |
| f_39_prepositions | 91.819 | 96.815 | -4.997 |
| f_08_third_person_pronouns | 8.671 | 4.154 | 4.517 |
| f_01_past_tense | 8.805 | 4.315 | 4.490 |
| f_40_adj_attr | 56.861 | 52.458 | 4.403 |
| f_14_nominalizations | 28.913 | 24.805 | 4.108 |
| f_07_second_person_pronouns | 6.786 | 3.456 | 3.330 |
| f_59_contractions | 5.798 | 2.640 | 3.158 |
| f_64_phrasal_coordination | 9.001 | 6.039 | 2.962 |
| f_06_first_person_pronouns | 11.934 | 9.360 | 2.573 |
| f_19_be_main_verb | 12.214 | 9.775 | 2.438 |
| f_65_clausal_coordination | 5.189 | 3.004 | 2.185 |
| f_03_present_tense | 53.432 | 51.300 | 2.132 |
| f_45_conjuncts | 3.658 | 5.713 | -2.055 |

## Selected Bootstrap Intervals

Nonparametric document-bootstrap intervals over retained rows, preserving
token weighting within each resample. Values are point estimates with
95% intervals for the first selected paper-facing features.

| feature | dolma_pretrain | sft_target | dpo_chosen | dpo_rejected |
| --- | --- | --- | --- | --- |
| f_01_past_tense | 57.065 [55.212, 59.078] | 4.315 [4.195, 4.460] | 6.421 [6.211, 6.618] | 8.805 [7.685, 9.963] |
| f_03_present_tense | 45.773 [44.561, 47.100] | 51.300 [50.984, 51.671] | 53.126 [52.699, 53.525] | 53.432 [52.363, 54.681] |
| f_06_first_person_pronouns | 44.732 [42.971, 46.451] | 9.360 [9.086, 9.792] | 9.223 [9.007, 9.449] | 11.934 [10.534, 13.503] |
| f_07_second_person_pronouns | 18.849 [18.246, 19.470] | 3.456 [3.331, 3.581] | 7.208 [7.031, 7.373] | 6.786 [6.267, 7.444] |
| f_08_third_person_pronouns | 51.809 [50.260, 53.734] | 4.154 [4.022, 4.297] | 6.138 [5.950, 6.338] | 8.671 [7.820, 9.649] |
| f_14_nominalizations | 8.944 [8.599, 9.303] | 24.805 [24.515, 25.119] | 27.373 [27.090, 27.642] | 28.913 [27.439, 31.086] |
| f_24_infinitives | 18.905 [18.662, 19.131] | 11.059 [10.954, 11.170] | 10.815 [10.707, 10.928] | 12.705 [12.285, 13.075] |
| f_39_prepositions | 88.294 [87.736, 88.968] | 96.815 [96.354, 97.219] | 94.630 [94.197, 95.004] | 91.819 [90.502, 93.122] |
| f_40_adj_attr | 40.742 [40.094, 41.364] | 52.458 [51.821, 53.321] | 63.965 [63.474, 64.386] | 56.861 [55.840, 58.157] |
| f_42_adverbs | 73.226 [72.535, 73.964] | 31.683 [31.424, 31.933] | 42.427 [42.079, 42.753] | 37.954 [36.292, 39.819] |

## Substantive Read

- SFT is far less narrative/personal than the Dolma pretraining sample:
  past tense, first-person pronouns, third-person pronouns, adverbs,
  emphatics, and clausal coordination all drop sharply.
- SFT is more nominalized and expository: nominalizations, attributive
  adjectives, present tense, prepositions, and broad noun density rise.
- DPO does not simply continue SFT in one direction. Both chosen and
  rejected responses differ from SFT on broader register dimensions,
  including more coordination, contractions, nominalizations, and
  analytic negation.
- DPO chosen responses are more descriptive/expository than rejected
  responses on this register layer: chosen is higher on attributive
  adjectives, adverbs, prepositions, and broad noun density.
- DPO rejected responses are more personal/narrative/mental-state framed:
  rejected is higher on first-person pronouns, third-person pronouns,
  past tense, private verbs, infinitives, nominalizations, and analytic
  negation.
- Hedges, amplifiers, and emphatics are higher in the Dolma pretraining
  sample than in SFT. So the register result does not support a generic
  claim that alignment simply adds hedging; it supports a narrower claim
  that alignment data shift toward polished, nominalized, answer-like
  exposition while specific slop lexemes and rhetorical forms move on top
  of that register shift.
