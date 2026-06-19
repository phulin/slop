# Camera-Ready Table Drafts

Date: 2026-06-18

Purpose: provide submission-facing table bodies for the current bounded paper
draft. These tables avoid repository paths in the table body and keep artifact
provenance in the reproducibility appendix.

## Table 1. Data And Measurement Scope

Bounded corpus, generation, detector, and teacher-forced rerun sizes used in
the paper. The scope is the reduced production estimand, not the originally
proposed exhaustive OLMo generation grid.

| Measurement layer | Scope | Size | Paper role |
|---|---|---:|---|
| Dolma 3 pretraining reference | English-filtered retained non-code sample from a bounded scan | 5,740 documents; 8,689,472 tokens | Pretraining-style baseline |
| Dolci SFT targets | English-filtered retained target responses | 40,000 documents; 8,279,115 tokens | SFT data register |
| Dolci DPO pairs | English-filtered retained chosen/rejected pairs | 40,000 pairs; 80,000 response rows; 27,448,016 tokens | Preference-data contrast |
| Phase 1 pybiber | Full pybiber over retained Phase 1 samples | 125,740 documents x 67 features | Corpus-side register layer |
| OLMo reduced SGLang panels | Bounded free-running generation scope | 16,384 generations | OLMo realized-output and compounding layer |
| SmolLM3 no-think panel | Production-shape no-think generation grid | 49,152 generated records | Cross-ladder replication pressure |
| Phase 4 detector attribution | ModernBERT detector attribution pass | 10,000 attributed documents; 77,013 document-span rows | Candidate Tier-3 discovery |
| Phase 4 Tier-3 exact rerun | OLMo exact sequence-mass teacher-forced grid | 512 reference documents x 4 stages | Detector-candidate propensity check |

Table note: Dolma counts refer to the retained English-filtered sample, not
the full Dolma mixture. SmolLM3 generations are no-think mode for
comparability.

## Table 2. Selected Full-Pybiber Register Means

Token-weighted means for selected pybiber features across retained Dolma,
Dolci SFT, and Dolci DPO chosen/rejected samples. The selected features
summarize the main register movement; the full 67-feature outputs remain in
the reproducibility appendix.

| Feature | Dolma | SFT | DPO chosen | DPO rejected | Interpretation |
|---|---:|---:|---:|---:|---|
| Past tense | 57.065 | 4.315 | 6.421 | 8.805 | SFT sharply suppresses narrative time orientation |
| First-person pronouns | 44.732 | 9.360 | 9.223 | 11.934 | SFT/DPO are much less personal than Dolma |
| Third-person pronouns | 51.809 | 4.154 | 6.138 | 8.671 | SFT removes much narrative/person reference |
| Nominalizations | 8.944 | 24.805 | 27.373 | 28.913 | Post-training data are more nominalized |
| Attributive adjectives | 40.742 | 52.458 | 63.965 | 56.861 | Chosen responses are especially descriptive |
| Prepositions | 88.294 | 96.815 | 94.630 | 91.819 | SFT shifts toward expository relation marking |
| Adverbs | 73.226 | 31.683 | 42.427 | 37.954 | Dolma is more adverbial; DPO partially rebounds |
| Hedges | 0.742 | 0.057 | 0.163 | 0.174 | Broad shift is not simply more hedging |
| Amplifiers | 2.292 | 0.416 | 0.818 | 1.454 | Dolma exceeds SFT on intensification |
| Emphatics | 8.843 | 1.490 | 3.019 | 3.030 | Dolma exceeds SFT/DPO on emphatic marking |
| Clausal coordination | 12.089 | 3.004 | 4.912 | 5.189 | SFT is less coordination-heavy |

Table note: selected-feature document-bootstrap intervals are available for
these means in the Phase 1 pybiber interval artifact. DPO chosen/rejected
contrasts are descriptive contrasts over the retained Dolci preference pairs;
they should not be interpreted as pure human preference effects.

## Table 4. Tier-1 Precision-Validation Status

Current manual-validation state for Tier-1 regex features and exploratory
addenda. Passing, demoted, derived, and unlabeled rows define the paper's
claim boundaries rather than serving as headline empirical results.

| Feature | Role | Status | Labeled | Exact | False positive | Ambiguous | Precision | Paper treatment |
|---|---|---|---:|---:|---:|---:|---:|---|
| `contrastive_negation` | Core | Pass | 61 | 57 | 3 | 0 | 0.934 | Interim precision-supported |
| `stock_openers` | Core | Pass | 68 | 67 | 1 | 0 | 0.985 | Interim precision-supported |
| `stock_closers` | Core | Pass | 50 | 48 | 2 | 0 | 0.960 | Interim precision-supported |
| `slop_lexicon` | Core | Pass | 50 | 50 | 0 | 0 | 1.000 | Interim span/item precision-supported |
| `rule_of_three_approx` | Core | Demote | 50 | 28 | 19 | 2 | 0.560 | Exploratory coordinated-triple surface only |
| `stock_openers_closers` | Derived | Derived | 0 | 0 | 0 | 0 | undefined | Treat as pooled convenience metric |
| `eqbench_slop_score` | Aggregate bridge | Unlabeled | 0 | 0 | 0 | 0 | undefined | Exploratory benchmark bridge |
| `slop_lexicon_v2_candidate` | Candidate lexicon | Unlabeled | 0 | 0 | 0 | 0 | undefined | Exploratory candidate lexicon |

Table note: The target interim precision gate is 0.90 with enough labeled
examples to support a publication-grade regex claim. Passing precision with
very few labels is not treated as sufficient for headline use.

## Appendix Table A1. Claim Strength Bands

| Band | Claim IDs | Allowed use |
|---|---|---|
| Headline-ready, bounded | C1, C2, C7, C8, C9 | Main text with bounded wording |
| Paper-usable with explicit caveat | C3, C4, C5, C6, C10, C12, C13 | Main text only when caveat is colocated |
| Methods/status rather than substantive result | C11, C14 | Methods, limitations, or appendix |

## Appendix Table A2. Selected Pybiber Bootstrap Intervals

Point estimates with nonparametric 95% document-bootstrap intervals over
retained rows for selected paper-facing pybiber features. These intervals
preserve token weighting within each resample.

| Feature | Dolma | SFT | DPO chosen | DPO rejected |
|---|---:|---:|---:|---:|
| Past tense | 57.065 [55.212, 59.078] | 4.315 [4.195, 4.460] | 6.421 [6.211, 6.618] | 8.805 [7.685, 9.963] |
| First-person pronouns | 44.732 [42.971, 46.451] | 9.360 [9.086, 9.792] | 9.223 [9.007, 9.449] | 11.934 [10.534, 13.503] |
| Third-person pronouns | 51.809 [50.260, 53.734] | 4.154 [4.022, 4.297] | 6.138 [5.950, 6.338] | 8.671 [7.820, 9.649] |
| Nominalizations | 8.944 [8.599, 9.303] | 24.805 [24.515, 25.119] | 27.373 [27.090, 27.642] | 28.913 [27.439, 31.086] |
| Attributive adjectives | 40.742 [40.094, 41.364] | 52.458 [51.821, 53.321] | 63.965 [63.474, 64.386] | 56.861 [55.840, 58.157] |
| Hedges | 0.742 [0.710, 0.772] | 0.057 [0.051, 0.064] | 0.163 [0.149, 0.177] | 0.174 [0.094, 0.265] |
| Amplifiers | 2.292 [2.226, 2.354] | 0.416 [0.400, 0.434] | 0.818 [0.795, 0.844] | 1.454 [0.550, 2.991] |
| Emphatics | 8.843 [8.696, 9.007] | 1.490 [1.447, 1.529] | 3.019 [2.971, 3.072] | 3.030 [2.760, 3.394] |

## Appendix Table A3. Paper-Safe Negative Claims

| Avoided claim | Replacement |
|---|---|
| DPO universally creates slop | Some markers amplify at specific stages and differ by ladder |
| Preference data are uniformly more slop-like | Dolci DPO chosen/rejected differences are feature-specific and register-specific |
| Alignment simply adds hedging | Pybiber shows a broader shift toward nominalized answer-register prose |
| EQ-Bench Slop Score is the outcome variable | EQ-Bench is an aggregate bridge; per-feature propensity remains the causal measurement layer |
| Detector clusters are human-perceived slop | Detector clusters are candidate Tier-3 features until human perceptibility labels exist |
| Reduced SGLang panels equal the original exhaustive grid | The paper estimand is the bounded production scope |
