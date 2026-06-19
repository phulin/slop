# Paper Tables Draft

Date: 2026-06-18

Purpose: collect manuscript-ready tables from the current bounded experiment
package. These tables are drafts for paper writing; the detailed phase reports
remain the authority for provenance and caveats.

## Table 1. Data And Measurement Scope

| Component | Scope | Size | Primary source | Paper use |
|---|---|---:|---|---|
| Dolma 3 pretraining reference | English-filtered retained non-code sample from bounded 80k scan | 5,740 docs; 8,689,472 tokens | `artifacts/stage1/census/census_summary.md` | Pretraining-style baseline |
| Dolci SFT targets | English-filtered retained target responses | 40,000 docs; 8,279,115 tokens | `artifacts/stage1/census/census_summary.md`; `phase1_pybiber_register_analysis.md` | SFT data register |
| Dolci DPO pairs | English-filtered retained chosen/rejected pairs | 40,000 pairs; 80,000 expanded rows; 27,448,016 tokens | `artifacts/stage1/census/census_summary.md` | Preference-data contrast |
| Phase 1 pybiber | Full pybiber over retained Phase 1 samples | 125,740 returned docs x 67 features | `artifacts/stage1/census/*_pybiber_full.csv` | Corpus-side register layer |
| OLMo reduced SGLang panels | Bounded free-running generation scope | 16,384 planned/completed generations | `docs/experiments/phase3_completion_audit.md` | OLMo realized-output and compounding layer |
| SmolLM3 no-think panel | Production-shape no-think generation grid | 49,152 generated records | `docs/experiments/phase3_completion_audit.md` | Cross-ladder replication pressure |
| Phase 4 detector attribution | ModernBERT detector attribution pass | 10,000 attributed docs; 77,013 doc-span rows | `docs/experiments/phase4_detector_discovery_report.md` | Candidate Tier-3 discovery |
| Phase 4 Tier-3 exact rerun | OLMo exact sequence-mass teacher-forced grid | 512 reference documents x 4 stages | `docs/experiments/phase4_completion_audit.md` | Detector-candidate propensity check |

## Table 2. Selected Full-Pybiber Register Means

Token-weighted means. Source:
`docs/experiments/phase1_pybiber_register_analysis.md`.

| Feature | Dolma pretrain | SFT target | DPO chosen | DPO rejected | Main read |
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
| Clausal coordination | 12.089 | 3.004 | 4.912 | 5.189 | SFT is less narrative/coordination-heavy |

## Table 3. Current Claim Strength Bands

Source: `docs/experiments/paper_claim_matrix.md`.

| Band | Claim IDs | What can be said |
|---|---|---|
| Headline-ready, bounded | C1, C2, C7, C8, C9 | OLMo/Dolci register shift, non-hedging interpretation, OLMo `slop_lexicon` DPO propensity peak, data-complicity caveat, and propensity/output divergence |
| Paper-usable with explicit caveat | C3, C4, C5, C6, C10, C12, C13 | DPO chosen/rejected register contrast, EQ-Bench scorecards, cross-ladder pressure, and selected Tier-3 propensity signals |
| Methods/status rather than substantive result | C11, C14 | Detector discovery status and exclusion of formatting features |

## Table 4. Tier-1 Precision-Validation Status

Source: `docs/experiments/precision_validation_status.md` and
`artifacts/stage1/validation/precision_validation_status.csv`.

| Feature | Core | Status | Queued | Labeled | Exact | FP | Ambig | Precision | Paper treatment |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `contrastive_negation` | yes | pass | 220 | 61 | 57 | 3 | 0 | 0.934426 | Interim precision-supported |
| `stock_openers` | yes | pass | 420 | 68 | 67 | 1 | 0 | 0.985294 | Interim precision-supported |
| `stock_closers` | yes | pass | 420 | 50 | 48 | 2 | 0 | 0.960000 | Interim precision-supported |
| `slop_lexicon` | yes | pass | 420 | 50 | 50 | 0 | 0 | 1.000000 | Interim span/item precision-supported |
| `rule_of_three_approx` | yes | demote | 420 | 50 | 28 | 19 | 2 | 0.560000 | Demoted for independent claims; exploratory coordinated-triple surface only |
| `stock_openers_closers` | derived | derived | 0 | 0 | 0 | 0 | 0 | undefined | Treat as derived pooled convenience metric |
| `eqbench_slop_score` | no | unlabeled | 20 | 0 | 0 | 0 | 0 | undefined | Exploratory aggregate benchmark bridge |
| `slop_lexicon_v2_candidate` | no | unlabeled | 20 | 0 | 0 | 0 | 0 | undefined | Exploratory candidate lexicon |

## Table 5. Paper-Safe Negative Claims

These are important because the project should not overstate the bounded
evidence.

| Avoided claim | Replacement |
|---|---|
| DPO universally creates slop | Some markers amplify at specific stages and differ by ladder |
| Preference data are uniformly more slop-like | Dolci DPO chosen/rejected differences are feature-specific and register-specific |
| Alignment simply adds hedging | Pybiber shows a broader shift toward nominalized answer-register prose |
| EQ-Bench Slop Score is the outcome variable | EQ-Bench is an aggregate bridge; per-feature propensity remains the causal measurement layer |
| Detector clusters are human-perceived slop | Detector clusters are candidate Tier-3 features until human perceptibility labels exist |
| Reduced SGLang panels equal the original exhaustive grid | The paper estimand is the bounded production scope |
