# Phase 3 Status

Date: 2026-06-15

## Scope

Phase 3 in `EXPERIMENTS.md` asks for the main amplification spectrum:
features x Phase 1 data rates x Phase 2 teacher-forced AF/free-running rates,
then feature-level classifications:

- inherited
- SFT-amplified
- preference-amplified, with signal-driven vs. dynamics-driven cause
- compounding-dominant

It also asks for statistics across the full feature set and SmolLM3
replication with cross-ladder AF rank correlation.

This status file records the first bounded Phase 3 implementation over the
retained OLMo 3/Dolci single-temperature Phase 2 artifacts. It is real Phase 3
progress, but it is not the full EXPERIMENTS.md Phase 3 completion because the
SmolLM3 ladder, p-value/FDR layer, and full production grid are not present.

## Implemented Phase 3 Layer

Added CLI:

- `slop-classify-amplification-spectrum`

Inputs:

- `artifacts/phase2/analysis/olmo3_amplification_spectrum_single_temp_t1_v6.csv`

Outputs:

- `artifacts/phase3/analysis/olmo3_phase3_bounded_feature_classification_t1.csv`
- `artifacts/phase3/analysis/olmo3_phase3_bounded_feature_classification_t1_summary.md`
- `artifacts/phase3/analysis/olmo3_phase3_bounded_artifact_manifest.csv`
- `artifacts/phase3/analysis/olmo3_phase3_bounded_artifact_manifest.json`
- `artifacts/phase3/analysis/olmo3_phase3_bounded_artifact_manifest.md`

W&B:

- `stage3-phase3-olmo3-bounded-feature-classification-t1-v2` (`3ptb39c2`)
- `stage3-phase3-olmo3-bounded-artifact-manifest` (`tqxvi3g1`)

The classifier emits one row per feature with:

- Phase 3 primary class.
- preference cause (`signal-driven`, `dynamics-driven`, or
  `not-preference-amplified`).
- Phase 1 data rates.
- base/SFT/DPO/final AF values.
- AF stage jumps.
- free-running stage rates.
- compounding summary fields where available.
- explicit `fdr_status=not_computed_missing_p_values`.

## Classification Rules

Default rule settings:

| Rule | Threshold |
|---|---:|
| High SFT-data rate | `>= 0.25` per 1k tokens |
| AF approximately 1 | within `+/- 0.25` |
| Amplified AF | `>= 1.25` |
| Preference-stage absolute AF jump | `>= 0.25` from max(base, SFT) to max(DPO, final) |
| Preference-stage relative AF jump | `>= 0.10` |
| Preference-data complicity | chosen - rejected `> 0.0` per 1k tokens |
| Compounding-dominant | max AF `<= 1.25`, max compounding excess `>= 0.1`, max realized AF `>= 1.1` |

The relative-jump rule is important for high-magnitude AF features such as
`stock_closers`, where a tiny relative change should not be labeled as a
preference-stage jump.

## Current Bounded OLMo Classifications

| Feature | Class | Cause | Key read |
|---|---|---|---|
| `slop_lexicon` | preference-amplified | dynamics-driven | DPO AF rises from SFT (`1.695`) to DPO (`1.999`), but DPO chosen data is slightly lower than rejected (`-0.032` per 1k), so the bounded label is dynamics-driven rather than signal-driven. |
| `stock_closers` | SFT-amplified | not preference-amplified | AF is already very high at base/SFT and does not clear the relative preference-jump rule. |
| `stock_openers_closers` | SFT-amplified | not preference-amplified | Pooled AF is high from the start; this remains mostly a closer-side effect from Phase 2. |
| `rule_of_three_approx` | measured-no-phase3-class | not preference-amplified | Teacher-forced comma-pair extension proxy is below 1 across stages and peaks at SFT, not DPO. |
| `stock_openers` | measured-no-phase3-class | not preference-amplified | Teacher-forced AF is far below 1 and declines through DPO/final. |
| `contrastive_negation` | observed-output-only | not preference-amplified | Free-running output exists, but teacher-forced support is missing and held-out references are sparse. |

Class counts:

| Class | Count |
|---|---:|
| preference-amplified | 1 |
| SFT-amplified | 2 |
| measured-no-phase3-class | 2 |
| observed-output-only | 1 |

## Phase 3 Requirement Audit

| EXPERIMENTS.md requirement | Current evidence | Status |
|---|---|---|
| Assemble features x data rates x AF/free-running rates | `olmo3_amplification_spectrum_single_temp_t1_v6.csv` has 24 feature-stage rows over six retained feature views and four OLMo stages. | Bounded OLMo done |
| Classify each feature as inherited/SFT/preference/compounding | `olmo3_phase3_bounded_feature_classification_t1.csv` classifies six retained feature views. | Bounded OLMo done |
| Cross-reference preference-amplified features with chosen-vs-rejected complicity | Classifier uses DPO chosen - rejected per-1k data-rate delta to label `slop_lexicon` dynamics-driven. | Bounded OLMo done |
| Compounding-dominant classification | Rule implemented; no current retained feature qualifies under default thresholds. | Implemented, no positive class |
| Cluster bootstrap CIs | Existing Phase 2 AF table includes bootstrap CIs where teacher-forced measurements exist. | Partially done from Phase 2 |
| Benjamini-Hochberg FDR across full feature set | Current retained spectrum does not include p-values, so classifier records `not_computed_missing_p_values`. | Missing |
| Paired designs wherever corpora share prompts | Phase 1 preference analysis exists separately; not folded into Phase 3 p-value/FDR output. | Partial |
| Replicate full spectrum on SmolLM3-3B no_think ladder | No SmolLM3 Phase 2 artifacts present. | Missing |
| Report cross-ladder AF rank correlation | Requires SmolLM3 spectrum. | Missing |
| Stretch Instruct vs. Think vs. RL Zero comparison | Not started. | Stretch missing |

## Current Interpretation

The bounded Phase 3 classification strengthens the final Phase 2 conclusion:

- The only retained feature view classified as preference-amplified is
  `slop_lexicon`.
- That preference-stage amplification is labeled dynamics-driven under the
  current rule because DPO chosen data is not higher than rejected for the
  feature.
- Pooled stock phrase behavior and stock closers are better described as
  already-amplified/high-AF features than as preference-stage jumps.
- `rule_of_three_approx` still does not support a DPO amplification claim.

The bounded Phase 3 result is therefore not "DPO makes everything sloppier."
It is: DPO is the clearest stage-localized propensity peak for the retained
`slop_lexicon` feature, while most other retained feature views are inherited,
already high, output-only, or unsupported by the teacher-forced proxy.

## Next Work Toward Full Phase 3

The next concrete work needed to complete Phase 3 from `EXPERIMENTS.md` is:

1. Add p-value production to the relevant Phase 1/2 analysis outputs, then
   compute Benjamini-Hochberg q-values in the Phase 3 classifier.
2. Decide whether full Phase 3 completion should use the bounded OLMo artifact
   shape or launch the original full production shape.
3. Build or run the SmolLM3 no_think Phase 1/2 ladder artifacts.
4. Assemble the SmolLM3 amplification spectrum with the same schema.
5. Add cross-ladder feature-level AF rank correlation and summarize DPO-vs-APO
   variation.
