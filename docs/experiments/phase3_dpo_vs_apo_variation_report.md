# Phase 3 DPO-vs-APO Variation Report

Date: 2026-06-16

## Scope

This report compares the current bounded OLMo 3/DPO and SmolLM3/APO evidence
using the already assembled Phase 3 artifacts:

- OLMo reduced SGLang spectrum:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv`
- OLMo reduced classifier:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_feature_classification_t07_long1024.csv`
- SmolLM3 no_think production spectrum:
  `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_production.csv`
- SmolLM3 no_think production classifier:
  `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_production.csv`
- Cross-ladder alignment:
  `artifacts/phase3/analysis/olmo3_reduced_sglang_vs_smollm3_no_think_production_aligned.csv`
- Cross-ladder summary:
  `artifacts/phase3/analysis/olmo3_reduced_sglang_vs_smollm3_no_think_production_summary.md`

This is not a claim that all `EXPERIMENTS.md` Phase 3 cells are complete. It
summarizes the DPO-vs-APO variation visible in the current bounded measured
overlap.

## Main Read

The shared result is not "DPO/APO uniformly creates slop." The measured result
is a stage-localization difference:

- OLMo 3/DPO shows a `slop_lexicon` preference-stage AF rise that is classified
  as dynamics-driven. Its normalized AF moves base `1.467`, SFT `1.695`, DPO
  `1.999`, final/RLVR `1.659`; paired Dolci preference data do not show
  chosen-greater-rejected support for `slop_lexicon`.
- SmolLM3/APO shows a much higher `slop_lexicon` AF level before preference
  optimization and is classified as SFT-amplified. Its normalized AF moves
  base `6.902`, SFT `7.647`, APO/DPO `8.141`, final `8.345`; paired
  SmolLM3 preference data are complicit, but SFT already explains the main
  class assignment.
- The cross-ladder AF rank correlation is positive over the measured shared
  AF cells: Spearman `0.762`, Pearson `0.978`, with 8 shared non-missing AF
  values.

The right interpretation is shared surface-style pressure with different
stage localization: DPO in OLMo exposes a clearer preference-stage dynamics
effect; APO/final in SmolLM3 continues an already SFT-amplified style profile.

## Feature-Level Comparison

| Feature | OLMo 3 class | SmolLM3 class | DPO/APO read |
|---|---|---|---|
| `slop_lexicon` | `preference-amplified`, `dynamics-driven` | `sft-amplified`, not preference-amplified | Strongest algorithm-path contrast. OLMo localizes the main jump to DPO without preference-data complicity; SmolLM3 starts high at SFT and remains high through APO/final. |
| `rule_of_three_approx` | measured, no Phase 3 class | measured, no Phase 3 class | Both ladders show output visibility but teacher-forced AF stays below one. Free-running rates rise at SFT/APO more strongly in SmolLM3 than OLMo. |
| `contrastive_negation` | observed-output-only | sparse addendum: `sft-amplified` | Preference data are complicit in both ladders, but SmolLM3 teacher-forced support has only 2 held-out target references in the 2,258-prompt addendum, so the main production read remains output-side/caveated. |
| `stock_closers` | `sft-amplified` | sparse addendum: `preference-amplified`, dynamics-driven | OLMo has measurable high AF. The SmolLM3 sparse addendum now has teacher-forced support, but only 3 held-out references; generated-output rates increase after base. |
| `stock_openers` | measured, no Phase 3 class | observed-output-only | OLMo generated rates rise strongly after SFT/DPO/final; SmolLM3 has weaker and less monotonic generated-output movement. |
| `stock_openers_closers` | `sft-amplified` | sparse addendum: `preference-amplified`, dynamics-driven | OLMo pooled stock features show SFT amplification and rising generated rates. SmolLM3 now has sparse teacher-forced support, but only 3 held-out references in the addendum scope. |

## Shared AF Cells

Only `rule_of_three_approx` and `slop_lexicon` currently have non-missing AFs
in both bounded ladders.

| Feature | Stage | OLMo AF | SmolLM3 AF | SmolLM3 - OLMo |
|---|---|---:|---:|---:|
| `rule_of_three_approx` | base | `0.721` | `0.673` | `-0.049` |
| `rule_of_three_approx` | SFT | `0.767` | `0.741` | `-0.027` |
| `rule_of_three_approx` | DPO/APO | `0.716` | `0.755` | `0.039` |
| `rule_of_three_approx` | final | `0.723` | `0.755` | `0.032` |
| `slop_lexicon` | base | `1.467` | `6.902` | `5.435` |
| `slop_lexicon` | SFT | `1.695` | `7.647` | `5.952` |
| `slop_lexicon` | DPO/APO | `1.999` | `8.141` | `6.142` |
| `slop_lexicon` | final | `1.659` | `8.345` | `6.686` |

`rule_of_three_approx` is consistently below one in both ladders, so it is not
an amplification example under the teacher-forced contract. `slop_lexicon` is
above one in both ladders, but the SmolLM3 level is much higher at every
stage.

## Generated-Output Stage Movement

Generated-output rates are not a substitute for teacher-forced AF, but they
are useful for comparing visible final-output style.

| Feature | OLMo SFT->DPO free-run delta | SmolLM3 SFT->APO free-run delta | OLMo DPO->final delta | SmolLM3 APO->final delta |
|---|---:|---:|---:|---:|
| `contrastive_negation` | `0.122` | `-0.013` | `0.023` | `0.014` |
| `rule_of_three_approx` | `0.490` | `0.954` | `-0.132` | `0.086` |
| `slop_lexicon` | `0.065` | `0.069` | `-0.015` | `0.011` |
| `stock_closers` | `0.008` | `0.020` | `-0.002` | `-0.004` |
| `stock_openers` | `0.137` | `0.077` | `0.053` | `-0.019` |
| `stock_openers_closers` | `0.145` | `0.098` | `0.051` | `-0.023` |

Visible generated-output movement broadly supports the idea that both ladders
move toward assistant-register surface forms after SFT/preference training.
However, the detailed direction is not identical. OLMo shows stronger
post-DPO/final movement in stock opener features, while SmolLM3 shows stronger
rule-of-three generated-output levels.

## Interpretation

The bounded DPO-vs-APO evidence separates three mechanisms:

- **Inherited or SFT-transmitted style:** SmolLM3 `slop_lexicon` is already
  high at SFT, and OLMo stock opener/closer pools are SFT-amplified.
- **Preference-stage dynamics:** OLMo `slop_lexicon` rises at DPO despite
  non-complicit Dolci chosen-vs-rejected data. This is the cleanest
  dynamics-driven result in the current Phase 3 slice.
- **Output-side register shifts without AF support:** several stock and
  contrastive features move in generated text but lack reliable teacher-forced
  opportunity support in one or both ladders.

This means DPO-vs-APO variation is real in the measured slice. The algorithms
do not produce the same stage path, but they land in overlapping regions of
assistant-style output space.

## Caveats

- SmolLM3's preference cell is APO/model-soup/final-path evidence, not a clean
  one-checkpoint DPO analogue.
- OLMo uses a reduced SGLang generation route rather than the abandoned
  5,000-prompt x 8-completion x 3-temperature full grid.
- Teacher-forced support is currently shared only for `slop_lexicon` and the
  `rule_of_three_approx` proxy. Other features should not be interpreted as
  cross-ladder AF comparisons until denominator support improves.
- The optional OLMo Instruct/Think/RL-Zero comparison is now measured as a
  reduced final-answer-mode SGLang panel. It is useful output-side path
  evidence, but it does not add teacher-forced AF cells to this DPO-vs-APO
  comparison.

## Bottom Line

The current evidence supports a nuanced Phase 3 answer: DPO and APO ladders
share a slop/style pressure, but the mechanism and stage localization differ.
OLMo's clearest `slop_lexicon` result is a DPO-stage dynamics effect; SmolLM3's
clearest `slop_lexicon` result is an already-high SFT-amplified profile that
persists through APO/final.
