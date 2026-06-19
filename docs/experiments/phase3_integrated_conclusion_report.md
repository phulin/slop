# Phase 3 Integrated Conclusion Report

Date: 2026-06-16

## Scope

This report summarizes the current Phase 3 evidence after replacing the
abandoned exhaustive OLMo generation grid with the completed reduced SGLang
plan. It is written as the current bounded Phase 3 conclusion, not as a claim
that every item in `EXPERIMENTS.md` is fully complete.

The current completed generation scope is:

- OLMo reduced SGLang main style panel: 8,192 generations.
- OLMo reduced SGLang temperature panel: 6,144 generations.
- OLMo reduced SGLang long compounding panel: 2,048 generations.
- SmolLM3 no_think production-shape panel: 49,152 generations.
- Optional OLMo Instruct/Think/RL-Zero SGLang stretch panel: 5,120
  final-answer-mode generations.

The current integrated OLMo spectrum is:

- `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv`
- `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_feature_classification_t07_long1024.csv`

The current SmolLM3 production spectrum is:

- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_production.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_production.csv`

The current SmolLM3 sparse teacher-forced addendum is:

- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_2258prompt_sparse_supported.csv`
- `artifacts/phase3/analysis/smollm3_no_think_feature_classification_2258prompt_sparse_supported.csv`

## Main Conclusion

The bounded Phase 3 evidence does not support the simple claim that preference
tuning makes every slop feature worse. The more precise result is stage- and
feature-specific:

- In OLMo 3, `slop_lexicon` is the cleanest preference-stage amplification
  signal. Its normalized teacher-forced AF rises from base `1.467` and SFT
  `1.695` to DPO `1.999`, then falls at final/RLVR to `1.659`.
- The OLMo `slop_lexicon` preference-stage increase is classified as
  dynamics-driven, not signal-driven, because paired Dolci chosen-vs-rejected
  evidence does not show chosen-greater-rejected support for the feature
  (`BH q=0.963`; chosen-minus-rejected is negative in aggregate).
- OLMo free-running `slop_lexicon` in the reduced long-output panel peaks at
  DPO: base `0.278`, SFT `0.317`, DPO `0.381`, final/RLVR `0.366` hits per
  1k generated tokens.
- OLMo self-conditioning is present for `slop_lexicon`: observed minus
  teacher-forced expected rates are positive at every stage, with excess per
  1k opportunities base `0.214`, SFT `0.223`, DPO `0.261`, and final/RLVR
  `0.198`.
- OLMo temperature-dependent realized AF is now measured on the reduced
  SGLang temperature panel. For `slop_lexicon`, the largest realized AF is SFT
  at temperature `1.0` (`1.359`); DPO remains near one across temperatures
  (`1.154`, `1.012`, `1.107` at `0.0`, `0.7`, and `1.0`), which means the
  teacher-forced DPO propensity peak does not translate into the largest
  realized temperature-panel AF.
- SmolLM3 no_think temperature-dependent realized AF is also measured. Its
  `slop_lexicon` realized AF is already high at SFT and post-preference
  checkpoints, with DPO `2.070`, `1.972`, `2.285` and final `1.645`,
  `2.011`, `2.279` at temperatures `0.0`, `0.7`, and `1.0`.
- SmolLM3 no_think `slop_lexicon` temperature CIs are now measured through a
  materialized counter-cache bootstrap. That gives uncertainty for the primary
  temperature curve without rescoring all generation text for every auxiliary
  feature.
- Stock openers and pooled stock openers/closers show strong generated-output
  stage shifts, but their teacher-forced AF pattern is not a preference-stage
  `slop_lexicon`-like rise. In the reduced classifier, `stock_closers` and
  `stock_openers_closers` are SFT-amplified, while `stock_openers` remains
  measured but not assigned a Phase 3 amplification class.
- The new SmolLM3 2,258-prompt sparse teacher-forced addendum changes the
  support picture for three previously output-only features, but not the main
  production conclusion. It finds raw AFs rising from base to SFT/APO/final for
  `contrastive_negation`, `stock_closers`, and `stock_openers_closers`; the
  sparse addendum classifier labels the two stock closer features as
  preference-amplified/dynamics-driven. This is low-reference evidence:
  held-out target references are 2 for `contrastive_negation` and 3 for each
  stock closer feature, while `stock_openers` still has zero references.
- `rule_of_three_approx` remains visible in outputs but uses a teacher-forced
  proxy contract and stays below one in normalized/raw AF terms across OLMo
  stages in the current bounded table.

## Cross-Ladder Read

The OLMo-reduced versus SmolLM3-production comparison aligns 24 feature-stage
rows and has 8 shared non-missing AF values. The AF rank correlation remains
positive:

- Spearman AF: `0.762`
- Pearson AF: `0.978`

This supports a bounded cross-ladder claim: some surface-style amplification
structure is shared across OLMo/DPO and SmolLM3/APO, but the stage path is not
identical. SmolLM3 no_think classifies `slop_lexicon` as SFT-amplified under
the current measured scope, whereas OLMo classifies it as preference-amplified
and dynamics-driven.

Detailed DPO-vs-APO variation report:
`docs/experiments/phase3_dpo_vs_apo_variation_report.md`.

## Statistical Evidence

The current Phase 3 table joins the available FDR-controlled evidence:

- Preference-data complicity uses paired chosen-vs-rejected tests from the
  Stage 1 pair analyses.
- Teacher-forced stage effects use paired opportunity-level sign tests where
  the opportunity contract exists.
- The SmolLM3 sparse teacher-forced addendum adds paired stage-effect tests for
  `contrastive_negation`, `stock_closers`, and `stock_openers_closers`: 9 rows,
  8 BH-FDR-significant at alpha `0.05`. These tests have many shared
  opportunity rows, but the AF denominator has only 2-3 target references per
  feature, so the AF intervals are wide and should be read as coverage-caveated
  addendum evidence.
- Free-running stage effects over the reduced OLMo long SGLang caches now use
  512 paired units per adjacent stage comparison, keyed by SGLang `prompt_id`.
  They report 10 BH-FDR-significant rows out of 18.
- Generated-output rate tables now carry document/prompt-cluster bootstrap
  CIs for per-1k rates when the corresponding JSONL cache is supplied, and
  the integrated spectra propagate those intervals into
  `free_run_per_1k_tokens_ci_low/high`.
- Compounding tables now carry document/prompt-cluster bootstrap CIs for
  observed/excess per-1k opportunity rates, realized AF, and prior/no-prior
  risk metrics in the reduced OLMo long panel, reduced OLMo temperature
  panel, SmolLM3 production `t=1.0` panel, and the supplemental SmolLM3
  three-temperature `slop_lexicon` table. The integrated spectra propagate
  the primary compounding CI columns.
- Biber-lite generated-output comparisons now carry document/prompt-cluster
  bootstrap CIs for generated register rates and generated-vs-corpus deltas
  and ratios, holding the aggregate corpus baselines fixed. The OLMo and
  SmolLM3 style signatures preserve those generated-side intervals as
  `value_ci_low/high` where available.

Primary generated-output `slop_lexicon` rate intervals:

- OLMo reduced long panel: base `0.278` `[0.183, 0.397]`, SFT `0.317`
  `[0.166, 0.477]`, DPO `0.381` `[0.269, 0.511]`, final/RLVR `0.366`
  `[0.254, 0.488]` hits per 1k generated tokens.
- SmolLM3 no_think production `t=1.0` panel: base `0.118`
  `[0.082, 0.165]`, SFT `0.310` `[0.268, 0.352]`, APO/DPO `0.379`
  `[0.332, 0.425]`, final `0.390` `[0.344, 0.445]` hits per 1k generated
  tokens.

Primary `slop_lexicon` compounding excess intervals:

- OLMo reduced long panel excess per 1k opportunities: base `0.214`
  `[0.066, 0.401]`, SFT `0.223` `[-0.042, 0.528]`, DPO `0.261`
  `[0.050, 0.488]`, final/RLVR `0.198` `[0.007, 0.383]`.
- SmolLM3 production `t=1.0` excess per 1k opportunities: base `0.090`
  `[0.003, 0.203]`, SFT `-0.372` `[-0.482, -0.248]`, APO/DPO `-0.456`
  `[-0.571, -0.332]`, final `-0.465` `[-0.590, -0.338]`.

Primary SmolLM3 `slop_lexicon` temperature compounding CI artifact:

- `artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_slop_lexicon_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3.csv`
- Counter cache:
  `artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_slop_lexicon_cluster_counts_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3.jsonl`

Important OLMo reduced free-running FDR reads:

- SFT to DPO significantly increases `stock_openers_closers`,
  `stock_openers`, `contrastive_negation`, `slop_lexicon`, and
  `rule_of_three_approx`.
- Base to SFT significantly increases stock opener features but decreases
  `contrastive_negation`, `rule_of_three_approx`, and `slop_lexicon` under
  the sign-test direction.
- DPO to final/RLVR has no BH-FDR-significant free-running changes in the
  reduced long panel.

## What Is Complete

The following bounded Phase 3 pieces are now complete and internally
consistent:

- Reduced OLMo SGLang generation panels.
- Reduced OLMo integrated amplification spectrum.
- Reduced OLMo classifier with preference, teacher-forced, free-run, and
  compounding evidence joined where available.
- Reduced OLMo `slop_lexicon` compounding decomposition and realized-AF plot.
- Reduced OLMo and SmolLM3 no_think temperature-dependent realized-AF
  decompositions and plots for supported features.
- SmolLM3 no_think production-shape generation, spectrum, classifier,
  Biber-lite/style-signature, and cross-ladder comparison.
- OLMo-reduced versus SmolLM3-production AF rank correlation.
- Generated-output document/prompt-cluster bootstrap CIs for the reduced OLMo
  long panel and SmolLM3 production `t=1.0` panel.
- Compounding document/prompt-cluster bootstrap CIs for the reduced OLMo
  long panel, reduced OLMo temperature panel, and SmolLM3 production `t=1.0`
  panel, plus the supplemental SmolLM3 three-temperature `slop_lexicon`
  curve.
- Generated-side Biber-lite bootstrap CIs for both OLMo and SmolLM3 style
  signatures.

## Practical Compute Strategy

The active strategy is to keep the bounded Phase 3 report useful now while
continuing original-scope parity through the SGLang path:

- Use SGLang for pure generation tasks. The completed OLMo reduced panels,
  optional path-comparison generations, and the original 5,000-prompt
  x 8-completion x 3-temperature OLMo recovery grid now all use SGLang
  fixed-budget generation. Transformers batch-size-32 recovery is legacy
  context, not the active production strategy.
- Keep generation panels purpose-specific: a paired main style panel, a small
  temperature panel, and a long-output compounding panel for the current
  report. These answer the style-progression, temperature, and compounding
  questions without waiting for the full 480,000-generation grid.
- Continue original-scope OLMo parity with resumable SGLang shards rather than
  Torch batch-size tuning. The control knobs are prompt chunk size, JSONL
  resume checkpoints, and shard order; the supervisor prioritizes a complete
  `t=1.0` four-stage slice for earlier downstream spectrum/classifier
  assembly.
- Separate generation from repeated analysis. The compounding analyzer can now
  materialize prompt-cluster counter caches and bootstrap from those counters,
  so rerunning CIs does not require rescanning every generated string.
- Treat `slop_lexicon` as the primary temperature-compounding target when
  compute is constrained; auxiliary-feature temperature CIs are lower priority
  unless they become decision-relevant.
- Stop spending Phase 3 effort on perfect pretraining-source coverage. The
  current baseline is sufficient for the bounded conclusions; remaining source
  gaps are audit caveats, not blockers.

## Remaining Full-Scope Gaps

Phase 3 is still not complete against the literal full `EXPERIMENTS.md` scope:

- Teacher-forced AF support is still missing or proxy-only for some retained
  features, especially sparse `contrastive_negation` and the
  `rule_of_three_approx` proxy contract.
- Cluster-bootstrap coverage is now present for teacher-forced AF where
  available, primary generated-output per-1k rate tables, and primary
  compounding tables, plus generated-side Biber-lite rates. The remaining CI
  gaps are corpus-side Biber baseline uncertainty, auxiliary-feature cells in
  the large SmolLM3 three-temperature compounding table, and any
  unmeasured/full-scope feature cells.
- The optional OLMo Instruct/Think/RL-Zero stretch comparison is now measured
  as a reduced final-answer-mode SGLang panel. It is output-side evidence only,
  not a new teacher-forced AF layer. In that panel, Instruct final is the
  maximum for every tracked Tier-1 output feature; for `slop_lexicon`, it
  reaches `0.807` per 1k final-answer tokens, while Think final is lower by
  `0.384` and the RL-Zero mean is lower by `0.244`.
- The abandoned 5,000-prompt x 8-completion x 3-temperature OLMo grid is not
  part of the active route; the completed reduced SGLang route is the current
  pragmatic replacement.

## Bottom Line

The bounded Phase 3 result is strong enough to support a current project
conclusion: slop amplification is not uniform, and the clearest OLMo
`slop_lexicon` effect localizes to the preference stage without direct
chosen-response complicity. That points to optimization or model-dynamics
effects rather than a simple imitation of higher-slop preference targets.

The SmolLM3 replication agrees that related style markers are amplified in
open post-training ladders, but it places the main `slop_lexicon` effect
earlier, at SFT. The cross-ladder result is therefore shared style-pressure
with different stage localization, not a single universal DPO/APO mechanism.
