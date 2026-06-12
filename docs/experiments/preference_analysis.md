# Result A Preference Analysis Notes

These notes define the first-pass analysis for Result A, using the per-pair
CSV emitted by `slop-census --pair-output`. The goal is an auditable early read
on chosen-vs-rejected feature differences before the full Wilcoxon and logistic
models in `EXPERIMENTS.md`.

## Inputs

Input is the `--pair-output` CSV from `slop-census`, with one row per
`source`, `subset`, `pair_id`, and `feature`:

- `chosen_count`
- `rejected_count`
- `chosen_tokens`
- `rejected_tokens`
- `chosen_rate_per_1k_tokens`
- `rejected_rate_per_1k_tokens`
- `chosen_minus_rejected`

`chosen_minus_rejected` is the primary first-pass effect size: chosen rate per
1k tokens minus rejected rate per 1k tokens for the same pair and feature.
Rows with both responses present but zero feature counts are still informative
for direction tests and denominator checks. Rows with missing pair IDs or
missing chosen/rejected denominators should fail validation rather than being
silently dropped.

## Feature Summaries

For each `source`, `subset`, and `feature`, summarize paired deltas with:

- pair count
- mean, median, and trimmed mean of `chosen_minus_rejected`
- standard deviation and bootstrap confidence interval for the mean delta
- share of pairs with positive, negative, and zero deltas
- aggregate chosen and rejected counts
- aggregate chosen and rejected tokens
- aggregate chosen and rejected rates per 1k tokens

The signed direction is always chosen minus rejected. Positive values mean the
feature is more frequent in chosen responses after per-pair token
normalization; negative values mean it is more frequent in rejected responses.
Report both normalized deltas and raw aggregate counts because sparse features
can have unstable per-pair rates.

## First-Pass Sign Test

Run a paired sign test per feature as the initial robustness screen:

1. Drop ties where `chosen_minus_rejected == 0` for the sign-test statistic.
2. Count positive and negative non-tied pairs.
3. Use a two-sided exact binomial test with null probability 0.5.
4. Record the effect direction from `sign(positive_pairs - negative_pairs)`.
5. Keep the tie count and non-tied pair count in the report.

Apply Benjamini-Hochberg FDR correction across features within the planned
Result A family. For early Dolci-only runs, correct across all Tier-1 features
in the run and label the family explicitly, for example
`dolci_dpo_tier1_first_pass`. When SmolLM3/Tulu preference data is added, keep
its correction family separate unless the analysis is intentionally pooled.

Minimum report fields:

- `source`
- `subset`
- `feature`
- `pairs`
- `positive_pairs`
- `negative_pairs`
- `zero_pairs`
- `mean_delta`
- `median_delta`
- `mean_chosen_rate`
- `mean_rejected_rate`
- `mean_chosen_tokens`
- `mean_rejected_tokens`
- `mean_token_delta`
- `sign_test_p`
- `bh_q`
- `direction`
- `fdr_significant`

This sign-test screen is deliberately conservative about magnitude: it tests
whether the paired direction is consistent, not whether large deltas are
concentrated in a small number of pairs.

## Length-Aware First-Pass Logistic Model

As a supplemental first-pass check, fit a response-side ridge logistic
diagnostic from the same pair-delta CSV. This is still an early diagnostic, not
the full Result A logistic analysis.

For each `source`, `subset`, and `feature`, expand every preference pair into
two response-side observations:

- chosen response: `chosen_label = 1`, `feature_rate =
  chosen_rate_per_1k_tokens`, `tokens = chosen_tokens`
- rejected response: `chosen_label = 0`, `feature_rate =
  rejected_rate_per_1k_tokens`, `tokens = rejected_tokens`

Fit a logistic model with an intercept, standardized response-side feature
rate, and standardized response token length. Apply a small ridge penalty to
the feature and length coefficients to keep rare or directional features from
dominating the smoke-run diagnostic. The current first-pass report records the
feature coefficient, feature standard error, feature z score, feature odds
ratio, length coefficient, iteration count, and convergence flag.

Feature coefficients describe whether higher response-side feature rates are
associated with the chosen label after accounting for response length in this
diagnostic. Positive coefficients mean the feature is enriched on chosen
responses; negative coefficients mean it is enriched on rejected responses.
Length coefficients should be interpreted as residual length imbalance checks,
not as evidence that length caused the preference label.

Preserve length diagnostics with the same grouping keys so the full model can
control for residual length effects:

- `chosen_tokens`
- `rejected_tokens`
- `length_delta_tokens = chosen_tokens - rejected_tokens`
- `length_delta_abs = abs(length_delta_tokens)`
- `length_ratio = chosen_tokens / rejected_tokens` when rejected tokens are
  nonzero
- `total_pair_tokens = chosen_tokens + rejected_tokens`
- optional binned length deltas for calibration plots

Summarize each length covariate per feature and at the pair level before
feature expansion. The pair-level table is important because feature-expanded
rows repeat the same length delta for every feature. Dolci length
pre-filtering partially controls this confound, but the first-pass report
should still expose residual length imbalance and the correlation between
`chosen_minus_rejected` and `length_delta_tokens`.

The later logistic model in `EXPERIMENTS.md` should use paired observations in
long form, with one row for each response side, a binary chosen label, feature
counts/rates, length covariates, and pair-level clustering or fixed effects as
appropriate.

For small samples, rare features, or perfectly directional features, expect
separation and unstable coefficients. Report non-convergence, large standard
errors, and any regularization used instead of treating these coefficients as
final evidence.

## W&B Logging Expectations

Do not log raw prompt or response text. Log aggregate metrics, compact tables,
and artifact references only.

Expected W&B contents for a first-pass Result A run:

- config: input artifact/path, matcher version, source dataset, subset, sample
  seed, correction family, and git commit when available
- metrics: `preference/features`, `preference/significant_features`, and
  `preference/pair_delta_rows`
- table: `preference_feature_analysis`, one row per source/subset/feature with
  the summary and sign-test/BH-FDR fields above, plus the first-pass ridge
  logistic diagnostic columns when available
- length diagnostics are included in `preference_feature_analysis` as
  `mean_chosen_tokens`, `mean_rejected_tokens`, and `mean_token_delta`; the
  richer pair-level diagnostics can be logged as `preference_length_diagnostics`
- artifact: local first-pass report CSV/Parquet plus the original
  `--pair-output` CSV reference or content hash

The census CLI may log up to 1,000 raw pair-delta rows as
`preference_pair_deltas`; the analysis run should log the complete derived
feature summaries and keep the full local pair CSV as the source of truth.

## Limitations Versus Full Result A Plan

This first pass is not the final Result A statistical analysis from
`EXPERIMENTS.md`.

- The sign test ignores delta magnitudes; the planned paired Wilcoxon
  signed-rank test uses signed ranks and is more sensitive to consistent
  nonzero magnitudes.
- BH-FDR here is a first-pass correction over the available feature family; the
  final paper analysis should correct across the frozen full feature set.
- Per-feature sign tests do not estimate marginal preference association when
  features co-occur. The planned logistic model is needed to estimate each
  feature's association with the chosen label while controlling for length and
  other covariates.
- The length-aware first-pass logistic model is a diagnostic bridge only; small
  samples and separation can dominate coefficients, so it does not replace the
  planned Wilcoxon/logistic Result A analysis.
- Per-1k token deltas can be noisy for very short responses or rare features;
  raw counts, token denominators, and sparse-feature flags must accompany any
  interpretation.
- For Dolci Delta-Learning data, chosen-vs-rejected differences measure the
  strong-vs-weak model style signal being distilled, not human preference per
  se. The SmolLM3/Tulu preference subset is needed for the human/RM-ranked
  comparison described in the main plan.

Use this note to produce an early Result A table and go/no-go check. Treat any
large, directional Tier-1 effects as hypotheses to confirm with the full
Wilcoxon/logistic analysis, not as standalone causal claims.
