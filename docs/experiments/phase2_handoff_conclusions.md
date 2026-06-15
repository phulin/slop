# Phase 2 Handoff Conclusions

Date: 2026-06-14

## Current Scope

Phase 2 is implemented and exercised for the OLMo 3 Instruct ladder as a
bounded measurement package, not as the full EXPERIMENTS.md production-scale
run. The final Phase 2 read fixes free-running generation at temperature
`1.0` and treats the earlier DPO temperature sweep only as sensitivity context.
The implemented components are:

- Held-out Dolci SFT prompt packaging with near-duplicate prompt filtering.
- Exact sequence teacher-forced propensity with shared-prefix KV caching,
  branch batching, bfloat16, `torch.compile`, bootstrap CIs, W&B logging, and
  grid assembly.
- Free-running emission generation with batched Transformers generation,
  `torch.compile`, W&B logging, generation-grid assembly, and compounding
  analysis.
- Result B compounding analysis joining free-running generations to
  teacher-forced propensity grids, including realized-AF temperature plots.

The scorer of record for exact teacher-forced propensity remains
torch/Transformers. SGLang or vLLM should only be revisited for free-running
generation throughput, or after a separate exactness benchmark reproduces the
torch fixed-branch probability-mass summaries.

## Final Single-Temperature Scope

The retained Phase 2 deliverable uses one free-running temperature:

- Temperature: `1.0`
- Top-p: `0.95`
- Main target-shape generation grid: base, SFT, DPO, and final/RLVR, each
  `512` prompts x `8` completions x up to `1,024` new tokens.
- Scale check: DPO-only `1,024` prompts x `8` completions at the same decoding
  settings.
- Teacher-forced grids: exact sequence scoring on the retained Tier-1
  opportunity contracts; no decoding temperature applies to these.

This is the final bounded OLMo/Dolci Phase 2 measurement package. The original
EXPERIMENTS.md three-temperature, 5,000-prompt, two-ladder production grid is
not part of this close-out.

## Retained Features

The Phase 2 core feature set inherited from the revised Phase 1 close-out is:

- `contrastive_negation`
- `rule_of_three_approx`
- `slop_lexicon`
- `stock_openers`
- `stock_closers`

Teacher-forced stage-localization claims are currently strongest for
`slop_lexicon` normalized by `neutral_common_controls`. `rule_of_three_approx`
now has a bounded comma-pair extension proxy, but that proxy measures
third-item continuation after a candidate two-item list rather than
open-vocabulary initiation of the whole construction.

## Teacher-Forced Read

The completed 1,024-prompt `slop_lexicon` vs `neutral_common_controls` grid
uses the 5,000-prompt held-out Dolci SFT package with `sample_size=1024`. All
four stages have matching denominators: 88,903 opportunities per feature, 52
slop references, and 6,694 neutral references.

W&B runs:

- Base: `4mn6ycwv`
- SFT: `hoggmwz5`
- DPO: `tu89kg31`
- Final/RLVR: `5v7x3180`
- Assembly: `1cuuzoax`

Neutral-normalized `slop_lexicon` AF:

| Stage | Normalized AF | 95% CI |
|---|---:|---:|
| Base | 1.467 | 1.114-2.046 |
| SFT | 1.695 | 1.315-2.289 |
| DPO | 1.999 | 1.446-2.837 |
| Final/RLVR | 1.659 | 1.189-2.351 |

Interpretation: DPO is the maximum point-estimate stage, but the confidence
intervals overlap and final/RLVR attenuates near SFT. Treat this as inherited
or base slop propensity with a modest DPO-stage peak, not as a clean
preference-only effect.

The full 5,000-prompt teacher-forced slop/neutral run remains a confirmatory
compute spend, not the automatic next step.

The completed 5,000-prompt `rule_of_three_approx` teacher-forced proxy grid
uses the comma-pair extension contract over the full held-out prompt package.
All four stages share 3,281 opportunities, 1,473 references, and reference rate
`0.449`.

W&B runs:

- Base: `hd0k5hfn`
- SFT: `ctpb9eu6`
- DPO: `l8indgao`
- Final/RLVR: `9b60jls5`
- Assembly: `sod6yjyj`

Raw AF:

| Stage | Raw AF | 95% CI |
|---|---:|---:|
| Base | 0.721 | 0.698-0.742 |
| SFT | 0.767 | 0.743-0.787 |
| DPO | 0.716 | 0.691-0.738 |
| Final/RLVR | 0.723 | 0.698-0.746 |

Interpretation: this proxy does not support a DPO-stage rule-of-three
amplification claim. SFT is the maximum point-estimate stage, DPO is slightly
below base, and all stages put less probability mass on extension than the
reference continuation rate.

## Free-Running Read

The largest current free-running grid uses the full 512-row held-out prompt
package, temperatures `0.0`, `0.7`, and `1.0`, top-p `0.95`, one completion per
prompt, and 128 generated tokens per completion.

W&B runs:

- Base: `8rovnqs9`
- SFT: `aevb3cbt`
- DPO: `6o6cpvm0`
- Final/RLVR: `8q42t9h3`
- Assembly: `o135ar2v`
- Compounding analysis joined to the 1024 teacher-forced grid: `6miductd`
  (supersedes the earlier 512-grid joins `fwj8byv8` and `y43cw1k9` for current
  Result B interpretation)

`slop_lexicon` rates per 1k generated tokens:

| Temperature | Base | SFT | DPO | Final/RLVR |
|---:|---:|---:|---:|---:|
| 0.0 | 0.214 | 0.320 | 0.412 | 0.366 |
| 0.7 | 0.244 | 0.259 | 0.427 | 0.305 |
| 1.0 | 0.320 | 0.366 | 0.458 | 0.443 |

Interpretation: DPO is the maximum free-running stage at all three
temperatures in the 512-prompt grid, aligning better with the teacher-forced
DPO point-estimate peak than the earlier 128-prompt generation slice did.
Final/RLVR remains close to DPO at temperature 1.0.

## Compounding Read

There are now two bounded compounding analyses joined to the 1,024-prompt
teacher-forced stage grid.

The temperature-sweep join uses the 512-prompt, one-completion, 128-token
generation grid. Observed `slop_lexicon` opportunity rates exceed the
teacher-forced expectation in every stage/temperature cell. Its headline cells:

- Max excess: SFT at temperature 1.0, observed `0.790` vs expected `0.336` per
  1k opportunities, excess `0.454`, observed/expected `2.35`, realized AF
  `1.35`.
- Max realized AF: DPO at temperature 1.0, realized AF `1.506`; final
  temperature 1.0 is effectively tied at `1.504`.
- Direct prior/no-prior window signal is positive in all `slop_lexicon` cells,
  but repeat counts are still sparse.

The target-shape join uses the newer 512-prompt x 8-completion x 1,024-token
generation caches at temperature 1.0. Observed `slop_lexicon` opportunity rates
again exceed teacher-forced expectation in all four stages:

| Stage | Observed/1k Opp | Expected/1k Opp | Excess/1k | Realized AF | Repeat Gens |
|---|---:|---:|---:|---:|---:|
| Base | 0.615 | 0.232 | 0.383 | 1.051 | 192 |
| SFT | 0.697 | 0.336 | 0.361 | 1.192 | 145 |
| DPO | 0.638 | 0.439 | 0.199 | 1.091 | 199 |
| Final/RLVR | 0.582 | 0.445 | 0.137 | 0.994 | 180 |

Interpretation: Result B has a bounded positive signal, and the target-shape
join gives much denser repeat counts than the shorter temperature sweep. The
effect is still not a clean DPO peak: base has the largest absolute excess, SFT
has the highest realized AF and strongest prior-window risk difference, and
final/RLVR is near the teacher-forced reference-rate baseline.

## Target-Shape Generation Shards

The first larger base, SFT, DPO, and final/RLVR target-shape generation shards
are complete on the 5,000-prompt package:

- Shape: 512 prompts, 8 completions per prompt, temperature `1.0`, top-p
  `0.95`, 1,024 generated tokens per completion.
- Base W&B: `nw6yoj5u`; 4,096 generations, 4,174,328 generated tokens,
  `357.5` generated tokens/sec.
- SFT W&B: `il07fjn1`; 4,096 generations, 3,845,808 generated tokens,
  `359.4` generated tokens/sec.
- DPO W&B: `8rud2kxl`; 4,096 generations, 4,194,304 generated tokens,
  `356.7` generated tokens/sec.
- Final/RLVR W&B: `b8xo8axn`; 4,096 generations, 4,187,136 generated tokens,
  `356.1` generated tokens/sec.
- Assembly W&B: `bojfccyx`.

Feature rates per 1k generated tokens:

| Feature | Base | SFT | DPO | Final/RLVR | Max Stage |
|---|---:|---:|---:|---:|---|
| `rule_of_three_approx` | 1.019 | 0.488 | 0.861 | 0.802 | Base |
| `slop_lexicon` | 0.233 | 0.171 | 0.229 | 0.193 | Base |
| `contrastive_negation` | 0.136 | 0.041 | 0.141 | 0.142 | Final/RLVR |
| `stock_openers_closers` | 0.092 | 0.072 | 0.108 | 0.091 | DPO |
| `stock_openers` | 0.064 | 0.052 | 0.064 | 0.059 | DPO |
| `stock_closers` | 0.028 | 0.020 | 0.043 | 0.032 | DPO |

Throughput is essentially unchanged from the 64-prompt target-shape benchmark
(`355.9` generated tokens/sec), so the current Torch/Transformers backend scales
linearly across prompts but remains slow in absolute terms for long generation.
The science read is more mixed than the 1,024-prompt teacher-forced grid:
base is narrowly above DPO for `slop_lexicon` (`0.233` vs. `0.229` per 1k
tokens) and clearly above all post-training checkpoints for
`rule_of_three_approx`. DPO remains the maximum for stock opener/closer
features, final/RLVR attenuates from DPO on most features, and SFT is the
low-emission point across all tracked features. Treat this as inherited/base
free-running style plus feature-specific post-training reshaping, not as a
clean DPO-stage generation peak.

The first scaled DPO target-shape shard is also complete:

- Shape: 1,024 prompts, 8 completions per prompt, temperature `1.0`, top-p
  `0.95`, 1,024 generated tokens per completion.
- Generation W&B: `p4v56vda`; 8,192 generations and 8,388,608 generated
  tokens.
- Scale-comparison W&B: `dezwh1xv`; corrected compounding W&B: `nd1lnfjy`.
- `slop_lexicon` is stable against the 512-prompt DPO shard:
  `0.226` vs. `0.229` hits per 1k generated tokens.
- The 1,024-prompt DPO compounding join reports observed `slop_lexicon`
  `0.662` vs. expected `0.439` per 1k opportunities, excess `0.223`,
  realized AF `1.133`, and 383 repeat generations. The direct window test is
  still strongly positive: `P(hit after prior)=0.0695` vs.
  `P(hit no prior)=0.0119`.
- The pooled `stock_openers_closers` compounding row is generation-side
  support only for this join; after the pooled-hit counting fix it has 755
  hits and 70 repeat generations.

The bounded target-shape DPO temperature expansion is complete for
temperatures `0.0`, `0.7`, and `1.0`:

- `t=0.0` W&B: `8s6spxu5`; 4,096 generations, 4,194,176 generated tokens,
  `0.206` `slop_lexicon` hits per 1k generated tokens.
- `t=0.7` W&B: `8no9vqyf`; 4,096 generations, 4,188,320 generated tokens,
  `0.223` `slop_lexicon` hits per 1k generated tokens.
- `t=1.0` W&B: `8rud2kxl`; 4,096 generations, 4,194,304 generated tokens,
  `0.229` `slop_lexicon` hits per 1k generated tokens.
- Assembly W&B: `5azmz6pq`.

Feature rates per 1k generated tokens:

| Feature | t=0.0 | t=0.7 | t=1.0 | Max Temp |
|---|---:|---:|---:|---|
| `slop_lexicon` | 0.206 | 0.223 | 0.229 | 1.0 |
| `stock_openers_closers` | 0.101 | 0.107 | 0.108 | 1.0 |
| `rule_of_three_approx` | 0.864 | 0.888 | 0.861 | 0.7 |
| `contrastive_negation` | 0.168 | 0.127 | 0.141 | 0.0 |

Interpretation: DPO target-shape temperature dependence is modest across this
range. Warmer decoding mildly increases `slop_lexicon` and pooled stock
opener/closer rates, but the effect is not large enough to explain the main
stage-localization results. Deterministic decoding does not remove these
features and is highest for `contrastive_negation`.

The matching DPO target-shape compounding join against the 1,024-prompt
teacher-forced grid is logged as W&B run `k6gcu75b`. `slop_lexicon` Result B is
positive at all three temperatures:

| Temp | Observed /1k Opp | Expected /1k Opp | Excess /1k | Realized AF | Repeat Gens |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.615 | 0.439 | 0.175 | 1.051 | 160 |
| 0.7 | 0.637 | 0.439 | 0.198 | 1.089 | 204 |
| 1.0 | 0.638 | 0.439 | 0.199 | 1.091 | 199 |

Interpretation: the self-conditioning/compounding signal is stable across DPO
temperatures at this target shape. Temperature moves the magnitude slightly,
but the larger result is that observed generation rates exceed teacher-forced
expectation even under deterministic decoding.

## Final Single-Temperature Amplification Spectrum

The final single-temperature headline table is assembled by
`slop-assemble-amplification-spectrum` and logged as W&B run `jxqcadhe`
(`stage2-phase2-olmo3-amplification-spectrum-single-temp-t1-v6`). Local
outputs:

- `artifacts/phase2/analysis/olmo3_amplification_spectrum_single_temp_t1_v6.csv`
- `artifacts/phase2/analysis/olmo3_amplification_spectrum_single_temp_t1_v6_summary.md`

It contains 24 rows: six retained feature views across base, SFT, DPO, and
final/RLVR. Each row joins Phase 1 corpus rates, available teacher-forced AF,
target-shape free-running rates, compounding summaries, and denominator-support
notes. The v6 table uses the corrected `stock_openers_closers` direct
prior/no-prior accounting in the target-shape compounding CSV, the 5,000-prompt
`rule_of_three_approx` comma-pair extension teacher-forced proxy, the
5,000-prompt pooled `stock_openers_closers` teacher-forced grid, and separate
5,000-prompt `stock_openers` and `stock_closers` teacher-forced grids.
Teacher-forced coverage is 20 cells. Blank cells mean missing measurements,
not zero effects.

The retained local artifact manifest is W&B run `oxc252zk`
(`stage2-phase2-olmo3-single-temp-t1-final-artifact-manifest`), superseded by
the Biber-inclusive manifest W&B run `p3ehymfc`
(`stage2-phase2-olmo3-single-temp-t1-final-artifact-manifest-v2`). Local
outputs:

- `artifacts/phase2/analysis/olmo3_phase2_single_temp_t1_final_artifact_manifest.csv`
- `artifacts/phase2/analysis/olmo3_phase2_single_temp_t1_final_artifact_manifest.json`
- `artifacts/phase2/analysis/olmo3_phase2_single_temp_t1_final_artifact_manifest.md`

Current spectrum read:

- `slop_lexicon` has the strongest teacher-forced evidence: neutral-normalized
  AF peaks at DPO (`1.999`) and falls at final/RLVR (`1.659`), while
  target-shape free-running rates put base (`0.233` per 1k generated tokens)
  just above DPO (`0.229`).
- `rule_of_three_approx` is the clearest target-shape free-running feature but
  currently peaks at base (`1.019` per 1k generated tokens). The 5,000-prompt
  comma-pair extension teacher-forced proxy peaks at SFT (`0.767` raw AF) and
  does not show a DPO peak.
- Pooled stock openers/closers have high teacher-forced raw AF in every stage,
  while target-shape free-running stock phrase rates are small and DPO-peaked.
  The split grids explain the pooled teacher-forced signal: `stock_openers`
  raw AF is far below 1 and declines from base (`0.067`) to DPO/final
  (`0.035`), while `stock_closers` raw AF is very high in every stage
  (`65.076`-`74.032`) because the model assigns roughly 3% mass to closer
  phrases against only 18 references in 41,187 opportunities.
- `contrastive_negation` remains too sparse in the 5k denominator audit for a
  strong bounded conclusion.

Treat this as the final bounded single-temperature OLMo/Dolci Phase 2
amplification spectrum, not the full production-scale EXPERIMENTS.md table.

The `rule_of_three_approx` comma-pair extension proxy supersedes the earlier
512-prompt mini-grid with a full 5,000-prompt teacher-forced grid. All stages
use the same 3,281 opportunities and 1,473 references, with reference rate
`0.449`.

| Stage | W&B | Mean Mass | Raw AF | 95% CI |
|---|---|---:|---:|---:|
| Base | `hd0k5hfn` | 0.324 | 0.721 | 0.698-0.742 |
| SFT | `ctpb9eu6` | 0.344 | 0.767 | 0.743-0.787 |
| DPO | `l8indgao` | 0.321 | 0.716 | 0.691-0.738 |
| Final/RLVR | `9b60jls5` | 0.325 | 0.723 | 0.698-0.746 |

Interpretation: the proxy is scoreable with the existing exact-sequence
harness, but it does not support DPO-stage amplification. SFT has the highest
point estimate, DPO is lowest, and every stage is below the reference extension
rate. Keep this separate from full `rule_of_three_approx` free-running rates
because it measures third-item extension after a candidate two-item list, not
open-vocabulary initiation of the whole construction.

The separate `stock_openers` teacher-forced grid uses the full 5,000-prompt
package and the document-start opportunity contract. All stages use 5,000
opportunities and 168 references, with reference rate `0.0336`.

| Stage | W&B | Mean Mass | Raw AF | 95% CI |
|---|---|---:|---:|---:|
| Base | `ul26mrrc` | 0.00226 | 0.067 | 0.059-0.078 |
| SFT | `140fbygd` | 0.00160 | 0.048 | 0.042-0.056 |
| DPO | `47jq4d4v` | 0.00117 | 0.035 | 0.030-0.041 |
| Final/RLVR | `8j59fpx5` | 0.00117 | 0.035 | 0.030-0.041 |

Interpretation: the model ladder assigns much less probability to the retained
stock opener starts than the held-out references actually use. This makes the
pooled `stock_openers_closers` teacher-forced AF suspect as a combined view:
the high pooled AF is not an opener effect.

The separate `stock_closers` teacher-forced grid uses the full 5,000-prompt
package and final-clause opportunity contract. All stages use 41,187
opportunities and 18 references, with reference rate `0.000437`.

| Stage | W&B | Mean Mass | Raw AF | 95% CI |
|---|---|---:|---:|---:|
| Base | `nlpxvpof` | 0.03204 | 73.304 | 47.983-132.664 |
| SFT | `jcp2nar5` | 0.02844 | 65.076 | 42.501-117.634 |
| DPO | `dcml121g` | 0.02999 | 68.634 | 45.192-123.947 |
| Final/RLVR | `y06wdbfa` | 0.03235 | 74.032 | 48.951-133.005 |

Interpretation: the high pooled stock opener/closer teacher-forced AF is a
closer-side effect. The magnitude should be read cautiously because the held
out reference denominator has only 18 positives, but the qualitative direction
is clear: stock closer phrases remain relatively cheap local continuations for
the model, with final/RLVR roughly back at base mass after the SFT dip.

## Biber-Lite Register Comparison

Biber-lite features are measured as a Phase 2 register comparison layer over
the same `t=1.0` target-shape generation caches, then joined to the Phase 1
corpus-role rates from `feature_rates_by_corpus.parquet`. This is not a
teacher-forced AF or compounding analysis.

W&B run:

- `stage2-phase2-olmo3-biber-lite-generation-vs-corpus-t1` (`eu4glzoq`)

Local outputs:

- `artifacts/phase2/analysis/olmo3_biber_lite_generation_vs_corpus_t1.csv`
- `artifacts/phase2/analysis/olmo3_biber_lite_generation_vs_corpus_t1_summary.md`

The table has 76 rows: 19 Biber-lite proxies across base, SFT, DPO, and
final/RLVR. Rates are regex-token-normalized per 1k tokens, matching the Phase
1 Biber-lite census.

Largest average generation-vs-SFT-target lifts across the four generation
stages:

| Feature | Avg Generation /1k | SFT Target /1k | Avg Delta | Ratio |
|---|---:|---:|---:|---:|
| `biber_lite_demonstratives` | 11.801 | 9.655 | 2.147 | 1.222 |
| `biber_lite_third_person_pronouns` | 5.507 | 3.575 | 1.932 | 1.540 |
| `biber_lite_second_person_pronouns` | 4.988 | 3.282 | 1.706 | 1.520 |
| `biber_lite_necessity_modals` | 3.105 | 1.431 | 1.674 | 2.170 |
| `biber_lite_first_person_pronouns` | 12.039 | 10.381 | 1.658 | 1.160 |
| `biber_lite_conditional_subordinators` | 6.741 | 5.179 | 1.562 | 1.302 |

Largest stage movements from base to DPO/final:

| Feature | Base /1k | DPO - Base | Final - Base |
|---|---:|---:|---:|
| `biber_lite_demonstratives` | 15.686 | -5.702 | -5.571 |
| `biber_lite_first_person_pronouns` | 15.325 | -5.023 | -3.687 |
| `biber_lite_infinitives` | 17.426 | -4.417 | -3.803 |
| `biber_lite_necessity_modals` | 5.037 | -2.622 | -2.532 |
| `biber_lite_second_person_pronouns` | 4.349 | 1.100 | 1.198 |
| `biber_lite_conditional_subordinators` | 6.133 | 1.732 | 1.525 |

Interpretation: the base checkpoint is high on demonstratives, first-person
pronouns, infinitives, necessity/possibility/prediction modals, hedges, and
that-complements relative to the post-training checkpoints. DPO/final move
many of those base-heavy register proxies downward, while increasing
second-person pronouns and conditional subordinators. The generated
nominalization rate is close to SFT targets and below DPO-chosen data, while
all generation stages are well above the pretrain sample for nominalizations.
Treat these as register/context shifts alongside the Tier-1 slop results, not
as claims about opportunity-normalized amplification.

## Current Compute Posture

Phase 2 is closed out at temperature `1.0`. Do not launch a full
5,000-prompt x 8-completion x 1,024-token generation grid blindly. The
target-shape DPO benchmark (`kji583m6`) plus the 512-prompt base/SFT/DPO/final
shards (`nw6yoj5u`, `il07fjn1`, `8rud2kxl`, `b8xo8axn`) validate the heavy
shape on the A100, but the full 5,000-prompt x 3-temperature OLMo target
remains expensive and is outside this Phase 2 close-out. Any next GPU work
should be a new, explicit follow-on question:

- For a stronger Result B estimate, the scaled 1,024-prompt DPO shard now
  stabilizes the DPO t=1.0 rate. Matched 1,024-prompt base/SFT/final shards are
  only needed if the next question is larger-sample stage localization.
- For stage-localization confirmation, the bounded four-stage target-shape
  comparison is now complete; additional free-running work should scale a
  specific comparison or temperature, not repeat the same cells.
- For teacher-forced precision, full 5k slop/neutral is confirmatory and should
  wait until a specific analysis requires tighter CIs.
- For generation throughput, benchmark SGLang or vLLM in an isolated
  environment against the 512-prompt DPO shard contract. Keep
  torch/Transformers as the exact teacher-forced scorer unless an exactness
  benchmark reproduces the fixed-branch probability-mass summaries.
- Use `slop-plan-phase2-generation` before launching more A100 generation
  shards. W&B run `eu65803t` confirms the existing 512-prompt x 8-completion
  x 1,024-token t=1.0 four-stage grid is complete locally. W&B run `3d8g6at4`
  estimates the full EXPERIMENTS.md OLMo generation grid at 12 shards,
  40,000 generations per shard, and about 383.5 A100-hours at current Torch
  throughput.
- Use `slop-launch-phase2-generation-shard` to execute one planned shard at a
  time. It is dry-run by default, requires `--execute` to launch, and refuses
  shards above the configured A100-hour cap unless `--force` is passed. For
  long shards, use `--detach --selection-output ...` and monitor with
  `slop-phase2-generation-status`.

## Open Caveats

- Phase 1 and Phase 2 are OLMo/Dolci-first. SmolLM3 replication remains future
  work.
- Biber-lite is available as register context, not as a primary Phase 2
  opportunity-normalized propensity target.
- Removed punctuation and list/header features should not be used for core
  claims without renewed validation.
- `contrastive_negation` is measurable but sparse in the 5k denominator audit.
- `rule_of_three_approx` now has a full 5,000-prompt comma-pair extension proxy
  grid, but it is still a proxy for third-item extension and should not be read
  as open-vocabulary initiation of the complete construction.
- `stock_openers` and `stock_closers` now have separate 5,000-prompt
  teacher-forced grids. The `stock_closers` magnitude is noisy because there
  are only 18 references in the current held-out package.
- The teacher-forced scorer now has a `--reference-subset NAME:FIELD=VALUE`
  sidecar summary path for all-SFT vs metadata-subset AF reporting. The
  current Dolci prompt package has usable `provenance`, `source_dataset`, and
  `stratum` metadata, but not a clean `human_written` flag, so human-written-SFT
  conclusions still require a provenance mapping or regenerated package with
  explicit labels. Prompt packaging can now materialize normalized fields with
  `--metadata-bucket-map`; use that path to add `reference_subset` before
  running subset AF summaries. For existing packages, use
  `slop-annotate-phase2-prompts` to preserve row identity. The current 5,000-row
  annotated package is logged as W&B run `ew7x38n8` with counts
  `code=1371`, `synthetic_llm=225`, and `unknown=3404`.
- `slop-summarize-propensity-subsets` can reuse existing opportunity CSVs for
  subset AF tables without a model rerun. The first slop/neutral subset summary
  is W&B run `316hbxbl`; it shows DPO highest for `synthetic_llm` and `unknown`
  subsets, while the `code` subset is denominator-sparse for slop references.
  Full 5,000-prompt subset summaries are also logged for rule-of-three
  (`0nklcliy`), stock openers (`jju0y0n5`), stock closers (`zxu196ge`), and
  pooled stock openers/closers (`ssa70ap1`). These preserve the all-reference
  read: rule-of-three does not show DPO amplification, stock openers are
  suppressed, and stock closers are the noisy closer-side stock effect.
- Raw artifacts under `artifacts/` are local and gitignored; durable result
  records live in config/docs and W&B artifacts.
