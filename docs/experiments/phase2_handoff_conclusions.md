# Phase 2 Handoff Conclusions

Date: 2026-06-13

## Current Scope

Phase 2 is implemented and exercised for the OLMo 3 Instruct ladder as a
bounded measurement package, not as the full EXPERIMENTS.md production-scale
run. The implemented components are:

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

## Retained Features

The Phase 2 core feature set inherited from the revised Phase 1 close-out is:

- `contrastive_negation`
- `rule_of_three_approx`
- `slop_lexicon`
- `stock_openers`
- `stock_closers`

Teacher-forced stage-localization claims are currently strongest for
`slop_lexicon` normalized by `neutral_common_controls`. `rule_of_three_approx`
remains free-running-only until a teacher-forced opportunity contract is frozen.

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

## Bounded Amplification Spectrum

The current bounded headline table is assembled by
`slop-assemble-amplification-spectrum` and logged as W&B run `v31wiggh`
(`stage2-phase2-olmo3-amplification-spectrum-bounded`). Local outputs:

- `artifacts/phase2/analysis/olmo3_amplification_spectrum_bounded.csv`
- `artifacts/phase2/analysis/olmo3_amplification_spectrum_bounded_summary.md`

It contains 24 rows: six retained feature views across base, SFT, DPO, and
final/RLVR. Each row joins Phase 1 corpus rates, available teacher-forced AF,
target-shape free-running rates, compounding summaries, and denominator-support
notes. Blank cells mean missing measurements, not zero effects.

Current spectrum read:

- `slop_lexicon` has the strongest teacher-forced evidence: neutral-normalized
  AF peaks at DPO (`1.999`) and falls at final/RLVR (`1.659`), while
  target-shape free-running rates put base (`0.233` per 1k generated tokens)
  just above DPO (`0.229`).
- `rule_of_three_approx` is the clearest target-shape free-running feature but
  currently peaks at base (`1.019` per 1k generated tokens), so it weakens a
  simple monotonic post-training story until a teacher-forced contract is
  available.
- Stock openers/closers are small but DPO-peaked in target-shape generation.
- `contrastive_negation` remains too sparse in the 5k denominator audit for a
  strong bounded conclusion.

Treat this as the current bounded amplification spectrum, not the full
production-scale EXPERIMENTS.md table.

## Current Compute Posture

Do not launch a full 5,000-prompt x 8-completion x 1,024-token generation grid
blindly. The target-shape DPO benchmark (`kji583m6`) plus the 512-prompt
base/SFT/DPO/final shards (`nw6yoj5u`, `il07fjn1`, `8rud2kxl`, `b8xo8axn`)
validate the heavy shape on the A100, but the full 5,000-prompt x 3-temperature
OLMo target remains expensive. The next GPU work should be selected by the
question it answers:

- For a stronger Result B estimate, scale the free-running generation shard
  before spending on more teacher-forced slop/neutral rows.
- For stage-localization confirmation, the bounded four-stage target-shape
  comparison is now complete; additional free-running work should scale a
  specific comparison or temperature, not repeat the same cells.
- For teacher-forced precision, full 5k slop/neutral is confirmatory and should
  wait until a specific analysis requires tighter CIs.
- For generation throughput, benchmark SGLang or vLLM in an isolated
  environment against the 512-prompt DPO shard contract. Keep
  torch/Transformers as the exact teacher-forced scorer unless an exactness
  benchmark reproduces the fixed-branch probability-mass summaries.

## Open Caveats

- Phase 1 and Phase 2 are OLMo/Dolci-first. SmolLM3 replication remains future
  work.
- Biber-lite is available as register context, not as a primary Phase 2
  opportunity-normalized propensity target.
- Removed punctuation and list/header features should not be used for core
  claims without renewed validation.
- `contrastive_negation` is measurable but sparse in the 5k denominator audit.
- `rule_of_three_approx` lacks a frozen teacher-forced opportunity contract.
- Raw artifacts under `artifacts/` are local and gitignored; durable result
  records live in config/docs and W&B artifacts.
