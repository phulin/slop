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

The current 512-prompt compounding analysis joins cached generations to the
newer 1,024-prompt teacher-forced stage grid. Observed `slop_lexicon`
opportunity rates exceed the teacher-forced expectation in every
stage/temperature cell.

Current headline cells:

- Max excess: SFT at temperature 1.0, observed `0.790` vs expected `0.336` per
  1k opportunities, excess `0.454`, observed/expected `2.35`, realized AF
  `1.35`.
- Max realized AF: DPO at temperature 1.0, realized AF `1.506`; final
  temperature 1.0 is effectively tied at `1.504`.
- Direct prior/no-prior window signal is positive in all `slop_lexicon` cells,
  but repeat counts are still sparse.

Interpretation: Result B has a bounded positive signal, but the conditional
window estimates are not yet stable enough for the final paper-scale claim.

## Current Compute Posture

Do not launch a full 5,000-prompt x 8-completion x 1,024-token generation grid
blindly. The target-shape DPO benchmark (`kji583m6`) validates the heavy shape
on 64 prompts, but the full OLMo four-stage target remains expensive. The next
GPU work should be selected by the question it answers:

- For a stronger Result B estimate, scale the free-running generation shard
  before spending on more teacher-forced slop/neutral rows.
- For stage-localization confirmation, a narrower DPO/final generation shard at
  the target 8-completion, 1,024-token shape is more informative than rerunning
  all stages immediately.
- For teacher-forced precision, full 5k slop/neutral is confirmatory and should
  wait until a specific analysis requires tighter CIs.

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
