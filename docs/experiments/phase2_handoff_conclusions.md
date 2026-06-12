# Phase 2 Handoff Conclusions

Date: 2026-06-12

## Task Plan

1. Treat Phase 1 as closed for the sampled OLMo Dolci SFT measurement package,
   with the caveats below carried forward into Phase 2 interpretation.
2. Use the retained Phase 1 feature scope as the first Phase 2 propensity and
   emission target set.
3. Run Phase 2 first on OLMo 3 Instruct-path checkpoints, starting with the SFT
   checkpoint for calibration and then extending across the full ladder.
4. Add free-running emission measurement alongside teacher-forced propensity so
   Phase 2 can support the compounding comparison.

## Phase 1 Closure

Phase 1 is closed as a sampled OLMo Dolci SFT measurement, not as an exhaustive
corpus census. The retained core feature set is:

- `contrastive_negation`
- `rule_of_three_approx`
- `slop_lexicon`
- `stock_openers`
- `stock_closers`

Biber-lite features remain useful as register covariates. They should not be
treated as primary slop propensity targets until a stronger opportunity contract
exists.

Removed or demoted features:

- punctuation/rhythm
- list/header formatting

Full SmolLM replication remains future work.

## Phase 2 Components

Phase 2 now has exact sequence teacher-forced propensity as the immediate model
measurement component. The next required component is free-running emission, so
teacher-forced propensity can be compared against realized generation rates and
self-conditioning effects.

Known Phase 2 W&B runs:

- Tiny GPT-2 exact-sequence smoke: `45ozhmum`.
- OLMo SFT tiny exact-sequence run: `pp0x4f2y`.
  - 8 opportunities.
  - Mean `slop_lexicon` mass approximately `1.103147e-06`.
  - Zero reference initiations.
  - Throughput approximately `0.094` opportunities/sec.
- Tiny GPT-2 free-running emission smoke: `4dt5kc3h`.
  - 1 prompt.
  - 16 generated tokens.
  - Zero `slop_lexicon` and `stock_openers` matches.
  - Uses retained text as a prompt for plumbing validation only.

## OLMo 3 Model IDs

Use these OLMo 3 Instruct-path checkpoint IDs:

- Base: `allenai/Olmo-3-1025-7B`
- SFT: `allenai/Olmo-3-7B-Instruct-SFT`
- DPO: `allenai/Olmo-3-7B-Instruct-DPO`
- Final: `allenai/Olmo-3-7B-Instruct`

## Open Caveats

- Phase 1 closure is sampled and OLMo Dolci focused; it does not close full
  SmolLM replication.
- Retained feature claims should stay limited to the revised core features and
  Biber-lite covariates listed above.
- Removed punctuation and list/header features should not be used for core
  Phase 2 claims without renewed validation.
- The OLMo SFT tiny run validates exact-sequence plumbing on a small shard but
  has zero reference initiations, so it is not yet an amplification result.
- Free-running emission now has a smoke-tested CLI, but held-out prompt
  packaging remains required before Phase 2 can support the compounding
  decomposition.
