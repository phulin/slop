# Phase 2 Strategy

Date: 2026-06-12

Phase 2 measures model propensity to initiate scoped slop features under
teacher forcing, then compares that decoding-free signal to free-running
emission rates.

## Current Slice

The first implementation slice is a smoke harness, not the full OLMo grid:

- Extract deterministic opportunity positions for revised core features.
- Compute first-token initiation probability mass at each opportunity using a
  causal LM.
- Log aggregate AF-style summaries to W&B without raw text.
- Run on a tiny model and a tiny retained Dolci SFT shard before promoting to
  OLMo checkpoints.

Current CLI:

```bash
uv run slop-teacher-forced-propensity \
  --model sshleifer/tiny-gpt2 \
  --input artifacts/stage1/corpora/olmo3_dolci_sft_10k_retained_sample.jsonl \
  --sample-size 8 \
  --feature contrastive_negation \
  --feature slop_lexicon \
  --feature stock_openers \
  --feature stock_closers \
  --max-opportunities 128 \
  --output artifacts/phase2/propensity/tiny_gpt2_smoke_opportunities.csv \
  --summary-output artifacts/phase2/propensity/tiny_gpt2_smoke_summary.csv \
  --wandb-mode online
```

The CLI defaults to `--torch-compile`. Use `--no-torch-compile` only to debug a
compiler failure, and record that failure before falling back.

## Opportunity Scope

Implemented smoke targets:

- `contrastive_negation`: clause-boundary opportunities.
- `slop_lexicon`: token-start opportunities, capped per document for smoke.
- `stock_openers`: document-start opportunities.
- `stock_closers`: final-clause-boundary opportunities.
- `stock_openers_closers`: pooled document-start plus final-boundary
  opportunities.

Held back from the first smoke:

- `rule_of_three_approx`: the Phase 1 regex is useful for rates, but a
  teacher-forced initiation event is not well-defined yet. Freeze a better
  opportunity contract before using it for AF claims.
- Biber-lite features: keep as register covariates, not propensity targets.

## Measurement Caveat

The first smoke computes probability mass over the first token of each
initiating phrase. Full Phase 2 should extend this to exact 1-3 token
initiating continuations with a tokenizer trie. The smoke is meant to validate
plumbing, W&B logging, CUDA/compile behavior, and opportunity counts.

## Promotion Criteria

Promote from tiny-model smoke to OLMo tiny shard only if:

- CUDA run completes on the A100.
- `torch.compile` succeeds or has a documented blocker.
- W&B logs model/run metadata, aggregate counts, throughput, and no raw text.
- Opportunity counts and reference initiation rates are nonzero where expected.
- Output CSV schemas are stable and tests pass.

Promote from OLMo tiny shard to full Phase 2 only after:

- Exact multi-token continuation mass is implemented.
- Held-out prompt splitting and near-duplicate filtering are in place.
- Neutral controls show AF near 1 on an SFT checkpoint.
- Positive controls reproduce expected amplification direction.

## Smoke Run Log

- `stage2-phase2-tiny-gpt2-compile-propensity-smoke-v2`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/w7b161sw`) completed on the
  A100 with `torch.compile` enabled.
- Inputs: 2 retained Dolci SFT rows, features `contrastive_negation` and
  `stock_openers`.
- Outputs:
  `artifacts/phase2/propensity/tiny_gpt2_compile_smoke_opportunities.csv` and
  `artifacts/phase2/propensity/tiny_gpt2_compile_smoke_summary.csv`.
- Shape: 64 opportunities, 2 feature summaries, 11.53 wall seconds, 5.55
  opportunities/sec, 1.35 GB peak RSS.
- The sampled rows had zero reference initiations for the selected features, so
  this run validates plumbing, CUDA, W&B, and compile behavior only.
- The first compile attempt failed because Triton could not discover
  `libcuda.so.1` while `LD_LIBRARY_PATH` was empty. The CLI now adds
  `/usr/lib/x86_64-linux-gnu` when `libcuda.so.1` exists there before invoking
  `torch.compile`.
