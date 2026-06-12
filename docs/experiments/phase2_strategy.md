# Phase 2 Strategy

Date: 2026-06-12

Phase 2 measures model propensity to initiate scoped slop features under
teacher forcing, then compares that decoding-free signal to free-running
emission rates.

## Current Slice

The first implementation slice is a smoke harness, not the full OLMo grid:

- Extract deterministic opportunity positions for revised core features.
- Compute exact initiation sequence probability mass at each opportunity using
  a causal LM. The first-token estimator remains a debug fallback.
- Run free-running emissions on held-out prompts so realized rates can be
  compared against teacher-forced propensity.
- Log aggregate AF-style summaries to W&B without raw text.
- Run on a tiny model and a tiny retained Dolci SFT shard before promoting to
  OLMo checkpoints.

Current CLI:

```bash
uv run slop-prepare-phase2-prompts \
  --input allenai/Dolci-Instruct-SFT-7B \
  --split train \
  --sample-size 8 \
  --max-scan 128 \
  --sampling-strategy hash_reservoir \
  --near-duplicate-threshold 0.85 \
  --output artifacts/phase2/prompts/olmo3_dolci_sft_phase2_prompt_package_8.jsonl \
  --manifest-output artifacts/phase2/prompts/olmo3_dolci_sft_phase2_prompt_package_8_manifest.csv \
  --summary-output artifacts/phase2/prompts/olmo3_dolci_sft_phase2_prompt_package_8_summary.json \
  --wandb-mode online
```

```bash
uv run slop-teacher-forced-propensity \
  --model sshleifer/tiny-gpt2 \
  --input artifacts/phase2/prompts/olmo3_dolci_sft_phase2_prompt_package_8.jsonl \
  --sample-size 8 \
  --feature contrastive_negation \
  --feature slop_lexicon \
  --feature stock_openers \
  --feature stock_closers \
  --max-opportunities 128 \
  --mass-mode sequence \
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
- Neutral controls: `neutral_for_example`, `neutral_such_as`,
  `neutral_in_particular`, `neutral_as_a_result`, and pooled
  `neutral_controls`; these use token-start opportunities and are calibration
  targets, not slop claims.

Held back from the first smoke:

- `rule_of_three_approx`: the Phase 1 regex is useful for rates, but a
  teacher-forced initiation event is not well-defined yet. Freeze a better
  opportunity contract before using it for AF claims.
- Biber-lite features: keep as register covariates, not propensity targets.

Free-running smoke CLI:

```bash
uv run slop-free-running-emission \
  --model sshleifer/tiny-gpt2 \
  --input artifacts/stage1/corpora/olmo3_dolci_sft_10k_retained_sample.jsonl \
  --sample-size 1 \
  --feature slop_lexicon \
  --feature stock_openers \
  --temperature 0.0 \
  --max-new-tokens 16 \
  --fallback-text-as-prompt \
  --generations-output artifacts/phase2/generations/tiny_gpt2_free_run_smoke.jsonl \
  --summary-output artifacts/phase2/generations/tiny_gpt2_free_run_smoke_summary.csv \
  --wandb-mode online
```

## Measurement Caveat

The current exact-sequence estimator sums up to the first three tokenizer
tokens of each initiating phrase variant. This is enough to avoid the inflated
first-token smoke estimator, but it is still deliberately small-shard code.
Current runs should use batched opportunity scoring with KV-cache continuation
reuse and a bounded `--cache-branch-batch-size` to avoid branch-fanout OOMs.
Before full Phase 2, keep benchmarking stable shapes, consider tokenizer-trie
deduplication, and keep neutral controls in the promotion gate.

## Promotion Criteria

Promote from tiny-model smoke to OLMo tiny shard only if:

- CUDA run completes on the A100.
- `torch.compile` succeeds or has a documented blocker.
- W&B logs model/run metadata, aggregate counts, throughput, and no raw text.
- Opportunity counts and reference initiation rates are nonzero where expected.
- Output CSV schemas are stable and tests pass.

Promote from OLMo tiny shard to full Phase 2 only after:

- Held-out prompt splitting and near-duplicate filtering are in place.
- Neutral controls show AF near 1 on an SFT checkpoint using the
  `neutral_controls` basket and individual control phrases.
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
- `stage2-phase2-tiny-gpt2-sequence-propensity-smoke`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/45ozhmum`) completed exact
  sequence-mass scoring on 1 retained Dolci SFT row, 16 `slop_lexicon`
  opportunities, with `torch.compile` enabled.
- `stage2-phase2-olmo3-7b-instruct-sft-sequence-tiny`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/pp0x4f2y`) completed the
  first real OLMo SFT exact-sequence tiny shard. It measured 8
  `slop_lexicon` opportunities, zero reference initiations, mean probability
  mass `1.103147e-06`, 85.1 wall seconds, and 0.094 opportunities/sec. This is
  a plumbing and throughput datapoint, not an amplification result.
- `stage2-phase2-tiny-gpt2-free-running-smoke-v2`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/4dt5kc3h`) completed the
  first free-running emission smoke with `torch.compile` enabled. It used one
  retained sample text as a prompt for plumbing validation, generated 16
  tokens, logged redacted W&B sample rows, and found zero `slop_lexicon` or
  `stock_openers` matches. The preceding free-running attempt
  (`rhfcryb6`) failed because this GPT-2 generation path rejected the
  `generator` kwarg; the CLI now seeds with `torch.manual_seed`/
  `torch.cuda.manual_seed_all` instead.
- `stage2-phase2-dolci-sft-prompt-package-8`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/7xkqxesu`) prepared the first
  prompt-bearing Phase 2 package from `allenai/Dolci-Instruct-SFT-7B`. It
  scanned 128 rows, found 125 eligible prompt/target rows, filtered 3
  near-duplicate prompts with MinHash, selected 8 prompts, and logged only the
  redacted manifest to W&B. Local ignored artifacts are under
  `artifacts/phase2/prompts/`.
- `stage2-phase2-olmo3-sft-promptpkg8-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/mbsjovsf`) completed the
  first prompt-package-backed OLMo SFT teacher-forced run. It scored 8 docs,
  227 opportunities, and 4 feature summaries in 376.8 seconds, for 0.602
  opportunities/sec. The package had zero reference initiations for these
  features, so this validates the held-out prompt package, OLMo loading,
  compiled exact-sequence scoring, and W&B plumbing; it is not an amplification
  result.
- `stage2-phase2-dolci-sft-prompt-package-128`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/dwqj49uk`) prepared a larger
  prompt package from a 2,048-row scan. It produced 128 prompts after filtering
  74 near duplicates. Local reference measurement found usable denominators for
  `slop_lexicon`, `stock_openers`, and pooled `neutral_controls`, but still
  zero positives for `contrastive_negation` and `stock_closers`.
- `stage2-phase2-dolci-sft-positive-control-package-32`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/wl0o62kn`) prepared a
  reference-positive package by retaining rows that initiate `slop_lexicon` or
  pooled `neutral_controls`. It scanned 4,096 rows, found 347 eligible
  reference-positive prompts, selected 32 prompts, and provides a compact
  positive/control shard. The first 4 rows at a 128 token-start cap contain 902
  total opportunities with 2 `slop_lexicon`, 2 `neutral_controls`, and 1
  `stock_openers` reference positives.
- `stage2-phase2-olmo3-sft-positive-control4-cached-branch2-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/bigtxtoj`) completed the
  first bounded AF-style OLMo SFT run with nonzero denominators. It scored the
  first 4 rows of the targeted positive/control package, 902 opportunities, and
  3 feature summaries in 92.3 seconds, for 9.78 opportunities/sec end to end.
  Reference initiations were 2 for `slop_lexicon`, 2 for pooled
  `neutral_controls`, and 1 for `stock_openers`; measured AFs were 0.434,
  0.296, and 0.00036 respectively. Because the package is intentionally
  reference-positive, this is a calibration/plumbing shard, not an unbiased
  corpus AF estimate.
