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

For the exact teacher-forced AF scorer, keep the current `torch`/Transformers
path unless a serving backend is explicitly benchmarked against our required
query shape. The scorer needs probability mass for a fixed set of candidate
continuation token sequences at many internal corpus offsets, with document
reference alignment and bootstrap summaries. vLLM or SGLang may be worthwhile
for the later free-running generation workload, but they should not replace the
teacher-forced scorer until they can return equivalent arbitrary continuation
mass without changing the estimator.

Teacher-forced initiator sequence enumeration now includes sentence-case
surface variants in addition to lowercase forms. This is required because the
reference regexes are case-insensitive; the first 512-row neutral breakdown
under-counted model mass for capitalized `For example` references. Rerun the
neutral calibration after this fix before treating the current neutral-control
failure as final.

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
- `stage2-phase2-positive-control32-opportunity-extraction-benchmark`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/4gy3euvz`) measured
  CPU-side opportunity extraction on the 32-row positive/control package. Full
  extraction over 8,774 opportunities peaked at 142.8k opportunities/sec, while
  raw offset enumeration was about 1.04M opportunities/sec. This confirms CPU
  extraction is not the bottleneck for the next OLMo shard; the matcher pass is
  the dominant CPU-side cost and now skips irrelevant Tier-1 regex families for
  narrow feature selections.
- `stage2-phase2-olmo3-sft-propensity-scorer-multifeature-branch2-benchmark-v2`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/a2lcsbg9`) measured the
  corrected shared-prefix cached scorer on the A100. At 256 prefix tokens,
  batch 16, and branch chunk size 2, scalar replay scored 2.73 opportunities/sec,
  single-feature cached scoring scored 24.9 opportunities/sec, and shared
  multi-feature cached scoring scored 53.2 feature-opportunities/sec across
  `slop_lexicon`, `neutral_controls`, and `stock_openers`. Max absolute
  difference versus scalar for the reference feature was `1.83e-04`, within the
  bfloat16 tolerance seen in previous cache benchmarks. Multi-feature Phase 2
  shards should keep exact sequence mass, fixed 256-token prefixes,
  opportunity batch size 16, KV-cache continuations, branch chunk size 2, and
  shared-prefix scoring enabled.
- `stage2-phase2-olmo3-sft-positive-control32-cached-shared-branch2-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/ce8i6wd9`) completed the
  32-row targeted positive/control shard with shared-prefix cached scoring and
  document-cluster bootstrap CIs. It scored 7,862 opportunities in 268.5
  seconds, for 29.3 opportunities/sec end to end. Point AFs were 0.289
  (`neutral_controls`), 0.298 (`slop_lexicon`), and 0.0014 (`stock_openers`).
  The targeted package is useful as a nonzero-denominator calibration shard but
  should not be read as an unbiased corpus estimate.
- `stage2-phase2-olmo3-sft-promptpkg128-cached-shared-branch2-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/n8zueel5`) completed the
  first broader held-out SFT prompt-package shard with the optimized exact
  sequence scorer. It scored 128 prompt-held-out documents and 23,656
  opportunities in 680.5 seconds, for 34.8 opportunities/sec end to end.
  Point AF was 0.629 for `neutral_controls` with bootstrap CI
  `[0.315, 2.028]`, and 0.746 for `slop_lexicon` with bootstrap CI
  `[0.296, 2.323]`. The neutral-control CI now spans 1, but denominators are
  still small (5 neutral-control and 10 slop-lexicon reference initiations), so
  this starts the meaningful Phase 2 SFT measurement but is not sufficient to
  promote to the full checkpoint grid.
- `stage2-phase2-dolci-sft-prompt-package-512`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/i6sh0rcv`) prepared a
  larger prompt-held-out Dolci SFT package by scanning 8,192 rows. It found
  7,753 eligible prompts, filtered 439 near-duplicate prompts, and selected
  512 prompts. A CPU denominator check
  (`https://wandb.ai/phulin-self/slop-stage1/runs/ukrfj8ud`) found 90,524
  total opportunities and 45 combined reference initiations for
  `slop_lexicon` plus `neutral_controls`.
- `stage2-phase2-olmo3-sft-promptpkg512-cached-shared-branch2-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/35b1isae`) completed the
  512-row held-out SFT scorer run with exact sequence mass, shared-prefix KV
  cache, branch chunk size 2, `torch.compile`, and document-bootstrap CIs. It
  scored 90,524 opportunities in 2,422.9 seconds, for 37.4 opportunities/sec
  end to end. Point AF was 0.357 for `neutral_controls` with CI
  `[0.258, 0.531]`, and 0.608 for `slop_lexicon` with CI `[0.361, 1.139]`.
  The tighter neutral-control interval no longer includes 1, so the current
  pooled neutral-control opportunity/initiator definition fails the SFT
  calibration gate. Do not promote to the full checkpoint grid until that
  neutral-control mismatch is diagnosed.
- `stage2-phase2-olmo3-sft-promptpkg512-neutral-breakdown-cached-shared-branch2-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/gkj0vlvc`) reran the 512-row
  held-out SFT package with individual neutral controls plus the pooled basket.
  It scored 226,310 feature-opportunities in 1,320.4 seconds, for 171.4
  feature-opportunities/sec. The observed neutral reference initiations were
  concentrated in `neutral_such_as` (16 references, AF 0.424, CI
  `[0.296, 0.713]`) and `neutral_for_example` (6 references, AF 0.103, CI
  `[0.051, 0.337]`). `neutral_in_particular` and `neutral_as_a_result` had
  zero reference initiations in this sample. The pooled neutral AF remains
  0.357 with CI `[0.258, 0.531]`, so the calibration failure is not just an
  artifact of mixing in unobserved controls.
- `stage2-phase2-tiny-gpt2-batched-free-running-smoke`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/pnjtz101`) validated the
  batched free-running CLI on the 8-row prompt package. It generated 8
  completions across 2 prompts, 2 completions per prompt, and temperatures
  0.0/0.7, logging redacted generation rows to W&B.
- `stage2-phase2-olmo3-sft-promptpkg512-free-running-16prompt-t0-t07-batched`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/2co91og0`) completed the
  first W&B-logged OLMo SFT free-running shard on the 512-row held-out prompt
  package. It generated 32 completions from 16 prompts at temperatures 0.0 and
  0.7, top-p 0.95, max 64 new tokens, batch size 4, bfloat16, and
  `torch.compile`. Throughput was 2,048 generated tokens in 207.7 seconds
  (9.86 tokens/sec including model load and compile). At this tiny sample,
  observed feature counts were 1 contrastive-negation and 2 rule-of-three hits
  at temperature 0.0, 1 rule-of-three hit at temperature 0.7, and zero
  `slop_lexicon`, `stock_openers`, or `stock_closers` hits.
- `stage2-phase2-olmo3-sft-promptpkg512-neutral-breakdown-case-variants-cached-shared-branch2-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/0rz3idjo`) reran the neutral
  breakdown after adding sentence-case initiator variants. It scored the same
  226,310 feature-opportunities in 1,875.1 seconds, for 120.7
  feature-opportunities/sec. The casing fix substantially changed
  `neutral_for_example`: AF moved from 0.103 to 0.711 with CI
  `[0.407, 2.076]`. The pooled neutral basket improved from 0.357 to 0.545,
  but its CI `[0.408, 0.800]` still excludes 1, and `neutral_such_as` stayed
  low at 0.429 with CI `[0.300, 0.722]`. Conclusion: casing was a real
  measurement bug, but the current pooled neutral basket still fails
  calibration because `such as` is underpredicted under this opportunity
  definition.
