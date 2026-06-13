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
For `slop_lexicon` plus `neutral_common_controls` on OLMo 3 SFT, branch size
8 is the best tested setting on the A100; branch size 16 OOMed, and branch
size 4 was slower. Before broader Phase 2, keep benchmarking stable shapes,
consider tokenizer-trie deduplication, and keep neutral-normalized controls in
the promotion gate.

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

The original `neutral_controls` basket is now treated as a failed
discourse-marker calibration basket, not as the only neutral gate. A separate
`neutral_common_controls` basket is defined for high-support functional
starts observed in the held-out SFT targets: `the`, `a`, `of the`,
`number of`, `in the`, `to the`, and `is a`. These are measured separately so
the old neutral runs remain comparable while the calibration gate can test a
lower-style, higher-denominator control set.

Because both discourse and common neutral controls are below raw AF 1 under
broad token-start opportunities, the teacher-forced CLI now supports
`--normalization-feature`. This adds neutral-normalized AF columns while
preserving raw AF and CIs. Use `neutral_common_controls` as the first
normalization baseline when comparing slop targets under the current
opportunity contract.

## Promotion Criteria

Promote from tiny-model smoke to OLMo tiny shard only if:

- CUDA run completes on the A100.
- `torch.compile` succeeds or has a documented blocker.
- W&B logs model/run metadata, aggregate counts, throughput, and no raw text.
- Opportunity counts and reference initiation rates are nonzero where expected.
- Output CSV schemas are stable and tests pass.

Promote from OLMo tiny shard to full Phase 2 only after:

- Held-out prompt splitting and near-duplicate filtering are in place.
- The SFT checkpoint is reported with raw AF plus neutral-normalized AF against
  `neutral_common_controls`, or the opportunity contract is revised. The old
  raw-AF-near-1 neutral gate failed for both discourse controls and common
  function-word controls under broad token-start opportunities.
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
- `stage2-phase2-olmo3-sft-promptpkg512-neutral-common-controls-cached-shared-branch2-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/xw8ihrej`) tested the
  high-support common-control basket on the same 512-row held-out prompt
  package. It scored 362,096 feature-opportunities in 1,857.5 seconds, for
  194.9 feature-opportunities/sec. The pooled common-control basket had 3,425
  reference initiations, but raw AF was 0.339 with CI `[0.319, 0.360]`.
  Individual controls were also below 1: `the` 0.284, `a` 0.408,
  `of the` 0.364, `number of` 0.295, `in the` 0.738, `to the` 0.583, and
  `is a` 0.499. Conclusion: the raw AF≈1 neutral sanity gate fails even for
  high-support function-word controls under broad token-start opportunities.
  The next implementation step should report raw AF plus a neutral-normalized
  AF or revise the opportunity definition before scaling to the full
  checkpoint grid.
- `stage2-phase2-olmo3-sft-promptpkg512-slop-vs-neutral-common-normalized-cached-shared-branch2-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/hdwg40tc`) was started as a
  full 512-row normalized `slop_lexicon` versus `neutral_common_controls`
  shard, then stopped early because slop-lexicon branch fanout made throughput
  too low for an efficient full run under the current implementation.
- `stage2-phase2-olmo3-sft-promptpkg512-sample128-slop-vs-neutral-common-normalized-cached-shared-branch2-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/wgsqtjcq`) completed the
  same normalized comparison on a 128-document sample from the 512-row prompt
  package. It scored 20,916 opportunities in 1,128.8 seconds, for 18.5
  opportunities/sec. `neutral_common_controls` had raw AF 0.331 with CI
  `[0.290, 0.380]`; `slop_lexicon` had raw AF 0.880 with CI `[0.000, 3.218]`.
  Normalized against the common-control basket, `slop_lexicon` had normalized
  AF 2.656 with CI `[0.000, 9.620]`. Interpretation: the point estimate is in
  the expected positive-control direction after common-control normalization,
  but the slop denominator is only 5 references on this sample, so this is a
  directional shard and performance datapoint rather than a stable full-grid
  result.
- `stage2-phase2-olmo3-sft-promptpkg512-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/pgxems2i`) completed the
  full 512-document normalized SFT shard with the branch cap raised to 8. It
  scored 90,524 opportunities in 3,269.9 seconds, for 27.7 opportunities/sec.
  `neutral_common_controls` reproduced the common-control raw AF failure with
  AF 0.339 and CI `[0.319, 0.360]`. `slop_lexicon` had raw AF 0.673 with CI
  `[0.414, 1.225]`, and neutral-normalized AF 1.982 with CI
  `[1.226, 3.663]`. This is the first stable Phase 2 SFT positive-control
  result under the current normalized contract: relative to the common-control
  calibration baseline, OLMo 3 SFT assigns about 2x the target-distribution
  initiation mass to the slop lexicon.
- `stage2-phase2-olmo3-base-promptpkg512-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/4bp5090m`) completed the
  matching 512-document base-checkpoint shard. It scored 90,524 opportunities
  in 3,273.0 seconds, for 27.7 opportunities/sec. `neutral_common_controls`
  had raw AF 0.272 with CI `[0.252, 0.293]`; `slop_lexicon` had raw AF 0.484
  with CI `[0.274, 0.957]`. Neutral-normalized `slop_lexicon` AF was 1.782
  with CI `[1.005, 3.519]`. First stage-localization read: the slop-lexicon
  signal is already present at the base checkpoint and is not introduced by
  SFT, though the SFT point estimate is modestly higher (1.982).
- `stage2-phase2-olmo3-dpo-promptpkg512-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/bo2i2g1l`) completed the
  matching 512-document DPO shard. It scored 90,524 opportunities in 3,293.9
  seconds, for 27.5 opportunities/sec. `neutral_common_controls` had raw AF
  0.376 with CI `[0.355, 0.400]`; `slop_lexicon` had raw AF 0.980 with CI
  `[0.605, 1.669]`. Neutral-normalized `slop_lexicon` AF was 2.605 with CI
  `[1.595, 4.485]`. Current stage-localization read for the narrow
  positive-control grid: base 1.782 -> SFT 1.982 -> DPO 2.605, consistent
  with additional preference-stage amplification on top of an inherited base
  signal.
- `stage2-phase2-olmo3-final-promptpkg512-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/zc5lbdzo`) completed the
  matching final/RLVR shard. It scored 90,524 opportunities in 3,274.3
  seconds, for 27.6 opportunities/sec. `neutral_common_controls` had raw AF
  0.455 with CI `[0.429, 0.481]`; `slop_lexicon` had raw AF 0.981 with CI
  `[0.605, 1.669]`. Neutral-normalized `slop_lexicon` AF was 2.156 with CI
  `[1.327, 3.717]`. The completed narrow OLMo stage-localization slice is:
  base 1.782, SFT 1.982, DPO 2.605, final 2.156. This suggests the largest
  slop-lexicon lift occurs by DPO, with the final checkpoint remaining above
  base/SFT but below the DPO point estimate.
- `stage2-phase2-olmo3-slop-neutral-common-normalized-stage-grid-assembly`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/bjpt4jy3`) assembled the
  four summary CSVs into a stage-grid table and primary-feature comparison
  under `artifacts/phase2/analysis/`. The assembled primary comparison
  identifies DPO as the maximum normalized-AF stage: base 1.782, SFT 1.982,
  DPO 2.605, final 2.156. This run logs aggregate tables only and does not
  rerun model scoring.
- Cached-only scorer microbenchmarks for the same OLMo SFT
  `slop_lexicon` plus `neutral_common_controls` shape showed that the previous
  branch cap of 2 is conservative. Branch 2
  (`https://wandb.ai/phulin-self/slop-stage1/runs/6g5evlht`) reached 20.5
  feature-opportunities/sec, branch 4
  (`https://wandb.ai/phulin-self/slop-stage1/runs/ytzn1c3b`) reached 25.4,
  and branch 8 (`https://wandb.ai/phulin-self/slop-stage1/runs/nwtcj61b`)
  reached 28.4 at batch size 16 and 256 prefix tokens. Branch 16
  (`https://wandb.ai/phulin-self/slop-stage1/runs/vstbnjwn`) OOMed on the
  cached path, so branch 8 is the next setting to use for the narrow
  normalized stage-localization shards.
- `stage2-phase2-olmo3-sft-promptpkg512-vllm-free-running-16prompt-t0-t07`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/r962btj8`) tested the
  current vLLM release path in an isolated `/tmp/slop-vllm-bench` environment
  created with `uv`. `vllm==0.22.1 --torch-backend=auto` resolved to
  `torch==2.11.0+cu129`; the project `.venv` was not modified. Import required
  adding the sidecar CUDA library paths because the wheel stack needed
  `libcudart.so.13`. Model load reached OLMo SFT initialization, then failed
  when FlashInfer sampling tried to JIT without a CUDA toolkit/nvcc at
  `/usr/local/cuda`. This run logged as a failed W&B backend attempt and did
  not emit generation artifacts.
- `stage2-phase2-olmo3-sft-promptpkg512-vllm-free-running-16prompt-t0-t07-pytorch-sampler`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/c6i9dqr9`) retried the same
  `vllm==0.22.1` environment with `VLLM_USE_FLASHINFER_SAMPLER=0`. This passed
  the previous sampler-JIT failure but then failed during CUDA graph profiling
  with `CUDA driver version is insufficient for CUDA runtime version`. Treat
  current vLLM/cu129 wheels as incompatible with this host's 570.124.06 driver
  unless the driver is upgraded or a compatibility-container path is validated.
- A safer sidecar install candidate exists for this host:
  `uv venv --python 3.12 /tmp/slop-vllm-cu128-bench` followed by
  `uv pip install --python /tmp/slop-vllm-cu128-bench/bin/python -e .
  'vllm==0.11.1' --torch-backend=cu128`. Import smoke passed with
  `vllm==0.11.1`, `torch==2.9.0+cu128`, CUDA runtime 12.8, and the A100
  visible. The first OLMo load attempt
  (`https://wandb.ai/phulin-self/slop-stage1/runs/6h1s5tq4`) was stopped after
  the parent process sat idle with no GPU work while an engine subprocess
  remained alive. Do not promote vLLM for free-running Phase 2 until this
  cu128 path completes a small generation benchmark, or use the existing
  Torch/Transformers CLI for the next generation shard.
- `stage2-phase2-olmo3-sft-promptpkg512-free-running-32prompt-t0-t07-t1-batched128`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/0dqgmeez`) completed the
  next Torch/Transformers free-running shard after the vLLM backend attempts.
  It used the 512-row held-out prompt package, 32 prompts, temperatures
  `0.0`, `0.7`, and `1.0`, top-p 0.95, max 128 new tokens, batch size 8,
  bfloat16, and `torch.compile`. It produced 96 generations and 12,288
  generated tokens in 231.3 seconds, or 53.1 tokens/sec including model load
  and compile. Feature rates remained sparse: contrastive-negation appeared
  once at each temperature (0.244 per 1k tokens), rule-of-three appeared 4
  times at temperature 0.0 and twice at 0.7/1.0, `slop_lexicon` appeared once
  at temperature 0.0 only, `stock_openers` appeared once at temperature 1.0
  only, and `stock_closers` had zero hits. No feature had repeated hits within
  a generation, so this shard does not yet show self-conditioning compounding.
  Local ignored artifacts are
  `artifacts/phase2/generations/olmo3_sft_promptpkg512_free_run_32prompt_t0_t07_t1_batched128.jsonl`
  and the matching `_summary.csv`.
