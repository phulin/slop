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
- `rule_of_three_approx`: comma-pair extension opportunities. The model is
  scored at the slot after a two-item comma-separated list (`X, Y <slot>`) for
  finite third-item extension initiators `, and`, `, or`, `and`, and `or`.
  Reference positives are aligned to the coordinator inside an existing
  `RULE_OF_THREE_PATTERN` hit. This is a teacher-forced proxy for extending an
  established two-item list into a three-item coordination, not a claim that
  the open-vocabulary first item can be scored as a closed initiator set.
- Neutral controls: `neutral_for_example`, `neutral_such_as`,
  `neutral_in_particular`, `neutral_as_a_result`, and pooled
  `neutral_controls`; these use token-start opportunities and are calibration
  targets, not slop claims.

Held back from the first smoke:

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

Free-running assembly CLI:

```bash
uv run slop-assemble-phase2-generation-grid \
  --generation-summary sft=artifacts/phase2/generations/olmo3_sft_promptpkg512_free_run_32prompt_t0_t07_t1_batched128_summary.csv \
  --checkpoint-label sft=allenai/Olmo-3-7B-Instruct-SFT \
  --primary-feature slop_lexicon \
  --output artifacts/phase2/analysis/olmo3_sft_generation_stage_grid_32prompt.csv \
  --comparison-output artifacts/phase2/analysis/olmo3_sft_generation_stage_comparison_32prompt.csv \
  --summary-output artifacts/phase2/analysis/olmo3_sft_generation_stage_grid_32prompt_summary.md \
  --wandb-mode online
```

Compounding decomposition CLI:

```bash
uv run slop-analyze-phase2-compounding \
  --generation-cache base=artifacts/phase2/generations/olmo3_base_promptpkg512_free_run_128prompt_t0_t07_t1_batched128.jsonl \
  --generation-cache sft=artifacts/phase2/generations/olmo3_sft_promptpkg512_free_run_128prompt_t0_t07_t1_batched128.jsonl \
  --generation-cache dpo=artifacts/phase2/generations/olmo3_dpo_promptpkg512_free_run_128prompt_t0_t07_t1_batched128.jsonl \
  --generation-cache final=artifacts/phase2/generations/olmo3_final_promptpkg512_free_run_128prompt_t0_t07_t1_batched128.jsonl \
  --propensity-grid artifacts/phase2/analysis/olmo3_slop_neutral_common_normalized_stage_grid.csv \
  --feature slop_lexicon \
  --window-tokens 32 \
  --output artifacts/phase2/analysis/olmo3_generation_compounding_128prompt.csv \
  --summary-output artifacts/phase2/analysis/olmo3_generation_compounding_128prompt_summary.md \
  --plot-output artifacts/phase2/analysis/olmo3_generation_compounding_128prompt_realized_af.svg \
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

For the current free-running Phase 2 shards, keep the optimized
Torch/Transformers path as the default backend. After batched generation,
`torch.compile`, and fixed generation shapes, the 128-prompt OLMo shards run at
about 285-286 generated tokens/sec on the A100 including load and compile. That
is fast enough for the present sparse-grid measurements. The later
512-prompt x 8-completion x 1,024-token target-shape shards run at about
356-359 generated tokens/sec including load/compile.

SGLang has now been benchmarked in a CUDA 12.8 sidecar. It should still be
treated as a candidate backend rather than the default science path. With
default stop behavior, SGLang stops on OLMo's additional chat stop token
`<|im_end|>` (`100265`), producing only 1,117 generated tokens on a
64-prompt x 128-token DPO benchmark. With `--ignore-eos`, it produces the full
8,192 tokens and reaches 1,969 decode tokens/sec, but that explicitly changes
the stop-token contract. A paired first-64 check now confirms that
Torch/Transformers and SGLang with `--ignore-eos` use the same prompt IDs/order
and both generate the full 8,192-token fixed-length shape. The generated text
and feature counts are not bit-identical because backend sampling RNG differs,
and the controlled target-shape SGLang pilot shows that the aggregate
differences are too large to ignore. Keep Torch/Transformers as the science
backend for Phase 2 generation until SGLang can match Torch's stop-token
contract, or until a separate SGLang-specific measurement definition is
approved. vLLM remains blocked for OLMo-3 offline generation on this host.

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

The teacher-forced CLI also supports reference-subset summaries via
`--reference-subset NAME:FIELD=VALUE` and `--reference-subset-summary-output`.
The main summary remains the all-reference AF table, while the sidecar summary
reuses the same scored opportunities for named metadata subsets and logs a
redacted `propensity_reference_subset_summary` table to W&B. This is the
implementation hook for the EXPERIMENTS.md requirement to report AF against
both all-SFT and human-written-SFT references. The current 5,000-row Dolci
prompt package exposes `provenance`, `source_dataset`, `stratum`, and related
source metadata, but it does not yet contain a reliable `human_written` boolean;
use provenance/source-dataset filters now, and prefer a regenerated package
with explicit human/synthetic labels before making a human-written-SFT claim.
Subset filters are exact string matches with one predicate per named subset; if
the intended reference bucket spans multiple mixture names, materialize a
normalized field such as `reference_subset=human_reference` before scoring.
Prompt packaging now supports this materialization through
`--metadata-bucket-map PATH`; maps use JSON fields `output_field`,
`source_field` or `source_fields`, `values`, and `default`. The checked-in
`configs/phase2_dolci_reference_subset_seed_map.json` is a conservative seed
map with `code`, `synthetic_llm`, and `unknown` buckets. It deliberately has no
`human_reference` bucket: local Dolci metadata identifies prompt/source-mixture
labels, not target-response authorship. Do not publish a human-written-SFT AF
claim until a regenerated package or upstream audit supplies explicit
human/synthetic labels.

For already-generated prompt packages, use `slop-annotate-phase2-prompts`
instead of re-running `slop-prepare-phase2-prompts`; it preserves existing rows
and only adds mapped metadata. The current 5,000-row annotated package was
logged as W&B run `ew7x38n8` and wrote:
`artifacts/phase2/prompts/olmo3_dolci_sft_phase2_prompt_package_5000_reference_subset.jsonl`.
Bucket counts are `code=1371`, `synthetic_llm=225`, and `unknown=3404`.

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
- `stage2-phase2-olmo3-sft-generation-grid-32prompt-assembly`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/nrxd4ash`) assembled the
  32-prompt free-running shard into the new generation stage-grid schema. It
  wrote 15 feature-by-temperature rows and a 3-row `slop_lexicon` primary
  comparison under `artifacts/phase2/analysis/`, and logged aggregate tables to
  W&B without raw generations. The comparison preserves the compounding inputs
  (`repeated_count`, `repeat_per_generation`, and repeat-generation shares) so
  the same artifact family can support Result B once multi-stage/free-running
  shards are available.
- `stage2-phase2-olmo3-dpo-promptpkg512-free-running-32prompt-t0-t07-t1-batched128`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/5d1aa3zi`) completed the
  matching DPO free-running shard with the same 32-prompt package, temperatures
  `0.0`, `0.7`, and `1.0`, top-p 0.95, max 128 new tokens, batch size 8,
  bfloat16, and `torch.compile`. It produced 96 generations and 12,288
  generated tokens in 74.6 seconds, or 164.8 tokens/sec including load and
  compile. DPO had zero `slop_lexicon` hits at temperature 0.0, one at 0.7,
  and two at 1.0; it had rule-of-three counts 6, 8, and 6 across the three
  temperatures, including repeated rule-of-three hits in two temperature-1.0
  generations. Contrastive negation and stock closers had zero hits; stock
  openers appeared once at temperature 1.0.
- `stage2-phase2-olmo3-sft-dpo-generation-grid-32prompt-assembly`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/lgxd92cg`) assembled the SFT
  and DPO free-running shards into the first two-stage generation grid. The
  primary `slop_lexicon` comparison has DPO above SFT at temperature 0.7
  (`0.244` vs. `0.000` per 1k generated tokens) and temperature 1.0 (`0.488`
  vs. `0.000`), while SFT has the only greedy temperature-0.0 hit (`0.244` vs.
  `0.000`). This is still a sparse 32-prompt shard; treat it as a stage-grid
  plumbing and directional signal rather than a stable free-running rate
  estimate.
- `stage2-phase2-olmo3-final-promptpkg512-free-running-32prompt-t0-t07-t1-batched128`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/7o64hjk4`) completed the
  matching final/RLVR free-running shard with the same 32-prompt package and
  generation settings. It produced 96 generations and 12,288 generated tokens
  in 74.2 seconds, or 165.7 tokens/sec including load and compile. Final had
  zero `slop_lexicon` hits at temperature 0.0, one at 0.7, and two at 1.0,
  matching DPO on that primary feature in this sparse shard. Rule-of-three
  counts were 4, 5, and 3 across the three temperatures, with repeated
  rule-of-three hits in one temperature-0.7 generation and one temperature-1.0
  generation.
- `stage2-phase2-olmo3-sft-dpo-final-generation-grid-32prompt-assembly`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/i6lltomx`) assembled the SFT,
  DPO, and final free-running shards into a three-stage generation grid. The
  `slop_lexicon` primary comparison is: at temperature 0.0, SFT `0.244` per 1k
  generated tokens while DPO/final are zero; at temperature 0.7, DPO and final
  are both `0.244` while SFT is zero; at temperature 1.0, DPO and final are
  both `0.488` while SFT is zero. This is the first free-running stage-grid
  artifact with post-DPO and final checkpoints, but it remains too sparse for a
  stable rate claim or compounding estimate.
- `stage2-phase2-olmo3-base-promptpkg512-free-running-32prompt-t0-t07-t1-batched128`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/8hde64zv`) completed the
  matching base-checkpoint free-running shard with the same 32-prompt package
  and generation settings. It produced 96 generations and 12,288 generated
  tokens in 74.3 seconds, or 165.3 tokens/sec including load and compile. Base
  had one `slop_lexicon` hit at temperature 0.0, two at 0.7, and zero at 1.0.
  Rule-of-three counts were 3, 6, and 6 across the three temperatures, with one
  repeated rule-of-three hit at temperature 1.0.
- `stage2-phase2-olmo3-generation-grid-32prompt-assembly`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/g3eprlp9`) assembled the
  first full four-stage free-running grid for the OLMo Instruct ladder. The
  primary `slop_lexicon` comparison is sparse but now stage-complete for the
  32-prompt slice: at temperature 0.0, base and SFT are both `0.244` per 1k
  generated tokens while DPO/final are zero; at temperature 0.7, base is
  highest at `0.488`, DPO/final are `0.244`, and SFT is zero; at temperature
  1.0, DPO/final are highest at `0.488`, while base/SFT are zero. No
  `slop_lexicon` repeats occurred in any 32-prompt stage shard, so this slice
  still does not support a `slop_lexicon` self-conditioning estimate. The
  observed temperature dependence and sparse denominators argue for expanding
  the sample size before making free-running rate claims.
- `stage2-phase2-olmo3-dpo-promptpkg512-free-running-128prompt-t0-t07-t1-batched128`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/pel3icre`) expanded the DPO
  free-running shard to 128 prompts while keeping the same prompt package,
  temperatures, top-p 0.95, max 128 new tokens, batch size 8, bfloat16, and
  `torch.compile`. It produced 384 generations and 49,152 generated tokens in
  171.8 seconds, or 286.0 tokens/sec including load and compile. The larger
  shard gives nonzero primary-feature rates at every temperature:
  `slop_lexicon` counts were 5 at temperature 0.0, 5 at 0.7, and 10 at 1.0,
  corresponding to 0.305, 0.305, and 0.610 per 1k generated tokens. Repeated
  `slop_lexicon` hits appeared in two greedy generations and two
  temperature-1.0 generations, giving a first direct compounding signal for
  DPO, though this is still a single-stage 128-prompt shard.
- `stage2-phase2-olmo3-dpo-generation-grid-128prompt-assembly`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/5ky7wb2q`) assembled the
  128-prompt DPO free-running shard into the generation-grid schema. The
  primary comparison identifies temperature 1.0 as the maximum DPO
  `slop_lexicon` rate in this shard (`0.610` per 1k generated tokens), with
  repeat columns preserved for the Result B compounding analysis.
- `stage2-phase2-olmo3-sft-promptpkg512-free-running-128prompt-t0-t07-t1-batched128`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/z4fbaqmg`) completed the
  matching 128-prompt SFT free-running shard with the same prompt package and
  generation settings as DPO-128. It produced 384 generations and 49,152
  generated tokens in 171.8 seconds, or 286.0 tokens/sec including load and
  compile. SFT `slop_lexicon` counts were 10 at temperature 0.0, 6 at 0.7, and
  5 at 1.0, corresponding to 0.610, 0.366, and 0.305 per 1k generated tokens.
  Repeated `slop_lexicon` hits appeared in two greedy generations, giving a
  direct repeat/compounding signal at temperature 0.0.
- `stage2-phase2-olmo3-sft-dpo-generation-grid-128prompt-assembly`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/ktjei0eo`) assembled the
  matching 128-prompt SFT and DPO free-running shards. At temperature 0.0, SFT
  is higher than DPO for `slop_lexicon` (`0.610` vs. `0.305` per 1k generated
  tokens); at temperature 0.7, SFT is slightly higher (`0.366` vs. `0.305`);
  at temperature 1.0, DPO is higher (`0.610` vs. `0.305`). Both stages now
  have nonzero repeat columns in the 128-prompt slice, but the temperature
  interaction means the free-running result is not a simple monotonic DPO lift
  at this sample size.
- `stage2-phase2-olmo3-final-promptpkg512-free-running-128prompt-t0-t07-t1-batched128`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/fmvqwb44`) completed the
  matching 128-prompt final/RLVR free-running shard with the same prompt
  package and generation settings. It produced 384 generations and 49,152
  generated tokens in 172.4 seconds, or 285.1 tokens/sec including load and
  compile. Final `slop_lexicon` counts were 7 at temperature 0.0, 5 at 0.7,
  and 10 at 1.0, corresponding to 0.427, 0.305, and 0.610 per 1k generated
  tokens. Repeated `slop_lexicon` hits appeared in three greedy generations and
  two temperature-1.0 generations.
- `stage2-phase2-olmo3-sft-dpo-final-generation-grid-128prompt-assembly`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/oly88f3w`) assembled the
  128-prompt SFT, DPO, and final free-running shards. At temperature 0.0, final
  sits between SFT and DPO for `slop_lexicon` (`0.427` vs. SFT `0.610` and DPO
  `0.305` per 1k generated tokens). At temperature 0.7, final matches DPO
  (`0.305`) and is slightly below SFT (`0.366`). At temperature 1.0, final
  matches DPO (`0.610`) and is above SFT (`0.305`). This strengthens the
  temperature-dependent picture from the 32-prompt grid: post-DPO/final
  checkpoints show the clearest free-running lift at higher temperature, while
  greedy decoding does not mirror the teacher-forced DPO peak.
- `stage2-phase2-olmo3-base-promptpkg512-free-running-128prompt-t0-t07-t1-batched128`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/c1n1qi9u`) completed the
  matching 128-prompt base-checkpoint free-running shard with the same prompt
  package and generation settings. It produced 384 generations and 49,152
  generated tokens in 171.9 seconds, or 285.9 tokens/sec including load and
  compile. Base `slop_lexicon` counts were 1 at temperature 0.0, 6 at 0.7, and
  5 at 1.0, corresponding to 0.061, 0.366, and 0.305 per 1k generated tokens.
  Repeated `slop_lexicon` hits appeared once at temperature 0.7.
- `stage2-phase2-olmo3-generation-grid-128prompt-assembly`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/oidwhky5`) assembled the
  full base, SFT, DPO, and final 128-prompt free-running grid. The primary
  `slop_lexicon` comparison is: at temperature 0.0, SFT is highest (`0.610`
  per 1k), final is next (`0.427`), DPO is lower (`0.305`), and base is lowest
  (`0.061`); at temperature 0.7, base and SFT tie at `0.366`, while DPO/final
  are `0.305`; at temperature 1.0, DPO and final tie at `0.610`, while
  base/SFT are `0.305`. This completed 128-prompt grid confirms that
  free-running slop-lexicon rates are temperature-dependent and do not form a
  simple monotonic training-stage ladder. It does, however, provide nonzero
  repeat columns for SFT, DPO, and final, so the next compounding work should
  use this artifact family or a larger prompt sample rather than rely on the
  sparse 32-prompt shard.
- `stage2-phase2-olmo3-generation-compounding-128prompt-v2`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/wouelim1`) ran the first
  Result B compounding decomposition over the completed 128-prompt generation
  caches, joined to the teacher-forced
  `olmo3_slop_neutral_common_normalized_stage_grid.csv`. For `slop_lexicon`,
  observed free-running rates exceed the teacher-forced expected opportunity
  rate at most non-base-greedy cells. The largest excess is SFT greedy:
  observed `1.313` per 1k generation opportunities versus expected `0.342`,
  excess `0.971`, observed/expected `3.84`, realized AF `2.58`. At
  temperature 1.0, DPO and final also show large excesses (`0.717` and
  `0.740` per 1k opportunities) and realized AFs around `2.39` and `2.44`.
  The direct prior/no-prior window test is still sparse but nonzero: final
  greedy has the largest conditional delta (`0.375` vs. `0.013`
  hit-window probability, 3 repeat generations), followed by SFT greedy and
  DPO temperature 1.0. Treat this as the first bounded Result B artifact, not
  the final compounding estimate; larger generation shards are needed for
  stable conditional rates.
- Full 512-prompt free-running generation shards completed for the OLMo
  Instruct ladder with the same prompt package, temperatures `0.0`, `0.7`,
  and `1.0`, top-p 0.95, max 128 new tokens, batch size 8, bfloat16, and
  `torch.compile`. W&B runs: base
  (`https://wandb.ai/phulin-self/slop-stage1/runs/8rovnqs9`), SFT
  (`https://wandb.ai/phulin-self/slop-stage1/runs/aevb3cbt`), DPO
  (`https://wandb.ai/phulin-self/slop-stage1/runs/6o6cpvm0`), and final
  (`https://wandb.ai/phulin-self/slop-stage1/runs/8q42t9h3`). Each run
  generated 1,536 completions and 196,608 tokens at about 362-363 generated
  tokens/sec including load and compile. This is the largest free-running
  cache so far and uses the full 512-row held-out prompt package.
- `stage2-phase2-olmo3-generation-grid-512prompt-assembly`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/o135ar2v`) assembled the
  full 512-prompt four-stage generation grid. The `slop_lexicon` comparison is
  now cleaner than the 128-prompt slice: DPO is the maximum stage at all three
  temperatures. Rates per 1k generated tokens are base/SFT/DPO/final =
  `0.214/0.320/0.412/0.366` at temperature 0.0,
  `0.244/0.259/0.427/0.305` at temperature 0.7, and
  `0.320/0.366/0.458/0.443` at temperature 1.0. This aligns the
  free-running stage ordering more closely with the teacher-forced normalized
  AF peak at DPO, while final remains close to DPO at high temperature.
- `stage2-phase2-olmo3-generation-compounding-512prompt`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/y43cw1k9`) reran Result B
  on the 512-prompt caches joined to the teacher-forced stage grid. Observed
  `slop_lexicon` opportunity rates exceed teacher-forced expectation in every
  stage/temperature cell, but the larger sample reduces the most extreme
  128-prompt estimate. The largest excess is now SFT at temperature 1.0:
  observed `0.790` per 1k generation opportunities versus expected `0.342`,
  excess `0.448`, observed/expected `2.31`, realized AF `1.55`. DPO and final
  at temperature 1.0 are similar but below that excess (`0.383` and `0.381`
  per 1k opportunities). The direct window-level prior/no-prior signal remains
  positive in all `slop_lexicon` cells; the largest conditional delta is base
  temperature 1.0 (`0.172` vs. `0.011`, 5 repeat generations). Treat the
  512-prompt artifact as the current best bounded Result B read; the full
  EXPERIMENTS.md target remains 5k prompts x 8 completions x 1024 tokens.
- `stage2-phase2-olmo3-generation-compounding-512prompt-realized-af-plot`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/fwj8byv8`) regenerated the
  512-prompt compounding artifact with the required realized-AF temperature
  plot. The SVG artifact is
  `artifacts/phase2/analysis/olmo3_generation_compounding_512prompt_realized_af.svg`
  and is uploaded to W&B as `phase2_compounding_realized_af_plot`. The plotted
  maximum realized AF is DPO at temperature 1.0 (`1.733`), with final
  temperature 1.0 close behind (`1.731`) and SFT temperature 1.0 at `1.554`.
- `stage2-phase2-olmo3-generation-compounding-512prompt-tf1024`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/6miductd`) rejoined the same
  512-prompt generation caches to the newer 1024-prompt teacher-forced
  slop/neutral grid. This is the current best bounded Result B artifact because
  it uses the larger teacher-forced stage grid. Observed `slop_lexicon`
  opportunity rates still exceed teacher-forced expectation in every
  stage/temperature cell. The max realized AF remains DPO at temperature 1.0
  (`1.506`), with final temperature 1.0 effectively tied (`1.504`); the max
  absolute excess remains SFT at temperature 1.0 (`0.454` per 1k
  opportunities). The direct prior/no-prior signal remains positive but sparse,
  with the largest conditional delta at base temperature 1.0.
- Stock opener/closer teacher-forced propensity now has a matching 512-prompt
  four-stage OLMo grid. W&B runs: base
  (`https://wandb.ai/phulin-self/slop-stage1/runs/mj3zhxsv`), SFT
  (`https://wandb.ai/phulin-self/slop-stage1/runs/209dxyh6`), DPO
  (`https://wandb.ai/phulin-self/slop-stage1/runs/tomad94n`), final
  (`https://wandb.ai/phulin-self/slop-stage1/runs/cxtn5mdi`), and assembly
  (`https://wandb.ai/phulin-self/slop-stage1/runs/f1vcmj2k`). The pooled
  `stock_openers_closers` raw AFs are base `6.078`, SFT `5.214`, DPO `5.469`,
  and final `5.929`, so this feature family is highly amplified relative to
  model sequence mass but does not show the slop-lexicon DPO-stage lift in the
  current held-out package. `stock_closers` alone has only two reference
  initiations; use the pooled opener/closer row for current stage comparisons.
- The same 512-prompt denominator check found zero `contrastive_negation`
  reference initiations despite 10,978 opportunities. Contrastive negation
  therefore remains a Phase 2 measurement gap for this prompt package rather
  than a meaningful AF grid result.
- `stage2-phase2-dolci-sft-prompt-package-5000`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/77rz02r7`) prepared the
  full-size held-out prompt package for the EXPERIMENTS.md free-running target.
  It scanned 65,536 Dolci SFT rows, found 60,011 eligible prompt/target rows,
  filtered 5,524 near-duplicate prompts with the same MinHash settings as the
  512-prompt package, and selected 5,000 prompts. The package contains 664,230
  prompt tokens and 923,210 target tokens.
- `stage2-phase2-olmo3-dpo-promptpkg5000-free-running-64prompt-8comp-t1-bench1024`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/kji583m6`) completed the
  first near-target-shape DPO generation benchmark: 64 prompts, 8 completions
  per prompt, temperature 1.0, top-p 0.95, max 1,024 new tokens, batch size 8,
  bfloat16, and `torch.compile`. It generated 512 completions and 524,288
  tokens in 1,473.1 seconds, or 355.9 generated tokens/sec including load and
  compile. Batch size 8 completed on the A100 but approached the memory limit
  during some batches, so larger sweeps should either keep this as the upper
  batch-size bound or add an automatic fallback to batch size 4. Feature rates
  per 1k generated tokens were: `rule_of_three_approx` 0.771,
  `slop_lexicon` 0.269, `contrastive_negation` 0.143,
  `stock_openers_closers` 0.130, `stock_openers` 0.084, and `stock_closers`
  0.046. This is a benchmark and sparse feature read, not the full
  free-running grid.
- `stage2-phase2-olmo3-promptpkg5000-denominator-support-v2`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/lq4yiy27`) measured Phase 2
  denominator/reference support over the full 5,000-prompt held-out package
  without logging raw text. The package has strong support for
  `slop_lexicon` (917,669 opportunities, 506 references),
  `stock_openers_closers` (45,404 opportunities, 186 references), and
  `neutral_common_controls` (917,669 opportunities, 66,671 references).
  `contrastive_negation` is now nonzero but sparse with 111,662 opportunities
  and 7 references. `rule_of_three_approx` was requested but omitted from the
  teacher-forced denominator table because it still lacks a frozen Phase 2
  opportunity spec; keep it as a free-running measurement until that contract is
  defined.
- `stage2-phase2-olmo3-dpo-promptpkg5000-stock-openers-closers-cached-shared-branch8-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/5c9cut6v`) completed the
  first 5,000-prompt teacher-forced OLMo run after the denominator audit. It
  scored 45,404 `stock_openers_closers` opportunities on
  `allenai/Olmo-3-7B-Instruct-DPO` in 1,245.8 seconds, or 36.45
  opportunities/sec, with bfloat16, `torch.compile`, shared prefix caching, and
  branch batch size 8. Raw AF is 7.115 with 95% bootstrap CI 6.140-8.362
  against 186 reference initiations. This supersedes the 512-prompt DPO stock
  point estimate for single-checkpoint support, but the full 5k stock stage grid
  still requires base, SFT, and final runs.
- `stage2-phase2-olmo3-base-promptpkg5000-stock-openers-closers-cached-shared-branch8-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/tn7nd70e`) completed the
  matching 5,000-prompt base checkpoint run. It scored the same 45,404
  `stock_openers_closers` opportunities on `allenai/Olmo-3-1025-7B` in
  1,190.3 seconds, or 38.15 opportunities/sec. Raw AF is 7.997 with 95%
  bootstrap CI 6.952-9.391. The base point estimate is higher than the DPO
  point estimate, so this 5k stock feature currently looks like a broad model
  propensity rather than DPO-specific evidence; SFT and final remain required
  for the stage grid.
- `stage2-phase2-olmo3-sft-promptpkg5000-stock-openers-closers-cached-shared-branch8-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/7mk7lu7k`) completed the
  matching 5,000-prompt SFT checkpoint run. It scored the same 45,404
  opportunities on `allenai/Olmo-3-7B-Instruct-SFT` in 1,191.3 seconds, or
  38.11 opportunities/sec. Raw AF is 6.738 with 95% bootstrap CI 5.837-7.857.
  The current 5k stock ordering is base > DPO > SFT by point estimate, with all
  three strongly amplified. This weakens any preference-stage interpretation
  for pooled stock openers/closers and makes the final checkpoint the remaining
  calibration point for this feature.
- `stage2-phase2-olmo3-final-promptpkg5000-stock-openers-closers-cached-shared-branch8-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/238ph0bt`) completed the
  matching 5,000-prompt final/RLVR checkpoint run. It scored the same 45,404
  opportunities on `allenai/Olmo-3-7B-Instruct` in 1,187.4 seconds, or 38.24
  opportunities/sec. Raw AF is 7.715 with 95% bootstrap CI 6.644-9.056. The
  completed 5k stock opener/closer ordering is base > final > DPO > SFT by
  point estimate, with all checkpoints strongly amplified. Treat this pooled
  feature as a high-support calibration/control result rather than evidence
  for preference-stage-specific amplification.
- `stage2-phase2-olmo3-promptpkg5000-stock-openers-closers-stage-grid-assembly`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/xl4b7ss5`) assembled the 5k
  stock opener/closer teacher-forced stage grid using raw AF. The comparison
  artifact is written under `artifacts/phase2/analysis/` and confirms base as
  the maximum raw-AF stage: base 7.997, SFT 6.738, DPO 7.115, final 7.715. Raw
  AF deltas versus base are SFT -1.259, DPO -0.882, and final -0.282.
- `stage2-phase2-olmo3-promptpkg1024-slop-neutral-denominator-support`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/gloz8jdd`) measured the
  bounded primary slop/neutral shape before launching more GPU work. At
  `max_token_start_opportunities=128`, the 1,024-prompt slice has 88,903
  opportunities per feature, 52 `slop_lexicon` references, and 6,694
  `neutral_common_controls` references. This gives a compute-efficient
  intermediate validation point between the completed 512 stage grid and the
  estimated 18-20 A100-hour full 5k DPO slop/neutral run.
- `stage2-phase2-olmo3-dpo-promptpkg1024-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/tu89kg31`) completed the
  bounded DPO validation on the same 1,024-prompt slice. It scored 177,806
  opportunities in 6,404.6 seconds, or 27.76 opportunities/sec. The
  `slop_lexicon` neutral-normalized AF is 1.999 with 95% bootstrap CI
  1.446-2.837; raw slop AF is 0.751, and neutral raw AF is 0.376. This keeps
  the positive-control slop excess alive, but weakens the 512-prompt DPO point
  estimate of 2.605 and moves the estimate toward the SFT-sized roughly 2x
  effect. Treat the full 5k DPO slop/neutral cell as a high-power confirmation
  run, not as the next automatic step; first benchmark the pushed planned
  cached scorer path on the target OLMo shape.
- `stage2-phase2-olmo3-dpo-scorer-planned-vs-dynamic-branch8-prefix256`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/wv1v44ku`) benchmarked that
  planned cached path on the DPO target shape. Dynamic cached scoring reached
  29.55 feature-opportunities/sec and the prebuilt-plan path reached 29.72 at
  batch size 16, prefix 256, branch size 8. This 1.006x speedup means the
  scorer remains model-forward dominated; the next compute-efficient
  stage-localization step is a matching 1,024-prompt SFT slop/neutral run, not
  the full 5k DPO cell.
- `stage2-phase2-olmo3-sft-promptpkg1024-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/hoggmwz5`) completed that
  matching SFT run. It scored the same 177,806 opportunities in 6,110.7
  seconds, or 29.10 opportunities/sec. The `slop_lexicon`
  neutral-normalized AF is 1.695 with 95% bootstrap CI 1.315-2.289; raw slop
  AF is 0.575, and neutral raw AF is 0.339. Compared with DPO's 1.999
  normalized AF, this is a modest `1.18x` point-estimate lift with heavy CI
  overlap. The next compute-efficient stage-localization cell is base 1,024,
  not full 5k DPO.
- `stage2-phase2-olmo3-base-promptpkg1024-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/4mn6ycwv`) completed the
  matching base run. It scored the same 177,806 opportunities in 6,109.4
  seconds, or 29.10 opportunities/sec. The `slop_lexicon`
  neutral-normalized AF is 1.467 with 95% bootstrap CI 1.114-2.046; raw slop
  AF is 0.396, and neutral raw AF is 0.270. The 1,024-point ordering is DPO
  1.999 > SFT 1.695 > base 1.467, but the confidence intervals overlap. Treat
  this as evidence for inherited/base slop propensity with modest
  post-training amplification rather than a clean preference-stage-only effect.
  For exact teacher-forced propensity, keep the torch/Transformers scorer as
  the scorer of record because it directly audits fixed-branch probability
  mass, shared-prefix accounting, and feature/control normalization. SGLang or
  vLLM should be considered for free-generation throughput, or only after a
  separate exactness benchmark reproduces the torch branch-mass summaries.
- `stage2-phase2-olmo3-final-promptpkg1024-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/5v7x3180`) completed the
  matching final/RLVR run. It scored the same 177,806 opportunities in 6,109.9
  seconds, or 29.10 opportunities/sec. The `slop_lexicon`
  neutral-normalized AF is 1.659 with 95% bootstrap CI 1.189-2.351; raw slop
  AF is 0.760, and neutral raw AF is 0.458. Denominators match all other 1024
  cells exactly.
- `stage2-phase2-olmo3-promptpkg1024-slop-neutral-common-normalized-stage-grid-assembly`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/1cuuzoax`) assembled the
  completed 1024 teacher-forced stage grid. The normalized-AF ordering is DPO
  1.999 > SFT 1.695 > final 1.659 > base 1.467, with overlapping confidence
  intervals. This supports inherited/base slop propensity plus a modest DPO
  point-estimate peak that attenuates at final/RLVR. The next compute-efficient
  Phase 2 step should be generation/compounding work or a narrower
  confirmatory teacher-forced cell, not an automatic full-5k slop/neutral run.
- `stage2-phase2-olmo3-dpo-promptpkg5000-free-running-512prompt-8comp-t1-bench1024`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/8rud2kxl`) completed the
  bounded target-shape DPO generation shard on the 5,000-prompt package:
  512 prompts, 8 completions per prompt, temperature 1.0, top-p 0.95, and
  1,024 generated tokens per completion. It produced 4,096 generations and
  4,194,304 generated tokens in 11,758.0 seconds, or 356.7 generated
  tokens/sec including model load and compile. The feature rates per 1k
  generated tokens were `rule_of_three_approx` 0.861, `slop_lexicon` 0.229,
  `contrastive_negation` 0.141, `stock_openers_closers` 0.108,
  `stock_openers` 0.064, and `stock_closers` 0.043. This throughput matches
  the 64-prompt benchmark almost exactly, so the Torch/Transformers generation
  path scales linearly over prompts but remains slow enough that the next
  generation optimization should be an isolated SGLang or vLLM microbenchmark
  against this same JSONL/summary contract. Do not move the exact
  teacher-forced scorer off torch/Transformers unless a separate exactness
  benchmark reproduces the fixed-branch probability-mass summaries.
- `stage2-phase2-olmo3-final-promptpkg5000-free-running-512prompt-8comp-t1-bench1024`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/b8xo8axn`) completed the
  matching final/RLVR target-shape generation shard: 512 prompts, 8 completions
  per prompt, temperature 1.0, top-p 0.95, and 1,024 generated tokens per
  completion. It produced 4,096 generations and 4,187,136 generated tokens in
  11,758.7 seconds, or 356.1 generated tokens/sec including model load and
  compile. Final/RLVR is lower than DPO on `slop_lexicon` (`0.193` vs. `0.229`
  per 1k generated tokens, ratio `0.84`), `rule_of_three_approx` (`0.802` vs.
  `0.861`, ratio `0.93`), and `stock_openers_closers` (`0.091` vs. `0.108`,
  ratio `0.84`). `contrastive_negation` is effectively tied and slightly
  higher at final (`0.142` vs. `0.141`). This bounded target-shape comparison
  supports the same stage-localization story as the 1024 teacher-forced grid:
  DPO is the point-estimate peak for core slop-lexicon and stock-phrase
  features, and final/RLVR attenuates rather than amplifying further.
- `stage2-phase2-olmo3-sft-promptpkg5000-free-running-512prompt-8comp-t1-bench1024`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/il07fjn1`) completed the
  matching SFT target-shape generation shard: 512 prompts, 8 completions per
  prompt, temperature 1.0, top-p 0.95, and 1,024 generated tokens per
  completion. It produced 4,096 generations and 3,845,808 generated tokens in
  10,701.3 seconds, or 359.4 generated tokens/sec including model load and
  compile. SFT is lower than DPO on every tracked feature: `slop_lexicon`
  (`0.171` vs. `0.229` per 1k generated tokens, DPO/SFT ratio `1.34`),
  `rule_of_three_approx` (`0.488` vs. `0.861`, ratio `1.77`),
  `contrastive_negation` (`0.041` vs. `0.141`, ratio `3.45`), and
  `stock_openers_closers` (`0.072` vs. `0.108`, ratio `1.50`). SFT is also
  lower than final/RLVR on every tracked feature. The bounded target-shape
  comparison is now SFT/DPO/final rather than only DPO/final: DPO remains the
  slop-lexicon peak, final/RLVR attenuates from DPO but mostly remains above
  SFT, and SFT is the low-emission baseline for long free-running samples.
- `stage2-phase2-olmo3-base-promptpkg5000-free-running-512prompt-8comp-t1-bench1024`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/nw6yoj5u`) completed the
  matching base target-shape generation shard: 512 prompts, 8 completions per
  prompt, temperature 1.0, top-p 0.95, and 1,024 generated tokens per
  completion. It produced 4,096 generations and 4,174,328 generated tokens in
  11,677.3 seconds, or 357.5 generated tokens/sec including model load and
  compile. The base feature rates per 1k generated tokens were
  `rule_of_three_approx` 1.019, `slop_lexicon` 0.233,
  `contrastive_negation` 0.136, `stock_openers_closers` 0.092,
  `stock_openers` 0.064, and `stock_closers` 0.028.
- `stage2-phase2-olmo3-generation-target-shape-512prompt-8comp-t1-stage-grid`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/bojfccyx`) assembled the
  completed four-stage target-shape generation grid. The `slop_lexicon`
  ordering is base 0.233 > DPO 0.229 > final/RLVR 0.193 > SFT 0.171 per 1k
  generated tokens. Base is also the clear maximum for `rule_of_three_approx`
  (1.019 vs. DPO 0.861, final/RLVR 0.802, SFT 0.488), while DPO is the maximum
  for stock opener/closer features. This changes the bounded target-shape
  interpretation from a DPO generation peak to inherited/base free-running
  style plus feature-specific post-training reshaping; SFT remains the
  low-emission checkpoint and final/RLVR generally attenuates from DPO.
- `stage2-phase2-olmo3-generation-compounding-target-shape-512prompt-8comp-t1-tf1024`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/0klmyip9`) joined the
  target-shape generation caches to the 1,024-prompt teacher-forced
  slop/neutral grid. Observed `slop_lexicon` opportunity rates exceed expected
  rates in every stage at temperature 1.0: base excess `0.383` per 1k
  opportunities, SFT `0.361`, DPO `0.199`, and final/RLVR `0.137`. Max
  realized AF is SFT (`1.192`), not DPO; base has the largest absolute excess;
  final/RLVR is near the teacher-forced reference-rate baseline (`0.994`
  realized AF). Compared with the shorter temperature sweep, the target-shape
  join has much denser repeat counts and a more conservative compounding
  magnitude.
- `stage2-phase2-olmo3-dpo-promptpkg5000-free-running-512prompt-8comp-t07-bench1024`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/8no9vqyf`) started the
  bounded target-shape temperature-dependence expansion by adding DPO at
  temperature 0.7. It produced 4,096 generations and 4,188,320 generated
  tokens in 11,702.7 seconds, or 357.9 generated tokens/sec including model
  load and compile. DPO temp 0.7 is close to the matching temp 1.0 shard for
  `slop_lexicon` (`0.223` vs. `0.229` per 1k generated tokens) and pooled stock
  openers/closers (`0.107` vs. `0.108`), while `rule_of_three_approx` is
  slightly higher at 0.7 (`0.888` vs. `0.861`).
- `stage2-phase2-olmo3-dpo-promptpkg5000-free-running-512prompt-8comp-t0-bench1024`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/8s6spxu5`) completed the
  matching deterministic DPO target-shape shard. It produced 4,096 generations
  and 4,194,176 generated tokens. `slop_lexicon` was lower than the warmer
  shards (`0.206` per 1k generated tokens), while `contrastive_negation` was
  higher (`0.168` per 1k).
- `stage2-phase2-olmo3-dpo-generation-target-shape-temperature-grid`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/5azmz6pq`) assembled the
  completed DPO target-shape temperature sweep. `slop_lexicon` rates per 1k
  generated tokens are temperature 0.0 `0.206`, 0.7 `0.223`, and 1.0 `0.229`;
  pooled stock opener/closer rates are 0.0 `0.101`, 0.7 `0.107`, and 1.0
  `0.108`. Treat temperature as a modest modulator at this target shape, not
  the primary explanation for the stage differences.
- `stage2-phase2-olmo3-dpo-generation-compounding-target-shape-temperature-tf1024-v2`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/k6gcu75b`) joined the DPO
  target-shape temperature sweep to the 1,024-prompt teacher-forced
  slop/neutral grid. `slop_lexicon` Result B is positive at all three
  temperatures: observed/expected per 1k opportunities are 0.615/0.439 at
  temperature 0.0, 0.637/0.439 at 0.7, and 0.638/0.439 at 1.0. Realized AF is
  correspondingly 1.051, 1.089, and 1.091, so temperature has a small effect
  on compounding magnitude but is not the main driver.
- `stage2-phase2-olmo3-dpo-sglang052-cu128-1prompt-smoke-clean-finish`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/29i0m563`) validated that
  SGLang 0.5.2 can load OLMo-3 DPO on this host with CUDA 12.8, Torch
  2.8.0+cu128, Transformers 4.57.6, nvcc 12.8, and GCC/G++ 14. The run
  synced W&B artifacts, but the single sampled prompt stopped after one token,
  so it is only a plumbing smoke.
- `stage2-phase2-olmo3-dpo-sglang052-cu128-64prompt-t1-128tok-bench`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/hkdjxvmm`) showed that
  SGLang default stop behavior is not contract-compatible with the current
  Torch generation cache: 61/64 prompts stopped on `<|im_end|>` and only
  1,117 tokens were generated.
- `stage2-phase2-olmo3-dpo-sglang052-cu128-64prompt-t1-128tok-ignoreeos-bench`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/qija648z`) generated the
  full 64 x 128 fixed-length benchmark with `--ignore-eos`: 8,192 tokens,
  4.16 decode seconds, 1,969 decode tokens/sec, and 31.77 wall seconds
  including load and graph capture. Use this as a backend-throughput candidate
  datapoint, not as a science result until the stop-token contract is settled.
- `stage2-phase2-olmo3-dpo-sglang052-cu128-first64-t1-128tok-ignoreeos-contract`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/pqctgupb`) and
  `stage2-phase2-olmo3-dpo-torch-first64-t1-128tok-contract`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/b0zz1hzo`) ran the paired
  first-64 stop-token contract check. Both backends used the same prompt IDs
  in the same order and generated the full 8,192 tokens. SGLang wall throughput
  was about 271.9 tokens/sec on this small run, compared with Torch at 58.8
  tokens/sec; SGLang decode-only throughput was 1,921 tokens/sec. Feature
  counts differed because backend sampling RNG is not bit-identical, so use a
  larger aggregate pilot before relying on SGLang for science shards.
- `stage2-phase2-olmo3-dpo-sglang052-cu128-512prompt-8comp-t1-ignoreeos-ctx4096-pilot-v2`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/v69n4twp`) completed the
  target-shape SGLang pilot: 512 prompts x 8 completions x 1,024 tokens,
  4,096 generations, 4,194,304 generated tokens, 1,654.8 wall seconds, and
  2,584 decode tokens/sec. It is about 7.1x faster than the matching Torch DPO
  target-shape shard, but feature rates diverge materially: `slop_lexicon`
  `0.402` vs. Torch `0.229` per 1k generated tokens, `rule_of_three_approx`
  `1.588` vs. `0.861`, and pooled stock openers/closers `0.440` vs. `0.108`.
  Do not treat current SGLang `--ignore-eos` generations as a drop-in
  replacement for the Torch Phase 2 science cache.
