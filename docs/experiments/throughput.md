# Stage 1 Throughput Calibration

This note defines the CPU-first calibration pass before scaling Stage 1 corpus
sampling and Tier-1 census jobs. The goal is to find a stable sample size for
Dolci and Dolma without spending time on full passes that are already likely to
miss the available compute window.

## Task Plan Checklist

- [ ] Run Dolci tiny calibration for SFT, DPO, and RL with W&B disabled or tagged
      `dry_run`.
- [ ] Run Dolci small calibration for any split that will feed Phase 1 tables.
- [ ] Run Dolma tiny calibration with `HF_TOKEN` and a persistent HF cache set.
- [ ] Run Dolma small calibration only if tiny throughput and retry rate pass the
      go/no-go rules below.
- [ ] Promote to medium samples only after schema fields, denominators, artifact
      sizes, and W&B metadata look correct.
- [ ] Record the final chosen sample caps, seeds, scan limits, cache path, and
      observed throughput in the Stage 1 run summary.

## Metrics To Record

Record these for every calibration run, including failed or aborted runs:

- `corpus/docs`: normalized corpus records emitted.
- `corpus/measurement_rows`: measured text rows emitted; preference pairs count
  as two measurement rows.
- `corpus/tokens`: estimated tokens measured by the Stage 1 tokenizer/counting
  path.
- `corpus/docs_per_sec`: normalized records per wall-clock second.
- `corpus/measurement_rows_per_sec`: measured text rows per wall-clock second.
- `corpus/tokens_per_sec`: estimated tokens per wall-clock second.
- `corpus/mb_per_sec`: input or decoded text throughput when available.
- `corpus/wall_seconds`: end-to-end command time.
- `corpus/peak_rss_mb`: process peak memory.
- `corpus/output_size_mb`: local artifact size.
- `corpus/retry_count`: HF/network retries or resumptions.
- `corpus/cache_hit`: boolean or fraction when measurable.
- `corpus/sample_seed`, `corpus/sample_cap`, and `corpus/max_scan`: log as
  config values as well as in local notes.

Also record qualitative notes when they affect scale-up: timeout messages,
schema surprises, missing fields, pair-alignment failures, or denominator
anomalies.

## Sample-Size Ladder

Use `seed: 1729` unless a run is intentionally testing seed sensitivity. Keep
`max_scan` no larger than needed for the selected sampling strategy.

| Corpus | Tiny | Small | Medium | Notes |
| --- | ---: | ---: | ---: | --- |
| Dolci SFT | 32 rows | 512 rows | 5,000 rows | Use this to validate chat normalization and SFT provenance tags. |
| Dolci DPO | 32 pairs | 512 pairs | 5,000 pairs | Count chosen and rejected separately, but preserve 1:1 pair IDs. |
| Dolci RL | 32 rows | 512 rows | 5,000 rows | Verify role/response fields before combining with SFT. |
| Dolma 3 | 8 docs | 128 docs | 1,024 docs | Remote streaming is the risk path; require HF cache and token before small. |

Recommended progression:

1. Tiny: schema, denominator, W&B metadata, and artifact-shape validation.
2. Small: throughput estimate stable enough to project medium and full Stage 1
   sample cost.
3. Medium: only for corpora that will feed non-smoke Phase 1 tables.

Do not use Dolci throughput to estimate Dolma cost. Dolci rows are smaller,
structured, and less likely to stress remote streaming or cache behavior.

## Hugging Face Token And Cache

Use `HF_TOKEN` and a persistent cache for any Dolma run beyond a local schema
smoke. Dolma remote streaming has already shown timeout risk on a two-document
smoke, so authenticated requests and cache reuse are part of calibration, not a
later optimization.

Recommended local setup:

```bash
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN before Dolma calibration}"
export HF_HOME="${HF_HOME:-/home/user/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
```

Use `HF_TOKEN` for Dolci too if the local environment already relies on
authenticated Hugging Face access, but do not block Dolci tiny runs on it unless
the dataset request fails.

Cache rules:

- Keep the same `HF_HOME` and `HF_DATASETS_CACHE` across tiny, small, and medium
  runs so cache-hit behavior is measured realistically.
- Log the cache path as W&B config, but never log token values or environment
  dumps.
- If a Dolma tiny run has read timeouts or low cache reuse, repeat tiny once
  before starting small; record both attempts.

## W&B Names And Tags

Use project `slop-stage1` and group `corpus-sampling` for these runs. Run names
should encode stage, corpus, split/stratum, sample size, and dry/full status:

```text
stage1-dolci-sft-tiny-dry_run
stage1-dolci-dpo-small-dry_run
stage1-dolma3-web_cc-tiny-dry_run
stage1-dolma3-web_cc-medium-full_run
```

Example command:

```bash
uv run slop-census \
  --input allenai/Dolci-Instruct-SFT \
  --sample-size 32 \
  --max-scan 32 \
  --wandb-group corpus-sampling \
  --wandb-job-type throughput \
  --wandb-tag throughput \
  --wandb-tag olmo3 \
  --wandb-tag dolci_sft \
  --wandb-tag tiny \
  --wandb-tag dry_run \
  --wandb-run-name stage1-dolci-sft-tiny-dry_run \
  --output artifacts/stage1/census/dolci_sft_tiny.csv
```

Required tags:

- `stage1`
- `throughput`
- ladder tag: `olmo3` or `smollm3` when applicable
- corpus tag: `dolci_sft`, `dolci_dpo`, `dolci_rl`, or `dolma3`
- size tag: `tiny`, `small`, or `medium`
- status tag: exactly one of `dry_run` or `full_run`

Use `dry_run` for calibration runs that should not count as final Stage 1
evidence. Use `full_run` only after the run is intended to produce retained
Stage 1 artifacts and has passed the preflight rules below.

## Go/No-Go Before Scaling

Go from tiny to small only if:

- Schema normalization emits expected text/role fields.
- Dolci DPO chosen and rejected rows align 1:1 by stable pair ID.
- Peak RSS stays below 50% of available memory.
- Output artifacts are valid JSONL/Parquet and have expected row counts.
- W&B run name, tags, config, and metrics are complete and secret-free.

Go from small to medium only if:

- Projected medium wall time fits the current compute window with at least 2x
  headroom.
- `corpus/tokens_per_sec` is within 25% across repeated runs or the variance has
  a documented cause.
- Retry count is zero for Dolci and no more than one transient retry for Dolma.
- Dolma uses `HF_TOKEN` and a persistent cache, and the cache path is recorded.
- Matcher smoke output contains nonzero denominators for every enabled core
  Tier-1 feature.

No-go until fixed:

- Any run logs `WANDB_API_KEY`, `HF_TOKEN`, raw `.env` contents, or unredacted
  large text samples.
- Dolci DPO pair alignment breaks or chosen/rejected interpretation is uncertain.
- Dolma tiny projection implies medium would exceed the compute window without a
  smaller sample cap or local mirror.
- Peak RSS grows superlinearly with sample size.
- Artifact row counts, token counts, or denominator totals disagree with command
  inputs by more than expected filtering can explain.

When a no-go triggers, keep the calibration artifact and W&B run, tag it
`blocked`, and record the blocker in the Stage 1 notes before rerunning.
