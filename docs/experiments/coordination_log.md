# Coordination Log

## 2026-06-12

### Committed

- `85a21c1` adds the experiment plan from `EXPERIMENTS.md` and keeps `.env`
  out of Git.
- `5978a62` adds the Stage 1 scaffold: `uv` project setup, Tier-1 matchers,
  corpus streaming/sampling helpers, sampled census CLI, W&B helper, Stage 1
  configs, and component notes.
- `590eeae` adds role-aware census behavior, canonical Dolci dataset notes,
  and initial tests.
- `3c33c76` validates Dolci chat-schema normalization, adds `ty` to the
  `uv` toolchain, and records the first live W&B census smoke run.

### Active Components

- Per-pair preference deltas: produce the Result A artifact with feature-level
  `chosen_minus_rejected` rows, not only aggregate role rates.
- Type/lint baseline: keep `ty`, `ruff`, and pytest passing through `uv`.
- Dataset schema follow-up: Dolma 3 tiny-stream schema remains unverified.

### Coordination Decisions

- Stage 1 stays CPU-first. Regex census and corpus sampling should not consume
  A100 time.
- A100 work starts only after the teacher-forced harness has a validated small
  shard and eager-vs-`torch.compile` benchmark plan.
- All non-smoke experiment runs must log to W&B through `WANDB_API_KEY` loaded
  from `.env`. Local smoke checks should use `--wandb-mode disabled`.
- Corpus measurements can use deterministic samples. Record sample caps, seeds,
  and scan limits in W&B config and output tables.
- The local verification bar is `uv run ty check src tests`,
  `uv run ruff check src tests`, and `uv run pytest -q`.

### Next Go/No-Go Questions

- Do the Dolci SFT/DPO/RL datasets expose stable response and pair fields?
- Can DPO chosen/rejected roles be verified well enough to interpret Result A?
- Do core Tier-1 matchers pass precision checks after 200-hit validation?
- Does sampled corpus throughput leave enough headroom before adding Tier-2 or
  model-facing jobs?

### 2026-06-12 Stage 1 Smoke Results

- Live Dolci DPO tiny census completed with
  `slop-census --input allenai/Dolci-Instruct-DPO --sample-size 2 --max-scan 2`.
- W&B run: `stage1-dolci-dpo-smoke`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/9adp4xaf`.
- Output shape: 16 aggregate feature-rate rows, split by `chosen` and
  `rejected`; 4 sampled response rows logged to W&B.
- Verification after schema/normalizer updates:
  `uv run ty check src tests`, `uv run ruff check src tests`, and
  `uv run pytest -q` all pass.
- Live Dolci DPO pair-delta smoke completed with `--pair-output`.
  W&B run: `stage1-dolci-dpo-pair-smoke`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/yff0l3a2`.
- Pair-delta smoke output shape: 16 aggregate feature-rate rows and 16
  per-pair delta rows for 2 preference pairs; W&B logged
  `preference_pair_deltas/rows = 16`.
- Live Dolci SFT metadata smoke completed with
  `slop-census --input allenai/Dolci-Instruct-SFT --sample-size 2 --max-scan 2`.
  W&B run: `stage1-dolci-sft-metadata-smoke`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/ojo0qa87`.
- Live Dolma 3 tiny census completed with
  `slop-census --input allenai/dolma3_mix-6T-1025-7B --sample-size 2 --max-scan 2`.
  W&B run: `stage1-dolma3-tiny-smoke`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/f6ozyjv3`.
  The run produced 8 aggregate rows grouped under inferred stratum `web_cc`.
  HF had one read-timeout retry and took about 51 seconds for two docs, so
  larger Dolma samples need an authenticated/cache-aware throughput calibration
  before spending more time on remote streaming.
- Throughput metrics smoke completed with
  `slop-census --input allenai/Dolci-Instruct-SFT --sample-size 1 --max-scan 1`.
  W&B run: `stage1-throughput-metrics-smoke`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/ytwl9j2g`.
  The run logged `corpus/docs`, `corpus/measurement_rows`,
  `corpus/tokens_per_sec`, `corpus/wall_seconds`, `corpus/peak_rss_mb`, and
  a `source_metrics` W&B table.
- Tagged throughput calibration smoke completed with group `corpus-sampling`,
  job type `throughput`, and tags `throughput`, `olmo3`, `dolci_sft`, `tiny`,
  `dry_run`. W&B run: `stage1-dolci-sft-throughput-tiny-dry_run`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/mgvgwscs`.
