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
- Dataset schema follow-up: SmolLM3/Tulu replication source identification
  remains open.
- SmolLM3 replication-source identification: in-progress doc-only pass has
  primary Hugging Face candidates for pretraining, SmolTalk2 SFT, and APO
  preference sources; retained sampling stays blocked until bounded schema and
  source-weight probes are W&B-logged.

### Coordination Decisions

- Stage 1 stays CPU-first. Regex census and corpus sampling should not consume
  A100 time.
- A100 work starts only after the teacher-forced harness has a validated small
  shard and eager-vs-`torch.compile` benchmark plan.
- All non-smoke experiment runs must log to W&B through `WANDB_API_KEY` loaded
  from `.env`. Local smoke checks should use `--wandb-mode disabled`.
- Corpus measurements can use deterministic samples. Record sample caps, seeds,
  and scan limits in W&B config and output tables.
- SmolLM3/Tulu live probes must be bounded metadata/schema probes before any
  retained sampling. Use primary Hugging Face model/dataset cards and linked
  training recipes; log source URLs, config/split names, schema fields,
  source-count summaries, pair-role counts, scan caps, and failure modes to
  W&B without raw full text.
- The local verification bar is `uv run ty check src tests`,
  `uv run ruff check src tests`, and `uv run pytest -q`.

### 2026-06-12 SmolLM3/Tulu Source-ID Planning

- No experiment or dataset-load commands were run for this pass.
- Primary metadata inspected for planning: `HuggingFaceTB/SmolLM3-3B`,
  `HuggingFaceTB/smollm3-pretraining-datasets`, and
  `HuggingFaceTB/smoltalk2`.
- Pretraining source status: candidate public sources are identified from the
  SmolLM3 pretraining collection, but exact stage weights and source-to-stratum
  mapping still need the `huggingface/smollm` training recipe and bounded
  W&B-logged metadata probes.
- SFT status: `HuggingFaceTB/smoltalk2` `SFT` is the candidate source. The
  card reports 25 datasets split into `think` and `no_think` variants with
  chat-style messages and `chat_template_kwargs`; exact schema/config names and
  assistant-response extraction still need bounded probing.
- Preference status: `HuggingFaceTB/smoltalk2` `Preference` is the candidate
  APO substrate. Keep `llama_3.1_tulu_3_8b_preference_mixture_no_think`
  separate from `tulu_3_8b_pref_mix_Qwen3_32B_Qwen3_0.6B_think`; the latter is
  synthetic Qwen strong-vs-weak data and should not be pooled with the
  candidate human/RM-ranked Tulu preference source.

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
- Precision hit-sampling smoke completed with
  `slop-sample-hits --input allenai/Dolci-Instruct-SFT --sample-size 1 --max-scan 1`.
  W&B run: `stage1-dolci-sft-hit-sampling-tiny-dry_run`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/awlqhf7r`.
  The run found 26 candidate hits and wrote 10 sampled hit rows locally for
  labeling; W&B logged aggregate precision sampling counts only.
- Precision label scoring smoke completed with `slop-score-labels` on a tiny
  synthetic labeled CSV. W&B run: `stage1-score-labels-tiny-dry_run`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/ssqrchea`.
  The run scored 2 features, demoted 1 below the configured precision target,
  and logged a `precision_report` table.
- Result A pair-analysis smoke completed with `slop-analyze-pairs` on a
  2-pair Dolci DPO pair-delta sample. W&B run:
  `stage1-dolci-dpo-pair-analysis-tiny-dry_run`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/v6ffzun5`.
  The run summarized 16 pair-delta rows into 8 feature rows with sign-test
  p-values and BH-FDR q-values.
- Small Dolci DPO dry-run census completed with 32 preference pairs:
  `stage1-dolci-dpo-32pair-census-dry_run`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/g6ukhu9g`.
  Throughput: 32 docs, 64 response rows, 24,014 simple tokens, 4.89 seconds,
  about 4,913 tokens/sec, and 396 MB peak RSS. Output: 32 aggregate census rows
  and 256 pair-delta rows.
- Small Dolci DPO first-pass Result A analysis completed:
  `stage1-dolci-dpo-32pair-analysis-dry_run`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/8zs1n52y`.
  It summarized 256 pair-delta rows into 16 source/subset/feature rows. No
  Tier-1 feature was BH-FDR significant at this dry-run size. The main
  diagnostic issue is length imbalance in the primary `unknown` subset: chosen
  responses averaged about 156 more simple tokens than rejected responses,
  reinforcing the need for length covariates in the full Result A model.
- Small Dolci DPO length-aware Result A diagnostic completed:
  `stage1-dolci-dpo-32pair-length-analysis-dry_run`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/mttll9ms`.
  It reused the 256-row pair-delta CSV, summarized 16 feature rows, and logged
  response-side ridge logistic diagnostics for feature rate and response
  length. No Tier-1 feature was BH-FDR significant at this dry-run size.
- Retained 10k-row Dolci SFT+DPO Phase 1 census completed:
  `stage1-olmo3-dolci-sft-dpo-10k-census-retained_sample`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/jnhxeiy2`.
  It scanned 10,000 SFT rows and 10,000 DPO rows, measured 29,995 response
  rows, processed 8,443,044 simple tokens in 73.15 seconds, logged about
  115,423 tokens/sec, peaked at 1,270.8 MB RSS, wrote 72 feature-rate rows, and
  emitted 79,960 DPO pair-delta rows. One Hugging Face read timeout recovered
  on retry.
- Retained 10k-row Dolci DPO Result A analysis completed:
  `stage1-olmo3-dolci-dpo-10k-pair-analysis-retained_sample`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/njkg77h9`.
  It summarized 79,960 pair-delta rows into 16 source/subset/feature rows and
  found 9 BH-FDR-significant rows before manual precision validation. In the
  main `unknown` subset, chosen responses averaged about 34.1 more simple
  tokens than rejected responses; length remains a required covariate.
- Retained 10k-row Dolci SFT+DPO hit sampling completed:
  `stage1-olmo3-dolci-sft-dpo-10k-hit-sampling-retained_sample`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/t6wdmyj2`.
  It found 750,738 candidate matcher hits across 20,000 docs and 29,995
  response rows, then wrote the configured cap of 1,400 sampled rows for manual
  labeling: 200 each for contrastive negation, list/header/bold lead-in,
  punctuation/rhythm, rule-of-three, slop lexicon, stock closers, and stock
  openers. The CSV contains multiline context fields, so line counts exceed
  record counts.
- Retained 10k-row Dolci artifact manifest completed:
  `stage1-olmo3-dolci-10k-retained-artifact-manifest`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/t9mjtmbo`.
  It logged 4 local artifact records, 11,810,749 total bytes, and 81,448 CSV
  records: 72 feature-rate rows, 79,960 DPO pair-delta rows, 16 pair-analysis
  rows, and 1,400 precision-labeling rows. Local generated manifests are under
  `artifacts/stage1/corpora/olmo3_dolci_10k_retained_manifest.*`.
- Retained 10k-row Dolci metadata probe completed:
  `stage1-olmo3-dolci-10k-source-metadata-probe`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/xieewrhy`.
  The first 10,000 SFT rows exposed 18 `source_dataset` values and 8 domains.
  The first 10,000 DPO rows exposed four `preference_type` values:
  `llm_judged` (4,856), `delta_learning` (4,730),
  `multiturn_self_talk` (240), and `multiturn_synthetic_context` (174).
  The top chosen/rejected model fields were `qwen3-no_reasoning-32b` (5,168)
  and `qwen3-no_reasoning-0.6b` (5,669), reinforcing that Dolci DPO Result A
  should be interpreted as mixed preference-construction signal, not a pure
  human-preference signal.
- Retained 10k-row Dolci DPO metadata-aware Result A analysis completed:
  `stage1-olmo3-dolci-dpo-10k-metadata-pair-analysis-retained_sample`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/sgszvu4u`.
  Input:
  `artifacts/stage1/census/olmo3_dolci_dpo_10k_metadata_pair_deltas.csv`.
  Output:
  `artifacts/stage1/census/olmo3_dolci_dpo_10k_metadata_pair_analysis.csv`.
  It grouped 79,960 pair-delta rows by
  `source/subset/preference_type/chosen_model/rejected_model/feature`,
  producing 3,352 grouped feature rows and 72 BH-FDR-significant rows before
  manual precision validation. Group counts by `preference_type` were
  `llm_judged` 3,312 rows, `multiturn_self_talk` 16 rows,
  `multiturn_synthetic_context` 16 rows, and `delta_learning` 8 rows. Top
  significant examples include `delta_learning`
  `qwen3-no_reasoning-32b` vs `qwen3-no_reasoning-0.6b` for `stock_closers`
  (`n=4,730`, `mean_delta=0.169827...`, `p=2.628e-15`, `q=8.809e-12`,
  `chosen_gt_rejected`) and `punctuation_rhythm` (`n=4,730`,
  `mean_delta=4.271586...`, `p=5.253e-11`, `q=8.804e-08`), plus
  `llm_judged` `gpt-120b` vs `olmo2-1b` `punctuation_rhythm` (`n=64`,
  `mean_delta=95.202...`, `p=7.672e-10`, `q=8.572e-07`). This narrows the
  DPO interpretation blocker by separating preference construction and model
  pair, but it does not close Phase 1 because manual precision validation,
  final chosen/rejected construction semantics, and SmolLM3/Tulu work remain.
- Retained 10k-row Dolci DPO metadata-aware artifact manifest completed:
  `stage1-olmo3-dolci-dpo-10k-metadata-artifact-manifest`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/e7cc238x`. It records the
  metadata feature-rate, pair-delta, and pair-analysis CSVs as 3 artifacts,
  16,219,490 bytes, and 83,360 CSV records. Local generated manifests are under
  `artifacts/stage1/corpora/olmo3_dolci_10k_metadata_retained_manifest.*`.
- Retained Dolma 3 20k-scan pretrain sample completed:
  `stage1-olmo3-dolma3-20k-scan-retained-sample`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/l0ms0k7t`. It streamed
  20,000 rows from `allenai/dolma3_mix-6T-1025-7B`, retained 1,401 normalized
  rows with seed `1729`, excluded code, and populated four non-code strata:
  `web_cc` 1,000 docs / 1,702,321 tokens, `forums_qa` 279 / 236,029, `wiki`
  77 / 155,497, and `scientific` 45 / 36,091. The run logged about 44,741
  tokens/sec including streaming setup and peaked at 1,209 MB RSS. Local
  retained outputs include
  `artifacts/stage1/corpora/olmo3_dolma3_20k_scan_retained_sample.jsonl` and
  the formal per-record `artifacts/stage1/corpora/sample_manifest.parquet`.
- Retained Dolma 3 stratum census completed:
  `stage1-olmo3-dolma3-20k-scan-stratum-census-retained_sample`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/3my331x7`. It measured the
  1,401 retained rows, processed 2,129,938 simple tokens in 16.61 seconds,
  logged about 128,199 tokens/sec, and wrote 32 feature-rate rows to
  `artifacts/stage1/census/olmo3_dolma3_20k_scan_feature_rates.csv`.
- Retained Dolma 3 artifact manifest completed:
  `stage1-olmo3-dolma3-20k-scan-artifact-manifest`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/75vyt0mb`. It records 4
  retained artifacts, 16,446,028 total bytes, and 4,235 counted records:
  retained JSONL, Parquet sample manifest, CSV sample manifest, and stratum
  census CSV. Local generated manifests are under
  `artifacts/stage1/corpora/olmo3_dolma3_20k_scan_artifact_manifest.*`.
- Dolci SFT retained corpus package completed:
  `stage1-olmo3-dolci-sft-10k-retained-corpus-package-schema-fix`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/t5l79cla`. It streamed
  10,000 rows from `allenai/Dolci-Instruct-SFT`, retained 10,000 normalized
  `target_response` rows, measured 1,806,697 simple tokens, and wrote
  `artifacts/stage1/corpora/olmo3_dolci_sft_10k_retained_sample.jsonl` plus
  Parquet/CSV/JSON/Markdown per-record manifests.
- Dolci DPO retained corpus package completed:
  `stage1-olmo3-dolci-dpo-10k-complete-pair-corpus-package-schema-fix`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/bucn0h4p`. It scanned 10,006
  rows from `allenai/Dolci-Instruct-DPO` to retain 10,000 complete preference
  pairs, wrote 20,000 normalized text rows split exactly into 10,000 `chosen`
  and 10,000 `rejected`, and measured 6,637,271 simple tokens. Original
  `prompt_id` is preserved separately from a unique row-index-qualified
  `pair_id`; validation found 10,000 pair IDs and zero bad role pairs.
- Dolci SFT/DPO retained corpus package manifest completed:
  `stage1-olmo3-dolci-sft-dpo-10k-corpus-package-manifest`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/dzrfn409`. It records 10
  local artifacts, 186,303,114 bytes, and 90,000 counted JSONL/Parquet/CSV
  records. Local generated manifests are under
  `artifacts/stage1/corpora/olmo3_dolci_sft_dpo_10k_corpus_package_manifest.*`.
- SmolLM3/Tulu source-identification probe completed:
  `stage1-smollm3-tulu-source-identification-probe`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/wiedc2oj`. Local artifact:
  `artifacts/stage1/corpora/smollm3_tulu_source_probe.json`. The probe covered
  5 dataset probes, 2 model probes, 3 dataset searches, 1 model search, 7
  sample rows, and 80 search-result rows.
- SmolTalk2 status from the probe: `HuggingFaceTB/smoltalk2` exposes `Mid`,
  `Preference`, and `SFT` configs. `Preference` splits include
  `llama_3.1_tulu_3_8b_preference_mixture_no_think` and
  `tulu_3_8b_pref_mix_Qwen3_32B_Qwen3_0.6B_think`; `SFT` exposes many
  source-labeled `think` and `no_think` splits, including
  `OpenThoughts3_1.2M_think`,
  `smoltalk_smollm3_everyday_conversations_no_think`,
  `smoltalk_smollm3_smol_magpie_ultra_no_think`, and
  `tulu_3_sft_personas_instruction_following_no_think`.
- Tulu preference-mixture status from the probe:
  `allenai/llama-3.1-tulu-3-8b-preference-mixture` and
  `allenai/llama-3.1-tulu-3-70b-preference-mixture` both load with default
  `train` rows shaped as `id`, `source`, `prompt`, `chosen`, and `rejected`;
  `chosen` and `rejected` are message lists.
- SmolLM3 metadata-source status from the probe:
  `HuggingFaceTB/smollm3-configs` is useful metadata/config-card evidence but
  not loadable as dataset files, and `HuggingFaceTB/smollm3-blueprint` is
  documentation/PDF-oriented with sample loading blocked on `pdfplumber`.
  Generic SmolTalk2 loading without a config fails, so the next implementation
  step is config/split-aware bounded probing.
- SmolTalk2 config-specific source probe completed:
  `stage1-smollm3-smoltalk2-config-source-probe`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/wo6496wj`. Local artifact:
  `artifacts/stage1/corpora/smollm3_tulu_config_source_probe.json`.
- Probe inputs were five exact dataset specs plus one model probe:
  `HuggingFaceTB/smoltalk2::Preference::llama_3.1_tulu_3_8b_preference_mixture_no_think`,
  `HuggingFaceTB/smoltalk2::Preference::tulu_3_8b_pref_mix_Qwen3_32B_Qwen3_0.6B_think`,
  `HuggingFaceTB/smoltalk2::SFT::smoltalk_smollm3_everyday_conversations_no_think`,
  `HuggingFaceTB/smoltalk2::SFT::OpenThoughts3_1.2M_think`,
  `HuggingFaceTB/smoltalk2::Mid::OpenThoughts3_1.2M`, and model
  `HuggingFaceTB/SmolLM3-3B`.
- Each dataset spec loaded successfully with 2 shape-only sample rows.
  Preference rows expose `chosen`, `rejected`, `prompt`,
  `chat_template_kwargs`, and `source`; SFT rows expose `messages`,
  `chat_template_kwargs`, and `source`; Mid `OpenThoughts3_1.2M` rows expose
  `messages` and `source`.
- The config-specific probe closes the generic-config loader issue and proves
  config/split-aware bounded probing works, but Phase 1 remains open pending
  broader split/source count census, target response extraction/normalization,
  Tulu construction semantics, and pretraining recipe weights.
- Revised Phase 1 scoped Biber-lite census runs completed. The required core
  scope is contrastive negation, rule-of-three approximation, slop lexicon, and
  stock openers/closers; punctuation/rhythm, list/header/bold lead-ins, and
  generic hedging are historical/exploratory only.
- Revised Dolci SFT retained run:
  `stage1-olmo3-dolci-sft-10k-revised-biber-lite-census`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/05zob6cx`. It measured
  10,000 retained `target_response` rows, 1,806,697 simple tokens, and wrote
  `artifacts/stage1/census/olmo3_dolci_sft_10k_revised_biber_lite_feature_rates.csv`.
- Revised Dolma 3 retained sample run:
  `stage1-olmo3-dolma3-20k-scan-revised-biber-lite-census`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/zrpkl0t7`. It measured 1,401
  retained pretrain rows, 2,129,938 simple tokens, and wrote
  `artifacts/stage1/census/olmo3_dolma3_20k_scan_revised_biber_lite_feature_rates.csv`.
- Revised Dolci DPO retained package run:
  `stage1-olmo3-dolci-dpo-10k-retained-revised-biber-lite-census`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/tmgvdy4t`. It measured
  20,000 retained rows, 6,637,271 simple tokens, 10,000 unique package pair
  IDs, wrote 100 feature-rate rows, and emitted 250,000 pair-feature delta rows
  to
  `artifacts/stage1/census/olmo3_dolci_dpo_10k_retained_revised_biber_lite_pair_deltas.csv`.
