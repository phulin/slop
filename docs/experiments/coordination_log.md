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
- Phase 2 teacher-forced smoke scaffold started. Added opportunity extraction
  and a compiled causal-LM smoke CLI. Successful W&B run:
  `stage2-phase2-tiny-gpt2-compile-propensity-smoke-v2`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/w7b161sw`. It ran
  `sshleifer/tiny-gpt2` on the A100 with `torch.compile`, processed 2 retained
  Dolci SFT rows and 64 opportunities, and wrote local ignored smoke artifacts
  under `artifacts/phase2/propensity/`. The first compile attempt exposed an
  empty-`LD_LIBRARY_PATH` Triton lookup issue for `libcuda.so.1`; the CLI now
  adds `/usr/lib/x86_64-linux-gnu` when that library exists.
- Phase 1 handoff conclusions were recorded in
  `docs/experiments/phase2_handoff_conclusions.md`. Phase 1 is closed as a
  sampled OLMo Dolci measurement package with caveats; full SmolLM remains
  future work, and punctuation plus list/header features are outside the core
  claims.
- Phase 2 exact-sequence teacher-forced propensity is now the default
  estimator. First-token propensity remains available as a smoke/debug fallback.
  Tiny GPT-2 exact-sequence W&B run:
  `stage2-phase2-tiny-gpt2-sequence-propensity-smoke`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/45ozhmum`.
- First real OLMo SFT tiny shard completed:
  `stage2-phase2-olmo3-7b-instruct-sft-sequence-tiny`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/pp0x4f2y`. It used
  `allenai/Olmo-3-7B-Instruct-SFT`, measured 8 `slop_lexicon` opportunities,
  found zero reference initiations, mean probability mass `1.103147e-06`,
  85.1 wall seconds, and 0.094 opportunities/sec. This validates model loading,
  CUDA, W&B logging, and exact-sequence plumbing only; it is not an
  amplification result.
- OLMo checkpoint IDs were corrected for Phase 2: base
  `allenai/Olmo-3-1025-7B`, SFT `allenai/Olmo-3-7B-Instruct-SFT`, DPO
  `allenai/Olmo-3-7B-Instruct-DPO`, and final
  `allenai/Olmo-3-7B-Instruct`.
- Phase 2 free-running emission CLI smoke completed:
  `stage2-phase2-tiny-gpt2-free-running-smoke-v2`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/4dt5kc3h`. It generated 16
  tokens from one retained-sample prompt with `torch.compile` enabled, logged
  redacted W&B sample rows, and wrote local ignored artifacts under
  `artifacts/phase2/generations/`. Summary rates were zero for `slop_lexicon`
  and `stock_openers`. The prior free-running attempt
  `https://wandb.ai/phulin-self/slop-stage1/runs/rhfcryb6` failed because this
  GPT-2 generation path rejected the `generator` kwarg; the CLI now uses
  `torch.manual_seed` and `torch.cuda.manual_seed_all` instead.
- Phase 2 held-out prompt packaging started. Added a dependency-free MinHash
  utility and `slop-prepare-phase2-prompts`, which writes local prompt/target
  JSONL plus a redacted manifest for W&B. First Dolci SFT prompt package run:
  `stage2-phase2-dolci-sft-prompt-package-8`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/7xkqxesu`. It scanned 128
  `allenai/Dolci-Instruct-SFT-7B` rows, found 125 eligible prompt/target rows,
  filtered 3 near-duplicate prompts, selected 8 prompts, and wrote local ignored
  artifacts under `artifacts/phase2/prompts/`.
- Teacher-forced exact-sequence mass was batched by continuation depth. For
  the current 1-3 token initiator sequences this reduces scoring from one model
  forward per sequence token to at most three forwards per opportunity, before
  any later trie/prefix-cache optimization.
- First prompt-package-backed OLMo SFT teacher-forced run completed:
  `stage2-phase2-olmo3-sft-promptpkg8-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/mbsjovsf`. It used
  `allenai/Olmo-3-7B-Instruct-SFT`, the 8-row Dolci SFT prompt package, exact
  sequence mass, `torch.compile`, and the A100. It scored 8 docs, 227
  opportunities, and 4 feature summaries in 376.8 seconds, for 0.602
  opportunities/sec. Mean probability masses were `1.47e-05` for
  `contrastive_negation`, `1.62e-04` for `slop_lexicon`, `3.32e-04` for
  `stock_closers`, and `8.79e-05` for `stock_openers`. There were zero
  reference initiations, so this is a plumbing and throughput run, not an AF
  result.
- Neutral controls were added to the Phase 2 propensity spec:
  `neutral_for_example`, `neutral_such_as`, `neutral_in_particular`,
  `neutral_as_a_result`, and pooled `neutral_controls`. These share the
  token-start opportunity contract and are calibration targets for SFT AF near
  1, not slop features.
- Prompt packaging now supports `--require-reference-feature`, which builds
  targeted smoke/control shards whose target text actually initiates selected
  features. This is for calibration and positive-control plumbing, not final
  unbiased AF estimates.
- Larger Dolci SFT package run:
  `stage2-phase2-dolci-sft-prompt-package-128`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/dwqj49uk`. It scanned 2,048
  rows, found 1,974 eligible prompt/target rows, filtered 74 near duplicates,
  and selected 128 prompts. Local reference counts at a 128 token-start cap:
  10 `slop_lexicon`, 4 `stock_openers`, and 5 pooled `neutral_controls`
  positives; still zero `contrastive_negation` and `stock_closers`.
- Positive/control Dolci SFT package run:
  `stage2-phase2-dolci-sft-positive-control-package-32`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/wl0o62kn`. It scanned 4,096
  rows, found 347 rows that initiate `slop_lexicon` or pooled
  `neutral_controls`, selected 32 prompts, and gives a compact next GPU shard.
  The first 4 rows at a 128 token-start cap have 902 total opportunities with
  2 `slop_lexicon`, 2 `neutral_controls`, and 1 `stock_openers` reference
  positives.
- The first positive/control OLMo scoring attempt
  `stage2-phase2-olmo3-sft-positive-control4-sequence`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/mgsyvegv`) was stopped before
  completion because exact sequence scoring remained too slow. The sequence
  mass loop no longer synchronizes once per initiator probability; it now keeps
  probability accumulation on the GPU and performs a single final scalar sync.
- Added `slop-benchmark-propensity-scorer` to compare scalar replay, batched
  replay, and batched KV-cache continuation scoring without running a full
  corpus job. Tiny GPT-2 benchmark:
  `stage2-phase2-tiny-gpt2-propensity-scorer-cache-benchmark`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/c0hgm5lm`. At batch 16,
  scalar replay measured 239.6 opportunities/sec, batched replay 1954.0, and
  batched KV-cache 2932.9, with max absolute difference
  `5.82e-11` versus scalar.
- OLMo SFT scorer microbenchmark:
  `stage2-phase2-olmo3-sft-propensity-scorer-cache-benchmark`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/fqeur4ct`. At 256 prefix
  tokens and batch 16, scalar replay measured 2.73 opportunities/sec, batched
  replay 3.10, and batched KV-cache 29.07, with max absolute difference
  `2.75e-04` versus scalar. Torch Dynamo hit the recompile limit during the
  mixed-mode benchmark, so full runs should keep scorer mode and shapes as
  stable as possible.
- A one-document cached OLMo integration attempt with unchunked branches
  (`https://wandb.ai/phulin-self/slop-stage1/runs/dhss4z9n`) OOMed under
  `torch.compile`; continuation branch fanout can still materialize too much
  attention state even when prefix KV cache is used.
- Added `--cache-branch-batch-size` to split cached continuation branches. OLMo
  SFT benchmark with branch chunk size 2:
  `stage2-phase2-olmo3-sft-propensity-scorer-cache-branch2-benchmark`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/6j4g3p0v`. At batch 16,
  scalar replay measured 2.73 opportunities/sec, batched replay 3.17, and
  branch-chunked KV-cache 25.35 with max absolute difference `2.44e-04`.
- Branch-chunked cached OLMo integration smoke succeeded:
  `stage2-phase2-olmo3-sft-positive-control1-cached-branch2-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/mbdegkow`. It scored one
  target document, 131 opportunities, and three feature summaries without OOM.
  End-to-end throughput including compile overhead was 1.23 opportunities/sec.
  The row had one pooled `neutral_controls` reference initiation and no
  `slop_lexicon` or `stock_openers` reference initiations, so this validates the
  optimized scorer path but is not an AF result.
- First bounded AF-style OLMo SFT run completed:
  `stage2-phase2-olmo3-sft-positive-control4-cached-branch2-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/bigtxtoj`. It used the
  optimized exact sequence scorer with opportunity batch size 16, KV-cache
  continuations, branch chunk size 2, and `torch.compile`. It scored 4 docs,
  902 opportunities, and 3 feature summaries in 92.3 seconds, for 9.78
  opportunities/sec end to end. Reference initiations were 2 for
  `slop_lexicon`, 2 for pooled `neutral_controls`, and 1 for `stock_openers`.
  AFs were 0.434 for `slop_lexicon`, 0.296 for pooled `neutral_controls`, and
  0.00036 for `stock_openers`. This is the first nonzero-denominator
  Phase 2 model measurement, but because the package was targeted to contain
  reference positives it is a bounded calibration/plumbing shard, not an
  unbiased corpus AF estimate.
- CPU-side opportunity extraction benchmark completed:
  `stage2-phase2-positive-control32-opportunity-extraction-benchmark`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/4gy3euvz`. On the 32-row
  positive/control package, full extraction measured up to 142.8k
  opportunities/sec and offsets-only enumeration measured up to 1.04M
  opportunities/sec. This rules out CPU opportunity extraction as the next
  full-run bottleneck. Opportunity extraction is now selected-feature aware, so
  neutral-only or narrow-feature runs skip irrelevant Tier-1 matcher families.
- Shared-prefix cached scorer benchmark completed:
  `stage2-phase2-olmo3-sft-propensity-scorer-multifeature-branch2-benchmark-v2`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/a2lcsbg9`. At batch 16 and
  256 fixed prefix tokens, single-feature cached OLMo scoring measured 24.9
  opportunities/sec; shared multi-feature cached scoring measured 53.2
  feature-opportunities/sec across three features with max absolute difference
  `1.83e-04` versus scalar on the reference feature. Use shared-prefix cached
  scoring for the next multi-feature OLMo shard.
- Document-cluster bootstrap CIs were added to
  `slop-teacher-forced-propensity` summaries. The summary CSV and W&B table now
  include point estimates plus 95% bootstrap intervals for reference rate,
  mean model probability mass, and AF. The bootstrap resamples measurement
  documents, not individual opportunities.
- Positive/control-32 OLMo SFT shard completed:
  `stage2-phase2-olmo3-sft-positive-control32-cached-shared-branch2-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/ce8i6wd9`. It scored 32
  targeted documents and 7,862 opportunities in 268.5 seconds, for 29.3
  opportunities/sec end to end. Point AFs were 0.289 for pooled
  `neutral_controls`, 0.298 for `slop_lexicon`, and 0.0014 for
  `stock_openers`. Because the package is reference-positive by construction,
  this remains a calibration/plumbing shard rather than an unbiased AF estimate.
- Held-out prompt-package-128 OLMo SFT shard completed:
  `stage2-phase2-olmo3-sft-promptpkg128-cached-shared-branch2-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/n8zueel5`. It scored 128
  prompt-held-out SFT targets and 23,656 opportunities in 680.5 seconds, for
  34.8 opportunities/sec end to end. Point AFs were 0.629 for
  `neutral_controls` and 0.746 for `slop_lexicon`; the bootstrap intervals
  were wide and included 1 (`neutral_controls`: 0.315-2.028,
  `slop_lexicon`: 0.296-2.323). Reference denominators are still too small for
  a calibration gate: only 5 neutral-control initiations and 10 slop-lexicon
  initiations.
- Larger held-out prompt package prepared:
  `stage2-phase2-dolci-sft-prompt-package-512`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/i6sh0rcv`. It scanned 8,192
  Dolci SFT rows, found 7,753 eligible prompt/target rows, filtered 439
  near-duplicate prompts, and selected 512 prompts. CPU denominator check
  `stage2-phase2-promptpkg512-opportunity-extraction`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/ukrfj8ud`) found 90,524
  opportunities and 45 combined reference initiations across `slop_lexicon`
  and pooled `neutral_controls`, enough to run a tighter calibration shard.
- Held-out prompt-package-512 OLMo SFT shard completed:
  `stage2-phase2-olmo3-sft-promptpkg512-cached-shared-branch2-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/35b1isae`. It scored 512
  prompt-held-out SFT targets and 90,524 opportunities in 2,422.9 seconds, for
  37.4 opportunities/sec end to end. Point AF was 0.357 for pooled
  `neutral_controls` with 95% bootstrap interval 0.258-0.531, and 0.608 for
  `slop_lexicon` with interval 0.361-1.139. This gives the first reasonably
  tight SFT shard, but it fails the neutral-control calibration gate: under
  the current pooled neutral-control opportunity/initiator definition, the SFT
  checkpoint underpredicts reference initiations.
- Neutral-control breakdown completed:
  `stage2-phase2-olmo3-sft-promptpkg512-neutral-breakdown-cached-shared-branch2-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/gkj0vlvc`. It scored the same
  512 prompt-held-out SFT targets with individual neutral features plus the
  pooled basket, measuring 226,310 feature-opportunities in 1,320.4 seconds
  for 171.4 feature-opportunities/sec. `neutral_such_as` had 16 reference
  initiations with AF 0.424 and 95% CI 0.296-0.713; `neutral_for_example` had
  6 reference initiations with AF 0.103 and CI 0.051-0.337.
  `neutral_in_particular` and `neutral_as_a_result` had zero observed
  reference initiations in this sample. The pooled neutral basket remains
  underpredicted with AF 0.357 and CI 0.258-0.531, so full-grid promotion
  remains blocked by the neutral calibration gate.
- Inference runtime decision: keep `torch`/Transformers for the current exact
  teacher-forced AF scorer. It needs arbitrary candidate continuation
  probability mass at fixed corpus offsets, not just generated tokens or
  top-k request logprobs. Evaluate vLLM or SGLang for the later free-running
  generation jobs, and only revisit teacher-forced serving backends after an
  isolated benchmark proves estimator-equivalent continuation mass.
- Neutral calibration follow-up: the neutral breakdown exposed a measurement
  bug in initiator coverage. Neutral reference matching is case-insensitive,
  but exact sequence initiators were lowercase-only; all 6 observed
  `neutral_for_example` references in the 512-row run were capitalized
  `For example`. Initiator enumeration now includes sentence-case variants for
  every phrase. Rerun the neutral calibration after this fix before promoting
  or permanently demoting the neutral basket.
- Batched free-running smoke completed:
  `stage2-phase2-tiny-gpt2-batched-free-running-smoke`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/pnjtz101`. It generated 8
  tiny GPT-2 completions on the Phase 2 prompt package with temperatures 0.0
  and 0.7, validating batched generation, top-p recording, repeat metrics, and
  redacted W&B logging.
- First W&B-logged OLMo SFT free-running shard completed:
  `stage2-phase2-olmo3-sft-promptpkg512-free-running-16prompt-t0-t07-batched`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/2co91og0`. It used the
  512-row held-out prompt package, 16 prompts, temperatures 0.0 and 0.7,
  top-p 0.95, max 64 new tokens, one completion per prompt, batch size 4,
  bfloat16, and `torch.compile`. It produced 32 generations and 2,048
  generated tokens in 207.7 seconds, with no raw text logged to W&B. This
  closes the gap from tiny free-running smoke to a bounded real OLMo
  free-running Phase 2 run; the full generation grid still needs a serving
  backend or a larger batching benchmark before scaling.
- Neutral case-variant rerun completed:
  `stage2-phase2-olmo3-sft-promptpkg512-neutral-breakdown-case-variants-cached-shared-branch2-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/0rz3idjo`. It used the same
  512 prompt-held-out documents, same opportunity cap, and same neutral
  features as `gkj0vlvc`, changing only initiator surface coverage to include
  sentence case. It scored 226,310 feature-opportunities in 1,875.1 seconds
  for 120.7 feature-opportunities/sec. `neutral_for_example` improved to AF
  0.711 with CI 0.407-2.076, confirming the casing bug. The pooled basket
  improved to AF 0.545 but still has CI 0.408-0.800, and `neutral_such_as`
  remains low at AF 0.429 with CI 0.300-0.722. Do not promote the full
  teacher-forced grid on the current neutral gate; either revise the neutral
  controls to higher-support low-style items or treat `such as` as a discourse
  feature rather than a neutral sanity check.
- Common neutral-control calibration completed:
  `stage2-phase2-olmo3-sft-promptpkg512-neutral-common-controls-cached-shared-branch2-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/xw8ihrej`. This tested
  `the`, `a`, `of the`, `number of`, `in the`, `to the`, `is a`, and pooled
  `neutral_common_controls` on the same 512 prompt-held-out SFT targets. The
  pooled basket had strong support (3,425 references), but AF was 0.339 with
  CI 0.319-0.360. `in the` was closest at 0.738; every individual common
  control was below 1. This shows the raw neutral AF≈1 gate is not valid under
  broad token-start opportunities, even with high-support function-word
  controls. The next Phase 2 implementation step is to add neutral-normalized
  reporting or change the opportunity contract before promoting to the full
  model grid.
- Teacher-forced summaries now support `--normalization-feature`, which adds
  neutral-normalized AF columns and bootstrap intervals for the ratio. This
  preserves raw AF while allowing Phase 2 shards to report ratios against the
  high-support `neutral_common_controls` baseline under the current
  opportunity contract.
- Full normalized `slop_lexicon` versus `neutral_common_controls` run
  `stage2-phase2-olmo3-sft-promptpkg512-slop-vs-neutral-common-normalized-cached-shared-branch2-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/hdwg40tc`, was stopped early
  because branch fanout made the full 512-row run inefficient under the current
  scorer.
- A bounded 128-document normalized shard completed:
  `stage2-phase2-olmo3-sft-promptpkg512-sample128-slop-vs-neutral-common-normalized-cached-shared-branch2-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/wgsqtjcq`. It scored 20,916
  opportunities in 1,128.8 seconds (18.5 opportunities/sec). Raw AF was 0.331
  for `neutral_common_controls` with CI 0.290-0.380 and 0.880 for
  `slop_lexicon` with CI 0.000-3.218. Neutral-normalized `slop_lexicon` AF
  was 2.656 with CI 0.000-9.620. This is directionally positive for the
  slop positive control after normalization, but the sample has only five
  `slop_lexicon` reference initiations, so it is not a stable grid result.
- The scorer benchmark CLI now supports `--mode`, allowing cached-only
  benchmarks without first running obsolete non-cached modes. This matters for
  slop/common-control shapes because the all-mode branch-4 benchmark
  (`https://wandb.ai/phulin-self/slop-stage1/runs/cqfl09k7`) OOMed in the
  non-cached batched path before reaching the production cached path.
- Cached-only OLMo SFT scorer microbenchmarks for `slop_lexicon` plus
  `neutral_common_controls` found branch size 8 to be the best tested setting:
  branch 2 (`https://wandb.ai/phulin-self/slop-stage1/runs/6g5evlht`) scored
  20.5 feature-opportunities/sec, branch 4
  (`https://wandb.ai/phulin-self/slop-stage1/runs/ytzn1c3b`) scored 25.4,
  branch 8 (`https://wandb.ai/phulin-self/slop-stage1/runs/nwtcj61b`) scored
  28.4, and branch 16
  (`https://wandb.ai/phulin-self/slop-stage1/runs/vstbnjwn`) OOMed on the
  cached path. Use `--cache-branch-batch-size 8` for the next narrow
  normalized stage-localization shard, while retaining branch 2 as the
  fallback if a different checkpoint has less memory headroom.
- Full 512-document SFT normalized shard completed with branch size 8:
  `stage2-phase2-olmo3-sft-promptpkg512-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/pgxems2i`. It scored 90,524
  opportunities in 3,269.9 seconds, for 27.7 opportunities/sec. The common
  controls again had raw AF 0.339 with CI 0.319-0.360. `slop_lexicon` had raw
  AF 0.673 with CI 0.414-1.225, and neutral-normalized AF 1.982 with CI
  1.226-3.663. This gives a stable SFT positive-control result under the
  neutral-normalized Phase 2 contract and is the baseline to compare against
  base, DPO, and final OLMo checkpoints.
- Full 512-document base-checkpoint normalized shard completed:
  `stage2-phase2-olmo3-base-promptpkg512-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/4bp5090m`. It scored 90,524
  opportunities in 3,273.0 seconds, for 27.7 opportunities/sec. The common
  controls had raw AF 0.272 with CI 0.252-0.293. `slop_lexicon` had raw AF
  0.484 with CI 0.274-0.957, and neutral-normalized AF 1.782 with CI
  1.005-3.519. Initial stage-localization read: slop-lexicon amplification
  relative to common controls is already present at base, with the SFT point
  estimate only modestly higher.
- Full 512-document DPO normalized shard completed:
  `stage2-phase2-olmo3-dpo-promptpkg512-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/bo2i2g1l`. It scored 90,524
  opportunities in 3,293.9 seconds, for 27.5 opportunities/sec. The common
  controls had raw AF 0.376 with CI 0.355-0.400. `slop_lexicon` had raw AF
  0.980 with CI 0.605-1.669, and neutral-normalized AF 2.605 with CI
  1.595-4.485. The current narrow stage grid is now base 1.782, SFT 1.982,
  DPO 2.605, supporting a preference-stage increase layered on top of inherited
  base propensity.
- Full 512-document final/RLVR normalized shard completed:
  `stage2-phase2-olmo3-final-promptpkg512-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/zc5lbdzo`. It scored 90,524
  opportunities in 3,274.3 seconds, for 27.6 opportunities/sec. The common
  controls had raw AF 0.455 with CI 0.429-0.481. `slop_lexicon` had raw AF
  0.981 with CI 0.605-1.669, and neutral-normalized AF 2.156 with CI
  1.327-3.717. The completed narrow OLMo stage-localization slice is base
  1.782, SFT 1.982, DPO 2.605, final 2.156. This supports a DPO-stage lift,
  with the final checkpoint still above base/SFT but below the DPO point
  estimate.
- Added `slop-assemble-phase2-grid` and ran the OLMo slop/common-control
  normalized grid assembly:
  `stage2-phase2-olmo3-slop-neutral-common-normalized-stage-grid-assembly`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/bjpt4jy3`. It wrote
  `artifacts/phase2/analysis/olmo3_slop_neutral_common_normalized_stage_grid.csv`,
  `artifacts/phase2/analysis/olmo3_slop_neutral_common_normalized_stage_comparison.csv`,
  and
  `artifacts/phase2/analysis/olmo3_slop_neutral_common_normalized_stage_grid_summary.md`.
  The primary comparison confirms DPO as the max normalized-AF stage.
- Completed the 512-prompt OLMo teacher-forced stock opener/closer grid. Runs:
  base `mj3zhxsv`, SFT `209dxyh6`, DPO `tomad94n`, final `cxtn5mdi`, assembly
  `f1vcmj2k` (`zhryxp9m` was superseded by the caveat-cleanup assembly). The
  pooled `stock_openers_closers` comparison uses raw AF because no stock
  neutral baseline was run: base `6.078`, SFT `5.214`, DPO `5.469`, final
  `5.929`. This is high raw amplification but not a post-training lift; base
  is the point-estimate max and final is close to base. The same denominator
  audit found zero `contrastive_negation` references in the 512 prompt package,
  so contrastive remains a measurement gap for this package rather than an AF
  result.
- Prepared the 5,000-prompt held-out package for scaling Phase 2:
  `stage2-phase2-dolci-sft-prompt-package-5000`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/77rz02r7`. It scanned 65,536
  Dolci SFT rows, retained 60,011 eligible prompt/target rows, filtered 5,524
  near-duplicate prompts, and selected 5,000 prompts. Local artifacts are
  `artifacts/phase2/prompts/olmo3_dolci_sft_phase2_prompt_package_5000.jsonl`,
  its manifest CSV, and its summary JSON.
- Ran the first target-shape DPO free-running benchmark on that package:
  `stage2-phase2-olmo3-dpo-promptpkg5000-free-running-64prompt-8comp-t1-bench1024`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/kji583m6`. Shape: 64 prompts,
  8 completions per prompt, temperature 1.0, top-p 0.95, max 1,024 new tokens,
  batch size 8, bfloat16, `torch.compile`. It completed 512 generations and
  524,288 generated tokens in 1,473.1 seconds, 355.9 generated tokens/sec
  including load and compile. Batch size 8 completed but was memory-tight on
  the 80GB A100. Feature rates per 1k generated tokens: rule-of-three 0.771,
  slop lexicon 0.269, contrastive negation 0.143, stock openers/closers 0.130.
- Added and ran the 5,000-prompt Phase 2 denominator-support measurement:
  `stage2-phase2-olmo3-promptpkg5000-denominator-support-v2`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/lq4yiy27`. The corrected
  summary distinguishes 5,000 unique measurement texts from 30,000
  feature-expanded measurement rows. Results: `slop_lexicon` has 917,669
  opportunities and 506 references; pooled `stock_openers_closers` has 45,404
  opportunities and 186 references; `neutral_common_controls` has 917,669
  opportunities and 66,671 references. `contrastive_negation` is nonzero but
  sparse with 111,662 opportunities and 7 references, including
  `not_x_but_y` and `not_x_its_y` subtypes. `rule_of_three_approx` was
  intentionally omitted from teacher-forced denominator support because it has
  no frozen Phase 2 opportunity spec yet.
- Started a DPO `contrastive_negation` teacher-forced diagnostic
  (`https://wandb.ai/phulin-self/slop-stage1/runs/1kz76tru`) and stopped it
  before meaningful GPU spend after the sidecar strategy audit flagged that 7
  references over the 5k package would make it a weak next run. Switched to
  `stock_openers_closers`, which had 186 references over 45,404 opportunities.
- Completed the first 5k teacher-forced OLMo run:
  `stage2-phase2-olmo3-dpo-promptpkg5000-stock-openers-closers-cached-shared-branch8-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/5c9cut6v`. It scored
  45,404 DPO opportunities in 1,245.8 seconds, 36.45 opportunities/sec, using
  bfloat16, `torch.compile`, shared prefix caching, and branch batch size 8.
  Result: raw AF `7.115`, CI `6.140-8.362`, against 186 reference
  initiations. This gives a well-supported single-checkpoint stock
  opener/closer result; base, SFT, and final remain to complete the 5k stock
  stage grid.
- Completed the matching 5k base checkpoint run:
  `stage2-phase2-olmo3-base-promptpkg5000-stock-openers-closers-cached-shared-branch8-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/tn7nd70e`. It scored the same
  45,404 opportunities in 1,190.3 seconds, 38.15 opportunities/sec. Result:
  raw AF `7.997`, CI `6.952-9.391`, against the same 186 reference
  initiations. The base point estimate is above DPO for this feature, so the
  5k stock grid should be treated as a calibration/control feature until SFT
  and final are measured.
- Completed the matching 5k SFT checkpoint run:
  `stage2-phase2-olmo3-sft-promptpkg5000-stock-openers-closers-cached-shared-branch8-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/7mk7lu7k`. It scored the same
  45,404 opportunities in 1,191.3 seconds, 38.11 opportunities/sec. Result:
  raw AF `6.738`, CI `5.837-7.857`, against the same 186 reference
  initiations. The 5k stock point-estimate order is now base > DPO > SFT, with
  all three strongly amplified; final remains to complete the stock stage grid.
- Completed the matching 5k final/RLVR checkpoint run:
  `stage2-phase2-olmo3-final-promptpkg5000-stock-openers-closers-cached-shared-branch8-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/238ph0bt`. It scored the same
  45,404 opportunities in 1,187.4 seconds, 38.24 opportunities/sec. Result:
  raw AF `7.715`, CI `6.644-9.056`, against the same 186 reference
  initiations. The completed 5k stock point-estimate order is base > final >
  DPO > SFT, with all four checkpoints strongly amplified; treat this as a
  high-support calibration/control feature, not as preference-stage-specific
  amplification evidence.
- Assembled the 5k stock opener/closer stage grid:
  `stage2-phase2-olmo3-promptpkg5000-stock-openers-closers-stage-grid-assembly`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/xl4b7ss5`. The raw-AF
  comparison artifacts are under `artifacts/phase2/analysis/` and confirm base
  as the maximum stage: base `7.997`, SFT `6.738`, DPO `7.115`, final `7.715`.
  Deltas versus base are SFT `-1.259`, DPO `-0.882`, and final `-0.282`.
- Measured a 1,024-prompt denominator slice for the primary
  `slop_lexicon`/`neutral_common_controls` teacher-forced shape:
  `stage2-phase2-olmo3-promptpkg1024-slop-neutral-denominator-support`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/gloz8jdd`. With
  `max_token_start_opportunities=128`, the slice has 88,903 opportunities per
  feature, 52 slop references, and 6,694 neutral references. A sidecar strategy
  audit recommended the full 5k DPO run as the eventual high-power check, but
  estimated it at roughly 18-20 A100-hours. Started a bounded 1,024-prompt DPO
  validation first to test whether the 512-prompt DPO slop lift survives a
  larger sample before spending overnight-scale compute.
- Completed that bounded 1,024-prompt DPO validation:
  `stage2-phase2-olmo3-dpo-promptpkg1024-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/tu89kg31`. It scored 177,806
  opportunities in 6,404.6 seconds, 27.76 opportunities/sec, on
  `allenai/Olmo-3-7B-Instruct-DPO`. Result: `slop_lexicon`
  neutral-normalized AF `1.999`, CI `1.446-2.837`; raw slop AF `0.751`, CI
  `0.546-1.058`; neutral raw AF `0.376`, CI `0.357-0.394`. This preserves a
  slop-vs-common-control excess but pulls the point estimate down from the
  earlier 512-prompt DPO normalized AF `2.605` toward the SFT-sized roughly
  2x effect. Before spending the estimated 18-20 A100-hours on the full 5k DPO
  slop/neutral cell, run the pushed planned-scorer benchmark/optimization path
  on the target model shape.
- Target-shape planned-scorer benchmark completed:
  `stage2-phase2-olmo3-dpo-scorer-planned-vs-dynamic-branch8-prefix256`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/wv1v44ku`. On
  `allenai/Olmo-3-7B-Instruct-DPO` with `slop_lexicon` plus
  `neutral_common_controls`, batch size 16, prefix 256, branch size 8, and
  bf16/compile, dynamic cached scoring reached 29.55 feature-opportunities/sec
  and the prebuilt-plan path reached 29.72. The 1.006x speedup confirms this
  patch reduces metadata churn but does not materially change the full 5k
  compute estimate; use it for correctness/cleanliness, not as a reason to
  launch the overnight-scale DPO cell before matching the 1,024-prompt SFT
  comparison.
- Completed the matching 1,024-prompt SFT slop/neutral run:
  `stage2-phase2-olmo3-sft-promptpkg1024-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/hoggmwz5`. It scored the
  same 177,806 opportunities in 6,110.7 seconds, 29.10 opportunities/sec, on
  `allenai/Olmo-3-7B-Instruct-SFT`. Result: `slop_lexicon`
  neutral-normalized AF `1.695`, CI `1.315-2.289`; raw slop AF `0.575`, CI
  `0.445-0.776`; neutral raw AF `0.339`, CI `0.322-0.357`. DPO remains higher
  by point estimate (`1.999`, ratio `1.18x`), but the CIs overlap heavily and
  the ratio is below the practical `1.25x` threshold for jumping straight to
  final or full 5k DPO. Next stage-localization cell: base 1,024.
- Completed the matching 1,024-prompt base slop/neutral run:
  `stage2-phase2-olmo3-base-promptpkg1024-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/4mn6ycwv`. It scored the same
  177,806 opportunities in 6,109.4 seconds, 29.10 opportunities/sec, on
  `allenai/Olmo-3-1025-7B`. Result: `slop_lexicon` neutral-normalized AF
  `1.467`, CI `1.114-2.046`; raw slop AF `0.396`, CI `0.302-0.552`; neutral
  raw AF `0.270`, CI `0.254-0.288`. The 1,024-point stage ordering is now DPO
  `1.999` > SFT `1.695` > base `1.467`, but all three intervals overlap. The
  conservative conclusion is inherited/base slop propensity plus modest
  post-training amplification, not a clean DPO-only effect. Keep the
  torch/Transformers scorer as the exact teacher-forced scorer of record;
  defer SGLang/vLLM to free-generation throughput work or a separate exactness
  benchmark that reproduces fixed-branch probability-mass summaries.
- Completed the matching 1,024-prompt final/RLVR slop/neutral run:
  `stage2-phase2-olmo3-final-promptpkg1024-slop-vs-neutral-common-normalized-cached-shared-branch8-sequence`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/5v7x3180`. It scored the
  same 177,806 opportunities in 6,109.9 seconds, 29.10 opportunities/sec, on
  `allenai/Olmo-3-7B-Instruct`. Result: `slop_lexicon`
  neutral-normalized AF `1.659`, CI `1.189-2.351`; raw slop AF `0.760`, CI
  `0.547-1.077`; neutral raw AF `0.458`, CI `0.439-0.478`. The denominators
  match base/SFT/DPO exactly: 88,903 opportunities per feature, 52 slop
  references, and 6,694 neutral references.
- Assembled the 1,024-prompt slop/neutral teacher-forced stage grid:
  `stage2-phase2-olmo3-promptpkg1024-slop-neutral-common-normalized-stage-grid-assembly`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/1cuuzoax`. The normalized-AF
  ordering is DPO `1.999` > SFT `1.695` > final `1.659` > base `1.467`, with
  all confidence intervals overlapping. DPO is the maximum point-estimate
  stage, but final/RLVR attenuates near SFT rather than preserving a large DPO
  lift. Treat the 1024 grid as inherited/base slop propensity with a modest
  DPO-stage point-estimate peak; do not spend the estimated full-5k
  slop/neutral compute unless there is a narrower confirmatory reason.
- Rejoined the completed 512-prompt free-running generation caches to the newer
  1024-prompt teacher-forced grid:
  `stage2-phase2-olmo3-generation-compounding-512prompt-tf1024`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/6miductd`. This supersedes
  the earlier 512-grid compounding join for current Result B interpretation.
  Observed `slop_lexicon` opportunity rates exceed teacher-forced expectation
  in every stage/temperature cell. Max realized AF is DPO at temperature 1.0
  (`1.506`), with final temperature 1.0 effectively tied (`1.504`); max
  absolute excess is SFT temperature 1.0 (`0.454` per 1k opportunities). This
  keeps the bounded compounding signal positive while reducing the older
  realized-AF magnitudes after the larger teacher-forced denominator update.
- Completed the bounded target-shape DPO generation shard:
  `stage2-phase2-olmo3-dpo-promptpkg5000-free-running-512prompt-8comp-t1-bench1024`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/8rud2kxl`. It generated 512
  prompts x 8 completions x 1,024 tokens, for 4,096 generations and 4,194,304
  generated tokens, in 11,758.0 seconds (`356.7` generated tokens/sec including
  load and compile). Feature rates per 1k generated tokens were
  `rule_of_three_approx` `0.861`, `slop_lexicon` `0.229`,
  `contrastive_negation` `0.141`, `stock_openers_closers` `0.108`,
  `stock_openers` `0.064`, and `stock_closers` `0.043`. Throughput matches the
  earlier 64-prompt target-shape benchmark, so scaling prompts did not degrade
  performance; the next optimization question is whether a serving-generation
  backend such as SGLang or vLLM can materially improve free-running generation
  throughput while preserving the same JSONL and feature-summary contracts.
- Hardened the vLLM sidecar benchmark script to better match the Torch
  free-running contract: revised six-feature defaults, prompt-token truncation,
  source-tagged summary rows, prompt token counts, and a unit test with a fake
  vLLM module. Retried the one-prompt OLMo DPO vLLM 0.11.1 cu128 smoke:
  `stage2-phase2-olmo3-dpo-vllm0111-cu128-1prompt-smoke`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/dibzdwyd`. It reached engine
  initialization and then made no generation progress before the 300-second
  timeout. No artifact was written and no GPU process remained. Treat vLLM as
  blocked on this host for OLMo-3 generation until a container, backend flag, or
  version change is validated; SGLang is now the next serving-backend candidate.
- Completed the matching final/RLVR target-shape generation shard:
  `stage2-phase2-olmo3-final-promptpkg5000-free-running-512prompt-8comp-t1-bench1024`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/b8xo8axn`. It generated 512
  prompts x 8 completions, 4,096 generations, and 4,187,136 generated tokens in
  11,758.7 seconds (`356.1` generated tokens/sec including load and compile).
  Feature rates per 1k generated tokens were `rule_of_three_approx` `0.802`,
  `slop_lexicon` `0.193`, `contrastive_negation` `0.142`,
  `stock_openers_closers` `0.091`, `stock_openers` `0.059`, and
  `stock_closers` `0.032`. Compared with the matching DPO target-shape shard,
  final/RLVR is lower on slop lexicon (`0.193` vs. `0.229`, ratio `0.84`),
  rule-of-three (`0.802` vs. `0.861`, ratio `0.93`), and pooled stock
  openers/closers (`0.091` vs. `0.108`, ratio `0.84`), while contrastive
  negation is effectively tied and slightly higher (`0.142` vs. `0.141`).
  This mirrors the teacher-forced read: DPO is the bounded peak and final/RLVR
  attenuates rather than amplifying further.
- Completed the matching SFT target-shape generation shard:
  `stage2-phase2-olmo3-sft-promptpkg5000-free-running-512prompt-8comp-t1-bench1024`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/il07fjn1`. It generated 512
  prompts x 8 completions, 4,096 generations, and 3,845,808 generated tokens
  in 10,701.3 seconds (`359.4` generated tokens/sec including load and
  compile). Feature rates per 1k generated tokens were
  `rule_of_three_approx` `0.488`, `slop_lexicon` `0.171`,
  `contrastive_negation` `0.041`, `stock_openers_closers` `0.072`,
  `stock_openers` `0.052`, and `stock_closers` `0.020`. Compared with DPO,
  SFT is lower on every tracked feature; DPO/SFT ratios are `1.77` for
  rule-of-three, `1.34` for slop lexicon, `3.45` for contrastive negation,
  and `1.50` for pooled stock openers/closers. Compared with final/RLVR, SFT
  is also lower on every tracked feature. This sharpens the bounded
  target-shape story: SFT is the low generation-emission point, DPO is the
  slop-lexicon peak, and final/RLVR attenuates from DPO while remaining above
  SFT for most generation features.
- Completed the matching base target-shape generation shard:
  `stage2-phase2-olmo3-base-promptpkg5000-free-running-512prompt-8comp-t1-bench1024`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/nw6yoj5u`. It generated 512
  prompts x 8 completions, 4,096 generations, and 4,174,328 generated tokens
  in 11,677.3 seconds (`357.5` generated tokens/sec including load and
  compile). Feature rates per 1k generated tokens were
  `rule_of_three_approx` `1.019`, `slop_lexicon` `0.233`,
  `contrastive_negation` `0.136`, `stock_openers_closers` `0.092`,
  `stock_openers` `0.064`, and `stock_closers` `0.028`.
- Assembled the four-stage target-shape generation grid:
  `stage2-phase2-olmo3-generation-target-shape-512prompt-8comp-t1-stage-grid`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/bojfccyx`. The assembled
  primary-feature comparison shows `slop_lexicon` rates of base `0.233`, SFT
  `0.171`, DPO `0.229`, and final/RLVR `0.193` per 1k generated tokens. Base
  is narrowly the slop-lexicon maximum and is clearly highest for
  `rule_of_three_approx`; DPO is highest for stock opener/closer features;
  final/RLVR attenuates from DPO on most features; SFT is the low-emission
  point across all tracked features. This completes the bounded
  base/SFT/DPO/final target-shape grid and weakens any simple monotonic
  post-training generation-amplification story.
- Ran the target-shape compounding join:
  `stage2-phase2-olmo3-generation-compounding-target-shape-512prompt-8comp-t1-tf1024`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/0klmyip9`. It joins the
  completed 512-prompt x 8-completion x 1,024-token generation caches against
  the 1,024-prompt teacher-forced slop/neutral grid. Observed `slop_lexicon`
  opportunity rates exceed teacher-forced expectation in all four stages:
  base `0.615` observed vs. `0.232` expected per 1k opportunities, SFT
  `0.697` vs. `0.336`, DPO `0.638` vs. `0.439`, and final/RLVR `0.582` vs.
  `0.445`. Max absolute excess is base (`0.383` per 1k opportunities); max
  realized AF is SFT (`1.192`); final/RLVR is near the teacher-forced
  reference-rate baseline (`0.994`). Repeat counts are now much denser than
  the 128-token temperature sweep: base `192`, SFT `145`, DPO `199`, final
  `180` repeat generations for `slop_lexicon`.
- Started the bounded target-shape temperature-dependence expansion with DPO
  at temperature 0.7:
  `stage2-phase2-olmo3-dpo-promptpkg5000-free-running-512prompt-8comp-t07-bench1024`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/8no9vqyf`. It generated 512
  prompts x 8 completions, 4,096 generations, and 4,188,320 generated tokens
  in 11,702.7 seconds (`357.9` generated tokens/sec including load and
  compile). Compared with the matching DPO temperature 1.0 shard, `slop_lexicon`
  is nearly unchanged (`0.223` vs. `0.229` per 1k generated tokens) and pooled
  stock openers/closers are effectively tied (`0.107` vs. `0.108`), while
  `rule_of_three_approx` is slightly higher at 0.7 (`0.888` vs. `0.861`).
- Completed the matching DPO target-shape temperature 0.0 shard:
  `stage2-phase2-olmo3-dpo-promptpkg5000-free-running-512prompt-8comp-t0-bench1024`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/8s6spxu5`. It generated 512
  prompts x 8 completions, 4,096 generations, and 4,194,176 generated tokens.
  Feature rates per 1k generated tokens were `contrastive_negation` `0.168`,
  `rule_of_three_approx` `0.864`, `slop_lexicon` `0.206`,
  `stock_closers` `0.044`, `stock_openers` `0.057`, and
  `stock_openers_closers` `0.101`.
- Assembled the DPO target-shape temperature grid:
  `stage2-phase2-olmo3-dpo-generation-target-shape-temperature-grid`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/5azmz6pq`. Across
  temperatures 0.0/0.7/1.0, `slop_lexicon` is `0.206`/`0.223`/`0.229` per
  1k generated tokens, pooled stock openers/closers are
  `0.101`/`0.107`/`0.108`, `rule_of_three_approx` is
  `0.864`/`0.888`/`0.861`, and `contrastive_negation` is
  `0.168`/`0.127`/`0.141`. The temperature dependence is modest: warmer
  decoding mildly increases slop lexicon and pooled stock phrase rates, but
  deterministic decoding does not remove the features and is highest for
  contrastive negation.
- Ran the matching DPO target-shape compounding temperature join:
  `stage2-phase2-olmo3-dpo-generation-compounding-target-shape-temperature-tf1024-v2`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/k6gcu75b`. It joins the three
  DPO target-shape generation caches against the 1,024-prompt teacher-forced
  slop/neutral grid. `slop_lexicon` observed rates exceed teacher-forced
  expectation at all temperatures: t=0.0 observed `0.615` vs expected `0.439`
  per 1k opportunities, realized AF `1.051`; t=0.7 observed `0.637` vs
  expected `0.439`, realized AF `1.089`; t=1.0 observed `0.638` vs expected
  `0.439`, realized AF `1.091`. The direct prior-window effect remains large
  at every temperature, with repeat generations t=0.0 `160`, t=0.7 `204`, and
  t=1.0 `199`.
- Implemented and tested the SGLang free-running benchmark harness
  (`scripts/benchmark_sglang_generation.py`) in an isolated `uv` sidecar. The
  working host stack is SGLang 0.5.2, Torch 2.8.0+cu128, Transformers 4.57.6,
  CUDA toolkit/nvcc 12.8, `libnuma1`, `ninja-build`, and GCC/G++ 14 pinned via
  `CC`, `CXX`, and `CUDAHOSTCXX`. The initial setup failures were CUDA 13
  wheels, missing `[srt]` dependencies, old Transformers without OLMo-3,
  missing `libnuma`, missing nvcc, and CUDA 12.8 rejecting default GCC 15.
- Completed SGLang OLMo-3 DPO W&B benchmarks. The clean one-prompt smoke
  `29i0m563` proved load/generate/W&B artifact sync. The 64-prompt default
  stop benchmark `hkdjxvmm` generated only 1,117 tokens because SGLang stops
  on `<|im_end|>` (`100265`) by default. The 64-prompt `--ignore-eos`
  benchmark `qija648z` generated the full 8,192 tokens and reached 1,969
  decode tokens/sec, or 257.9 wall tokens/sec including load and graph capture.
  Treat SGLang as a promising fixed-length generation backend candidate, but
  settle the stop-token contract against Torch before using it for science
  shards.
- Ran the paired first-64 stop-token contract check. SGLang with
  `--sampling-strategy first --ignore-eos` logged W&B run `pqctgupb`; the
  matching Torch/Transformers CLI run logged `b0zz1hzo`. The prompt IDs and
  order matched exactly, and both backends generated 8,192 tokens. SGLang was
  much faster on this small wall-clock check (`271.9` tokens/sec vs. Torch
  `58.8` tokens/sec), with decode-only throughput `1,921` tokens/sec. Feature
  counts are not exact-match comparable because backend sampling RNG differs,
  so the next step is a controlled target-shape SGLang pilot and aggregate-rate
  comparison against the existing Torch DPO temp-1 cache.
- Completed the controlled target-shape SGLang DPO pilot after fixing the
  harness to handle flattened multi-completion SGLang responses:
  `stage2-phase2-olmo3-dpo-sglang052-cu128-512prompt-8comp-t1-ignoreeos-ctx4096-pilot-v2`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/v69n4twp`. It generated the
  full 512 prompts x 8 completions x 1,024 tokens: 4,096 generations and
  4,194,304 tokens. Throughput was excellent (`2,584` decode tokens/sec,
  `2,535` wall tokens/sec), about 7.1x the matching Torch target-shape DPO
  shard. The science contract fails, though: SGLang `--ignore-eos` rates are
  materially higher than Torch for `slop_lexicon` (`0.402` vs. `0.229` per
  1k tokens), `rule_of_three_approx` (`1.588` vs. `0.861`), and pooled stock
  openers/closers (`0.440` vs. `0.108`). Keep Torch as the Phase 2 science
  backend unless SGLang can be configured to match the Torch stop-token
  contract without `ignore_eos` changing the distribution.
- Added the bounded amplification-spectrum assembler and ran it over the
  current OLMo/Dolci Phase 1 and Phase 2 artifacts:
  `stage2-phase2-olmo3-amplification-spectrum-bounded`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/v31wiggh`. It writes
  `artifacts/phase2/analysis/olmo3_amplification_spectrum_bounded.csv` and
  `artifacts/phase2/analysis/olmo3_amplification_spectrum_bounded_summary.md`,
  with 24 feature-stage rows covering retained feature data rates, the
  four-stage `slop_lexicon` teacher-forced grid, target-shape free-running
  rates, target-shape compounding, and denominator-support notes. The assembled
  spectrum records the current caveats explicitly: `rule_of_three_approx`
  remains free-running-only until its teacher-forced initiation contract is
  frozen, `contrastive_negation` is sparse in the 5k denominator package, and
  SGLang generations are excluded from science results until the stop-token
  contract matches Torch.
- Froze a pilot teacher-forced contract for `rule_of_three_approx` as a
  comma-pair extension proxy. Opportunities are the slot after a candidate
  two-item comma-separated list (`X, Y <slot>`); the scorer sums exact sequence
  mass over `, and`, `, or`, `and`, and `or`; reference positives align to the
  coordinator inside the existing `RULE_OF_THREE_PATTERN` span. This measures
  whether the model extends an established two-item list into a third
  coordinated item, not open-vocabulary first-item initiation.
- Ran CPU denominator-support pilots for the new `rule_of_three_approx`
  contract on the 5,000-prompt Dolci SFT Phase 2 package. The initial unfiltered
  pilot logged W&B run `9879iqjj` with 4,424 opportunities, 1,832 references,
  and reference rate `0.414`. After adding a conservative code/data-line filter,
  pilot v2 logged W&B run `99kvrho6` with 3,281 opportunities, 1,473 references,
  and reference rate `0.449`. The filtered support is large enough to justify a
  small GPU teacher-forced pilot, but sampled negatives still include some
  bibliography/list-heading cases, so treat this as a proxy contract until
  precision is sampled.
- Ran the first GPU teacher-forced pilot for the new rule-of-three completion
  contract on `allenai/Olmo-3-7B-Instruct-SFT`:
  `stage2-phase2-olmo3-sft-rule-of-three-completion-promptpkg512`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/yi498mc3`. It used 512 prompts
  from the 5,000-prompt package, bfloat16, `torch.compile`, fixed prefix 256,
  exact sequence mass, and branch batch size 4. The run scored 360
  opportunities in 76.9 seconds (`4.68` opportunities/sec after load/compile),
  with 155 references, reference rate `0.431`, mean model probability mass
  `0.348`, and raw AF `0.808` with bootstrap CI `0.745`-`0.865`. This is a
  useful SFT pilot but not yet a stage-localization grid.
- Completed the matching 512-prompt rule-of-three completion teacher-forced
  mini-grid. Base W&B run `8mggip2w` scored raw AF `0.775`
  (`0.710`-`0.829`); SFT run `yi498mc3` scored `0.808`
  (`0.745`-`0.865`); DPO run `e72tp5np` scored `0.761`
  (`0.687`-`0.821`); final/RLVR run `7n60plb0` scored `0.768`
  (`0.693`-`0.831`). All four runs used the same 512-prompt sample, 360
  opportunities, and 155 references. The pilot proxy does not show
  DPO-stage amplification; SFT has the highest point estimate, and all stages
  are below the reference extension rate.
- Assembled the rule-of-three completion teacher-forced mini-grid as a first
  class Phase 2 grid:
  `stage2-phase2-olmo3-rule-of-three-completion-stage-grid`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/bi1jm0qf`. Outputs:
  `artifacts/phase2/analysis/olmo3_promptpkg512_rule_of_three_completion_stage_grid.csv`,
  `artifacts/phase2/analysis/olmo3_promptpkg512_rule_of_three_completion_stage_comparison.csv`,
  and
  `artifacts/phase2/analysis/olmo3_promptpkg512_rule_of_three_completion_stage_grid_summary.md`.
- Rebuilt the bounded amplification spectrum with both teacher-forced grids:
  the 1,024-prompt `slop_lexicon`/`neutral_common_controls` grid and the
  512-prompt `rule_of_three_approx` comma-pair extension proxy grid. W&B run:
  `stage2-phase2-olmo3-amplification-spectrum-bounded-v2`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/jkie1xy6`. It still has 24
  feature-stage rows, but teacher-forced coverage increases from 4 cells to 8
  cells. Outputs:
  `artifacts/phase2/analysis/olmo3_amplification_spectrum_bounded_v2.csv` and
  `artifacts/phase2/analysis/olmo3_amplification_spectrum_bounded_v2_summary.md`.
- Scaled the `rule_of_three_approx` comma-pair extension teacher-forced proxy
  to the full 5,000-prompt held-out package. The four OLMo stages all scored
  3,281 opportunities and 1,473 references with the same filtered denominator
  support. W&B runs: base `hd0k5hfn`, SFT `ctpb9eu6`, DPO `l8indgao`,
  final/RLVR `9b60jls5`. Raw AFs were base `0.721` (`0.698`-`0.742`), SFT
  `0.767` (`0.743`-`0.787`), DPO `0.716` (`0.691`-`0.738`), and final/RLVR
  `0.723` (`0.698`-`0.746`). Throughput for the full runs was about 24
  opportunities/sec after model load and compile. This full proxy preserves the
  512-prompt read: SFT has the highest point estimate, DPO is slightly below
  base, and the proxy does not support a DPO-stage rule-of-three amplification
  claim.
- Assembled the full 5,000-prompt rule-of-three completion grid:
  `stage2-phase2-olmo3-rule-of-three-completion-promptpkg5000-stage-grid`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/sod6yjyj`. Outputs:
  `artifacts/phase2/analysis/olmo3_promptpkg5000_rule_of_three_completion_stage_grid.csv`,
  `artifacts/phase2/analysis/olmo3_promptpkg5000_rule_of_three_completion_stage_comparison.csv`,
  and
  `artifacts/phase2/analysis/olmo3_promptpkg5000_rule_of_three_completion_stage_grid_summary.md`.
- Rebuilt the bounded amplification spectrum as v3 with the 1,024-prompt
  slop/neutral grid, the full 5,000-prompt rule-of-three proxy grid, the
  5,000-prompt pooled stock opener/closer grid, target-shape generation rates,
  compounding, and denominator support. W&B run:
  `stage2-phase2-olmo3-amplification-spectrum-bounded-v3`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/4agew61j`. It still has 24
  feature-stage rows and now has 12 teacher-forced cells. Outputs:
  `artifacts/phase2/analysis/olmo3_amplification_spectrum_bounded_v3.csv` and
  `artifacts/phase2/analysis/olmo3_amplification_spectrum_bounded_v3_summary.md`.
- Ran a separate 5,000-prompt `stock_openers` teacher-forced grid to split the
  pooled stock opener/closer signal. W&B runs: base `ul26mrrc`, SFT
  `140fbygd`, DPO `47jq4d4v`, final/RLVR `8j59fpx5`. All four stages scored
  the same 5,000 document-start opportunities and 168 references. Raw AFs were
  base `0.067` (`0.059`-`0.078`), SFT `0.048` (`0.042`-`0.056`), DPO `0.035`
  (`0.030`-`0.041`), and final/RLVR `0.035` (`0.030`-`0.041`). This shows that
  document-start stock openers are not responsible for the high pooled
  `stock_openers_closers` teacher-forced AF; the remaining candidate is the
  sparse closer side or the pooled denominator interaction.
- Assembled the `stock_openers` stage grid:
  `stage2-phase2-olmo3-stock-openers-promptpkg5000-stage-grid`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/echjq5b8`. Outputs:
  `artifacts/phase2/analysis/olmo3_promptpkg5000_stock_openers_stage_grid.csv`,
  `artifacts/phase2/analysis/olmo3_promptpkg5000_stock_openers_stage_comparison.csv`,
  and
  `artifacts/phase2/analysis/olmo3_promptpkg5000_stock_openers_stage_grid_summary.md`.
- Rebuilt the bounded amplification spectrum as v4 with the separate
  `stock_openers` teacher-forced grid. W&B run:
  `stage2-phase2-olmo3-amplification-spectrum-bounded-v4`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/rg4mattp`. It still has 24
  feature-stage rows and now has 16 teacher-forced cells. Outputs:
  `artifacts/phase2/analysis/olmo3_amplification_spectrum_bounded_v4.csv` and
  `artifacts/phase2/analysis/olmo3_amplification_spectrum_bounded_v4_summary.md`.
- Ran the matching 5,000-prompt `stock_closers` teacher-forced grid to resolve
  the pooled stock opener/closer signal. W&B runs: base `nlpxvpof`, SFT
  `jcp2nar5`, DPO `dcml121g`, final/RLVR `y06wdbfa`. All four stages scored
  the same 41,187 final-clause opportunities and 18 references. Raw AFs were
  base `73.304` (`47.983`-`132.664`), SFT `65.076`
  (`42.501`-`117.634`), DPO `68.634` (`45.192`-`123.947`), and final/RLVR
  `74.032` (`48.951`-`133.005`). Mean probability mass stayed near 3% across
  stages (`0.0284`-`0.0324`) despite a reference rate of only `0.000437`.
  This confirms that the high pooled stock AF is coming from the closer side,
  not from document-start openers, but the absolute AF magnitude is
  denominator-sensitive because the reference count is sparse.
- Assembled the `stock_closers` stage grid:
  `stage2-phase2-olmo3-stock-closers-promptpkg5000-stage-grid`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/5j8q9u59`. Outputs:
  `artifacts/phase2/analysis/olmo3_promptpkg5000_stock_closers_stage_grid.csv`,
  `artifacts/phase2/analysis/olmo3_promptpkg5000_stock_closers_stage_comparison.csv`,
  and
  `artifacts/phase2/analysis/olmo3_promptpkg5000_stock_closers_stage_grid_summary.md`.
- Rebuilt the bounded amplification spectrum as v5 with the separate
  `stock_closers` teacher-forced grid. W&B run:
  `stage2-phase2-olmo3-amplification-spectrum-bounded-v5`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/87y3gz6m`. It still has 24
  feature-stage rows and now has 20 teacher-forced cells. Outputs:
  `artifacts/phase2/analysis/olmo3_amplification_spectrum_bounded_v5.csv` and
  `artifacts/phase2/analysis/olmo3_amplification_spectrum_bounded_v5_summary.md`.
- Made the artifact-manifest CLI stage-aware via `--stage-tag` so Phase 2
  manifests no longer log with a hard-coded `stage1` tag. Logged the current
  OLMo Phase 2 headline artifact manifest as
  `stage2-phase2-olmo3-headline-artifact-manifest`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/c7pc9mz2`. The local
  gitignored manifest files are:
  `artifacts/phase2/analysis/olmo3_phase2_headline_artifact_manifest.csv`,
  `artifacts/phase2/analysis/olmo3_phase2_headline_artifact_manifest.json`, and
  `artifacts/phase2/analysis/olmo3_phase2_headline_artifact_manifest.md`.
- Implemented reference-subset summaries in `slop-teacher-forced-propensity`.
  The new `--reference-subset NAME:FIELD=VALUE` / `FIELD!=VALUE` option and
  `--reference-subset-summary-output` sidecar write additional AF summaries for
  named metadata subsets while preserving the existing all-reference summary
  schema and opportunity CSV. W&B logs the sidecar as
  `propensity_reference_subset_summary` without raw text. Focused verification:
  `uv run pytest -q tests/test_teacher_forced_propensity.py` (`11 passed`),
  `uv run pytest -q tests/test_teacher_forced_propensity.py
  tests/test_assemble_phase2_grid.py tests/test_imports.py` (`15 passed`), and
  `uv run ruff check src/slop_sftdiv/cli/teacher_forced_propensity.py
  tests/test_teacher_forced_propensity.py` (`All checks passed`). The current
  5,000-row Dolci prompt package exposes `provenance`, `source_dataset`, and
  `stratum`, but not an explicit `human_written` flag, so the implementation
  closes the scorer gap while the human-written-SFT science claim still depends
  on a provenance mapping or regenerated package.
- Added prompt-package metadata bucket materialization via
  `slop-prepare-phase2-prompts --metadata-bucket-map PATH`. The JSON map schema
  is `output_field`, `source_field` or `source_fields`, `values`, and
  `default`; mapped fields are written into both the prompt JSONL and manifest
  CSV. Checked in `configs/phase2_dolci_reference_subset_seed_map.json` as a
  conservative seed map for `reference_subset` buckets (`code`,
  `synthetic_llm`, `unknown`). A sidecar provenance audit found that the
  current Dolci package's `source_dataset` and `provenance` fields are identical
  source-mixture labels, not independent target-response authorship evidence.
  The seed map therefore assigns only code-domain labels plus `WildGuardMix`
  (`synthetic_llm`); it intentionally does not assign a `human_reference`
  bucket. Smoke verification on the existing 128-row Phase 2 package wrote a
  16-row gitignored package with bucket counts `code=4`, `unknown=12`.
  Focused verification: `uv run pytest -q tests/test_prepare_phase2_prompts.py
  tests/test_teacher_forced_propensity.py tests/test_imports.py` (`16 passed`)
  and `uv run ruff check src/slop_sftdiv/cli/prepare_phase2_prompts.py
  tests/test_prepare_phase2_prompts.py
  src/slop_sftdiv/cli/teacher_forced_propensity.py
  tests/test_teacher_forced_propensity.py` (`All checks passed`).
- Added `slop-annotate-phase2-prompts` for row-preserving annotation of
  existing prompt packages with the same metadata bucket-map schema. This avoids
  accidental resampling or duplicate-ID filtering when adding fields to already
  frozen Phase 2 packages. Ran the full 5,000-row annotation with W&B online as
  `stage2-phase2-olmo3-dolci-sft-promptpkg5000-reference-subset`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/ew7x38n8`. Outputs are
  gitignored under `artifacts/phase2/prompts/`:
  `olmo3_dolci_sft_phase2_prompt_package_5000_reference_subset.jsonl`,
  `_manifest.csv`, and `_summary.json`. The annotated package preserves all
  5,000 rows and has bucket counts `code=1371`, `synthetic_llm=225`, and
  `unknown=3404`.
- Added `slop-summarize-propensity-subsets`, an offline summarizer that joins
  existing `slop-teacher-forced-propensity` opportunity CSVs to annotated prompt
  metadata and emits subset AF tables with bootstrap CIs. This lets Phase 2
  subgroup checks reuse already-scored logits rather than rerunning the model.
  Ran it on the four-stage 1,024-prompt `slop_lexicon` /
  `neutral_common_controls` opportunity grid using the annotated 5,000-row
  package. W&B run:
  `stage2-phase2-olmo3-slop-neutral-reference-subset-summary`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/316hbxbl`. It joined 711,224
  opportunity rows with zero missing metadata and wrote 24 subset-stage-feature
  rows to gitignored outputs:
  `artifacts/phase2/analysis/olmo3_promptpkg1024_slop_neutral_reference_subset_summary.csv`
  and `_summary.md`. `slop_lexicon` neutral-normalized AF by subset/stage:
  `code` base/SFT/DPO/final `1.643`/`2.078`/`1.699`/`1.301`,
  `synthetic_llm` `1.189`/`1.518`/`2.033`/`1.612`, and
  `unknown` `1.636`/`1.867`/`2.326`/`1.895`. The `code` subset has only 5 slop
  references in this denominator and should be treated as diagnostic.
- Ran the same reference-subset summarizer over the remaining full 5,000-prompt
  teacher-forced grids:
  `rule_of_three_approx` W&B `0nklcliy`,
  `stock_openers` W&B `jju0y0n5`,
  `stock_closers` W&B `zxu196ge`, and
  pooled `stock_openers_closers` W&B `ssa70ap1`. All runs joined with zero
  missing metadata. The subset read preserves the all-reference conclusions:
  rule-of-three completion is SFT-high and DPO-low across `code`,
  `synthetic_llm`, and `unknown`; stock openers are below reference rate in
  every subset and decline after base; stock closers drive the large pooled
  stock AF but have sparse subset references (`code=2`, `synthetic_llm=0`,
  `unknown=16`). Pooled `stock_openers_closers` in `unknown` remains closer-side
  dominated with raw AF base/SFT/DPO/final
  `18.862`/`15.407`/`17.060`/`18.507`, while `code` is much lower
  (`2.228`/`2.065`/`1.735`/`1.867`) because opener references dominate.
- Added `slop-plan-phase2-generation`, a row/checklist planner that emits
  resumable `slop-free-running-emission` commands, expected generation/token
  counts, local completion status, and A100-hour estimates. W&B run
  `stage2-phase2-olmo3-generation-plan-512prompt-t1`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/eu65803t`) verified the
  existing 512-prompt t=1.0 target-shape grid as 4/4 complete. W&B run
  `stage2-phase2-olmo3-generation-plan-full-5000prompt-8comp-3temp`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/3d8g6at4`) planned the full
  EXPERIMENTS.md OLMo generation grid as 12 missing shards, 40,000 generations
  per shard, and `383.52` estimated A100-hours at 356 generated tokens/sec.
  This argues against launching the full grid blindly on the current Torch
  backend.
- Added `slop-launch-phase2-generation-shard`, a guarded launcher for one row
  from a generation plan. It prints by default, requires `--execute` to launch,
  skips completed shards unless `--allow-completed` is set, and refuses shards
  over `--max-estimated-a100-hours` unless `--force` is passed. Validation:
  the full 5,000-prompt DPO t=1.0 shard was refused under the default 8-hour
  cap because it is estimated at `31.96` A100-hours, while the completed
  512-prompt DPO t=1.0 shard was selectable with `--allow-completed` and printed
  the matching generation command.
- Extended the launcher with explicit `--detach` support and added
  `slop-phase2-generation-status`. Detached launches write a selection JSON
  containing PID, log path, expected outputs, and command; the status command
  checks PID liveness and output completion. Validation on the completed
  bounded DPO t=1.0 selection reported `alive=False`, `completed=True`, and
  `4096/4096` generations.
- Started the first scaled post-planner target-shape Phase 2 generation shard:
  DPO, 1,024 prompts from the annotated 5,000-row Dolci SFT package, 8
  completions per prompt, temperature `1.0`, top-p `0.95`, and 1,024 max new
  tokens. The guarded planner estimated `6.55` A100-hours for 8,192 expected
  generations, under the 8-hour cap, but that estimate assumed the previously
  benchmarked throughput and did not capture compile-time memory cliffs after
  the host reset. The first detached attempt reached W&B run `xbd6yk7j` but
  exited at `0/8192` generations without writing generation outputs. A reset
  diagnostic with explicit CUDA wheel `LD_LIBRARY_PATH` completed successfully
  as W&B run `zacdzzwn`, producing 1 generation and 6 feature-rate rows.
  Relaunches with explicit `LD_LIBRARY_PATH` and batch sizes `1024`, `512`,
  `256`, and `64` each reached W&B but failed at `0/8192` generations with
  CUDA OOM during compiled generation. The successful live launch candidate is
  batch size `16`, planned as W&B run `eahhgpae` and launched as W&B run
  `p4v56vda`:
  `https://wandb.ai/phulin-self/slop-stage1/runs/p4v56vda`. Selection:
  `artifacts/phase2/analysis/olmo3_generation_launch_selection_dpo_1024prompt_8comp_t1_batched16.json`.
  Log:
  `artifacts/phase2/analysis/olmo3_generation_launch_selection_dpo_1024prompt_8comp_t1_batched16.log`.
  Status at 2026-06-14 22:03 UTC: alive, not complete, `0/8192` generations
  written yet, Python worker PID `5950`, and A100 utilization `96%` at about
  39.8 GiB. The process inherited the explicit CUDA wheel `LD_LIBRARY_PATH`;
  output rows are expected only after the first batch flush.
- Fixed the free-running generation cache writer for future long shards:
  completed generation batches are now appended to the JSONL cache immediately
  instead of waiting until the entire run finishes. This lets
  `slop-phase2-generation-status` observe partial progress and preserves
  completed batches if a later batch crashes. Focused verification:
  `uv run pytest -q tests/test_free_running_emission.py
  tests/test_phase2_generation_status.py` (`4 passed`) and
  `uv run ruff check src/slop_sftdiv/cli/free_running_emission.py
  tests/test_free_running_emission.py` (`All checks passed`). This code change
  does not affect the already-running `p4v56vda` process, which was launched
  before the patch.
- Improved `slop-phase2-generation-status` for inherited pre-streaming detached
  runs by parsing tqdm prompt progress from the launch log and estimating
  completed generations from the recorded `--completions-per-prompt` command
  argument. This makes the already-running DPO 1,024-prompt shard monitorable
  even though its JSONL cache will not flush until process completion. Focused
  verification: `uv run pytest -q tests/test_phase2_generation_status.py`
  (`2 passed`) and `uv run ruff check
  src/slop_sftdiv/cli/phase2_generation_status.py
  tests/test_phase2_generation_status.py` (`All checks passed`). Status at
  2026-06-14 22:37 UTC: W&B run `p4v56vda` alive, A100 utilization about 96%,
  worker PID `5950`, `0/8192` JSONL rows written, and latest log-derived
  progress `128` prompts (`~1024` generation completions at 8 completions per
  prompt). The next Phase 2 step after this shard completes is analysis first:
  compare its DPO feature rates against the existing 512-prompt target-shape
  DPO cell and run the matching DPO compounding join against the existing
  1,024-prompt teacher-forced `slop_lexicon` grid before launching matched
  base/SFT/final 1,024-prompt shards.
- Added `docs/experiments/phase2_post_shard_analysis.md` with the exact
  post-completion runbook for the active DPO 1,024-prompt shard: completion
  checks, a DPO 512-vs-1024 target-shape generation scale comparison using
  `slop-assemble-phase2-generation-grid`, a DPO-only 1,024-prompt compounding
  join using `slop-analyze-phase2-compounding` against
  `olmo3_promptpkg1024_slop_neutral_common_normalized_stage_grid.csv`, and the
  decision rule to analyze before launching matched base/SFT/final shards.
- Added `slop-run-phase2-post-shard-analysis`, a CPU-side post-shard runner
  that waits for a detached generation selection to reach completion, then
  calls the existing generation-grid assembler and compounding analyzer with
  the DPO 512-vs-1024 and 1,024-prompt DPO Result B paths documented in the
  runbook. The command is explicit: without `--execute` it only checks
  readiness and prints planned outputs. Focused verification:
  `uv run pytest -q tests/test_run_phase2_post_shard_analysis.py
  tests/test_phase2_generation_status.py` (`5 passed`) and
  `uv run ruff check src/slop_sftdiv/cli/run_phase2_post_shard_analysis.py
  tests/test_run_phase2_post_shard_analysis.py pyproject.toml` (`All checks
  passed`).
- Launched the post-shard runner as a detached CPU-side watcher with
  `setsid -f` after the plain `nohup ... &` attempt did not survive the shell
  session. Watcher command:
  `uv run slop-run-phase2-post-shard-analysis --selection
  artifacts/phase2/analysis/olmo3_generation_launch_selection_dpo_1024prompt_8comp_t1_batched16.json
  --wait --poll-seconds 300 --timeout-seconds 43200 --execute --wandb-mode
  online`. Watcher Python PID: `13708`; PID file:
  `artifacts/phase2/analysis/olmo3_dpo_post_shard_analysis_watcher.pid`; log:
  `artifacts/phase2/analysis/olmo3_dpo_post_shard_analysis_watcher.log`.
  Status at 2026-06-14 22:47 UTC: generation shard alive, A100 utilization
  about 94-95%, `0/8192` JSONL rows written, and latest log-derived progress
  `160` prompts (`~1280` generation completions at 8 completions per prompt).
- Patched `slop-run-phase2-post-shard-analysis` to flush wait-progress lines
  so redirected watcher logs are useful, verified
  `tests/test_run_phase2_post_shard_analysis.py` (`3 passed`) and ruff, then
  restarted only the CPU-side watcher. New watcher Python PID: `14233`; PID
  file still `artifacts/phase2/analysis/olmo3_dpo_post_shard_analysis_watcher.pid`.
  The watcher log now shows `waiting for shard completion: 0/8192 rows`.
  Status at 2026-06-14 22:50 UTC: generation shard alive, A100 utilization
  about 97%, `0/8192` JSONL rows written, and latest log-derived progress
  `176` prompts (`~1408` generation completions at 8 completions per prompt).
- Extended the post-shard watcher to parse the generation launch log just like
  `slop-phase2-generation-status`, so wait-progress lines now include
  `log_prompts` and estimated generation completions even for this
  pre-streaming run. Focused verification:
  `uv run pytest -q tests/test_run_phase2_post_shard_analysis.py
  tests/test_phase2_generation_status.py` (`5 passed`) and
  `uv run ruff check src/slop_sftdiv/cli/run_phase2_post_shard_analysis.py
  tests/test_run_phase2_post_shard_analysis.py` (`All checks passed`).
  Restarted only the CPU-side watcher. New watcher Python PID: `14748`.
  First flushed wait line:
  `waiting for shard completion: 0/8192 rows; log_prompts=176;
  log_generation_estimate=1408`.
- Restarted the CPU-side post-shard watcher with `--poll-seconds 60` so the
  analysis begins within about one minute of generation completion instead of
  waiting up to five minutes. New watcher Python PID: `15117`; PID file:
  `artifacts/phase2/analysis/olmo3_dpo_post_shard_analysis_watcher.pid`.
  First wait line after restart:
  `waiting for shard completion: 0/8192 rows; log_prompts=192;
  log_generation_estimate=1536`.
- Added `scripts/phase2_cuda_env.sh` as a sourceable recovery helper for shell
  or host resets. It reconstructs `LD_LIBRARY_PATH` from the installed NVIDIA
  CUDA 12.8 wheel libraries in `.venv`, sets `UV_CACHE_DIR` to `.uv-cache` if
  unset, and preserves `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  Updated `docs/experiments/phase2_post_shard_analysis.md` to use the helper
  instead of static machine-specific `LD_LIBRARY_PATH` examples. Verified with
  `bash -n scripts/phase2_cuda_env.sh` and a live
  `uv run slop-phase2-generation-status` invocation after sourcing the helper.
  Status at 2026-06-14 23:16 UTC: generation shard alive, A100 utilization
  about 95%, `0/8192` JSONL rows written, and latest log-derived progress
  `272` prompts (`~2176` generation completions at 8 completions per prompt).
- Extended `slop-phase2-generation-status` and
  `slop-run-phase2-post-shard-analysis` wait logs to parse tqdm elapsed time
  and seconds-per-prompt, then report ETA fields (`latest_log_elapsed_seconds`,
  `latest_log_seconds_per_prompt`, `eta_seconds`, `eta_hms`). Focused
  verification: `uv run pytest -q tests/test_phase2_generation_status.py
  tests/test_run_phase2_post_shard_analysis.py` (`5 passed`) and
  `uv run ruff check src/slop_sftdiv/cli/phase2_generation_status.py
  src/slop_sftdiv/cli/run_phase2_post_shard_analysis.py
  tests/test_phase2_generation_status.py
  tests/test_run_phase2_post_shard_analysis.py` (`All checks passed`). Live
  status after sourcing `scripts/phase2_cuda_env.sh`: generation shard alive,
  `0/8192` JSONL rows written, `288` log prompts (`~2304` generation
  completions), latest tqdm rate `15.93` seconds/prompt, ETA `3:15:24`.
- Restarted only the CPU-side post-shard watcher so the live redirected log
  uses the new ETA-aware wait message. Generation worker PID `5950` was left
  untouched. New watcher Python PID: `21450`; PID file:
  `artifacts/phase2/analysis/olmo3_dpo_post_shard_analysis_watcher.pid`.
  First ETA-aware wait line:
  `waiting for shard completion: 0/8192 rows; log_prompts=288;
  log_generation_estimate=2304; eta=3:15:24`.
- Added average-rate ETA fields to the Phase 2 status and watcher code:
  `latest_log_avg_seconds_per_prompt`, `eta_avg_seconds`, and `eta_avg_hms`.
  The original `eta_hms` remains based on tqdm's latest reported
  seconds-per-prompt, while `eta_avg_hms` uses elapsed time divided by completed
  prompts for a steadier estimate. Focused verification:
  `uv run pytest -q tests/test_phase2_generation_status.py
  tests/test_run_phase2_post_shard_analysis.py` (`5 passed`) and
  `uv run ruff check src/slop_sftdiv/cli/phase2_generation_status.py
  src/slop_sftdiv/cli/run_phase2_post_shard_analysis.py
  tests/test_phase2_generation_status.py
  tests/test_run_phase2_post_shard_analysis.py` (`All checks passed`).
  Restarted only the CPU-side post-shard watcher; generation worker PID `5950`
  was left untouched. New watcher Python PID: `22144`. First wait line:
  `waiting for shard completion: 0/8192 rows; log_prompts=304;
  log_generation_estimate=2432; eta=3:15:22; eta_avg=3:22:51`.
- Post-reset audit at 2026-06-15 00:11 UTC: sourced
  `scripts/phase2_cuda_env.sh`, confirming `LD_LIBRARY_PATH` is rebuilt from
  the CUDA 12.8 wheel libraries. The active DPO 1,024-prompt target-shape
  generation worker is still alive as Python PID `5950`, using the A100 at
  about 96% GPU utilization and 48.9 GiB VRAM. The CPU-side post-shard watcher
  remains alive as Python PID `22144`. The active generation log has advanced
  to `464` prompts (`~3712` generation completions at 8 completions per
  prompt), while JSONL and summary outputs remain absent because this shard was
  launched before the streaming-writer fix. Re-ran
  `uv run pytest -q tests/test_phase2_generation_status.py
  tests/test_run_phase2_post_shard_analysis.py` (`5 passed`). Post-shard
  analysis should still run automatically when the shard writes `8192` JSONL
  rows. Caveat for interpreting that analysis: the default compounding
  propensity grid,
  `artifacts/phase2/analysis/olmo3_promptpkg1024_slop_neutral_common_normalized_stage_grid.csv`,
  contains exact teacher-forced AF for `slop_lexicon` and
  `neutral_common_controls`; other retained Tier-1 features in the compounding
  invocation provide generation-side/support context rather than fully
  AF-backed estimates.
- Completed the scaled DPO 1,024-prompt target-shape generation shard on
  2026-06-15. Generation W&B run:
  `stage2-phase2-olmo3-dpo-promptpkg5000-free-running-1024prompt-8comp-t1-bench1024`
  (`p4v56vda`). Output:
  `artifacts/phase2/generations/olmo3_dpo_promptpkg5000_free_run_1024prompt_8comp_t1_batched16.jsonl`
  with `8192/8192` rows and summary
  `artifacts/phase2/generations/olmo3_dpo_promptpkg5000_free_run_1024prompt_8comp_t1_batched16_summary.csv`.
  The run produced 8,388,608 generated tokens. The DPO 512-vs-1024 scale
  comparison W&B run is `dezwh1xv`; the corrected DPO 1024 compounding W&B run
  is `nd1lnfjy`.
- The scaled DPO result stabilizes the t=1.0 target-shape `slop_lexicon` rate:
  512 prompts `0.229` vs. 1,024 prompts `0.226` hits per 1k generated tokens
  (`0.235` vs. `0.231` per generation). Other 1,024-prompt DPO rates per 1k
  generated tokens are `rule_of_three_approx=0.784`,
  `contrastive_negation=0.134`, `stock_openers_closers=0.090`,
  `stock_openers=0.058`, and `stock_closers=0.032`.
- The 1,024-prompt DPO compounding join reports `slop_lexicon` observed
  `0.662` vs. expected `0.439` per 1k opportunities, excess `0.223`, realized
  AF `1.133`, and 383 repeat generations. The empirical window test remains
  strongly positive: `P(hit after prior)=0.0695` vs.
  `P(hit no prior)=0.0119` (risk ratio `5.83`). This supports a bounded
  positive Result B signal at the larger DPO generation scale without changing
  the earlier caveat that the 512-prompt four-stage target-shape grid is not a
  clean DPO-stage generation peak.
- Fixed `slop-analyze-phase2-compounding` pooled-hit accounting for
  `stock_openers_closers`. The compounding analyzer already counted separate
  `stock_openers` and `stock_closers` hits, but did not emit pooled hit starts
  for the direct prior/no-prior test. Added a regression test and reran the
  post-shard analysis. Corrected pooled row now matches the generation summary:
  755 pooled hits, 70 repeat generations, observed `0.822` per 1k pooled
  opportunities, and risk ratio `1.15`. Focused verification:
  `uv run pytest -q tests/test_analyze_phase2_compounding.py
  tests/test_run_phase2_post_shard_analysis.py` (`7 passed`) and
  `uv run ruff check src/slop_sftdiv/cli/analyze_phase2_compounding.py
  tests/test_analyze_phase2_compounding.py` (`All checks passed`).
- Closed Phase 2 as a single-temperature OLMo/Dolci package at generation
  temperature `1.0`. Reran the four-stage target-shape compounding join after
  the pooled `stock_openers_closers` accounting fix. W&B run:
  `stage2-phase2-olmo3-generation-compounding-target-shape-512prompt-8comp-t1-tf1024-v2`
  (`66a0iy2z`). The primary `slop_lexicon` Result B rows are unchanged in
  interpretation: base observed/expected/excess per 1k opportunities
  `0.615/0.232/0.383`, SFT `0.697/0.336/0.361`, DPO `0.638/0.439/0.199`, and
  final/RLVR `0.582/0.445/0.137`. Corrected pooled `stock_openers_closers`
  direct-test rows now have nonzero hits across all stages: base `382`, SFT
  `277`, DPO `452`, final/RLVR `380`.
- Rebuilt the final single-temperature amplification spectrum as v6. W&B run:
  `stage2-phase2-olmo3-amplification-spectrum-single-temp-t1-v6`
  (`jxqcadhe`). Outputs:
  `artifacts/phase2/analysis/olmo3_amplification_spectrum_single_temp_t1_v6.csv`
  and
  `artifacts/phase2/analysis/olmo3_amplification_spectrum_single_temp_t1_v6_summary.md`.
  The table has 24 feature-stage rows, all 24 target-shape free-running cells,
  and 20 teacher-forced cells.
- Logged the final retained single-temperature Phase 2 artifact manifest as
  W&B run `stage2-phase2-olmo3-single-temp-t1-final-artifact-manifest`
  (`oxc252zk`). Local manifest outputs:
  `artifacts/phase2/analysis/olmo3_phase2_single_temp_t1_final_artifact_manifest.csv`,
  `artifacts/phase2/analysis/olmo3_phase2_single_temp_t1_final_artifact_manifest.json`,
  and
  `artifacts/phase2/analysis/olmo3_phase2_single_temp_t1_final_artifact_manifest.md`.
  This manifest supersedes the older v5 headline manifest for Phase 2 close-out.
- Added `slop-compare-phase2-biber` to measure Biber-lite register proxies on
  cached Phase 2 generations and compare them with Phase 1 corpus-role samples
  from `artifacts/stage1/census/feature_rates_by_corpus.parquet`. Ran it over
  the final single-temperature `t=1.0` target-shape generation caches for
  base/SFT/DPO/final. W&B run:
  `stage2-phase2-olmo3-biber-lite-generation-vs-corpus-t1` (`eu4glzoq`).
  Outputs:
  `artifacts/phase2/analysis/olmo3_biber_lite_generation_vs_corpus_t1.csv`
  and
  `artifacts/phase2/analysis/olmo3_biber_lite_generation_vs_corpus_t1_summary.md`.
  The table has 76 rows: 19 Biber-lite features across four stages.
- Main Biber-lite register read: averaged across generated stages, the largest
  generation-vs-SFT-target lifts are demonstratives (`+2.147` per 1k tokens),
  third-person pronouns (`+1.932`), second-person pronouns (`+1.706`),
  necessity modals (`+1.674`), first-person pronouns (`+1.658`), and
  conditional subordinators (`+1.562`). From base to DPO/final, demonstratives,
  first-person pronouns, infinitives, necessity modals, hedges, and
  that-complements decline, while second-person pronouns and conditional
  subordinators rise. This is register context only, not teacher-forced AF or
  Result B compounding.
- Regenerated the final Phase 2 artifact manifest to include the Biber-lite
  comparison. W&B run:
  `stage2-phase2-olmo3-single-temp-t1-final-artifact-manifest-v2`
  (`p3ehymfc`). The manifest now covers 14 retained artifacts and 192 records.
- Added `slop-build-style-signature` to combine Tier-1 slop emission,
  empirical compounding metrics, and Biber-lite register rates into a final
  generated-output style signature. Ran it on the single-temperature `t=1.0`
  Phase 2 close-out artifacts. W&B run:
  `stage2-phase2-olmo3-style-signature-t1` (`bt0zelup`). Outputs:
  `artifacts/phase2/analysis/olmo3_style_signature_t1.csv`,
  `artifacts/phase2/analysis/olmo3_style_signature_t1_stage_distances.csv`,
  and `artifacts/phase2/analysis/olmo3_style_signature_t1_summary.md`. The
  signature has 172 rows over 25 feature names and four stages; the distance
  matrix has 16 rows.
- Style-signature read: on raw metric scales, final/RLVR is closest to DPO
  (`30.567` Euclidean distance), then SFT (`35.564`), then base (`85.139`).
  Final vs. base mainly lowers demonstratives, infinitives, first-person
  pronouns, necessity/prediction/possibility modals, and stock opener
  opportunities, while increasing rule-of-three observed compounding
  opportunities and nominalizations. Final vs. DPO lowers rule-of-three
  observed compounding opportunities and stock opener observed opportunities,
  but raises stock-closer prior risk ratio and several Biber register proxies
  including first-person pronouns, causal subordinators, hedges, infinitives,
  passive-voice approximation, public verbs, and possibility modals.
- Regenerated the final retained artifact manifest to include the style
  signature. W&B run:
  `stage2-phase2-olmo3-single-temp-t1-final-artifact-manifest-v3`
  (`5qonr7io`). The manifest now covers 17 artifacts and 380 records.
- Added `docs/experiments/phase2_final_conclusion_report.md` as a standalone
  reader-facing Phase 2 conclusion report. It explains the Phase 2 question,
  OLMo 3 checkpoint ladder, measurement views, retained feature set, main
  results, checkpoint progression, caveats, final conclusion, and primary
  artifacts for readers new to the project. The report keeps the final read
  aligned with the retained artifacts: DPO is the clearest `slop_lexicon`
  propensity peak, final/RLVR is closest to DPO in aggregate generated-output
  style, and final/RLVR does not broadly amplify slop beyond DPO.

## 2026-06-15 - Phase 3 bounded OLMo classification layer

- Added `slop-classify-amplification-spectrum`, which reads an assembled
  amplification-spectrum table and emits feature-level Phase 3 classifications
  (`inherited`, `sft-amplified`, `preference-amplified`,
  `compounding-dominant`, or coverage-limited fallback classes). The classifier
  records chosen-vs-rejected preference-data complicity, AF jumps, free-running
  maxima, compounding summary fields, and AF-stage FDR availability status.
- Ran the classifier over
  `artifacts/phase2/analysis/olmo3_amplification_spectrum_single_temp_t1_v6.csv`,
  joined with
  `artifacts/stage1/census/olmo3_dolci_dpo_10k_pair_analysis.csv` for paired
  Result A sign-test p-values and BH-FDR q-values.
  W&B run:
  `stage3-phase3-olmo3-bounded-feature-classification-t1-v4-pref-fdr`
  (`4oabv7j8`).
  Outputs:
  `artifacts/phase3/analysis/olmo3_phase3_bounded_feature_classification_t1.csv`
  and
  `artifacts/phase3/analysis/olmo3_phase3_bounded_feature_classification_t1_summary.md`.
- Bounded classification read: `slop_lexicon` is the only
  preference-amplified retained feature view, labeled dynamics-driven because
  paired Result A evidence is nonsignificant and rejected-greater-chosen
  (`BH q=0.963`) for the feature.
  `stock_closers` and pooled `stock_openers_closers` are SFT-amplified/high-AF
  rather than preference-jump features; `rule_of_three_approx` and
  `stock_openers` remain measured but unclassified under Phase 3 rules; and
  `contrastive_negation` is output-only due missing teacher-forced support.
- Added `docs/experiments/phase3_status.md` with the bounded result, current
  requirement audit, and remaining work for full EXPERIMENTS.md Phase 3:
  AF-stage p-value/FDR layer, SmolLM3 no_think replication, and cross-ladder
  AF rank correlation.
- Logged a retained Phase 3 bounded artifact manifest as
  `stage3-phase3-olmo3-bounded-artifact-manifest-v3` (`nlh83sd9`), covering
  the classification CSV and summary markdown.

## 2026-06-15 - Phase 3 cross-ladder comparison tooling

- Added `slop-compare-phase3-ladders`, which aligns two amplification-spectrum
  CSVs by feature-stage pair and reports AF Spearman/Pearson correlations
  overall and by stage. It uses normalized AF when available and raw AF
  otherwise.
- Ran an OLMo-vs-OLMo self-check with W&B disabled:
  `artifacts/phase3/analysis/olmo3_phase3_cross_ladder_selfcheck_aligned.csv`,
  `artifacts/phase3/analysis/olmo3_phase3_cross_ladder_selfcheck_correlations.csv`,
  and
  `artifacts/phase3/analysis/olmo3_phase3_cross_ladder_selfcheck_summary.md`.
  The self-check aligned 24 feature-stage rows, 20 non-missing AF values, and
  produced overall Spearman AF `1.000`.
- This completes the reusable cross-ladder correlation tooling for Phase 3,
  but the actual OLMo-vs-SmolLM3 result still requires the SmolLM3 no_think
  Phase 1/2 artifacts and assembled spectrum.

## 2026-06-15 - SmolLM3 checkpoint-revision harness support

- Added optional Hugging Face revision support to the Phase 2 model-loading
  CLIs needed for Phase 3 replication:
  `slop-free-running-emission --model-revision REVISION` and
  `slop-teacher-forced-propensity --model-revision REVISION`. Both pass the
  revision to tokenizer and model `from_pretrained` and record it in W&B
  config.
- Updated `slop-plan-phase2-generation` to accept `STAGE=MODEL@REVISION`
  stage specs, emit a separate `model_revision` column, and include
  `--model-revision` in planned commands. Existing `STAGE=MODEL` specs remain
  backward compatible.
- Added `slop-plan-phase2-propensity`, a matching teacher-forced AF planner
  that accepts the same `STAGE=MODEL@REVISION` format, writes
  `model_revision`, estimates rough opportunity throughput, tracks existing
  opportunity/summary outputs, and emits `slop-teacher-forced-propensity`
  commands with the requested features, normalization feature, reference
  subsets, cache settings, bootstrap settings, and revision flags.
- Re-verified current SmolLM3 branch refs from Hugging Face on 2026-06-15:
  base `HuggingFaceTB/SmolLM3-3B-Base` main
  `d78a42f79198603e614095753484a04c10c2b940`, checkpoint repo branches
  `it-SFT` `f6ddaa5f2e99f24ea507596c214595769fb06387`,
  `it-soup-APO` `cfb32d505f5025ec9be4e704f70cfbf5bdf8da94`,
  `it-mid-training` `0485ec16b618f88f6dbb27b23dbdecca1d6a1cdd`,
  `it-LC-expert` `99d07b0a954f07f1237176440b0a61149c7d03d3`, and final
  `HuggingFaceTB/SmolLM3-3B` main
  `a07cc9a04f16550a088caea529712d1d335b0ac1`.
- Updated `docs/experiments/phase3_status.md` with the SmolLM3 checkpoint
  table, example generation-plan stage specs, and W&B-disabled generation and
  teacher-forced planner smoke artifacts. This removes a harness blocker for
  SmolLM3 no_think replication, but the SmolLM3 prompt package,
  teacher-forced summaries, free-running summaries, assembled spectrum, and
  OLMo-vs-SmolLM3 AF rank correlation are still missing.

## 2026-06-15 - Phase 3 paired free-running stage-effect FDR

- Added `slop-analyze-phase3-free-run-effects`, which loads stage-tagged
  free-running generation JSONL caches, aligns generations by shared
  `(record_id, completion_index, temperature, top_p)`, computes paired
  per-completion feature-rate deltas, runs two-sided sign tests, and applies
  Benjamini-Hochberg FDR across feature/comparison rows.
- Ran it on the retained OLMo target-shape `t=1.0` caches for base, SFT, DPO,
  and final/RLVR. Local outputs:
  `artifacts/phase3/analysis/olmo3_phase3_free_run_stage_effects_t1.csv` and
  `artifacts/phase3/analysis/olmo3_phase3_free_run_stage_effects_t1_summary.md`.
- Result: 18 feature-comparison rows over adjacent ladder comparisons and six
  retained feature views; 15 rows are BH-FDR significant at alpha `0.05`.
  SFT -> DPO significantly increases all six retained free-running feature
  views. DPO -> final/RLVR significantly decreases `slop_lexicon`,
  `stock_closers`, and pooled `stock_openers_closers`, while
  `rule_of_three_approx`, `stock_openers`, and `contrastive_negation` are not
  significant in the DPO -> final/RLVR comparison.
- This closes the paired/FDR layer for retained OLMo target-shape free-running
  stage effects, but AF-stage p-values/FDR and SmolLM3 replication remain
  incomplete Phase 3 requirements.

## 2026-06-15 - Phase 3 paired teacher-forced stage-effect FDR

- Added `slop-analyze-phase3-teacher-forced-effects`, which loads one or more
  stage-tagged `slop-teacher-forced-propensity` opportunity CSVs per stage,
  aligns rows by source/record/role/feature/opportunity kind/character offset,
  computes paired probability-mass and reference-rate-derived AF deltas, runs
  two-sided sign tests, and applies Benjamini-Hochberg FDR across
  feature/comparison rows.
- Ran it over the retained OLMo teacher-forced opportunity grids:
  1,024-prompt `slop_lexicon`/`neutral_common_controls`, 5,000-prompt
  `rule_of_three_approx` comma-pair extension, and 5,000-prompt stock
  opener/closer grids. Local outputs:
  `artifacts/phase3/analysis/olmo3_phase3_teacher_forced_stage_effects_t1.csv`
  and
  `artifacts/phase3/analysis/olmo3_phase3_teacher_forced_stage_effects_t1_summary.md`.
- Result: 18 feature-comparison rows; 17 are BH-FDR significant at alpha
  `0.05`. SFT -> DPO has significant teacher-forced stage changes for every
  retained measured feature view, including `slop_lexicon`. DPO -> final/RLVR
  is significant for every measured feature view except `stock_openers`.
  `contrastive_negation` remains absent from teacher-forced effect rows
  because its retained Phase 2 teacher-forced support is missing.
- Regenerated
  `artifacts/phase3/analysis/olmo3_phase3_bounded_feature_classification_t1.*`
  with preference, teacher-forced stage, and free-running stage q-values joined
  into the feature-level classifier table. Local run name:
  `stage3-phase3-olmo3-bounded-feature-classification-t1-v5-stage-fdr`
  with W&B disabled during regeneration.
- This closes the retained OLMo bounded paired/FDR requirements for both
  teacher-forced and free-running stage effects. SmolLM3 no_think replication,
  assembled SmolLM3 spectrum, and real cross-ladder AF rank correlation remain
  incomplete.
