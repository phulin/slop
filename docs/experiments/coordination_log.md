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

## 2026-06-15 - SmolLM3 no_think prompt package and launch plans

- Prepared the first real SmolLM3 no_think Phase 2 prompt package from
  `HuggingFaceTB/smoltalk2`, config `SFT`, split
  `smoltalk_smollm3_everyday_conversations_no_think`, using
  `sample_size=512`, `max_scan=5000`, `sampling_strategy=hash_reservoir`, and
  seed `1729`. The split yielded 2,260 scanned rows before exhaustion, 2,258
  eligible prompts after near-duplicate filtering, and 512 selected prompts
  with 20,081 prompt tokens and 43,390 target tokens. Outputs:
  `artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_512.jsonl`,
  `artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_512_manifest.csv`,
  and
  `artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_512_summary.json`.
- Generated full 512-prompt SmolLM3 no_think launch plans for the base, SFT,
  APO-soup, and final stages. Generation plan outputs:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512_t1.csv`
  and `.md`, estimating 13.09 missing A100-hours for `512` prompts x `8`
  completions at `t=1.0`, `top_p=0.95`, `max_new_tokens=1024`. Teacher-forced
  slop/neutral propensity plan outputs:
  `artifacts/phase3/analysis/smollm3_no_think_propensity_plan_512_slop_neutral.csv`
  and `.md`, estimating 23.30 missing A100-hours.
- Verified a tiny final-checkpoint SmolLM3 generation smoke on the free A100:
  `2` prompts, `1` completion, `64` max new tokens, `t=1.0`, `top_p=0.95`,
  `dtype=bfloat16`, W&B disabled, no compile. Outputs:
  `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_2prompt_1comp_t1_64tok_smoke.jsonl`
  and
  `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_2prompt_1comp_t1_64tok_smoke_summary.csv`.
  The smoke wrote valid feature-count JSON and the sampled generations had no
  obvious `<think>`/`</think>`/`reasoning` markup.
- Verified a matching tiny final-checkpoint SmolLM3 teacher-forced smoke on
  the same prompt package: `2` prompts, `slop_lexicon` plus
  `neutral_common_controls`, `max_opportunities=32`, exact sequence mass,
  `dtype=bfloat16`, W&B disabled, no compile. Outputs:
  `artifacts/phase2/propensity/smollm3_final_smoltalk2_everyday_no_think_2prompt_slop_neutral_smoke_opportunities.csv`,
  `artifacts/phase2/propensity/smollm3_final_smoltalk2_everyday_no_think_2prompt_slop_neutral_smoke_summary.csv`,
  and
  `artifacts/phase2/propensity/smollm3_final_smoltalk2_everyday_no_think_2prompt_slop_neutral_smoke_reference_subset_summary.csv`.
  It wrote 64 opportunity rows and two feature summaries. Both features had
  zero reference initiations in the 2-prompt smoke sample, so this is a
  harness/schema check only, not an interpretable AF result.
- Updated `docs/experiments/phase3_status.md` so the SmolLM3 replication row is
  now "started, data mostly missing" rather than fully missing. Full Phase 3
  still needs the four-stage SmolLM3 free-running and teacher-forced summaries,
  SmolLM3 amplification-spectrum assembly, and the real OLMo-vs-SmolLM3 AF
  rank-correlation result.

## 2026-06-15 - SmolLM3 no_think chat-template generation support

- Added `--apply-chat-template` and `--chat-template-kwargs-json` to
  `slop-free-running-emission`, plus passthrough support in
  `slop-plan-phase2-generation`. This is needed for SmolLM3 no_think
  generation because the plain-prompt smoke can make the final checkpoint
  continue the dialogue as user text instead of answering as the assistant.
- The chat-template implementation handles tokenizer return variants observed
  in practice: raw strings, tensors/lists, and `BatchEncoding`/mapping objects
  with `input_ids`. Focused tests:
  `uv run pytest -q tests/test_free_running_emission.py
  tests/test_plan_phase2_generation.py` (`7 passed`) and
  `uv run ruff check src/slop_sftdiv/cli/free_running_emission.py
  src/slop_sftdiv/cli/plan_phase2_generation.py
  tests/test_free_running_emission.py tests/test_plan_phase2_generation.py`
  (`All checks passed`).
- Regenerated the canonical SmolLM3 no_think generation plan with
  `--apply-chat-template --chat-template-kwargs-json '{"enable_thinking":
  false}'`: `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512_t1_chat.csv`
  and `.md`. It supersedes the earlier plain-prompt generation plan for actual
  SmolLM3 no_think free-running work.
- Re-ran the 2-prompt final-checkpoint generation smoke through the chat
  template. Outputs:
  `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_2prompt_1comp_t1_64tok_chat_smoke.jsonl`
  and
  `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_2prompt_1comp_t1_64tok_chat_smoke_summary.csv`.
  The generated text is assistant-style, writes valid feature-count JSON, and
  has no obvious generated reasoning markup.

## 2026-06-15 - SmolLM3 all-stage calibration spectrum

- Added `--missing-chat-template {error,plain}` support to
  `slop-free-running-emission` and `slop-plan-phase2-generation`. The
  canonical SmolLM3 no_think generation plan now uses
  `--apply-chat-template --chat-template-kwargs-json '{"enable_thinking":
  false}' --missing-chat-template plain`, which keeps the base checkpoint in
  the same grid despite its missing chat template while preserving
  chat-template rendering for SFT/APO/final.
- Updated `slop-assemble-amplification-spectrum` to use a model-neutral
  summary title because the same assembler now produces both OLMo and SmolLM3
  spectra.
- Ran a W&B-disabled all-stage SmolLM3 no_think generation calibration:
  base/SFT/APO/final, `4` prompts, `1` completion per prompt, `64` max new
  tokens, `t=1.0`, `top_p=0.95`, bfloat16, no compile. Local summaries:
  `artifacts/phase2/generations/smollm3_base_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration_summary.csv`,
  `artifacts/phase2/generations/smollm3_sft_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration_summary.csv`,
  `artifacts/phase2/generations/smollm3_dpo_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration_summary.csv`,
  and
  `artifacts/phase2/generations/smollm3_final_smoltalk2_everyday_no_think_4prompt_1comp_t1_64tok_chat_calibration_summary.csv`.
  The calibration produced 256 generated tokens per stage. `rule_of_three`
  rates were 0.000/base, 7.812/SFT, 11.719/APO, and 15.625/final per 1k
  generated tokens; `slop_lexicon` appeared once in base only
  (3.906/1k). These are calibration counts, not stable estimates.
- Ran a matching W&B-disabled all-stage SmolLM3 teacher-forced slop/neutral
  calibration: base/SFT/APO/final, `4` prompts, `slop_lexicon` plus
  `neutral_common_controls`, `32` opportunities per feature, exact sequence
  mass, bfloat16, no compile. Local summaries:
  `artifacts/phase2/propensity/smollm3_base_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_summary.csv`,
  `artifacts/phase2/propensity/smollm3_sft_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_summary.csv`,
  `artifacts/phase2/propensity/smollm3_dpo_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_summary.csv`,
  and
  `artifacts/phase2/propensity/smollm3_final_smoltalk2_everyday_no_think_4prompt_slop_neutral_calibration_summary.csv`.
  All four summaries had zero reference initiations for both features, so raw
  AF and normalized AF are `0.0`. This verifies scoring and schema only; it is
  not an interpretable propensity result.
- Assembled the SmolLM3 calibration generation grid, propensity grid,
  amplification spectrum, and feature classifications:
  `artifacts/phase3/analysis/smollm3_no_think_generation_stage_grid_4prompt_1comp_t1_64tok_chat_calibration.csv`,
  `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_4prompt_slop_neutral_calibration.csv`,
  `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_4prompt_calibration.csv`,
  and
  `artifacts/phase3/analysis/smollm3_no_think_feature_classification_4prompt_calibration.csv`.
- Ran the real OLMo-vs-SmolLM3 comparator path:
  `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_4prompt_calibration_aligned.csv`,
  `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_4prompt_calibration_correlations.csv`,
  and
  `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_4prompt_calibration_summary.md`.
  It aligned 24 feature-stage rows, but Spearman/Pearson AF are blank because
  the SmolLM3 AF vector is constant zero. Full Phase 3 still requires the
  512-prompt SmolLM3 no_think teacher-forced and free-running ladder before an
  interpretable cross-ladder AF rank correlation can be reported.
- Verification after this pass:
  `uv run ty check src tests`, `uv run ruff check src tests`,
  `uv run pytest -q tests/test_free_running_emission.py tests/test_plan_phase2_generation.py tests/test_assemble_amplification_spectrum.py tests/test_compare_phase3_ladders.py tests/test_classify_amplification_spectrum.py tests/test_teacher_forced_propensity.py`,
  and `git diff --check` all pass.

## 2026-06-15 - SmolLM3 512-prompt teacher-forced slop/neutral ladder

- Completed the full 512-prompt SmolLM3 no_think teacher-forced slop/neutral
  ladder over the SmolTalk2 no_think prompt package. Stages and W&B runs:
  base `q7xrq9x6`, SFT `rmwh8p2z`, APO `e4xukin8`, final `cy8si8df`.
  Each stage scored 43,294 opportunities for `slop_lexicon` and 43,294 for
  `neutral_common_controls`; the prompt package has 19 slop reference
  initiations and 2,915 neutral-control reference initiations.
- Assembled the 512-prompt SmolLM3 teacher-forced stage grid:
  `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_512prompt_slop_neutral.csv`
  and `.md`, plus the stage-comparison CSV. Neutral-normalized
  `slop_lexicon` AF rises from base `6.902` to SFT `7.647`, APO `8.141`, and
  final `8.345`.
- Assembled a TF-only SmolLM3 amplification spectrum and feature
  classification:
  `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_slop_neutral.csv`
  and
  `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_slop_neutral.csv`.
  The classifier labels `slop_lexicon` as `sft-amplified` under the current
  rules because AF is already high at SFT and the APO/final relative increase
  over SFT is modest. This is provisional because SmolLM3 corpus,
  free-running, and compounding layers are not yet present.
- Ran the first interpretable OLMo-vs-SmolLM3 teacher-forced cross-ladder
  comparison:
  `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_slop_neutral_aligned.csv`,
  `_correlations.csv`, and `_summary.md`. It aligns four `slop_lexicon` stage
  rows and reports overall Spearman AF `0.400` and Pearson AF `0.685`.
  This satisfies the command path and provides a narrow slop-lexicon TF
  comparison, but not the full Phase 3 cross-feature rank-correlation result.
- Made the amplification-spectrum assembler and classifier summaries
  model-neutral so SmolLM3 summaries no longer claim to be OLMo/Dolci tables.
  Full Phase 3 still needs the SmolLM3 512-prompt free-running ladder,
  compounding, and broader SmolLM3 teacher-forced feature coverage where
  denominator support permits it.

## 2026-06-15 - SmolLM3 512-prompt rule-of-three teacher-forced proxy

- Added a full 512-prompt SmolLM3 no_think teacher-forced run for
  `rule_of_three_approx` using the same comma-pair extension proxy as the OLMo
  bounded Phase 2 spectrum. Denominator support is good for this proxy on the
  SmolTalk2 no_think prompt package: 754 opportunities and 628 references.
  W&B runs: base `xud6jfor`, SFT `9alxeeh7`, APO `6j3vuu0e`, final
  `jdruuqay`.
- SmolLM3 `rule_of_three_approx` raw AFs are base `0.673`, SFT `0.741`, APO
  `0.755`, and final `0.755`. The proxy remains below reference rate at every
  stage and does not create a strong APO/final preference-stage jump.
- Assembled the rule-of-three grid and the broader TF-only SmolLM3 spectrum:
  `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_512prompt_rule_of_three.csv`,
  `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_slop_neutral_rule3.csv`,
  and
  `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_slop_neutral_rule3.csv`.
  The classifier labels `slop_lexicon` as `sft-amplified`,
  `rule_of_three_approx` as `measured-no-phase3-class`, and
  `neutral_common_controls` as `measured-no-phase3-class`.
- Ran paired SmolLM3 teacher-forced stage-effect tests over the combined
  slop/neutral/rule3 opportunity rows:
  `artifacts/phase3/analysis/smollm3_no_think_teacher_forced_stage_effects_512prompt_tf_slop_neutral_rule3.csv`.
  It produced 9 adjacent comparison rows, 8 BH-FDR-significant at alpha
  `0.05`. Note the existing caveat: sign-test direction can differ from mean
  AF delta when many small probability changes oppose fewer larger changes.
- Re-ran OLMo-vs-SmolLM3 comparison on the two-feature teacher-forced slice:
  `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_slop_neutral_rule3_aligned.csv`,
  `_correlations.csv`, and `_summary.md`. It aligns 8 feature-stage AF rows
  across `slop_lexicon` and `rule_of_three_approx`, with overall Spearman AF
  `0.762` and Pearson AF `0.978`; per-stage correlations are `1.000` because
  each stage has only two shared feature values. This is an interpretable
  bounded teacher-forced cross-ladder read, not the full production
  free-running/compounding replication.

## 2026-06-15 - SmolLM3 512-prompt free-running and compounding ladder

- Completed the bounded SmolLM3 no_think free-running ladder on the same
  512-prompt SmolTalk2 prompt package: base, SFT, APO-soup, and final;
  `8` completions per prompt; `t=1.0`; `top_p=0.95`; `max_new_tokens=1024`;
  bfloat16; `torch.compile`; chat-template rendering with
  `{"enable_thinking": false}` and plain fallback for the base checkpoint.
  Retained generation W&B runs: base `stcaaobk`, SFT `prb8bw1b`, APO
  `mewe19y4`, final `4oxiq1xr`.
- Added an inference-cache safeguard to `slop-free-running-emission`: when the
  loaded model exposes `config.use_cache` or `generation_config.use_cache`,
  the harness sets it to `True`. The SmolLM3 SFT/APO branch configs advertise
  `use_cache=False`, which made full APO generation impractically slow before
  this override. A focused unit test now covers the cache override.
- Assembled the SmolLM3 no_think generation stage grid:
  `artifacts/phase3/analysis/smollm3_no_think_generation_stage_grid_512prompt_8comp_t1_chat.csv`
  and `_summary.md`. Free-running rates per 1k generated tokens:
  `slop_lexicon` base `0.118`, SFT `0.269`, APO `0.376`, final `0.391`;
  `rule_of_three_approx` base `1.248`, SFT `4.081`, APO `5.516`, final
  `5.513`; `contrastive_negation` base `0.092`, SFT `0.094`, APO `0.096`,
  final `0.105`; pooled stock phrases base `0.193`, SFT `0.174`, APO
  `0.284`, final `0.271`.
- Ran paired SmolLM3 free-running stage-effect tests:
  `artifacts/phase3/analysis/smollm3_no_think_free_run_stage_effects_512prompt_8comp_t1_chat.csv`.
  It produced 18 rows and 8 BH-FDR-significant rows. Strongest reads:
  base -> SFT increases `rule_of_three_approx` and `slop_lexicon`; SFT -> APO
  increases `rule_of_three_approx`, `stock_openers`, pooled stock phrases, and
  `slop_lexicon`; APO -> final changes are not significant for the retained
  feature family.
- Ran the generation-inclusive compounding join:
  `artifacts/phase3/analysis/smollm3_no_think_generation_compounding_512prompt_8comp_t1_chat_tf_slop_neutral_rule3.csv`
  and `_summary.md`. For `slop_lexicon`, direct prior/no-prior windows remain
  positive in every stage, but observed-vs-expected excess is positive only at
  base (`+0.090` per 1k opportunities); SFT/APO/final have negative
  observed-vs-expected excess but realized AF around `2.05` to `2.27`.
  `rule_of_three_approx` has negative base excess (`-229.840`) and positive
  SFT/APO/final excess (`+187.735`, `+149.541`, `+151.376`) under the
  comma-pair extension proxy.
- Assembled the generation-inclusive SmolLM3 amplification spectrum and
  classifier:
  `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_slop_neutral_rule3.csv`
  and
  `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_slop_neutral_rule3.csv`.
  The classifier labels `slop_lexicon` as `sft-amplified`,
  `rule_of_three_approx` and `neutral_common_controls` as
  `measured-no-phase3-class`, and the remaining retained features as
  `observed-output-only` because teacher-forced support is missing or sparse.
- Re-ran OLMo-vs-SmolLM3 on the generation-inclusive SmolLM3 spectrum:
  `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_tf_generation_compounding_slop_neutral_rule3_aligned.csv`,
  `_correlations.csv`, and `_summary.md`. It aligns 24 feature-stage rows and
  has 8 shared AF values from `slop_lexicon` and the
  `rule_of_three_approx` teacher-forced proxy. Overall Spearman AF is `0.762`;
  Pearson AF is `0.978`. This was the best bounded cross-ladder result before
  adding the SmolLM3 data-rate layer below, but not full Phase 3 completion
  because the original production grid, SmolLM3 data-rate baselines, broader
  teacher-forced support, and stretch Think/RL Zero comparison remained
  missing.

## 2026-06-15 - SmolLM3 SFT/preference data-rate layer

- Sampled the SmolTalk2 no_think SFT split for the bounded SmolLM3 Phase 3
  data-rate layer:
  `artifacts/stage1/corpora/smollm3_smoltalk2_sft_everyday_no_think_2260.jsonl`
  and `_summary.md`. The retained sample has 2,260 target responses and
  188,621 simple tokens from
  `smoltalk_smollm3_everyday_conversations_no_think`.
- Sampled 10,000 SmolTalk2 no_think Tulu preference pairs, expanded into
  chosen/rejected response rows:
  `artifacts/stage1/corpora/smollm3_smoltalk2_preference_tulu_no_think_10k_pairs.jsonl`
  and `_summary.md`. The retained sample has 20,000 response rows and
  4,469,915 simple tokens from
  `llama_3.1_tulu_3_8b_preference_mixture_no_think`.
- Ran Tier-1 census and paired preference analysis:
  `artifacts/stage1/census/smollm3_smoltalk2_sft2260_pref10k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_smoltalk2_pref10k_tier1_pair_deltas.csv`,
  and
  `artifacts/stage1/census/smollm3_smoltalk2_pref10k_tier1_pair_analysis.csv`.
  Key SFT/Chosen/Rejected rates per 1k tokens:
  `slop_lexicon` `0.419`/`1.609`/`1.293`,
  `rule_of_three_approx` `17.045`/`5.443`/`4.877`,
  `contrastive_negation` `0.095`/`0.379`/`0.352`, and pooled stock phrases
  `0.376`/`0.788`/`0.502`.
- Paired preference evidence is chosen-greater-rejected and BH-FDR-significant
  for `slop_lexicon` (`q=3.367e-15`), `rule_of_three_approx`
  (`q=9.437e-13`), `contrastive_negation` (`q=0.039`), `stock_closers`
  (`q=3.016e-04`), and pooled stock phrases (`q=0.003`). `stock_openers` is
  chosen-greater-rejected in mean rate but not BH-FDR-significant (`q=0.161`).
- Reassembled the SmolLM3 no_think data-rate spectrum and classifier:
  `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_data_rates_slop_neutral_rule3.csv`
  and
  `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_data_rates_slop_neutral_rule3.csv`.
  The classifier still labels `slop_lexicon` as `sft-amplified`, not
  preference-amplified: the no_think Tulu preference data are complicit, but
  SmolLM3 SFT already has high normalized AF and APO/final only add a modest
  relative increase under the current rule.
- Re-ran the OLMo-vs-SmolLM3 bounded comparison on the data-rate spectrum:
  `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_data_rates_tf_generation_compounding_slop_neutral_rule3_aligned.csv`,
  `_correlations.csv`, and `_summary.md`. It keeps the same 24 aligned
  feature-stage rows and 8 shared AF values as the generation-inclusive
  comparison, with overall Spearman AF `0.762` and Pearson AF `0.978`; the new
  value is in the SmolLM3 SFT/preference data-rate fields and classifier
  interpretation.
- Caveat: the normalized SmolTalk2 preference rows preserve split/source
  identity but do not expose per-row `preference_type`, `chosen_model`, or
  `rejected_model`; those columns are `None` in the paired analysis. Full
  Phase 3 still needs the original production grid, a SmolLM3
  pretraining/mid-training baseline, broader teacher-forced support where
  denominator support permits it, and the stretch Think/RL Zero comparison.

## 2026-06-15 - SmolLM3 bounded pretrain/Mid baselines

- Probed the SmolLM3 pretraining collection path and confirmed
  `HuggingFaceTB/smollm3-pretraining-datasets` is a Hugging Face collection,
  not a loadable dataset ID. Bounded pretraining baselines therefore sample
  named source datasets from the collection rather than an exact weighted mix.
- Sampled two pretraining-source baselines and one Mid baseline:
  `artifacts/stage1/corpora/smollm3_pretrain_fineweb_edu_2k.jsonl`
  (2,000 rows, 1,610,774 tokens),
  `artifacts/stage1/corpora/smollm3_pretrain_stackexchange_apple_2k.jsonl`
  (2,000 rows, 1,013,267 tokens), and
  `artifacts/stage1/corpora/smollm3_smoltalk2_mid_llama_nemotron_reasoning_2k.jsonl`
  (2,000 rows, 6,041,904 tokens). The StackExchange sample uses
  `ThreadText`; the Mid sample uses SmolTalk2 `Mid` split
  `Llama_Nemotron_Post_Training_Dataset_reasoning_r1`.
- Ran the Tier-1 census:
  `artifacts/stage1/census/smollm3_pretrain_mid_baselines_2k_tier1_feature_rates.csv`.
  Key rates per 1k tokens:
  `slop_lexicon` pretrain aggregate `0.455`, FineWeb-Edu `0.536`,
  StackExchange `0.327`, Mid `0.050`;
  `rule_of_three_approx` pretrain aggregate `4.424`, FineWeb-Edu `6.211`,
  StackExchange `1.583`, Mid `1.273`;
  `contrastive_negation` pretrain aggregate `0.433`, FineWeb-Edu `0.385`,
  StackExchange `0.509`, Mid `0.177`.
- Updated `slop-assemble-amplification-spectrum` so repeated same-role data
  rates are token-weighted instead of overwritten. It now supports
  `mid_target_response` as `mid_target_per_1k_tokens` and emits
  source-specific pretrain/Mid columns such as
  `pretrain_smollm3_pretrain_fineweb_edu_2k_per_1k_tokens`.
- Reassembled the current best bounded SmolLM3 spectrum and classifier:
  `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3.csv`
  and
  `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3.csv`.
  Class labels are unchanged: `slop_lexicon` remains `sft-amplified`,
  `rule_of_three_approx` remains `measured-no-phase3-class`, and sparse
  teacher-forced features remain `observed-output-only`.
- Re-ran OLMo-vs-SmolLM3 on the baseline data-rate spectrum:
  `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_baselines_data_rates_tf_generation_compounding_slop_neutral_rule3_aligned.csv`,
  `_correlations.csv`, and `_summary.md`. AF support is unchanged from the
  prior data-rate comparison: 24 aligned feature-stage rows, 8 shared AF
  values, overall Spearman AF `0.762`, Pearson AF `0.978`.
- Caveat: this closes the bounded pretrain/Mid baseline gap but not the full
  production baseline. The pretrain aggregate is token-weighted over two
  selected source samples, not the exact weighted SmolLM3 11T pretraining mix.

## 2026-06-15 - Phase 2 report readability pass

- Expanded `docs/experiments/phase2_final_conclusion_report.md` for readers
  new to the project. The report now opens with a plain-language project
  context, explains why Phase 2 separates corpus rates, local propensity,
  sampled output, and compounding, and adds an evidence-strength table for the
  strongest and weakest Phase 2 claim areas.
- Added a "What We Were Looking For" bridge section that maps the original
  Phase 2 plan onto the four causal stories the measurements were designed to
  distinguish: inherited/base-heavy features, SFT amplification, preference
  amplification, and generation compounding. No numeric measurements or
  artifact references changed in this pass.

## 2026-06-15 - SmolLM3 recipe-weight extraction

- Added `slop-extract-smollm3-config-weights`, a reproducible parser for the
  published SmolLM3 Nanotron YAML configs. It derives data-stage spans from
  `start_training_step`, computes tokens per step from config batch geometry,
  normalizes aligned `dataset_weights`, and writes detailed source rows,
  aggregate source weights, heuristic source-group weights, and a Markdown
  summary.
- Ran the extractor on `HuggingFaceTB/smollm3-configs` files
  `stage3_9T_11T.yaml`, `long_context_4k_to_32k.yaml`, and
  `long_context_32k_to_64.yaml`. Outputs:
  `artifacts/phase3/analysis/smollm3_config_source_weights_detail.csv`,
  `artifacts/phase3/analysis/smollm3_config_source_weights_aggregate.csv`,
  `artifacts/phase3/analysis/smollm3_config_source_weights_groups.csv`, and
  `artifacts/phase3/analysis/smollm3_config_source_weights_summary.md`.
- The extraction reports 256 detail rows, 115 aggregate sources, and
  `11.234968T` config-implied tokens when including the two long-context
  extension stages. Top aggregate sources are `dclm` (`35.486%`),
  `fineweb-edu` (`31.140%`), `fw2-deu` (`2.209%`), `fw2-spa` (`2.003%`),
  `stack-edu-Python` (`1.811%`), and `pes2o` (`1.724%`). At that point,
  sampled pretraining feature-rate coverage was quantifiable: `fineweb-edu`
  covers `31.140%` of the extracted recipe and `stackexchange` covers
  `0.333%`.
- Updated `docs/experiments/phase3_status.md` and
  `docs/experiments/source_card_notes.md` to distinguish the now-resolved
  recipe-weight discovery step from the still-missing production weighted
  feature-rate baseline across the remaining SmolLM3 sources.

## 2026-06-15 - SmolLM3 weighted baseline coverage proxy

- Added `slop-assemble-weighted-pretrain-baseline`, which joins sampled
  feature-rate rows to exact recipe source weights through explicit source
  maps and reports coverage-aware weighted rates. It emits both
  `weighted_per_1k_tokens_covered_only` and
  `weighted_per_1k_tokens_missing_as_zero`, so unsampled recipe sources are
  not silently imputed.
- Ran it on
  `artifacts/phase3/analysis/smollm3_config_source_weights_aggregate.csv` and
  `artifacts/stage1/census/smollm3_pretrain_mid_baselines_2k_tier1_feature_rates.csv`
  with source maps `smollm3_pretrain_fineweb_edu_2k=fineweb-edu` and
  `smollm3_pretrain_stackexchange_apple_2k=stackexchange`. Outputs:
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy_summary.md`.
- Current weighted feature-rate coverage is `31.473%` of the extracted recipe
  and missing share is `68.527%`. Covered-only rates per 1k tokens include
  `slop_lexicon` `0.534`, `rule_of_three_approx` `6.162`,
  `contrastive_negation` `0.386`, `stock_closers` `0.041`, `stock_openers`
  `0.027`, and pooled stock phrases `0.069`. These are coverage-normalized
  diagnostics, not full-mixture estimates. Updated Phase 3 docs to identify
  `dclm` (`35.486%`) as the next highest-impact source for pretraining
  feature-rate sampling.

## 2026-06-15 - DCLM pretraining feature-rate coverage

- Fixed `slop-sample-corpus` retained-row metadata so the final outer sampling
  strategy is stamped onto JSONL and manifest rows. This matters for bounded
  samples where the command scans with `first` internally and then applies its
  own `hash_reservoir` selection. Added test coverage in
  `tests/test_sample_corpus.py`.
- Sampled DCLM for the SmolLM3 pretraining baseline:
  `artifacts/stage1/corpora/smollm3_pretrain_dclm_2k.jsonl`, with manifests
  and summary. Source: `mlfoundations/dclm-baseline-1.0`, config `default`,
  split `train`, 20,000 scanned rows, 2,000 retained rows, 1,780,498 simple
  tokens, `hash_reservoir` seed `1729`. Strata were `unknown` 1,123,
  `web_cc` 482, `forums_qa` 272, `wiki` 76, `scientific` 32, and `code` 15.
- Ran Tier-1 census:
  `artifacts/stage1/census/smollm3_pretrain_dclm_2k_tier1_feature_rates.csv`.
  Source-aggregated DCLM rates per 1k tokens are `slop_lexicon` `0.354`,
  `rule_of_three_approx` `3.746`, `contrastive_negation` `0.487`,
  `stock_closers` `0.037`, `stock_openers` `0.037`, and pooled stock phrases
  `0.074`.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and summary with the DCLM source map. Current coverage is now `66.959%` of
  the extracted SmolLM3 recipe, with missing share `33.041%`. Covered-only
  weighted rates per 1k tokens are `slop_lexicon` `0.439`,
  `rule_of_three_approx` `4.882`, `contrastive_negation` `0.440`,
  `stock_closers` `0.039`, `stock_openers` `0.032`, and pooled stock phrases
  `0.071`.

## 2026-06-15 - FineWeb2 German and Spanish pretraining feature-rate coverage

- Sampled the next highest SmolLM3 recipe source after DCLM:
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_deu_2k.jsonl`, with
  manifests and summary. Source: `HuggingFaceFW/fineweb-2`, config
  `deu_Latn`, split `train`, 20,000 scanned rows, 2,000 retained rows,
  986,326 simple tokens, `hash_reservoir` seed `1729`. The sample was a
  single `unknown` stratum because FineWeb2 rows expose language metadata but
  no local source-domain stratum.
- Ran retained Tier-1 census:
  `artifacts/stage1/census/smollm3_pretrain_fw2_deu_2k_tier1_feature_rates.csv`.
  Source-aggregated FineWeb2 German rates per 1k tokens are `slop_lexicon`
  `0.047`, `rule_of_three_approx` `0.184`, `contrastive_negation` `0.006`,
  `stock_closers` `0.002`, `stock_openers` `0.000`, and pooled stock phrases
  `0.002`.
- Sampled the next FineWeb2 source:
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_spa_2k.jsonl`, with
  manifests and summary. Source: `HuggingFaceFW/fineweb-2`, config
  `spa_Latn`, split `train`, 20,000 scanned rows, 2,000 retained rows,
  1,271,133 simple tokens, `hash_reservoir` seed `1729`.
- Ran retained Tier-1 census:
  `artifacts/stage1/census/smollm3_pretrain_fw2_spa_2k_tier1_feature_rates.csv`.
  Source-aggregated FineWeb2 Spanish rates per 1k tokens are `slop_lexicon`
  `0.005`, `rule_of_three_approx` `0.038`, `contrastive_negation` `0.002`,
  `stock_closers` `0.000`, `stock_openers` `0.000`, and pooled stock phrases
  `0.000`.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and summary with source maps `smollm3_pretrain_fw2_deu_2k=fw2-deu` and
  `smollm3_pretrain_fw2_spa_2k=fw2-spa`. Current retained Tier-1 coverage is
  now `71.171%` of the extracted SmolLM3 recipe, with missing share `28.829%`.
  Covered-only weighted rates per 1k tokens are `slop_lexicon` `0.415`,
  `rule_of_three_approx` `4.599`, `contrastive_negation` `0.414`,
  `stock_closers` `0.037`, `stock_openers` `0.030`, and pooled stock phrases
  `0.067`.
- FineWeb2 German and Spanish are much lower on the retained English-oriented
  Tier-1 markers than DCLM or FineWeb-Edu, so adding them improves coverage
  while lowering the covered-only weighted baseline rates.

## 2026-06-15 - PES2O pretraining feature-rate coverage

- Probed `HuggingFaceTB/stack-edu` config `Python`, the largest remaining
  SmolLM3 recipe source by weight (`1.811%`). The dataset exposes blob
  metadata fields (`blob_id`, `language`, `repo_name`, `path`, score/license
  fields) but no source text field, so it cannot be measured by the current
  `slop-sample-corpus` text sampler without a blob-hydration path or another
  text-bearing mirror.
- Sampled the next directly measurable source:
  `artifacts/stage1/corpora/smollm3_pretrain_pes2o_2k.jsonl`, with manifests
  and summary. Source: `allenai/dolmino-mix-1124`, config `pes2o`, split
  `train`, 20,000 scanned rows, 2,000 retained rows, 399,454 simple tokens,
  `hash_reservoir` seed `1729`. Strata were `unknown` 1,267, `scientific`
  455, `web_cc` 210, `code` 57, `forums_qa` 9, and `wiki` 2.
- Ran retained Tier-1 census:
  `artifacts/stage1/census/smollm3_pretrain_pes2o_2k_tier1_feature_rates.csv`.
  Source-aggregated PES2O rates per 1k tokens are `slop_lexicon` `0.716`,
  `rule_of_three_approx` `5.520`, `contrastive_negation` `0.238`,
  `stock_closers` `0.060`, `stock_openers` `0.000`, and pooled stock phrases
  `0.060`.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and summary with source map `smollm3_pretrain_pes2o_2k=pes2o`. Current
  retained Tier-1 coverage is now `72.895%` of the extracted SmolLM3 recipe,
  with missing share `27.105%`. Covered-only weighted rates per 1k tokens are
  `slop_lexicon` `0.422`, `rule_of_three_approx` `4.621`,
  `contrastive_negation` `0.410`, `stock_closers` `0.037`, `stock_openers`
  `0.029`, and pooled stock phrases `0.067`.

## 2026-06-15 - FineWeb2 French pretraining feature-rate coverage

- Sampled the next directly measurable SmolLM3 recipe source:
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_fra_2k.jsonl`, with
  manifests and summary. Source: `HuggingFaceFW/fineweb-2`, config
  `fra_Latn`, split `train`, 20,000 scanned rows, 2,000 retained rows,
  1,152,313 simple tokens, `hash_reservoir` seed `1729`. The sample was a
  single `unknown` stratum because FineWeb2 rows expose language metadata but
  no local source-domain stratum.
- Ran retained Tier-1 census:
  `artifacts/stage1/census/smollm3_pretrain_fw2_fra_2k_tier1_feature_rates.csv`.
  Source-aggregated FineWeb2 French rates per 1k tokens are `slop_lexicon`
  `0.088`, `rule_of_three_approx` `0.100`, `contrastive_negation` `0.002`,
  `stock_closers` `0.001`, `stock_openers` `0.000`, and pooled stock phrases
  `0.001`.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and summary with source map `smollm3_pretrain_fw2_fra_2k=fw2-fra`. Current
  retained Tier-1 coverage is now `74.502%` of the extracted SmolLM3 recipe,
  with missing share `25.498%`. Covered-only weighted rates per 1k tokens are
  `slop_lexicon` `0.414`, `rule_of_three_approx` `4.524`,
  `contrastive_negation` `0.401`, `stock_closers` `0.037`, `stock_openers`
  `0.029`, and pooled stock phrases `0.065`.

## 2026-06-15 - FineMath pretraining feature-rate coverage

- Sampled the next directly measurable SmolLM3 recipe source:
  `artifacts/stage1/corpora/smollm3_pretrain_finemath_2k.jsonl`, with
  manifests and summary. Source: `HuggingFaceTB/finemath`, config
  `finemath-3plus`, split `train`, 20,000 scanned rows, 2,000 retained rows,
  2,302,447 simple tokens, `hash_reservoir` seed `1729`. The sample was a
  single `unknown` stratum.
- Ran retained Tier-1 census:
  `artifacts/stage1/census/smollm3_pretrain_finemath_2k_tier1_feature_rates.csv`.
  Source-aggregated FineMath rates per 1k tokens are `slop_lexicon` `0.150`,
  `rule_of_three_approx` `1.817`, `contrastive_negation` `0.181`,
  `stock_closers` `0.036`, `stock_openers` `0.065`, and pooled stock phrases
  `0.100`.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and summary with source map `smollm3_pretrain_finemath_2k=finemath`.
  Current retained Tier-1 coverage is now `75.912%` of the extracted SmolLM3
  recipe, with missing share `24.088%`. Covered-only weighted rates per 1k
  tokens are `slop_lexicon` `0.410`, `rule_of_three_approx` `4.473`,
  `contrastive_negation` `0.397`, `stock_closers` `0.037`, `stock_openers`
  `0.030`, and pooled stock phrases `0.066`.
- `stack-edu-Python` (`1.811%`) remains the largest unresolved source, but it
  needs code-text hydration because the public HF config exposes blob metadata
  only. The next directly sampleable source by recipe weight is FineWeb2
  Italian (`1.062%`); `stack-edu-Cpp` (`1.304%`) carries the same hydration
  caveat as Python.

## 2026-06-15 - FineWeb2 Italian pretraining feature-rate coverage

- Verified `HuggingFaceFW/fineweb-2` exposes the Italian config as
  `ita_Latn`.
- Sampled the next directly measurable SmolLM3 recipe source:
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_ita_2k.jsonl`, with
  manifests and summary. Source: `HuggingFaceFW/fineweb-2`, config
  `ita_Latn`, split `train`, 20,000 scanned rows, 2,000 retained rows,
  1,053,366 simple tokens, `hash_reservoir` seed `1729`. The sample was a
  single `unknown` stratum because FineWeb2 rows expose language metadata but
  no local source-domain stratum.
- Ran retained Tier-1 census:
  `artifacts/stage1/census/smollm3_pretrain_fw2_ita_2k_tier1_feature_rates.csv`.
  Source-aggregated FineWeb2 Italian rates per 1k tokens are `slop_lexicon`
  `0.036`, `rule_of_three_approx` `0.066`, `contrastive_negation` `0.003`,
  `stock_closers` `0.000`, `stock_openers` `0.000`, and pooled stock phrases
  `0.000`.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and summary with source map `smollm3_pretrain_fw2_ita_2k=fw2-ita`. Current
  retained Tier-1 coverage is now `76.974%` of the extracted SmolLM3 recipe,
  with missing share `23.026%`. Covered-only weighted rates per 1k tokens are
  `slop_lexicon` `0.404`, `rule_of_three_approx` `4.413`,
  `contrastive_negation` `0.391`, `stock_closers` `0.036`, `stock_openers`
  `0.029`, and pooled stock phrases `0.065`.
- `stack-edu-Python` (`1.811%`) and `stack-edu-Cpp` (`1.304%`) remain blocked
  on code-text hydration. The next directly sampleable recipe sources are
  other FineWeb2 language shards, `infiwebmath` (`0.903%`), and
  `finemath-4plus` (`0.606%`).

## 2026-06-15 - FineWeb2 Chinese pretraining feature-rate coverage

- Verified `HuggingFaceFW/fineweb-2` exposes the Chinese config as
  `cmn_Hani`; Russian is `rus_Cyrl`.
- Sampled the next directly measurable SmolLM3 recipe source:
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_cmn_2k.jsonl`, with
  manifests and summary. Source: `HuggingFaceFW/fineweb-2`, config
  `cmn_Hani`, split `train`, 20,000 scanned rows, 2,000 retained rows,
  269,299 simple tokens, `hash_reservoir` seed `1729`. The sample was a
  single `unknown` stratum because FineWeb2 rows expose language metadata but
  no local source-domain stratum.
- Ran retained Tier-1 census:
  `artifacts/stage1/census/smollm3_pretrain_fw2_cmn_2k_tier1_feature_rates.csv`.
  Source-aggregated FineWeb2 Chinese rates per 1k tokens are `slop_lexicon`
  `0.022`, `rule_of_three_approx` `0.152`, `contrastive_negation` `0.004`,
  `stock_closers` `0.000`, `stock_openers` `0.007`, and pooled stock phrases
  `0.007`.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and summary with source map `smollm3_pretrain_fw2_cmn_2k=fw2-cmn`. Current
  retained Tier-1 coverage is now `77.964%` of the extracted SmolLM3 recipe,
  with missing share `22.036%`. Covered-only weighted rates per 1k tokens are
  `slop_lexicon` `0.400`, `rule_of_three_approx` `4.358`,
  `contrastive_negation` `0.386`, `stock_closers` `0.036`, `stock_openers`
  `0.029`, and pooled stock phrases `0.064`.
- `stack-edu-Python` (`1.811%`) and `stack-edu-Cpp` (`1.304%`) remain blocked
  on code-text hydration. The next directly sampleable recipe sources are
  FineWeb2 Russian (`0.991%`), FineWeb2 Portuguese (`0.931%`), `infiwebmath`
  (`0.903%`), and `finemath-4plus` (`0.606%`).

## 2026-06-15 - Phase 2 conclusion report newcomer pass

- Expanded `docs/experiments/phase2_final_conclusion_report.md` with a reader
  map, plain-language Phase 2 questions, a glossary, a clearer bounded-scope
  explanation, and stronger bridges between Phase 1 corpus rates, Phase 2
  teacher-forced propensity, free-running generation, compounding, and
  Biber-lite register evidence.
- The report conclusions did not change: the retained OLMo/Dolci Phase 2
  package supports a selective DPO-stage `slop_lexicon` propensity peak and a
  strong `slop_lexicon` self-conditioning signal, but not broad monotonic slop
  growth from Base to SFT to DPO to Final/RLVR.

## 2026-06-15 - FineWeb2 Russian pretraining feature-rate coverage

- Sampled the next directly measurable SmolLM3 recipe source:
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_rus_2k.jsonl`, with
  manifests and summary. Source: `HuggingFaceFW/fineweb-2`, config
  `rus_Cyrl`, split `train`, 20,000 scanned rows, 2,000 retained rows,
  1,587,847 simple tokens, `hash_reservoir` seed `1729`. The sample was a
  single `unknown` stratum because FineWeb2 rows expose language metadata but
  no local source-domain stratum.
- Ran retained Tier-1 census:
  `artifacts/stage1/census/smollm3_pretrain_fw2_rus_2k_tier1_feature_rates.csv`.
  Source-aggregated FineWeb2 Russian rates per 1k tokens are `slop_lexicon`
  `0.005`, `rule_of_three_approx` `0.052`, `contrastive_negation` `0.003`,
  `stock_closers` `0.000`, `stock_openers` `0.000`, and pooled stock phrases
  `0.000`.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and summary with source map `smollm3_pretrain_fw2_rus_2k=fw2-rus`. Current
  retained Tier-1 coverage is now `78.955%` of the extracted SmolLM3 recipe,
  with missing share `21.045%`. Covered-only weighted rates per 1k tokens are
  `slop_lexicon` `0.395`, `rule_of_three_approx` `4.304`,
  `contrastive_negation` `0.382`, `stock_closers` `0.035`, `stock_openers`
  `0.028`, and pooled stock phrases `0.064`.
- `stack-edu-Python` (`1.811%`) and `stack-edu-Cpp` (`1.304%`) remain blocked
  on code-text hydration. The next directly sampleable recipe sources are
  FineWeb2 Portuguese (`0.931%`), `infiwebmath` (`0.903%`),
  `finemath-4plus` (`0.606%`), and smaller FineWeb2 language shards.

## 2026-06-15 - FineWeb2 Portuguese pretraining feature-rate coverage

- Verified `HuggingFaceFW/fineweb-2` exposes the Portuguese config as
  `por_Latn`.
- Sampled the next directly measurable SmolLM3 recipe source:
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_por_2k.jsonl`, with
  manifests and summary. Source: `HuggingFaceFW/fineweb-2`, config
  `por_Latn`, split `train`, 20,000 scanned rows, 2,000 retained rows,
  1,344,559 simple tokens, `hash_reservoir` seed `1729`. The sample was a
  single `unknown` stratum because FineWeb2 rows expose language metadata but
  no local source-domain stratum.
- Ran retained Tier-1 census:
  `artifacts/stage1/census/smollm3_pretrain_fw2_por_2k_tier1_feature_rates.csv`.
  Source-aggregated FineWeb2 Portuguese rates per 1k tokens are
  `slop_lexicon` `0.027`, `rule_of_three_approx` `0.162`,
  `contrastive_negation` `0.013`, `stock_closers` `0.000`,
  `stock_openers` `0.000`, and pooled stock phrases `0.000`.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and summary with source map `smollm3_pretrain_fw2_por_2k=fw2-por`. Current
  retained Tier-1 coverage is now `79.886%` of the extracted SmolLM3 recipe,
  with missing share `20.114%`. Covered-only weighted rates per 1k tokens are
  `slop_lexicon` `0.390`, `rule_of_three_approx` `4.256`,
  `contrastive_negation` `0.377`, `stock_closers` `0.035`, `stock_openers`
  `0.028`, and pooled stock phrases `0.063`.
- `stack-edu-Python` (`1.811%`) and `stack-edu-Cpp` (`1.304%`) remain blocked
  on code-text hydration. The next directly sampleable recipe sources are
  `infiwebmath` (`0.903%`), `finemath-4plus` (`0.606%`), and smaller
  FineWeb2 language shards.

## 2026-06-15 - FineMath-4plus pretraining feature-rate coverage

- Checked the public `HuggingFaceTB/finemath` configs. It exposes
  `finemath-3plus`, `finemath-4plus`, `infiwebmath-3plus`, and
  `infiwebmath-4plus`, but not an exact `infiwebmath` config for the extracted
  `infiwebmath` recipe source row (`0.903%`). I therefore sampled the next
  exact public recipe source, `finemath-4plus` (`0.606%`).
- Sampled `artifacts/stage1/corpora/smollm3_pretrain_finemath_4plus_2k.jsonl`,
  with manifests and summary. Source: `HuggingFaceTB/finemath`, config
  `finemath-4plus`, split `train`, 20,000 scanned rows, 2,000 retained rows,
  1,774,598 simple tokens, `hash_reservoir` seed `1729`.
- Ran retained Tier-1 census:
  `artifacts/stage1/census/smollm3_pretrain_finemath_4plus_2k_tier1_feature_rates.csv`.
  Source-aggregated FineMath-4plus rates per 1k tokens are `slop_lexicon`
  `0.170`, `rule_of_three_approx` `1.909`, `contrastive_negation` `0.159`,
  `stock_closers` `0.025`, `stock_openers` `0.078`, and pooled stock phrases
  `0.104`.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and summary with source map
  `smollm3_pretrain_finemath_4plus_2k=finemath-4plus`. Current retained
  Tier-1 coverage is now `80.492%` of the extracted SmolLM3 recipe, with
  missing share `19.508%`. Covered-only weighted rates per 1k tokens are
  `slop_lexicon` `0.389`, `rule_of_three_approx` `4.238`,
  `contrastive_negation` `0.376`, `stock_closers` `0.035`, `stock_openers`
  `0.029`, and pooled stock phrases `0.063`.
- `stack-edu-Python` (`1.811%`) and `stack-edu-Cpp` (`1.304%`) remain blocked
  on code-text hydration. The next exact public math source is
  `infiwebmath-4plus` (`0.383%`), and the next smaller web sources are
  FineWeb2 language shards.

## 2026-06-16 - InfiWebMath-4plus pretraining feature-rate coverage

- Sampled
  `artifacts/stage1/corpora/smollm3_pretrain_infiwebmath_4plus_2k.jsonl`,
  with manifests and summary. Source: `HuggingFaceTB/finemath`, config
  `infiwebmath-4plus`, split `train`, 20,000 scanned rows, 2,000 retained
  rows, 1,868,874 simple tokens, `hash_reservoir` seed `1729`. Retained
  strata are `web_cc` (1,516 rows), `forums_qa` (308), `wiki` (86), `code`
  (47), and `scientific` (43).
- Ran retained Tier-1 census:
  `artifacts/stage1/census/smollm3_pretrain_infiwebmath_4plus_2k_tier1_feature_rates.csv`.
  Token-weighted source rates per 1k tokens are `slop_lexicon` `0.177`,
  `rule_of_three_approx` `1.896`, `contrastive_negation` `0.158`,
  `stock_closers` `0.029`, `stock_openers` `0.073`, and pooled stock phrases
  `0.102`.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and summary with source map
  `smollm3_pretrain_infiwebmath_4plus_2k=infiwebmath-4plus`. Current retained
  Tier-1 coverage is now `80.875%` of the extracted SmolLM3 recipe, with
  missing share `19.125%`. Covered-only weighted rates per 1k tokens are
  `slop_lexicon` `0.388`, `rule_of_three_approx` `4.227`,
  `contrastive_negation` `0.375`, `stock_closers` `0.035`, `stock_openers`
  `0.029`, and pooled stock phrases `0.063`.
- Added a regression test so multi-stratum source rows are token-weighted when
  they map to one recipe source. The larger `infiwebmath` recipe row
  (`0.903%`) remains unmapped to an exact public `HuggingFaceTB/finemath`
  config; `infiwebmath-4plus` covers only its exact `0.383%` recipe source
  row.

## 2026-06-16 - Weighted SmolLM3 baseline downstream refresh

- Updated `slop-assemble-amplification-spectrum` with a
  `--weighted-pretrain-baseline` input. The weighted covered-only rate now
  overrides the aggregate `pretrain_per_1k_tokens` field, while raw
  source-specific pretraining columns remain in the spectrum for audit.
- Fixed source-specific pretraining columns to token-weight repeated rows from
  the same source before writing values. This avoids reporting a single last
  stratum for multi-stratum samples such as `infiwebmath-4plus`.
- Regenerated the current preferred bounded SmolLM3 baseline data-rate
  spectrum, classifier, and OLMo-vs-SmolLM3 comparison:
  `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3.csv`,
  `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3.csv`,
  and
  `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_baselines_data_rates_tf_generation_compounding_slop_neutral_rule3_summary.md`.
- The regenerated spectrum uses the `80.875%` covered-recipe weighted proxy as
  the aggregate pretrain baseline. The OLMo-vs-SmolLM3 AF correlation remains
  Spearman `0.762` and Pearson `0.978`, because the AF layer did not change.

## 2026-06-16 - MegaMath pretraining feature-rate coverage

- Probed `LLM360/MegaMath` and confirmed the exact public source directories
  needed for the SmolLM3 recipe rows. The default dataset loader exposes
  metadata-like rows for the first data directory, so `slop-sample-corpus` now
  accepts `--hf-data-files` and can target text-bearing HF parquet
  subdirectories through the generic `parquet` loader.
- Sampled exact `LLM360/MegaMath` source paths:
  `artifacts/stage1/corpora/smollm3_pretrain_megamath_text_code_block_2k.jsonl`
  from `megamath-text-code-block/*.parquet` and
  `artifacts/stage1/corpora/smollm3_pretrain_megamath_web_pro_2k.jsonl` from
  `megamath-web-pro/*.parquet`. Both use `hash_reservoir`, seed `1729`, and a
  2,000-row target. The text-code sample scanned 19,150 rows and retained
  844,480 simple tokens; web-pro scanned 20,000 rows and retained 1,182,841
  simple tokens.
- Ran retained Tier-1 census:
  `artifacts/stage1/census/smollm3_pretrain_megamath_text_code_block_2k_tier1_feature_rates.csv`
  and
  `artifacts/stage1/census/smollm3_pretrain_megamath_web_pro_2k_tier1_feature_rates.csv`.
  Source-aggregated MegaMath text-code rates per 1k tokens are
  `slop_lexicon` `0.124`, `rule_of_three_approx` `0.817`,
  `contrastive_negation` `0.109`, `stock_closers` `0.033`,
  `stock_openers` `0.000`, and pooled stock phrases `0.033`. MegaMath
  web-pro rates are `slop_lexicon` `0.332`, `rule_of_three_approx` `4.451`,
  `contrastive_negation` `0.099`, `stock_closers` `0.085`,
  `stock_openers` `0.008`, and pooled stock phrases `0.093`.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and summary with source maps
  `smollm3_pretrain_megamath_text_code_block_2k=megamath-text-code-block`
  and `smollm3_pretrain_megamath_web_pro_2k=megamath-web-pro`. Current
  retained Tier-1 coverage is now `82.259%` of the extracted SmolLM3 recipe,
  with missing share `17.741%`. Covered-only weighted rates per 1k tokens are
  `slop_lexicon` `0.384`, `rule_of_three_approx` `4.191`,
  `contrastive_negation` `0.370`, `stock_closers` `0.035`,
  `stock_openers` `0.028`, and pooled stock phrases `0.063`.
- Regenerated the preferred bounded SmolLM3 baseline data-rate spectrum,
  classifier, and OLMo-vs-SmolLM3 comparison. The OLMo-vs-SmolLM3 AF
  correlation remains Spearman `0.762` and Pearson `0.978`, because the
  teacher-forced AF layer did not change.

## 2026-06-16 - Additional FineWeb2 pretraining feature-rate coverage

- Sampled seven additional directly measurable FineWeb2 language shards for
  the SmolLM3 recipe-weighted pretraining baseline. Each sample used
  `hash_reservoir`, seed `1729`, a 2,000-row target, and a 20,000-row scan cap:
  `fw2-fas` (`fas_Arab`, 1,674,442 simple tokens), `fw2-hin` (`hin_Deva`,
  1,781,688 tokens), `fw2-jpn` (`jpn_Jpan`, 258,131 tokens), `fw2-kor`
  (`kor_Hang`, 709,095 tokens), `fw2-tha` (`tha_Thai`, 1,180,666 tokens),
  `fw2-vie` (`vie_Latn`, 1,984,302 tokens), and `fw2-ell` (`ell_Grek`,
  1,054,624 tokens).
- Ran retained Tier-1 census for all seven new samples:
  `artifacts/stage1/census/smollm3_pretrain_fw2_fas_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_fw2_hin_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_fw2_jpn_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_fw2_kor_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_fw2_tha_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_fw2_vie_2k_tier1_feature_rates.csv`,
  and
  `artifacts/stage1/census/smollm3_pretrain_fw2_ell_2k_tier1_feature_rates.csv`.
- Source rates per 1k tokens are low for the retained English-oriented
  detectors. `slop_lexicon`: `fw2-fas` `0.002`, `fw2-hin` `0.004`,
  `fw2-jpn` `0.008`, `fw2-kor` `0.024`, `fw2-tha` `0.023`, `fw2-vie`
  `0.005`, `fw2-ell` `0.018`. `rule_of_three_approx`: `fw2-fas` `0.045`,
  `fw2-hin` `0.053`, `fw2-jpn` `0.256`, `fw2-kor` `0.118`, `fw2-tha`
  `0.042`, `fw2-vie` `0.033`, `fw2-ell` `0.117`.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and summary with new source maps for `fw2-fas`, `fw2-hin`, `fw2-jpn`,
  `fw2-kor`, `fw2-tha`, `fw2-vie`, and `fw2-ell`. Current retained Tier-1
  coverage is now `84.350%` of the extracted SmolLM3 recipe, with missing
  share `15.650%`. Covered-only weighted rates per 1k tokens are
  `slop_lexicon` `0.375`, `rule_of_three_approx` `4.090`,
  `contrastive_negation` `0.361`, `stock_closers` `0.034`,
  `stock_openers` `0.028`, and pooled stock phrases `0.062`.
- Regenerated the preferred bounded SmolLM3 baseline data-rate spectrum,
  classifier, and OLMo-vs-SmolLM3 comparison:
  `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3.csv`,
  `artifacts/phase3/analysis/smollm3_no_think_feature_classification_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3.csv`,
  and
  `artifacts/phase3/analysis/olmo3_vs_smollm3_no_think_512prompt_baselines_data_rates_tf_generation_compounding_slop_neutral_rule3_summary.md`.
  The OLMo-vs-SmolLM3 AF correlation remains Spearman `0.762` and Pearson
  `0.978`, because the teacher-forced AF layer did not change.

## 2026-06-16 - Stack-Edu mirror coverage for SmolLM3 weighted baseline

- Sampled thirteen text-bearing Stack-Edu language mirrors for SmolLM3
  pretraining-source coverage: Python, Cpp, Java, JavaScript, C, PHP,
  TypeScript, SQL, Go, Ruby, Rust, Shell, and Swift. Each sample used
  `hash_reservoir`, seed `1729`, a 2,000-row target, and a 20,000-row scan cap
  over the mirror `text` field. The canonical `HuggingFaceTB/stack-edu`
  configs expose blob metadata rather than hydrated code text; Python and Cpp
  mirror probes matched canonical first blob IDs, so these are documented as
  mirror-hydrated Stack-Edu source samples.
- Ran retained Tier-1 census for all thirteen Stack-Edu samples:
  `artifacts/stage1/census/smollm3_pretrain_stack_edu_python_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_stack_edu_cpp_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_stack_edu_java_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_stack_edu_javascript_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_stack_edu_c_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_stack_edu_php_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_stack_edu_typescript_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_stack_edu_sql_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_stack_edu_go_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_stack_edu_ruby_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_stack_edu_rust_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_stack_edu_shell_2k_tier1_feature_rates.csv`,
  and
  `artifacts/stage1/census/smollm3_pretrain_stack_edu_swift_2k_tier1_feature_rates.csv`.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and summary with new source maps for the thirteen Stack-Edu language
  sources. Current retained Tier-1 coverage is now `91.161%` of the extracted
  recipe, with missing share `8.839%`. Covered-only weighted rates per 1k
  tokens are `slop_lexicon` `0.357`, `rule_of_three_approx` `3.805`,
  `contrastive_negation` `0.335`, `stock_closers` `0.032`, `stock_openers`
  `0.026`, and pooled stock phrases `0.058`.
- Regenerated the preferred bounded SmolLM3 baseline data-rate spectrum,
  classifier, and OLMo-vs-SmolLM3 comparison. The OLMo-vs-SmolLM3 AF
  correlation remains Spearman `0.762` and Pearson `0.978`, because this pass
  changed the data-rate baseline rather than the teacher-forced AF layer.

2026-06-16:

- Probed `HuggingFaceTB/issues-kaggle-notebooks` and confirmed two public
  text-bearing configs: `issues` and `kaggle`, both with split `train`.
- Added bounded exact source samples:
  `artifacts/stage1/corpora/smollm3_pretrain_github_issues_2k.jsonl`
  retained 2,000 rows and 565,111 simple tokens from a 20,000-row
  hash-reservoir scan; `artifacts/stage1/corpora/smollm3_pretrain_kaggle_2k.jsonl`
  retained 2,000 rows and 2,259,555 simple tokens from the same scan cap.
- Ran Tier-1 censi:
  `artifacts/stage1/census/smollm3_pretrain_github_issues_2k_tier1_feature_rates.csv`
  and
  `artifacts/stage1/census/smollm3_pretrain_kaggle_2k_tier1_feature_rates.csv`.
  GitHub issues rates per 1k tokens: `contrastive_negation` `0.324`,
  `rule_of_three_approx` `0.743`, `slop_lexicon` `0.251`,
  `stock_openers` `0.071`, `stock_closers` `0.016`, pooled stock phrases
  `0.087`. Kaggle rates per 1k tokens: `contrastive_negation` `0.050`,
  `rule_of_three_approx` `0.993`, `slop_lexicon` `0.105`,
  `stock_openers` `0.003`, `stock_closers` `0.074`, pooled stock phrases
  `0.077`.
- Regenerated the coverage-aware weighted pretraining baseline with exact
  maps `smollm3_pretrain_github_issues_2k=github-issues` and
  `smollm3_pretrain_kaggle_2k=kaggle`. Retained Tier-1 recipe coverage is now
  `91.553%` through 38 mapped source samples, with missing share `8.447%`.
  Covered-only weighted rates per 1k tokens are `slop_lexicon` `0.356`,
  `rule_of_three_approx` `3.792`, `contrastive_negation` `0.335`,
  `stock_closers` `0.032`, `stock_openers` `0.026`, and pooled stock phrases
  `0.058`.
- Regenerated the preferred bounded SmolLM3 baseline data-rate spectrum,
  classifier, and OLMo-vs-SmolLM3 comparison after the weighted-baseline
  refresh. The SmolLM3 spectrum now carries source-specific
  `pretrain_smollm3_pretrain_github_issues_2k_per_1k_tokens` and
  `pretrain_smollm3_pretrain_kaggle_2k_per_1k_tokens` columns. The
  OLMo-vs-SmolLM3 AF correlation remains Spearman `0.762` and Pearson
  `0.978`, because the teacher-forced AF layer did not change.

## 2026-06-16 - Wiki, InfiWebMath-3plus, and OpenMathInstruct-2 baseline coverage

- Probed and sampled three additional exact public SmolLM3 recipe sources for
  the retained Tier-1 weighted pretraining baseline:
  `wiki`, `infiwebmath-3plus`, and `openmathinstruct-2`.
- Added bounded source samples with `hash_reservoir`, seed `1729`, a
  2,000-row target, and a 20,000-row scan cap:
  `artifacts/stage1/corpora/smollm3_pretrain_wiki_2k.jsonl` from
  `wikimedia/wikipedia`, config `20231101.en`, text field `text`
  (`2,183,945` simple tokens);
  `artifacts/stage1/corpora/smollm3_pretrain_infiwebmath_3plus_2k.jsonl`
  from `HuggingFaceTB/finemath`, config `infiwebmath-3plus`, text field
  `text` (`1,990,392` simple tokens); and
  `artifacts/stage1/corpora/smollm3_pretrain_openmathinstruct_2_solution_2k.jsonl`
  from `nvidia/OpenMathInstruct-2`, text field `generated_solution`
  (`319,125` simple tokens). The OpenMathInstruct measurement is a bounded
  solution-text sample, not a reconstruction of combined problem-plus-solution
  training records.
- Ran retained Tier-1 censi:
  `artifacts/stage1/census/smollm3_pretrain_wiki_2k_tier1_feature_rates.csv`,
  `artifacts/stage1/census/smollm3_pretrain_infiwebmath_3plus_2k_tier1_feature_rates.csv`,
  and
  `artifacts/stage1/census/smollm3_pretrain_openmathinstruct_2_solution_2k_tier1_feature_rates.csv`.
  Wiki rates per 1k tokens: `contrastive_negation` `0.219`,
  `rule_of_three_approx` `6.121`, `slop_lexicon` `0.356`,
  `stock_openers` `0.002`, `stock_closers` `0.019`, pooled stock phrases
  `0.021`. OpenMathInstruct solution-text rates per 1k tokens:
  `contrastive_negation` `0.066`, `rule_of_three_approx` `0.417`,
  `slop_lexicon` `0.034`, `stock_openers` `0.019`, `stock_closers` `0.038`,
  pooled stock phrases `0.056`. InfiWebMath-3plus rates are token-weighted
  across inferred `web_cc`, `forums_qa`, `wiki`, `code`, and `scientific`
  strata by the weighted-baseline assembler.
- Regenerated the coverage-aware weighted pretraining baseline with source
  maps `smollm3_pretrain_wiki_2k=wiki`,
  `smollm3_pretrain_infiwebmath_3plus_2k=infiwebmath-3plus`, and
  `smollm3_pretrain_openmathinstruct_2_solution_2k=openmathinstruct-2`.
  Retained Tier-1 recipe coverage is now `91.720%` through 41 mapped source
  samples, with missing share `8.280%`. Covered-only weighted rates per 1k
  tokens are `slop_lexicon` `0.356`, `rule_of_three_approx` `3.792`,
  `contrastive_negation` `0.335`, `stock_closers` `0.032`,
  `stock_openers` `0.026`, and pooled stock phrases `0.058`.
- Regenerated the preferred bounded SmolLM3 baseline data-rate spectrum,
  classifier, and OLMo-vs-SmolLM3 comparison after the weighted-baseline
  refresh. The SmolLM3 spectrum now carries source-specific
  `pretrain_smollm3_pretrain_wiki_2k_per_1k_tokens`,
  `pretrain_smollm3_pretrain_infiwebmath_3plus_2k_per_1k_tokens`, and
  `pretrain_smollm3_pretrain_openmathinstruct_2_solution_2k_per_1k_tokens`
  columns. The OLMo-vs-SmolLM3 AF correlation remains Spearman `0.762` and
  Pearson `0.978`, because the teacher-forced AF layer did not change.

## 2026-06-16 - OpenMathReasoning-4k and Natural Reasoning baseline coverage

- Added one generic corpus-reader extraction path for response-style mappings:
  list elements with a `response` key now coerce as text, alongside existing
  `content`, `text`, and `value` keys. This supports bounded sampling of
  `facebook/natural_reasoning` through `--text-field responses`.
- Sampled two additional exact public SmolLM3 recipe sources with
  `hash_reservoir`, seed `1729`, a 2,000-row target, and a 20,000-row scan
  cap:
  `artifacts/stage1/corpora/smollm3_pretrain_openmathreasoning_4k_2k.jsonl`
  from `LLMcompe-Team-Watanabe/math_OpenMathReasoning_preprocess_4k-8k`, text
  field `answer` (`4,401,574` simple tokens), and
  `artifacts/stage1/corpora/smollm3_pretrain_natural_reasoning_responses_2k.jsonl`
  from `facebook/natural_reasoning`, text field `responses` (`1,000,263`
  simple tokens). The OpenMathReasoning sample includes answer text as
  published, including `<think>` reasoning markup where present; the Natural
  Reasoning sample joins `responses[].response` strings per row.
- Ran retained Tier-1 censi:
  `artifacts/stage1/census/smollm3_pretrain_openmathreasoning_4k_2k_tier1_feature_rates.csv`
  and
  `artifacts/stage1/census/smollm3_pretrain_natural_reasoning_responses_2k_tier1_feature_rates.csv`.
  OpenMathReasoning-4k rates per 1k tokens: `contrastive_negation` `0.080`,
  `rule_of_three_approx` `0.814`, `slop_lexicon` `0.025`,
  `stock_openers` `0.005`, `stock_closers` `0.020`, pooled stock phrases
  `0.025`, `list_header_bold_lead_in` `6.260`, and `punctuation_rhythm`
  `28.077`. Natural Reasoning joined-response rates per 1k tokens:
  `contrastive_negation` `0.198`, `rule_of_three_approx` `3.977`,
  `slop_lexicon` `0.524`, `stock_openers` `0.023`, `stock_closers` `0.108`,
  pooled stock phrases `0.131`, `list_header_bold_lead_in` `17.822`, and
  `punctuation_rhythm` `23.859`.
- Regenerated the coverage-aware weighted pretraining baseline with source
  maps `smollm3_pretrain_openmathreasoning_4k_2k=openmathreasoning-4k` and
  `smollm3_pretrain_natural_reasoning_responses_2k=natural_reasoning`.
  Retained Tier-1 recipe coverage is now `91.794%` through 43 mapped source
  samples, with missing share `8.206%`. Covered-only weighted rates per 1k
  tokens are `slop_lexicon` `0.356`, `rule_of_three_approx` `3.790`,
  `contrastive_negation` `0.334`, `stock_closers` `0.032`,
  `stock_openers` `0.026`, and pooled stock phrases `0.058`.
- Regenerated the preferred bounded SmolLM3 baseline data-rate spectrum,
  classifier, and OLMo-vs-SmolLM3 comparison. The SmolLM3 spectrum now carries
  source-specific
  `pretrain_smollm3_pretrain_openmathreasoning_4k_2k_per_1k_tokens` and
  `pretrain_smollm3_pretrain_natural_reasoning_responses_2k_per_1k_tokens`
  columns. The OLMo-vs-SmolLM3 AF correlation remains Spearman `0.762` and
  Pearson `0.978`, because this refresh changed data-rate baselines rather
  than teacher-forced AF values.

## 2026-06-16 - Multilingual Wikipedia baseline coverage refresh

- Probed the remaining SmolLM3 source gaps. Public `HuggingFaceTB/stack-edu`
  configs expose blob metadata only for the missing CSharp/Markdown/HTML-style
  rows; the tested BigCode code datasets are gated; and the public
  `verify-ppt` Jupyter datasets do not expose supported data files through the
  local dataset loader. The next exact public source chosen for bounded
  coverage was `multilingual_wiki`.
- Sampled three `wikimedia/wikipedia` non-English configs with
  `hash_reservoir`, seed `1729`, 1,000 retained rows per language, and a
  10,000-row scan cap per language:
  `smollm3_pretrain_multilingual_wiki_de_1k` (`2,653,575` simple tokens),
  `smollm3_pretrain_multilingual_wiki_es_1k` (`2,407,799` simple tokens),
  and `smollm3_pretrain_multilingual_wiki_fr_1k` (`2,672,780` simple tokens).
  This is a bounded three-language proxy for the `multilingual_wiki` recipe
  source, not a full multilingual mixture reconstruction.
- Ran retained Tier-1 censi for all three samples and regenerated the
  coverage-aware weighted pretraining baseline with all three source maps
  pointing to `multilingual_wiki`. Retained Tier-1 recipe coverage is now
  `91.969%` through 47 mapped source samples, with missing share `8.031%`.
  Covered-only weighted rates per 1k tokens are `slop_lexicon` `0.358`,
  `rule_of_three_approx` `3.789`, `contrastive_negation` `0.334`,
  `stock_closers` `0.032`, `stock_openers` `0.026`, and pooled stock phrases
  `0.058`. `list_header_bold_lead_in` and `punctuation_rhythm` coverage rose
  to `31.722%`.
- Regenerated the preferred bounded SmolLM3 baseline data-rate spectrum,
  classifier, and OLMo-vs-SmolLM3 comparison. The SmolLM3 spectrum now carries
  source-specific multilingual wiki columns for the German, Spanish, and
  French bounded samples. The OLMo-vs-SmolLM3 AF correlation remains Spearman
  `0.762` and Pearson `0.978`, because this refresh changed data-rate
  baselines rather than teacher-forced AF values.

## 2026-06-16 - MegaMath QA Qwen, Cosmopedia2, and OpenCodeReasoning coverage refresh

- Added three additional bounded public SmolLM3 recipe-source samples:
  `smollm3_pretrain_megamath_qa_qwen_2k` from `LLM360/MegaMath`
  `megamath-qa/qwen-2.5/*.parquet` (`298,474` simple tokens),
  `smollm3_pretrain_cosmopedia2_20k_2k` from
  `HuggingFaceTB/cosmopedia-20k` (`1,057,350` simple tokens), and
  `smollm3_pretrain_open_codereasoning_4k_output_2k` from
  `nvidia/OpenCodeReasoning`, config/split `split_0`, text field `output`
  (`9,993,355` simple tokens). Cosmopedia2 is represented by the public 20k
  sample rather than a full corpus reconstruction; OpenCodeReasoning measures
  output text rather than exact 4k tokenizer packing.
- Regenerated FineWeb-Edu and StackExchange censi from their existing local
  corpus JSONL files because the current worktree no longer had their census
  CSVs, then regenerated the weighted pretraining baseline with source maps
  `megamath-qa-qwen`, `cosmopedia2`, and `open-codereasoning-4k`.
- Retained Tier-1 recipe coverage is now `91.882%` through 46 mapped source
  samples, with missing share `8.118%`. Covered-only weighted rates per 1k
  tokens are `slop_lexicon` `0.358`, `rule_of_three_approx` `3.793`,
  `contrastive_negation` `0.335`, `stock_closers` `0.032`,
  `stock_openers` `0.026`, and pooled stock phrases `0.058`.
  `list_header_bold_lead_in` and `punctuation_rhythm` coverage rose to
  `31.635%`.
- Regenerated the preferred bounded SmolLM3 baseline data-rate spectrum,
  classifier, and OLMo-vs-SmolLM3 comparison. The SmolLM3 spectrum now carries
  source-specific `pretrain_smollm3_pretrain_megamath_qa_qwen_2k_per_1k_tokens`,
  `pretrain_smollm3_pretrain_cosmopedia2_20k_2k_per_1k_tokens`, and
  `pretrain_smollm3_pretrain_open_codereasoning_4k_output_2k_per_1k_tokens`
  columns. The OLMo-vs-SmolLM3 AF correlation remains Spearman `0.762` and
  Pearson `0.978`, because this refresh changed data-rate baselines rather
  than teacher-forced AF values.

## 2026-06-16 - Explicit SmolLM3 source-proxy weighted-baseline refresh

- Added `--source-proxy MEASURED_RECIPE_SOURCE=PROXY_RECIPE_SOURCE` to
  `slop-assemble-weighted-pretrain-baseline`. The weighted baseline now
  reports direct and proxy coverage separately via
  `exact_covered_recipe_share`, `proxy_covered_recipe_share`,
  `exact_matched_source_count`, `proxy_source_count`, and
  `proxy_recipe_sources`, while preserving the combined
  `covered_recipe_share` used by downstream spectra.
- Regenerated
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  with explicit same-family proxies for `infiwebmath` from
  `infiwebmath-4plus` and inaccessible Stack-Edu `real`/`real-shuffled`
  sibling recipe rows from the measured same-language Stack-Edu samples.
  Retained Tier-1 direct coverage remains `91.969%` through 47 mapped source
  samples; explicit proxies add `4.601%`, bringing total covered recipe share
  to `96.571%` and reducing missing recipe share to `3.429%`.
- Proxy-aware covered-only weighted rates per 1k tokens are now
  `slop_lexicon` `0.347`, `rule_of_three_approx` `3.638`,
  `contrastive_negation` `0.320`, `stock_closers` `0.031`,
  `stock_openers` `0.025`, and pooled stock phrases `0.056`. Structural
  exploratory features remain at `31.722%` direct coverage because the proxy
  source censi cover retained Tier-1 rows only.
- Propagated the exact/proxy coverage split into
  `slop-assemble-amplification-spectrum` and regenerated the preferred
  bounded SmolLM3 baseline data-rate spectrum, classifier, and
  OLMo-vs-SmolLM3 comparison. The OLMo-vs-SmolLM3 AF correlation remains
  Spearman `0.762` and Pearson `0.978` because this refresh only changed the
  pretraining data-rate layer, not teacher-forced AF values.

## 2026-06-16 - SmolLM3 Biber-lite style signature and non-pretrain Phase 3 pivot

- Stopped treating the remaining `3.429%` proxy-aware SmolLM3 pretraining
  recipe-source gap as the active Phase 3 blocker. The current bounded
  analysis keeps the `96.571%` proxy-aware pretraining baseline as sufficient
  for interpretation and shifts remaining work to teacher-forced support,
  generation grids, and reporting.
- Ran a SmolLM3 Biber-lite corpus census over the local SmolTalk2 no_think
  SFT target pool plus the 10k Tulu no_think preference-pair sample:
  `artifacts/stage1/census/smollm3_smoltalk2_sft2260_pref10k_biber_lite_feature_rates.csv`
  and
  `artifacts/stage1/census/smollm3_smoltalk2_pref10k_biber_lite_pair_deltas.csv`.
- Compared Biber-lite generated-output rates for the four 512-prompt
  SmolLM3 no_think `t=1.0` generation caches against that corpus baseline:
  `artifacts/phase2/analysis/smollm3_biber_lite_generation_vs_corpus_512prompt_8comp_t1_chat.csv`.
- Built the combined SmolLM3 output-style signature from Tier-1 generation
  rates, compounding, and Biber-lite register rates:
  `artifacts/phase3/analysis/smollm3_no_think_style_signature_512prompt_8comp_t1_chat.csv`.
  The distance table has 43 shared coordinates; Final/RLVR is closest to
  APO/DPO (`6.398`), then SFT (`31.315`), and far from Base (`454.800`).
- Wrote a feature-support audit separating raw target occurrence from
  teacher-forced opportunity support:
  `artifacts/phase3/analysis/smollm3_no_think_phase3_feature_support_audit_512prompt_and_full_sft.md`.
  In the actual 512-prompt package, `slop_lexicon`,
  `rule_of_three_approx`, and neutral controls have teacher-forced support;
  `contrastive_negation` and stock opener/closer families remain
  generation-side only under the current opportunity contract.
- Added a runnable SmolLM3 no_think production-shape generation plan for the
  available 512-prompt package across four stages, 8 completions, and
  temperatures `0.0`, `0.7`, and `1.0`:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`.
- Added `docs/experiments/phase3_completion_audit.md` as the strict
  requirement-by-requirement completion audit for `EXPERIMENTS.md` Phase 3.
  It records the current state as a bounded Tier-1 Phase 3 slice, not full
  completion, and identifies the real remaining blockers as unrun generation
  grids, post-grid compounding/style rebuilds, and broader teacher-forced
  support only where opportunity contracts have nonzero references.
- Dry-ran the guarded launcher against the SmolLM3 3-temperature plan and
  recorded the next executable shard selection without launching GPU work:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_next_shard.json`.
  The selected shard is base, temperature `0.0`, 4,096 expected generations,
  estimated at `3.273` A100-hours.
- Added `docs/experiments/phase3_production_runbook.md` with the guarded
  launch, status, plan-refresh, and rebuild commands needed after the
  remaining generation shards complete. Also generated
  `artifacts/phase3/analysis/phase3_generation_plan_status_summary.csv` and
  `.md` from the current OLMo and SmolLM3 production generation plans.
- Added `slop-summarize-phase3-generation-plans` so that combined Phase 3
  generation-plan status summaries can be regenerated from plan CSVs without
  ad hoc scripting. The current status summary was regenerated with this CLI.

## 2026-06-16 - SmolLM3 Phase 3 base/t0 production shard launched

- Per the decision to stop prioritizing the remaining SmolLM3 pretraining
  source-coverage gap, launched the next actual Phase 3 execution step: the
  SmolLM3 no_think production-shape 512-prompt, 8-completion base checkpoint
  shard at temperature `0.0`.
- The guarded launcher wrote
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_next_shard.json`
  with `executed=true`, `detached=true`, expected generations `4,096`, and
  estimated cost `3.273` A100-hours.
- Detached log:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_next_shard.log`.
  The job had loaded model weights at launch verification time.
- This launch does not make Phase 3 complete. The shard must finish, the plan
  must be refreshed, and downstream generation-grid, compounding,
  Biber-lite/style-signature, spectrum, classification, and cross-ladder
  artifacts must be rebuilt at the chosen scope.

## 2026-06-16 - Live Phase 3 generation status accounting

- Extended `slop-phase2-generation-status` with a generated-row ETA fallback
  based on the detached launch `started_at` timestamp. This makes active
  shards report useful ETA even when the tqdm log does not expose prompt-level
  timing.
- Extended `slop-summarize-phase3-generation-plans` with
  `--selection-status PLAN=CSV`, allowing the cross-plan summary to include
  in-flight shard counts, live generated-row totals, and active ETAs.
- Regenerated
  `artifacts/phase3/analysis/phase3_generation_plan_status_summary.csv` and
  `.md` with the active SmolLM3 base/t0 status merged in. The summary now
  reports one in-flight SmolLM3 shard rather than treating the whole plan as
  merely unstarted.

## 2026-06-16 - SmolLM3 no_think generation markup audit helper

- Added `slop-audit-no-think-generations`, a small JSONL audit for explicit
  thinking/reasoning markup in generated text caches. It flags tags such as
  `<think>`, `</think>`, `<reasoning>`, and fenced `thinking`/`reasoning`
  blocks without treating ordinary prose uses of `think` as failures.
- Ran the audit on the active partial SmolLM3 base/t0 production cache:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_base_t0_no_think_audit.csv`
  and `.md`. At 1,280 records, it found zero explicit thinking-marker records
  and zero marker hits. This is a no_think mode-compliance check for the
  partial cache, not a completion or generation-quality verdict.

## 2026-06-16 - SmolLM3 production grid base/t0 complete, base/t0.7 launched

- The SmolLM3 no_think production-grid base checkpoint at temperature `0.0`
  completed: 4,096/4,096 generation records plus summary CSV exist.
- Re-ran the no_think markup audit on the completed base/t0 cache:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_base_t0_no_think_audit.csv`
  and `.md`. The completed cache has zero explicit thinking-marker records
  and zero marker hits.
- Regenerated
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`
  and `.md`; the SmolLM3 plan now records 1/12 completed shards and `36.00`
  estimated A100-hours remaining.
- Launched the next SmolLM3 production-grid shard: base checkpoint,
  temperature `0.7`, 4,096 expected generations. Selection/status/log files:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t07_shard.json`,
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t07_shard_status.csv`,
  and
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t07_shard.log`.

## 2026-06-16 - SmolLM3 production grid base/t0.7 complete, base/t1 launched

- The SmolLM3 no_think production-grid base checkpoint at temperature `0.7`
  completed: 4,096/4,096 generation records plus summary CSV exist.
- Re-ran the no_think markup audit on the completed base/t0.7 cache:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_base_t07_no_think_audit.csv`
  and `.md`. The completed cache has zero explicit thinking-marker records
  and zero marker hits.
- Regenerated
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`
  and `.md`; the SmolLM3 plan now records 2/12 completed shards and `32.73`
  estimated A100-hours remaining.
- Launched the next SmolLM3 production-grid shard: base checkpoint,
  temperature `1.0`, 4,096 expected generations. Selection/status/log files:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t1_shard.json`,
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t1_shard_status.csv`,
  and
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_base_t1_shard.log`.

## 2026-06-16 - SmolLM3 production grid base/t1 complete, SFT/t0 launched

- The SmolLM3 no_think production-grid base checkpoint at temperature `1.0`
  completed: 4,096/4,096 generation records plus summary CSV exist.
- Re-ran the no_think markup audit on the completed base/t1 cache:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_base_t1_no_think_audit.csv`
  and `.md`. The completed cache has zero explicit thinking-marker records
  and zero marker hits.
- Regenerated
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`
  and `.md`; the SmolLM3 plan now records 3/12 completed shards and `29.45`
  estimated A100-hours remaining.
- Launched the first SFT production-grid shard: SFT checkpoint,
  temperature `0.0`, 4,096 expected generations. Selection/status/log files:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t0_shard.json`,
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t0_shard_status.csv`,
  and
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t0_shard.log`.

## 2026-06-16 - SmolLM3 production grid SFT/t0 complete, SFT/t0.7 launched

- The SmolLM3 no_think production-grid SFT checkpoint at temperature `0.0`
  completed: 4,096/4,096 generation records plus summary CSV exist.
- Re-ran the no_think markup audit on the completed SFT/t0 cache:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_sft_t0_no_think_audit.csv`
  and `.md`. The completed cache has zero explicit thinking-marker records
  and zero marker hits.
- Regenerated
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`
  and `.md`; the SmolLM3 plan now records 4/12 completed shards and `26.18`
  estimated A100-hours remaining.
- Launched the next SFT production-grid shard: SFT checkpoint,
  temperature `0.7`, 4,096 expected generations. Selection/status/log files:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t07_shard.json`,
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t07_shard_status.csv`,
  and
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t07_shard.log`.

## 2026-06-16 - SmolLM3 production grid SFT/t0.7 complete, SFT/t1 launched

- The SmolLM3 no_think production-grid SFT checkpoint at temperature `0.7`
  completed: 4,096/4,096 generation records plus summary CSV exist.
- Re-ran the no_think markup audit on the completed SFT/t0.7 cache:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_sft_t07_no_think_audit.csv`
  and `.md`. The completed cache has zero explicit thinking-marker records
  and zero marker hits.
- Regenerated
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`
  and `.md`; the SmolLM3 plan now records 5/12 completed shards and `22.91`
  estimated A100-hours remaining.
- Launched the next SFT production-grid shard: SFT checkpoint,
  temperature `1.0`, 4,096 expected generations. Selection/status/log files:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t1_shard.json`,
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t1_shard_status.csv`,
  and
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_sft_t1_shard.log`.

## 2026-06-16 - SmolLM3 production grid SFT/t1 complete, APO/DPO t0 launched

- The SmolLM3 no_think production-grid SFT checkpoint at temperature `1.0`
  completed: 4,096/4,096 generation records plus summary CSV exist.
- Re-ran the no_think markup audit on the completed SFT/t1 cache:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_sft_t1_no_think_audit.csv`
  and `.md`. The completed cache has zero explicit thinking-marker records
  and zero marker hits.
- Regenerated
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`
  and `.md`; the SmolLM3 plan now records 6/12 completed shards and `19.64`
  estimated A100-hours remaining.
- Launched the first APO/DPO production-grid shard: APO/DPO checkpoint,
  temperature `0.0`, 4,096 expected generations. Selection/status/log files:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t0_shard.json`,
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t0_shard_status.csv`,
  and
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t0_shard.log`.

## 2026-06-16 - SmolLM3 production grid APO/DPO t0 complete, APO/DPO t0.7 launched

- The SmolLM3 no_think production-grid APO/DPO checkpoint at temperature
  `0.0` completed: 4,096/4,096 generation records plus summary CSV exist.
- Re-ran the no_think markup audit on the completed APO/DPO t0 cache:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_dpo_t0_no_think_audit.csv`
  and `.md`. The completed cache has zero explicit thinking-marker records
  and zero marker hits.
- Regenerated
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`
  and `.md`; the SmolLM3 plan now records 7/12 completed shards and `16.36`
  estimated A100-hours remaining.
- Launched the next APO/DPO production-grid shard: APO/DPO checkpoint,
  temperature `0.7`, 4,096 expected generations. Selection/status/log files:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t07_shard.json`,
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t07_shard_status.csv`,
  and
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t07_shard.log`.

## 2026-06-16 - SmolLM3 production grid APO/DPO t0.7 complete, APO/DPO t1 launched

- The SmolLM3 no_think production-grid APO/DPO checkpoint at temperature
  `0.7` completed: 4,096/4,096 generation records plus summary CSV exist.
- Re-ran the no_think markup audit on the completed APO/DPO t0.7 cache:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_dpo_t07_no_think_audit.csv`
  and `.md`. The completed cache has zero explicit thinking-marker records
  and zero marker hits.
- Regenerated
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`
  and `.md`; the SmolLM3 plan now records 8/12 completed shards and `13.09`
  estimated A100-hours remaining.
- Launched the next APO/DPO production-grid shard: APO/DPO checkpoint,
  temperature `1.0`, 4,096 expected generations. Selection/status/log files:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t1_shard.json`,
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t1_shard_status.csv`,
  and
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_dpo_t1_shard.log`.

## 2026-06-16 - SmolLM3 production grid APO/DPO t1 complete, final/RLVR t0 launched

- The SmolLM3 no_think production-grid APO/DPO checkpoint at temperature
  `1.0` completed: 4,096/4,096 generation records plus summary CSV exist.
- Re-ran the no_think markup audit on the completed APO/DPO t1 cache:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_dpo_t1_no_think_audit.csv`
  and `.md`. The completed cache has zero explicit thinking-marker records
  and zero marker hits.
- Regenerated
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`
  and `.md`; the SmolLM3 plan now records 9/12 completed shards and `9.82`
  estimated A100-hours remaining.
- Launched the first final/RLVR production-grid shard: final/RLVR checkpoint,
  temperature `0.0`, 4,096 expected generations. Selection/status/log files:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t0_shard.json`,
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t0_shard_status.csv`,
  and
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t0_shard.log`.

## 2026-06-16 - SmolLM3 production grid final/RLVR t0 complete, final/RLVR t0.7 launched

- The SmolLM3 no_think production-grid final/RLVR checkpoint at temperature
  `0.0` completed: 4,096/4,096 generation records plus summary CSV exist.
- Re-ran the no_think markup audit on the completed final/RLVR t0 cache:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_final_t0_no_think_audit.csv`
  and `.md`. The completed cache has zero explicit thinking-marker records
  and zero marker hits.
- Regenerated
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`
  and `.md`; the SmolLM3 plan now records 10/12 completed shards and `6.55`
  estimated A100-hours remaining.
- Launched the next final/RLVR production-grid shard: final/RLVR checkpoint,
  temperature `0.7`, 4,096 expected generations. Selection/status/log files:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t07_shard.json`,
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t07_shard_status.csv`,
  and
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t07_shard.log`.

## 2026-06-16 - SmolLM3 production grid final/RLVR t0.7 complete, final/RLVR t1 launched

- The SmolLM3 no_think production-grid final/RLVR checkpoint at temperature
  `0.7` completed: 4,096/4,096 generation records plus summary CSV exist.
- Re-ran the no_think markup audit on the completed final/RLVR t0.7 cache:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_final_t07_no_think_audit.csv`
  and `.md`. The completed cache has zero explicit thinking-marker records
  and zero marker hits.
- Regenerated
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat.csv`
  and `.md`; the SmolLM3 plan now records 11/12 completed shards and `3.27`
  estimated A100-hours remaining.
- Launched the last final/RLVR production-grid shard: final/RLVR checkpoint,
  temperature `1.0`, 4,096 expected generations. Selection/status/log files:
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t1_shard.json`,
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t1_shard_status.csv`,
  and
  `artifacts/phase3/analysis/smollm3_no_think_generation_plan_512prompt_8comp_3temp_chat_final_t1_shard.log`.

## 2026-06-16 - SmolLM3 production grid complete and downstream rebuilt

- The SmolLM3 no_think production-grid final/RLVR checkpoint at temperature
  `1.0` completed: 4,096/4,096 generation records plus summary CSV exist.
  The full SmolLM3 512-prompt production-shape grid is now 12/12 shards
  complete, with 49,152/49,152 generated records.
- Re-ran the no_think markup audit on the completed final/RLVR t1 cache:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg512_chat_production_grid_final_t1_no_think_audit.csv`
  and `.md`. The completed cache has zero explicit thinking-marker records
  and zero marker hits.
- Rebuilt per-temperature production generation grids:
  `artifacts/phase3/analysis/smollm3_no_think_generation_stage_grid_512prompt_8comp_t0_chat_production.csv`,
  `artifacts/phase3/analysis/smollm3_no_think_generation_stage_grid_512prompt_8comp_t07_chat_production.csv`,
  and
  `artifacts/phase3/analysis/smollm3_no_think_generation_stage_grid_512prompt_8comp_t1_chat_production.csv`.
- Rebuilt production `t=1.0` free-run effects, compounding, Biber-lite
  comparison, style signature, spectrum, classifier, and cross-ladder
  comparison. The production classifier still labels `slop_lexicon` as
  `sft-amplified`, `rule_of_three_approx` as
  `measured-no-phase3-class`, and sparse teacher-forced features as
  `observed-output-only`. The production OLMo-vs-SmolLM3 AF comparison remains
  Spearman `0.762`, Pearson `0.978` over 8 shared AF values.

## 2026-06-16 - OLMo full production grid started with lower batch size

- Attempted the first full OLMo production-grid shard from the original
  `batched1024` plan: base checkpoint, temperature `0.0`, 40,000 expected
  generations. The launch failed before writing usable rows because compiled
  prefill tried to allocate an additional 21.50 GiB and hit CUDA OOM on the
  A100. The failed output file is zero bytes and no summary CSV exists.
- Regenerated the full OLMo 5,000-prompt x 8-completion x 3-temperature plan
  with `--generation-batch-size 256`:
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched256.csv`
  and `.md`. That compiled lower-batch launch also failed before writing rows,
  with CUDA OOM during efficient attention.
- Regenerated the full plan again with `--generation-batch-size 128` and
  `--no-torch-compile`:
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched128_no_compile.csv`
  and `.md`. That no-compile lower-batch launch also failed before writing
  usable rows, with CUDA OOM in the MLP.
- Regenerated the full plan again with `--generation-batch-size 64` and
  `--no-torch-compile`:
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched64_no_compile.csv`
  and `.md`.
- Launched the replacement OLMo base/t0 shard detached from the
  `batched64_no_compile` plan. Selection/status/log files:
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched64_no_compile_base_t0_shard.json`,
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched64_no_compile_base_t0_shard_status.csv`,
  and
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched64_no_compile_base_t0_shard.log`.
  This run passed the early OOM point and wrote initial rows: latest checked
  status was alive with `384/40000` generations and an existing-row ETA of
  `16:48:10`.
- Refreshed
  `artifacts/phase3/analysis/phase3_generation_plan_status_summary.md`; it
  now records OLMo `0` completed shards, `1` in-flight shard, `11` missing
  shards, and SmolLM3 `12/12` complete.

## 2026-06-16 - OLMo continuation guard added

- Added `slop-continue-phase2-generation-plan`, a guarded continuation helper
  for the long OLMo production grid. It detects active detached shards from
  tracked selection JSON files, refuses to overwrite partial missing shard
  output by default, applies the estimated A100-hour cap, and writes the next
  selection payload only when a clean missing shard can be launched.
- Dry-ran the helper against the active OLMo `batched64_no_compile` plan with
  selection prefix
  `olmo3_generation_plan_full_5000prompt_8comp_3temp_batched64_no_compile`.
  It correctly reported `active_shard_running` for the current base/t0 shard
  instead of selecting a new shard.
- Refreshed the live OLMo status and combined plan summary. The active OLMo
  base/t0 shard is alive at `512/40000` generations with log-derived active
  ETA `10:04:15`; SmolLM3 remains `12/12` complete.

## 2026-06-16 - OLMo base/t0 resumed after batch64 OOM

- The active OLMo base/t0 `batched64_no_compile` attempt died after writing
  512/40,000 generation rows. The log shows CUDA OOM during KV-cache growth:
  the process held about 78.30 GiB and failed on an additional 1002 MiB
  allocation.
- Added `--resume` to `slop-free-running-emission`. Resume mode preserves the
  existing generations JSONL, loads existing rows into summary counters, skips
  already written source/record/completion/temperature/top-p cells, and
  appends only missing generations.
- Updated `slop-continue-phase2-generation-plan` so explicit partial retries
  append `--resume`, can override `--generation-batch-size`, and record
  `resume_initial_generations` for sane post-restart throughput accounting.
- Relaunched OLMo base/t0 with the same output cache, `--resume`, and
  `--generation-batch-size 32`. New selection:
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched64_no_compile_base_t0p0.json`.
  Latest refreshed status is alive at `608/40000` generations, active ETA
  `30:00:37`, with GPU memory around 61 GiB.
- Added the effective OLMo continuation plan
  `artifacts/phase2/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_batched64_outputs_batched32_resume.csv`
  and `.md`. It preserves the existing `batched64` output paths, switches all
  commands to `--generation-batch-size 32`, and enables `--resume`, so future
  launches do not strand the partial base/t0 cache.
- Dry-ran the continuation helper against the effective plan. It correctly
  reports the active base/t0 shard at `704/40000` generations rather than
  selecting a new shard. The combined Phase 3 plan summary now uses this
  effective plan and reports OLMo active ETA `31:50:47`.

## 2026-06-16 - OLMo batch-size policy clarified

- Rechecked the resumed OLMo base/t0 shard: it is alive at `1312/40000`
  generations with active ETA `15:01:51`, with GPU memory around 61 GiB.
- Clarified that `--generation-batch-size 32` is the recovery-safe setting
  after the observed batch-64 KV-cache OOM, not a final throughput optimum.
- Added a runbook gate for profiling `--generation-batch-size-override 48` at
  the next clean handoff rather than interrupting the healthy active shard.
- Added `slop-materialize-phase2-generation-continuation-plan` so the
  effective batch-32/resume OLMo continuation plan is reproducible from the
  original batch-64 plan without ad hoc scripting.

## 2026-06-16 - OLMo reduced SGLang generation plan adopted

- Stopped the resumed full-grid OLMo base/t0 shard after confirming it was
  alive at `1472/40000` rows and using the A100 heavily. The partial JSONL is
  preserved at
  `artifacts/phase2/generations/olmo3_base_promptpkg5000_free_run_5000prompt_8comp_t0_batched64.jsonl`
  as pilot/cache data, but the 5,000-prompt x 8-completion x 3-temperature
  grid is no longer the active operational target.
- Materialized a reduced OLMo Phase 3 generation design: 8,192 main paired
  style-signature generations, 6,144 temperature-sensitivity generations, and
  2,048 long-output compounding generations, for 16,384 planned OLMo
  generations total.
- Added `slop-materialize-sglang-generation-plan`, which rewrites ordinary
  Phase 2 generation plan rows into SGLang launch commands while preserving
  output paths, expected row counts, and the existing guarded launcher/status
  workflow.
- Materialized the active SGLang OLMo plans:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_main_signature_1024prompt_2comp_t07_512_sglang.csv`,
  `artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_512prompt_1comp_3temp_512_sglang.csv`,
  and
  `artifacts/phase3/analysis/olmo3_phase3_reduced_long_compounding_256prompt_2comp_t07_1024_sglang.csv`.
  The combined status summary is
  `artifacts/phase3/analysis/phase3_reduced_sglang_generation_plan_status_summary.md`.
- The SGLang plans use `.venvs/sglang-cu128/bin/python`,
  `scripts/benchmark_sglang_generation.py`, and `--ignore-eos` fixed-budget
  generation. The repo-local SGLang sidecar still needs to be created before
  launch; the runbook records the validated CUDA 12.8 setup commands. Current
  host check found `python3.12`, but not `gcc-14`, `g++-14`, or
  `/usr/local/cuda-12.8`, so the SGLang sidecar/toolchain must be installed or
  exposed before launching the reduced OLMo shards.

## 2026-06-16 - Reduced OLMo SGLang main panel completed

- Built the repo-local `.venvs/sglang-cu128` sidecar with Torch
  `2.8.0+cu128`, SGLang `0.5.2`, and a forced Transformers `4.57.6` upgrade
  for OLMo 3 support. SGLang initially failed on missing `libnuma.so.1`; fixed
  that without root by downloading and extracting Ubuntu's
  `libnuma1_2.0.18-1build1_amd64.deb` under the sidecar sysroot. SGLang then
  failed on FlashInfer's `nvcc` JIT path, so the working command shape uses
  `--attention-backend triton` and `--disable-cuda-graph`.
- Updated `scripts/benchmark_sglang_generation.py` to expose
  `--attention-backend` and `--disable-cuda-graph`, and updated
  `slop-materialize-sglang-generation-plan` so generated plan commands include
  the required `env LD_LIBRARY_PATH=... TORCH_CUDA_ARCH_LIST=8.0` prefix.
- Ran a one-prompt OLMo base SGLang smoke successfully:
  `artifacts/phase3/generations/olmo3_phase3_sglang_base_1prompt_smoke.jsonl`
  and summary CSV.
- Completed all four reduced OLMo main style-signature shards with SGLang:
  base, SFT, DPO, and final/RLVR at temperature `0.7`, 1,024 prompts, two
  completions, and 512 max new tokens each. The main panel is now 4/4 shards
  complete with 8,192/8,192 generations.
- Built the reduced main Tier-1 generation grid:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_main_generation_stage_grid_1024prompt_2comp_t07_sglang.csv`
  and comparison/summary artifacts. The `slop_lexicon` rate per 1k generated
  tokens is base `0.283`, SFT `0.366`, DPO `0.362`, and final/RLVR `0.312`.
- Started the reduced temperature panel and completed the base checkpoint at
  temperatures `0.0`, `0.7`, and `1.0`, for 1,536/6,144 temperature-panel
  generations. The combined reduced status now records OLMo main complete,
  OLMo temperature 3/12 shards complete, and OLMo long compounding not
  started:
  `artifacts/phase3/analysis/phase3_reduced_sglang_generation_plan_status_summary.md`.

## 2026-06-16 - Reduced OLMo SGLang generation plan completed

- Completed the remaining reduced OLMo SGLang generation panels:
  temperature-sensitivity reached 12/12 shards and 6,144/6,144 generations;
  long-output compounding reached 4/4 shards and 2,048/2,048 generations.
  Together with the previously completed main panel, the reduced OLMo SGLang
  plan is now 16,384/16,384 generations complete.
- Refreshed the combined reduced status summary:
  `artifacts/phase3/analysis/phase3_reduced_sglang_generation_plan_status_summary.md`.
- Assembled reduced OLMo generated-output grids:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_generation_stage_grid_512prompt_1comp_3temp_sglang.csv`
  and
  `artifacts/phase3/analysis/olmo3_phase3_reduced_long_generation_stage_grid_256prompt_2comp_t07_1024_sglang.csv`.
- Ran the reduced OLMo long-output compounding analysis over the four 1,024-token
  caches:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_long_compounding_analysis_256prompt_2comp_t07_1024_sglang.csv`.
  The optional realized-AF SVG was not produced because no teacher-forced
  propensity grid was supplied for expected-rate columns in this reduced run;
  the empirical prior/no-prior window rates were written successfully.

## 2026-06-16 - Reduced OLMo SGLang spectrum integrated

- Fixed `slop-analyze-phase3-free-run-effects` to pair SGLang generation
  caches by `prompt_id` when `record_id` is absent. The previous reducer
  collapsed SGLang rows to two completion-index keys; the corrected reduced
  long-output free-run stage effects now use 512 paired units per adjacent
  comparison and report 10/18 BH-FDR-significant rows:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_long_free_run_stage_effects_256prompt_2comp_t07_1024_sglang.csv`.
- Reran reduced OLMo long-output compounding with the
  `slop_lexicon` teacher-forced propensity grid, producing expected/excess
  columns and the realized-AF SVG:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_long_compounding_analysis_256prompt_2comp_t07_1024_sglang.csv`
  and
  `artifacts/phase3/analysis/olmo3_phase3_reduced_long_compounding_analysis_256prompt_2comp_t07_1024_sglang_realized_af.svg`.
  The primary `slop_lexicon` compounding rows are base
  observed/expected/excess `0.446/0.232/0.214`, SFT `0.559/0.336/0.223`,
  DPO `0.700/0.439/0.261`, and final/RLVR `0.643/0.445/0.198` per 1k
  opportunities.
- Assembled the integrated reduced OLMo spectrum and classifier:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv`
  and
  `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_feature_classification_t07_long1024.csv`.
  The reduced classifier keeps `slop_lexicon` as preference-amplified and
  dynamics-driven, with DPO as the free-running and teacher-forced peak.
- Rebuilt the OLMo-reduced-vs-SmolLM3 production cross-ladder comparison:
  `artifacts/phase3/analysis/olmo3_reduced_sglang_vs_smollm3_no_think_production_summary.md`.
  It aligns 24 feature-stage rows and reports overall Spearman AF `0.762` and
  Pearson AF `0.978` over 8 shared AF values.

## 2026-06-16 - Temperature-dependent realized AF completed for bounded Phase 3

- Updated `slop-analyze-phase2-compounding` to accept repeated
  `--generation-cache stage=path` arguments. This is required for temperature
  panels where stage is constant but temperature differs; aggregation already
  keys rows by `(stage, source, temperature, top_p, feature)`.
- Ran reduced OLMo temperature-dependent compounding over the 12 completed
  SGLang temperature caches. Outputs:
  `artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_compounding_analysis_512prompt_1comp_3temp_512_sglang.csv`,
  summary markdown, and
  `artifacts/phase3/analysis/olmo3_phase3_reduced_temperature_compounding_analysis_512prompt_1comp_3temp_512_sglang_realized_af.svg`.
  For `slop_lexicon`, realized AF is base `0.537/1.177/1.065`, SFT
  `1.226/1.135/1.359`, DPO `1.154/1.012/1.107`, and final/RLVR
  `0.972/1.086/1.138` at temperatures `0.0/0.7/1.0`.
- Ran SmolLM3 no_think temperature-dependent compounding over the completed
  12-shard production-shape grid. Outputs:
  `artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_analysis_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3.csv`,
  summary markdown, and
  `artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_analysis_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3_realized_af.svg`.
  For `slop_lexicon`, realized AF is base `0.309/0.370/0.649`, SFT
  `1.744/1.937/2.133`, APO/DPO `2.070/1.972/2.285`, and final
  `1.645/2.011/2.279` at temperatures `0.0/0.7/1.0`.

## 2026-06-16 - Generated-output bootstrap CIs propagated into Phase 3 spectra

- Added `--generation-cache`, `--bootstrap-samples`, and `--bootstrap-seed`
  to `slop-assemble-phase2-generation-grid`, with document/prompt-cluster
  bootstrap intervals for generated-output per-1k feature rates.
- Updated `slop-assemble-amplification-spectrum` to propagate those intervals
  as `free_run_per_1k_tokens_ci_low/high`. The assembler now also infers the
  intended OLMo data-rate role for raw Dolma pretraining and Dolci SFT census
  rows that are labeled as generic `text`.
- Regenerated the reduced OLMo long-output grid, SmolLM3 no_think production
  `t=1.0` grid, both current spectra, both classifiers, and the
  OLMo-reduced-vs-SmolLM3-production cross-ladder comparison.
- The OLMo reduced long-panel `slop_lexicon` generated-output rates are base
  `0.278` `[0.183, 0.397]`, SFT `0.317` `[0.166, 0.477]`, DPO `0.381`
  `[0.269, 0.511]`, and final/RLVR `0.366` `[0.254, 0.488]` hits per 1k
  generated tokens.
- The SmolLM3 no_think production `t=1.0` `slop_lexicon` rates are base
  `0.118` `[0.082, 0.165]`, SFT `0.310` `[0.268, 0.352]`, APO/DPO `0.379`
  `[0.332, 0.425]`, and final `0.390` `[0.344, 0.445]` hits per 1k
  generated tokens.
- The refreshed cross-ladder comparison still aligns 24 feature-stage rows and
  reports overall Spearman AF `0.762` and Pearson AF `0.978` over 8 shared AF
  values.

## 2026-06-16 - Compounding bootstrap CIs added to primary Phase 3 artifacts

- Added `--bootstrap-samples` and `--bootstrap-seed` to
  `slop-analyze-phase2-compounding`. The analyzer now resamples
  document/prompt clusters from cached generation JSONL and emits CIs for
  observed per-1k opportunities, excess per-1k opportunities, realized AF,
  observed/expected ratio, and prior/no-prior risk metrics.
- Updated `slop-assemble-amplification-spectrum` to propagate primary
  compounding CI fields into the integrated spectra.
- Regenerated reduced OLMo long-output compounding with 1,000 bootstrap
  samples, reduced OLMo temperature compounding with 1,000 bootstrap samples,
  and SmolLM3 production `t=1.0` compounding with 1,000 bootstrap samples.
  Rebuilt both current spectra, both classifiers, and the current
  OLMo-reduced-vs-SmolLM3-production cross-ladder comparison.
- OLMo reduced long-panel `slop_lexicon` excess per 1k opportunity CIs are:
  base `0.214` `[0.066, 0.401]`, SFT `0.223` `[-0.042, 0.528]`, DPO
  `0.261` `[0.050, 0.488]`, and final/RLVR `0.198` `[0.007, 0.383]`.
- SmolLM3 production `t=1.0` `slop_lexicon` excess per 1k opportunity CIs
  are: base `0.090` `[0.003, 0.203]`, SFT `-0.372` `[-0.482, -0.248]`,
  APO/DPO `-0.456` `[-0.571, -0.332]`, and final `-0.465`
  `[-0.590, -0.338]`.
- The all-temperature SmolLM3 compounding bootstrap was attempted with 250
  samples but stopped after several minutes because rescoring all 49,152
  generated rows was too slow for the lightweight Phase 3 route. The existing
  SmolLM3 temperature realized-AF table remains the point-estimate
  temperature-dependence artifact; adding CIs there should use a cached-counter
  bootstrap rather than rescoring raw text.

## 2026-06-16 - Biber-lite generated-side bootstrap CIs added

- Added `--bootstrap-samples` and `--bootstrap-seed` to
  `slop-compare-phase2-biber`. The comparison now resamples generation
  document/prompt clusters and emits generated-side CIs for Biber-lite
  generated rates, generated per-doc rates, document shares, and
  generated-vs-corpus deltas/ratios while holding aggregate corpus baselines
  fixed.
- Updated `slop-build-style-signature` to preserve source-layer uncertainty as
  `value_ci_low/high` for Tier-1, compounding, and Biber-lite rows where the
  input artifact supplies intervals.
- Regenerated SmolLM3 no_think production Biber-lite comparison and style
  signature with 1,000 bootstrap samples:
  `artifacts/phase2/analysis/smollm3_biber_lite_generation_vs_corpus_512prompt_8comp_t1_chat_production.csv`
  and
  `artifacts/phase3/analysis/smollm3_no_think_style_signature_512prompt_8comp_t1_chat_production.csv`.
- Built a combined OLMo Biber-lite corpus-rate table from the existing Dolma
  pretraining, Dolci SFT, and retained Dolci DPO Biber-lite censi:
  `artifacts/stage1/census/olmo3_revised_biber_lite_combined_feature_rates.csv`.
  Regenerated the OLMo Biber-lite comparison and style signature with 1,000
  bootstrap samples:
  `artifacts/phase2/analysis/olmo3_biber_lite_generation_vs_corpus_t1.csv`
  and `artifacts/phase2/analysis/olmo3_style_signature_t1.csv`.
- The Biber-lite layer now has generated-output uncertainty for both ladders;
  corpus-side Biber baselines remain aggregate point rates in these comparison
  artifacts.

## 2026-06-16 - Lightweight SGLang/counter-cache strategy recorded

- Confirmed SGLang as the preferred backend for pure generation tasks. The
  reduced OLMo Phase 3 route remains the active replacement for the abandoned
  480,000-generation OLMo grid; further Transformers batch-size tuning is not
  the right use of the A100 for generation-only shards.
- Added reusable compounding counter caches to
  `slop-analyze-phase2-compounding` through `--counter-cache-output` and
  `--counter-cache-input`. Bootstrap sampling now uses deterministic
  prompt-cluster ordering, so cached reruns are byte-reproducible with the
  same seed.
- Regenerated the supplemental SmolLM3 no_think three-temperature
  `slop_lexicon` compounding CI table from the materialized counter cache in
  about 20 seconds:
  `artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_slop_lexicon_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3.csv`.
- Verified the cached regeneration byte-for-byte against an independent
  `/tmp` rerun. The counter cache is:
  `artifacts/phase3/analysis/smollm3_no_think_temperature_compounding_slop_lexicon_cluster_counts_512prompt_8comp_3temp_chat_production_tf_slop_neutral_rule3.jsonl`.
- Updated the Phase 3 conclusion report, completion audit, status file, and
  production runbook to state the lightweight compute policy: SGLang for pure
  generation, reduced purpose-built panels, cached counters for repeated
  compounding bootstraps, and no blocking on perfect pretraining-source
  coverage.

## 2026-06-16 - DPO-vs-APO variation report added

- Added `docs/experiments/phase3_dpo_vs_apo_variation_report.md` as the
  bounded Phase 3 writeup comparing OLMo 3/DPO and SmolLM3/APO over the
  current measured overlap.
- The report records the central stage-localization contrast: OLMo
  `slop_lexicon` is preference-amplified and dynamics-driven, while SmolLM3
  `slop_lexicon` is already SFT-amplified and remains high through APO/final.
- Updated the Phase 3 status, completion audit, and integrated conclusion to
  point to the report and removed the DPO-vs-APO writeup from the active
  remaining-work list.

## 2026-06-16 - Optional Think/RL-Zero stretch plan materialized

- Verified the current OLMo 3 Think and RL-Zero model IDs from Hugging Face
  model cards before planning the optional stretch path.
- Materialized a reduced optional OLMo Instruct-vs-Think-vs-RL-Zero
  generation plan:
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_256prompt_2comp_t07_1024.csv`.
- Materialized the matching SGLang command plan:
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_256prompt_2comp_t07_1024_sglang.csv`,
  with summary
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_256prompt_2comp_t07_1024_sglang.md`.
- The plan covers base, Instruct final, Think SFT, Think DPO, Think final, and
  RL-Zero General/IF/Math/Code/Mix: 10 shards, 512 expected generations per
  shard, 5,120 generations total, and `4.09` source-estimated A100-hours
  before SGLang speedup.
- The stretch remains unrun. After generation, analysis must strip or
  separately score reasoning traces so feature extraction uses final-answer
  text only.

## 2026-06-16 - Final-answer feature extraction implemented for reasoning outputs

- Added `slop_sftdiv.generation_text` with `raw` and `final_answer` feature
  text modes. `final_answer` strips explicit `<think>`/`<reasoning>` blocks
  and fenced thinking/reasoning blocks before Tier-1 feature extraction.
- Added `--feature-text-mode {raw,final_answer}` to
  `slop-free-running-emission` and `scripts/benchmark_sglang_generation.py`.
  Raw generations remain in JSONL; feature counts and per-1k denominators use
  the selected feature text. In final-answer mode, denominators use the
  final-answer token count rather than full reasoning-trace generated tokens.
- Added `--feature-text-mode` passthrough to
  `slop-materialize-sglang-generation-plan` and regenerated the optional
  Think/RL-Zero stretch SGLang plan with
  `--feature-text-mode final_answer` embedded in every command:
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_256prompt_2comp_t07_1024_sglang.csv`.
- The optional stretch is still unrun, but the planned generation commands now
  satisfy the `EXPERIMENTS.md` requirement that Think/RL-Zero feature
  extraction use final-response text only.

## 2026-06-16 - Think/RL-Zero stretch comparison assembler added

- Added `slop-assemble-olmo-stretch-comparison`, a post-generation assembler
  for the optional OLMo Instruct/Think/RL-Zero stretch path. It consumes the
  final-answer-mode generation grid and writes a per-stage comparison table,
  feature-summary table, and Markdown report.
- The assembler reports per-feature maxima, deltas/ratios versus Base,
  Instruct final, and Think final, and an RL-Zero family mean. This makes the
  downstream stretch comparison deterministic once the 10 SGLang shards
  complete.
- Updated the Phase 3 production runbook with the exact commands to assemble
  the stretch generation grid and the path-comparison report after generation.

## 2026-06-16 - Think/RL-Zero stretch SGLang generation completed

- Ran the reduced OLMo Instruct-vs-Think-vs-RL-Zero SGLang stretch plan:
  10/10 stages, 5,120/5,120 generated records, with every JSONL row verified
  as `feature_text_mode=final_answer`.
- Completed stages: base, Instruct final, Think SFT, Think DPO, Think final,
  RL-Zero General, RL-Zero IF, RL-Zero Math, RL-Zero Code, and RL-Zero Mix.
- The run initially filled `/home/user` because the Hugging Face cache held
  multiple 7B checkpoint copies. Cleared only disposable OLMo model-cache
  directories, then resumed from completed summaries.
- `allenai/Olmo-3-7B-RL-Zero-Mix` failed under SGLang/Transformers `4.57.6`
  because its downloaded config used `model_type=olmo2-retrofit`; the other
  RL-Zero checkpoints use `model_type=olmo3`. Patched the local downloaded
  cache config for Mix to the same OLMo 3 loader identifiers and reran the
  shard successfully.
- Rebuilt the stretch generation grid with document/prompt-cluster bootstrap
  CIs from the JSONL caches:
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_generation_grid_256prompt_2comp_t07_1024_sglang.csv`.
- Rebuilt the path comparison:
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_comparison_256prompt_2comp_t07_1024_sglang.csv`,
  feature summary
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_feature_summary_256prompt_2comp_t07_1024_sglang.csv`,
  and Markdown summary
  `artifacts/phase3/analysis/olmo3_phase3_stretch_think_rlzero_comparison_256prompt_2comp_t07_1024_sglang_summary.md`.
- Result: Instruct final is the maximum for every tracked Tier-1 output
  feature in this reduced final-answer-mode path comparison. For
  `slop_lexicon`, Instruct final is `0.807` per 1k final-answer tokens; Think
  final is lower by `0.384`, and the RL-Zero family mean is lower by `0.244`.

## 2026-06-16 - Sparse SmolLM3 teacher-forced addendum completed

- Materialized a full eligible SmolTalk2 everyday no_think SFT prompt package:
  `artifacts/phase2/prompts/smollm3_smoltalk2_sft_everyday_no_think_phase2_prompt_package_2258.jsonl`.
  It contains 2,258 selected prompts, 87,458 prompt tokens, and 188,435 target
  tokens.
- Measured denominator support on the 2,258-prompt package:
  `artifacts/phase3/analysis/smollm3_no_think_promptpkg2258_denominator_support.csv`.
  Sparse nonzero support exists for `contrastive_negation` (16,526
  opportunities, 2 references), `stock_closers` (4,139 opportunities, 3
  references), and `stock_openers_closers` (6,397 opportunities, 3 references).
  `stock_openers` remains unsupported with zero target references.
- Regenerated the sparse teacher-forced plan with the corrected SmolLM3 model
  specs:
  base `HuggingFaceTB/SmolLM3-3B-Base`, SFT
  `HuggingFaceTB/SmolLM3-3B-checkpoints@it-SFT`, APO
  `HuggingFaceTB/SmolLM3-3B-checkpoints@it-soup-APO`, and final
  `HuggingFaceTB/SmolLM3-3B`.
- Completed all four sparse teacher-forced shards for
  `contrastive_negation`, `stock_closers`, and `stock_openers_closers`.
  Practical runtime was about 11.5-11.75 minutes per stage, because actual
  sparse opportunity rows were about 27k per stage rather than the planner's
  pessimistic capped estimate.
- Assembled the sparse stage grid:
  `artifacts/phase3/analysis/smollm3_no_think_propensity_stage_grid_2258prompt_sparse_supported.csv`.
  Raw AFs:
  `contrastive_negation` base `1.967`, SFT `3.124`, APO `3.026`, final
  `3.254`; `stock_closers` base `42.206`, SFT `91.178`, APO `100.983`,
  final `96.294`; `stock_openers_closers` base `53.770`, SFT `105.893`,
  APO `128.318`, final `123.391`.
- Ran paired opportunity-level teacher-forced stage effects:
  `artifacts/phase3/analysis/smollm3_no_think_teacher_forced_stage_effects_2258prompt_sparse_supported.csv`.
  The test produced 9 rows and 8 BH-FDR-significant rows at alpha `0.05`.
- Assembled sparse addendum spectrum and classifier:
  `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_2258prompt_sparse_supported.csv`
  and
  `artifacts/phase3/analysis/smollm3_no_think_feature_classification_2258prompt_sparse_supported.csv`.
  The addendum labels `contrastive_negation` as `sft-amplified`, and labels
  `stock_closers` and `stock_openers_closers` as
  `preference-amplified`/`dynamics-driven`.
- Interpretation caveat: this is useful broader teacher-forced support, but
  it is still sparse addendum evidence. Held-out references are only 2 or 3
  per measured sparse feature, every raw-AF CI overlaps 1, and there is no
  matching generated-output compounding expectation join for this 2,258-prompt
  sparse package.

## 2026-06-16 - Full OLMo SGLang plan made resumable and launched

- Materialized a SGLang-backed version of the original OLMo 5,000-prompt x
  8-completion x 3-temperature generation grid:
  `artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.csv`
  and
  `artifacts/phase3/analysis/olmo3_generation_plan_full_5000prompt_8comp_3temp_sglang.md`.
- Added `--resume` support to `scripts/benchmark_sglang_generation.py`. The
  SGLang harness now reads existing JSONL rows, reconstructs feature-rate
  summaries from them, skips already written
  source/prompt/completion/temperature/top-p keys, and appends only missing
  rows.
- Added `--prompt-batch-size` to the SGLang harness and to
  `slop-materialize-sglang-generation-plan`. This prevents full-grid shards
  from waiting for one monolithic 5,000-prompt `engine.generate` call before
  writing any output.
- Tested the resume/chunk behavior with a pre-populated JSONL row and a fake
  SGLang engine. Focused tests:
  `uv run pytest tests/test_benchmark_sglang_generation.py tests/test_materialize_sglang_generation_plan.py tests/test_continue_phase2_generation_plan.py`.
- A first attempt at the full base/t0 shard with one monolithic request was
  stopped before writing rows because it was not checkpointing soon enough.
  A second attempt with `--prompt-batch-size 512` was also stopped before
  writing rows because each checkpoint would still require about 4.2M
  generated tokens.
- Relaunched the first full-grid shard with `--prompt-batch-size 128`:
  base `allenai/Olmo-3-1025-7B`, temperature `0.0`, 40,000 expected
  generations.
  Selection:
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_generation_shard_base_t0p0.json`.
  Log:
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_generation_shard_base_t0p0.log`.
  Output:
  `artifacts/phase2/generations/olmo3_base_promptpkg5000_free_run_5000prompt_8comp_t0_batched1024.jsonl`.
- First live checkpoint at `2026-06-16T20:36:28Z`: the shard is still alive
  and has written `1024/40000` rows. This is one 128-prompt chunk with 8
  completions per prompt, `1,048,576` generated tokens, and all rows
  length-capped at 1,024 tokens under the fixed-budget `--ignore-eos`
  contract. The status helper projected about `7:22:13` remaining from the
  existing row rate, so the full-grid SGLang recovery path is hours per shard
  rather than weeks per shard.
- Second live checkpoint at `2026-06-16T20:41:57Z`: the shard is still alive
  and has written `2048/40000` rows. The refreshed SGLang plan now records
  base/t0 `existing_generations=2048`, and the status helper projects about
  `5:27:19` remaining from the existing row rate.
- Third live checkpoint at `2026-06-16T20:50:10Z`: the shard is still alive
  and has written `3072/40000` rows. The refreshed SGLang plan now records
  base/t0 `existing_generations=3072`, and the status helper projects about
  `5:18:27` remaining from the existing row rate.
- Fourth live checkpoint at `2026-06-16T20:56:05Z`: the shard is still alive
  and has written `4096/40000` rows. The refreshed SGLang plan now records
  base/t0 `existing_generations=4096`, and the status helper projects about
  `5:00:11` remaining from the existing row rate.
- Added `scripts/phase3_full_sglang_supervisor.sh` and started it detached via
  `setsid -f` as PID `3890` after confirming that plain `nohup ... &` did not
  survive this execution environment. The supervisor checks the full SGLang
  plan every 600 seconds, refreshes the plan summary from current JSONL caches,
  uses `slop-continue-phase2-generation-plan --execute --allow-partial-retry`,
  refuses to proceed below `12` GiB free on `/home/user`, and relies on the
  launcher's active-shard detection to avoid duplicate GPU jobs. Supervisor
  PID and log paths are
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_supervisor.pid` and
  `artifacts/phase3/analysis/olmo3_full_sglang_chunk128_supervisor.log`.
- Added `slop-materialize-phase3-full-sglang-finalize-plan` and generated the
  first full-grid post-generation finalization plan:
  `artifacts/phase3/analysis/olmo3_full_sglang_5000prompt_8comp_3temp_finalize.sh`
  and
  `artifacts/phase3/analysis/olmo3_full_sglang_5000prompt_8comp_3temp_finalize.md`.
  The readiness summary currently has zero enabled commands because no
  complete four-stage temperature slice exists yet: temperature `0.0` is
  `5120/160000` rows, and temperatures `0.7` and `1.0` are `0/160000`.
  The generated script comments out incomplete assembly/compounding commands
  and will enable generation-grid, compounding, spectrum, classification, and
  cross-ladder commands after the relevant slices complete.
- Added `--priority-temperature` to `slop-continue-phase2-generation-plan` and
  restarted the full-grid supervisor as PID `6211`. The supervisor now passes
  `--priority-temperature 1.0`, so active-shard detection still prevents
  duplicates, but after the current base/t0 shard exits the next launches will
  prioritize the primary t=1.0 slice across stages. This should make the main
  spectrum/classification path ready earlier while preserving the full
  12-shard grid.
- Recorded the generation backend policy explicitly: SGLang is the default for
  pure generation tasks, while Torch/Transformers batch-size recovery is legacy
  context unless SGLang hits a concrete loader/runtime blocker. For the
  original full-grid recovery, the control knobs are SGLang prompt chunk size,
  JSONL resume checkpoints, and temperature-slice ordering rather than
  returning to batch-size-32 Transformers generation.
- Fifth live checkpoint at `2026-06-16T21:07:00Z`: the base/t0 SGLang shard is
  still alive as PID `1032`, has written `5120/40000` rows, and the detached
  supervisor is alive as PID `6211`. `/home/user` has `24` GiB free, and the
  A100 is at `100%` utilization with about `70.8/81.9` GiB used. The status
  helper projects about 5 hours remaining from the existing row rate.
- Extended `slop-materialize-phase3-full-sglang-finalize-plan` so the
  readiness summary now includes a per-stage table in addition to the
  temperature-level command gate. The regenerated summary currently shows only
  `base/t0` with partial rows (`6144/40000`) and no complete stage summaries,
  so the finalization script still has zero enabled commands.
- Extended `scripts/phase3_full_sglang_supervisor.sh` with
  `REFRESH_FINALIZATION=1` by default. Each supervisor loop now regenerates
  `artifacts/phase3/analysis/olmo3_full_sglang_5000prompt_8comp_3temp_finalize.sh`
  and `.md` after refreshing the SGLang generation plan, keeping downstream
  readiness current without a manual materializer run. Restarted only the
  supervisor, not the active generation process; the live supervisor is PID
  `8230`, the active SGLang generation process remains PID `1032`, and no
  stale supervisor process remains.
- Sixth live checkpoint at `2026-06-16T21:15:00Z`: the base/t0 SGLang shard is
  still alive as PID `1032`, has written `6144/40000` rows, or six
  128-prompt chunks. The refreshed SGLang plan and finalization readiness now
  record temperature `0.0` at `6144/160000` rows, temperatures `0.7` and `1.0`
  at `0/160000`, and zero enabled downstream commands.
- Updated the finalization materializer so complete slices skip downstream
  commands whose declared output artifacts already exist. This keeps the
  generated finalization script focused on pending work and prevents automatic
  readiness refreshes or manual reruns from redoing already completed
  generation-grid, compounding, spectrum, classification, or cross-ladder
  outputs.
- Added `slop-audit-phase3-completion`, which writes
  `artifacts/phase3/analysis/phase3_completion_audit_snapshot.csv` and `.md`
  from the current full SGLang generation plan, finalization readiness summary,
  and named required artifacts. It now also treats enabled finalization
  commands as pending work. The current snapshot reports `8/10` tracked
  requirements complete: bounded OLMo/SmolLM3 outputs, the conclusion report,
  finalization readiness, and zero pending finalization commands are verified,
  while `full_olmo_sglang_generation_grid` and
  `primary_temperature_1_four_stage_slice` remain incomplete.
- Extended `scripts/phase3_full_sglang_supervisor.sh` with
  `REFRESH_COMPLETION_AUDIT=1` by default. Restarted only the supervisor; the
  live supervisor is PID `12474`, the active generation process remains PID
  `1032`, and the supervisor log confirms it refreshed the SGLang plan,
  finalization readiness, completion audit, and then detected the active shard
  rather than launching a duplicate.
- Seventh live checkpoint at `2026-06-16T21:23:00Z`: the base/t0 SGLang shard
  is still alive as PID `1032`, has written `7168/40000` rows, or seven
  128-prompt chunks. The refreshed plan and finalization readiness record
  temperature `0.0` at `7168/160000`, temperatures `0.7` and `1.0` at
  `0/160000`, and zero enabled downstream commands.
- Expanded the completion audit gate set to include the expected full-grid
  primary t=1 downstream outputs: generation grid, compounding table,
  amplification spectrum, feature classification, and cross-ladder summary.
  The current machine-readable snapshot now reports `8/15` complete; the
  newly added full-grid primary downstream gates are correctly `missing` until
  the t=1 four-stage slice finishes and finalization runs.
- Tightened `slop-audit-phase3-completion` so named required artifacts must
  exist and have nonzero file size, with CSV files requiring at least one data
  row and JSONL files requiring at least one non-empty record. Missing, empty,
  or header-only artifacts now report `missing_or_empty_or_header_only`, which
  prevents empty placeholders from satisfying Phase 3 gates.
- Eighth live checkpoint at `2026-06-16T21:32:00Z`: the base/t0 SGLang shard
  is still alive as PID `1032`, has written `8192/40000` rows, or eight
  128-prompt chunks. The refreshed plan and finalization readiness record
  temperature `0.0` at `8192/160000`, temperatures `0.7` and `1.0` at
  `0/160000`, and zero enabled downstream commands. The completion audit
  remains `8/15` complete, with the full-grid primary downstream gates
  `missing_or_empty_or_header_only`.
- Added `severity` and `blocker` columns to `slop-audit-phase3-completion`.
  The current incomplete full-grid and full-grid-primary rows are now marked
  `critical`, with concise blocker text explaining whether generation,
  finalization, or downstream artifacts are missing.
- Ninth live checkpoint at `2026-06-16T21:39:00Z`: the base/t0 SGLang shard is
  still alive as PID `1032`, has written `9216/40000` rows, or nine
  128-prompt chunks. The refreshed plan and finalization readiness record
  temperature `0.0` at `9216/160000`, temperatures `0.7` and `1.0` at
  `0/160000`, and zero enabled downstream commands. The completion audit
  remains `8/15` complete with the new severity/blocker columns.
- Clarified the written status to match the active compute policy: the
  abandoned path is the Torch/Transformers OLMo full-grid route, not
  original-scope OLMo parity itself. Pure generation work now uses SGLang by
  default, including original-scope full-grid recovery; Transformers
  batch-size-32 recovery remains legacy context unless SGLang hits a concrete
  loader or runtime blocker.
- Expanded the automatic and manual Phase 3 completion audit gates from
  `15` to `39` requirements so the snapshot now explicitly checks retained
  data-rate inputs, reduced OLMo generation grids, temperature and long-output
  compounding, SmolLM3 production generation/compounding, Biber-lite/style
  signature outputs, sparse-support addendum artifacts, cross-ladder
  correlations, and the OLMo Think/RL-Zero stretch comparison. The refreshed
  audit reports `32/39` complete; the seven incomplete gates are the full
  OLMo grid, primary t=1 four-stage slice, and five full-grid primary
  downstream artifacts.
- Tenth live checkpoint at `2026-06-16T21:49:00Z`: the base/t0 SGLang shard is
  still alive as PID `1032`, has written `10240/40000` rows, or ten
  128-prompt chunks. The refreshed plan and finalization readiness record
  temperature `0.0` at `10240/160000`, temperatures `0.7` and `1.0` at
  `0/160000`, and zero enabled downstream commands.
- Restarted only the detached supervisor so the live monitor picks up the
  expanded 39-gate completion audit command. The active SGLang generation
  process remains PID `1032`; the new supervisor is PID `15597`. Its first
  loop refreshed the SGLang plan, finalization readiness, and `32/39`
  completion audit, then detected the active shard rather than launching a
  duplicate.
- Added a focused continuation-plan regression test for the next critical
  handoff: after a `t=0` shard has complete JSONL and summary outputs, a stale
  completed selection file is ignored and `--priority-temperature 1.0` selects
  the `t=1.0` shard before `t=0.7`. Verification:
  `uv run pytest tests/test_continue_phase2_generation_plan.py`,
  `uv run ruff check src/slop_sftdiv/cli/continue_phase2_generation_plan.py tests/test_continue_phase2_generation_plan.py`,
  and `uv run ty check src/slop_sftdiv/cli/continue_phase2_generation_plan.py`.
- Factored the 39-gate Phase 3 completion audit command into
  `scripts/phase3_refresh_completion_audit.sh` and changed
  `scripts/phase3_full_sglang_supervisor.sh` to call that helper. The runbook
  now points manual refreshes at the same helper, reducing drift between the
  automatic and manual audit paths.
- Eleventh live checkpoint at `2026-06-16T21:57:00Z`: the base/t0 SGLang shard
  is still alive as PID `1032`, has written `11264/40000` rows, or eleven
  128-prompt chunks. The refreshed plan and finalization readiness record
  temperature `0.0` at `11264/160000`, temperatures `0.7` and `1.0` at
  `0/160000`, and zero enabled downstream commands. Restarted only the
  supervisor to use the new helper; the live supervisor is PID `16614`, and
  its first loop refreshed the `32/39` completion audit and detected the
  active shard instead of launching a duplicate.
- Added guarded downstream finalization automation to
  `scripts/phase3_full_sglang_supervisor.sh`. Each loop now checks the
  refreshed finalization readiness summary; when `Enabled commands` is
  nonzero, the supervisor runs the generated pending-work finalization script,
  refreshes the finalization summary again, and then refreshes the completion
  audit. With the current `11264/160000` temperature-0 slice and incomplete
  primary t=1 slice, this remains a no-op until a complete slice is ready.
- Restarted only the supervisor to pick up the ready-finalization hook. The
  active SGLang generation process remains PID `1032`; the new supervisor is
  PID `17195`. Its first loop refreshed the plan, finalization summary, and
  `32/39` audit, saw zero enabled finalization commands, and detected the
  active shard instead of launching a duplicate.
- Twelfth live checkpoint at `2026-06-16T22:03:00Z`: the base/t0 SGLang shard
  is still alive as PID `1032`, has written `12288/40000` rows, or twelve
  128-prompt chunks. The refreshed plan and finalization readiness record
  temperature `0.0` at `12288/160000`, temperatures `0.7` and `1.0` at
  `0/160000`, and zero enabled downstream commands. The completion audit
  remains `32/39` complete.
- Added a PID-file singleton guard to
  `scripts/phase3_full_sglang_supervisor.sh` so an accidental second
  supervisor exits if the PID file points at a live supervisor process.
  Verified by attempting a duplicate `setsid -f` launch: the PID file remained
  `17195`, the existing supervisor stayed alive, and the log recorded
  `supervisor already running pid=17195; exiting`.
- Added a bounded supervisor-log rotation guard. On startup the monitor now
  rotates `LOG_OUTPUT` when it exceeds `MAX_LOG_BYTES` (default `5000000`),
  keeping future diagnostics focused on the current run rather than the full
  sequence of previous supervisor restarts.
- Verified the rotation guard with a temporary-log smoke test using
  `MAX_LOG_BYTES=4` and a disk-guard exit path, then restarted only the
  supervisor so the guard is live. The active SGLang generation process
  remains PID `1032`; the live supervisor is PID `18430`. Its first loop
  refreshed the current `12288/40000` state and detected the active shard
  instead of launching a duplicate.
- Recorded the final compute-policy clarification in the integrated
  conclusion report and production runbook: SGLang is the active backend for
  pure generation tasks, including the original-scope OLMo recovery grid;
  Transformers batch-size-32 generation is legacy context. The bounded report
  remains driven by completed purpose-built SGLang panels, while original
  parity continues through resumable SGLang shards, prompt chunk sizing,
  JSONL resume checkpoints, and prioritized `t=1.0` shard ordering.
- Thirteenth live checkpoint at `2026-06-16T22:11:00Z`: the base/t0 SGLang
  shard is still alive as PID `1032`, has written `13312/40000` rows, or
  thirteen 128-prompt chunks. Refreshed the SGLang plan, full-grid
  finalization readiness, and completion audit manually; the plan and
  readiness now record `13312` existing rows, the finalization script still
  has zero enabled commands, and the completion audit remains `32/39`
  complete.
- Added `slop-audit-sglang-generation-cache`, a structural JSONL integrity
  audit for SGLang generation caches. It checks required fields, duplicate
  `(source, prompt_id, source_row_index, completion_index, temperature,
  top_p)` keys, expected backend/model/temperature/top-p metadata,
  completion-index bounds, generated-token/feature-token sanity, empty
  generations, and parsed `features_json`/`repeated_features_json`.
- Wired the new integrity audit into `scripts/phase3_full_sglang_supervisor.sh`.
  Each monitor loop now refreshes
  `artifacts/phase3/analysis/olmo3_full_sglang_existing_cache_integrity.{csv,md}`
  for existing full-grid caches before refreshing the completion audit.
  Verification: `uv run pytest tests/test_audit_sglang_generation_cache.py`,
  `uv run ruff check src/slop_sftdiv/cli/audit_sglang_generation_cache.py tests/test_audit_sglang_generation_cache.py`,
  `uv run ty check src/slop_sftdiv/cli/audit_sglang_generation_cache.py`,
  and `bash -n scripts/phase3_full_sglang_supervisor.sh`.
- Restarted only the detached supervisor to pick up the integrity-audit
  refresh. The active SGLang generation process remained PID `1032`; the new
  supervisor is PID `20500`. Its first loop refreshed the plan, finalization
  readiness, integrity audit, and `32/39` completion audit, then detected the
  active shard instead of launching a duplicate.
- Fourteenth live checkpoint at `2026-06-16T22:19:00Z`: the base/t0 SGLang
  shard has written `14336/40000` rows, or fourteen 128-prompt chunks. The
  refreshed plan and readiness record temperature `0.0` at `14336/160000`;
  the integrity audit reports `1/1` existing caches passed with `14336`
  unique generation keys and zero structural failures.
- Promoted the SGLang cache-integrity audit into the Phase 3 completion audit.
  `slop-audit-phase3-completion` now accepts `--integrity-audit` and adds a
  `full_grid_sglang_cache_integrity` gate that is critical if the audit is
  missing or any cache fails. `scripts/phase3_refresh_completion_audit.sh`
  passes the live integrity CSV, and the supervisor forwards
  `INTEGRITY_AUDIT` overrides into that helper. Refreshed snapshot:
  `33/40` requirements complete; the new integrity gate is complete, and the
  remaining seven blockers are unchanged.
- Restarted only the detached supervisor to pick up the promoted integrity
  gate in the automatic completion-audit path. The active SGLang generation
  process remained PID `1032`; the new supervisor is PID `21542`. Its first
  loop refreshed the plan, finalization readiness, integrity audit, and
  `33/40` completion audit, then detected the active shard instead of
  launching a duplicate.
- Added explicit prerequisite checking to
  `slop-materialize-phase3-full-sglang-finalize-plan`. The finalization
  readiness summary now includes a `Prerequisite Inputs` table, and commands
  with missing auxiliary inputs are commented out instead of being enabled.
  Downstream spectrum/classifier/cross-ladder commands inherit upstream
  prerequisite failures, so the supervisor will not launch a command that is
  guaranteed to fail because an input is absent.
- Promoted finalization prerequisites into the Phase 3 completion audit. The
  refreshed audit reports `34/41` complete: the new prerequisite gate and the
  SGLang cache-integrity gate both pass, while the seven critical full-grid
  generation/downstream blockers remain.
- Fifteenth live checkpoint at `2026-06-16T22:28:00Z`: the base/t0 SGLang
  shard has written `15360/40000` rows, or fifteen 128-prompt chunks. The
  refreshed plan, finalization readiness, integrity audit, and completion
  audit all agree on the `15360` row count; integrity remains `1/1` passing.
- Extended `slop-audit-sglang-generation-cache` to verify prompt-group
  completion coverage. In addition to duplicate generation keys and row
  metadata, the audit now checks that every observed
  `(source, prompt_id, source_row_index, temperature, top_p)` prompt group has
  exactly the plan's expected number of completion indices. The live base/t0
  cache reports `1920` prompt groups and zero incomplete or overfull groups.
- Added an integrity-before-finalization hard gate to
  `scripts/phase3_full_sglang_supervisor.sh`. If the finalization readiness
  summary has enabled commands, the supervisor now requires the refreshed
  SGLang cache-integrity CSV to exist and have every existing cache marked
  passing before it runs the generated finalization script. A failing isolated
  smoke test exited with status `3`, wrote the expected block message, and did
  not invoke the fake finalization script.
- Restarted only the detached supervisor to pick up the new hard gate. The
  active SGLang generation process remained PID `1032`; the new supervisor is
  PID `23911`. Its first loop refreshed the plan, finalization readiness,
  integrity audit, and `34/41` completion audit, then detected the active
  shard instead of launching a duplicate.
- Sixteenth live checkpoint at `2026-06-16T22:37:00Z`: the base/t0 SGLang
  shard has written `16384/40000` rows, or sixteen 128-prompt chunks. The
  integrity audit reports `1/1` existing caches passed with `16384` unique
  generation keys, `2048` prompt groups, and zero incomplete or overfull
  prompt groups.
- Hardened `slop-continue-phase2-generation-plan` so current filesystem
  evidence wins over a stale `completed=True` value in the plan CSV. A shard
  is now skipped only when its current JSONL row count and summary output prove
  completion. This prevents a stale sidecar plan from silently skipping a
  required full-grid shard after the active SGLang job exits. Added the
  regression test `test_continue_phase2_generation_plan_ignores_stale_completed_flag`;
  `uv run pytest tests/test_continue_phase2_generation_plan.py`,
  `uv run ruff check`, and `uv run ty check` all pass for the touched launcher
  files.
- Switched full-grid compute ordering toward the Phase 3-critical primary
  temperature slice. Paused the supervisor, waited for the live base/t0 SGLang
  shard to flush its next JSONL checkpoint, and stopped PID `1032` after
  `18432/40000` rows. The integrity audit passes on this partial cache with
  `18432` unique generation keys, `2304` prompt groups, and zero structural
  failures.
- Refreshed the SGLang plan and launched base/t1 as PID `25449` using
  `slop-continue-phase2-generation-plan --execute --allow-partial-retry
  --priority-temperature 1.0`. Restarted the detached supervisor as PID
  `25515`; its first loop detected the active base/t1 shard at `0/40000`
  instead of launching a duplicate. The completion audit remains `34/41`, with
  full-grid generation now at `18432/480000` and the primary t=1 slice still
  at `0/160000` until base/t1 starts flushing rows.
- Base/t1 flushed its first 128-prompt checkpoint: `1024/40000` rows. After
  refreshing the SGLang plan, finalization readiness, integrity audit, and
  completion audit, the live blockers are full-grid generation
  `19456/480000`, primary t=1 generation `1024/160000`, and the five
  full-grid primary downstream artifacts. The integrity audit now covers two
  partial caches and passes `2/2`: base/t0 at `18432` rows and base/t1 at
  `1024` rows, with zero bad prompt groups or structural failures.
- Base/t1 flushed a second 128-prompt checkpoint: `2048/40000` rows. Refreshed
  plan/finalization/integrity/completion artifacts now report full-grid
  generation `20480/480000`, primary t=1 generation `2048/160000`, and
  integrity `2/2` passing. Base/t1 remains active as PID `25449`, the
  supervisor remains active as PID `25515`, and GPU utilization is `100%`.
- Base/t1 flushed a third 128-prompt checkpoint: `3072/40000` rows. Refreshed
  plan/finalization/integrity/completion artifacts now report full-grid
  generation `21504/480000`, primary t=1 generation `3072/160000`, and
  integrity `2/2` passing. The active worker and supervisor remain PID `25449`
  and PID `25515`, respectively.
- Base/t1 flushed a fourth 128-prompt checkpoint: `4096/40000` rows. Refreshed
  plan/finalization/integrity/completion artifacts now report full-grid
  generation `22528/480000`, primary t=1 generation `4096/160000`, and
  integrity `2/2` passing. The base/t1 cache now has `512` complete prompt
  groups and zero structural failures.
- Base/t1 flushed a fifth 128-prompt checkpoint: `5120/40000` rows. Refreshed
  plan/finalization/integrity/completion artifacts now report full-grid
  generation `23552/480000`, primary t=1 generation `5120/160000`, and
  integrity `2/2` passing. The base/t1 cache now has `640` complete prompt
  groups and zero structural failures.
- Base/t1 flushed a sixth 128-prompt checkpoint: `6144/40000` rows. Refreshed
  plan/finalization/integrity/completion artifacts now report full-grid
  generation `24576/480000`, primary t=1 generation `6144/160000`, and
  integrity `2/2` passing. The base/t1 cache now has `768` complete prompt
  groups and zero structural failures.
- Base/t1 flushed a seventh 128-prompt checkpoint: `7168/40000` rows.
  Refreshed plan/finalization/integrity/completion artifacts now report
  full-grid generation `25600/480000`, primary t=1 generation
  `7168/160000`, and integrity `2/2` passing. The base/t1 cache now has
  `896` complete prompt groups and zero structural failures.
- Base/t1 flushed an eighth 128-prompt checkpoint: `8192/40000` rows.
  Refreshed plan/finalization/integrity/completion artifacts now report
  full-grid generation `26624/480000`, primary t=1 generation
  `8192/160000`, and integrity `2/2` passing. The base/t1 cache now has
  `1024` complete prompt groups and zero structural failures.
- Paused the original-scope OLMo full-grid runner after the user clarified
  that the intended Phase 3 production path should be much smaller than
  `480000` generations. Stopped supervisor PID `25515`; base/t1 worker PID
  `25449` required SIGKILL after SIGTERM did not release the GPU. The partial
  caches remain intact at `26624` total full-grid rows (`18432` base/t0 and
  `8192` base/t1), and the A100 memory was released.
- Realigned the Phase 3 completion audit to the lightweight production scope:
  `scripts/phase3_refresh_completion_audit.sh` now passes
  `--full-grid-scope optional`, keeps the original full-grid rows as
  optional/addendum evidence, and requires the reduced OLMo, SmolLM3,
  Biber-lite/style-signature, cross-ladder, sparse-support, and stretch
  artifacts. Regenerated
  `artifacts/phase3/analysis/phase3_completion_audit_snapshot.md`; it now
  reports `30/30` required rows complete, `34/41` all tracked rows complete,
  and `4/11` optional/addendum rows complete.

## 2026-06-18 - Phase 1 English-filtered fixed-sample rerun completed

- Replaced the previous character-composition language screen with
  source-metadata-first Lingua language detection and reran the revised Tier-1
  Phase 1 census over the new fixed retained English samples:
  40,000 Dolci SFT target rows, 40,000 Dolci DPO pairs expanded to 80,000
  chosen/rejected rows, and 5,740 retained Dolma 3 documents from the 80k scan.
  The sampling pass filtered 13,125 SFT source records, 7,541 DPO source pairs,
  and 3,868 Dolma records. W&B was disabled for these local reruns.
- Regenerated the scoped Tier-1 inputs:
  `artifacts/stage1/census/olmo3_dolci_sft_40k_revised_tier1_feature_rates.csv`,
  `artifacts/stage1/census/olmo3_dolci_dpo_40k_retained_revised_tier1_feature_rates.csv`,
  `artifacts/stage1/census/olmo3_dolci_dpo_40k_retained_revised_tier1_pair_deltas.csv`,
  and
  `artifacts/stage1/census/olmo3_dolma3_80k_scan_revised_tier1_feature_rates.csv`.
- Reassembled the canonical Phase 1 artifacts:
  `artifacts/stage1/census/feature_rates_by_corpus.parquet` (`36` rows),
  `artifacts/stage1/census/feature_rates_by_stratum.parquet` (`99` rows),
  `artifacts/stage1/census/preference_pair_deltas.parquet` (`360,000` rows),
  and `artifacts/stage1/census/census_summary.md`.
- Ran full pybiber over all three retained samples after fixing duplicate
  document IDs in the pybiber CLI by row-qualifying colliding IDs before
  calling pybiber. Full pybiber outputs now exist with complete retained-row
  coverage for Dolma (`5,740` returned docs x `67` features), Dolci SFT
  (`40,000` returned docs x `67` features), and Dolci DPO (`80,000` returned
  docs x `67` features), each with wide and long CSV outputs under
  `artifacts/stage1/census/`.
- The earlier pybiber row shortfall is resolved for the English-filtered
  samples; language filtering is now part of the Phase 1 sample definition.
- Added the token-weighted full-pybiber register analysis:
  `artifacts/stage1/census/phase1_pybiber_register_means.csv`,
  `artifacts/stage1/census/phase1_pybiber_register_deltas.csv`, and
  `docs/experiments/phase1_pybiber_register_analysis.md`. The substantive read
  is that alignment data shift away from narrative/personal web prose toward
  nominalized, adjectival, answer-like exposition; this does not support a
  generic "alignment adds hedging" claim.
- Updated `docs/experiments/phase1_conclusions.md`,
  `docs/experiments/phase1_gap_audit.md`, `EXPERIMENTS.md`, and
  `artifacts/stage1/census/census_summary.md` to reflect the fixed-sample
  rerun and full pybiber support. Focused verification passed:
  `uv run pytest tests/test_assemble_phase1.py` and
  `uv run pytest tests/test_pybiber_full.py`.

## 2026-06-18 - Phase 4 512-document exact Tier-3 production run completed

- Completed the production-grade Phase 4 exact sequence-mass Tier-3 OLMo grid
  under
  `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/`.
  All four stages now have opportunity and summary CSVs: base, SFT, DPO, and
  final.
- Assembled:
  `olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv`,
  `olmo3_phase4_tier3_512_exact_sequence_primary_comparison.csv`,
  `olmo3_phase4_tier3_512_exact_sequence_stage_grid_summary.md`,
  `olmo3_phase4_tier3_512_exact_sequence_stage_effects.csv`, and
  `olmo3_phase4_tier3_512_exact_sequence_stage_effects_summary.md`.
- The paired stage-effects table contains `33` comparison-feature rows, with
  `31` BH-FDR-significant rows at `alpha=0.05`.
- Primary process-framing raw AF rises across the OLMo ladder from `1.401`
  (base) to `1.957` (final). Neutral-normalized process-framing AF declines
  from `7.701` to `6.056` because the neutral common-control AF rises faster
  than process framing.
- Refreshed `docs/experiments/phase4_tier3_teacher_forced_addendum.md`,
  `docs/experiments/phase4_completion_audit.md`, and
  `docs/experiments/phase4_production_runbook.md` with the 512-document
  results.

## 2026-06-18 - Paper claim matrix added

- Added `docs/experiments/paper_claim_matrix.md` as the current
  paper-facing authority for bounded claims, evidence sources, claim strength,
  caveats, and allowed wording.
- Added `docs/experiments/paper_scaffold.md` to turn the claim matrix into a
  manuscript structure, including title, thesis, abstract skeleton, section
  plan, and figure/table inventory.
- Updated `EXPERIMENTS.md` and the status memos to point readers to the claim
  matrix. The matrix keeps the central thesis bounded: post-training shifts
  style toward answer-register prose, selected surface markers amplify at
  feature- and ladder-specific stages, and detector-discovered Tier-3 features
  remain provisional until human perceptibility and precision validation are
  complete.

## 2026-06-18 - Draft paper figures rendered

- Added `slop-render-paper-figures`, a reproducible renderer for paper draft
  SVGs from cached artifacts, with a smoke test in
  `tests/test_render_paper_figures.py`.
- Rendered five draft figures under `artifacts/paper/figures/`:
  `figure1_pybiber_register_selected.svg`,
  `figure2_eqbench_stage_scores.svg`,
  `figure3_olmo_slop_lexicon_views.svg`,
  `figure4_cross_ladder_af_scatter.svg`, and
  `figure5_phase4_tier3_raw_af.svg`.
- Updated `docs/experiments/paper_scaffold.md` so the figure inventory points
  at the rendered draft SVGs. Remaining figure work is polish and deciding
  whether Figure 3 should expand from the headline `slop_lexicon` view to a
  broader spectrum panel.

## 2026-06-18 - Precision-validation status table refreshed

- Added `slop-summarize-precision-validation`, a local status summarizer that
  combines multiple label CSVs and hit queues into
  `artifacts/stage1/validation/precision_validation_status.csv` and
  `docs/experiments/precision_validation_status.md`.
- Refreshed the status table with the labels available at that point plus four
  hit queues, including the bounded direct `stock_openers_closers` attempt.
  The then-current independent core gate status was `2/6` passing:
  `contrastive_negation` and `stock_openers` passed the interim gate, while
  `rule_of_three_approx`, `slop_lexicon`, `stock_closers`, and the pooled
  `stock_openers_closers` view still needed additional treatment.
- The direct bounded queue for `stock_openers_closers` produced zero retained
  hits, so the pooled feature should remain a derived convenience metric unless
  a broader scan is deliberately budgeted.
- Updated `docs/experiments/phase1_gap_audit.md`,
  `docs/experiments/precision_validation.md`,
  `docs/experiments/status_memo_2026_06_17.md`,
  `docs/experiments/project_status_memo_2026-06-17.md`,
  `docs/experiments/paper_claim_matrix.md`, and
  `docs/experiments/paper_scaffold.md` to reference the versioned status
  table and the current validation caveat.

## 2026-06-18 - Paper table draft added

- Added `docs/experiments/paper_tables.md` with manuscript-oriented draft
  tables for data/measurement scope, selected full-pybiber register means,
  claim strength bands, Tier-1 precision-validation status, and paper-safe
  negative claims.
- Updated `docs/experiments/paper_scaffold.md` to point the figure/table
  inventory and immediate paper-prep queue at the new table pack.

## 2026-06-18 - Paper Methods draft added

- Added `docs/experiments/paper_methods_draft.md`, a manuscript-facing Methods
  draft that explains the bounded study design, OLMo and SmolLM3 ladders,
  feature tiers, pybiber, EQ-Bench score bridge, teacher-forced propensity,
  SGLang generation, compounding, amplification-spectrum assembly, detector
  discovery, statistical treatment, and caveats.
- Updated `docs/experiments/paper_scaffold.md` and `EXPERIMENTS.md` to list
  the Methods draft as part of the active paper artifact set.

## 2026-06-18 - Paper Results draft added

- Added `docs/experiments/paper_results_draft.md`, a manuscript-facing Results
  draft organized by claim IDs. It covers the pybiber register shift,
  EQ-Bench score bridge, OLMo `slop_lexicon` DPO propensity peak and
  compounding, SmolLM3 stage-localization difference, Phase 4 Tier-3 detector
  candidates, and the bounded overall thesis.
- Updated `docs/experiments/paper_scaffold.md` and `EXPERIMENTS.md` to list
  the Results draft as part of the active paper artifact set.

## 2026-06-18 - Paper introduction/discussion draft added

- Added `docs/experiments/paper_intro_discussion_draft.md`, a
  manuscript-facing draft for the Introduction, Discussion, Limitations, and
  Conclusion. It frames "slop style" as a localization problem across corpus
  inheritance, fixed-context propensity, free-running emission, and
  self-conditioning.
- The draft preserves the bounded thesis: post-training data shift toward
  answer-register prose; selected markers amplify feature- and
  ladder-specifically; EQ-Bench is an aggregate bridge; and detector-derived
  Tier-3 features remain candidates until human perceptibility and precision
  validation.
- Updated `docs/experiments/paper_scaffold.md` and `EXPERIMENTS.md` to list
  the introduction/discussion draft as part of the active paper artifact set.

## 2026-06-18 - Integrated manuscript draft added

- Added `docs/experiments/paper_manuscript_draft.md`, an end-to-end paper
  draft that stitches the introduction, methods, results, discussion,
  limitations, conclusion, figure/table callouts, and appendix pointers into a
  single readable manuscript skeleton.
- Updated `docs/experiments/paper_scaffold.md` and `EXPERIMENTS.md` to list
  the integrated manuscript draft as the current stitched paper artifact.

## 2026-06-18 - Citation plan added

- Added `docs/experiments/paper_citation_plan.md`, a related-work and
  bibliography plan organized by citation slot: AI-writing detection, Biber
  register work, preference optimization, open model/data ladders, EQ-Bench,
  teacher-forced/generation methods, and detector attribution/clustering.
- The plan intentionally records source requirements rather than unverified
  bibliography entries; each slot requires checking title, authors, year,
  venue, and URL/DOI against a primary source before entering a final
  references file.
- Updated `docs/experiments/paper_scaffold.md`,
  `docs/experiments/paper_manuscript_draft.md`, and `EXPERIMENTS.md` to point
  to the citation plan.

## 2026-06-18 - Initial checked bibliography added

- Added `docs/experiments/paper_reference_sources.md`, a primary-source
  inventory for the initial verified citation set. It records source URLs,
  what each source supports, and which citation slots still need verification.
- Added `docs/experiments/paper_references.bib`, an initial BibTeX file for
  checked sources including OLMo 3, OLMo/Dolci Hugging Face cards, SmolLM3,
  EQ-Bench Slop Score, DPO, ModernBERT, SGLang, HAP-E/LLM style work,
  pybiber, Integrated Gradients, and HDBSCAN.
- Updated `docs/experiments/paper_citation_plan.md`,
  `docs/experiments/paper_scaffold.md`,
  `docs/experiments/paper_manuscript_draft.md`, and `EXPERIMENTS.md` to link
  the checked source inventory and BibTeX draft.

## 2026-06-18 - Integrated manuscript citation pass

- Added a compact Related Work section to
  `docs/experiments/paper_manuscript_draft.md` using only the verified initial
  BibTeX set.
- Inserted citation keys for open model/data provenance, DPO, SmolLM3,
  EQ-Bench Slop Score, pybiber, HAP-E, ModernBERT, Integrated Gradients,
  HDBSCAN, and SGLang.
- Renumbered the integrated manuscript after adding Related Work and validated
  that all 18 cited keys resolve to `docs/experiments/paper_references.bib`
  with no duplicate or unused BibTeX keys.
- Updated `docs/experiments/paper_scaffold.md` and
  `docs/experiments/paper_citation_plan.md` so the remaining citation work is
  limited to the still-unverified slots rather than the already completed
  initial citation pass.

## 2026-06-18 - Second bibliography verification pass

- Added checked BibTeX/source-inventory entries for Biber's
  `Variation across Speech and Writing`, Shaib et al.'s AI slop taxonomy
  paper, Anchored Preference Optimization, verbosity bias in LLM preference
  labeling, and the `Alibaba-NLP/gte-large-en-v1.5` model card.
- Wired those references into `docs/experiments/paper_manuscript_draft.md`
  for register-analysis background, human-perceived slop taxonomy framing,
  APO/SmolLM3 method context, preference-style-bias background, and Phase 4
  embedding-model provenance.
- Updated `docs/experiments/paper_citation_plan.md` and
  `docs/experiments/paper_scaffold.md` so the remaining citation gaps are
  optional background slots rather than missing core provenance for the current
  manuscript.

## 2026-06-18 - Calibration and generation-background citations added

- Added checked BibTeX/source-inventory entries for Guo et al. on neural
  calibration, Holtzman et al. on neural text degeneration/nucleus sampling,
  and Welleck et al. on unlikelihood training for repetitive text generation.
- Cited these in `docs/experiments/paper_manuscript_draft.md` as background
  for neutral-normalized AF caveats and generation-time compounding/repetition
  context.
- Updated `docs/experiments/paper_citation_plan.md` and
  `docs/experiments/paper_scaffold.md` so the remaining optional citation gap
  is mainly teacher-forcing/LM-scoring background if the final Methods section
  needs it.

## 2026-06-18 - Figure/table manifest added

- Added `docs/experiments/paper_figure_table_manifest.md`, defining the
  current five-figure and five-table paper package with draft captions, source
  artifacts, intended claims, caveats, and manuscript placement.
- Replaced draft "callout" prose in
  `docs/experiments/paper_manuscript_draft.md` with normal numbered
  figure/table references tied to the manifest.
- Updated `docs/experiments/paper_scaffold.md` and `EXPERIMENTS.md` so the
  manifest is discoverable from the top-level paper artifact set.

## 2026-06-18 - Claim evidence map added

- Added `docs/experiments/paper_claim_evidence_map.md`, mapping C1-C14 from
  the paper claim matrix to manuscript sections, figure/table support, primary
  evidence artifacts, and residual caveats.
- Linked the map from `EXPERIMENTS.md`,
  `docs/experiments/paper_scaffold.md`, and
  `docs/experiments/paper_manuscript_draft.md`.
- This gives the manuscript a review trail from bounded claim wording to the
  exact current evidence package without changing the claim matrix itself.

## 2026-06-18 - Paper readiness audit added

- Added `docs/experiments/paper_readiness_audit.md`, separating paper-grade
  evidence, usable-with-caveat evidence, submission blockers, non-blocking
  deferred work, recommended next actions, and submission exit criteria.
- Linked the readiness audit from `EXPERIMENTS.md`,
  `docs/experiments/paper_scaffold.md`, and
  `docs/experiments/paper_manuscript_draft.md`.
- The audit records the current state as paper-draft ready but
  submission-blocked by validation and polish, especially Tier-1 precision
  validation, Phase 4 human perceptibility labels, camera-ready figures/tables,
  and final manuscript formatting.

## 2026-06-18 - Methods draft citation/equation refresh

- Updated `docs/experiments/paper_methods_draft.md` to match the integrated
  manuscript's supported citation set for OLMo/Dolci, SmolLM3/APO, Biber and
  pybiber, EQ-Bench, SGLang, calibration, generation-degeneration background,
  HAP-E, ModernBERT, Integrated Gradients, GTE-large, and HDBSCAN.
- Replaced the inline amplification-factor definition with explicit displayed
  formulas for `AF_f(m)` and `normalized_AF_f(m)`.
- Updated `docs/experiments/paper_scaffold.md` so the Methods queue now
  reflects final prose polish rather than missing citation/equation parity.

## 2026-06-18 - Results draft figure/table alignment

- Updated `docs/experiments/paper_results_draft.md` to link the claim evidence
  map and figure/table manifest, and to identify Figure 1-Figure 5 plus
  Table 2/Table 3/Table 5 placements for the result sections.
- Preserved the existing claim language while making the standalone Results
  draft align with `docs/experiments/paper_manuscript_draft.md` and
  `docs/experiments/paper_figure_table_manifest.md`.
- Updated `docs/experiments/paper_scaffold.md` so the Results queue now
  reflects final prose polish rather than missing figure/table placement.

## 2026-06-18 - Caption and reproducibility appendix draft added

- Added `docs/experiments/paper_caption_appendix_draft.md` with
  submission-facing captions for Figure 1-Figure 5 and Table 1-Table 5, plus a
  reproducibility appendix draft grouping claim controls, Phase 1/pybiber,
  precision validation, EQ-Bench, Phase 3, Phase 4, figures/tables, and
  references.
- Linked the caption/appendix draft from `EXPERIMENTS.md`,
  `docs/experiments/paper_scaffold.md`,
  `docs/experiments/paper_manuscript_draft.md`, and
  `docs/experiments/paper_figure_table_manifest.md`.
- This separates paper-facing caption text from repo artifact provenance, so
  the main manuscript can later replace repo-path appendix pointers with
  submission-style appendix references.

## 2026-06-18 - LaTeX table draft pack wired into checks

- Added `docs/experiments/paper_latex_table_drafts.tex` as a generic
  booktabs/threeparttable draft pack for Table 1, Table 2, Table 4, and the
  current appendix tables.
- Linked the LaTeX table pack from the figure/table manifest, caption appendix,
  readiness audit, and paper scaffold so the Markdown table bodies and
  submission-oriented LaTeX bodies are discoverable together.
- Extended `slop-check-paper-package` and its tests to require the LaTeX table
  pack, reject repo/provenance markers in it, and assert the expected
  table labels/captions are present.
- Verification passed:
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run slop-check-paper-package --root /home/user/slop`; and the broader
  focused paper utility suite covering pybiber intervals, figure rendering,
  EQ-Bench intervals, and Phase 4 human-label summaries.

## 2026-06-18 - Stock closer precision gate advanced

- Added 30 additional manual labels for clear `Let me know if...`
  `stock_closers` hits in
  `artifacts/stage1/validation/labels/revised_phase1_stock_closers_additional_labels_2026-06-18.csv`.
- Regenerated
  `artifacts/stage1/validation/precision_validation_status.csv` and
  `docs/experiments/precision_validation_status.md` with the new label file.
  `stock_closers` now passes the interim gate with 50 labels, 48 exact hits, 2
  false positives, precision `0.960`, and ambiguous rate `0.000`.
- Updated the paper-facing policy, camera-ready Markdown tables, LaTeX table
  draft, manuscript, methods/discussion drafts, readiness audit, claim matrix,
  scaffold, and Phase 1 gap audit from `2/6` to `3/6` core features passing.
  The remaining Tier-1 blockers are `rule_of_three_approx`, `slop_lexicon`,
  and the derived pooled `stock_openers_closers` view.
- Extended `slop-check-paper-package` to require the precision-validation
  status artifacts and fail if `stock_closers` regresses to the stale
  incomplete state.
- Verification passed:
  `uv run pytest tests/test_check_paper_package.py tests/test_summarize_precision_validation.py`;
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py src/slop_sftdiv/cli/summarize_precision_validation.py tests/test_check_paper_package.py tests/test_summarize_precision_validation.py`;
  `uv run slop-check-paper-package --root /home/user/slop`; and a stale
  language scan over the paper-facing docs.

## 2026-06-18 - Slop lexicon span precision gate advanced

- Added 42 additional manual labels for `slop_lexicon` hits in
  `artifacts/stage1/validation/labels/revised_phase1_slop_lexicon_additional_labels_2026-06-18.csv`.
  These labels validate the frozen lexicon word-family/span definition rather
  than making a semantic judgment that every matched word is always
  human-perceived slop in context.
- Regenerated
  `artifacts/stage1/validation/precision_validation_status.csv` and
  `docs/experiments/precision_validation_status.md`. `slop_lexicon` now
  passes the interim span/item precision gate with 50 labels, 50 exact hits,
  precision `1.000`, and ambiguous rate `0.000`.
- Updated the Tier-1 publication policy, camera-ready Markdown tables, LaTeX
  table draft, manuscript, methods/discussion drafts, readiness audit, claim
  matrix, scaffold, precision-validation guide, and Phase 1 gap audit from
  `3/6` to `4/6` core features passing under the previous taxonomy. The
  remaining issue was `rule_of_three_approx`, with the pooled
  `stock_openers_closers` view already best treated as derived.
- Extended `slop-check-paper-package` to fail if `slop_lexicon` regresses to
  the stale underlabeled state and to require the then-current `4/6` precision
  summary.
- Verification passed:
  `uv run pytest tests/test_check_paper_package.py tests/test_summarize_precision_validation.py`;
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py src/slop_sftdiv/cli/summarize_precision_validation.py tests/test_check_paper_package.py tests/test_summarize_precision_validation.py`;
  `uv run slop-check-paper-package --root /home/user/slop`; and a stale
  language scan over paper-facing docs.

## 2026-06-18 - Pooled stock feature demoted to derived validation status

- Updated the precision-validation source of truth so
  `stock_openers_closers` is no longer an independent core regex-validation
  blocker. It remains measured as a pooled opener/closer convenience view, but
  precision claims should be made through the already validated component
  features unless a broader direct scan is deliberately budgeted.
- Regenerated
  `artifacts/stage1/validation/precision_validation_status.csv` and
  `docs/experiments/precision_validation_status.md`. The current independent
  core status is now 4/5 passing: `contrastive_negation`, `slop_lexicon`,
  `stock_openers`, and `stock_closers` pass the interim gate; only
  `rule_of_three_approx` remains blocking for publication-grade independent
  regex claims.
- Updated paper-facing policy/status docs, table drafts, the manuscript, and
  the paper-package checker to require the `4/5` summary and the derived pooled
  stock row.
- Verification passed:
  `uv run pytest tests/test_check_paper_package.py tests/test_summarize_precision_validation.py tests/test_score_labels.py`;
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py src/slop_sftdiv/cli/summarize_precision_validation.py src/slop_sftdiv/cli/score_labels.py tests/test_check_paper_package.py tests/test_summarize_precision_validation.py tests/test_score_labels.py`;
  `uv run slop-check-paper-package --root /home/user/slop`; and a stale
  language scan over current paper/status docs.

## 2026-06-18 - Rule-of-three interim validation completed and demoted

- Added 38 additional sequential labels for `rule_of_three_approx` hits in
  `artifacts/stage1/validation/labels/revised_phase1_rule_of_three_additional_labels_2026-06-18.csv`,
  bringing the interim labeled sample to 50 rows.
- Regenerated
  `artifacts/stage1/validation/precision_validation_status.csv` and
  `docs/experiments/precision_validation_status.md`. The rule-of-three regex
  now has 50 labels, 28 exact hits, 1 partial, 19 false positives, and 2
  ambiguous cases, for precision `0.560` and ambiguous rate `0.040`; it is
  marked `demote`.
- Updated the publication policy, table drafts, manuscript/methods/discussion
  prose, Phase 1 conclusions, Phase 1 gap audit, and paper-package checker to
  treat `rule_of_three_approx` as an exploratory coordinated-triple surface
  rather than a publication-grade independent rhetorical-triad matcher.
- Verification passed:
  `uv run pytest tests/test_check_paper_package.py tests/test_summarize_precision_validation.py tests/test_score_labels.py`;
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py src/slop_sftdiv/cli/summarize_precision_validation.py src/slop_sftdiv/cli/score_labels.py tests/test_check_paper_package.py tests/test_summarize_precision_validation.py tests/test_score_labels.py`;
  `uv run slop-check-paper-package --root /home/user/slop`; and a targeted
  stale-language scan over current paper/status docs.

## 2026-06-18 - Tier-1 validation removed as a submission blocker

- Updated the paper readiness audit, claim matrix, claim-evidence map,
  manuscript, methods draft, discussion draft, Phase 1 conclusions, and Phase
  1 gap audit to reflect the new scoped Tier-1 disposition: four independent
  regex rows pass the interim gate, `rule_of_three_approx` is demoted, and
  `stock_openers_closers` is derived.
- Reclassified the readiness state from a generic validation blocker to a
  Phase 4 human-validation/polish blocker. Tier-1 is now a claim-boundary
  issue enforced by `docs/experiments/paper_tier1_publication_policy.md`, not
  an unresolved submission blocker for the current scoped paper.
- Verification passed:
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_summarize_precision_validation.py tests/test_score_labels.py`;
  and a stale-language scan for old Tier-1 incomplete/blocking phrasing in
  current paper/status docs.

## 2026-06-18 - Manuscript and table-package prose polish

- Polished the integrated manuscript and companion Results/caption/table docs
  to remove remaining report-style wording such as "corpus censi", "Paper
  interpretation", "Figure N should", stale "incomplete validation" captions,
  and local-artifact phrasing in the main methods prose.
- Corrected the camera-ready table pack so Tier-1 precision validation is
  consistently Table 4, matching the manuscript and figure/table manifest, and
  added a `slop-check-paper-package` guard plus test coverage for missing main
  table headings.
- Updated the caption appendix and manifest so Table 4 is described as a
  pass/demote/derived/unlabeled claim-boundary table rather than an incomplete
  validation limitation.
- Verification passed:
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_summarize_precision_validation.py tests/test_score_labels.py`;
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  and a targeted prose scan over current paper docs.

## 2026-06-18 - Phase 4 human-annotation readiness audit added

- Added `slop-audit-phase4-human-package`, which audits the 1,000-row Phase 4
  human-perceptibility annotation package for required fields, 100-row
  per-feature balance, blank label state, duplicate annotation IDs, and
  per-candidate source/record coverage.
- Ran the audit on the current package, producing
  `docs/experiments/phase4_human_annotation_readiness.md`,
  `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_annotation_readiness.csv`,
  and a deterministic 20-row-per-candidate pilot sheet at
  `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_pilot_20_each.csv`.
- Updated the Phase 4 protocol, completion audit, paper readiness audit, claim
  matrix, claim-evidence map, and reproducibility appendix to cite the
  readiness/pilot artifacts while preserving the claim boundary: the package
  is ready for labeling, but human perceptibility remains unlabeled.

## 2026-06-18 - Paper scaffold/status wording tightened

- Cleaned remaining paper-facing scaffold/manuscript wording so Tier-1
  validation is described as scoped pass/demote/derived status rather than as
  an incomplete blocker, and Figure 2 interval generation is no longer listed
  as pending now that EQ-Bench intervals are integrated.
- Updated the integrated manuscript and intro/discussion draft to use
  declarative limitation language for derived pooled features, bounded SGLang
  scope, and Dolci DPO caveats.
- Extended `slop-check-paper-package` with stale-phrase guards for the old
  Tier-1 incomplete wording and stale Figure 2 interval task.
- Verification passed:
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  and `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Main manuscript internal-prose cleanup

- Cleaned remaining workflow-facing phrasing from the integrated manuscript,
  caption appendix, and figure/table manifest, including references to "local
  retained artifacts", "project vocabulary", "current paper package", and
  writing-guardrail table language.
- Reworded Methods and Limitations prose to describe the evidence as filtered
  retained samples, OLMo/SmolLM3 generation/reference slices, SGLang generation
  panels, and an analysis taxonomy rather than internal project machinery.
- Extended `slop-check-paper-package` so those internal phrases are rejected in
  future manuscript/control-doc edits.
- Verification passed:
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  a targeted stale-phrase scan over paper-facing docs; and
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Paper figure readiness audit added

- Added `slop-audit-paper-figures`, which audits the five rendered paper SVGs
  for dimensions, viewBox, editable text, expected title strings,
  deterministic date metadata, and local path/provenance leakage.
- Ran the audit on the current rendered figures. All five figures are marked
  `ready_for_visual_review` in
  `docs/experiments/paper_figure_readiness_audit.md` and
  `artifacts/paper/figures/paper_figure_readiness_audit.csv`.
- Wired the figure-readiness audit into `slop-check-paper-package`, the paper
  readiness audit, the paper scaffold, the figure/table manifest, and the
  reproducibility appendix. The remaining figure blocker is now narrowed to
  human visual review and venue sizing, not basic SVG package hygiene.
- Verification passed:
  `uv run pytest tests/test_check_paper_package.py tests/test_audit_paper_figures.py tests/test_render_paper_figures.py`;
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py src/slop_sftdiv/cli/audit_paper_figures.py tests/test_check_paper_package.py tests/test_audit_paper_figures.py`;
  and `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Paper table readiness audit added

- Added `slop-audit-paper-tables`, which audits the submission-facing Markdown
  and generic booktabs table drafts for required headings, minimum data rows,
  LaTeX captions/labels, and local path/provenance leakage.
- Ran the audit on the current table package. Table 1, Table 2, Table 4, and
  appendix tables A1-A3 are marked `ready_for_venue_styling` in
  `docs/experiments/paper_table_readiness_audit.md` and
  `artifacts/paper/tables/paper_table_readiness_audit.csv`.
- Wired the table-readiness audit into `slop-check-paper-package`, the paper
  readiness audit, the paper scaffold, the figure/table manifest, and the
  reproducibility appendix. The remaining table blocker is now venue-specific
  typography and placement, not basic table package hygiene.

## 2026-06-18 - Manuscript-facing wording cleanup

- Tightened the integrated manuscript, standalone Methods draft, Results
  draft, Intro/Discussion draft, and caption appendix to remove internal
  "current/project/local" status phrasing in favor of submission-facing
  bounded-evidence language.
- Extended `slop-check-paper-package` stale-phrase coverage to the standalone
  manuscript section drafts, so internal phrasing such as
  "project-specific generation slices" and stale "current interim gate"
  wording is rejected outside the integrated manuscript too.
- Verification passed:
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_audit_paper_tables.py`;
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  and a targeted scan over manuscript-facing drafts.

## 2026-06-18 - Citation coverage gate broadened

- Extended `slop-check-paper-package` citation validation from the integrated
  manuscript alone to the manuscript plus standalone Methods, Results, and
  Intro/Discussion drafts, so all manuscript-facing citation keys must resolve
  to `paper_references.bib`.
- Updated `paper_citation_plan.md` to distinguish covered citation slots from
  optional expansion slots; the current bounded draft is citation-complete
  under the verified BibTeX/source inventory, while teacher-forcing,
  SmolTalk/Tulu, and deeper RLVR citations remain optional only if final prose
  expands those claims.
- Added `paper_citation_plan.md` to the required paper package paths and linked
  path checks.
- Verification passed:
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py`; and
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`.

## 2026-06-18 - Claim-language audit added

- Added `slop-audit-paper-claim-language`, which checks the integrated
  manuscript for concrete C1-C14 claim/caveat phrases and the seven
  explicitly forbidden overclaims from the paper claim matrix.
- Ran the audit on the current manuscript, producing
  `docs/experiments/paper_claim_language_audit.md` and
  `artifacts/paper/claims/paper_claim_language_audit.csv`; all 14 claim rows
  and all 7 forbidden-overclaim checks are marked `ready`.
- Wired the claim-language audit into `slop-check-paper-package`, the paper
  readiness audit, the paper scaffold, and the reproducibility appendix so
  claim wording remains mechanically checked as the manuscript changes.
- Verification passed:
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_audit_paper_claim_language.py`; and
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py src/slop_sftdiv/cli/audit_paper_claim_language.py tests/test_check_paper_package.py tests/test_audit_paper_claim_language.py`.

## 2026-06-18 - Manuscript-structure audit added

- Added `slop-audit-paper-manuscript-structure`, which checks the integrated
  manuscript for expected main section order, Result 4.x subsection coverage,
  word-count bounds, figure/table caption counts, and reproducibility appendix
  presence.
- Ran the audit on the manuscript, producing
  `docs/experiments/paper_manuscript_structure_audit.md` and
  `artifacts/paper/manuscript/paper_manuscript_structure_audit.csv`; all nine
  structure checks are marked `ready`.
- Wired the manuscript-structure audit into `slop-check-paper-package`, the
  paper readiness audit, the paper scaffold, and the reproducibility appendix
  so future manuscript edits cannot silently drop sections, captions, or the
  appendix surface.

## 2026-06-18 - Submission-exit audit added

- Added `slop-audit-paper-submission-exit`, which converts the submission exit
  criteria into machine-readable rows for figure/table finalization,
  manuscript structure, claim discipline, Tier-1 boundaries, Phase 4 human
  perceptibility, bibliography/package state, and overall submission status.
- Ran the audit on the paper package, producing
  `docs/experiments/paper_submission_exit_audit.md` and
  `artifacts/paper/submission/paper_submission_exit_audit.csv`.
- The audit intentionally marks `overall_submission_status` as `blocked`,
  with `figure_table_finalization=venue_review_pending` and
  `phase4_human_perceptibility=human_validation_pending`, while manuscript,
  claim, Tier-1, and bibliography/package rows are `ready`.
- Wired the audit into `slop-check-paper-package`, the paper readiness audit,
  paper scaffold, reproducibility appendix, and `EXPERIMENTS.md` so draft-ready
  and submission-ready states remain mechanically distinct.

## 2026-06-18 - Figure rendered-layout review and Figure 4 label cleanup

- Rendered PNG sidecars from the same Matplotlib code used for the paper SVGs
  under `artifacts/paper/figures/visual_review_png/` and inspected Figure 1
  through Figure 5 for legibility, label overlap, legend placement, and data
  occlusion.
- Found repeated per-point labels colliding in Figure 4's lower-left `rule3`
  cluster and upper `slop` cluster. Updated `slop-render-paper-figures` so the
  cross-ladder scatter labels each feature cluster once instead of labeling
  every stage point.
- Re-rendered Figure 1-Figure 5 and reran `slop-audit-paper-figures`; all SVG
  package-readiness rows remain `ready_for_visual_review`.
- Added `docs/experiments/paper_figure_visual_review.md` and
  `artifacts/paper/figures/paper_figure_visual_review.csv`; all five figures
  are marked `visual_review_passed`. The remaining figure-side blocker is now
  final venue sizing/export, not visible layout defects in the rendered draft.

## 2026-06-18 - Table typography review added

- Added `slop-audit-paper-table-typography`, which reviews the LaTeX table
  draft for generic submission-neutral typography conventions: booktabs rules,
  `tabularx`, expected table/table* environments, expected width targets,
  compact sizing, table notes where expected, and absence of vertical rules.
- Ran the audit on `paper_latex_table_drafts.tex`, producing
  `docs/experiments/paper_table_typography_review.md` and
  `artifacts/paper/tables/paper_table_typography_review.csv`; Table 1, Table
  2, Table 4, and appendix tables A1-A3 are marked
  `typography_review_passed`.
- Wired the table typography review into `slop-check-paper-package` and
  `slop-audit-paper-submission-exit`. The submission-exit audit still keeps
  figure/table finalization at `venue_review_pending`, but the remaining table
  work is now target-venue template adaptation rather than generic table-body
  typography.

## 2026-06-18 - Integrated manuscript polish pass

- Edited `paper_manuscript_draft.md` to remove several internal/project-status
  turns of phrase, including the auxiliary-tooling paragraph, the
  opener/closer validation-boundary paragraph, and colloquial wording in the
  pybiber discussion.
- Reran `slop-audit-paper-claim-language` and
  `slop-audit-paper-manuscript-structure`; all C1-C14 claim checks, all seven
  forbidden-overclaim checks, and all nine manuscript-structure checks remain
  `ready`.
- Updated `paper_readiness_audit.md` and `paper_scaffold.md` so the remaining
  manuscript work is described as final venue copyediting rather than an
  unstarted editorial pass.

## 2026-06-18 - Standalone section-draft polish pass

- Tightened `paper_methods_draft.md` so the Methods preamble, Tier-1
  validation-boundary paragraph, pybiber implementation paragraph, EQ-Bench
  bridge description, Tier-3 rerun description, and reproducibility section
  read as manuscript-facing methods rather than command notes.
- Tightened `paper_results_draft.md` so figure/table references use final
  paper-facing language and the closing result section no longer reads as a
  manuscript assembly note.
- Tightened `paper_intro_discussion_draft.md` to match the polished integrated
  manuscript language for the pybiber discussion and Tier-1 limitation.
- Updated `paper_scaffold.md` and `paper_readiness_audit.md` to mark Methods
  Results, and Intro/Discussion as polished drafts whose remaining work is
  final venue copyediting.

## 2026-06-18 - Venue-decision checklist added

- Added `docs/experiments/paper_venue_decision_checklist.md` and
  `artifacts/paper/submission/paper_venue_decision_checklist.csv` to make
  unresolved target-venue choices explicit rather than implicit blockers.
- The checklist tracks `decision_pending` rows for target venue, figure
  dimensions, figure export format, table placement, appendix-table inclusion,
  manuscript template, and the human-labeling requirement for detector
  candidates.
- Wired the checklist into `slop-check-paper-package`, `EXPERIMENTS.md`, the
  paper readiness audit, the paper scaffold, the reproducibility appendix, and
  the regenerated submission-exit audit. The package remains submission-blocked
  until venue decisions and Phase 4 human labels are resolved.

## 2026-06-18 - Abstract/conclusion alignment audit added

- Added `slop-audit-paper-abstract-conclusion`, which checks that the
  manuscript abstract and conclusion preserve the bounded slop-style thesis,
  pybiber/EQ-Bench framing, DPO chosen-enrichment caveat, and Phase 4
  human-validation caveat.
- Ran the audit on the integrated manuscript, producing
  `docs/experiments/paper_abstract_conclusion_audit.md` and
  `artifacts/paper/manuscript/paper_abstract_conclusion_audit.csv`; both
  abstract and conclusion rows are marked `ready`.
- Wired the audit into `slop-check-paper-package`, the paper readiness audit,
  the paper scaffold, the reproducibility appendix, and `EXPERIMENTS.md` so
  abstract/conclusion edits cannot silently drop the key bounded-claim
  language.

## 2026-06-18 - Historical Biber-lite report language retired

- Updated the Phase 2 final/new-reader reports and the Phase 3 completion
  audit so older generated-output Biber-lite/register-proxy artifacts are
  described as archived diagnostics rather than the active Tier-2 feature
  surface.
- Preserved the historical proxy tables for provenance, but redirected active
  paper interpretation to the full pybiber Phase 1 corpus-side register result
  and repeated that generated-output full pybiber is not claimed.
- Rechecked the active paper package after the edits with
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Paper presentation decisions resolved

- Replaced the remaining review-question block in
  `paper_claim_evidence_map.md` with resolved presentation decisions: Figure 3
  remains the headline OLMo `slop_lexicon` mechanism panel; Table 3 and Table
  5 remain appendix/review controls by default; no separate
  teacher-forcing/LM-scoring citation is needed unless final prose broadens
  into a literature claim.
- Updated the paper scaffold, figure/table manifest, and caption appendix so
  the immediate queue no longer asks future readers to re-decide those points.

## 2026-06-18 - Paper limitations audit wired into readiness controls

- Added the integrated-manuscript limitations audit to the paper readiness
  audit, paper scaffold inventory, reproducibility appendix key artifacts, and
  `EXPERIMENTS.md` so bounded open-ladder/sample, Tier-1, sparse denominator,
  DPO, pybiber-generation, EQ-Bench, Phase 4 human-validation, and reduced-grid
  caveats are visible paper controls.
- Tightened `slop-audit-paper-submission-exit` so the `manuscript_body` row now
  requires both the manuscript-structure audit and the limitations audit.
- Regenerated `docs/experiments/paper_submission_exit_audit.md` and
  `artifacts/paper/submission/paper_submission_exit_audit.csv`; the audit now
  marks manuscript body `ready` under the stricter condition while preserving
  the real submission blockers: venue review and Phase 4 human perceptibility
  labels.
- Added `--root` support to `slop-audit-paper-limitations` so it can be run
  with the same repository-root invocation style as the other paper audits.

## 2026-06-18 - Paper reproducibility manifest added

- Added `slop-audit-paper-reproducibility-manifest`, which writes a
  categorized paper-facing evidence manifest with file sizes and SHA-256
  checksums for paper controls, manuscript drafts, audits, figures, Phase 1
  pybiber evidence, EQ-Bench interval evidence, and Phase 4 human-labeling
  materials.
- Generated `docs/experiments/paper_reproducibility_manifest.md` and
  `artifacts/paper/submission/paper_reproducibility_manifest.csv`; the
  manifest currently has 53 ready rows and no missing files.
- Wired the manifest into `slop-check-paper-package`, the paper readiness
  audit, paper scaffold, reproducibility appendix, submission-exit audit, and
  `EXPERIMENTS.md` so final evidence packaging has a checksum surface without
  pretending venue review or Phase 4 human labels are complete.

## 2026-06-18 - Paper numeric claims audit integrated

- Added `slop-audit-paper-numeric-claims`, which checks 15 headline manuscript numeric claims against their source CSV/JSON artifacts after manuscript rounding.
- Generated `docs/experiments/paper_numeric_claims_audit.md` and `artifacts/paper/manuscript/paper_numeric_claims_audit.csv`; all 15 rows currently mark `ready`.
- Wired the numeric audit into `EXPERIMENTS.md`, the paper readiness audit, paper scaffold, reproducibility appendix inventory, package checker, and reproducibility manifest defaults so numeric drift is checked before final paper edits.

## 2026-06-18 - Paper claim-evidence audit integrated

- Added `slop-audit-paper-claim-evidence`, which checks that the paper
  claim-to-evidence map covers C1-C14, that mapped backticked source paths or
  directories exist, and that claim-to-claim evidence inheritance is explicit.
- Generated `docs/experiments/paper_claim_evidence_audit.md` and
  `artifacts/paper/claims/paper_claim_evidence_audit.csv`; all 14 claim rows
  currently mark `ready`.
- Wired the claim-evidence audit into `EXPERIMENTS.md`, the paper readiness
  audit, paper scaffold, reproducibility appendix inventory, submission-exit
  audit, package checker, and reproducibility manifest defaults so evidence-map
  drift is checked before final paper edits.

## 2026-06-18 - Paper citation/source audit integrated

- Added `slop-audit-paper-citations`, which checks manuscript-facing citation
  keys against `paper_references.bib`, checks every BibTeX key against a
  verified source row, checks every source row against BibTeX, and verifies
  populated source/URL/note cells.
- Generated `docs/experiments/paper_citation_audit.md` and
  `artifacts/paper/references/paper_citation_audit.csv`; all six citation and
  source-integrity rows currently mark `ready`.
- Wired the citation audit into `EXPERIMENTS.md`, the paper readiness audit,
  paper scaffold, reproducibility appendix inventory, submission-exit audit,
  package checker, and reproducibility manifest defaults so reference drift is
  checked before final paper edits.

## 2026-06-18 - Paper terminology/readability audit integrated

- Added `slop-audit-paper-terminology`, which checks that the integrated
  manuscript defines slop-style scope, separates corpus/propensity/generation/
  compounding measurements, and preserves pybiber, EQ-Bench, Tier-1, and
  Phase 4 interpretation boundaries for new readers.
- Generated `docs/experiments/paper_terminology_audit.md` and
  `artifacts/paper/manuscript/paper_terminology_audit.csv`; all eight
  terminology rows currently mark `ready`.
- Wired the terminology audit into `EXPERIMENTS.md`, the paper readiness
  audit, paper scaffold, reproducibility appendix inventory, submission-exit
  audit, package checker, and reproducibility manifest defaults so final
  readability edits cannot silently drop the core measurement definitions.

## 2026-06-18 - Pybiber result prose tightened

- Tightened the integrated manuscript and standalone Results draft so the
  pybiber result reads as a substantive register finding: post-training data
  shift away from narrative/personal prose toward noun-heavy, descriptive
  assistant-answer prose, while narrower lexical/rhetorical markers vary on
  top of that broader voice.
- Preserved the claim-language audit's exact C1/C3 guard phrases while adding
  plainer reader-facing explanation for what the pybiber deltas mean.
- Regenerated terminology, claim-language, manuscript-structure,
  abstract/conclusion, submission-exit, and reproducibility-manifest artifacts;
  all manuscript-facing audit rows remain `ready`.

## 2026-06-18 - Phase 4 human annotation codebook added

- Added `docs/experiments/phase4_human_annotation_codebook.md` with concrete
  yes/no/context-dependent decision rules, taxonomy shortcuts, and
  candidate-specific labeling boundaries for the 10 detector-derived Tier-3
  candidate families.
- Wired the codebook into the Phase 4 perceptibility protocol, paper readiness
  audit, paper scaffold, reproducibility appendix, C11 claim evidence trail,
  package checker, and reproducibility manifest defaults.
- The package still remains human-label blocked: the codebook reduces
  annotation ambiguity but does not replace the prepared pilot or full
  1,000-row human-labeling pass.

## 2026-06-18 - Venue adaptation runbook added

- Added `docs/experiments/paper_venue_adaptation_runbook.md`, which converts
  the remaining venue blocker into explicit post-decision steps for figure
  sizing/export, table placement/typography, manuscript template adaptation,
  and the Phase 4 human-labeling decision.
- Wired the runbook into `EXPERIMENTS.md`, the paper readiness audit, paper
  scaffold, reproducibility appendix, package checker, and reproducibility
  manifest defaults.
- The package remains venue-blocked until a target venue is selected and the
  decision checklist rows are resolved; the runbook documents the execution
  path without pretending those decisions have been made.

## 2026-06-18 - Paper reproduction runbook added

- Added `docs/experiments/paper_reproduction_runbook.md`, which records the
  command order for refreshing the paper-facing evidence layer: EQ-Bench
  intervals, rendered figures, table audits, manuscript/claim/citation audits,
  Phase 4 human-label summaries, submission-exit audit, reproducibility
  manifest, and final package check.
- Wired the runbook into `EXPERIMENTS.md`, paper scaffold, readiness audit,
  reproducibility appendix, package checker, and reproducibility manifest
  defaults.
- The runbook explicitly treats cached Phase 1-4 outputs as inputs and does
  not relaunch generation, teacher-forced propensity, full pybiber extraction,
  or detector-training jobs.

## 2026-06-18 - Reviewer FAQ added

- Added `docs/experiments/paper_reviewer_faq.md`, a concise scope and
  interpretation FAQ for likely review questions: slop-style terminology,
  pybiber's corpus-side role, EQ-Bench as aggregate bridge, Dolci DPO caveats,
  reduced-grid scope, SmolLM3 replication pressure, and remaining blockers.
- Wired the FAQ into `EXPERIMENTS.md`, the paper scaffold, readiness audit,
  reproducibility appendix, package checker, and reproducibility manifest
  defaults.
- The FAQ is documentation only; `paper_claim_matrix.md` remains the authority
  for allowed paper claims.

## 2026-06-18 - Phase 4 blind pilot packet added

- Extended `uv run slop-audit-phase4-human-package` to emit a randomized
  annotator-facing blind pilot CSV and a separate unblinding map for the
  existing 20-per-candidate Phase 4 human-perceptibility pilot.
- The blind sheet contains only `blind_id`, matched span, snippet, and blank
  label fields; feature names, candidate labels, source names, and record IDs
  stay in the map until after adjudication.
- Updated the Phase 4 human-perceptibility protocol, annotation codebook, and
  reproducibility manifest defaults so the labeling handoff is checksum-tracked
  and does not rely on the internal pilot sheet.

## 2026-06-18 - Phase 4 blind-label import path added

- Added `uv run slop-import-phase4-blind-labels`, which joins a filled blind
  pilot CSV to the unblinding map and emits canonical JSONL rows compatible
  with `uv run slop-summarize-phase4-human-labels`.
- Verified the importer on the current blank blind pilot as a smoke test:
  200 canonical rows are produced and 0 rows are labeled, preserving the
  current human-validation boundary.
- Updated the Phase 4 protocol, venue adaptation runbook, reproduction
  runbook, paper readiness audit, scaffold, claim matrix, and claim evidence
  map to describe the blind-label import workflow without claiming completed
  human labels.

## 2026-06-18 - Phase 4 label-agreement tooling added

- Added `uv run slop-summarize-phase4-label-agreement`, which compares two
  independently labeled blind pilot CSVs against the unblinding map and writes
  per-feature agreement rows, Cohen's kappa over `human_perceived_slop`, and a
  disagreement CSV for adjudication.
- Updated the human-perceptibility protocol, annotation codebook, reproduction
  runbook, venue adaptation runbook, readiness audit, and paper scaffold so the
  two-annotator workflow has an explicit agreement/adjudication step.
- This remains annotation infrastructure only: current Phase 4 human labels are
  still absent, and detector-derived families remain candidate-only until real
  labels are collected and adjudicated.

## 2026-06-18 - Phase 4 label-adjudication apply path added

- Added `uv run slop-apply-phase4-label-adjudication`, which merges two
  independently labeled blind pilot sheets with filled adjudication rows and
  emits canonical JSONL for `uv run slop-summarize-phase4-human-labels`.
- Extended the disagreement CSV produced by
  `uv run slop-summarize-phase4-label-agreement` with blank
  `adjudicated_human_perceived_slop`, `adjudicated_shaib_taxonomy_label`, and
  `adjudication_notes` columns so it can be used directly as an adjudication
  worksheet.
- Updated the Phase 4 protocol, codebook, reproduction runbook, venue
  adaptation runbook, readiness audit, and scaffold. Human labels are still not
  complete; this only closes the workflow gap between agreement measurement and
  final label summarization.

## 2026-06-18 - Paper package index added

- Added `docs/experiments/paper_package_index.md` as a compact start-here guide
  for new readers: core thesis, main evidence objects, claim boundaries,
  active blockers, and validation commands.
- Wired the index into the paper scaffold, readiness audit, package checker,
  reproducibility manifest defaults, and package-check tests.
- The index deliberately delegates claim authority to
  `docs/experiments/paper_claim_matrix.md`; it is orientation documentation,
  not a new claim surface.

## 2026-06-18 - Package-index alignment audit added

- Extended `uv run slop-audit-paper-abstract-conclusion` with a
  `package_index_alignment` row so the new start-here index is checked against
  the same bounded thesis, pybiber/EQ-Bench boundaries, Phase 4 candidate-only
  caveat, and package-check command.
- Updated `uv run slop-check-paper-package` and its fixtures to require the
  third abstract/conclusion/package-index alignment row.
- Regenerated abstract/conclusion, claim-evidence, claim-language,
  submission-exit, and reproducibility-manifest artifacts; the paper package
  still passes while retaining the human-label and venue blockers.

## 2026-06-18 - Phase 4 full blind annotation packet added

- Extended `uv run slop-audit-phase4-human-package` to emit a blinded
  full-package annotation CSV and separate ID map for all 1,000 Phase 4 human
  perceptibility rows, in addition to the existing 20-per-candidate blind
  pilot packet.
- Wired the full blind packet into the Phase 4 protocol/codebook, paper
  readiness docs, claim evidence wording, package checker, and reproducibility
  manifest. The manifest now includes both full blind files alongside the
  pilot packet.
- Verified the new full blind files parse as 1,000 rows with annotator-facing
  columns limited to blind ID, matched text, snippet, labels, and notes. Human
  labels remain absent; this removes a workflow blocker for collecting them.

## 2026-06-18 - Phase 4 blind packet package checks tightened

- Extended `uv run slop-check-paper-package` to parse both Phase 4 blinded
  annotation packets and their ID maps, rather than only requiring the files to
  exist.
- The package gate now verifies expected row counts, exact annotator-facing
  columns, absence of unblinded feature/source metadata in blind sheets,
  unique blind IDs, matching blind-ID sets between sheets and maps, and
  nonblank feature/annotation IDs in maps.
- Added a regression test that fails when a blind full-package sheet leaks the
  `feature` column. The full paper package still passes.

## 2026-06-18 - Venue-neutral submission draft bundle added

- Added `uv run slop-materialize-paper-submission-bundle`, which stages the
  current manuscript draft, caption appendix, table drafts, BibTeX file,
  SVG/PDF/PNG figure candidates, and claim/readiness controls under
  `artifacts/paper/submission/draft_bundle/`.
- The bundle writes its own `MANIFEST.csv` with source paths, staged paths,
  file sizes, and SHA-256 checksums, plus a README that preserves the
  venue-neutral and human-labeling caveats.
- Wired the bundle into `uv run slop-check-paper-package`, the paper
  reproducibility manifest, readiness/scaffold docs, and reproduction/venue
  runbooks. The bundle currently contains 24 ready files and 0 missing files.

## 2026-06-18 - Paper review-hygiene audit added

- Added `uv run slop-audit-paper-review-hygiene` to check that the integrated
  manuscript main body and paper-facing staged bundle files avoid local paths,
  internal tool commands, draft placeholders, and author placeholder text.
- Wired the review-hygiene CSV/Markdown outputs into
  `uv run slop-check-paper-package`, `uv run slop-audit-paper-reproducibility-manifest`,
  `EXPERIMENTS.md`, the package index, readiness audit, and reproduction runbook.
- Added the hygiene audit to the venue-neutral draft bundle review controls,
  moving the bundle to 25 ready files and 0 missing files after regeneration.

## 2026-06-18 - Paper open-release inventory added

- Added `uv run slop-audit-paper-release-inventory` to map the open-release
  deliverable to concrete feature definitions, opportunity definitions,
  harness code, phase runbooks, and cached paper statistics.
- Wired `paper_release_inventory.md` and
  `artifacts/paper/submission/paper_release_inventory.csv` into the paper
  package checker, reproducibility manifest, package index, readiness audit,
  scaffold, and reproduction runbook.
- Added the release inventory to the venue-neutral draft bundle review
  controls, moving the bundle to 26 ready files and 0 missing files after
  regeneration.

## 2026-06-18 - Paper source/provenance audit added

- Added `uv run slop-audit-paper-source-provenance` to make source-card and
  manuscript interpretation boundaries explicit for Dolci DPO, Dolma sampling,
  full pybiber, and SmolLM3 APO/source claims.
- The audit checks that Dolci DPO is not treated as pure human-preference
  evidence, the retained Dolma reference is not treated as full Dolma, and
  generated-output full pybiber is not claimed.
- Wired `paper_source_provenance_audit.md` and
  `artifacts/paper/submission/paper_source_provenance_audit.csv` into the
  paper package checker, reproducibility manifest, package index, readiness
  audit, scaffold, and reproduction runbook. This was later superseded by the
  reader-glossary bundle update; the current regenerated draft bundle contains
  28 ready files and 0 missing files.

## 2026-06-18 - Phase 4 human annotation handoff bundle added

- Added `uv run slop-materialize-phase4-human-handoff`, which creates
  `artifacts/phase4/modernbert_detector_combined_v2_clean/human_annotation_handoff/`.
- The handoff separates annotator-facing blinded CSVs and the codebook under
  `annotator/` from coordinator-only maps/source identifiers under
  `coordinator/private_maps/`.
- The generated manifest records source paths, bundle paths, SHA-256 hashes,
  and redaction status; after the two-annotator assignment update, the current
  run produces 12 ready files, 0 missing files, and 6 redacted annotator CSVs.

## 2026-06-18 - Reader glossary wired into paper package

- Added `docs/experiments/paper_reader_glossary.md` as a new-reader map for
  slop style, register, pybiber, EQ-Bench Slop Score, Tier 1/2/3 features,
  amplification factors, model ladders, and evidence layers.
- Wired the glossary into `uv run slop-check-paper-package`,
  `uv run slop-audit-paper-reproducibility-manifest`, and
  `uv run slop-materialize-paper-submission-bundle`.
- Regenerated the venue-neutral draft bundle and reproducibility manifest. The
  current bundle has 28 ready files and 0 missing files; the reproducibility
  manifest has 99 ready rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py src/slop_sftdiv/cli/audit_paper_reproducibility_manifest.py src/slop_sftdiv/cli/materialize_paper_submission_bundle.py tests/test_check_paper_package.py tests/test_materialize_paper_submission_bundle.py`;
  `uv run pytest tests/test_check_paper_package.py tests/test_materialize_paper_submission_bundle.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Pybiber conclusion wording tightened

- Revised the manuscript conclusion and the source introduction/discussion
  conclusion draft so the final paper-facing takeaway states the substantive
  pybiber result directly: OLMo/Dolci data shift away from narrative/personal
  prose toward nominalized, adjectival assistant-answer exposition; this is
  not an aggregate hedging increase, and DPO chosen responses are not
  uniformly more slop-like than rejected responses.
- Regenerated the abstract/conclusion audit, the venue-neutral draft bundle,
  and the reproducibility manifest after the manuscript edit.
- Validation passed:
  `uv run slop-audit-paper-abstract-conclusion`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_materialize_paper_submission_bundle.py`.

## 2026-06-18 - Claim-matrix pybiber boundary tightened

- Narrowed the claim-matrix scope summary from "post-training shifts model
  outputs toward polished answer-register prose" to the evidence-supported
  distinction: post-training data shift toward assistant-answer prose, while
  some surface markers in model propensity and sampled output amplify at
  particular stages.
- Regenerated claim-language and claim-evidence audits, then refreshed the
  venue-neutral draft bundle and reproducibility manifest.
- Validation passed:
  `uv run slop-audit-paper-claim-language`;
  `uv run slop-audit-paper-claim-evidence`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_source_provenance.py`.

## 2026-06-18 - Data-vs-output register wording consistency pass

- Scanned paper-facing docs for phrasing that could blur corpus-side full
  pybiber evidence with generated-output style evidence.
- Tightened `paper_results_draft.md` Result 6 so its summary matches the
  integrated manuscript and claim matrix: post-training data shift away from
  narrative/personal prose toward nominalized, adjectival answer-register
  exposition.
- Regenerated claim-language, review-hygiene, and reproducibility-manifest
  artifacts. Validation passed:
  `uv run slop-audit-paper-claim-language`;
  `uv run slop-audit-paper-review-hygiene --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_audit_paper_source_provenance.py`.

## 2026-06-18 - Pybiber output-boundary regression guard added

- Extended `slop-audit-paper-source-provenance` so the paper-scope docs now
  explicitly fail review if they reintroduce the over-broad wording
  "post-training shifts model outputs toward polished answer-register prose."
- Added a regression test covering that exact failure mode in
  `tests/test_audit_paper_source_provenance.py`.
- Regenerated the source-provenance audit, venue-neutral draft bundle, and
  reproducibility manifest. Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_source_provenance.py tests/test_audit_paper_source_provenance.py`;
  `uv run pytest tests/test_audit_paper_source_provenance.py`;
  `uv run slop-audit-paper-source-provenance --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_source_provenance.py`.

## 2026-06-18 - EQ-Bench causal-overclaim guard added

- Tightened `slop-audit-paper-claim-language` with two additional forbidden
  overclaim checks: "The EQ-Bench Slop Score is the causal measurement layer."
  and "EQ-Bench is the causal estimand."
- Added a regression test for the causal-measurement-layer overclaim and
  updated `slop-check-paper-package` to expect 9 forbidden-overclaim rows.
- Regenerated the claim-language audit and reproducibility manifest. The
  current claim-language audit reports C1-C14 ready and forbidden-overclaim
  checks `9/9` passed.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_claim_language.py src/slop_sftdiv/cli/check_paper_package.py tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-claim-language`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_audit_paper_claim_language.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_source_provenance.py`.

## 2026-06-18 - Phase 4 human-perception overclaim guard added

- Tightened `slop-audit-paper-claim-language` with two additional forbidden
  overclaim checks: "Detector clusters are human-perceived slop." and
  "Phase 4 detector clusters are validated human-perceived slop."
- Added a regression test for the detector-clusters overclaim and updated
  `slop-check-paper-package` to expect 11 forbidden-overclaim rows.
- Regenerated the claim-language audit and reproducibility manifest. The
  current claim-language audit reports C1-C14 ready and forbidden-overclaim
  checks `11/11` passed.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_claim_language.py src/slop_sftdiv/cli/check_paper_package.py tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-claim-language`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_audit_paper_claim_language.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_source_provenance.py`.

## 2026-06-18 - Full-grid overclaim guard added

- Tightened `slop-audit-paper-claim-language` with two additional forbidden
  overclaim checks: "We ran the full 5,000 prompt x 8 completion x 3
  temperature OLMo grid." and "The original exhaustive OLMo generation grid is
  complete."
- Added a regression test for the full-grid overclaim and updated
  `slop-check-paper-package` to expect 13 forbidden-overclaim rows.
- Regenerated the claim-language audit and reproducibility manifest. The
  current claim-language audit reports C1-C14 ready and forbidden-overclaim
  checks `13/13` passed.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_claim_language.py src/slop_sftdiv/cli/check_paper_package.py tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-claim-language`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_audit_paper_claim_language.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_source_provenance.py`.

## 2026-06-18 - DPO human-preference overclaim guard added

- Tightened `slop-audit-paper-claim-language` with two additional forbidden
  overclaim checks: "Dolci DPO chosen responses reveal human preference." and
  "Chosen responses are direct evidence of human taste."
- Added a regression test for the Dolci DPO human-preference overclaim and
  updated `slop-check-paper-package` to expect 15 forbidden-overclaim rows.
- Regenerated the claim-language audit and reproducibility manifest. The
  current claim-language audit reports C1-C14 ready and forbidden-overclaim
  checks `15/15` passed.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_claim_language.py src/slop_sftdiv/cli/check_paper_package.py tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-claim-language`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_audit_paper_claim_language.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_source_provenance.py`.

## 2026-06-18 - Universal-scope overclaim guard added

- Tightened `slop-audit-paper-claim-language` with three additional forbidden
  overclaim checks: "All LLMs exhibit the same slop style.", "All alignment
  methods produce the same slop style.", and "Slop style is a single global
  score."
- Added a regression test for the all-LLMs overclaim and updated
  `slop-check-paper-package` to expect 18 forbidden-overclaim rows.
- Regenerated the claim-language audit and reproducibility manifest. The
  current claim-language audit reports C1-C14 ready and forbidden-overclaim
  checks `18/18` passed.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_claim_language.py src/slop_sftdiv/cli/check_paper_package.py tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-claim-language`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_audit_paper_claim_language.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_source_provenance.py`.

## 2026-06-18 - Tier-1 feature-boundary overclaim guard added

- Tightened `slop-audit-paper-claim-language` with four additional forbidden
  overclaim checks for Tier-1 feature boundaries: demoted
  `rule_of_three_approx`, derived `stock_openers_closers`, global Tier-1
  precision, and broad semantic `slop_lexicon` detector claims.
- Added regression tests for the Tier-1 precision and derived-feature
  overclaim cases and updated `slop-check-paper-package` to expect 22
  forbidden-overclaim rows.
- Regenerated the claim-language audit and reproducibility manifest. The
  current claim-language audit reports C1-C14 ready and forbidden-overclaim
  checks `22/22` passed.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_claim_language.py src/slop_sftdiv/cli/check_paper_package.py tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-claim-language`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Replication and sparse-Tier-3 overclaim guard added

- Tightened `slop-audit-paper-claim-language` with four additional forbidden
  overclaim checks for SmolLM3 and Tier-3 boundaries: no universal SmolLM3
  replication claim, no identical OLMo/SmolLM3 stage-localization claim, no
  standalone-headline follow-up-offer claim, and no denominator-stable sparse
  Tier-3 AF claim.
- Added regression tests for the SmolLM3 replication and sparse follow-up-offer
  overclaim cases and updated `slop-check-paper-package` to expect 26
  forbidden-overclaim rows.
- Regenerated the claim-language audit and reproducibility manifest. The
  current claim-language audit reports C1-C14 ready and forbidden-overclaim
  checks `26/26` passed.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_claim_language.py src/slop_sftdiv/cli/check_paper_package.py tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-claim-language`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Reviewer FAQ expanded for claim-boundary readability

- Expanded `docs/experiments/paper_reviewer_faq.md` with reader-facing answers
  for corpus-side pybiber versus generated-output evidence, publication-safe
  Tier-1 feature use, SmolLM3 as replication pressure rather than identical
  stage localization, and sparse Tier-3 AF interpretation.
- Tightened `slop-check-paper-package` so the FAQ must retain those boundary
  phrases instead of regressing to a generic scope FAQ.
- Regenerated the reproducibility manifest after the FAQ content change.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Draft bundle refreshed after FAQ and audit updates

- Refreshed the draft submission bundle after the FAQ and claim-language audit
  changes so copied paper-control files and bundle metadata stay aligned with
  source docs.
- Regenerated the reproducibility manifest after the bundle refresh.
- Validation passed:
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`
  reported 28 ready files and 0 missing;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`
  reported 99 rows;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Dolma retained-sample FAQ boundary added

- Expanded `docs/experiments/paper_reviewer_faq.md` with a direct answer that
  the OLMo pretraining reference is an English-filtered retained Dolma sample
  from a bounded 80k scan, not a full Dolma/source-inventory measurement.
- Tightened `slop-check-paper-package` so the reviewer FAQ must retain this
  retained-sample/source-coverage boundary.
- Refreshed the draft submission bundle and reproducibility manifest after the
  FAQ update.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`
  reported 28 ready files and 0 missing;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`
  reported 99 rows;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Full local test suite passed after paper hardening

- Ran the full local test suite after the claim-language, FAQ, bundle, and
  package-checker updates.
- Validation passed:
  `uv run pytest` collected 334 tests and passed all 334. The only warnings
  were pybiber/spaCy/polars deprecation warnings from the pybiber smoke tests.

## 2026-06-18 - Paper audit refresh and pybiber numeric-claim fix

- Refreshed the paper manuscript/control audits after the FAQ and claim
  hardening pass: abstract/conclusion, manuscript structure, limitations,
  terminology, review hygiene, source provenance, claim evidence, numeric
  claims, citations, release inventory, submission exit, figure readiness,
  table readiness, table typography, and claim language.
- The refreshed numeric-claims audit exposed a wording mismatch for the
  `pybiber_dpo_contrast` manuscript sentence. Updated the manuscript to carry
  the exact source-backed DPO pybiber contrast for broad nouns, attributive
  adjectives, adverbs, and prepositions.
- Refreshed the draft submission bundle and reproducibility manifest after the
  audit outputs changed.
- Validation passed:
  `uv run slop-audit-paper-numeric-claims`;
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`
  reported 28 ready files and 0 missing;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`
  reported 99 rows;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Readiness-audit next action updated

- Updated `docs/experiments/paper_readiness_audit.md` so the recommended next
  work no longer treats claim-safe manuscript wording as open. The next action
  now emphasizes venue copyediting while preserving claim-language,
  numeric-claims, and structure audits.
- Refreshed the draft submission bundle and reproducibility manifest after the
  readiness-audit wording change.
- Validation passed:
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`
  reported 28 ready files and 0 missing;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`
  reported 99 rows;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Reproduction runbook bundle/manifest order clarified

- Updated `docs/experiments/paper_reproduction_runbook.md` to run
  `slop-audit-paper-submission-exit` before bundle materialization, and to run
  reproducibility-manifest generation only after the draft bundle is stable.
- Added an explicit note not to parallelize bundle materialization and manifest
  generation because the reproducibility manifest hashes the bundle manifest.
- Validation passed with the documented sequential order:
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`
  reported 28 ready files and 0 missing;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`
  reported 99 rows;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Package index aligned with retained-sample and bundle order

- Updated `docs/experiments/paper_package_index.md` so the claim boundaries
  include the retained English-filtered Dolma sample boundary and the
  validation command block includes draft-bundle materialization before
  reproducibility-manifest generation.
- Tightened `slop-check-paper-package` so the package index must retain those
  two entry-point guardrails.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`
  reported 28 ready files and 0 missing;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`
  reported 99 rows;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Readiness executive blocker wording aligned

- Updated `docs/experiments/paper_readiness_audit.md` executive status to match
  the current submission-exit audit: paper-draft ready, submission-blocked by
  venue finalization and Phase 4 human-validation.
- Refreshed submission-exit, draft bundle, and reproducibility manifest after
  the readiness-audit wording change.
- Validation passed:
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`
  reported 28 ready files and 0 missing;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`
  reported 99 rows;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Final validation after package-control edits

- Re-ran the full local test suite after the package-index, reproduction
  runbook, readiness-audit, claim-language, FAQ, and bundle/manifest updates.
- Validation passed:
  `uv run pytest` collected 334 tests and passed all 334. The only warnings
  were pybiber/spaCy/polars deprecation warnings from the pybiber smoke tests.
  `uv run slop-check-paper-package --root /home/user/slop` passed at
  2026-06-18T11:09:17Z.

## 2026-06-18 - Phase 4 annotation codebook fill rules tightened

- Expanded `docs/experiments/phase4_human_annotation_codebook.md` with
  annotator fill rules: edit only `human_perceived_slop`,
  `shaib_taxonomy_label`, and `notes`; preserve blinded IDs, matched text, and
  snippets for import/agreement/adjudication.
- Cleaned up the pilot procedure so agreement, yes/no disagreement review,
  adjudication fields, and full-package escalation are ordered unambiguously.
- Tightened `slop-check-paper-package` so the codebook must retain the new
  fill-rule and agreement-sequencing phrases.
- Refreshed Phase 4 annotation readiness outputs, rematerialized the redacted
  human annotation handoff, refreshed submission-exit, the draft paper bundle,
  and the reproducibility manifest.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-audit-phase4-human-package`;
  `uv run slop-materialize-phase4-human-handoff --root /home/user/slop`
  reported 12 ready files, 0 missing, and 6 redacted annotator CSVs;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_materialize_phase4_human_handoff.py tests/test_audit_phase4_human_package.py`.

## 2026-06-18 - Phase 4 handoff README fill-rule reminder added

- Updated `src/slop_sftdiv/cli/materialize_phase4_human_handoff.py` so the
  generated handoff README repeats the annotator fill rules: edit only
  `human_perceived_slop`, `shaib_taxonomy_label`, and `notes`; preserve
  `blind_id`, `matched_text`, and `snippet`.
- Tightened the package checker and handoff tests so the README-level reminder
  remains present in regenerated handoff bundles.
- Refreshed the Phase 4 human handoff, submission-exit audit, draft paper
  bundle, and reproducibility manifest.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/materialize_phase4_human_handoff.py src/slop_sftdiv/cli/check_paper_package.py tests/test_materialize_phase4_human_handoff.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_materialize_phase4_human_handoff.py tests/test_check_paper_package.py`;
  `uv run slop-materialize-phase4-human-handoff --root /home/user/slop`
  reported 12 ready files, 0 missing, and 6 redacted annotator CSVs;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`
  reported 28 ready files and 0 missing;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`
  reported 99 rows;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Venue sizing inventory added

- Added `docs/experiments/paper_venue_sizing_inventory.md` to make the
  remaining venue-formatting blocker concrete: current Figure 1-Figure 5 SVG
  sizes in points/inches, PDF/PNG export availability, table float/width
  assumptions, and venue-adaptation decisions.
- Linked the sizing inventory from the package index, venue adaptation runbook,
  and readiness audit.
- Added the sizing inventory to required package paths, the draft submission
  bundle, and the reproducibility manifest. The current draft bundle now has
  29 ready files and 0 missing; the reproducibility manifest now has 100 rows.
- Added package-check coverage so stale sizing-inventory content is flagged.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py src/slop_sftdiv/cli/materialize_paper_submission_bundle.py src/slop_sftdiv/cli/audit_paper_reproducibility_manifest.py tests/test_check_paper_package.py tests/test_materialize_paper_submission_bundle.py`;
  `uv run pytest tests/test_check_paper_package.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_reproducibility_manifest.py`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`
  reported 29 ready files and 0 missing;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`
  reported 100 rows;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Venue sizing inventory cross-check hardened

- Tightened `slop-check-paper-package` so
  `docs/experiments/paper_venue_sizing_inventory.md` is cross-checked against
  `paper_figure_readiness_audit.csv` and `paper_table_typography_review.csv`.
  The checker now verifies figure-specific path/size rows, inch conversions,
  and table environment/width rows.
- Added regression coverage for stale figure dimensions in the sizing
  inventory.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_reproducibility_manifest.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Venue sizing inventory added to release inventory

- Added `docs/experiments/paper_venue_sizing_inventory.md` to
  `slop-audit-paper-release-inventory` as a runbook/control item, so the
  release-facing inventory covers the venue-formatting adaptation artifact.
- Regenerated the release inventory, draft bundle, and reproducibility
  manifest. The release inventory now reports 19 ready rows and 0 missing.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_release_inventory.py tests/test_audit_paper_release_inventory.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_audit_paper_release_inventory.py tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-release-inventory --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`
  reported 29 ready files and 0 missing;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`
  reported 100 rows;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Venue checklist linked to sizing inventory

- Updated `docs/experiments/paper_venue_decision_checklist.md` and
  `artifacts/paper/submission/paper_venue_decision_checklist.csv` so the
  figure-dimensions and table-placement rows explicitly cite
  `paper_venue_sizing_inventory.md` as current evidence.
- Tightened `slop-check-paper-package` so the venue checklist must retain the
  sizing-inventory links for both figures and tables.
- Refreshed submission-exit, draft bundle, and reproducibility manifest after
  the checklist update.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`
  reported 29 ready files and 0 missing;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`
  reported 100 rows;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Submission-exit figure/table blocker cites sizing inventory

- Updated `slop-audit-paper-submission-exit` so the
  `figure_table_finalization` evidence row names the venue sizing inventory in
  addition to figure/table audits, rendered-layout review, export candidates,
  and venue-decision checklist.
- Regenerated submission-exit, draft bundle, and reproducibility manifest.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_submission_exit.py tests/test_audit_paper_submission_exit.py`;
  `uv run pytest tests/test_audit_paper_submission_exit.py`;
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`
  reported 29 ready files and 0 missing;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`
  reported 100 rows;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Phase 4 production runbook denominator guard added

- Corrected `docs/experiments/phase4_production_runbook.md` so the opening
  status sentence names the current 512-document exact teacher-forced
  continuation rather than the historical 128-document scoping run.
- Added `slop-check-paper-package` coverage for the Phase 4 production runbook
  so stale 128-document "already exists" wording is flagged while the
  historical 128-document artifact remains allowed as a scoping note.
- Refreshed submission-exit, the draft paper bundle, and the reproducibility
  manifest after the runbook edit. The draft bundle still reports 29 ready
  files and 0 missing; the reproducibility manifest still reports 100 rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_reproducibility_manifest.py tests/test_audit_paper_release_inventory.py tests/test_audit_paper_submission_exit.py`;
  `uv run pytest` collected 337 tests and passed all 337, with only the known
  pybiber/spaCy/polars deprecation warnings.

## 2026-06-18 - Package index links Phase 4 production runbook

- Updated `docs/experiments/paper_package_index.md` so the Phase 4 detector
  evidence row links `phase4_production_runbook.md` and tells new readers that
  it records the completed 512-document exact Tier-3 rerun.
- Tightened `slop-check-paper-package` so the package index must keep both the
  production-runbook link and the 512-document rerun description.
- Refreshed submission-exit, the draft paper bundle, and the reproducibility
  manifest after the index edit. The draft bundle still reports 29 ready files
  and 0 missing; the reproducibility manifest still reports 100 rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_reproducibility_manifest.py tests/test_audit_paper_release_inventory.py tests/test_audit_paper_submission_exit.py`.

## 2026-06-18 - Retired Biber-lite spelling guard broadened

- Broadened `slop-check-paper-package` retired-feature checks over
  paper-facing docs so both hyphenated/underscored and plain-space
  Biber-lite spellings are rejected.
- Added regression coverage for the plain-space `Biber lite` spelling.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Phase 4 handoff coordinator workflow made explicit

- Updated the generated Phase 4 human-annotation handoff README so the
  coordinator path names the agreement, adjudication, and final-summary
  commands directly instead of only referring to the protocol.
- Tightened `slop-check-paper-package` so the handoff README must retain the
  coordinator command sequence and adjudication-column reminder.
- Regenerated the Phase 4 human handoff, submission-exit audit, draft paper
  bundle, and reproducibility manifest. The handoff still reports 12 ready
  files, 0 missing, and 6 redacted annotator CSVs; the draft bundle still
  reports 29 ready files and 0 missing; the reproducibility manifest still
  reports 100 rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/materialize_phase4_human_handoff.py src/slop_sftdiv/cli/check_paper_package.py tests/test_materialize_phase4_human_handoff.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_materialize_phase4_human_handoff.py tests/test_check_paper_package.py`;
  `uv run slop-materialize-phase4-human-handoff --root /home/user/slop`;
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_materialize_phase4_human_handoff.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_reproducibility_manifest.py tests/test_audit_paper_release_inventory.py tests/test_audit_paper_submission_exit.py`.

## 2026-06-18 - Venue adaptation final gate uses safe bundle order

- Updated `docs/experiments/paper_venue_adaptation_runbook.md` so the final
  post-venue command sequence rematerializes the draft submission bundle before
  regenerating the reproducibility manifest.
- Added package-check coverage so the venue adaptation runbook must keep the
  bundle materialization command and the refreshed `draft_bundle/MANIFEST.csv`
  requirement.
- Refreshed submission-exit, the draft paper bundle, and the reproducibility
  manifest after the runbook edit. The draft bundle still reports 29 ready
  files and 0 missing; the reproducibility manifest still reports 100 rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_reproducibility_manifest.py tests/test_audit_paper_release_inventory.py tests/test_audit_paper_submission_exit.py`.

## 2026-06-18 - Additional full-grid overclaim guard added

- Added a claim-language forbidden-overclaim check for the affirmative sentence
  "We completed the 5,000 prompt x 8 completion x 3 temperature OLMo grid."
  without banning the manuscript's safe negative wording that the reduced
  panels are not the same evidence as such a completed grid.
- Regenerated the claim-language audit. It now reports 41 rows total:
  C1-C14 plus 27 forbidden-overclaim checks.
- Refreshed submission-exit, the draft paper bundle, and the reproducibility
  manifest after the claim-language audit update. The draft bundle still
  reports 29 ready files and 0 missing; the reproducibility manifest still
  reports 100 rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_claim_language.py src/slop_sftdiv/cli/check_paper_package.py tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-claim-language`;
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_audit_paper_claim_language.py tests/test_check_paper_package.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_reproducibility_manifest.py tests/test_audit_paper_submission_exit.py`.

## 2026-06-18 - Phase 4 full-package label summary path fixed

- Corrected `docs/experiments/phase4_human_perceptibility_protocol.md` so the
  full-package annotation flow imports the blinded full sheet to
  `phase4_human_perceptibility_blind_full_labeled.jsonl` and summarizes the
  post-agreement/adjudicated full-package JSONL rather than the original blank
  annotation package.
- Added `slop-check-paper-package` coverage so the protocol must retain the
  full labeled/adjudicated paths and must not use the blank
  `phase4_human_perceptibility_annotation_package.jsonl` as the full-package
  post-label summary input.
- Regenerated the Phase 4 human handoff, submission-exit audit, draft paper
  bundle, and reproducibility manifest. The handoff still reports 12 ready
  files, 0 missing, and 6 redacted annotator CSVs; the draft bundle still
  reports 29 ready files and 0 missing; the reproducibility manifest still
  reports 100 rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-materialize-phase4-human-handoff --root /home/user/slop`;
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run pytest tests/test_check_paper_package.py tests/test_materialize_phase4_human_handoff.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_reproducibility_manifest.py tests/test_audit_paper_submission_exit.py`.

## 2026-06-18 - EXPERIMENTS pybiber scope boundary guarded

- Tightened `EXPERIMENTS.md` so Phase 1 explicitly separates Tier-1 matcher
  use on generated checkpoint outputs from full pybiber's active paper role:
  corpus-side register extraction over retained pretrain/SFT/DPO samples only.
- Added `slop-check-paper-package` coverage for that boundary. The checker now
  requires the corpus-side full-pybiber scope sentence in `EXPERIMENTS.md` and
  flags the old wording that bundled full pybiber with generated checkpoint
  measurements.
- Refreshed submission-exit, the draft paper bundle, and the reproducibility
  manifest after the `EXPERIMENTS.md` edit. The draft bundle still reports 29
  ready files and 0 missing; the reproducibility manifest still reports 100
  rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Phase 4 annotator quickstart added to handoff bundle

- Added an annotator-facing quickstart to the Phase 4 human annotation handoff
  bundle. The quickstart tells annotators which assigned CSV to fill, which
  columns may be edited, how to interpret the `human_perceived_slop` labels,
  and not to use or request coordinator maps.
- Regenerated the handoff bundle with `uv run slop-materialize-phase4-human-handoff
  --root /home/user/slop`. The bundle now reports 13 ready files, 0 missing
  files, and 6 redacted annotator CSVs.
- Updated the readiness audit and package checker so the high-level paper
  package explicitly preserves the annotator quickstart in addition to the
  codebook, blinded sheets, redacted handoff, agreement tooling, and
  adjudication tooling.
- Regenerated the submission draft bundle and reproducibility manifest. The
  draft bundle still reports 30 ready files and 0 missing; the reproducibility
  manifest now reports 114 rows.

## 2026-06-18 - Phase 4 blind-label import rejects protected-field edits

- Hardened the Phase 4 blinded-label workflow so coordinator maps now retain
  the protected `snippet` field alongside `matched_text`, while annotator
  sheets remain stripped of source identifiers.
- Updated blind-label import to reject edits to protected `matched_text` or
  `snippet` fields before writing canonical labeled JSONL rows. This makes the
  annotator quickstart rule machine-enforced rather than only documentary.
- Regenerated the Phase 4 annotation-readiness outputs, blind pilot/full
  sheets, blind pilot/full maps, and the redacted handoff bundle. The handoff
  still reports 13 ready files, 0 missing files, and 6 redacted annotator CSVs.
- Refreshed the venue-neutral submission draft bundle and reproducibility
  manifest after the generated artifact and source changes.

## 2026-06-18 - Phase 4 import now requires protected map fields

- Tightened `slop-import-phase4-blind-labels` so blind maps must contain the
  protected `matched_text` and `snippet` fields before import proceeds.
  Older or incomplete maps can no longer bypass protected-field checks.
- Updated the agreement and adjudication test fixtures to use the current
  blind-map schema, and added a regression test that rejects maps missing the
  protected `snippet` field.
- Validation passed for the import/agreement/adjudication tests before
  refreshing the paper draft bundle and reproducibility manifest.

## 2026-06-18 - Phase 4 adjudication rejects stale or typo rows

- Hardened `slop-apply-phase4-label-adjudication` so adjudication rows must
  reference real `blind_id` values that actually disagreed between annotators.
  Unknown IDs and stale adjudication rows for consensus examples now fail
  before canonical JSONL is written.
- Added regression tests for typo adjudication IDs and stale consensus-row
  adjudications, in addition to the existing missing-adjudication strict-mode
  check.
- Updated the human-perceptibility protocol and reproduction runbook to record
  the adjudication integrity contract, and added a paper-package checker guard
  so the protocol keeps that warning.

## 2026-06-18 - Phase 4 handoff manifest integrity checked

- Tightened `slop-check-paper-package` so the Phase 4 human handoff manifest
  now verifies each ready bundle file exists and that manifest `size_bytes` and
  `sha256` match the actual file.
- For copied handoff rows, the checker also verifies the declared source path
  exists and that the copied bundle file still matches the source file. The
  generated annotator quickstart is skipped only for source-copy comparison.
- Updated the package-checker fixture to generate real handoff manifest hashes
  from fixture bundle files, and added regression tests for handoff hash drift
  and missing bundle files.

## 2026-06-18 - Phase 4 handoff redaction validated from file headers

- Extended `slop-check-paper-package` to inspect the actual headers of every
  annotator-facing CSV in the Phase 4 handoff bundle, rather than trusting only
  the manifest `redaction_status` field.
- Added a regression test where the handoff manifest hash and redaction status
  match a leaked annotator CSV, but the checker still rejects the file because
  its header contains an unblinded `feature` column.
- This keeps the handoff package safe even if the manifest is regenerated
  after an accidental annotator-file edit.

## 2026-06-18 - Submission-exit audit requires venue controls

- Tightened `slop-audit-paper-submission-exit` so the figure/table
  finalization row only reports `venue_review_pending` when the venue-decision
  checklist CSV, venue-decision checklist document, adaptation runbook, and
  sizing inventory exist.
- The venue checklist must include the expected decision items and keep them
  `decision_pending`; otherwise the figure/table row is `blocked`, not merely
  pending final venue review.
- The audit now reads the current venue-checklist schema's `item` column while
  remaining compatible with the older `decision_item` fixture name.
- Added a regression test for missing venue controls, then regenerated the
  submission-exit audit, venue-neutral draft bundle, and reproducibility
  manifest.

## 2026-06-18 - Submission-exit package row checks audit readiness

- Tightened the `bibliography_and_package` row in
  `slop-audit-paper-submission-exit`. The row now requires citation,
  claim-evidence, and source-provenance audit CSVs to contain ready statuses,
  rather than checking only for file existence.
- The same row now validates the reproducibility manifest's row statuses,
  positive sizes, and SHA-256 shape before reporting `ready`.
- Added regression coverage for a non-ready citation audit blocking the
  `bibliography_and_package` row, then regenerated the submission-exit audit,
  draft bundle, and reproducibility manifest.

## 2026-06-18 - Submission-exit manifest row checks file hashes

- Strengthened the `bibliography_and_package` row again so the
  reproducibility manifest must match the actual files named in each row. The
  audit now checks row status, positive size, SHA-256 shape, file existence,
  actual size, and actual SHA-256.
- Added a regression test that mutates a manifest-tracked file after manifest
  creation and verifies that `bibliography_and_package` becomes `blocked`.
- Refreshed the reproducibility manifest before regenerating the
  submission-exit audit, then refreshed the draft bundle and manifest again
  after the audit output changed.

## 2026-06-18 - Phase 3 generated-output pybiber runbook retired

- Replaced the stale Phase 3 runbook section that instructed extracting full
  pybiber over generated caches with an explicit boundary note: generated-output
  full pybiber is not part of the active Phase 3 route, and older
  register-proxy/style-signature artifacts remain archived diagnostics only.
- Added `slop-check-paper-package` coverage so
  `docs/experiments/phase3_production_runbook.md` must retain the boundary and
  must not reintroduce the old generated-output pybiber instruction.
- Refreshed the release inventory, submission-exit audit, draft paper bundle,
  and reproducibility manifest after the runbook edit. The release inventory
  still reports 19 ready rows and 0 missing; the draft bundle still reports 29
  ready files and 0 missing; the reproducibility manifest still reports 100
  rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-release-inventory --root /home/user/slop`;
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Phase 3 status Biber-lite language superseded

- Updated `docs/experiments/phase3_status.md` with a 2026-06-18 supersession
  note: active Tier-2 register evidence is corpus-side full pybiber from Phase
  1, while generated-output Biber-lite/register-proxy artifacts in the status
  memo are historical diagnostics only.
- Reworded the Phase 3 status scope, completion-gate summary, bootstrap-CI
  audit row, and SmolLM3 diagnostic section so they refer to archived
  generated-output register-proxy/style-signature diagnostics rather than an
  active generated-output Biber layer.
- Validation passed:
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Phase 2 handoff register-proxy language superseded

- Updated `docs/experiments/phase2_handoff_conclusions.md` with the same
  2026-06-18 boundary: active Tier-2 register evidence is corpus-side full
  pybiber from Phase 1, while the older Biber-lite/register-proxy section is
  historical generated-output diagnostic context only.
- Reworded the archived register-proxy section, style-signature bullets, and
  open-caveat language so the handoff report no longer describes Biber-lite as
  an active Phase 2 register layer or full-pybiber evidence.
- Validation passed:
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Phase 4 human-summary package guard added

- Tightened `slop-check-paper-package` so
  `phase4_human_perceptibility_summary.csv` must match the current unlabeled
  Phase 4 state: 10 candidate rows, 100 total examples per candidate, 0 labeled
  rows, blank rates/taxonomy labels, and `venn_region=unlabeled`.
- Added Markdown checks for the generated Phase 4 human-perceptibility summary
  so it must keep the detector-only warning and the 0/100 follow-up-offer row.
- Added regression coverage that catches accidental promotion of a candidate to
  `detector_and_human_perceived` while the paper package still claims
  candidate-only Phase 4 status.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Reproduction runbook Phase 4 summary commands guarded

- Tightened `slop-check-paper-package` so
  `docs/experiments/paper_reproduction_runbook.md` must include the Phase 4
  handoff materialization command, the human-label summarizer command, and the
  reminder that the current human-perceptibility summary remains unlabeled
  unless real labels have been added.
- Updated the package-check fixture so the reproduction-runbook guard reflects
  the current paper refresh sequence.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Open-release inventory includes package and label utilities

- Added the paper package checker and Phase 4 blind-label workflow utilities
  to the open-release inventory:
  `src/slop_sftdiv/cli/check_paper_package.py`,
  `src/slop_sftdiv/cli/import_phase4_blind_labels.py`,
  `src/slop_sftdiv/cli/summarize_phase4_label_agreement.py`, and
  `src/slop_sftdiv/cli/apply_phase4_label_adjudication.py`.
- Tightened `slop-check-paper-package` so the release inventory must carry at
  least 23 rows and include the new control/handoff utility paths.
- Regenerated the release inventory, submission-exit audit, draft bundle, and
  reproducibility manifest. The release inventory now reports 23 ready rows and
  0 missing; the draft bundle still reports 29 ready files and 0 missing; the
  reproducibility manifest still reports 100 rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_release_inventory.py src/slop_sftdiv/cli/check_paper_package.py tests/test_audit_paper_release_inventory.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_audit_paper_release_inventory.py tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-release-inventory --root /home/user/slop`;
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Reproducibility manifest now checksums release code

- Added a `release_code` category to
  `artifacts/paper/submission/paper_reproducibility_manifest.csv` and
  `docs/experiments/paper_reproducibility_manifest.md` for the source files
  already advertised in the open-release inventory: Tier-1 matchers,
  EQ-Bench scoring, full pybiber extraction, propensity definitions,
  generation text normalization, teacher-forced/free-running harnesses, the
  Phase 4 blind-label utilities, and the package checker.
- Tightened `slop-check-paper-package` so the reproducibility manifest must
  include that `release_code` category and the concrete release-code paths,
  and so each release-code source row must actually be categorized as
  `release_code`. This prevents the paper package from looking reproducible
  while only checksumming generated artifacts and prose.
- Regenerated the reproducibility manifest. It now reports 111 rows.
- Updated `docs/experiments/paper_package_index.md` and
  `docs/experiments/paper_reproduction_runbook.md` so a new reader can see
  that the reproducibility manifest covers both release-code source files and
  generated paper artifacts.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_reproducibility_manifest.py src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py tests/test_audit_paper_reproducibility_manifest.py`;
  `uv run pytest tests/test_audit_paper_reproducibility_manifest.py tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Draft bundle includes package index and source-copy guard

- Added `docs/experiments/paper_package_index.md` to the venue-neutral draft
  bundle under `review_controls/paper_package_index.md`, so the staged bundle
  now carries its own start-here orientation file.
- Tightened `slop-check-paper-package` for the draft bundle. The checker now
  expects 30 bundle rows, requires the package-index bundle path, verifies each
  ready bundle file against both its manifest hash and the current
  `source_path` hash, and requires the bundle README to mention the package
  index plus claim/readiness controls.
- Updated the reproduction runbook and paper scaffold from 29 to 30 ready
  staged files.
- Regenerated the draft bundle and reproducibility manifest. The draft bundle
  now reports 30 ready files and 0 missing; the reproducibility manifest still
  reports 111 rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/materialize_paper_submission_bundle.py src/slop_sftdiv/cli/check_paper_package.py tests/test_materialize_paper_submission_bundle.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_materialize_paper_submission_bundle.py tests/test_check_paper_package.py`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Release surface includes paper-package materializers

- Added the venue-neutral draft-bundle materializer and reproducibility
  manifest writer to the open-release inventory:
  `src/slop_sftdiv/cli/materialize_paper_submission_bundle.py` and
  `src/slop_sftdiv/cli/audit_paper_reproducibility_manifest.py`.
- Added both utilities to the `release_code` checksum surface in the
  reproducibility manifest and tightened `slop-check-paper-package` so the
  release inventory and manifest must retain those paths.
- Regenerated the release inventory, draft bundle, and reproducibility
  manifest. The release inventory now reports 25 ready rows and 0 missing; the
  draft bundle reports 30 ready files and 0 missing; the reproducibility
  manifest reports 113 rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/materialize_paper_submission_bundle.py src/slop_sftdiv/cli/audit_paper_release_inventory.py src/slop_sftdiv/cli/audit_paper_reproducibility_manifest.py src/slop_sftdiv/cli/check_paper_package.py tests/test_materialize_paper_submission_bundle.py tests/test_audit_paper_release_inventory.py tests/test_audit_paper_reproducibility_manifest.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_audit_paper_release_inventory.py tests/test_audit_paper_reproducibility_manifest.py tests/test_materialize_paper_submission_bundle.py tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-release-inventory --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Readiness audit reflects expanded release package

- Updated `docs/experiments/paper_readiness_audit.md` so the draft-bundle row
  names the package index as part of the staged bundle, the release-inventory
  row names paper-package materializers, and the reproducibility-manifest row
  names release-code source files as part of the checksum surface.
- Added `slop-check-paper-package` coverage for those readiness-audit phrases
  so the high-level readiness page cannot drift behind the release inventory,
  draft bundle, and manifest.
- Regenerated the reproducibility manifest after the readiness-audit and
  checker updates. The manifest still reports 113 rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Venue adaptation runbook adds per-artifact matrices

- Added per-figure and per-table decision matrices to
  `docs/experiments/paper_venue_adaptation_runbook.md`. The figure matrix now
  records each figure's current paper role, likely placement, required venue
  decision, and post-decision verification. The table matrix does the same for
  Table 1, Table 2, Table 4, and appendix tables A1-A3.
- Updated `docs/experiments/paper_venue_sizing_inventory.md` to point readers
  from the current venue-neutral dimensions to those per-artifact decision
  matrices.
- Tightened `slop-check-paper-package` so the venue runbook must retain the
  per-artifact matrices, including the Phase 4 Tier-3 Figure 5 row and the
  pybiber-interval Appendix Table A2 row, and so the sizing inventory must
  keep the decision-matrix cross-reference.
- Regenerated the draft bundle and reproducibility manifest after the runbook
  edit. The draft bundle still reports 30 ready files and 0 missing; the
  reproducibility manifest still reports 113 rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Release inventory detects file-size drift

- Tightened `slop-check-paper-package` so every `ready` row in
  `artifacts/paper/submission/paper_release_inventory.csv` must point to an
  existing file whose current byte size matches the recorded `size_bytes`.
  This makes the release inventory a live drift check, not only a declarative
  list of paths.
- Added regression coverage that mutates a release-inventory source file after
  fixture materialization and verifies the package checker reports the stale
  `size_bytes` value.
- Updated the package-check fixture so release-inventory rows are generated
  after all listed files are materialized and include the Phase 4 Tier-3 matcher
  spec plus cached statistics rows.
- Regenerated the release inventory, submission-exit audit, draft bundle, and
  reproducibility manifest in dependency order. The release inventory reports
  25 ready rows and 0 missing; the draft bundle reports 30 ready files and 0
  missing; the reproducibility manifest reports 114 rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py tests/test_audit_paper_release_inventory.py`;
  `uv run pytest tests/test_check_paper_package.py tests/test_audit_paper_release_inventory.py`;
  `uv run slop-audit-paper-release-inventory --root /home/user/slop`;
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Paper measurement synthesis added

- Added `docs/experiments/paper_measurement_synthesis.md` as a paper-facing
  review-control note that connects the core measurement layers: Phase 1 corpus
  rates, full pybiber register, EQ-Bench Slop Score, teacher-forced
  propensity, free-running emission/compounding, and detector-derived Tier 3.
- The synthesis records the intended interpretation hierarchy: pybiber supplies
  the broad corpus-side register backbone, EQ-Bench is an aggregate benchmark
  bridge, feature-specific AF remains the model-side localization layer,
  sampled emissions and compounding answer generation questions, and detector
  clusters remain candidate-only until human labels exist.
- Wired the synthesis into the new-reader path in
  `docs/experiments/paper_package_index.md`, staged it in the venue-neutral
  submission draft bundle, added it to the reproducibility manifest, and
  tightened `slop-check-paper-package` so the synthesis cannot be dropped or
  weakened silently.
- The draft bundle now reports 31 ready files and 0 missing. The
  reproducibility manifest now reports 115 rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py src/slop_sftdiv/cli/materialize_paper_submission_bundle.py src/slop_sftdiv/cli/audit_paper_reproducibility_manifest.py tests/test_check_paper_package.py tests/test_materialize_paper_submission_bundle.py`;
  `uv run pytest tests/test_check_paper_package.py tests/test_materialize_paper_submission_bundle.py`;
  `uv run slop-audit-paper-release-inventory --root /home/user/slop`;
  `uv run slop-materialize-paper-submission-bundle --root /home/user/slop`;
  `uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop`;
  `uv run slop-audit-paper-submission-exit --root /home/user/slop`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Readiness audit surfaces measurement synthesis

- Updated `docs/experiments/paper_readiness_audit.md` so the Reader
  orientation row names `paper_measurement_synthesis.md` alongside the package
  index, glossary, scaffold, and reviewer FAQ.
- Updated the venue-neutral draft-bundle readiness row so it explicitly says
  the bundle includes the measurement synthesis, not only generic
  claim/readiness controls.
- Tightened `slop-check-paper-package` to require the readiness audit to keep
  the pybiber/EQ-Bench/propensity/generation/compounding/detector hierarchy
  language.
- Regenerated release inventory, the 31-file draft bundle, submission-exit
  audit, and 115-row reproducibility manifest.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Manuscript methods align to measurement hierarchy

- Updated `docs/experiments/paper_manuscript_draft.md` and
  `docs/experiments/paper_methods_draft.md` so the methods framing now
  distinguishes four localization layers from two auxiliary measurement
  surfaces.
- The four localization layers remain corpus inheritance, fixed-context
  propensity, free-running emission, and self-conditioning. EQ-Bench is now
  explicitly identified as an aggregate public benchmark bridge, while
  detector-derived Tier 3 is identified as a candidate-feature discovery
  surface pending human labels and matcher precision checks.
- Regenerated the 31-file venue-neutral draft bundle, submission-exit audit,
  and 115-row reproducibility manifest after the manuscript/methods edits.
- Validation passed:
  `uv run slop-check-paper-package --root /home/user/slop`;
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`.

## 2026-06-18 - Terminology audit guards measurement hierarchy

- Added a `localization_surface_boundary` row to
  `slop-audit-paper-terminology`. The row requires the manuscript to preserve
  the four-localization-layer framing and the separate EQ-Bench/Tier-3
  measurement-surface boundaries.
- Tightened `slop-check-paper-package` so
  `artifacts/paper/manuscript/paper_terminology_audit.csv` must include the
  new row and `docs/experiments/paper_terminology_audit.md` must report it as
  ready.
- Regenerated the terminology audit, release inventory, 31-file draft bundle,
  submission-exit audit, and 115-row reproducibility manifest. The terminology
  audit now reports 9 ready rows.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_terminology.py src/slop_sftdiv/cli/check_paper_package.py tests/test_audit_paper_terminology.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_audit_paper_terminology.py tests/test_check_paper_package.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Abstract/conclusion audit guards compounding and synthesis path

- Updated the integrated manuscript abstract so the measurement list includes
  compounding tests alongside pybiber, teacher-forced propensity,
  free-running generation, EQ-Bench, and detector discovery.
- Tightened `slop-audit-paper-abstract-conclusion` so the abstract must keep
  the compounding phrase and the package-index alignment row must keep the
  measurement-synthesis link.
- Tightened `slop-check-paper-package` so the abstract/conclusion audit must
  report `abstract_alignment` as 8/8 and `package_index_alignment` as 8/8.
- Regenerated the abstract/conclusion audit, manuscript-structure audit,
  release inventory, 31-file draft bundle, submission-exit audit, and 115-row
  reproducibility manifest. The manuscript remains within structure-audit word
  bounds: 182 abstract words, 213 conclusion words, and 5,584 total words.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/audit_paper_abstract_conclusion.py src/slop_sftdiv/cli/check_paper_package.py tests/test_audit_paper_abstract_conclusion.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_audit_paper_abstract_conclusion.py tests/test_check_paper_package.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Reviewer FAQ aligns to measurement hierarchy

- Updated `docs/experiments/paper_reviewer_faq.md` so the opening explanation
  now separates four localization layers from the two auxiliary measurement
  surfaces.
- The FAQ now names self-conditioning as its own localization layer and
  states that EQ-Bench is an aggregate public benchmark bridge while
  detector-derived Tier 3 remains a candidate-feature discovery surface.
- Added reviewer-facing language that aggregate EQ-Bench movement can disagree
  with a feature-specific propensity path, reinforcing that slop style is not
  one scalar quantity.
- Tightened `slop-check-paper-package` and its fixture so the reviewer FAQ
  cannot drift back to the stale three-layer framing.
- Regenerated release inventory, the 31-file draft bundle, submission-exit
  audit, and 115-row reproducibility manifest.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Venue adaptation docs get default packing plan

- Added a venue-neutral packing default to
  `docs/experiments/paper_venue_adaptation_runbook.md` and
  `docs/experiments/paper_venue_sizing_inventory.md`.
- The default keeps Figure 1, Figure 2, Figure 3, Figure 4, Table 1, Table 2,
  and Table 4 as the main-text starting point, with Figure 5 and appendix
  guardrail tables moving to appendix/supplement first if page limits are
  tight.
- The docs now explicitly say not to remove caveat columns, candidate-only
  labels, or measurement-boundary notes just to fit the venue template.
- Tightened `slop-check-paper-package` and its fixture so this practical
  venue-adaptation guidance remains present while the target venue is still
  unresolved.
- Regenerated release inventory, the 31-file draft bundle, submission-exit
  audit, and 115-row reproducibility manifest.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Phase 4 human-label coordinator checklist added

- Added `docs/experiments/phase4_human_labeling_execution_checklist.md` as a
  coordinator-facing bridge from the existing blind-label handoff to the
  paper-safe claim update path.
- The checklist records the external-label execution status, the rule to share
  only `human_annotation_handoff/annotator/`, the full-package agreement,
  adjudication, and summary command templates, and the rule to keep detector
  claims machine-detectable until real labeled rows exist.
- Updated `slop-materialize-phase4-human-handoff` so the coordinator checklist
  is copied into `human_annotation_handoff/coordinator/`. The handoff now
  reports 14 ready files, 0 missing files, and 6 redacted annotator CSVs.
- Updated paper-facing orientation docs (`EXPERIMENTS.md`, package index,
  readiness audit, scaffold, and caption appendix) so they name the checklist
  alongside the protocol, codebook, readiness audit, handoff, and summary.
- Tightened `slop-check-paper-package` and focused tests to require the
  checklist source, the copied coordinator checklist, and the key safety and
  claim-boundary phrases.
- Regenerated release inventory, the 31-file draft bundle, submission-exit
  audit, Phase 4 human handoff, and 117-row reproducibility manifest.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py src/slop_sftdiv/cli/materialize_phase4_human_handoff.py src/slop_sftdiv/cli/audit_paper_reproducibility_manifest.py tests/test_check_paper_package.py tests/test_materialize_phase4_human_handoff.py`;
  `uv run pytest tests/test_materialize_phase4_human_handoff.py tests/test_check_paper_package.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Pybiber substantive interpretation guarded in paper controls

- Updated `docs/experiments/paper_measurement_synthesis.md` so the
  reader-facing synthesis now records both substantive pybiber lessons:
  the Phase 1 register shift is not simply more hedging/intensification, and
  Dolci DPO chosen responses are more descriptive/expository while rejected
  responses are more personal, narrative, and mental-state framed.
- Updated `docs/experiments/paper_reviewer_faq.md` with the same DPO-register
  interpretation, explicitly separating it from a claim that chosen responses
  are uniformly more slop-like.
- Tightened `slop-check-paper-package` and its fixture so the measurement
  synthesis and reviewer FAQ continue to carry those pybiber interpretation
  boundaries.
- Regenerated release inventory, the 31-file draft bundle, submission-exit
  audit, and 117-row reproducibility manifest.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Pybiber DPO-register read added to new-reader path

- Updated `docs/experiments/paper_package_index.md` so the shortest
  new-reader path now states that the pybiber DPO contrast is
  register-specific: chosen responses are more descriptive/expository, while
  rejected responses are more personal, narrative, and mental-state framed.
- Updated `docs/experiments/paper_reader_glossary.md` so the Dolci DPO entry
  carries the same boundary alongside the existing warning that Dolci DPO is
  not a pure human-preference contrast.
- Tightened `slop-check-paper-package` and its fixture so package-index and
  glossary drift cannot silently drop this pybiber DPO-register interpretation.
- Regenerated release inventory, the 31-file draft bundle, submission-exit
  audit, and 117-row reproducibility manifest.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - EQ-Bench aggregate-score boundary added to new-reader path

- Updated `docs/experiments/paper_package_index.md` so the core-thesis and
  claim-boundary sections now state the substantive EQ-Bench lesson: the
  public aggregate score is useful for benchmark comparison, but its trajectory
  can disagree with feature-specific propensity and is mostly lexical-density
  evidence rather than a mechanism claim.
- Updated `docs/experiments/paper_reader_glossary.md` so the EQ-Bench entry
  says the score is not AF and its stage path can disagree with
  feature-specific propensity.
- Tightened `slop-check-paper-package` and its fixture so the package index
  and glossary preserve the EQ-Bench aggregate-score boundary.
- Regenerated release inventory, the 31-file draft bundle, submission-exit
  audit, and 117-row reproducibility manifest.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.

## 2026-06-18 - Submission blocker burn-down order added

- Added a `Submission Blocker Burn-Down Order` section to
  `docs/experiments/paper_readiness_audit.md` so the remaining blockers are
  operationalized as two explicit tracks: Phase 4 human labels and venue
  adaptation.
- The human-label track now names the submission-minimum path: collect two
  independent full-package blind label sheets, run agreement before
  discussion, adjudicate disagreements, apply adjudication to canonical JSONL,
  regenerate the human-perceptibility summary, and update C11/C12/C13 before
  promoting detector-derived families.
- The venue track now names the done condition: resolve every
  `decision_pending` venue row, apply figure/table sizing and export
  requirements, rerun figure/table/submission/reproducibility/package checks,
  and require `overall_submission_status` to become `ready`.
- Tightened `slop-check-paper-package` and its fixture so these blocker
  burn-down criteria remain in the readiness audit.
- Regenerated release inventory, the 31-file draft bundle, submission-exit
  audit, and 117-row reproducibility manifest.
- Validation passed:
  `uv run ruff check src/slop_sftdiv/cli/check_paper_package.py tests/test_check_paper_package.py`;
  `uv run pytest tests/test_check_paper_package.py`;
  `uv run slop-check-paper-package --root /home/user/slop`.
