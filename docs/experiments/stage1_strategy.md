# Stage 1 Implementation Strategy

This document translates `EXPERIMENTS.md` Week 1-2 into executable slices owned by experiment orchestration. Stage 1 produces validated corpora, Tier-1 matcher measurements, and a Phase 1 data-corpus census. It deliberately avoids model inference except for small tokenizer/dataset smoke tests; later model jobs should reuse the same run metadata and efficiency conventions.

## Objectives

1. Build sampled, stratified corpora for the OLMo 3 primary ladder and SmolLM3 replication ladder.
2. Port Tier-1 feature matchers with frozen feature specs: span matcher, opportunity context, and normalization rules.
3. Precision-validate each Tier-1 matcher on 200 sampled hits, with documented demotion criteria.
4. Run Phase 1 frequency census over data corpora: pretrain strata, SFT targets, preference chosen, and preference rejected.
5. Emit auditable artifacts for Week 3 chosen-vs-rejected analysis and later teacher-forced propensity harness work.

## Executable Slices

### Slice 1: Corpus Inventory And Sampling

Config: `configs/stage1/corpus_samples.yaml`

Tasks:
- Verify dataset cards and splits for OLMo 3 data: Dolma 3 sample source, `allenai/Dolci-Instruct-SFT-7B`, `allenai/Dolci-Instruct-DPO-7B`, and `allenai/Dolci-Instruct-RL-7B`.
- Verify SmolLM3 replication data sources: public pretraining mixture equivalent, SmolTalk2 SFT data, and Tulu-3 preference subset plus synthetic Qwen3-32B/0.6B pairs where available.
- Build deterministic samples using `seed: 1729`.
- Tag every row with `corpus`, `source_dataset`, `split`, `stratum`, `provenance`, `prompt_id`, `doc_id`, and token-count fields.
- Preserve chosen/rejected pair IDs for paired tests.

Expected artifacts:
- `artifacts/stage1/corpora/*.jsonl`
- `artifacts/stage1/corpora/sample_manifest.parquet`
- `artifacts/stage1/corpora/source_card_notes.md`

Go/no-go:
- Go if all primary OLMo corpora are loadable, pair IDs are stable for DPO, and at least three pretrain strata are populated.
- No-go if Dolci DPO construction cannot be verified well enough to label chosen/rejected interpretation. Continue corpus work, but block Result A interpretation until resolved.

### Slice 2: Tier-1 Matcher Port

Config: `configs/stage1/tier1_matchers.yaml`

Tasks:
- Implement matchers for contrastive negation, slop lexicon, rule-of-three, list/header onset, generic hedging, stock openers/closers, punctuation density, and paragraph rhythm.
- For each feature, freeze the default normalization unit: per 1k tokens, per sentence, per opportunity, or document-level scalar.
- For each feature, define `opportunity_context` even if Stage 1 only uses corpus frequencies. These definitions become the contract for Week 3-5 teacher-forced propensity.
- Add a neutral-control basket for later AF calibration: `for example`, `such as`, `in particular`, and `as a result`.

Expected artifacts:
- `artifacts/stage1/matchers/feature_specs.json`
- `artifacts/stage1/matchers/matcher_version.txt`
- `artifacts/stage1/matchers/unit_smoke_results.json`

Go/no-go:
- Go if every Tier-1 feature has a deterministic matcher, opportunity definition, and smoke-test fixture.
- Demote individual features to exploratory if the opportunity context cannot be defined clearly before measurement.

### Slice 3: Precision Validation

Config: `configs/stage1/phase1_census.yaml`

Tasks:
- Sample up to 200 hits per feature across corpora, stratified by corpus and source where possible.
- Hand-label each sampled hit as exact, partial, false positive, or ambiguous.
- Refine matchers until precision is at least 0.90 or demote the feature.
- Record known systematic false positives so downstream analysis can explain exclusions.

Expected artifacts:
- `artifacts/stage1/validation/hit_samples/*.jsonl`
- `artifacts/stage1/validation/labels/*.csv`
- `artifacts/stage1/validation/precision_report.md`

Go/no-go:
- Go if core features have precision >= 0.90: contrastive negation, slop lexicon pooled index, list/header onset, stock openers/closers, and punctuation density.
- No-go for Phase 1 publication if more than two core features fail precision. Continue with passing features only.

### Slice 4: Phase 1 Data-Corpus Census

Config: `configs/stage1/phase1_census.yaml`

Tasks:
- Run validated Tier-1 matchers over sampled data corpora.
- Report per-1k-token rates for all features where meaningful.
- Report per-sentence rates for clause-level features.
- Report per-opportunity rates wherever Stage 1 opportunity extraction is implemented.
- Emit chosen-vs-rejected per-pair deltas without fitting final statistics yet.
- Include length fields so Week 3 can control for token-count effects.

Expected artifacts:
- `artifacts/stage1/census/feature_rates_by_corpus.parquet`
- `artifacts/stage1/census/feature_rates_by_stratum.parquet`
- `artifacts/stage1/census/preference_pair_deltas.parquet`
- `artifacts/stage1/census/census_summary.md`

Go/no-go:
- Go if chosen/rejected rows align 1:1 by pair ID and length distributions match dataset-card expectations.
- Bug-suspect no-go if no feature, including length and markdown/list markers, differs between chosen and rejected on either ladder.

## Component Task Notes

### Corpus Component

Responsibilities:
- Dataset loading, filtering, deterministic sampling, schema normalization, and manifest generation.
- Pretrain source-domain stratification: web/CC, forums/Q&A, wiki, scientific/olmOCR, and optional code-excluded audit bucket.
- SFT provenance tagging: synthetic/LLM-generated, human-written, unknown, and mixed.

Implementation notes:
- Use streaming reads for large pretrain corpora.
- Measure sampled corpus throughput before full runs: rows/sec, MB/sec, tokens/sec, peak RSS, and compressed artifact size.
- Start with small calibration samples, then scale only after matcher smoke tests pass.

### Matcher Component

Responsibilities:
- Feature-specific span extraction.
- Opportunity extraction contract for later teacher-forced jobs.
- Token, sentence, paragraph, and document-level denominator calculation.

Implementation notes:
- Prefer deterministic regex/POS pipelines with explicit versioning.
- Keep tokenizer-dependent logic out of Stage 1 matchers except token counts; Week 3-5 initiating-token sets are model-tokenizer-specific.
- Store raw spans and denominators so rates can be recomputed without rerunning extraction.

### Validation Component

Responsibilities:
- Hit sampling, labeling templates, precision calculation, and demotion decisions.
- Precision report with examples of accepted and rejected hits.

Implementation notes:
- Keep validation samples credential-free and free of raw secrets.
- Use the same random seed for reproducible hit sets.
- Ambiguous labels count against precision for go/no-go unless a feature spec explicitly treats them as valid.

### Census Component

Responsibilities:
- Batch matcher execution over normalized corpora.
- Rate table emission and pair-delta table emission.
- W&B logging of aggregate metrics and artifact references.

Implementation notes:
- Use content hashes for corpus samples and matcher specs.
- Do not log raw full text to W&B by default. Log aggregate metrics, manifests, and small redacted examples only.

## W&B Logging Conventions

See `docs/experiments/wandb.md` for setup details. Stage 1 runs should:
- Load `WANDB_API_KEY` from `.env` at runtime.
- Never write the API key into configs, command histories, summaries, artifacts, or crash logs.
- Use project `slop-stage1`.
- Use run groups matching the executable slice: `corpus-sampling`, `matcher-port`, `precision-validation`, or `phase1-census`.
- Use tags for ladder, corpus, feature tier, and dry-run/full-run status.
- Log sampled corpus measurements before each full corpus pass.

Recommended metric names:
- `corpus/rows_per_sec`
- `corpus/tokens_per_sec`
- `corpus/sample_tokens`
- `corpus/peak_rss_gb`
- `matcher/hits`
- `matcher/opportunities`
- `matcher/precision`
- `census/rate_per_1k_tokens`
- `census/rate_per_sentence`
- `census/rate_per_opportunity`
- `preference/mean_chosen_minus_rejected`

## Compute-Efficiency Guidance

Stage 1:
- Run a sampled-corpus measurement before every new full corpus pass. Record wall time, rows/sec, tokens/sec, peak RSS, and output artifact size.
- Use streaming dataset reads and bounded writer buffers for Dolma-scale sources.
- Cache normalized text, token counts, sentence counts, paragraph counts, matcher spans, and denominator tables separately.
- Prefer a two-pass plan: small smoke sample, then calibrated full Stage 1 sample.
- Keep Biber/Tier-2 extraction out of the default critical path unless Tier-1 throughput has enough headroom.

Later model jobs:
- Use vLLM for generation grids when possible.
- Use a logits-capture harness for teacher-forced propensity, with cached tokenized batches per checkpoint tokenizer.
- Evaluate `torch.compile` after the teacher-forced harness is numerically validated. Treat it as an optimization toggle, not a default assumption.
- Benchmark eager vs. `torch.compile` on a fixed 1k-prompt shard before committing full A100 time. Track compile warmup, tokens/sec, memory, and exact-logit agreement tolerance.
- Keep Tier-1 first. Biber/Tier-3 should reuse cached generations and denominators.

## Stage 1 Exit Criteria

Go to Week 3 if:
- Primary OLMo Stage 1 corpora are sampled and manifest-backed.
- SmolLM3 replication sources are either sampled or have a documented blocker and fallback.
- Core Tier-1 matchers pass precision gates.
- Phase 1 census tables exist with stable row counts, denominators, and pair IDs.
- W&B contains aggregate metrics and artifact references for all non-dry-run Stage 1 jobs.
- Known limitations are recorded in `census_summary.md` and `precision_report.md`.

Do not proceed to statistical claims if:
- Pair alignment is broken.
- Dataset-card verification changes chosen/rejected interpretation.
- Precision validation fails for the core feature set.
- Sampled corpus throughput suggests the planned full run would exceed the available compute window without a revised sample size.
