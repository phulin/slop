# Phase 1 Gap Audit

Date: 2026-06-12

Scope: Phase 1 completion gaps only, based on `EXPERIMENTS.md` sections 1, 2, 3, and 7; `docs/experiments/stage1_strategy.md`; `docs/experiments/coordination_log.md`; and `configs/stage1/*.yaml`. No experiment commands were run for this audit.

## Summary

Phase 1 is scaffolded and now has a retained 10k-row Dolci SFT+DPO sample, but
it is not complete for publishable frequency-census claims. The current state
supports CLI/schema confidence, W&B logging, retained Dolci SFT/DPO census
tables, retained pair-delta output, length-aware Result A diagnostics, and a
1,400-row manual labeling queue. The main gaps are manifest-backed corpus
samples, manual precision labels/scoring, exact source identification for
replication data, Dolma stratification, and dataset-card/source verification
needed to interpret chosen/rejected roles.

## Completed

- Stage 1 plan and success criteria are defined in `stage1_strategy.md`, including corpus sampling, Tier-1 matcher port, precision validation, and Phase 1 census slices.
- Stage 1 configs exist for corpus sampling, Tier-1 matcher specs, Phase 1 census outputs, and later model-job efficiency defaults:
  - `configs/stage1/corpus_samples.yaml`
  - `configs/stage1/tier1_matchers.yaml`
  - `configs/stage1/phase1_census.yaml`
  - `configs/stage1/model_job_efficiency.yaml`
- Tier-1 feature inventory is frozen at the config/spec level: contrastive negation, slop lexicon, rule-of-three, list/header onset, generic hedging, stock openers/closers, punctuation density, paragraph rhythm, plus neutral controls.
- W&B conventions are specified for non-smoke Stage 1 jobs, including project, groups, tags, aggregate-only logging, and required throughput metrics.
- The coordination log records code-level scaffold completion: project setup, Tier-1 matchers, corpus streaming/sampling helpers, sampled census CLI, W&B helper, role-aware census behavior, pair-delta support, canonical Dolci notes, type/lint/test baseline, and initial tests.
- The coordination log records successful local verification after schema/normalizer updates: `uv run ty check src tests`, `uv run ruff check src tests`, and `uv run pytest -q`.
- Pair-delta plumbing exists at smoke/dry-run scale: the log records per-pair `chosen_minus_rejected` rows, feature-level pair analysis, sign-test p-values, BH-FDR q-values, and a length-aware diagnostic path.
- Retained 10k-row Dolci SFT+DPO census completed and logged to W&B:
  `stage1-olmo3-dolci-sft-dpo-10k-census-retained_sample`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/jnhxeiy2`). It scanned
  20,000 source rows, measured 29,995 response rows, processed 8,443,044 simple
  tokens, wrote 72 feature-rate rows, and emitted 79,960 DPO pair-delta rows.
- Retained 10k-row Dolci DPO Result A analysis completed and logged to W&B:
  `stage1-olmo3-dolci-dpo-10k-pair-analysis-retained_sample`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/njkg77h9`). It summarized
  79,960 pair-delta rows into 16 source/subset/feature rows and found 9
  BH-FDR-significant rows before manual precision validation.
- Retained 10k-row Dolci SFT+DPO hit sampling completed and logged to W&B:
  `stage1-olmo3-dolci-sft-dpo-10k-hit-sampling-retained_sample`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/t6wdmyj2`). It produced 200
  sampled rows for each of 7 features, 1,400 CSV records total, for manual
  precision labeling.

## Dry-Run Only

- Dolci DPO tiny census:
  - Input: `allenai/Dolci-Instruct-DPO`
  - Size: `--sample-size 2 --max-scan 2`
  - Output: 16 aggregate feature-rate rows split by `chosen` and `rejected`
  - Status: smoke only
- Dolci DPO pair-delta smoke:
  - Size: 2 preference pairs
  - Output: 16 aggregate feature-rate rows and 16 per-pair delta rows
  - Status: smoke only
- Dolci SFT metadata smoke:
  - Input: `allenai/Dolci-Instruct-SFT`
  - Size: `--sample-size 2 --max-scan 2`
  - Status: smoke only
- Dolma 3 tiny census:
  - Input: `allenai/dolma3_mix-6T-1025-7B`
  - Size: `--sample-size 2 --max-scan 2`
  - Output: 8 aggregate rows under inferred stratum `web_cc`
  - Status: smoke only; one HF read-timeout retry and about 51 seconds for two docs
- Throughput metrics smoke:
  - Input: Dolci SFT
  - Size: `--sample-size 1 --max-scan 1`
  - Output: corpus throughput and RSS metrics in W&B
  - Status: smoke only
- Tagged throughput calibration smoke:
  - Group: `corpus-sampling`
  - Tags include `dry_run`
  - Status: smoke only
- Precision hit-sampling smoke:
  - Input: Dolci SFT
  - Size: `--sample-size 1 --max-scan 1`
  - Output: 26 candidate hits and 10 sampled hit rows locally for labeling
  - Status: dry run only; not a retained 200-hit validation set
- Precision label scoring smoke:
  - Input: tiny synthetic labeled CSV
  - Output: 2 scored features, 1 demoted below target
  - Status: scoring-path smoke only; not real manual validation
- Result A pair-analysis smoke:
  - Input: 2-pair Dolci DPO pair-delta sample
  - Output: 16 pair-delta rows summarized into 8 feature rows
  - Status: dry run only
- Dolci DPO 32-pair census dry run:
  - Size: 32 preference pairs, 64 response rows, 24,014 simple tokens
  - Output: 32 aggregate census rows and 256 pair-delta rows
  - Status: dry run only
- Dolci DPO 32-pair first-pass and length-aware Result A diagnostics:
  - Output: 16 feature rows; no Tier-1 feature BH-FDR significant at dry-run size
  - Status: diagnostic only; length imbalance remains material

## Retained-Sample Needed

- The Phase 1 corpus artifacts named by `stage1_strategy.md` and `corpus_samples.yaml` still need retained, manifest-backed outputs:
  - `artifacts/stage1/corpora/*.jsonl`
  - `artifacts/stage1/corpora/sample_manifest.parquet`
  - `artifacts/stage1/corpora/source_card_notes.md`
- The Phase 1 matcher artifacts named by `tier1_matchers.yaml` still need retained outputs, unless produced elsewhere outside the audited evidence:
  - `artifacts/stage1/matchers/feature_specs.json`
  - `artifacts/stage1/matchers/matcher_version.txt`
  - `artifacts/stage1/matchers/unit_smoke_results.json`
- The Phase 1 validation artifacts need retained real samples and labels:
  - `artifacts/stage1/validation/hit_samples/*.jsonl`
  - `artifacts/stage1/validation/labels/*.csv`
  - `artifacts/stage1/validation/precision_report.md`
- The Phase 1 census artifacts need retained non-dry-run tables:
  - `artifacts/stage1/census/feature_rates_by_corpus.parquet`
  - `artifacts/stage1/census/feature_rates_by_stratum.parquet`
  - `artifacts/stage1/census/preference_pair_deltas.parquet`
  - `artifacts/stage1/census/census_summary.md`
  The retained 10k Dolci run currently writes CSVs under
  `artifacts/stage1/census/olmo3_dolci_*_10k_*.csv`; the named parquet/summary
  artifacts are still needed for the formal Phase 1 package.
- Primary OLMo retained samples still need enough rows and metadata coverage to
  satisfy the Stage 1 go/no-go:
  - Dolma 3 with at least three populated non-code strata
  - Dolci SFT target responses with verified provenance fields
  - Dolci DPO chosen/rejected rows with verified role semantics
  - Dolci RL if kept for final-stage data-rate context
- The 10k Dolci SFT+DPO retained sample is large enough for a useful Result A
  development artifact, but it is not enough by itself to close Phase 1 because
  it lacks manual precision validation, manifest-backed corpus packaging,
  Dolma strata, SmolLM3/Tulu replication sources, and chosen/rejected
  construction verification.

Recommended next commands/artifacts, not run:

```bash
uv run slop-census --input allenai/Dolci-Instruct-DPO --sample-size <retained_pair_count> --max-scan <scan_cap> --pair-output <path> --wandb-mode online
uv run slop-census --input allenai/Dolci-Instruct-SFT --sample-size <retained_row_count> --max-scan <scan_cap> --wandb-mode online
uv run slop-census --input allenai/dolma3_mix-6T-1025-7B --sample-size <retained_row_count> --max-scan <scan_cap> --wandb-mode online
```

Expected retained artifacts from the next pass:

- A sample manifest with content hashes, seeds, scan caps, source datasets, split names, strata, provenance, text roles, and token/sentence/paragraph counts.
- Census tables grouped by corpus, ladder, source dataset, split, stratum, provenance, text role, feature ID, and subtype.
- Pair-delta table with one-to-one chosen/rejected alignment and length covariates.
- W&B runs without `dry_run` tags for retained Stage 1 jobs.

## Blocked By Manual Labeling

- Tier-1 matcher precision has not passed the required 200-hit manual validation per feature.
- The retained 10k hit-sampling run produced 1,400 sampled CSV records: 200
  each for contrastive negation, list/header/bold lead-in, punctuation/rhythm,
  rule-of-three, slop lexicon, stock closers, and stock openers. These rows
  still require manual labels before precision can be scored.
- The label-scoring smoke used a tiny synthetic CSV, so it validates the scoring path but not real matcher precision.
- Core features cannot be treated as Phase 1-validated until the precision report shows at least 0.90 precision for:
  - `contrastive_negation`
  - `slop_lexicon`
  - `list_header_onset`
  - `stock_openers_closers`
  - `punctuation_density`
- Standard features still need validation or demotion decisions:
  - `rule_of_three`
  - `generic_hedging`
- `paragraph_rhythm` is already exploratory in config, but still needs documented behavior if included in tables.

Recommended next commands/artifacts, not run:

```bash
uv run slop-sample-hits --input <retained_corpus_or_manifest> --sample-size <enough_for_200_hits_per_feature> --max-scan <scan_cap> --output artifacts/stage1/validation/hit_samples/<run_id>.jsonl --wandb-mode online
uv run slop-score-labels --labels artifacts/stage1/validation/labels/<label_file>.csv --output artifacts/stage1/validation/precision_report.md --wandb-mode online
```

Required manual artifact:

- `artifacts/stage1/validation/labels/*.csv` with labels `exact`, `partial`, `false_positive`, or `ambiguous`, where ambiguous labels count against precision unless a feature spec explicitly treats them as valid.

## Blocked By Missing Source Identification

- Dolci DPO construction and chosen/rejected role interpretation remain unresolved for final Result A claims. `EXPERIMENTS.md` says to verify whether `Dolci-Instruct-DPO-7B` uses Delta-Learning strong-vs-weak model responses before locking interpretation.
- `corpus_samples.yaml` explicitly blocks Result A interpretation if Dolci DPO construction is not verified or chosen/rejected role mapping is ambiguous.
- The coordination log still lists open go/no-go questions about whether Dolci SFT/DPO/RL datasets expose stable response and pair fields, and whether DPO chosen/rejected roles can be verified well enough to interpret Result A.
- Dolma 3 tiny-stream schema remains unverified according to the coordination log. The tiny census inferred `web_cc`, but this does not satisfy the required stratified Dolma 3 sample across at least three populated strata.
- SmolLM3 pretraining source identification is incomplete:
  - Config status: `identify_exact_public_sources`
  - Needed for equivalent strata: `web_cc`, `forums_qa`, `wiki`, `scientific`, with code excluded
- SmolLM3 preference source identification is incomplete:
  - Config status: `identify_exact_source`
  - The exact Tulu-3/APO preference source must separate RM/human-ranked pairs from synthetic Qwen3-32B/0.6B pairs.
- Dolci SFT provenance split remains to be verified from mixture metadata. Phase 1 requires rates by provenance, and the SFT human-written subset is important for the target-distribution baseline.

Recommended next artifacts, not run:

- `artifacts/stage1/corpora/source_card_notes.md` with explicit citations/notes for:
  - Dolma 3 sample source and usable stratum fields
  - Dolci SFT response fields and provenance metadata
  - Dolci DPO prompt, chosen, rejected, and pair ID fields
  - Dolci DPO construction semantics for Delta-Learning versus human/RM preference interpretation
  - Dolci RL schema and whether it remains in Stage 1
  - SmolLM3 exact pretraining mixture sources
  - SmolTalk2 SFT target-response fields
  - Exact SmolLM3 APO/Tulu-3 preference source and synthetic-pair separation

## Phase 1 Exit Status

Current status: not ready to exit Stage 1 / Phase 1.

Exit criteria still unmet:

- Primary OLMo corpora are not yet retained, sampled, and manifest-backed.
- SmolLM3 replication sources are not sampled and still have source-identification blockers.
- Core Tier-1 matchers do not yet have real 200-hit precision validation.
- Retained Dolci SFT/DPO census and pair-analysis CSVs now exist for a 10k
  sample and are logged to W&B, but the formal named parquet/manifest/summary
  artifacts are still missing.
- Pair IDs and chosen/rejected alignment have retained-run evidence for Dolci
  DPO, but chosen/rejected construction semantics are still not verified.
- W&B now has retained Dolci SFT/DPO Stage 1 records, but retained Dolma,
  SmolLM3/Tulu, precision-scoring, and manifest-backed corpus artifacts are
  still needed.

Do not proceed to statistical claims until:

- `source_card_notes.md`, retained corpus samples, precision labels, precision report, census summary, and census parquet outputs exist.
- Chosen/rejected role semantics are verified.
- Pair alignment is checked on the retained sample.
- Length covariates are included in retained Result A analysis, given the observed 32-pair dry-run length imbalance.
