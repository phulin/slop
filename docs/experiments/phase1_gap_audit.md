# Phase 1 Gap Audit

Date: 2026-06-12

Scope: Phase 1 completion gaps only, based on `EXPERIMENTS.md` sections 1, 2, 3, and 7; `docs/experiments/stage1_strategy.md`; `docs/experiments/coordination_log.md`; and `configs/stage1/*.yaml`. No experiment commands were run for this audit.

## Summary

Phase 1 is scaffolded and now has retained Dolci SFT/DPO and Dolma 3 sampled
artifacts, but it is not complete for publishable frequency-census claims. The
current state supports CLI/schema confidence, W&B logging, retained Dolci
SFT/DPO census tables, retained pair-delta output, metadata-aware Result A
diagnostics, a retained Dolma 3 pretrain sample/census, and a 1,400-row manual
labeling queue. The main gaps are manual precision labels/scoring, exact source
identification for replication data, formal package summaries, and final
dataset-card/source verification needed to make chosen/rejected interpretation
claims.

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
- Retained 10k-row Dolci artifact manifest completed and logged to W&B:
  `stage1-olmo3-dolci-10k-retained-artifact-manifest`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/t9mjtmbo`). It hashes and
  counts the 10k feature-rate, pair-delta, pair-analysis, and hit-sample CSVs.
  The manifest records 4 local artifacts, 11,810,749 total bytes, and 81,448
  CSV records.
- Retained 10k-row Dolci source metadata probe completed and logged to W&B:
  `stage1-olmo3-dolci-10k-source-metadata-probe`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/xieewrhy`). It verifies
  retained-prefix SFT source/domain counts and DPO preference-type/model
  counts.
- Retained 10k-row Dolci DPO metadata-aware Result A analysis completed and
  logged to W&B:
  `stage1-olmo3-dolci-dpo-10k-metadata-pair-analysis-retained_sample`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/sgszvu4u`). Input:
  `artifacts/stage1/census/olmo3_dolci_dpo_10k_metadata_pair_deltas.csv`.
  Output:
  `artifacts/stage1/census/olmo3_dolci_dpo_10k_metadata_pair_analysis.csv`.
  It grouped 79,960 pair-delta rows by
  `source/subset/preference_type/chosen_model/rejected_model/feature`,
  producing 3,352 grouped feature rows and 72 BH-FDR-significant rows before
  manual precision validation. Grouped row counts by `preference_type` were
  `llm_judged` 3,312, `multiturn_self_talk` 16,
  `multiturn_synthetic_context` 16, and `delta_learning` 8.
- Retained Dolma 3 20k-scan pretrain sample completed and logged to W&B:
  `stage1-olmo3-dolma3-20k-scan-retained-sample`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/l0ms0k7t`). It streamed
  20,000 rows from `allenai/dolma3_mix-6T-1025-7B`, retained 1,401 normalized
  rows with seed `1729`, excluded code, and populated four non-code strata:
  `web_cc`, `forums_qa`, `wiki`, and `scientific`. Local retained outputs
  include `artifacts/stage1/corpora/olmo3_dolma3_20k_scan_retained_sample.jsonl`
  and `artifacts/stage1/corpora/sample_manifest.parquet`.
- Retained Dolma 3 stratum census completed and logged to W&B:
  `stage1-olmo3-dolma3-20k-scan-stratum-census-retained_sample`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/3my331x7`). It processed
  2,129,938 simple tokens and wrote 32 feature-rate rows to
  `artifacts/stage1/census/olmo3_dolma3_20k_scan_feature_rates.csv`.
- Retained Dolma 3 artifact manifest completed and logged to W&B:
  `stage1-olmo3-dolma3-20k-scan-artifact-manifest`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/75vyt0mb`). It records 4
  retained artifacts, 16,446,028 total bytes, and 4,235 counted records.
- Retained Dolci SFT corpus package completed and logged to W&B:
  `stage1-olmo3-dolci-sft-10k-retained-corpus-package-schema-fix`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/t5l79cla`). It retained
  10,000 normalized `target_response` rows from
  `allenai/Dolci-Instruct-SFT`, measured 1,806,697 simple tokens, and wrote
  `artifacts/stage1/corpora/olmo3_dolci_sft_10k_retained_sample.jsonl` plus
  Parquet/CSV/JSON/Markdown per-record manifests.
- Retained Dolci DPO corpus package completed and logged to W&B:
  `stage1-olmo3-dolci-dpo-10k-complete-pair-corpus-package-schema-fix`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/bucn0h4p`). It scanned
  10,006 source rows to retain 10,000 complete preference pairs and wrote
  20,000 normalized text rows: 10,000 `chosen` and 10,000 `rejected`. Original
  `prompt_id` is preserved separately from a unique row-index-qualified
  `pair_id`; validation found 10,000 pair IDs and zero bad role pairs.
- Retained Dolci SFT/DPO corpus package manifest completed and logged to W&B:
  `stage1-olmo3-dolci-sft-dpo-10k-corpus-package-manifest`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/dzrfn409`). It records 10
  local artifacts, 186,303,114 total bytes, and 90,000 counted
  JSONL/Parquet/CSV records.
- SmolLM3/Tulu source-identification probe completed and logged to W&B:
  `stage1-smollm3-tulu-source-identification-probe`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/wiedc2oj`). Local artifact:
  `artifacts/stage1/corpora/smollm3_tulu_source_probe.json`. It recorded 5
  dataset probes, 2 model probes, 3 dataset searches, 1 model search, 7 sample
  rows, and 80 search-result rows. This narrows the replication-source search
  space, but does not close source verification or authorize retained
  SmolLM3/Tulu sampling.
- SmolTalk2 config-specific source probe completed and logged to W&B:
  `stage1-smollm3-smoltalk2-config-source-probe`
  (`https://wandb.ai/phulin-self/slop-stage1/runs/wo6496wj`). Local artifact:
  `artifacts/stage1/corpora/smollm3_tulu_config_source_probe.json`. It probed
  five exact dataset specs and model `HuggingFaceTB/SmolLM3-3B`. Each dataset
  spec loaded successfully with 2 shape-only sample rows:
  `HuggingFaceTB/smoltalk2::Preference::llama_3.1_tulu_3_8b_preference_mixture_no_think`,
  `HuggingFaceTB/smoltalk2::Preference::tulu_3_8b_pref_mix_Qwen3_32B_Qwen3_0.6B_think`,
  `HuggingFaceTB/smoltalk2::SFT::smoltalk_smollm3_everyday_conversations_no_think`,
  `HuggingFaceTB/smoltalk2::SFT::OpenThoughts3_1.2M_think`, and
  `HuggingFaceTB/smoltalk2::Mid::OpenThoughts3_1.2M`.
- The config-specific SmolTalk2 probe closes the generic-config loader issue
  and proves config/split-aware bounded probing works. It does not finish Phase
  1: remaining SmolLM3/Tulu gaps are broader split/source count census, target
  response extraction/normalization, Tulu construction semantics, and
  pretraining recipe weights.

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

- The Phase 1 corpus artifacts named by `stage1_strategy.md` and `corpus_samples.yaml` are partially retained:
  - `artifacts/stage1/corpora/*.jsonl`
  - `artifacts/stage1/corpora/sample_manifest.parquet`
  - `artifacts/stage1/corpora/source_card_notes.md`
  Dolma 3 now has a retained JSONL sample and the formal
  `sample_manifest.parquet`; Dolci SFT/DPO now have retained corpus-package
  JSONL and per-corpus manifests; replication-ladder corpus packaging remains
  incomplete. SmolTalk2 config-specific probing works, but retained
  replication samples still need source-count census and response extraction
  rules.
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
  artifacts are still needed for the formal Phase 1 package. A retained
  artifact manifest exists locally and in W&B for the current CSV outputs.
- Primary OLMo retained samples still need enough rows and metadata coverage to
  satisfy the Stage 1 go/no-go:
  - Dolma 3 now has a bounded 20k-scan sample with four populated non-code
    strata, but rare strata are prefix-skewed and underfilled relative to the
    1,000-row cap
  - Dolci SFT target responses now have retained package artifacts, with
    source/domain metadata suitable for retained-prefix provenance analysis
  - Dolci DPO chosen/rejected rows now have retained package artifacts with
    exactly two roles per unique package `pair_id`, but final construction
    semantics still require careful qualification
  - Dolci RL if kept for final-stage data-rate context
- The 10k Dolci SFT+DPO retained sample is large enough for a useful Result A
  development artifact, but it is not enough by itself to close Phase 1 because
  it lacks manual precision validation, retained SmolLM3/Tulu replication
  samples, and final chosen/rejected construction verification.

## Dolci Retained Corpus Packaging Pass

Status: completed for retained 10k SFT rows and 10k complete DPO pairs.

- SFT package paths:
  - `artifacts/stage1/corpora/olmo3_dolci_sft_10k_retained_sample.jsonl`
  - `artifacts/stage1/corpora/olmo3_dolci_sft_10k_sample_manifest.parquet`
  - `artifacts/stage1/corpora/olmo3_dolci_sft_10k_sample_manifest.csv`
  - `artifacts/stage1/corpora/olmo3_dolci_sft_10k_sample_manifest.json`
  - `artifacts/stage1/corpora/olmo3_dolci_sft_10k_sample_manifest.md`
- DPO package paths:
  - `artifacts/stage1/corpora/olmo3_dolci_dpo_10k_retained_sample.jsonl`
  - `artifacts/stage1/corpora/olmo3_dolci_dpo_10k_sample_manifest.parquet`
  - `artifacts/stage1/corpora/olmo3_dolci_dpo_10k_sample_manifest.csv`
  - `artifacts/stage1/corpora/olmo3_dolci_dpo_10k_sample_manifest.json`
  - `artifacts/stage1/corpora/olmo3_dolci_dpo_10k_sample_manifest.md`
- Package-level manifest:
  `artifacts/stage1/corpora/olmo3_dolci_sft_dpo_10k_corpus_package_manifest.*`.
- Acceptance checks passed locally:
  - SFT rows have only `target_response` measured text.
  - DPO rows align as two rows per unique package `pair_id`, one `chosen` and
    one `rejected`.
  - Original DPO `prompt_id` is preserved separately from unique `pair_id`,
    because 37 sampled prompt IDs appeared more than once.
  - Manifest counts match JSONL row counts and W&B artifact metadata.
  - Package is retained/non-dry-run and logged under `corpus-sampling`.

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
- The dataset-card and retained-prefix probe now verify that Dolci DPO is a
  mixed construction with Delta Learning, LLM-judged, and multiturn subsets.
  The retained metadata-aware pair analysis further narrows the blocker by
  stratifying Result A rows by `preference_type`, `chosen_model`, and
  `rejected_model`, but it does not close the blocker for final Phase 1 claims:
  manual precision validation, final chosen/rejected construction semantics,
  and SmolLM3/Tulu replication-source work remain.
- Top metadata-aware significant examples include `delta_learning`
  `qwen3-no_reasoning-32b` vs `qwen3-no_reasoning-0.6b` for `stock_closers`
  (`n=4,730`, `mean_delta=0.169827...`, `p=2.628e-15`, `q=8.809e-12`,
  `chosen_gt_rejected`) and `punctuation_rhythm` (`n=4,730`,
  `mean_delta=4.271586...`, `p=5.253e-11`, `q=8.804e-08`), plus
  `llm_judged` `gpt-120b` vs `olmo2-1b` `punctuation_rhythm` (`n=64`,
  `mean_delta=95.202...`, `p=7.672e-10`, `q=8.572e-07`).
- `corpus_samples.yaml` explicitly blocks final Result A interpretation if
  chosen/rejected role mapping is ambiguous. Current schema evidence supports
  response extraction and pair identity, but construction-aware interpretation
  remains required.
- The coordination log still lists open go/no-go questions about whether Dolci
  RL remains in scope and whether DPO chosen/rejected construction semantics
  can be verified well enough to interpret Result A without overclaiming.
- Dolma 3 tiny-stream schema and retained stratified sampling now have local
  evidence. The 20k-scan sample satisfies the at-least-three-strata Stage 1
  sample condition, while preserving a note that rare strata are underfilled in
  the streamed prefix.
- SmolLM3 pretraining source identification is incomplete:
  - Config status: `identify_exact_public_sources`
  - Needed for equivalent strata: `web_cc`, `forums_qa`, `wiki`, `scientific`, with code excluded
- Primary Hugging Face metadata now identifies candidate SmolLM3 pretraining
  sources via the `HuggingFaceTB/smollm3-pretraining-datasets` collection, but
  this is not yet enough to sample. The exact source IDs, stage assignment, and
  mixture weights must be confirmed from the linked `huggingface/smollm`
  training recipe before Phase 1 replication sampling.
- The SmolLM3/Tulu source-identification probe adds two useful metadata
  sources but does not resolve pretraining weights:
  `HuggingFaceTB/smollm3-configs` is useful metadata/config-card evidence but
  not loadable as dataset files, and `HuggingFaceTB/smollm3-blueprint` is
  documentation/PDF-oriented with sample loading currently blocked on
  `pdfplumber`.
- Candidate SmolLM3 pretraining source families to verify are web/CC
  (`fineweb-edu`, `dclm-baseline-1.0`, `FineWeb2-HQ`, `fineweb-2`,
  `smollm-corpus`, `dolmino-mix-1124`), Q&A/forum-like sources
  (`stackexchange_2025_md`, `stack-edu`), math/reasoning sources
  (`finemath`, `MegaMath`, `OpenMathReasoning`, `natural_reasoning`), and code
  sources for exclusion or an audit bucket (`the-stack-v2`,
  `issues-kaggle-notebooks`, `OpenCodeReasoning`).
- SmolTalk2 SFT source identification is incomplete:
  - Dataset candidate: `HuggingFaceTB/smoltalk2`, config/subset `SFT`
  - Primary card facts: 25 SFT datasets, split into `think` and `no_think`
    variants, with chat-style `messages` and `chat_template_kwargs`
  - Needed before sampling: exact config names, split names, assistant-response
    extraction schema, reasoning-trace tagging, source/provenance fields, and
    row-count reconciliation against the card
- The live probe confirms `HuggingFaceTB/smoltalk2` configs `Mid`,
  `Preference`, and `SFT`. Generic SmolTalk2 sample loading without a config
  fails, so the next implementation step is config/split-aware bounded probing
  rather than retained sampling.
- SmolTalk2 SFT splits are source-labeled and include many `think` and
  `no_think` splits, including `OpenThoughts3_1.2M_think`,
  `smoltalk_smollm3_everyday_conversations_no_think`,
  `smoltalk_smollm3_smol_magpie_ultra_no_think`, and
  `tulu_3_sft_personas_instruction_following_no_think`. Remaining work is to
  probe those splits' schemas and response extraction consistently.
- SmolLM3 preference source identification is incomplete:
  - Config status: `identify_exact_source`
  - Candidate dataset: `HuggingFaceTB/smoltalk2`, config/subset `Preference`
  - The exact Tulu-3/APO preference source must separate the no-think
    `llama_3.1_tulu_3_8b_preference_mixture_no_think` source from the synthetic
    `tulu_3_8b_pref_mix_Qwen3_32B_Qwen3_0.6B_think` source.
  - The Qwen source must be interpreted as synthetic strong-vs-weak pair data:
    chosen generated by Qwen3-32B and rejected generated by Qwen3-0.6B, pending
    bounded schema confirmation. Do not pool it with the candidate human/RM
    Tulu preference source for Result A.
- The probe also confirms that
  `allenai/llama-3.1-tulu-3-8b-preference-mixture` and
  `allenai/llama-3.1-tulu-3-70b-preference-mixture` load with default `train`
  rows shaped as `id`, `source`, `prompt`, `chosen`, and `rejected`; `chosen`
  and `rejected` are message lists. This supports bounded Tulu schema work but
  still does not close source construction semantics.
- All SmolLM3 live probes must be bounded and W&B-logged before retained
  sampling: fixed scan caps, deterministic seed, no raw-text W&B logging,
  source/card URLs in config, schema fields, split/config names, source counts,
  pair-role counts, and failure modes. Use primary Hugging Face dataset/model
  cards and linked training recipes as the authority for source semantics.
- Dolci SFT provenance split remains to be verified from mixture metadata. Phase 1 requires rates by provenance, and the SFT human-written subset is important for the target-distribution baseline.
- `docs/experiments/source_card_notes.md` now records primary-source card facts
  and retained-prefix metadata probe results for Dolci SFT, Dolci DPO, Dolma
  3, and the in-progress SmolLM3 replication-source plan.

Recommended next artifacts, not run:

- `artifacts/stage1/corpora/source_card_notes.md` with explicit citations/notes for:
  - Dolma 3 sample source and usable stratum fields
  - Dolci SFT response fields and provenance metadata
  - Dolci DPO prompt, chosen, rejected, and pair ID fields
  - Dolci DPO construction semantics for Delta-Learning versus human/RM preference interpretation
  - Dolci RL schema and whether it remains in Stage 1
  - SmolLM3 exact pretraining mixture sources
  - SmolTalk2 SFT config/schema and target-response fields
  - Exact SmolLM3 APO/Tulu-3 preference source and synthetic Qwen-pair separation

## Phase 1 Exit Status

Current status: not ready to exit Stage 1 / Phase 1.

Exit criteria still unmet:

- Primary OLMo corpora are partly retained, sampled, and manifest-backed.
  Dolma 3 and Dolci SFT/DPO now have retained sample/census or package
  artifacts; Dolci RL remains unresolved if kept in scope.
- SmolLM3 replication sources are not sampled and still have source-identification blockers.
- Core Tier-1 matchers do not yet have real 200-hit precision validation.
- Retained Dolci SFT/DPO census, pair-analysis, JSONL package, and per-record
  Parquet/CSV/JSON/Markdown manifest artifacts now exist for a 10k sample and
  are logged to W&B. Formal combined census-summary files are still missing.
- Pair IDs, chosen/rejected alignment, and metadata-aware grouping have
  retained-run evidence for Dolci DPO, but chosen/rejected construction
  semantics are still not sufficient for final Phase 1 claims without the
  remaining validation/package work.
- W&B now has retained Dolci SFT/DPO and Dolma Stage 1 records plus a
  SmolLM3/Tulu source-identification probe, but SmolLM3/Tulu config-specific
  schema probes, retained sampling, precision scoring, and formal package
  summaries are still needed.

Do not proceed to statistical claims until:

- `source_card_notes.md`, retained corpus samples, precision labels, precision report, census summary, and census parquet outputs exist.
- Chosen/rejected role semantics are verified.
- Pair alignment is checked on the retained sample.
- Length covariates are included in retained Result A analysis, given the observed 32-pair dry-run length imbalance.
