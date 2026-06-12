# Stage 1 Precision Validation

Stage 1 precision validation is the gate between Tier-1 matcher exploration and
Phase 1 corpus claims. It validates sampled matcher hits before aggregate rates
are interpreted.

## Task Plan

- [ ] Generate reproducible hit samples for each Tier-1 feature.
- [ ] Hand-label sampled hits with the shared label schema.
- [ ] Compute per-feature precision and ambiguous rate. Ready: scoring inputs,
  formulas, report fields, and W&B metric/table names are specified below.
- [ ] Demote features that miss the precision or reviewability gates. Planned:
  scoring report emits core-feature gate status and demotion reason fields.
- [ ] Log validation metrics and artifacts to W&B. Planned: scoring step logs
  aggregate metrics, per-feature metrics, and labeled-hit/report tables.
- [ ] Apply go/no-go rules before making Phase 1 claims. Planned: use the
  scoring report's core gate summary before interpreting Phase 1 rates.

## Hit Samples

Generate up to 200 hits per feature from Stage 1 sampled corpora using the
frozen matcher version and `seed: 1729`. Stratify by corpus, source dataset, and
chosen/rejected side where possible. If a feature has fewer than 200 hits, label
all hits and record the observed sample size.

Each sampled row should include:

- `feature_id`
- `matcher_version`
- `corpus`
- `source_dataset`
- `split`
- `stratum`
- `doc_id` or stable sample key
- `span_start` and `span_end` when available
- matched span text plus a short redacted context window
- denominator fields used for the feature rate

Do not include raw secrets, full prompts, or full responses in validation files
unless the corpus artifact is already approved for local raw-text handling.

## Label Schema

Use exactly one label per sampled hit:

- `exact`: the hit matches the intended feature definition.
- `partial`: the hit captures a related pattern but is too broad, too narrow, or
  missing required context.
- `false_positive`: the hit does not match the intended feature definition.
- `ambiguous`: the labeler cannot decide from the sampled context or the feature
  spec is underspecified.

Precision is `exact / total_labeled`. `partial`, `false_positive`, and
`ambiguous` all count against the Stage 1 precision gate unless a feature spec
explicitly declares a narrower valid-use calculation.

## Fast Labeling TUI

Use the terminal UI for fast single-keystroke labeling. It saves after every
label and resumes from the labeled output file if it already exists:

```bash
uv run slop-label-hits \
  --input artifacts/stage1/validation/hit_samples/olmo3_dolci_sft_dpo_10k_hits_for_labeling.csv \
  --output artifacts/stage1/validation/labels/olmo3_dolci_sft_dpo_10k_labels.csv \
  --labeler <initials>
```

Keys:

- `e`: exact
- `p`: partial
- `f`: false positive
- `a`: ambiguous
- `n`: edit note for current row
- `u`: clear current label
- `j`/down: next unlabeled row
- `k`/up: previous unlabeled row
- space/right: next row
- `b`/left: previous row
- `q`: save and quit

To focus one feature at a time, pass `--feature <feature>`; repeat the option
to include multiple features. The TUI preserves the scorer-compatible CSV
schema, including `label`, `labeler`, and `notes`.

## Label Scoring Step

Run the scoring step after label CSVs are complete and before any Phase 1 claim
language is drafted. Command shape:

```bash
uv run slop-score-labels \
  --input artifacts/stage1/validation/hit_samples/labeled_hits.csv \
  --output artifacts/stage1/validation/precision_report.csv \
  --wandb-project slop-stage1 \
  --wandb-group precision-validation \
  --wandb-job-type score-labels
```

The scorer accepts an input label CSV, an output report CSV, threshold options,
and optional W&B project/group/job-type arguments.

Required input label fields:

- `feature`
- `sample_id` or stable row key
- `label`
- `labeler`
- source/corpus fields from `slop-sample-hits`
- `role` and `pair_id` when the source data is preference-paired
- `notes` for ambiguous labels or systematic false-positive patterns

Allowed `label` values are exactly `exact`, `partial`, `false_positive`, and
`ambiguous`. Rows with missing labels or unknown label values fail scoring
unless `--allow-incomplete` is supplied for in-progress labeling audits.

For each feature:

- `total_labeled = exact + partial + false_positive + ambiguous`
- `precision = exact / total_labeled`
- `ambiguous_rate = ambiguous / total_labeled`
- `false_positive_rate = false_positive / total_labeled`
- `partial_rate = partial / total_labeled`

If `total_labeled` is zero, precision and rates are undefined and the feature
fails the reportability gate. Core Tier-1 features pass only when precision is
`>= 0.90`, ambiguous rate is `<= 0.10`, label counts are auditable, and the
feature has a stated denominator/opportunity context. Non-core failures should
be marked exploratory rather than blocking the core Phase 1 go/no-go decision.

The precision report should include one row per feature with:

- `feature`
- `is_core`
- `status` (`pass` or `demote`)
- `labeled`
- `exact`
- `partial`
- `false_positive`
- `ambiguous`
- `precision`
- `ambiguous_rate`
- `target_precision`
- `max_ambiguous_rate`
- `min_labeled`
- `precision_pass`
- `ambiguity_pass`
- `sample_size_pass`

## Precision Target

The target precision is `>= 0.90` for every feature used in Phase 1 claims.
Core Tier-1 features must meet the target before census results are interpreted:
contrastive negation, pooled slop lexicon, list/header onset, stock
openers/closers, and punctuation density.

## Demotion Rules

Demote a feature to exploratory if any condition holds:

- precision is below `0.90` after one matcher refinement pass
- fewer than 50 total hits are available and the feature is not central to a
  pre-registered claim
- ambiguous labels exceed `0.10`
- false positives are systematic and cannot be fixed without changing the
  frozen feature definition
- the opportunity context or denominator cannot be stated clearly

Demoted features may still be reported as exploratory diagnostics, but they
must not support Phase 1 headline claims or go/no-go language.

## W&B Logging

Use project `slop-stage1`, group `precision-validation`, and job type
`score-labels`. Log aggregate metrics and tables only:

- `matcher/features_scored`
- `matcher/features_demoted`
- `matcher/core_features_scored`
- `matcher/core_features_demoted`
- W&B table `precision_report` with one row per feature and the report fields
  listed in the scoring section
- artifact references for hit samples, label CSVs, and the precision report

Artifacts should include matcher version, sample seed, source dataset IDs,
content hashes, and the git commit when available. Do not log unredacted full
text, `.env` contents, API keys, or command environments.

## Go/No-Go Before Phase 1 Claims

Go if all core Tier-1 features reach precision `>= 0.90`, label counts are
auditable, and W&B contains the validation metrics plus artifact references.

No-go for Phase 1 claims if any core feature fails precision, pair-side
sampling cannot be audited for preference data, or the precision report is
missing systematic false-positive notes. If only non-core features fail, proceed
with passing features and mark failed features exploratory.
