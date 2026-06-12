# Phase 1 Census CLI

`slop-census` measures Tier-1 feature counts over sampled corpus records and
writes aggregate rate rows to CSV. Plain text records keep the existing behavior:
the CLI measures `record.text` once and emits aggregate rows with `role=text`.

For the revised Phase 1 close-out, run only the scoped Tier-1 features plus
Biber-lite sampled register proxies:

```bash
uv run slop-census \
  --include-biber-lite \
  --feature contrastive_negation \
  --feature rule_of_three_approx \
  --feature slop_lexicon \
  --feature stock_openers \
  --feature stock_closers \
  --feature stock_openers_closers \
  --feature biber_lite_first_person_pronouns \
  --feature biber_lite_second_person_pronouns \
  --feature biber_lite_third_person_pronouns \
  --feature biber_lite_demonstratives \
  --feature biber_lite_possibility_modals \
  --feature biber_lite_necessity_modals \
  --feature biber_lite_prediction_modals \
  --feature biber_lite_hedges \
  --feature biber_lite_amplifiers \
  --feature biber_lite_private_verbs \
  --feature biber_lite_public_verbs \
  --feature biber_lite_suasive_verbs \
  --feature biber_lite_nominalizations \
  --feature biber_lite_that_complements \
  --feature biber_lite_infinitives \
  --feature biber_lite_causal_subordinators \
  --feature biber_lite_conditional_subordinators \
  --feature biber_lite_wh_questions \
  --feature biber_lite_passive_voice_approx \
  ...
```

List/header and punctuation/rhythm matchers remain implemented but are excluded
from revised Phase 1 claims. The Biber layer is `biber_lite`, not the full
66-category pybiber extractor; pybiber is deferred on this CPython 3.14
environment because its spaCy dependency lacks a compatible wheel.

Preference records are expanded when both `chosen` and `rejected` text are
present. Retained expanded package rows with `pair_id` plus
`text_role=chosen/rejected` are joined back into pair deltas by `pair_id`. The
CLI measures each response separately and aggregates rows by `source`, `subset`,
and `role`, where `role` is `chosen` or `rejected`. This keeps the
chosen/rejected denominators separate for paired analysis while leaving
non-preference corpora on the `text` role.

Use `--pair-output PATH` to also write a per-pair preference delta CSV. The
aggregate `--output` CSV is unchanged. Pair rows are emitted for direct paired
records and for retained expanded rows once both `chosen` and `rejected` sides
have been seen. Pair rows include:

`source`, `subset`, `pair_id`, `feature`, `chosen_count`, `rejected_count`,
`chosen_tokens`, `rejected_tokens`, `chosen_rate_per_1k_tokens`,
`rejected_rate_per_1k_tokens`, and `chosen_minus_rejected`.

`chosen_minus_rejected` is the chosen rate per 1k tokens minus the rejected
rate per 1k tokens for that pair and feature. Counts and token denominators are
included separately so downstream analysis can choose raw-count or normalized
deltas.

Pair identity is preserved through `pair_id` when the input row provides it.
The census CLI asks the corpus reader to prefer `pair_id`, `pair.id`, and
`preference_pair_id` as record IDs before falling back to the standard ID
fields or generated stable row IDs. Sampled W&B metadata includes `pair_id` for
expanded preference rows. Aggregate feature-rate rows intentionally remain
corpus-level; per-pair rows are written only when `--pair-output` is supplied.

W&B raw text logging remains off for the census path. Sample tables exclude
`text`, `prompt`, `chosen`, `rejected`, and `raw`; they retain IDs, roles, field
names, character counts, and sampling metadata. Pair deltas contain only IDs,
counts, denominators, and rates, so the CLI logs up to 1,000 pair-delta rows to
W&B as `preference_pair_deltas` and always logs `pair_rows` plus
`pair_rows_logged`. The complete per-pair output remains the local CSV supplied
by `--pair-output`.

Assemble the formal revised Phase 1 parquet artifacts from the scoped CSVs:

```bash
uv run slop-assemble-phase1 \
  --feature-rates artifacts/stage1/census/olmo3_dolci_sft_10k_revised_biber_lite_feature_rates.csv \
  --feature-rates artifacts/stage1/census/olmo3_dolci_dpo_10k_retained_revised_biber_lite_feature_rates.csv \
  --feature-rates artifacts/stage1/census/olmo3_dolma3_20k_scan_revised_biber_lite_feature_rates.csv \
  --pair-deltas artifacts/stage1/census/olmo3_dolci_dpo_10k_retained_revised_biber_lite_pair_deltas.csv \
  --wandb-mode online \
  --wandb-run-name stage1-revised-phase1-assembled-census
```
