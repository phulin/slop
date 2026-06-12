# Phase 1 Census CLI

`slop-census` measures Tier-1 feature counts over sampled corpus records and
writes aggregate rate rows to CSV. Plain text records keep the existing behavior:
the CLI measures `record.text` once and emits aggregate rows with `role=text`.

Preference records are expanded when both `chosen` and `rejected` text are
present. The CLI measures each response separately and aggregates rows by
`source`, `subset`, and `role`, where `role` is `chosen` or `rejected`. This
keeps the chosen/rejected denominators separate for paired analysis while
leaving non-preference corpora on the `text` role.

Use `--pair-output PATH` to also write a per-pair preference delta CSV. The
aggregate `--output` CSV is unchanged. Pair rows are emitted only for records
with both `chosen` and `rejected` text and include:

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
