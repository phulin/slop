# Phase 1 Census CLI

`slop-census` measures Tier-1 feature counts over sampled corpus records and
writes aggregate rate rows to CSV. Plain text records keep the existing behavior:
the CLI measures `record.text` once and emits aggregate rows with `role=text`.

Preference records are expanded when both `chosen` and `rejected` text are
present. The CLI measures each response separately and aggregates rows by
`source`, `subset`, and `role`, where `role` is `chosen` or `rejected`. This
keeps the chosen/rejected denominators separate for paired analysis while
leaving non-preference corpora on the `text` role.

Pair identity is preserved through `pair_id` when the input row provides it.
The census CLI asks the corpus reader to prefer `pair_id`, `pair.id`, and
`preference_pair_id` as record IDs before falling back to the standard ID
fields or generated stable row IDs. Sampled W&B metadata includes `pair_id` for
expanded preference rows, but aggregate feature-rate rows intentionally remain
corpus-level and do not include one row per pair.

W&B raw text logging remains off for the census path. Sample tables exclude
`text`, `prompt`, `chosen`, `rejected`, and `raw`; they retain IDs, roles, field
names, character counts, and sampling metadata.
