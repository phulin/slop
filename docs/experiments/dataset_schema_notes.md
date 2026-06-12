# Dataset Schema Notes

Live check date: 2026-06-12.

## Task Plan

- [x] Probe Hugging Face dataset availability for primary OLMo first-stage sources.
- [x] Record canonical dataset IDs, configs, splits, and visible schema fields.
- [x] Avoid large downloads; use repository metadata and dataset-card metadata only.
- [x] Re-run with `.venv/bin/uv run python` plus `datasets` once available locally.
- [x] Confirm row-level normalization on tiny streamed samples before Stage 1 census runs.
- [x] Add mock schema tests for representative SFT, DPO, and RL row shapes.
- [x] Confirm current DPO extraction uses assistant-only chosen/rejected text, not full chat transcripts.

## Environment Notes

- Prior metadata was collected with Python standard-library HTTPS calls to Hugging Face API and raw README endpoints. No data shards were downloaded.
- On 2026-06-12, `.venv/bin/uv run python` was available with `datasets` installed.
- Live streaming command used `datasets.load_dataset(..., split="train", streaming=True)` and consumed at most 2 rows from each Dolci dataset. No full dataset download was requested.

## OLMo SFT/Preference/RL Data

### `allenai/Dolci-Instruct-SFT-7B`

- Availability: public, not gated, not disabled.
- Canonical API ID: `allenai/Dolci-Instruct-SFT`.
- Configs/splits: `default`, `train`, `data/train-*`.
- Size in card metadata: 2,152,112 train examples; 15 parquet shards.
- Visible columns/features:
  - `id: string`
  - `messages: list[{content: string, function_calls: string, functions: string, role: string}]`
  - `source_dataset: string`
  - `domain: string`
- Notes: good SFT target candidate; row text normalization should read chat `messages` rather than a flat `text` field.
- Live sample: each streamed row had `id`, `messages`, `source_dataset`, and `domain`; `messages` was a 2-message user/assistant chat.
- Current normalizer: default `iter_corpus_records` reads `messages`; `text` uses assistant-role content, `prompt` uses user/system-role content, and `id` is preserved as `record_id`.

### `allenai/Dolci-Instruct-DPO-7B`

- Availability: public, not gated, not disabled.
- Canonical API ID: `allenai/Dolci-Instruct-DPO`.
- Configs/splits: `default`, `train`, `data/train-*`.
- Size in card metadata: 259,922 train examples; 4 parquet shards.
- Visible top-level columns/features:
  - `chosen: list[message]`
  - `rejected: list[message]`
  - `chosen_model: string`
  - `rejected_model: string`
  - `prompt_id: string`
  - `preference_type: string`
- Visible message fields include `content`, `role`, `language`, `country`, `hashed_ip`, `header.accept-language`, `header.user-agent`, `redacted`, `state`, `toxic`, and `turn_identifier`, plus several nullable OpenAI/chat metadata fields.
- Notes: good preference-pair candidate. `prompt_id` appears to be the stable pair key to preserve for chosen-vs-rejected deltas.
- Live sample: `preference_type` was `delta_learning`; `chosen_model` was `qwen3-no_reasoning-32b`; `rejected_model` was `qwen3-no_reasoning-0.6b`; `prompt_id` was present and stable-looking.
- Current normalizer: default `chosen`/`rejected` extraction selects assistant-role content from each chat transcript, excluding the shared user prompt. With `id_fields=("prompt_id",)`, `record_id` becomes the pair key. `text` defaults to the assistant-only `chosen` response when no flat `text` field exists.
- Implication: current normalization is suitable for response-only chosen-vs-rejected feature rates if callers pass `id_fields=("prompt_id",)`.

### `allenai/Dolci-Instruct-RL-7B`

- Availability: public, not gated, not disabled.
- Canonical API ID: `allenai/Dolci-Instruct-RL`.
- Configs/splits: `default`, `train`, `data/train-*`.
- Size in card metadata: 169,964 train examples; 3 parquet shards.
- Visible columns/features:
  - Prompt/task fields: `prompt: string`, `source_prompt: list[{content: string, role: string}]`, `input_ids_prompt: list[int64]`
  - Labels/answers: `ground_truth: list[string]`, `solution: string`, `outputs: list[string]`, `predicted_label: string`
  - Provenance/task metadata: `id`, `dataset`, `dataset_source`, `data_source`, `original_dataset`, `setting_key`, `setting_name`, `ability`, `topic`, `constraint_type`, `constraint`, `custom_id`, `key`
  - Difficulty/rollout metadata: `difficulty`, `difficulty_explanation`, `total_rollouts`, `total_correct_rollouts`, `passrate`
  - Structured fields: `reward_model.{ground_truth, style}`, `extra_info.index`
- Notes: this is not a chosen/rejected pair table. For Stage 1 census, use `prompt`, `source_prompt`, `solution`, or `outputs` deliberately depending on the measurement question.
- Live sample: first rows had code tasks with `prompt`, `solution`, `ground_truth`, `dataset`, `dataset_source`, `input_ids_prompt`, and many nullable rollout/metadata fields.
- Current normalizer: default extraction selects `solution` as `text` before falling back to prompt-like fields; `prompt` remains available separately. If measuring generated `outputs`, callers must pass an explicit field selection or downstream expansion policy.

## Dolma 3 Discovery

Hugging Face search for AllenAI Dolma 3 returned an active family. Obvious first-stage candidates:

- `allenai/dolma3_pool`: README says it contains the full pool of documents considered to train the first stage of OLMo 3 7B. Public, not gated. Config `default`, split `train`, path `data/**/*`. Card metadata did not expose `dataset_info` features. Repository has about 99,587 siblings, mostly `jsonl.zst` shards.
- `allenai/dolma3_mix-6T-1025-7B`: README says it is the full mix of documents used to train OLMo 3 7B, with a note that some olmOCR science PDF texts are now redacted as `[REMOVED]`. Public, not gated. Config `default`, split `train`, path `data/**/*.jsonl.zst`. Visible configured features: `id`, `text`, `metadata`, `source`, `version`, `created`, `added`, `doc`, `attributes`, all strings. Repository has about 99,834 siblings.
- `allenai/dolma3_mix-6T`: README for the 7B mix recommends this as the primary Dolma 3 data mix for training from scratch and notes it uses the same sampling strategy with complete olmOCR science PDFs. Public, not gated. Config `default`, split `train`, path `data/**/*.jsonl.zst`. Visible configured features match `dolma3_mix-6T-1025-7B`: `id`, `text`, `metadata`, `source`, `version`, `created`, `added`, `doc`, `attributes`.
- Related pool components discovered: `allenai/dolma3_dolmino_pool` and `allenai/dolma3_longmino_pool`. The dolmino pool exposes many named configs, including `stem_heavy_crawl`, `stack_edu_fim`, `cranecode`, `cranemath`, `megamatt`, `dolmino1_math`, `tulu_3_sft`, reasoning-trace configs, `olmocr_science_pdfs`, and `common_crawl_hq`.

## Blockers / Follow-Up

- No source bug found in current Dolci chat normalization. The current helper extracts assistant-only response text for SFT and DPO chat rows.
- Dolma 3 pool card metadata exposes configs but not feature schemas; schema should be verified with `datasets` streaming or a tiny remote shard sample before implementing final normalizers.
- Do not assume the user-facing `-7B` Dolci names are the canonical Hugging Face IDs; API canonicalizes them to IDs without `-7B`.
