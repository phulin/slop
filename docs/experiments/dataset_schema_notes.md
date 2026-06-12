# Dataset Schema Notes

Live check date: 2026-06-12.

## Task Plan

- [x] Probe Hugging Face dataset availability for primary OLMo first-stage sources.
- [x] Record canonical dataset IDs, configs, splits, and visible schema fields.
- [x] Avoid large downloads; use repository metadata and dataset-card metadata only.
- [ ] Re-run with `uv run python` plus `datasets`/`huggingface_hub` once available locally.
- [ ] Confirm row-level normalization on tiny streamed samples before Stage 1 census runs.

## Environment Notes

- `uv` was not installed in this workspace PATH.
- `python` was not installed; `python3` was available at `/usr/bin/python3`.
- `datasets`, `huggingface_hub`, and `requests` were not installed for `python3`.
- Metadata was collected with Python standard-library HTTPS calls to Hugging Face API and raw README endpoints. No data shards were downloaded.

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

## Dolma 3 Discovery

Hugging Face search for AllenAI Dolma 3 returned an active family. Obvious first-stage candidates:

- `allenai/dolma3_pool`: README says it contains the full pool of documents considered to train the first stage of OLMo 3 7B. Public, not gated. Config `default`, split `train`, path `data/**/*`. Card metadata did not expose `dataset_info` features. Repository has about 99,587 siblings, mostly `jsonl.zst` shards.
- `allenai/dolma3_mix-6T-1025-7B`: README says it is the full mix of documents used to train OLMo 3 7B, with a note that some olmOCR science PDF texts are now redacted as `[REMOVED]`. Public, not gated. Config `default`, split `train`, path `data/**/*.jsonl.zst`. Visible configured features: `id`, `text`, `metadata`, `source`, `version`, `created`, `added`, `doc`, `attributes`, all strings. Repository has about 99,834 siblings.
- `allenai/dolma3_mix-6T`: README for the 7B mix recommends this as the primary Dolma 3 data mix for training from scratch and notes it uses the same sampling strategy with complete olmOCR science PDFs. Public, not gated. Config `default`, split `train`, path `data/**/*.jsonl.zst`. Visible configured features match `dolma3_mix-6T-1025-7B`: `id`, `text`, `metadata`, `source`, `version`, `created`, `added`, `doc`, `attributes`.
- Related pool components discovered: `allenai/dolma3_dolmino_pool` and `allenai/dolma3_longmino_pool`. The dolmino pool exposes many named configs, including `stem_heavy_crawl`, `stack_edu_fim`, `cranecode`, `cranemath`, `megamatt`, `dolmino1_math`, `tulu_3_sft`, reasoning-trace configs, `olmocr_science_pdfs`, and `common_crawl_hq`.

## Blockers / Follow-Up

- Could not use `uv run python` because `uv` is absent.
- Could not call `datasets.get_dataset_config_names`, `get_dataset_split_names`, or `get_dataset_infos` because the `datasets` package is absent.
- Dolma 3 pool card metadata exposes configs but not feature schemas; schema should be verified with `datasets` streaming or a tiny remote shard sample before implementing final normalizers.
- Do not assume the user-facing `-7B` Dolci names are the canonical Hugging Face IDs; API canonicalizes them to IDs without `-7B`.
