# Corpora IO Notes

## Task Plan

- [x] Create a lightweight corpus IO package under `src/slop_sftdiv/data/`.
- [x] Support local JSONL streaming with standard-library JSON parsing.
- [x] Support Hugging Face datasets through a lazy optional `datasets` import, using streaming mode by default.
- [x] Add deterministic caps for first-N and stable hash-reservoir sampling.
- [x] Normalize common `text`/`prompt`/`chosen`/`rejected` row shapes.
- [x] Include metadata fields intended for W&B tables.
- [ ] Validate against the target Dolma/Dolci/SmolTalk dataset schemas once access details are pinned down.

## Implemented Interface

`CorpusSource.jsonl(path)` streams local JSONL records from disk. `CorpusSource.hf(name, split=..., streaming=True)` streams Hugging Face datasets when the optional `datasets` package is installed.

`SamplingConfig(cap=N, strategy="first")` is the default low-overhead path for already shuffled sources. `SamplingConfig(cap=N, strategy="hash_reservoir", seed=S, max_scan=M)` gives deterministic stable sampling over the scanned rows with O(N) memory.

`iter_corpus_records(source, sampling)` yields normalized `ExtractedRecord` objects. Each record preserves:

- `text`, `prompt`, `chosen`, and `rejected` fields when present.
- `record_id`, `row_index`, source name/kind, split, and source metadata.
- field-name provenance for extracted columns.
- `wandb_row()` output with text lengths, corpus identifiers, sampling configuration, and a stable hexadecimal sample key.

## Field Extraction Notes

The first pass handles common flat fields (`text`, `prompt`, `chosen`, `rejected`, `response`, `completion`, `instruction`, `question`) and dotted paths supplied by callers. Sequence fields are joined with newlines, and chat-like mappings prefer `content`, `text`, then `value`.

Preference records with `chosen`/`rejected` but no explicit `text` use `chosen` as the measurement text by default while preserving both responses. Downstream paired analyses should consume `chosen` and `rejected` directly instead of relying only on `text`.

## Current Limitations

- Hugging Face support depends on the external `datasets` package and was built as a lazy optional dependency.
- Exact schema mappings for Dolma 3, Dolci, SmolTalk2, and Tulu preference subsets still need dataset-card verification.
- Hash-reservoir sampling is deterministic and bounded in memory, but it must scan the selected input range. Use `max_scan` for very large streaming datasets.
- No stratified sampling API is included yet; source/domain strata should be represented as separate `CorpusSource` instances for the first measurement pass.
