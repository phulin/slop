# Dolma 3 Tiny-Stream Notes

Live check date: 2026-06-12.

## Task Plan

- [x] Inspect existing corpus extraction and schema-test patterns.
- [x] Try tiny streaming reads for `allenai/dolma3_mix-6T-1025-7B` first.
- [x] Try tiny streaming reads for `allenai/dolma3_mix-6T` second.
- [x] Consume at most 2 rows total per dataset and avoid full downloads.
- [x] Record command attempts, visible fields, row shapes, metadata examples, and blockers.
- [x] Add offline mock tests for representative Dolma 3 row shapes.
- [x] Run the focused Dolma 3 schema tests with `.venv/bin/uv run`.

## Environment

- Workspace: `/home/user/slop`
- Python/test runner: `.venv/bin/uv run`
- Access mode: Hugging Face `datasets.load_dataset(..., split="train", streaming=True)`
- Safety limit: `itertools.islice(ds, 2)` per dataset; no full dataset download requested.

## Command Attempts

### Tiny streaming shape probe

Command:

```bash
.venv/bin/uv run python - <<'PY'
from itertools import islice
from datasets import load_dataset

DATASETS = [
    "allenai/dolma3_mix-6T-1025-7B",
    "allenai/dolma3_mix-6T",
]

def shape(v):
    if isinstance(v, str):
        return {"type": "str", "chars": len(v), "preview": v[:80].replace("\n", " ")}
    if isinstance(v, dict):
        return {"type": "dict", "keys": list(v)[:12], "size": len(v)}
    if isinstance(v, list):
        return {"type": "list", "len": len(v), "first": shape(v[0]) if v else None}
    return {"type": type(v).__name__, "value": v if v is None or isinstance(v, (int, float, bool)) else repr(v)[:80]}

for name in DATASETS:
    print(f"=== {name} ===")
    ds = load_dataset(name, split="train", streaming=True)
    print("features:", getattr(ds, "features", None))
    for i, row in enumerate(islice(ds, 2)):
        print("row", i)
        print("columns:", list(row.keys()))
        for key in row:
            print(key, shape(row[key]))
PY
```

Result: succeeded for both datasets. Hugging Face emitted an unauthenticated-request warning, but no rate-limit error or access block occurred.

Warning observed:

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

## `allenai/dolma3_mix-6T-1025-7B`

- Access: public streaming succeeded unauthenticated.
- Rows consumed: 2.
- Visible features:
  - `id: string`
  - `text: string`
  - `metadata: string`
  - `source: string`
  - `version: string`
  - `created: string`
  - `added: string`
  - `doc: string`
  - `attributes: string`
- Row columns matched the feature list exactly.
- `id`: UUID-like string, 36 characters in both sampled rows.
- `text`: plain document text. Sampled lengths were 5,667 chars and 12 chars. Long text was not retained in these notes.
- `metadata`: JSON-encoded string. Sampled lengths were 1,512 and 1,873 chars.
- `metadata` high-level shape: included scalar and structured quality/provenance keys such as `cc_dump`, `dolma2_qc`, and duplicate/quality metadata.
- `source`: first sampled row was `None`; second sampled row was a short source string, `dclm-hero-run-fasttext_for_HF`.
- `version`: first sampled row was `None`; second sampled row was `1.0`.
- `created` and `added`: first sampled row was `None`; second sampled row used empty strings.
- `doc`: `None` in both sampled rows.
- `attributes`: `None` in both sampled rows.

## `allenai/dolma3_mix-6T`

- Access: public streaming succeeded unauthenticated.
- Rows consumed: 2.
- Visible features:
  - `id: string`
  - `text: string`
  - `metadata: string`
  - `source: string`
  - `version: string`
  - `created: string`
  - `added: string`
  - `doc: string`
  - `attributes: string`
- Row columns matched the feature list exactly.
- `id`: UUID-like string, 36 characters in both sampled rows.
- `text`: plain document text. Sampled lengths were 3,744 chars and 1 char. Long text was not retained in these notes.
- `metadata`: JSON-encoded string. Sampled lengths were 1,363 and 1,487 chars.
- `metadata` high-level shape: included scalar and structured Common Crawl quality/provenance keys such as `cc_dump`, `dolma2_qc`, and duplicate/quality metadata.
- `source`, `version`, `created`, `added`, `doc`, and `attributes`: all `None` in both sampled rows.

## Normalization Implications

- The existing corpus extractor can read Dolma 3 `text` directly with the default `text_fields`.
- `id` is covered by the default ID fields and becomes `record_id`.
- Top-level scalar provenance fields `source`, `version`, `created`, and `added` are promoted into record metadata.
- JSON strings in `metadata`, `doc`, and `attributes` are parsed, and scalar first-level keys are preserved as `metadata.<key>`, `doc.<key>`, and `attributes.<key>`.
- Nested JSON values are intentionally not flattened by the current helper, so keys such as `metadata.dolma2_qc` are not promoted unless a later change broadens flattening.

## Blockers / Notes

- No dataset access blocker occurred for the two requested candidates.
- The only warning was unauthenticated Hugging Face access, which may matter for larger or repeated probes.
- Sampled `source`, `doc`, and `attributes` values were often `None`; mock tests include non-null JSON-string variants for preservation coverage because the visible schema allows them and the extractor already supports them.
