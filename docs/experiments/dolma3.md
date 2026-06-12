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

## Retained Sample Plan

Status: completed for a bounded 20k-row scan retained sample; larger or more
balanced strata remain optional follow-up, not a blocker for the current Stage
1 sampled-corpus proof.

- Target dataset: use `allenai/dolma3_mix-6T-1025-7B` for the OLMo 3
  pretrain retained sample. This matches the successful tiny census and the
  gap audit recommendation; `configs/stage1/corpus_samples.yaml` still lists
  `allenai/dolma3` as a dataset-card verification placeholder.
- Sampling: deterministic seed `1729`, streaming read, bounded scan/sample
  caps, and bounded writer buffers. Do a small authenticated/cache-aware
  throughput calibration before any retained remote-streaming pass because the
  two-document smoke run saw an HF read-timeout retry and took about 51
  seconds.
- Stratification: aim to populate at least three non-code strata from
  `web_cc`, `forums_qa`, `wiki`, and `scientific_olmocr`, with code excluded.
  If the streamed prefix is skewed, keep the bounded run but record the skew,
  inferred stratum counts, scan cap, and whether the Stage 1 go/no-go remains
  blocked.
- Normalized rows: write retained JSONL with the Stage 1 schema fields where
  available, including `corpus`, `ladder`, `source_dataset`, `split`,
  `stratum`, `provenance`, `doc_id`, `text_role`, `text`, token/sentence/
  paragraph counts, and `content_hash`.
- Manifests: emit local CSV, JSON, and Markdown manifests for retained outputs,
  including source dataset, seed, scan cap, sample cap, stratum counts, content
  hashes, row counts, byte sizes, and known blockers. Include W&B run and
  artifact references in the manifests.
- W&B: log aggregate metrics and artifact references only under project
  `slop-stage1`, group `corpus-sampling`; do not log raw full text by default.

## Retained 20k-Scan Sample

Command shape:

```bash
.venv/bin/uv run slop-sample-corpus \
  --input allenai/dolma3_mix-6T-1025-7B \
  --split train \
  --seed 1729 \
  --sampling-strategy hash_reservoir \
  --target-rows-per-stratum 1000 \
  --max-scan 20000 \
  --exclude-stratum code \
  --corpus dolma3_pretrain_sample \
  --ladder olmo3 \
  --source-dataset allenai/dolma3_mix-6T-1025-7B \
  --text-role pretrain_document
```

- W&B sample run:
  `stage1-olmo3-dolma3-20k-scan-retained-sample`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/l0ms0k7t`.
- Local retained JSONL:
  `artifacts/stage1/corpora/olmo3_dolma3_20k_scan_retained_sample.jsonl`.
- Formal per-record manifest:
  `artifacts/stage1/corpora/sample_manifest.parquet`.
- Sidecar manifests:
  `artifacts/stage1/corpora/olmo3_dolma3_20k_scan_sample_manifest.csv`,
  `.json`, and `.md`.
- Scan metrics: 20,000 streamed rows, 1,401 retained rows, 2,129,938 simple
  tokens, 47.61 seconds, about 44,741 tokens/sec, and 1,209 MB peak RSS.
- Retained strata:
  - `web_cc`: 1,000 docs, 1,702,321 tokens
  - `forums_qa`: 279 docs, 236,029 tokens
  - `wiki`: 77 docs, 155,497 tokens
  - `scientific`: 45 docs, 36,091 tokens
- The 20k streamed prefix populated at least three non-code strata, satisfying
  the Stage 1 sampled-Dolma go/no-go condition. The rare strata remain
  underfilled relative to the 1,000-row cap and should be reported as prefix
  skew, not as evidence of source prevalence in the full Dolma 3 mix.

## Historical Retained 20k-Scan All-Feature Census

- W&B census run:
  `stage1-olmo3-dolma3-20k-scan-stratum-census-retained_sample`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/3my331x7`.
- Local output:
  `artifacts/stage1/census/olmo3_dolma3_20k_scan_feature_rates.csv`.
- Shape: 32 feature-rate rows, grouped by retained stratum and text role.
- Throughput: 1,401 docs, 2,129,938 simple tokens, 16.61 seconds, about
  128,199 tokens/sec, 161 MB peak RSS.
- Highest per-1k-token rates by stratum in the historical all-feature census.
  `punctuation_rhythm` and `list_header_bold_lead_in` are excluded from the
  revised Phase 1 core scope:
  - `web_cc`: `punctuation_rhythm` 13.53, `rule_of_three_approx` 3.85,
    `list_header_bold_lead_in` 0.65
  - `forums_qa`: `punctuation_rhythm` 18.29, `rule_of_three_approx` 3.55,
    `list_header_bold_lead_in` 1.64
  - `wiki`: `punctuation_rhythm` 16.08, `list_header_bold_lead_in` 10.48,
    `rule_of_three_approx` 5.74
  - `scientific`: `punctuation_rhythm` 16.54, `rule_of_three_approx` 3.55,
    `list_header_bold_lead_in` 1.50

## Retained Artifact Manifest

- W&B artifact manifest run:
  `stage1-olmo3-dolma3-20k-scan-artifact-manifest`,
  `https://wandb.ai/phulin-self/slop-stage1/runs/75vyt0mb`.
- Local manifest files:
  `artifacts/stage1/corpora/olmo3_dolma3_20k_scan_artifact_manifest.csv`,
  `.json`, and `.md`.
- Manifest records 4 retained artifacts, 16,446,028 total bytes, and 4,235
  counted records: 1,401 retained JSONL rows, 1,401 Parquet manifest rows,
  1,401 CSV manifest rows, and 32 census rows.
- Artifact hashes:
  - retained JSONL:
    `dc4d14448e4e05dbb643a3f7bc5e85aca1203a8cee0e1351be8d0cc85a98d36c`
  - `sample_manifest.parquet`:
    `0dcc4ba6ff63a99c38d9e196bc8889c861c35ddf1941f617ba92e21a1da52c4c`
  - sample manifest CSV:
    `f5e00e7760ce4a4d939b9faceb90a974c991c8a31cb1081f40ba148e6b49eaa6`
  - feature-rate CSV:
    `9b11498c29c60426d130b3e71091bfdb70b3170948a431208a7c102ed37d3509`

## Blockers / Notes

- No dataset access blocker occurred for the two requested candidates.
- The only warning was unauthenticated Hugging Face access, which may matter for larger or repeated probes.
- Sampled `source`, `doc`, and `attributes` values were often `None`; mock tests include non-null JSON-string variants for preservation coverage because the visible schema allows them and the extractor already supports them.
- A first 5k retained sample run (`hs82zcqp`) exposed that top-level
  `full_CC` needed to be classified as web/CC. The stratum inference was fixed
  and rerun as `0r6dvpwk` before the 20k retained sample above.
