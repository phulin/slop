# W&B Conventions

Stage 1 uses W&B for aggregate experiment tracking only. Do not log raw secrets, raw `.env` contents, or unredacted large text samples.

## Authentication

Load the API key from `.env` at runtime:

```bash
set -a
source .env
set +a
wandb login "$WANDB_API_KEY"
```

The `.env` file is gitignored. Config files should refer to the variable name `WANDB_API_KEY`, never the value.

## Default Run Metadata

- Entity: set locally with `WANDB_ENTITY` if needed.
- Project: `slop-stage1`
- Groups: `corpus-sampling`, `matcher-port`, `precision-validation`, `phase1-census`
- Job types: `sample`, `validate`, `census`, `smoke`
- Required tags: `stage1`, ladder name, corpus name, and `dry_run` or `full_run`

## Safe Logging Rules

- Log aggregate metrics, schema manifests, content hashes, and artifact paths.
- Do not log full prompt/response text by default.
- Redacted examples are allowed only when needed for matcher debugging.
- Do not include environment dumps in W&B configs.
- Scrub command logs for values matching `WANDB_API_KEY` before attaching them.

## Metric Names

Use stable slash-separated names:

- `corpus/rows`
- `corpus/tokens`
- `corpus/rows_per_sec`
- `corpus/tokens_per_sec`
- `corpus/peak_rss_gb`
- `corpus/output_size_mb`
- `matcher/hits`
- `matcher/opportunities`
- `matcher/precision`
- `matcher/ambiguous_rate`
- `census/rate_per_1k_tokens`
- `census/rate_per_sentence`
- `census/rate_per_opportunity`
- `preference/pair_count`
- `preference/mean_chosen_minus_rejected`

## Artifact Names

Use deterministic artifact names:

- `stage1-corpus-manifest:{run_id}`
- `stage1-feature-specs:{matcher_version}`
- `stage1-validation-report:{matcher_version}`
- `stage1-census-tables:{run_id}`

Artifacts should contain metadata for source dataset IDs, sample seed, git commit when available, matcher version, and content hashes.
