# Paper Reproduction Runbook

Date: 2026-06-18

Purpose: provide the command order for regenerating the paper-facing evidence
bundle from the current repository state. This complements
`docs/experiments/paper_reproducibility_manifest.md`, which records file sizes
and SHA-256 checksums for paper artifacts, evidence files, and release-code
source files.

This runbook is for paper-package refreshes, not for relaunching large
generation or detector-training jobs. Cached Phase 1-4 artifacts are treated
as inputs unless a section explicitly says otherwise.

## Preconditions

Run commands from the repository root:

```bash
cd /home/user/slop
```

Use the project environment:

```bash
uv sync
```

If cached analysis artifacts are absent, do not use this runbook as a
substitute for the phase production runbooks. Restore or regenerate the phase
artifacts first.

## Paper Evidence Refresh

### 1. EQ-Bench intervals and figures

```bash
uv run slop-summarize-eqbench-intervals --bootstrap-samples 500 --seed 1729
uv run slop-render-paper-figures
uv run slop-audit-paper-figures
```

Rendered PNG sidecars for visual review should be refreshed if figure sizes,
labels, or plotting logic changed. The existing visual-review audit remains a
human-inspection artifact; do not mark new rendered layouts as reviewed without
actually inspecting them.

### 2. Tables

```bash
uv run slop-audit-paper-tables
uv run slop-audit-paper-table-typography
```

These commands check generic table package hygiene. Target-venue spacing and
float placement are handled by
`docs/experiments/paper_venue_adaptation_runbook.md`.

### 3. Manuscript and claim audits

```bash
uv run slop-audit-paper-claim-language
uv run slop-audit-paper-claim-evidence
uv run slop-audit-paper-citations
uv run slop-audit-paper-abstract-conclusion
uv run slop-audit-paper-manuscript-structure
uv run slop-audit-paper-limitations --root /home/user/slop
uv run slop-audit-paper-terminology --root /home/user/slop
uv run slop-audit-paper-numeric-claims --root /home/user/slop
uv run slop-audit-paper-review-hygiene --root /home/user/slop
uv run slop-audit-paper-release-inventory --root /home/user/slop
uv run slop-audit-paper-source-provenance --root /home/user/slop
```

If any row moves to `review`, fix the manuscript or evidence map before
continuing. Do not silence a review row by weakening the audit unless the
claim matrix has been deliberately updated.

### 4. Phase 4 human-labeling materials

```bash
uv run slop-audit-phase4-human-package
uv run slop-materialize-phase4-human-handoff --root /home/user/slop
uv run slop-summarize-phase4-human-labels
```

The current package is intentionally unlabeled. The summary should remain
`unlabeled` until real human labels are entered into the blinded pilot or
blinded full-package sheet. The annotation rules are in
`docs/experiments/phase4_human_annotation_codebook.md`. If a blinded sheet
has been filled in, first import it with `uv run slop-import-phase4-blind-labels`
using the matching map file, and then run
`uv run slop-summarize-phase4-human-labels` on the imported canonical JSONL
output. If two annotators labeled the same blinded sheet, run
`uv run slop-summarize-phase4-label-agreement` before adjudication and keep
the resulting disagreement rows with the annotation record. After adjudication,
run `uv run slop-apply-phase4-label-adjudication` and summarize the resulting
canonical JSONL with `uv run slop-summarize-phase4-human-labels`. The
adjudication applier rejects unknown `blind_id` values and stale adjudication
rows for consensus examples.

### 5. Submission and manifest controls

Run submission-exit before materializing the bundle, then run the manifest
after all other generated paper files and the draft bundle are stable. Do not
parallelize bundle materialization and manifest generation; the manifest hashes
the bundle manifest.

```bash
uv run slop-audit-paper-submission-exit --root /home/user/slop
uv run slop-materialize-paper-submission-bundle --root /home/user/slop
uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop
uv run slop-check-paper-package --root /home/user/slop
```

If `slop-check-paper-package` reports manifest hash mismatches, rerun
`slop-materialize-paper-submission-bundle`, then
`slop-audit-paper-reproducibility-manifest`, then the package check after the
files named in the error are stable.

## Expected Terminal State

For the current draft package:

- `slop-check-paper-package` passes.
- `artifacts/paper/submission/draft_bundle/MANIFEST.csv` records 31 ready
  staged files.
- `paper_reproducibility_manifest.csv` has all rows `ready`.
- The reproducibility manifest includes `release_code` rows for the source
  files named in the release inventory.
- `paper_submission_exit_audit.csv` keeps
  `overall_submission_status=blocked` until target-venue decisions and Phase 4
  human perceptibility labels are resolved.
- `phase4_human_perceptibility_summary.csv` remains unlabeled unless real
  human labels have been added.

## What This Does Not Reproduce

This runbook does not relaunch:

- OLMo/SmolLM3 generation panels.
- Teacher-forced propensity grids.
- Full pybiber extraction over Phase 1 samples.
- ModernBERT detector training, attribution, embedding, or clustering.

Those are phase-production tasks with separate cached artifacts and runbooks.
This document refreshes the paper-facing layer built from those cached
outputs.
