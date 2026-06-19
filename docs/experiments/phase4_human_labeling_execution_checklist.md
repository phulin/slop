# Phase 4 Human Labeling Execution Checklist

Date: 2026-06-18

Purpose: give the coordinator a single execution checklist for collecting
Phase 4 human-perceptibility labels and converting them into paper-safe claim
updates. This checklist does not report completed labels.

Execution status: `ready_for_external_labels`.

Primary handoff directory:

- `artifacts/phase4/modernbert_detector_combined_v2_clean/human_annotation_handoff/`

## Coordinator Rule

Share only `human_annotation_handoff/annotator/` with annotators. Do not share
`coordinator/private_maps/` until independent labels are returned and the
agreement/adjudication step is ready.

Use the full-package annotator-specific CSVs for the submission-minimum plan:

- `annotator/annotator_a/phase4_human_perceptibility_blind_full_annotator_a.csv`
- `annotator/annotator_b/phase4_human_perceptibility_blind_full_annotator_b.csv`

Use the pilot-specific CSVs only for a dry run or pilot report:

- `annotator/annotator_a/phase4_human_perceptibility_blind_pilot_20_each_annotator_a.csv`
- `annotator/annotator_b/phase4_human_perceptibility_blind_pilot_20_each_annotator_b.csv`

## Label Collection Steps

1. Give each annotator their assigned CSV and
   `annotator/phase4_human_annotation_codebook.md`.
2. Confirm annotators edit only `human_perceived_slop`,
   `shaib_taxonomy_label`, and `notes`.
3. Keep the map files private until both independent sheets are returned.
4. Run agreement before any discussion or adjudication.
5. Fill adjudication columns only in disagreement rows.
6. Apply adjudication to produce canonical JSONL.
7. Summarize the canonical labels into the paper-facing summary CSV/Markdown.
8. Update the claim matrix, manuscript, Figure 5 caption, and limitations only
   after the summary exists.

## Full-Package Command Template

```bash
uv run slop-summarize-phase4-label-agreement \
  --annotator-a artifacts/phase4/modernbert_detector_combined_v2_clean/human_annotation_handoff/annotator/annotator_a/phase4_human_perceptibility_blind_full_annotator_a.csv \
  --annotator-b artifacts/phase4/modernbert_detector_combined_v2_clean/human_annotation_handoff/annotator/annotator_b/phase4_human_perceptibility_blind_full_annotator_b.csv \
  --blind-map artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full_map.csv \
  --output artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_full_agreement.csv \
  --disagreements-output artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_full_disagreements.csv \
  --summary-output docs/experiments/phase4_human_perceptibility_full_agreement.md \
  --strict
```

After adjudicating disagreement rows:

```bash
uv run slop-apply-phase4-label-adjudication \
  --annotator-a artifacts/phase4/modernbert_detector_combined_v2_clean/human_annotation_handoff/annotator/annotator_a/phase4_human_perceptibility_blind_full_annotator_a.csv \
  --annotator-b artifacts/phase4/modernbert_detector_combined_v2_clean/human_annotation_handoff/annotator/annotator_b/phase4_human_perceptibility_blind_full_annotator_b.csv \
  --blind-map artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full_map.csv \
  --adjudications artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_full_disagreements.csv \
  --output artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full_adjudicated.jsonl \
  --strict
```

Then summarize:

```bash
uv run slop-summarize-phase4-human-labels \
  --input artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full_adjudicated.jsonl \
  --output artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_summary.csv \
  --summary-output docs/experiments/phase4_human_perceptibility_summary.md
```

## Paper Update Rules

Keep detector claims machine-detectable only until the label summary contains
real labeled rows. After labels exist:

- Promote a candidate only if it meets the protocol thresholds.
- Keep candidates below threshold as detector-only or domain/context artifacts.
- Update C11/C12/C13 in `paper_claim_matrix.md` before changing manuscript
  claims.
- Re-run claim-language, claim-evidence, numeric-claims, limitations,
  submission-exit, reproducibility-manifest, and package checks.
- Preserve the unlabeled/candidate-only caveat for any candidate whose
  adjudicated summary remains below threshold or sparse.
