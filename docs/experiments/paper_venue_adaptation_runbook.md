# Paper Venue Adaptation Runbook

Date: 2026-06-18

Purpose: define the concrete steps for converting the current venue-neutral
paper package into a target-venue submission package. This runbook does not
choose a venue and does not mark the package submission-ready. It makes the
remaining venue blocker executable after a target template is selected.

Primary decision record:

- `docs/experiments/paper_venue_decision_checklist.md`
- `artifacts/paper/submission/paper_venue_decision_checklist.csv`
- `docs/experiments/paper_venue_sizing_inventory.md`

Current status: venue decisions are `decision_pending`. The package remains
submission-blocked until those rows are resolved and the submission-exit audit
reports `overall_submission_status=ready`.

## Required Venue Decisions

Record these before final formatting:

| Decision | Needed value |
|---|---|
| Target venue | Conference/journal/workshop and exact track or template. |
| Manuscript template | Class file or style, anonymity mode, page limit, supplement policy. |
| Figure size | Single-column/two-column widths, max height, minimum font size. |
| Figure format | PDF/SVG/PNG/EPS requirements and raster DPI if applicable. |
| Table placement | Which tables are main text, appendix, supplement, or review-only. |
| Appendix policy | Whether appendix tables A1-A3 can ship with the submission. |
| Human-labeling requirement | Whether detector-derived Phase 4 claims require human labels before submission. |

## Figure Adaptation

Inputs:

- `artifacts/paper/figures/figure1_pybiber_register_selected.svg`
- `artifacts/paper/figures/figure2_eqbench_stage_scores.svg`
- `artifacts/paper/figures/figure3_olmo_slop_lexicon_views.svg`
- `artifacts/paper/figures/figure4_cross_ladder_af_scatter.svg`
- `artifacts/paper/figures/figure5_phase4_tier3_raw_af.svg`
- `docs/experiments/paper_figure_readiness_audit.md`
- `docs/experiments/paper_figure_visual_review.md`
- `docs/experiments/paper_venue_sizing_inventory.md`

Procedure after venue selection:

1. Set target width/height for Figure 1-Figure 5 according to the venue
   template.
2. Re-render with `uv run slop-render-paper-figures`.
3. Export any venue-required alternate formats from the rendered SVGs.
4. Re-run `uv run slop-audit-paper-figures`.
5. Recreate visual-review PNG sidecars if figure dimensions or labels changed.
6. Update `docs/experiments/paper_figure_table_manifest.md` if captions,
   figure placement, or figure formats change.

Do not change figure claims while adapting size. If a label must be shortened,
check that the claim matrix still covers the resulting caption.

## Per-Figure Decision Matrix

Use this matrix after a target template is selected. Fill the decision in the
venue-specific branch or submission notes before claiming the figure set is
final.

| Figure | Current role | Likely placement | Required venue decision | Verification after decision |
|---|---|---|---|---|
| Figure 1 | Phase 1 pybiber register shift | Two-column or supplement-width | Final width/height and whether selected-feature labels remain readable. | Re-render, export, inspect labels, rerun figure audit and visual review. |
| Figure 2 | EQ-Bench aggregate score bridge | Two-column | Final width/height, legend placement, and interval visibility. | Re-render, export, inspect interval bars, rerun figure audit and visual review. |
| Figure 3 | OLMo `slop_lexicon` propensity/generation views | Two-column landscape or supplement-width | Whether all three panels stay together or move to supplement. | Re-render, export, inspect panel labels, rerun figure audit and visual review. |
| Figure 4 | OLMo/SmolLM3 cross-ladder AF comparison | One-column if labels fit; otherwise two-column | One-column versus two-column placement and label density. | Re-render, export, inspect point labels, rerun figure audit and visual review. |
| Figure 5 | Phase 4 Tier-3 candidate AF | Two-column | Whether candidate-only status and denominator labels remain visible. | Re-render, export, inspect denominator labels and legend, rerun figure audit and visual review. |

## Venue-Neutral Packing Default

Use this only as a starting point before the target template is selected; it
does not resolve the venue-decision checklist. The default packing assumption
is:

- Main text figures: Figure 1, Figure 2, Figure 3, and Figure 4.
- Appendix or supplement figure: Figure 5, unless the venue allows a fifth
  main-text evidence figure and the Phase 4 candidate-only caveat stays in the
  caption.
- Main text tables: Table 1, Table 2, and Table 4.
- Appendix or supplement tables: Appendix Table A2 if uncertainty intervals
  are useful for reviewers; Appendix Table A1 and Appendix Table A3 remain
  review-control guardrails unless space and venue policy allow them.

If page limits are tight, preserve Figure 1, Table 1, Table 2, and the OLMo
`slop_lexicon` evidence from Figure 3 first. Figure 5 and the appendix guardrail tables should move out of the main text before any caveat column, candidate-only label, or measurement-boundary note is removed.

## Table Adaptation

Inputs:

- `docs/experiments/paper_camera_ready_tables.md`
- `docs/experiments/paper_latex_table_drafts.tex`
- `docs/experiments/paper_table_readiness_audit.md`
- `docs/experiments/paper_table_typography_review.md`

Procedure after venue selection:

1. Decide whether Table 1, Table 2, and Table 4 remain main-text tables.
2. Decide whether appendix tables A1-A3 are submitted, moved to supplement, or
   retained only as review-control material.
3. Adapt `paper_latex_table_drafts.tex` to the venue template while preserving
   captions, labels, and table notes.
4. Re-run `uv run slop-audit-paper-tables`.
5. Re-run `uv run slop-audit-paper-table-typography`.

Do not remove caveat columns or notes only to save space. If a caveat must be
shortened, move the full caveat to the main text or appendix.

## Per-Table Decision Matrix

Use this matrix after the target template and appendix/supplement policy are
known.

| Table | Current role | Likely placement | Required venue decision | Verification after decision |
|---|---|---|---|---|
| Table 1 | Data and measurement scope | Main text | Main-text width, whether scope/caveat columns fit. | Rerun table readiness and typography checks. |
| Table 2 | Selected full-pybiber register means | Main text | Main-text width and whether uncertainty/caveat notes stay attached. | Rerun table readiness and typography checks. |
| Table 4 | Tier-1 precision-validation status | Main text | Main-text width and whether pass/demote/derived labels stay visible. | Rerun table readiness and typography checks. |
| Appendix Table A1 | Claim strength bands | Appendix, supplement, or review-control only | Whether claim-band table ships with the submission. | Rerun table readiness and typography checks if included. |
| Appendix Table A2 | Selected pybiber bootstrap intervals | Appendix or supplement | Whether selected uncertainty intervals ship with the submission. | Rerun table readiness and typography checks if included. |
| Appendix Table A3 | Paper-safe negative claims | Appendix, supplement, or review-control only | Whether negative-claim guardrails ship with the submission. | Rerun table readiness and typography checks if included. |

## Manuscript Adaptation

Inputs:

- `docs/experiments/paper_manuscript_draft.md`
- `docs/experiments/paper_methods_draft.md`
- `docs/experiments/paper_results_draft.md`
- `docs/experiments/paper_intro_discussion_draft.md`
- `docs/experiments/paper_claim_matrix.md`

Procedure after venue selection:

1. Apply the target template, anonymity policy, and page limit.
2. Keep the main claim order: pybiber register shift, EQ-Bench aggregate
   bridge, OLMo `slop_lexicon`, SmolLM3 cross-ladder pressure, and Phase 4
   detector candidates.
3. Preserve the limitations that keep generated-output full pybiber,
   EQ-Bench, sparse Tier-3 AF, and detector-candidate claims bounded.
4. Re-run the manuscript audits:

```bash
uv run slop-audit-paper-claim-language
uv run slop-audit-paper-claim-evidence
uv run slop-audit-paper-abstract-conclusion
uv run slop-audit-paper-manuscript-structure
uv run slop-audit-paper-limitations --root /home/user/slop
uv run slop-audit-paper-terminology --root /home/user/slop
uv run slop-audit-paper-numeric-claims --root /home/user/slop
```

## Human-Labeling Decision

If the target venue or paper framing requires detector-discovered features to
be called human-perceived slop, complete annotation before submission:

1. Read `docs/experiments/phase4_human_annotation_codebook.md`.
2. For the pilot, label the blinded sheet
   `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each.csv`,
   keep the map separate until adjudication, then import labels with
   `uv run slop-import-phase4-blind-labels`.
3. If two annotators label the blind pilot, run
   `uv run slop-summarize-phase4-label-agreement` before adjudicating
   disagreement rows.
4. Apply adjudicated pilot labels with
   `uv run slop-apply-phase4-label-adjudication`, then summarize the resulting
   canonical JSONL with `uv run slop-summarize-phase4-human-labels`.
5. For full-package labeling, distribute the blinded full sheet
   `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full.csv`,
   keep
   `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full_map.csv`
   separate until adjudication, then use the same import/agreement/adjudication
   commands with the full-package paths.
6. Summarize labels with `uv run slop-summarize-phase4-human-labels`.
7. Keep candidate-only language if labels are not collected.

## Final Gate

Materialize a venue-neutral draft bundle before final venue-specific edits:

```bash
uv run slop-materialize-paper-submission-bundle --root /home/user/slop
```

This writes `artifacts/paper/submission/draft_bundle/` with manuscript,
tables, references, SVG/PDF/PNG figure candidates, review controls, and a
bundle manifest. It is a staging bundle, not a venue submission certificate.

After venue adaptation and any human-labeling decision, run:

```bash
uv run slop-audit-paper-submission-exit --root /home/user/slop
uv run slop-materialize-paper-submission-bundle --root /home/user/slop
uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop
uv run slop-check-paper-package --root /home/user/slop
```

Submission readiness requires:

- `paper_venue_decision_checklist.csv` rows no longer left as unresolved
  placeholders in the final submission branch.
- `paper_submission_exit_audit.csv` reports `overall_submission_status=ready`.
- `artifacts/paper/submission/draft_bundle/MANIFEST.csv` has been refreshed
  after the final venue-specific manuscript, figure, table, and reference
  edits.
- `slop-check-paper-package` passes after all figure, table, manuscript,
  citation, and manifest changes.
