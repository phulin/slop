# Paper Venue Sizing Inventory

Date: 2026-06-18

Purpose: make the remaining venue-formatting blocker concrete. This inventory
records the current venue-neutral figure dimensions, export candidates, and
table float assumptions. It does not choose a target venue and does not replace
`docs/experiments/paper_venue_adaptation_runbook.md`.

Use the per-figure and per-table decision matrices in
`docs/experiments/paper_venue_adaptation_runbook.md` when converting these
venue-neutral dimensions into target-template dimensions and placements.

## Figure Sizing Snapshot

All five paper figures currently pass SVG readiness and rendered-layout visual
review. The current dimensions are Matplotlib/SVG points. Inches are computed
as points divided by 72.

| Figure | SVG artifact | Current size | Approx. inches | Exports present | Venue adaptation action |
|---|---|---:|---:|---|---|
| Figure 1 | `artifacts/paper/figures/figure1_pybiber_register_selected.svg` | 709.835 x 481.837 pt | 9.86 x 6.69 in | PDF+PNG | Likely two-column or supplement-width; verify grouped labels after resizing. |
| Figure 2 | `artifacts/paper/figures/figure2_eqbench_stage_scores.svg` | 568.361 x 366.637 pt | 7.89 x 5.09 in | PDF+PNG | Likely two-column; preserve bootstrap interval legibility and legend placement. |
| Figure 3 | `artifacts/paper/figures/figure3_olmo_slop_lexicon_views.svg` | 854.394 x 308.702 pt | 11.87 x 4.29 in | PDF+PNG | Wide three-panel figure; likely two-column landscape-width or supplement-width. |
| Figure 4 | `artifacts/paper/figures/figure4_cross_ladder_af_scatter.svg` | 495.857 x 395.798 pt | 6.89 x 5.50 in | PDF+PNG | Candidate for one-column if labels remain legible; otherwise two-column. |
| Figure 5 | `artifacts/paper/figures/figure5_phase4_tier3_raw_af.svg` | 779.724 x 438.637 pt | 10.83 x 6.09 in | PDF+PNG | Likely two-column; preserve denominator labels and external legend. |

## Export Inventory

Each figure has:

- editable SVG in `artifacts/paper/figures/`;
- PDF and PNG candidates in `artifacts/paper/figures/submission_exports/`;
- visual-review PNG sidecar in `artifacts/paper/figures/visual_review_png/`.

The current PDF/PNG candidates are useful staging exports. They should be
re-exported after final venue dimensions are chosen.

## Table Sizing Snapshot

The current LaTeX table drafts use generic booktabs/tabularx conventions and
pass generic typography review. They still need target-template placement.

| Table | Current environment | Current width | Notes present | Venue adaptation action |
|---|---|---|---|---|
| Table 1 | `table*` | `\textwidth` | yes | Main-text scope table; keep caveat columns visible. |
| Table 2 | `table*` | `\textwidth` | yes | Main-text pybiber table; preserve selected-feature intervals/caveats. |
| Table 4 | `table*` | `\textwidth` | yes | Main-text feature-status table; keep pass/demote/derived statuses visible. |
| Appendix Table A1 | `table` | `\linewidth` | no | Appendix/review-control unless venue permits supplement. |
| Appendix Table A2 | `table*` | `\textwidth` | no | Appendix uncertainty support for Figure 1/Table 2. |
| Appendix Table A3 | `table*` | `\textwidth` | no | Appendix negative-claim guardrail. |

## Venue Decision Checklist

Before claiming submission readiness:

1. Choose target venue, template, anonymity mode, page limit, and supplement
   policy.
2. Decide whether Figure 1-Figure 5 are one-column, two-column, or supplement
   figures.
3. Re-render or re-export figures at target dimensions and rerun figure
   readiness plus visual review.
4. Decide whether Table 1, Table 2, and Table 4 stay in main text.
5. Decide whether Appendix Tables A1-A3 ship in the submission, supplement, or
   review package only.
6. Rerun table readiness and typography checks after final template adaptation.

## Default Packing Assumption

Before a venue is selected, use this as the working layout assumption for
copyediting and figure/table prioritization:

| Artifact | Default placement | Rationale |
|---|---|---|
| Figure 1 | Main text | Primary Phase 1 pybiber register result. |
| Figure 2 | Main text | Public aggregate EQ-Bench bridge; useful for readers familiar with slop-score benchmarks. |
| Figure 3 | Main text | Primary OLMo `slop_lexicon` propensity/generation/compounding evidence. |
| Figure 4 | Main text if space permits; otherwise supplement | Cross-ladder pressure is useful but caveated by small shared support. |
| Figure 5 | Appendix or supplement by default | Detector-derived Tier-3 candidates remain candidate-only until human labels exist. |
| Table 1 | Main text | Scope and measurement definitions anchor the bounded claim surface. |
| Table 2 | Main text | Compact pybiber register means for Result 1. |
| Table 4 | Main text | Preserves Tier-1 pass/demote/derived boundaries. |
| Appendix Table A1 | Review-control or supplement | Claim-strength guardrail, not necessary for the main narrative. |
| Appendix Table A2 | Appendix or supplement | Optional uncertainty support for selected pybiber intervals. |
| Appendix Table A3 | Review-control or supplement | Negative-claim guardrail. |

This packing assumption is deliberately conservative. If page limits force a smaller main text, move Figure 5, Figure 4, and appendix guardrail tables before removing caveats from Table 1, Table 2, Table 4, or any figure caption.

Current blocker summary: figure/table assets are draft-ready and internally
consistent, but target-venue dimensions, export format, float placement, and
appendix/supplement placement remain unresolved; the venue adaptation runbook
now records the per-artifact decisions needed to close those rows.
