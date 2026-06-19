# Paper Figure Rendered-Layout Review

Date: 2026-06-18

Machine-readable review: `artifacts/paper/figures/paper_figure_visual_review.csv`

Review status: `visual_review_passed`.
This review inspects PNG sidecars rendered from the same Matplotlib figure code
as the paper SVGs. It is a rendered-layout review for legibility, overlap, and
data occlusion; it does not replace final venue sizing.

PNG review sidecars:

- `artifacts/paper/figures/visual_review_png/figure1_pybiber_register_selected.png`
- `artifacts/paper/figures/visual_review_png/figure2_eqbench_stage_scores.png`
- `artifacts/paper/figures/visual_review_png/figure3_olmo_slop_lexicon_views.png`
- `artifacts/paper/figures/visual_review_png/figure4_cross_ladder_af_scatter.png`
- `artifacts/paper/figures/visual_review_png/figure5_phase4_tier3_raw_af.png`

| Figure | Status | Finding | Remaining action |
|---|---|---|---|
| Figure 1 | visual_review_passed | Labels, legend, grouped bars, and bootstrap intervals are legible; no visible overlap or data occlusion. | Final venue sizing. |
| Figure 2 | visual_review_passed | Lines, markers, error bars, axes, and legend are legible; no visible overlap or data occlusion. | Final venue sizing. |
| Figure 3 | visual_review_passed | Three-panel layout is readable; panel titles fit and stage labels do not overlap. | Final venue sizing. |
| Figure 4 | visual_review_passed | Scatterplot is readable after replacing repeated per-point labels with one label per feature cluster; no visible label collisions remain. | Final venue sizing. |
| Figure 5 | visual_review_passed | Symlog line plot is readable; legend sits outside the plotting area and does not occlude data. | Final venue sizing. |
