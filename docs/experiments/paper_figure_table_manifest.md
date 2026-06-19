# Paper Figure And Table Manifest

Date: 2026-06-18

Purpose: define the current paper-facing figure and table set, with captions,
source artifacts, and caveats. The current SVGs are rendered by
`uv run slop-render-paper-figures` with editable text, deterministic metadata,
and a shared visual style. `uv run slop-audit-paper-figures` records SVG
package readiness in `docs/experiments/paper_figure_readiness_audit.md`.
`uv run slop-audit-paper-tables` records Markdown/LaTeX table-package
readiness in `docs/experiments/paper_table_readiness_audit.md`.
`uv run slop-audit-paper-table-typography` records generic booktabs/tabularx
typography review in `docs/experiments/paper_table_typography_review.md`.
Submission-facing caption and reproducibility prose is split out in
`docs/experiments/paper_caption_appendix_draft.md`.

## Figures

| ID | Short title | Draft artifact | Source artifacts | Intended claim | Caveat |
|---|---|---|---|---|---|
| Figure 1 | Phase 1 register shift in pybiber features | `artifacts/paper/figures/figure1_pybiber_register_selected.svg` | `artifacts/stage1/census/phase1_pybiber_register_means.csv`; `artifacts/stage1/census/phase1_pybiber_register_intervals.csv`; `docs/experiments/phase1_pybiber_register_analysis.md` | OLMo/Dolci post-training data shift from narrative/personal prose toward nominalized answer-register exposition. | Corpus-side register result only; selected-feature intervals are document-bootstrap intervals over retained rows; does not measure generated-output full pybiber. |
| Figure 2 | EQ-Bench Slop Score by checkpoint | `artifacts/paper/figures/figure2_eqbench_stage_scores.svg` | `artifacts/phase2/analysis/eqbench_slop/phase1_phase2_eqbench_slop_comparison.csv`; `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv`; EQ-Bench score JSON/CSV outputs under `artifacts/phase2/analysis/eqbench_slop/` | Aggregate public slop-score trajectories differ by ladder and do not replace feature-level propensity. | Bootstrap intervals are over valid generated outputs only; use as diagnostic/benchmark bridge. |
| Figure 3 | OLMo `slop_lexicon` amplification views | `artifacts/paper/figures/figure3_olmo_slop_lexicon_views.svg` | `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv`; OLMo compounding and generation-rate artifacts | OLMo `slop_lexicon` has a DPO-stage teacher-forced propensity peak plus positive free-running compounding. | Headline feature view, not the full amplification spectrum. |
| Figure 4 | Cross-ladder AF comparison | `artifacts/paper/figures/figure4_cross_ladder_af_scatter.svg` | OLMo/SmolLM3 amplification-spectrum comparison artifacts under `artifacts/phase3/analysis/` | OLMo and SmolLM3 show related style pressure but different stage localization. | Shared non-missing AF support is small; claim as suggestive replication pressure. |
| Figure 5 | Phase 4 Tier-3 exact AF with denominators | `artifacts/paper/figures/figure5_phase4_tier3_raw_af.svg` | `docs/experiments/phase4_tier3_teacher_forced_addendum.md`; Phase 4 exact 512 stage grid and primary comparison artifacts | Detector-discovered Tier-3 candidates include real exact sequence-mass propensity shifts. | Human perceptibility labels are absent; sparse denominators make several large AFs provisional. |

## Draft Captions

**Figure 1. Phase 1 register shift in selected pybiber features.** Token-weighted
pybiber means for the retained English-filtered Dolma, Dolci SFT, and Dolci
DPO chosen/rejected samples. The main movement is away from narrative/personal
web prose and toward nominalized, adjectival answer-register exposition. Error
bars show document-bootstrap intervals over retained rows for plotted features.

**Figure 2. EQ-Bench Slop Score by checkpoint.** Aggregate EQ-Bench Slop Score
for bounded OLMo and SmolLM3 generation panels with 95% nonparametric
bootstrap intervals over valid generated outputs. The score is useful for
public benchmark comparison, but its stage path differs from the strongest
feature-level propensity result.

**Figure 3. OLMo `slop_lexicon` amplification views.** Teacher-forced
normalized amplification factor, generated-output rates, and compounding
evidence for the original `slop_lexicon` in the bounded OLMo ladder. The DPO
teacher-forced peak is not explained by measured chosen-response enrichment in
the Dolci preference data.

**Figure 4. Cross-ladder AF comparison.** Matched OLMo and SmolLM3
amplification-factor coordinates where both ladders have non-missing support.
The positive rank correlation is suggestive but should not be treated as a
large-sample universal replication.

**Figure 5. Phase 4 Tier-3 exact AF with denominators.** Exact sequence-mass
teacher-forced amplification factors for detector-discovered candidate
features over the 512-reference OLMo rerun. Denominator labels are part of the
figure because sparse candidates remain provisional.

## Tables

| ID | Short title | Draft location | Source artifacts | Intended claim | Caveat |
|---|---|---|---|---|---|
| Table 1 | Data and measurement scope | `docs/experiments/paper_camera_ready_tables.md`; `docs/experiments/paper_latex_table_drafts.tex`; `docs/experiments/paper_tables.md` | Phase reports, sample manifests, generation plans, detector reports | Defines the bounded data/model/generation/detector scope of the paper. | Must not be described as the abandoned exhaustive generation grid. |
| Table 2 | Selected full-pybiber register means | `docs/experiments/paper_camera_ready_tables.md`; `docs/experiments/paper_latex_table_drafts.tex`; `docs/experiments/paper_tables.md` | `docs/experiments/phase1_pybiber_register_analysis.md`; `phase1_pybiber_register_means.csv`; `phase1_pybiber_register_intervals.csv` | Supports the answer-register shift and non-hedging interpretation. | Selected features only; full 67-feature outputs remain in artifacts; appendix Table A2 gives selected bootstrap intervals. |
| Table 3 | Claim strength bands | `docs/experiments/paper_camera_ready_tables.md`; `docs/experiments/paper_latex_table_drafts.tex`; `docs/experiments/paper_claim_matrix.md` | `docs/experiments/paper_claim_matrix.md` | Separates headline-ready, caveated, and methods/status-only claims. | Appendix/review-control table by default; main text should carry caveats locally. |
| Table 4 | Tier-1 precision-validation status | `docs/experiments/paper_camera_ready_tables.md`; `docs/experiments/paper_latex_table_drafts.tex`; `docs/experiments/precision_validation_status.md` | `artifacts/stage1/validation/precision_validation_status.csv` | Records which regex features are precision-supported, demoted, derived, or unlabeled. | Validation status is a scope-control result, not a headline result. |
| Table 5 | Paper-safe negative claims | `docs/experiments/paper_camera_ready_tables.md`; `docs/experiments/paper_latex_table_drafts.tex`; `docs/experiments/paper_tables.md` | Claim matrix and phase reports | Prevents overclaiming beyond the bounded evidence. | Appendix/review-control table by default; use main text caveats for core claims. |

The table package passes the current hygiene and generic typography reviews:
`artifacts/paper/tables/paper_table_readiness_audit.csv` and
`artifacts/paper/tables/paper_table_typography_review.csv`. These checks cover
required Markdown headings, minimum data rows, LaTeX captions/labels,
local-path leakage, booktabs/tabularx use, table width conventions, table
notes where expected, compact sizing, and absence of vertical rules. They do
not replace final target-venue template adaptation.

## Draft Table Captions

**Table 1. Data and measurement scope.** Bounded corpus, generation, detector,
and teacher-forced rerun sizes used in the study.

**Table 2. Selected full-pybiber register means.** Token-weighted means for
selected pybiber features across Dolma, Dolci SFT, and Dolci DPO
chosen/rejected samples. Appendix Table A2 reports selected document-bootstrap
intervals.

**Table 3. Current claim strength bands.** Paper claim IDs grouped by evidence
strength and required caveat language.

**Table 4. Tier-1 precision-validation status.** Current manual-validation
state for Tier-1 regex features and exploratory addenda; pass, demote,
derived, and unlabeled rows define claim boundaries.

**Table 5. Paper-safe negative claims.** Claims the paper should avoid and the
bounded replacement wording supported by current evidence.

## Manuscript Placement

| Section | Figures | Tables |
|---|---|---|
| Methods: model/data scope | none | Table 1 |
| Methods: feature surface | none | Table 4 |
| Result 1: register shift | Figure 1 | Table 2 |
| Result 2: EQ-Bench bridge | Figure 2 | none |
| Result 3: OLMo amplification | Figure 3 | none |
| Result 4: cross-ladder comparison | Figure 4 | none |
| Result 5: detector discovery | Figure 5 | none |
| Appendix / review package | none | Table 3, Table 5 |
