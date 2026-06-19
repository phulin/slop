# Slop Style Paper Scaffold

Date: 2026-06-18

This scaffold converts the current experiment package into a manuscript plan.
Use `docs/experiments/paper_claim_matrix.md` as the authority for allowed claim
language; this file gives the order of the argument and the figure/table set.
The shortest reader-facing orientation is
`docs/experiments/paper_package_index.md`.
The reviewer-facing scope FAQ is `docs/experiments/paper_reviewer_faq.md`.
The current manuscript claim-to-evidence map is
`docs/experiments/paper_claim_evidence_map.md`; the Tier-1 publication-use
policy is `docs/experiments/paper_tier1_publication_policy.md`; the current
submission readiness audit is `docs/experiments/paper_readiness_audit.md`.
The current claim-evidence path-integrity audit is
`docs/experiments/paper_claim_evidence_audit.md`.
The current citation/source integrity audit is
`docs/experiments/paper_citation_audit.md`.
The current integrated-manuscript limitations-boundary audit is
`docs/experiments/paper_limitations_audit.md`.
The current integrated-manuscript terminology/readability audit is
`docs/experiments/paper_terminology_audit.md`.
The current headline numeric-claims audit is
`docs/experiments/paper_numeric_claims_audit.md`.
The current reproducibility manifest is
`docs/experiments/paper_reproducibility_manifest.md`.
The current paper reproduction runbook is
`docs/experiments/paper_reproduction_runbook.md`.
The current manuscript-facing Methods draft is
`docs/experiments/paper_methods_draft.md`; the current Results prose draft is
`docs/experiments/paper_results_draft.md`; the current
Introduction/Discussion/Limitations draft is
`docs/experiments/paper_intro_discussion_draft.md`. The integrated
end-to-end manuscript draft is `docs/experiments/paper_manuscript_draft.md`.
The current figure/table manifest is
`docs/experiments/paper_figure_table_manifest.md`; the current caption and
reproducibility appendix draft is
`docs/experiments/paper_caption_appendix_draft.md`.
The citation-slot plan is `docs/experiments/paper_citation_plan.md`.
Checked reference artifacts are `docs/experiments/paper_reference_sources.md`
and `docs/experiments/paper_references.bib`. Run
`uv run slop-check-paper-package --root /home/user/slop` after paper-doc edits
to catch broken local paths, missing citation keys, stale blocker wording,
C1-C14 claim-map drift, and figure/table reference drift.

## Working Title

Localizing "Slop Style" in Open Post-Training Ladders

## One-Sentence Thesis

Open post-training ladders show that "slop style" is not one monolithic
artifact: alignment data shift register toward polished answer-like exposition,
while individual lexical and rhetorical markers amplify at different stages
depending on the ladder and measurement mode.

## Abstract Skeleton

Large language model outputs often share recognizable stylistic markers, but
raw output counts cannot distinguish inherited training-data style from
model-side amplification. We study this problem in open model/data ladders
where pretraining, supervised fine-tuning, preference data, and checkpoints can
be compared directly. We combine corpus censuses, full pybiber register
measurement, teacher-forced propensity, free-running generation, EQ-Bench Slop
Score diagnostics, and detector-driven feature discovery. In OLMo/Dolci data,
post-training corpora move away from narrative and personal web prose toward
nominalized, adjectival, answer-like exposition; this is not simply an
aggregate increase in hedging. In the OLMo checkpoint ladder, the original
slop lexicon shows the cleanest preference-stage propensity peak, but the
corresponding preference data do not show chosen-response enrichment, pointing
away from a simple imitation account. SmolLM3 no-think shows related style
pressure with different stage localization. Detector attribution identifies
candidate Tier-3 style families, several of which have real teacher-forced
sequence-mass support, but human perceptibility validation remains open. The
result is an amplification spectrum: slop style is feature-specific,
stage-specific, and partly a register shift rather than a single global score.

## Contributions

1. A measurement design that separates corpus inheritance, fixed-context model
   propensity, and free-running self-conditioning for stylistic features.
2. A full pybiber register read over English-filtered OLMo Phase 1 samples,
   showing that alignment data shift toward answer-like exposition rather than
   simply increasing hedges or intensifiers.
3. A bounded OLMo and SmolLM3 amplification spectrum showing that
   `slop_lexicon` amplifies in both ladders but localizes to different stages.
4. An EQ-Bench Slop Score bridge that supports benchmark comparison while
   keeping aggregate scores separate from causal propensity claims.
5. A detector-driven Tier-3 discovery pass that turns attribution clusters
   into provisional features and validates selected candidates with exact
   teacher-forced sequence mass.

## Section Plan

### 1. Introduction

Open with the empirical problem: people recognize repeated LLM stylistic tics,
but the source is ambiguous. A phrase can appear because it is in pretraining
data, because SFT teaches an assistant register, because preference
optimization changes local propensities, or because generation
self-conditions after the first occurrence.

Main framing:

- "Slop style" means measurable stylistic markers, not factual incorrectness.
- The paper is not a universal indictment of DPO or post-training.
- The key object is an amplification spectrum, not one scalar slop score.

End the introduction with the three main findings:

- Data-side alignment register shift: narrative/personal to answer-like
  exposition.
- Feature-specific amplification: OLMo `slop_lexicon` peaks at DPO in
  teacher-forced propensity, while SmolLM3 localizes the main effect earlier.
- Detector-discovered features include real propensity shifts, but validation
  is the remaining boundary between machine-detectable and human-perceived
  slop.

Draft source: `docs/experiments/paper_intro_discussion_draft.md`.

### 2. Related Work

The integrated manuscript now contains a compact first-pass Related Work
section using the verified BibTeX set. Second-pass citation work added
foundational Biber metadata, Shaib taxonomy provenance, APO method background,
verbosity-bias background, calibration background, generation-degeneration
background, and GTE-large provenance. Remaining gaps are the optional
unverified citation slots listed in
`docs/experiments/paper_citation_plan.md`, especially broader AI-writing
detection survey context and teacher-forcing/LM-scoring background if the
final Methods section needs them.

Covered clusters:

- Stylistic artifacts and "AI writing" detection.
- Biber/register analysis and computational register features.
- Preference optimization and open checkpoint/data provenance.
- EQ-Bench Slop Score and public slop lexicon efforts.
- Attribution-driven feature discovery in classifiers.
- SGLang generation-runtime provenance.

### 3. Data And Model Ladders

Describe:

- OLMo 3 base/SFT/DPO/final with Dolma and Dolci.
- SmolLM3 no-think replication ladder and why its preference/final cells are
  not structurally identical to OLMo.
- English filtering and fixed retained samples.
- Bounded generation panels and why they replaced the abandoned exhaustive
  grid.

Table 1 summarizes rows/tokens/generations:

- Dolma retained pretrain: 5,740 docs, 8,689,472 tokens.
- Dolci SFT: 40,000 docs, 8,279,115 tokens.
- Dolci DPO: 40,000 pairs, 80,000 expanded rows, 27,448,016 tokens.
- OLMo reduced generation panels.
- SmolLM3 no-think production panel.
- Phase 4 detector/attribution sample sizes.

### 4. Features And Measurement

Split the feature surface into:

- Tier 1: contrastive negation, slop lexicon, rule-of-three approximation,
  stock openers/closers, EQ-Bench addenda.
- Tier 2: full pybiber register features.
- Tier 3: detector-discovered features.

Then define the three measurement modes:

- Corpus rate: per 1k tokens / per opportunity where defined.
- Teacher-forced propensity: model probability mass on initiating tokens or
  exact candidate sequences in fixed contexts.
- Free-running emission and compounding: realized counts and prior-hit risk.

Important methods caveat:

> EQ-Bench Slop Score is reported as a benchmark bridge and aggregate
> diagnostic; it is not the replacement for feature-level propensity.

Draft source: `docs/experiments/paper_methods_draft.md`.

### 5. Result 1: Alignment Data Shift Register

Claim IDs: C1, C2, C3.

Draft source: `docs/experiments/paper_results_draft.md`.

Core result:

- SFT is much lower than Dolma on past tense, first-person pronouns,
  third-person pronouns, adverbs, emphatics, contractions, and clausal
  coordination.
- SFT is higher on nominalizations, attributive adjectives, present tense,
  prepositions, and broad noun density.
- DPO chosen and rejected differ in register, but chosen is not uniformly
  more slop-like.

Figure 1:

- Pybiber selected-feature horizontal bar plot with document-bootstrap
  intervals:
  Dolma, SFT, DPO chosen, and DPO rejected.
- Draft artifact:
  `artifacts/paper/figures/figure1_pybiber_register_selected.svg`.

Table 2:

- Selected token-weighted pybiber means from
  `docs/experiments/phase1_pybiber_register_analysis.md`.
- Selected-feature interval artifact:
  `artifacts/stage1/census/phase1_pybiber_register_intervals.csv`.

Interpretation:

> Alignment data move toward a polished assistant-answer register. Specific
> slop markers sit on top of that register shift, but the broad register
> shift is not reducible to hedging.

### 6. Result 2: Aggregate Slop Score Is Useful But Insufficient

Claim IDs: C4, C5, C6.

Use EQ-Bench to show the value and limitation of aggregate scoring:

- OLMo score: base 5.179, SFT 7.579, DPO 7.008, final 6.649.
- SmolLM3 no-think score: base 2.803, SFT 7.173, DPO/APO 8.354, final 8.254.
- Component rates are mostly driven by EQ-Bench slop-list word density.

Figure 2:

- Two-line stage plot: EQ-Bench Slop Score by checkpoint for OLMo and
  SmolLM3.
- Draft artifact:
  `artifacts/paper/figures/figure2_eqbench_stage_scores.svg`.

Main interpretation:

> The score captures a real public benchmark notion of slop, but it hides
> mechanism. OLMo and SmolLM3 have different stage paths, and the score is not
> a fixed-context propensity estimate.

### 7. Result 3: Feature-Level Amplification Spectrum

Claim IDs: C7, C8, C9, C10.

Primary OLMo story:

- `slop_lexicon` normalized teacher-forced AF rises base 1.467, SFT 1.695,
  DPO 1.999, final 1.659.
- Free-running long-panel rates peak at DPO but are not identical to the
  teacher-forced pattern.
- Compounding excess is positive at all OLMo stages.
- Dolci chosen/rejected evidence does not show chosen-response enrichment for
  `slop_lexicon`.

Replication story:

- SmolLM3 also shows slop-lexicon pressure, but the main localization is SFT
  rather than OLMo's DPO.
- Cross-ladder AF rank correlation is positive but thin: Spearman 0.762 over
  8 shared non-missing AF values.

Figure 3:

- OLMo amplification spectrum for supported Tier-1 features:
  data rate, teacher-forced AF, free-run rate, compounding excess.
- Current draft focuses on the headline OLMo `slop_lexicon` result:
  `artifacts/paper/figures/figure3_olmo_slop_lexicon_views.svg`.

Figure 4:

- Cross-ladder comparison scatterplot or paired feature-stage chart.
- Draft artifact:
  `artifacts/paper/figures/figure4_cross_ladder_af_scatter.svg`.

Main interpretation:

> The cleanest OLMo `slop_lexicon` signal is dynamics-driven in the narrow
> sense that the model-side DPO peak is not matched by chosen-response
> enrichment. This rules out the simplest data-imitation story but does not
> identify the exact optimization mechanism.

### 8. Result 4: Detector-Discovered Tier-3 Features

Claim IDs: C11, C12, C13.

Describe the pipeline:

- ModernBERT classifier.
- Integrated gradients over 10,000 generated docs.
- GTE/HDBSCAN clustering.
- Matcher specs and candidate census.
- 512-document exact OLMo teacher-forced sequence-mass rerun.

Key supported candidates:

- Process framing: raw AF 1.401 -> 1.957 from base to final.
- Additive transition: 0.565 -> 1.012.
- Prescriptive instruction: 0.301 -> 0.763.
- Follow-up offer: 6.171 -> about 39 after SFT, but only 11 reference
  initiations.

Figure 5:

- Tier-3 candidate AF table/forest plot with denominator annotations.
- Draft artifact:
  `artifacts/paper/figures/figure5_phase4_tier3_raw_af.svg`.

Main interpretation:

> Detector-derived features are not merely free-generation counting artifacts.
> Some correspond to fixed-context propensity shifts. They remain provisional
> until human perceptibility and precision validation are complete.

### 9. Discussion

Key discussion points:

- Slop style is partly a register transition and partly feature-level
  amplification.
- "Preference optimization creates slop" is too broad; evidence points to
  feature- and ladder-specific stage localization.
- Word "dynamics-driven" carefully: it means not explained by the
  measured chosen-vs-rejected data contrast, not that the optimizer mechanism
  is fully identified.
- Aggregate scores are valuable for comparability but do not drive causal
  interpretation.
- Machine-detectable style and human-perceived slop need to be separated.

Draft source: `docs/experiments/paper_intro_discussion_draft.md`.

### 10. Limitations

Required limitations:

- Tier-1 precision validation is scoped rather than incomplete: the current
  status table has 4/5 independent core features passing the interim gate,
  `rule_of_three_approx` demoted, and `stock_openers_closers` treated as a
  derived pooled view.
- Human perceptibility validation for Phase 4 is not complete; the 1,000-row
  annotation package is readiness-audited and a 20-per-candidate pilot sheet
  is prepared.
- Dolma and generation panels are bounded samples.
- SmolLM3 preference/final stage attribution is blurred by APO soup and
  long-context merge.
- Sparse feature denominators make several large AFs provisional.
- Generated-output full pybiber has not been run, so generated-output register
  proxy artifacts are excluded from main-paper claims.
- EQ-Bench aggregate score now has output-bootstrap intervals, but remains a
  descriptive benchmark bridge rather than the causal estimand.

### 11. Conclusion

Close with:

> Slop style is a spectrum of measurable stylistic features
> rather than a single defect. In open post-training ladders, broad register
> shifts can be separated from feature-level propensity amplification and
> self-conditioning. That separation changes the causal story: some markers
> are inherited, some are amplified, and some are artifacts of the assistant
> register itself.

## Figure And Table Inventory

The authoritative figure/table manifest is
`docs/experiments/paper_figure_table_manifest.md`; the table below is a compact
scaffold summary.

| ID | Artifact to produce | Source artifacts | Status |
|---|---|---|---|
| Table 1 | Data/model scope table | Phase reports and sample manifests | Needs final formatting |
| Figure 1 | Pybiber selected-feature register bars with intervals | `phase1_pybiber_register_means.csv`; `phase1_pybiber_register_intervals.csv` | Draft SVG rendered: `artifacts/paper/figures/figure1_pybiber_register_selected.svg` |
| Table 2 | Selected pybiber means | `docs/experiments/phase1_pybiber_register_analysis.md` | Text table exists |
| Figure 2 | EQ-Bench score by stage | `phase2_eqbench_slop_stage_comparison.csv` | Draft SVG rendered: `artifacts/paper/figures/figure2_eqbench_stage_scores.svg` |
| Figure 3 | OLMo `slop_lexicon` amplification views | `olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv` | Draft SVG rendered: `artifacts/paper/figures/figure3_olmo_slop_lexicon_views.svg` |
| Figure 4 | OLMo vs SmolLM3 cross-ladder AF | Phase 3 cross-ladder artifacts | Draft SVG rendered: `artifacts/paper/figures/figure4_cross_ladder_af_scatter.svg` |
| Figure 5 | Tier-3 teacher-forced AF with denominators | Phase 4 exact 512 stage grid/primary comparison | Draft SVG rendered: `artifacts/paper/figures/figure5_phase4_tier3_raw_af.svg` |
| Table 3 | Claim/evidence/caveat matrix | `paper_claim_matrix.md` | Exists as doc, may become appendix |
| Table 4 | Tier-1 precision-validation status | `precision_validation_status.md` and `precision_validation_status.csv` | Exists as status table; pass/demote/derived validation status is itself the current result |
| Table pack | Manuscript-ready draft tables | `paper_tables.md` | Exists as combined provenance-oriented draft |
| Camera-ready table pack | Submission-facing Table 1/Table 2/Table 4 plus appendix tables | `paper_camera_ready_tables.md`; `paper_latex_table_drafts.tex` | Exists as path-free Markdown tables plus generic booktabs LaTeX drafts, including a selected pybiber interval appendix table |
| Figure/table manifest | Captions, source artifacts, caveats, manuscript placement | `paper_figure_table_manifest.md` | Exists as paper-facing manifest |
| Figure readiness audit | SVG size, editable-text, title, metadata, path-hygiene, and PDF/PNG export check | `paper_figure_readiness_audit.md`; `paper_figure_readiness_audit.csv`; `slop-audit-paper-figures` | Exists and marks Figure 1-Figure 5 `ready_for_visual_review` with PDF/PNG export candidates; complements rendered-layout review; remaining action is final venue sizing/export choice |
| Figure rendered-layout review | PNG sidecar review for rendered legibility, overlap, and data occlusion | `paper_figure_visual_review.md`; `paper_figure_visual_review.csv`; `artifacts/paper/figures/visual_review_png/` | Exists and marks Figure 1-Figure 5 `visual_review_passed`; remaining action is final venue sizing |
| Table readiness audit | Markdown heading/row-count and LaTeX caption/label/path-hygiene check | `paper_table_readiness_audit.md`; `paper_table_readiness_audit.csv`; `slop-audit-paper-tables` | Exists and marks Table 1/Table 2/Table 4 plus appendix tables A1-A3 `ready_for_venue_styling`; does not replace final venue typography |
| Table typography review | Generic booktabs/tabularx typography check | `paper_table_typography_review.md`; `paper_table_typography_review.csv`; `slop-audit-paper-table-typography` | Exists and marks Table 1/Table 2/Table 4 plus appendix tables A1-A3 `typography_review_passed`; remaining action is target venue template adaptation |
| Caption/appendix draft | Submission-facing captions and reproducibility artifact index | `paper_caption_appendix_draft.md` | Exists as manuscript appendix material |
| Figure renderer | Cached-artifact renderer for Figure 1-Figure 5 | `slop-render-paper-figures` | Exists; current SVGs render with editable text, deterministic metadata, consistent styling, interval error bars for Figure 1/Figure 2, and PDF/PNG submission-export candidates |
| EQ-Bench interval summarizer | Output-bootstrap interval generator for Figure 2 | `slop-summarize-eqbench-intervals` | Exists; current Figure 2 intervals use 500 bootstrap resamples over valid outputs |
| Phase 4 human-label protocol | Annotation schema, codebook, coordinator execution checklist, blind-label importer, agreement summarizer, adjudication applier, and label summarizer for detector-candidate perceptibility labels | `phase4_human_perceptibility_protocol.md`; `phase4_human_annotation_codebook.md`; `phase4_human_labeling_execution_checklist.md`; `slop-import-phase4-blind-labels`; `slop-summarize-phase4-label-agreement`; `slop-apply-phase4-label-adjudication`; `slop-summarize-phase4-human-labels` | Exists; current generated summary records 0/100 labeled rows for each candidate, while the blinded pilot/full-package packets, importer, agreement/adjudication tools, codebook, and coordinator checklist define the path from annotation to canonical summaries and paper claim updates |
| Claim evidence map | Claim IDs mapped to manuscript sections, figures/tables, evidence artifacts, and caveats | `paper_claim_evidence_map.md` | Exists as paper review trail |
| Claim language audit | Required claim/caveat phrases and forbidden-overclaim checks for the integrated manuscript | `paper_claim_language_audit.md`; `paper_claim_language_audit.csv`; `slop-audit-paper-claim-language` | Exists and marks C1-C14 plus all forbidden-overclaim checks `ready` |
| Claim evidence audit | Claim-to-evidence map coverage and source-path integrity for C1-C14 | `paper_claim_evidence_audit.md`; `paper_claim_evidence_audit.csv`; `slop-audit-paper-claim-evidence` | Exists and marks all C1-C14 evidence-map rows `ready` with existing paths or explicit evidence inheritance |
| Abstract/conclusion audit | Abstract and conclusion bounded-claim check | `paper_abstract_conclusion_audit.md`; `paper_abstract_conclusion_audit.csv`; `slop-audit-paper-abstract-conclusion` | Exists and marks abstract and conclusion alignment `ready` |
| Manuscript structure audit | Integrated-manuscript section order, length bounds, Result 4.x subsection coverage, captions, and appendix presence | `paper_manuscript_structure_audit.md`; `paper_manuscript_structure_audit.csv`; `slop-audit-paper-manuscript-structure` | Exists and marks all nine structure checks `ready` |
| Limitations audit | Integrated-manuscript limitations boundary coverage | `paper_limitations_audit.md`; `paper_limitations_audit.csv`; `slop-audit-paper-limitations` | Exists and marks bounded open-ladder/sample, Tier-1, sparse denominator, DPO, pybiber-generation, EQ-Bench, Phase 4 human-validation, reduced-grid, and forbidden-overclaim checks `ready` |
| Terminology audit | Reader-facing definitions for slop-style scope, measurement layers, amplification spectrum, and interpretation boundaries | `paper_terminology_audit.md`; `paper_terminology_audit.csv`; `slop-audit-paper-terminology` | Exists and marks all eight terminology checks `ready` |
| Numeric claims audit | Source-backed check for headline manuscript numbers | `paper_numeric_claims_audit.md`; `paper_numeric_claims_audit.csv`; `slop-audit-paper-numeric-claims` | Exists and marks 15 manuscript numeric claims `ready` against source CSV/JSON artifacts |
| Source/provenance audit | Source-card and manuscript-facing interpretation boundaries for Dolci DPO, Dolma sampling, full pybiber, and SmolLM3 APO/source claims | `paper_source_provenance_audit.md`; `paper_source_provenance_audit.csv`; `slop-audit-paper-source-provenance` | Exists and marks all five provenance checks `ready`; prevents pure-human-preference, full-Dolma, and generated-output full-pybiber overclaims |
| Tier-1 publication policy | Main-text versus exploratory use rules for undervalidated regex features | `paper_tier1_publication_policy.md` | Exists; separates `slop_lexicon` model-side index use from corpus-rate precision claims |
| Readiness audit | Paper-grade areas, caveated areas, submission blockers, and exit criteria | `paper_readiness_audit.md` | Exists as submission-readiness control |
| Reader glossary | New-reader definitions for AF, pybiber, EQ-Bench, Tier 1/Tier 2/Tier 3, model ladders, and evidence layers | `paper_reader_glossary.md` | Exists as orientation aid; claim wording remains governed by the claim matrix |
| Submission-exit audit | Machine-readable submission exit rows separating paper-draft readiness from venue/human-validation blockers | `paper_submission_exit_audit.md`; `paper_submission_exit_audit.csv`; `slop-audit-paper-submission-exit` | Exists and marks the package `blocked` only on venue review plus Phase 4 human perceptibility labels |
| Phase 4 human annotation handoff | Redacted annotator-facing bundle plus coordinator-only maps and checklist for Phase 4 human labels | `phase4_human_annotation_handoff.md`; `human_annotation_handoff/`; `phase4_human_labeling_execution_checklist.md`; `slop-materialize-phase4-human-handoff` | Exists with 14 ready files, 0 missing files, and 6 redacted annotator CSVs, including an annotator quickstart, coordinator checklist, and annotator-specific pilot/full copies for two independent labelers; labels are still absent |
| Venue-decision checklist and runbook | Explicit unresolved target-venue choices plus post-decision adaptation steps | `paper_venue_decision_checklist.md`; `paper_venue_adaptation_runbook.md`; `paper_venue_decision_checklist.csv` | Exists and marks target venue, figure dimensions/export, table placement, appendix inclusion, manuscript template, and human-labeling requirement `decision_pending`; runbook defines figure/table/manuscript adaptation steps after venue selection |
| Venue-neutral submission draft bundle | One-command staging directory for manuscript, tables, references, figures, and review controls | `artifacts/paper/submission/draft_bundle/README.md`; `artifacts/paper/submission/draft_bundle/MANIFEST.csv`; `slop-materialize-paper-submission-bundle` | Exists with 31 ready files and 0 missing; useful for review and venue adaptation, but not a venue submission certificate |
| Open-release inventory | Release-facing map of feature definitions, opportunity definitions, harness code, runbooks, and cached statistics | `paper_release_inventory.md`; `paper_release_inventory.csv`; `slop-audit-paper-release-inventory` | Exists and marks release inventory rows ready; useful for artifact release planning but not a license review or archival deposit |
| Reproducibility manifest | File-size and SHA-256 index for paper-facing evidence bundle | `paper_reproducibility_manifest.md`; `paper_reproducibility_manifest.csv`; `slop-audit-paper-reproducibility-manifest` | Exists and marks paper controls, manuscript drafts, audits, figure SVGs plus PDF/PNG exports, Phase 1 pybiber evidence, EQ-Bench evidence, and Phase 4 human-labeling materials ready |
| Reproduction runbook | Command order for refreshing paper-facing figures, audits, summaries, manifest, and package checks | `paper_reproduction_runbook.md` | Exists as reviewer-facing regeneration guidance for the paper layer; phase production jobs remain separate |
| Paper package index | Short start-here guide for claims, evidence objects, blockers, and validation commands | `paper_package_index.md` | Exists as the compact reader-facing orientation; it delegates claim authority to `paper_claim_matrix.md` |
| Reviewer FAQ | Scope and interpretation answers for likely review questions | `paper_reviewer_faq.md` | Exists; summarizes slop-style terminology, pybiber/EQ-Bench roles, DPO caveats, reduced-grid scope, SmolLM3 role, and remaining blockers |
| Citation audit | Citation-key, BibTeX, verified-source row, and source-URL integrity check | `paper_citation_audit.md`; `paper_citation_audit.csv`; `slop-audit-paper-citations` | Exists and marks all six citation/source checks `ready` |
| Methods draft | Manuscript-facing methods prose | `paper_methods_draft.md` | Exists with citation parity, AF equation formatting, and a manuscript-facing polish pass; remaining work is final venue copyediting |
| Results draft | Manuscript-facing results prose | `paper_results_draft.md` | Exists with claim-map coverage, figure/table-manifest references, and a manuscript-facing polish pass; remaining work is final venue copyediting |
| Intro/discussion draft | Manuscript-facing framing, limitations, and conclusion prose | `paper_intro_discussion_draft.md` | Exists with a manuscript-facing polish pass; remaining work is final venue copyediting |
| Integrated manuscript | End-to-end paper draft with numbered figure/table references, captions, and citations | `paper_manuscript_draft.md` | Exists as stitched draft with Related Work section, formal figure/table captions, reproducibility appendix references, and passing structure, limitations, and terminology audits |
| Paper package checker | Required-path, citation, manuscript-structure, figure/table, caption-coverage, C1-C14 claim coverage, and path-hygiene verifier | `slop-check-paper-package` | Exists and passes on the current package |
| Citation plan | Related-work and bibliography verification slots | `paper_citation_plan.md` | Exists; initial citation insertion complete, remaining slots explicit |
| Reference sources | Checked primary-source inventory | `paper_reference_sources.md` | Exists for initial source set |
| BibTeX draft | Checked bibliography subset | `paper_references.bib` | Exists for initial source set |

## Immediate Paper-Prep Queue

1. Keep Figure 3 as the headline OLMo `slop_lexicon` mechanism view, then do
   final venue sizing/export of the SVGs that now pass the figure-readiness
   and rendered-layout reviews; resolve the corresponding rows in
   `paper_venue_decision_checklist.md` and then follow
   `paper_venue_adaptation_runbook.md`.
2. Adapt `paper_latex_table_drafts.tex` to the final venue template. The table
   package now passes readiness and generic typography reviews, so remaining
   table work is venue-specific spacing, placement, and appendix inclusion.
3. Copyedit `paper_methods_draft.md`, `paper_results_draft.md`, and
   `paper_intro_discussion_draft.md` against the final submission template;
   citation parity, AF equation formatting, claim-map references,
   figure/table-manifest references, and manuscript-facing polish are already
   in place.
4. Continue final copyediting of `paper_manuscript_draft.md`: the integrated
   manuscript has passed an initial editorial polish pass plus structure,
   limitations, and claim-language audits, but final venue copyediting should
   still check caption wording against final rendered figures/tables and keep
   artifact paths confined to reproducibility appendix materials.
5. Expand `paper_reference_sources.md` and `paper_references.bib` only for
   citation slots that the final prose actually uses; the current bounded
   Methods section does not need a separate teacher-forcing/LM-scoring citation
   unless it broadens into a literature claim.
6. Run `uv run slop-check-paper-package --root /home/user/slop` after each
   substantial paper-doc edit; a passing check is a consistency guardrail, not
   a substitute for Tier-1 precision or Phase 4 human-perceptibility
   validation.
7. Regenerate `uv run slop-audit-paper-reproducibility-manifest --root
   /home/user/slop` after final manuscript, figure, table, or audit changes so
   the checksum index reflects the submitted evidence bundle.
8. Use `paper_reproduction_runbook.md` as the command-order checklist for any
   reviewer-facing refresh of the paper package.
9. Label the prepared Phase 4 20-per-candidate pilot sheet if the next paper
   push needs human-perceptibility evidence; otherwise keep detector claims
   candidate-only.
