# Paper Readiness Audit

Date: 2026-06-18

Purpose: assess whether the current experiment/document package is ready to
become a research paper characterizing elements of "slop style." This audit is
not a new analysis result. It is a submission-readiness control surface over
`EXPERIMENTS.md`, the claim matrix, the integrated manuscript, and the current
phase reports.

## Executive Status

The project now supports a bounded paper, but not a final submission. The
paper-grade core is the amplification-spectrum framing:

- OLMo/Dolci Phase 1 full-pybiber register shift is strong enough for a main
  result.
- OLMo `slop_lexicon` teacher-forced preference-stage peak plus compounding is
  strong enough for a bounded feature-level main result.
- EQ-Bench Slop Score is implemented and useful as an aggregate benchmark
  bridge, but not as the causal estimand.
- SmolLM3 provides useful cross-ladder pressure, not a full symmetric
  replication of every OLMo stage claim.
- Phase 4 detector discovery is a real executed pipeline with exact
  teacher-forced follow-up for candidates, but the human-perceived slop
  boundary remains open.

The current package is therefore best described as **paper-draft ready,
submission-blocked by venue finalization and Phase 4 human-validation**.
Tier-1 validation is dispositioned for the current scoped paper through
pass/demote/derived status.

## Paper-Grade Now

| Area | Status | Evidence | Paper treatment |
|---|---|---|---|
| Reader orientation | Ready | `docs/experiments/paper_package_index.md`; `docs/experiments/paper_reader_glossary.md`; `docs/experiments/paper_measurement_synthesis.md`; `docs/experiments/paper_scaffold.md`; `docs/experiments/paper_reviewer_faq.md` | Start new readers at the package index, use the glossary for recurring terms, use the measurement synthesis for the pybiber/EQ-Bench/propensity/generation/compounding/detector hierarchy, use the scaffold for manuscript structure, and use the FAQ for scope questions; claim authority remains the claim matrix. |
| Claim discipline | Ready | `docs/experiments/paper_claim_matrix.md`; `docs/experiments/paper_reviewer_faq.md`; `docs/experiments/paper_claim_evidence_map.md`; `docs/experiments/paper_claim_evidence_audit.md`; `docs/experiments/paper_claim_language_audit.md`; `artifacts/paper/claims/paper_claim_evidence_audit.csv`; `artifacts/paper/claims/paper_claim_language_audit.csv` | Use C1-C14 wording and caveats as the review authority; the FAQ answers common scope questions without expanding the claim surface; the integrated manuscript passes the claim-language audit, forbidden-overclaim checks, and claim-evidence path-integrity audit. |
| Source/provenance interpretation | Ready | `docs/experiments/paper_source_provenance_audit.md`; `artifacts/paper/submission/paper_source_provenance_audit.csv`; `docs/experiments/source_card_notes.md` | Source-card constraints for Dolci DPO, Dolma sampling, full pybiber, and SmolLM3 APO/source claims are explicit package checks rather than implicit prose caveats. |
| Tier-1 publication-use policy | Ready | `docs/experiments/paper_tier1_publication_policy.md`; `docs/experiments/precision_validation_status.md` | Distinguishes precision-supported corpus-rate claims from frozen-index model-side uses and exploratory rows. |
| Tier-1 validation disposition | Scoped-ready | `docs/experiments/precision_validation_status.md`; `artifacts/stage1/validation/precision_validation_status.csv` | Four independent core regex rows pass the interim gate; `rule_of_three_approx` is demoted; `stock_openers_closers` is derived. |
| Integrated manuscript | Draft ready | `docs/experiments/paper_manuscript_draft.md`; `docs/experiments/paper_manuscript_structure_audit.md`; `artifacts/paper/manuscript/paper_manuscript_structure_audit.csv` | Main body now avoids repo-path prose, includes formal figure/table captions, and ends with reproducibility appendix references; the structure audit marks heading order, Result 4.x subsection coverage, length bounds, caption counts, and appendix presence `ready`. |
| Abstract/conclusion alignment | Ready | `docs/experiments/paper_abstract_conclusion_audit.md`; `artifacts/paper/manuscript/paper_abstract_conclusion_audit.csv` | Abstract and conclusion preserve the bounded slop-style thesis, pybiber/EQ-Bench framing, and Phase 4 human-validation caveat. |
| Limitations coverage | Ready | `docs/experiments/paper_limitations_audit.md`; `artifacts/paper/manuscript/paper_limitations_audit.csv` | Limitations preserve the bounded open-ladder/sample, Tier-1 validation, sparse denominator, DPO preference-data, pybiber generation, EQ-Bench aggregate, Phase 4 human-validation, and reduced-grid boundaries. |
| Terminology/readability audit | Ready | `docs/experiments/paper_terminology_audit.md`; `artifacts/paper/manuscript/paper_terminology_audit.csv` | Integrated manuscript preserves the slop-style scope, measurement-layer definitions, amplification-spectrum framing, Tier-1 policy, and pybiber/EQ-Bench/Phase 4 boundaries for new readers. |
| Numeric claims audit | Ready | `docs/experiments/paper_numeric_claims_audit.md`; `artifacts/paper/manuscript/paper_numeric_claims_audit.csv` | Fifteen headline manuscript numeric claims match their source CSV/JSON artifacts after manuscript rounding. |
| Review-hygiene audit | Ready | `docs/experiments/paper_review_hygiene_audit.md`; `artifacts/paper/manuscript/paper_review_hygiene_audit.csv` | Integrated manuscript main body and paper-facing staged bundle files avoid local paths, internal tool commands, draft placeholders, and author placeholder text; review-control files and reproducibility appendices remain allowed to preserve evidence paths. |
| Figure/table inventory | Draft ready | `docs/experiments/paper_figure_table_manifest.md`; `docs/experiments/paper_figure_readiness_audit.md`; `docs/experiments/paper_figure_visual_review.md`; `docs/experiments/paper_table_readiness_audit.md`; `docs/experiments/paper_table_typography_review.md`; `artifacts/paper/figures/`; `docs/experiments/paper_camera_ready_tables.md`; `docs/experiments/paper_latex_table_drafts.tex` | Five polished draft SVGs with PDF/PNG submission-export candidates, paper-facing Markdown tables, and generic booktabs LaTeX table drafts are defined with captions and caveats; SVG/export hygiene, rendered-layout visual review, table-package hygiene, and generic table typography review pass. |
| Bibliography provenance | Ready | `docs/experiments/paper_citation_audit.md`; `artifacts/paper/references/paper_citation_audit.csv`; `docs/experiments/paper_reference_sources.md`; `docs/experiments/paper_references.bib` | Current cited source set is internally consistent: cited keys, BibTeX entries, verified source rows, and source URLs pass the citation audit. |
| Submission-exit audit | Blocked honestly | `docs/experiments/paper_submission_exit_audit.md`; `artifacts/paper/submission/paper_submission_exit_audit.csv` | Machine-readable exit audit separates paper-draft readiness from submission readiness; current blockers are final venue styling/sizing and Phase 4 human perceptibility labels. |
| Venue-decision checklist | Decision pending | `docs/experiments/paper_venue_decision_checklist.md`; `docs/experiments/paper_venue_adaptation_runbook.md`; `docs/experiments/paper_venue_sizing_inventory.md`; `artifacts/paper/submission/paper_venue_decision_checklist.csv` | Target venue, figure dimensions/export format, table placement, appendix-table inclusion, manuscript template, and human-labeling requirement are explicit `decision_pending` rows rather than implicit blockers; the runbook and sizing inventory give the post-decision adaptation steps. |
| Venue-neutral submission draft bundle | Ready as staging artifact | `artifacts/paper/submission/draft_bundle/README.md`; `artifacts/paper/submission/draft_bundle/MANIFEST.csv`; `uv run slop-materialize-paper-submission-bundle --root /home/user/slop` | A single staging directory now contains the package index, measurement synthesis, manuscript draft, caption appendix, table drafts, references, SVG/PDF/PNG figure candidates, and claim/readiness controls; it is not a venue submission certificate. |
| Open-release inventory | Ready as inventory | `docs/experiments/paper_release_inventory.md`; `artifacts/paper/submission/paper_release_inventory.csv`; `uv run slop-audit-paper-release-inventory --root /home/user/slop` | Maps the open-release deliverable to concrete feature definitions, opportunity definitions, harness code, paper-package materializers, runbooks, and cached paper statistics; this is not a license review or archival deposit. |
| Reproducibility manifest | Ready | `docs/experiments/paper_reproducibility_manifest.md`; `docs/experiments/paper_reproduction_runbook.md`; `artifacts/paper/submission/paper_reproducibility_manifest.csv` | Paper-facing docs, figures, audits, release-code source files, Phase 1 pybiber evidence, EQ-Bench interval evidence, and Phase 4 human-labeling materials are indexed with file sizes and SHA-256 checksums; the runbook records the refresh command order. |
| Paper package consistency checker | Ready | `uv run slop-check-paper-package --root /home/user/slop` | Current package passes required-path, manuscript/table hygiene, manuscript-structure, limitations, terminology, numeric-claims, review-hygiene, source-provenance audits, Phase 4 blind-packet redaction/row-alignment checks, figure/table reference, caption-coverage, C1-C14 claim coverage, citation-key, stale-phrase, and linked-path checks. |
| Phase 1 pybiber register result | Main-result ready with bounded sample caveat | `docs/experiments/phase1_pybiber_register_analysis.md`; `artifacts/stage1/census/phase1_pybiber_register_means.csv`; `artifacts/stage1/census/phase1_pybiber_register_deltas.csv`; `artifacts/stage1/census/phase1_pybiber_register_intervals.csv` | Main paper Result 1; selected-feature document-bootstrap intervals now support Figure 1/Table 2 uncertainty language. |
| OLMo `slop_lexicon` amplification result | Main-result ready within bounded scope | `docs/experiments/phase3_integrated_conclusion_report.md`; `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv` | Main paper Result 3. |
| Reduced SGLang scope | Methods-ready | `EXPERIMENTS.md`; `docs/experiments/phase3_completion_audit.md`; `docs/experiments/phase3_production_runbook.md` | Present as the bounded production estimand, not the abandoned exhaustive grid. |

## Usable With Explicit Caveat

| Area | Current evidence | Required caveat |
|---|---|---|
| DPO chosen/rejected interpretation | `docs/experiments/phase1_pybiber_register_analysis.md`; `docs/experiments/phase1_conclusions.md` | Dolci DPO is mixed/synthetic/Delta Learning-style data, not a pure human-preference contrast. |
| EQ-Bench Slop Score | `src/slop_sftdiv/features/eqbench_slop.py`; `src/slop_sftdiv/cli/eqbench_slop_score.py`; `src/slop_sftdiv/cli/summarize_eqbench_intervals.py`; `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_stage_comparison.md`; `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.md` | Aggregate diagnostic with output-bootstrap intervals; do not use as the causal measurement layer. |
| SmolLM3 cross-ladder comparison | `docs/experiments/phase3_dpo_vs_apo_variation_report.md`; SmolLM3 spectrum artifacts under `artifacts/phase3/analysis/` | Small shared AF support and different checkpoint/training construction; claim replication pressure, not universal replication. |
| Phase 4 detector candidates | `docs/experiments/phase4_detector_discovery_report.md`; `docs/experiments/phase4_tier3_teacher_forced_addendum.md`; `docs/experiments/phase4_human_perceptibility_protocol.md`; `docs/experiments/phase4_human_annotation_codebook.md`; `docs/experiments/phase4_human_annotation_readiness.md`; `docs/experiments/phase4_human_annotation_handoff.md`; `docs/experiments/phase4_human_labeling_execution_checklist.md`; `docs/experiments/phase4_human_perceptibility_summary.md`; `artifacts/phase4/modernbert_detector_combined_v2_clean/` | Candidate machine-detectable style families until human perceptibility labels and matcher precision validation exist; the labeling protocol, codebook, readiness audit, annotator quickstart, blinded pilot/full-package sheets, redacted handoff bundle, coordinator checklist, and summarizer are now ready. |
| Follow-up offers and sparse Tier-3 features | `docs/experiments/phase4_tier3_teacher_forced_addendum.md` | Large point estimates but sparse denominator support; do not headline. |
| Tier-1 regex features | `docs/experiments/precision_validation_status.md`; `docs/experiments/paper_tier1_publication_policy.md` | `contrastive_negation`, `slop_lexicon`, `stock_openers`, and `stock_closers` currently pass the interim span/item precision gate; `slop_lexicon` still requires item-level interpretation. |

## Submission Blockers

The machine-readable submission-exit audit is
`docs/experiments/paper_submission_exit_audit.md`, with CSV output at
`artifacts/paper/submission/paper_submission_exit_audit.csv`. It currently
marks the overall submission status `blocked` because figure/table
finalization still needs final venue styling/sizing and Phase 4 human
perceptibility is still `human_validation_pending`.

| Blocker | Why it matters | Current evidence | Smallest useful next action |
|---|---|---|---|
| Human perceptibility labels for Phase 4 absent | Detector clusters cannot be called human-perceived slop yet. | `docs/experiments/phase4_completion_audit.md`; `docs/experiments/phase4_human_perceptibility_protocol.md`; `docs/experiments/phase4_human_annotation_codebook.md`; `docs/experiments/phase4_human_annotation_readiness.md`; `docs/experiments/phase4_human_annotation_handoff.md`; `docs/experiments/phase4_human_labeling_execution_checklist.md`; `docs/experiments/phase4_human_perceptibility_summary.md`; annotation package exists, the codebook defines candidate-specific label boundaries, readiness audit reports 10 balanced blank candidate rows plus blinded pilot and full-package packets, the annotator quickstart defines allowed edit columns and return rules, the redacted handoff bundle separates annotator files from coordinator maps, the coordinator checklist defines post-label claim updates, the agreement and adjudication tools are available, and summary records 0/100 labeled rows per candidate. | Label the blinded 20-per-candidate pilot or the blinded full 1,000-example package from the handoff bundle; for blind labels, run `uv run slop-summarize-phase4-label-agreement`, fill adjudication columns, apply them with `uv run slop-apply-phase4-label-adjudication`, then summarize with `uv run slop-summarize-phase4-human-labels`; until then, keep detector claims machine-detectable only. |
| Figures need final venue sizing | SVGs pass package hygiene checks, PDF/PNG export candidates exist, and rendered-layout visual review passes, but final dimensions/export choice still depends on the venue. | `artifacts/paper/figures/`; `artifacts/paper/figures/submission_exports/`; `src/slop_sftdiv/cli/render_paper_figures.py`; `src/slop_sftdiv/cli/audit_paper_figures.py`; `docs/experiments/paper_figure_table_manifest.md`; `docs/experiments/paper_venue_sizing_inventory.md`; `docs/experiments/paper_figure_readiness_audit.md`; `docs/experiments/paper_figure_visual_review.md` | Adjust final dimensions and choose the venue-required export format. |
| Tables need final venue template adaptation | The Markdown and generic booktabs table package passes heading, row-count, caption/label, path-hygiene, and generic typography checks; it still needs adaptation to the target venue template. | `docs/experiments/paper_camera_ready_tables.md`; `docs/experiments/paper_latex_table_drafts.tex`; `docs/experiments/paper_table_readiness_audit.md`; `docs/experiments/paper_table_typography_review.md`; `artifacts/paper/tables/paper_table_readiness_audit.csv`; `artifacts/paper/tables/paper_table_typography_review.csv` | Apply target venue spacing/placement and decide whether appendix tables A1-A3 stay in the submission. |
| Manuscript prose still needs final venue copyedit | Draft is readable, passes the manuscript-structure, limitations, terminology, claim-language, and review-hygiene audits, and has had an initial editorial polish pass to remove several internal/status-report turns of phrase. The standalone Methods, Results, and Intro/Discussion drafts have also been tightened for manuscript-facing prose. | `docs/experiments/paper_manuscript_draft.md`; `docs/experiments/paper_methods_draft.md`; `docs/experiments/paper_results_draft.md`; `docs/experiments/paper_intro_discussion_draft.md`; `docs/experiments/paper_manuscript_structure_audit.md`; `docs/experiments/paper_limitations_audit.md`; `docs/experiments/paper_terminology_audit.md`; `docs/experiments/paper_claim_language_audit.md`; `docs/experiments/paper_review_hygiene_audit.md`; `docs/experiments/paper_caption_appendix_draft.md` | Do final copyediting against the target venue template, check caption wording against final rendered figures/tables, and keep artifact paths confined to reproducibility appendix materials. |
| Venue decisions explicit but unresolved | Submission-specific choices should not be inferred from draft artifacts. | `docs/experiments/paper_venue_decision_checklist.md`; `docs/experiments/paper_venue_adaptation_runbook.md`; `artifacts/paper/submission/paper_venue_decision_checklist.csv` | Select target venue, follow the adaptation runbook, and resolve all `decision_pending` rows before claiming submission readiness. |
| EQ-Bench remains aggregate despite intervals | Figure 2 now has output-bootstrap intervals, but the score still hides feature mechanism. | `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_stage_comparison.md`; `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.md` | Keep Figure 2 framed as a benchmark bridge and avoid treating score intervals as causal AF evidence. |

## Submission Blocker Burn-Down Order

Treat the remaining blockers as two tracks, and do not promote the package to
submission-ready until both tracks have explicit evidence.

1. **Human-label track.** Collect two independent full-package blind label
   sheets, run agreement before discussion, adjudicate disagreements, apply the
   adjudication to canonical JSONL, and regenerate the Phase 4 human
   perceptibility summary. Done means the summary contains real labeled rows,
   C11/C12/C13 have been updated if any detector family is promoted, and
   detector-only wording remains for all unpromoted candidates.
2. **Venue track.** Choose the venue/template, resolve every
   `decision_pending` row in the venue-decision checklist, apply the venue's
   figure/table sizing and export requirements, and rerun figure, table,
   submission-exit, reproducibility-manifest, and package checks. Done means
   `slop-audit-paper-submission-exit` reports `overall_submission_status` as
   `ready`.

If the human-label track is deliberately deferred for a venue submission, the
manuscript must keep Phase 4 as detector-discovery evidence only and must not
describe detector clusters as human-perceived slop.

## Non-Blocking Deferred Work

| Deferred item | Treatment |
|---|---|
| Full pybiber over generated outputs | Not required for the current bounded thesis; generated-output register proxy artifacts are excluded from main-paper claims. |
| Larger Phase 4 denominator expansion | Useful only after candidate features are validated or selected for headline claims. |
| SAE detector-attribution path | Stretch methods contribution, not required for the current paper. |
| Original 5,000 x 8 x 3 OLMo generation grid | Superseded by the bounded SGLang production scope in `EXPERIMENTS.md`. |
| Remaining SmolLM3 pretraining source coverage | No longer a blocker; current framing avoids 100% source reconstruction claims. |

## Recommended Next Three Work Items

1. **Venue-copyedit without weakening claim controls.** Use
   `docs/experiments/paper_claim_evidence_map.md`,
   `docs/experiments/paper_claim_language_audit.md`, and
   `docs/experiments/paper_numeric_claims_audit.md` while adapting prose to the
   target template; the main manuscript now passes the claim-language,
   forbidden-overclaim, numeric-claims, and structure checks.
2. **Camera-ready the main evidence objects.** Polish Figure 1-Figure 5 and
   adapt the generic booktabs Table 1/Table 2/Table 4 drafts to the final
   venue template, then update `docs/experiments/paper_figure_table_manifest.md`
   with final captions.
3. **Preserve validation boundaries.** Tier-1 is now dispositioned for the
   current scope: use passing rows as precision-supported, keep
   `rule_of_three_approx` exploratory, and keep `stock_openers_closers`
   derived. For Phase 4, either collect human labels or preserve
   candidate-only language.

## Claim Readiness Summary

| Claim band | Claim IDs | Readiness |
|---|---|---|
| Main-text ready | C1, C2, C7, C8, C9 | Ready with bounded language already in the claim matrix. |
| Main-text ready with nearby caveat | C3, C4, C5, C6, C10, C12, C13 | Usable if caveats are colocated with claims. |
| Methods/status only | C11, C14 | Keep out of headline findings. |

## Exit Criteria For Submission-Ready Package

A submission-ready package should have:

1. Final figure/table captions and camera-ready artifacts for Figure 1-Figure 5
   and final-styled Table 1/Table 2/Table 4; paper-facing Markdown and generic
   LaTeX table bodies now exist.
2. A manuscript draft with no repo-path prose in the main body and a passing
   structure audit for section order, length bounds, result subsections,
   captions, appendix presence, limitations-boundary coverage, and
   reader-facing terminology coverage.
3. Claim IDs C1-C14 either represented in final text with allowed wording or
   explicitly omitted, with the claim-evidence audit still passing.
4. Tier-1 pass/demote/derived boundaries preserved in final text and captions.
5. Phase 4 detector claims explicitly marked as candidate-only unless human
   labels are added.
6. Numeric-claims audit passing after final manuscript edits.
7. Citation and source inventories passing the citation audit and existing
   key/path consistency checks.
8. A ready reproducibility manifest with file sizes and SHA-256 checksums for
   paper-facing docs and artifacts, plus a reproduction runbook for refreshing
   the paper-facing layer from cached phase outputs.
9. `uv run slop-check-paper-package --root /home/user/slop` passing after final
   manuscript, table, figure-manifest, and bibliography edits.
10. `uv run slop-audit-paper-submission-exit --root /home/user/slop` reporting
   `overall_submission_status` as `ready`.
