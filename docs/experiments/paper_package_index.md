# Paper Package Index

Date: 2026-06-18

Purpose: give a new reader the shortest reliable path through the current
slop-style paper package. This is an orientation index, not a new claim source.
Use `docs/experiments/paper_claim_matrix.md` as the authority for allowed
claims and caveats.

## Start Here

1. Read `docs/experiments/paper_manuscript_draft.md` for the integrated paper
   draft.
2. Read `docs/experiments/paper_reader_glossary.md` for recurring terms such
   as AF, pybiber, EQ-Bench, Tier 1/Tier 2/Tier 3, and compounding.
3. Read `docs/experiments/paper_measurement_synthesis.md` for the measurement
   hierarchy linking pybiber, EQ-Bench, propensity, sampled output,
   compounding, and detector candidates.
4. Read `docs/experiments/paper_claim_matrix.md` for the supported claim
   surface.
5. Read `docs/experiments/paper_claim_evidence_map.md` for claim-to-evidence
   traceability.
6. Read `docs/experiments/paper_readiness_audit.md` for readiness and
   blockers.
7. Run `uv run slop-check-paper-package --root /home/user/slop` before
   treating the package as internally consistent.

## Core Thesis

The bounded paper argues that slop style is feature-specific, stage-specific,
and partly a register shift. The strongest supported claims are:

- OLMo/Dolci post-training data shift from narrative/personal web prose toward
  nominalized assistant-answer exposition.
- That broad pybiber register shift is not simply more hedging.
- The pybiber DPO contrast is register-specific: chosen responses are more
  descriptive/expository, while rejected responses are more personal,
  narrative, and mental-state framed.
- EQ-Bench is useful as a public aggregate score, but its trajectory can
  disagree with feature-specific propensity; in the bounded panels, its
  movement is mostly lexical-density evidence rather than a mechanism claim.
- OLMo `slop_lexicon` shows a DPO-stage teacher-forced propensity peak in the
  bounded production scope.
- The OLMo `slop_lexicon` peak is not explained by measured chosen-response
  enrichment in Dolci DPO.
- Propensity, sampled output, and compounding answer different questions and
  should not be collapsed into one score.

## Main Evidence Objects

| Evidence layer | Primary artifacts | Paper role |
|---|---|---|
| Claim controls | `paper_claim_matrix.md`; `paper_claim_evidence_map.md`; `paper_reader_glossary.md`; `paper_measurement_synthesis.md`; `paper_reviewer_faq.md`; `paper_source_provenance_audit.md` | Keeps result wording bounded, source-card aware, readable, and reviewable. |
| Integrated writing | `paper_manuscript_draft.md`; `paper_methods_draft.md`; `paper_results_draft.md`; `paper_intro_discussion_draft.md`; `paper_review_hygiene_audit.md` | Current manuscript text, section drafts, and paper-facing review-hygiene check. |
| Phase 1 register | `phase1_pybiber_register_analysis.md`; `phase1_pybiber_register_means.csv`; `phase1_pybiber_register_intervals.csv` | Main corpus-side pybiber result. |
| EQ-Bench bridge | `phase2_eqbench_slop_intervals.csv`; `phase2_eqbench_slop_intervals.md` | Aggregate diagnostic, not causal estimand. |
| Phase 3 amplification | `phase3_integrated_conclusion_report.md`; OLMo/SmolLM3 spectrum artifacts under `artifacts/phase3/analysis/` | Feature-level propensity/generation/compounding result. |
| Phase 4 detector candidates | `phase4_detector_discovery_report.md`; `phase4_tier3_teacher_forced_addendum.md`; `phase4_production_runbook.md`; `phase4_human_perceptibility_protocol.md`; `phase4_human_annotation_handoff.md`; `phase4_human_labeling_execution_checklist.md` | Candidate machine-detectable style families until human labels exist; the production runbook records the completed 512-document exact Tier-3 rerun, and the handoff bundle separates annotator files from coordinator maps while the checklist gives the post-label claim-update path. |
| Figures and tables | `paper_figure_table_manifest.md`; `paper_venue_sizing_inventory.md`; `paper_camera_ready_tables.md`; `paper_latex_table_drafts.tex`; `artifacts/paper/figures/` | Draft submission evidence objects plus current venue-sizing inventory. |
| Reproducibility and release | `paper_reproduction_runbook.md`; `paper_reproducibility_manifest.md`; `paper_release_inventory.md`; `paper_source_provenance_audit.md`; `paper_submission_exit_audit.md` | Refresh order, release-code and artifact checksums, release-facing object inventory, provenance guardrails, and submission blockers. |

## Claim Boundaries

- Full pybiber is a Phase 1 corpus-side register result; generated-output full
  pybiber is not claimed.
- EQ-Bench Slop Score is a benchmark bridge and aggregate lexical-density
  diagnostic, not the causal measurement layer.
- EQ-Bench bootstrap intervals quantify valid-output sampling variability;
  they do not turn the aggregate score into AF or mechanism evidence.
- Dolci DPO chosen/rejected contrasts are preference-data evidence, not a pure
  human-preference estimate.
- The Dolci DPO pybiber contrast should not be reduced to
  chosen-responses-are-more-slop-like.
- The OLMo pretraining reference is a retained English-filtered Dolma sample
  from a bounded scan, not exhaustive source-corpus coverage.
- Phase 4 detector-derived families remain machine-detectable candidates until
  human perceptibility labels are collected and adjudicated.
- The reduced SGLang panels are the production estimand; the abandoned
  exhaustive 5,000 x 8 x 3 OLMo grid is not claimed.

## Current Blockers

The package is paper-draft ready but not submission ready. The active blockers
are:

- Phase 4 human perceptibility labels are absent.
- Target venue decisions and final figure/table template adaptation are still
  unresolved.

The Phase 4 labeling path is operational: blinded pilot packet, label import,
two-annotator agreement, adjudication application, and final label summary
commands exist. The blocker is real labels, not missing tooling.

## Validation Commands

Run these after paper-facing edits:

```bash
uv run slop-audit-paper-claim-evidence
uv run slop-audit-paper-claim-language
uv run slop-audit-paper-review-hygiene --root /home/user/slop
uv run slop-audit-paper-release-inventory --root /home/user/slop
uv run slop-audit-paper-submission-exit --root /home/user/slop
uv run slop-materialize-paper-submission-bundle --root /home/user/slop
uv run slop-audit-paper-reproducibility-manifest --root /home/user/slop
uv run slop-check-paper-package --root /home/user/slop
```
