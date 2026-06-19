# Paper Claim Evidence Map

Date: 2026-06-18

Purpose: connect the paper-facing claim matrix to the integrated manuscript,
figures, tables, and source artifacts. The authoritative allowed wording and
claim strength remain in `docs/experiments/paper_claim_matrix.md`; this file
is the review trail for where each claim is used.

## Claim-To-Manuscript Map

| Claim ID | Claim role | Manuscript location | Figure/table support | Primary evidence artifacts | Residual caveat |
|---|---|---|---|---|---|
| C1 | Data-side register shift | Section 4.1 Result 1; Discussion | Figure 1; Table 2 | `docs/experiments/phase1_pybiber_register_analysis.md`; `artifacts/stage1/census/phase1_pybiber_register_means.csv`; `artifacts/stage1/census/phase1_pybiber_register_deltas.csv`; `artifacts/stage1/census/phase1_pybiber_register_intervals.csv` | Dolma is an English-filtered retained sample from a bounded scan, not full Dolma; intervals cover selected paper-facing features. |
| C2 | Register shift is not just hedging | Section 4.1 Result 1; Discussion; Limitations | Figure 1; Table 2; Table 5 | Same pybiber register means, deltas, and selected-feature intervals as C1 | Aggregate pybiber result does not rule out hedging in individual generated answers. |
| C3 | DPO chosen/rejected register contrast is not uniformly slop-like | Section 4.1 Result 1; Limitations | Figure 1; Table 2 | `docs/experiments/phase1_pybiber_register_analysis.md`; Phase 1 DPO pair deltas | Dolci DPO is not a pure human-preference contrast. |
| C4 | EQ-Bench is a benchmark bridge, not the causal estimand | Section 3.4 Methods; Section 4.2 Result 2; Discussion | Figure 2; Table 5 | `src/slop_sftdiv/features/eqbench_slop.py`; `src/slop_sftdiv/cli/eqbench_slop_score.py`; `src/slop_sftdiv/cli/summarize_eqbench_intervals.py`; `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_stage_comparison.md`; `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.md`; `docs/experiments/paper_reference_sources.md` | Aggregate diagnostic with output-bootstrap intervals; still not a causal estimand. |
| C5 | OLMo EQ-Bench score peaks at SFT in bounded output panel | Section 4.2 Result 2 | Figure 2 | `artifacts/phase2/analysis/eqbench_slop/phase1_phase2_eqbench_slop_comparison.csv`; `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv`; EQ-Bench OLMo generation score outputs under `artifacts/phase2/analysis/eqbench_slop/` | Bounded 512-prompt/8-completion temperature-1 panel; intervals are over valid outputs. |
| C6 | SmolLM3 EQ-Bench score rises after SFT and remains high | Section 4.2 Result 2; Section 4.4 Result 4 | Figure 2 | `artifacts/phase2/analysis/eqbench_slop/phase1_phase2_eqbench_slop_comparison.csv`; `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv`; SmolLM3 EQ-Bench score outputs under `artifacts/phase2/analysis/eqbench_slop/` | Bounded no-think panel; aggregate score mostly lexical-density driven; DPO and final intervals overlap. |
| C7 | OLMo `slop_lexicon` teacher-forced AF peaks at DPO | Section 4.3 Result 3; Discussion | Figure 3 | `docs/experiments/phase3_integrated_conclusion_report.md`; `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv`; `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_feature_classification_t07_long1024.csv` | Applies to bounded supported opportunity contract. |
| C8 | OLMo `slop_lexicon` DPO peak is not explained by chosen-response enrichment | Section 4.3 Result 3; Discussion; Limitations | Figure 3; Table 5 | Phase 3 integrated spectrum joined to Phase 1 paired Dolci tests; `docs/experiments/phase3_integrated_conclusion_report.md`; `docs/experiments/phase1_conclusions.md` | Delta Learning means this is a preference-data contrast, not a pure human-taste estimate. |
| C9 | Propensity and free-running output can diverge; compounding matters | Section 3.1 Study Design; Section 3.4 Methods; Section 4.3 Result 3; Discussion | Figure 3 | OLMo generation-rate and compounding artifacts summarized in `docs/experiments/phase3_integrated_conclusion_report.md` and `docs/experiments/paper_results_draft.md` | Reduced panels answer the bounded estimand, not the abandoned exhaustive grid. |
| C10 | OLMo and SmolLM3 show related style pressure with different stage localization | Section 4.4 Result 4; Discussion | Figure 4 | `docs/experiments/phase3_integrated_conclusion_report.md`; `docs/experiments/phase3_dpo_vs_apo_variation_report.md`; SmolLM3 amplification spectrum CSV under `artifacts/phase3/analysis/` | Small shared non-missing AF support; ladders differ in scale, algorithm, prompts, and checkpoint construction. |
| C11 | Phase 4 detector discovery produced candidate Tier-3 features, not validated human-perceived slop | Section 3.3 Feature Surface; Section 4.5 Result 5; Limitations | Figure 5; Table 5 | `docs/experiments/phase4_detector_discovery_report.md`; `docs/experiments/phase4_human_perceptibility_protocol.md`; `docs/experiments/phase4_human_annotation_codebook.md`; `docs/experiments/phase4_human_annotation_readiness.md`; `docs/experiments/phase4_human_perceptibility_summary.md`; `artifacts/phase4/modernbert_detector_combined_v2_clean/`; annotation package, blinded pilot/full-package packets, import maps, and pilot sheet under Phase 4 artifacts | Human perceptibility labels are still absent; readiness audit reports a balanced blank package and current summary records 0/100 labeled rows per candidate. |
| C12 | Some detector candidates have real teacher-forced sequence-mass support | Section 4.5 Result 5 | Figure 5 | `docs/experiments/phase4_tier3_teacher_forced_addendum.md`; `docs/experiments/phase4_completion_audit.md`; Phase 4 exact 512 stage grid and primary comparison artifacts | Stronger for denominator-supported candidates than sparse ones. |
| C13 | Follow-up offers are large but sparse | Section 4.5 Result 5; Limitations | Figure 5 | `docs/experiments/phase4_tier3_teacher_forced_addendum.md`; Phase 4 exact 512 comparison artifacts | Only 11 reference initiations; provisional, not a standalone headline. |
| C14 | Formatting features are excluded from the active thesis | Section 3.3 Feature Surface; Limitations | Table 5 | `EXPERIMENTS.md`; `docs/experiments/phase1_gap_audit.md`; `docs/experiments/manual_feature_regex_audit_2026-06-17.md` | Historical artifacts may still contain retired formatting-feature measurements. |

## Manuscript Coverage By Section

| Manuscript section | Claim IDs | Evidence objects |
|---|---|---|
| Section 3.1 Study Design | C9 | Defines corpus inheritance, fixed-context propensity, free-running emission, and self-conditioning. |
| Section 3.2 Model And Data Ladders | C1, C3, C5, C6, C10 | Establishes OLMo/Dolci and SmolLM3 bounded scope; Table 1. |
| Section 3.3 Feature Surface | C4, C11, C14 | Defines Tier 1/Tier 2/Tier 3, pybiber, EQ-Bench, detector candidates, and retired formatting features; Table 4. |
| Section 3.4 Corpus Rates, EQ-Bench, Propensity, And Generation | C4, C9 | Defines corpus rates, EQ-Bench bridge, AF, neutral-normalized AF, SGLang generation, and compounding. |
| Section 4.1 Result 1 | C1, C2, C3 | Figure 1; Table 2. |
| Section 4.2 Result 2 | C4, C5, C6 | Figure 2. |
| Section 4.3 Result 3 | C7, C8, C9 | Figure 3. |
| Section 4.4 Result 4 | C10 | Figure 4. |
| Section 4.5 Result 5 | C11, C12, C13 | Figure 5. |
| Section 5 Discussion | C1, C2, C7, C8, C9, C10, C11, C12 | Explains the amplification-spectrum thesis and mechanism separation. |
| Section 6 Limitations | C3, C4, C10, C11, C12, C13, C14 | Records bounded samples, Tier-1 pass/demote/derived boundaries, sparse denominators, DPO caveats, pybiber/generation limits, EQ-Bench scorecard limits, and detector-validation limits. |

## Claim Strength For Main Text

| Placement | Claim IDs | Treatment |
|---|---|---|
| Main headline claims | C1, C2, C7, C8, C9 | Use in abstract/introduction/results with bounded language. |
| Main text with explicit caveat | C3, C4, C5, C6, C10, C12, C13 | Use where the caveat appears nearby or in the same paragraph. |
| Methods/status or limitation framing | C11, C14 | Do not promote to headline empirical conclusions. |

## Resolved Presentation Decisions

1. Figure 3 remains the headline OLMo `slop_lexicon` mechanism panel. It is
   the strongest feature-level result because it joins fixed-context
   propensity, free-running output, compounding, and the DPO chosen/rejected
   caveat in one view. A broader full-spectrum panel can remain a supplement
   if space allows, but it should not replace the cleaner mechanism figure.
2. Table 3 and Table 5 remain appendix/review controls by default. Table 3 is
   useful for claim governance, and Table 5 is useful for preventing
   overclaiming, but the main text should carry the relevant caveats locally
   rather than relying on a control table to do the argumentative work.
3. A separate teacher-forcing/LM-scoring citation is not required for the
   current bounded manuscript. The paper defines its own amplification-factor
   estimand, uses calibration and generation-background citations, and should
   add a new citation only if the final Methods prose expands into a broader
   literature claim about teacher-forced scoring.
4. Figure 1 and Figure 2 already plot bootstrap intervals, and the
   rendered-layout figure review passes. Remaining figure work is final venue
   sizing/export, not interval integration.
