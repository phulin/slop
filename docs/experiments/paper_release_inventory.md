# Paper Release Inventory

Machine-readable inventory: `artifacts/paper/submission/paper_release_inventory.csv`

Readiness status: `ready`.
This inventory maps the EXPERIMENTS.md open-release deliverable to concrete repository paths: feature definitions, opportunity definitions, harness code, runbooks, and cached paper statistics. It is an inventory, not a license review or final archival deposit.

| Category | Items |
|---|---:|
| cached_statistics | 7 |
| feature_definitions | 3 |
| harness_code | 9 |
| opportunity_definitions | 2 |
| runbook | 4 |

| Category | Path | Status | Role | Caveat |
|---|---|---|---|---|
| feature_definitions | `src/slop_sftdiv/features/tier1_matchers.py` | ready | Tier-1 regex/span matchers used for corpus and generation scoring. | Rule-of-three remains demoted for independent publication-grade claims. |
| feature_definitions | `src/slop_sftdiv/features/eqbench_slop.py` | ready | Portable EQ-Bench Slop Score implementation and aggregate scoring helpers. | Benchmark bridge only; not the causal propensity estimand. |
| feature_definitions | `src/slop_sftdiv/features/pybiber_full.py` | ready | Full pybiber feature extraction wrapper for corpus-side register analysis. | Corpus-side Phase 1 use only; generated-output full pybiber is not claimed. |
| opportunity_definitions | `src/slop_sftdiv/propensity.py` | ready | Opportunity definitions and initiator sets for teacher-forced propensity. | Some Tier-3 opportunities remain provisional until human validation. |
| opportunity_definitions | `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_tier3_matcher_specs.json` | ready | Detector-derived Tier-3 matcher specs and opportunity definitions. | Candidate features require human perceptibility labels and precision validation. |
| harness_code | `src/slop_sftdiv/cli/teacher_forced_propensity.py` | ready | Teacher-forced propensity harness. | Production conclusions use bounded retained reference panels. |
| harness_code | `src/slop_sftdiv/cli/free_running_emission.py` | ready | Free-running generation/scoring harness. | Reduced SGLang grids are the production estimand. |
| harness_code | `src/slop_sftdiv/generation_text.py` | ready | Generation text extraction and normalization helpers. | Used to keep generated-output scoring consistent across backends. |
| harness_code | `src/slop_sftdiv/cli/import_phase4_blind_labels.py` | ready | Phase 4 blinded-label import utility. | Use only after annotators return blinded sheets with matching private maps. |
| harness_code | `src/slop_sftdiv/cli/summarize_phase4_label_agreement.py` | ready | Phase 4 dual-annotator agreement summarizer. | Agreement rows require adjudication before final human-perceptibility claims. |
| harness_code | `src/slop_sftdiv/cli/apply_phase4_label_adjudication.py` | ready | Phase 4 adjudication applier for human labels. | Only adjudicated canonical labels should feed final human-perceptibility summaries. |
| harness_code | `src/slop_sftdiv/cli/check_paper_package.py` | ready | Paper package consistency checker. | Guardrail only; passing checks do not resolve venue or human-label blockers. |
| harness_code | `src/slop_sftdiv/cli/materialize_paper_submission_bundle.py` | ready | Venue-neutral draft-bundle materializer. | Staging helper only; final venue submission still requires template adaptation. |
| harness_code | `src/slop_sftdiv/cli/audit_paper_reproducibility_manifest.py` | ready | Paper reproducibility manifest writer. | Checksum inventory only; does not certify external archival completeness. |
| runbook | `docs/experiments/paper_reproduction_runbook.md` | ready | Paper-facing refresh command order. | Refreshes cached paper layer; does not rerun large production jobs. |
| runbook | `docs/experiments/paper_venue_sizing_inventory.md` | ready | Venue-facing figure/table sizing and export inventory. | Inventory only; final dimensions still depend on target venue. |
| runbook | `docs/experiments/phase3_production_runbook.md` | ready | Reduced SGLang generation and amplification-spectrum production runbook. | Original exhaustive OLMo grid remains optional addendum. |
| runbook | `docs/experiments/phase4_production_runbook.md` | ready | Detector discovery and Tier-3 follow-up production runbook. | Human labels remain external to the current cached run. |
| cached_statistics | `artifacts/stage1/census/feature_rates_by_corpus.parquet` | ready | Phase 1 corpus feature rates. | Bounded English-filtered retained samples, not full-corpus coverage. |
| cached_statistics | `artifacts/stage1/census/preference_pair_deltas.parquet` | ready | Phase 1 Dolci DPO chosen/rejected pair deltas. | Preference-data contrast is not a pure human-preference estimate. |
| cached_statistics | `artifacts/stage1/census/phase1_pybiber_register_intervals.csv` | ready | Selected full-pybiber bootstrap intervals. | Selected paper-facing features, not all 67 features. |
| cached_statistics | `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv` | ready | EQ-Bench Slop Score stage intervals. | Aggregate diagnostic with output-bootstrap intervals. |
| cached_statistics | `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv` | ready | Primary reduced OLMo amplification spectrum. | Bounded SGLang production scope. |
| cached_statistics | `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_production.csv` | ready | Primary SmolLM3 no-think amplification spectrum. | Cross-ladder comparison has small shared non-missing AF support. |
| cached_statistics | `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv` | ready | Phase 4 Tier-3 exact sequence-mass teacher-forced stage grid. | Tier-3 features remain candidate-only pending human labels. |
