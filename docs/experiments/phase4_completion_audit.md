# Phase 4 Completion Audit

This audit reflects the corrected Phase 4 execution after making concrete
detector/attribution attempts. The earlier `bounded_discovery_v1` n-gram pass
remains a fallback artifact, but the primary Phase 4 evidence is now the
ModernBERT detector pipeline in
`artifacts/phase4/modernbert_detector_combined_v2_clean/`.

| Requirement from EXPERIMENTS.md | Status | Evidence |
|---|---|---|
| Fine-tune ModernBERT-large detector on HAP-E plus generations | Completed | `modernbert_detector_combined_v2_clean/checkpoint`, `phase4_detector_summary.json`, `phase4_detector_train_log.csv`. |
| Evaluate cross-model detector generalization | Completed with caveats | `phase4_detector_metrics.csv` includes held-out HAP-E authors plus OLMo/SmolLM3 external generation/reference evaluation. |
| Integrated-gradients attribution over at least 10k documents | Completed | `ig_10000doc/phase4_detector_ig_summary.json` reports 10,000 attributed documents, 77,013 doc-span rows, and 52,109 unique spans. |
| Embed spans and cluster with HDBSCAN | Completed | `ig_10000doc_hdbscan_gte/phase4_detector_ig_hdbscan_summary.json` reports GTE-large embeddings over 5,000 selected spans and 56 non-noise HDBSCAN clusters. |
| Auto-label clusters with LLM and manual review | Partially completed | Clusters were manually summarized in `docs/experiments/phase4_detector_discovery_report.md` and mapped into 10 provisional matcher families. An automated LLM-labeling workflow was not run; formal manual review remains limited to this report-level triage. |
| SAE attribution path | Deferred stretch | No SAE artifacts produced; stretch item remains non-blocking in EXPERIMENTS.md. |
| Human-perceptibility validation, 100 examples per cluster | Protocol, readiness audit, blinded pilot/full-package sheets, import path, and summary completed; human labels not completed | `phase4_human_perceptibility_annotation_package.jsonl` contains 1,000 examples, 100 per candidate feature. `docs/experiments/phase4_human_perceptibility_protocol.md` defines the label schema, blind-label import command, and promotion rules. `docs/experiments/phase4_human_annotation_readiness.md` reports a balanced blank package plus blinded 20-per-candidate pilot and full-package packets. `docs/experiments/phase4_human_perceptibility_summary.md` currently records 0/100 labeled rows for each candidate. `phase4_perceptibility_proxy_labels.jsonl` contains rule-based Codex/Shaib-taxonomy proxy labels. |
| Detector-vs-human-perceived Venn diagram | Proxy completed; human Venn not completed | `phase4_detector_vs_proxy_perceived_venn.csv` and `.svg` summarize detector-vs-proxy perceived overlap. Human-label Venn still requires human annotations. |
| Write matchers and opportunity definitions for top perceptible clusters | Completed as provisional candidates | `phase4_detector_tier3_matcher_specs.json` contains 10 candidate matcher specs and opportunity definitions, all marked as requiring 200-hit precision validation. |
| Rerun Phases 1-3 on top clusters | Completed as production bounded rerun | `phase4_detector_tier3_candidate_census.csv` reruns provisional matchers over available reference/generation sources. `tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv` reruns the 10 Tier-3 candidates through the OLMo base/SFT/DPO/final exact sequence-mass teacher-forced propensity grid on the 512-document reference package. Larger post-validation denominator expansion remains future work only for sparse features and validated headline claims. |
| Reader-facing Phase 4 conclusion | Completed | `docs/experiments/phase4_detector_discovery_report.md`. |

Primary completion evidence:

- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_metrics.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/ig_10000doc/phase4_detector_ig_summary.json`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/ig_10000doc_hdbscan_gte/phase4_detector_ig_hdbscan_clusters.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_tier3_matcher_specs.json`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_tier3_candidate_census.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_annotation_package.jsonl`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_annotation_readiness.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_pilot_20_each.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each_map.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full_map.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_perceptibility_proxy_labels.jsonl`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_vs_proxy_perceived_venn.svg`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_stage_effects.csv`
- `docs/experiments/phase4_tier3_teacher_forced_addendum.md`

Verification commands run:

- `uv run pytest tests/test_phase4_discover_style_markers.py`
- `uv run ruff check src/slop_sftdiv/cli/phase4_discover_style_markers.py tests/test_phase4_discover_style_markers.py`
- `uv run ty check src/slop_sftdiv/cli/phase4_discover_style_markers.py tests/test_phase4_discover_style_markers.py`
- `uv run ruff check src/slop_sftdiv/cli/phase4_train_detector.py`
- `uv run ty check src/slop_sftdiv/cli/phase4_train_detector.py`
- `uv run ruff check src/slop_sftdiv/cli/phase4_attribute_detector.py`
- `uv run ty check src/slop_sftdiv/cli/phase4_attribute_detector.py`
- `uv run ruff check src/slop_sftdiv/cli/phase4_cluster_attributed_spans.py`
- `uv run ty check src/slop_sftdiv/cli/phase4_cluster_attributed_spans.py`
- `uv run pytest tests/test_generation_text.py`
- `uv run pytest tests/test_propensity.py tests/test_teacher_forced_propensity.py tests/test_phase4_discover_style_markers.py tests/test_generation_text.py`
- `uv run ruff check src/slop_sftdiv/propensity.py src/slop_sftdiv/cli/teacher_forced_propensity.py tests/test_propensity.py tests/test_teacher_forced_propensity.py`
- `uv run ty check src/slop_sftdiv/propensity.py src/slop_sftdiv/cli/teacher_forced_propensity.py tests/test_propensity.py tests/test_teacher_forced_propensity.py`

Conclusion: the detector, cross-model evaluation, integrated-gradients
attribution, GTE/HDBSCAN clustering, candidate matcher writing, lightweight
census, 512-document exact sequence-mass Tier-3 teacher-forced OLMo rerun,
human-annotation package, and proxy perceptibility/Venn analysis were actually
run/prepared and fit under the 24h A100 cap. The remaining original Phase 4
items are completed human labels / human-label Venn analysis, larger
post-validation Tier-3 AF denominator expansion for sparse features, and the
SAE stretch path.
