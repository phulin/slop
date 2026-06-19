# Phase 4 Human Annotation Readiness

Input package: `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_annotation_package.jsonl`
Machine-readable audit: `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_annotation_readiness.csv`
Pilot labeling sheet: `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_pilot_20_each.csv`
Blind pilot labeling sheet: `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each.csv`
Blind pilot ID map: `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each_map.csv`
Blind full-package labeling sheet: `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full.csv`
Blind full-package ID map: `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full_map.csv`
Safe handoff bundle: `artifacts/phase4/modernbert_detector_combined_v2_clean/human_annotation_handoff`

Readiness status: `ready_for_labeling`.
Package rows: `1000` across `10` candidate features.
Pilot sheet size: `20` rows per feature.
Blind pilot randomization seed: `17`.

| Feature | Rows | Schema | Labeled | Sources | Records | Candidate label |
|---|---:|---|---:|---:|---:|---|
| phase4_ig_additive_transition | 100/100 | ok | 0 | 1 | 24 | Additive/explanatory transition |
| phase4_ig_benefit_intensifier | 100/100 | ok | 0 | 1 | 43 | Benefit/intensifier framing |
| phase4_ig_careful_reasoning | 100/100 | ok | 0 | 1 | 45 | Careful reasoning instruction |
| phase4_ig_code_boilerplate | 100/100 | ok | 0 | 1 | 23 | Code boilerplate and IO framing |
| phase4_ig_conversation_greeting | 100/100 | ok | 0 | 1 | 49 | Conversational greeting or acknowledgment |
| phase4_ig_followup_offer | 100/100 | ok | 0 | 1 | 32 | Follow-up assistance offer |
| phase4_ig_prescriptive_instruction | 100/100 | ok | 0 | 1 | 20 | Prescriptive instruction or requirement |
| phase4_ig_process_framing | 100/100 | ok | 0 | 1 | 21 | Explicit assistant process framing |
| phase4_ig_quantity_time_recipe | 100/100 | ok | 0 | 1 | 36 | Quantified time/recipe guidance |
| phase4_ig_response_constraint | 100/100 | ok | 0 | 1 | 36 | Response/answer constraint wording |

No package-balance or schema warnings were found. The package remains
unlabeled and should still be treated as detector-only evidence until
human labels are collected and summarized. Use the blind pilot sheet
for annotator-facing dry runs, or the blind full-package sheet for
the recommended two-annotator labeling plan; keep ID maps separate
until after adjudication. The handoff bundle materializes this
separation: share only its `annotator/` directory with annotators
and keep `coordinator/private_maps/` private.
