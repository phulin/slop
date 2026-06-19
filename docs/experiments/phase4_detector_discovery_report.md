# Phase 4 Detector Discovery Report

Phase 4 was completed as an actual detector-driven discovery pass, not just the
earlier n-gram fallback. The run fine-tuned ModernBERT-large, evaluated
cross-model transfer, ran integrated-gradients attribution over 10,000 generated
documents, embedded attributed spans with GTE-large, clustered them with
HDBSCAN, and wrote provisional Tier-3 matcher specs plus a lightweight census.

Human annotation and the SAE stretch path are not complete. Those require human
labels / additional methods work and are called out separately below.

## Detector Training

Two ModernBERT-large detectors were attempted:

1. `modernbert_detector_attempt_v1`: HAP-E-only training. It trained on
   HAP-E human vs. GPT-4o/Llama-3-8B-Instruct and evaluated on held-out
   GPT-4o-mini/Llama-3-70B-Instruct plus OLMo/SmolLM3 generations.
2. `modernbert_detector_combined_v2_clean`: HAP-E plus disjoint slices of our
   OLMo/SmolLM3 generations and SFT references, after stripping chat-role
   marker lines from generated text.

The cleaned combined detector is the primary Phase 4 detector artifact:

- Checkpoint: `artifacts/phase4/modernbert_detector_combined_v2_clean/checkpoint`
- Summary: `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_summary.json`
- Metrics: `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_metrics.csv`
- Training log: `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_train_log.csv`

It trained on 3,328 examples: 2,048 HAP-E examples plus 1,280 external
OLMo/SmolLM3 reference/generation examples. It evaluated on 1,024 held-out
HAP-E examples and 2,560 external examples.

Held-out HAP-E performance stayed strong:

- Llama-3-70B-Instruct LLM texts: accuracy `0.977`, mean LLM probability
  `0.962`.
- GPT-4o-mini LLM texts: accuracy `0.984`, mean LLM probability `0.975`.
- Held-out human counterparts: accuracy `1.000` and `0.996`, mean LLM
  probability `0.015` and `0.010`.

External checkpoint evaluation is mixed but useful:

- SmolLM3 generation transfer is strong: base `0.914`, SFT `0.977`, DPO
  `0.996`, final `0.992` accuracy on LLM-labeled generations.
- SmolLM3 SFT references are correctly reference-like: accuracy `1.000`, mean
  LLM probability `0.00035`.
- OLMo generation transfer is weaker after role-marker cleaning: base `0.469`,
  SFT `0.297`, DPO `0.363`, final `0.383` accuracy on LLM-labeled generations.
- OLMo Dolci SFT references remain partly detector-positive: `0.676` accuracy
  on reference labels, with a `0.351` mean LLM probability and `0.324`
  positive prediction rate. This is a calibration caveat and probably reflects
  the synthetic/assistant-like nature of Dolci SFT.

The key detector conclusion is that HAP-E style transfer alone is insufficient
for OLMo/SmolLM3. Adding our generation/reference slices makes the detector
usable for attribution, but also means discovered features must be treated as
candidate style markers rather than clean human-vs-LLM universals.

## Attribution And Clustering

The cleaned combined detector was attributed with embedding-level integrated
gradients:

- Attribution artifact: `artifacts/phase4/modernbert_detector_combined_v2_clean/ig_10000doc/`
- Documents attributed: 10,000.
- IG steps: 8.
- Max length: 384.
- Document-span rows: 77,013.
- Unique spans: 52,109.

The top 5,000 attributed spans were embedded with GTE-large and clustered with
HDBSCAN:

- Cluster artifact:
  `artifacts/phase4/modernbert_detector_combined_v2_clean/ig_10000doc_hdbscan_gte/`
- Embedding model: `thenlper/gte-large`.
- Non-noise HDBSCAN clusters: 56.
- Noise spans: 3,819 of the 5,000 selected spans.

The highest-support HDBSCAN clusters mix real style markers with topic/code
artifacts. The useful perceptible style clusters include:

| Cluster | Examples | Interpretation |
|---|---|---|
| `cluster_54`, `cluster_47`, `cluster_44` | `hey!`, `hi there`, `sure`, `certainly`, `great question` | Conversational greeting / acknowledgment. |
| `cluster_41`, `cluster_38` | `step by step`, `problem step by step`, `to solve` | Explicit process framing. |
| `cluster_51`, `cluster_50`, `cluster_52`, `cluster_32` | `also`, `additionally`, `moreover`, `however`, `therefore`, `thus`, `typically` | Additive and explanatory transitions. |
| `cluster_36`, `cluster_43`, `cluster_29`, `cluster_35` | `should`, `need to`, `consider`, `ensure`, `make sure` | Prescriptive instruction style. |
| `cluster_49`, `cluster_55`, `cluster_48`, `cluster_31` | `you can`, `if you want`, `happy to help`, `i'm glad` | Follow-up offer / friendly assistant framing. |
| `cluster_34` | `your response`, `your answer`, `must` | Response constraint wording. |
| `cluster_16`, `cluster_39` | `significantly enhance`, `greatly enhance`, `greatly benefit` | Benefit/intensifier framing. |
| `cluster_22` | `think carefully`, `let's think`, `carefully first` | Careful reasoning instruction. |

The main artifacts are:

- `phase4_detector_ig_doc_spans.csv`
- `phase4_detector_ig_span_summary.csv`
- `phase4_detector_ig_clusters.csv`
- `phase4_detector_ig_hdbscan_members.csv`
- `phase4_detector_ig_hdbscan_clusters.csv`
- `phase4_detector_ig_hdbscan_summary.json`

## Tier-3 Candidate Specs

Ten provisional matcher specs were written from the detector/IG/HDBSCAN
clusters:

- `phase4_ig_conversation_greeting`
- `phase4_ig_process_framing`
- `phase4_ig_additive_transition`
- `phase4_ig_prescriptive_instruction`
- `phase4_ig_followup_offer`
- `phase4_ig_response_constraint`
- `phase4_ig_code_boilerplate`
- `phase4_ig_quantity_time_recipe`
- `phase4_ig_benefit_intensifier`
- `phase4_ig_careful_reasoning`

Spec file:
`artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_tier3_matcher_specs.json`

Every spec is marked:
`requires_200_hit_manual_validation_before_core_claim`.

## Lightweight Rerun Census

The provisional specs were rerun over the available references and generations:

`artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_tier3_candidate_census.csv`

Main generated-output patterns:

- Conversation greetings are much more SmolLM3-like than OLMo-like. SmolLM3
  base/SFT/DPO/final are `2.239`, `1.846`, `1.199`, `1.191` per 1k tokens;
  OLMo base/SFT/DPO/final are `0.433`, `0.238`, `0.259`, `0.288`.
- Process framing is more OLMo-like and rises into OLMo final:
  OLMo base/SFT/DPO/final are `2.486`, `1.846`, `2.543`, `3.550`; SmolLM3 is
  lower at `0.817`, `0.839`, `0.732`, `0.694`.
- Additive transitions are high in both ladders, with a strong SmolLM3 SFT
  spike: OLMo final `3.648`, SmolLM3 SFT `6.657`, SmolLM3 final `4.569`.
- Prescriptive instruction is high everywhere but especially base-heavy for
  OLMo: OLMo base `8.528`, final `3.838`; SmolLM3 base `4.751`, final `4.516`.
- Follow-up offers are strongly SmolLM3-like: SmolLM3 SFT/DPO/final
  `4.739`, `3.054`, `3.146`; OLMo SFT/DPO/final `0.661`, `0.925`, `0.939`.
- Response-constraint wording is OLMo-like: OLMo `0.235`-`0.487`, SmolLM3
  near zero after base.
- Code boilerplate is OLMo/code-source heavy and should be treated as a
  domain/source artifact unless validated separately.
- Careful reasoning instructions are base-heavy and attenuate after alignment:
  OLMo base `0.175` vs. DPO/final `0.020`/`0.040`; SmolLM3 near zero.

## Bounded Teacher-Forced Rerun

The top 10 Phase 4 candidate features were also wired into the existing
teacher-forced propensity harness and scored on the OLMo 3 Instruct checkpoint
ladder:

- Model stages: base, SFT, DPO, final.
- Reference package: 128 Dolci SFT target responses.
- Scoring mode: exact sequence probability mass. An earlier first-token pass is
  retained as a scoping artifact.
- Features: 10 Phase 4 candidates plus `neutral_common_controls`.
- Main artifact:
  `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact/olmo3_phase4_tier3_128_exact_sequence_stage_grid.csv`
- Paired stage effects:
  `artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact/olmo3_phase4_tier3_128_exact_sequence_stage_effects.csv`
- Reader addendum:
  `docs/experiments/phase4_tier3_teacher_forced_addendum.md`

Main teacher-forced results:

| Feature | Ref inits | Raw AF base | Raw AF SFT | Raw AF DPO | Raw AF final |
|---|---:|---:|---:|---:|---:|
| Follow-up offer | 4 | 4.120 | 25.830 | 26.870 | 26.450 |
| Process framing | 30 | 1.583 | 1.909 | 1.940 | 2.162 |
| Additive transition | 55 | 0.585 | 0.706 | 0.895 | 1.030 |
| Prescriptive instruction | 20 | 0.312 | 0.664 | 0.663 | 0.781 |
| Response constraint | 2 | 1.135 | 2.287 | 2.600 | 2.837 |
| Code boilerplate | 60 | 0.407 | 0.467 | 0.461 | 0.496 |
| Conversation greeting | 2 | 0.200 | 0.146 | 0.150 | 0.133 |

Three candidates had zero reference initiations in this 128-document slice:
benefit intensifier, careful reasoning, and quantity/time/recipe. Their
probability masses are still in the stage grid, but AF is not interpretable
without a larger or targeted denominator.

The strongest provisional Tier-3 teacher-forced signal is follow-up-offer
language. The most stable denominator-supported signal is process framing,
which has raw AF above 1 at every OLMo stage and rises into final. Additive
transitions approach reference-matched raw AF by final. Conversation greetings
do not show OLMo teacher-forced amplification in this slice, matching the
free-running census where greetings were much more SmolLM3-like than OLMo-like.

## Conclusions

Phase 4 did find new candidate style families beyond the original Tier-1 set,
but it also showed why detector discovery needs validation. The detector
surfaces three different kinds of features:

- Human-perceptible assistant style: greetings, follow-up offers, process
  framing, additive transitions, prescriptive instructions, helpful closure.
- Register/domain artifacts: code boilerplate, math/problem-solving phrasing,
  recipe/time quantities, topic words.
- Detector calibration artifacts: OLMo Dolci SFT references are partly
  detector-positive, so OLMo results need extra caution.

The strongest candidate additions to Tier 3 are conversation greetings,
follow-up offers, process framing, additive transitions, prescriptive
instructions, benefit/intensifier framing, and careful-reasoning instructions.
After the bounded teacher-forced rerun, the strongest OLMo-side candidates are
follow-up offers, process framing, additive transitions, prescriptive
instruction, and response-constraint wording. They are still not final claims
until 200-hit precision validation, human perceptibility annotation, and
larger/exact AF denominator support are done.

## Remaining Work

Human-validation package prepared:

- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_annotation_package.jsonl`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_annotation_package_summary.csv`
- `docs/experiments/phase4_human_perceptibility_protocol.md`
- `docs/experiments/phase4_human_perceptibility_summary.md`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_summary.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_perceptibility_proxy_labels.jsonl`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_perceptibility_proxy_summary.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_vs_proxy_perceived_venn.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_detector_vs_proxy_perceived_venn.svg`

The package contains 1,000 examples total: 100 examples for each of the 10
detector-derived candidate features. The label fields are intentionally blank
because no human annotator has completed them.

A separate proxy validation pass maps the candidates to the Shaib et al.
7-way collapsed taxonomy: Density, Relevance, Factuality, Bias, Structure,
Coherence, and Tone. This is rule-based Codex triage, not human annotation.
It marks 7 detector-derived features as likely human-perceptible slop,
2 as context-dependent, and 1 as detector-only/domain artifact:

| Proxy region | Feature count | Features |
|---|---:|---|
| Detector and proxy-perceived | 7 | conversation greeting; process framing; additive transition; prescriptive instruction; follow-up offer; benefit intensifier; careful reasoning |
| Context-dependent proxy | 2 | response constraint; quantity/time/recipe guidance |
| Detector-only proxy | 1 | code boilerplate |

Not complete:

- Human-perceptibility validation against the Shaib et al. taxonomy.
- 100 human annotations per discovered cluster.
- Detector-vs-human-perceived Venn diagram.
- SAE attribution stretch path.
- Larger post-validation teacher-forced Phase 1-3 rerun for validated Tier-3
  features.

The parts above were not deferred because of A100 time. They require human
annotation, post-validation denominator expansion, or a separate stretch-methods
pass. The detector, integrated gradients, GTE-large embedding, HDBSCAN
clustering, candidate matcher writing, lightweight rerun census, and bounded
exact sequence-mass Tier-3 teacher-forced OLMo rerun were all actually run.
