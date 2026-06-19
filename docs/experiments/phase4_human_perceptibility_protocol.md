# Phase 4 Human-Perceptibility Annotation Protocol

Date: 2026-06-18

Purpose: define the human-labeling protocol for promoting Phase 4
detector-discovered Tier-3 candidates from "machine-detectable style families"
to "human-perceived slop style" claims. This document does not claim that
human annotation is complete; it specifies how to complete and summarize it
using the existing 1,000-example package.

## Annotation Package

Primary package:

- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_annotation_package.jsonl`

The package contains 1,000 rows: 100 examples for each of the 10 provisional
Tier-3 detector candidates.

Readiness audit and pilot sheet:

- `docs/experiments/phase4_human_annotation_readiness.md`
- `docs/experiments/phase4_human_annotation_codebook.md`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_annotation_readiness.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_pilot_20_each.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each_map.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full_map.csv`

The readiness audit verifies the package schema, row balance, blank label
state, and per-candidate source/record coverage. The codebook gives
candidate-specific yes/no/context-dependent decision boundaries. The pilot
sheet is a deterministic 20-row-per-candidate subset for an author-pilot or
annotation dry run; it is not a substitute for the full 1,000-row labeling
plan. The blind pilot sheet randomizes those same rows and hides feature,
candidate, source, and record identifiers from annotators; the map should stay
separate until after adjudication. The blind full-package sheet applies the
same redaction and randomization to all 1,000 rows and is the annotator-facing
input for the recommended full labeling plan.

For handoff, run:

```bash
uv run slop-materialize-phase4-human-handoff --root /home/user/slop
```

This writes
`artifacts/phase4/modernbert_detector_combined_v2_clean/human_annotation_handoff/`.
Distribute only the `annotator/` subdirectory to labelers. The
`coordinator/private_maps/` subdirectory contains maps and source identifiers
needed only after independent labels have been collected.
For the two-annotator plan, send `annotator/annotator_a/` to the first
annotator and `annotator/annotator_b/` to the second; the files are identical
except for filenames, so annotators can return independent filled sheets
without overwriting each other.

Current row fields:

| Field | Meaning |
|---|---|
| `annotation_id` | Stable row identifier, grouped by candidate feature. |
| `feature` | Machine-readable Tier-3 candidate feature name. |
| `cluster` | Human-readable detector cluster family. |
| `candidate_label` | Short candidate interpretation. |
| `source` | Generation/reference source for the snippet. |
| `record_id` | Underlying record identifier. |
| `matched_text` | Span that triggered the candidate matcher. |
| `snippet` | Local context shown to annotators. |
| `human_perceived_slop` | Blank field to label. |
| `shaib_taxonomy_label` | Blank field to label when applicable. |
| `notes` | Optional free-text note. |

## Candidate Features

| Feature | Candidate label | Default paper status before human labels |
|---|---|---|
| `phase4_ig_conversation_greeting` | Conversational greeting or acknowledgment | Candidate; likely human-perceptible only when gratuitous or formulaic. |
| `phase4_ig_process_framing` | Explicit assistant process framing | Candidate; likely human-perceptible when it adds procedural framing without need. |
| `phase4_ig_additive_transition` | Additive/explanatory transition | Candidate; likely human-perceptible when transition density feels formulaic. |
| `phase4_ig_prescriptive_instruction` | Prescriptive instruction style | Candidate; likely human-perceptible when it overuses advice imperatives. |
| `phase4_ig_followup_offer` | Follow-up offer/friendly closure | Candidate; likely human-perceptible when it adds unsolicited assistance/closure. |
| `phase4_ig_response_constraint` | Response-constraint wording | Context-dependent; may reflect prompt/instruction wording. |
| `phase4_ig_code_boilerplate` | Code/task boilerplate | Candidate domain artifact until humans judge otherwise. |
| `phase4_ig_quantity_time_recipe` | Quantity/time/recipe guidance | Context-dependent; often task/domain-specific. |
| `phase4_ig_benefit_intensifier` | Benefit/intensifier framing | Candidate; likely human-perceptible when praise/intensification is generic. |
| `phase4_ig_careful_reasoning` | Careful reasoning instruction | Candidate; likely human-perceptible when metacognitive caution is formulaic. |

## Annotator Task

Annotators see `matched_text` and `snippet`. They should answer:

> Does this matched span contribute to a recognizable "AI slop style" effect
> in this local context?

Annotators should judge style, not factual correctness. A span can be
detector-positive but not human-perceived slop if it is demanded by the task,
domain, or prompt.

## Required Labels

### `human_perceived_slop`

Allowed values:

| Value | Meaning |
|---|---|
| `yes` | The matched span contributes to a recognizable slop-style effect in context. |
| `no` | The matched span is natural, task-required, or not stylistically salient. |
| `context_dependent` | The span can be slop-like, but this row is ambiguous without broader prompt/output context. |
| `unclear` | The annotator cannot decide from the snippet. |

Blank values mean the row has not been labeled.

### `shaib_taxonomy_label`

Allowed values:

| Value | Meaning |
|---|---|
| `tone` | Overfriendly, sycophantic, generic, or assistant-persona style. |
| `structure` | Formulaic organization, openings/closings, list-like scaffolding, or generic transitions. |
| `coherence` | Reasoning/framing that feels mechanically sequenced or over-explained. |
| `relevance` | Content that feels unnecessary, off-target, or gratuitous for the task. |
| `density` | Overcompressed buzzwords, overpacked adjectives, or generic intensification. |
| `factuality` | Unsupported factual confidence or hallucination-like framing. |
| `bias` | Normative, stereotyped, or value-loaded framing. |
| `none` | Human-perceived slop is `no`; no taxonomy category applies. |
| `domain_artifact` | The span reflects task/domain conventions rather than style slop. |
| `unclear` | Taxonomy assignment is unclear. |

For `human_perceived_slop=yes`, annotators should choose the closest Shaib
taxonomy label rather than `none`. For `context_dependent`, use the best label
if one is apparent, otherwise `unclear`.

## Minimum Labeling Plan

Recommended submission-minimum plan:

1. Two independent annotators label all 1,000 examples.
2. Run two-annotator agreement before discussion or adjudication.
3. Disagreements on `human_perceived_slop` are adjudicated by a third pass.
4. Report per-feature human-perceived rate, context-dependent rate, and
   unresolved rate.
5. Promote a candidate to "human-perceived slop family" only if at least 50%
   of adjudicated rows are `yes` or at least 65% are `yes` plus
   `context_dependent`.
6. Keep candidates below that threshold as detector-only or domain/context
   artifacts.

If only a pilot is possible, label 20 examples per candidate and present it as
an author-pilot validation, not as completed human perceptibility.

The prepared pilot sheet already contains this 20-per-candidate subset:

```bash
uv run slop-audit-phase4-human-package
```

This command regenerates the readiness audit and pilot CSV from the current
annotation package, plus blinded pilot/full-package CSVs and unblinding maps.
Annotators should read
`docs/experiments/phase4_human_annotation_codebook.md` before labeling the
pilot or full package.

## Adjudication Rules

- Prefer `no` when the phrase is plainly required by the prompt or domain.
- Prefer `context_dependent` when the snippet is too short to know whether the
  phrase is gratuitous.
- Use `domain_artifact` when the matched text is code boilerplate, recipe
  units, mathematical setup, or other task-conventional language.
- Do not mark a span `yes` merely because it is common in LLM output; it must
  be locally perceptible as stylistic excess, formula, or generic assistant
  prose.
- Keep notes short and concrete: mention the reason, not a general impression.
- For blind labeling, distribute only the blind pilot or blind full-package
  CSV. Do not share the feature names, candidate labels, source, record IDs,
  or map file until after independent labels are collected.

## Paper Outputs

After blind pilot or full-package annotation, import the filled blind CSV back
into canonical JSONL rows. For a pilot:

```bash
uv run slop-import-phase4-blind-labels \
  --blind-labels artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each.csv \
  --blind-map artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each_map.csv \
  --output artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_labeled.jsonl \
  --strict
```

For the full 1,000-row package, use
`phase4_human_perceptibility_blind_full.csv` and
`phase4_human_perceptibility_blind_full_map.csv` instead, writing to a
full-package labeled JSONL path:

```bash
uv run slop-import-phase4-blind-labels \
  --blind-labels artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full.csv \
  --blind-map artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full_map.csv \
  --output artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full_labeled.jsonl \
  --strict
```

If two independent blind pilot or full-package sheets exist, summarize
agreement before adjudication:

```bash
uv run slop-summarize-phase4-label-agreement \
  --annotator-a artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each_annotator_a.csv \
  --annotator-b artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each_annotator_b.csv \
  --blind-map artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each_map.csv \
  --output artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_pilot_agreement.csv \
  --disagreements-output artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_pilot_disagreements.csv \
  --summary-output docs/experiments/phase4_human_perceptibility_pilot_agreement.md \
  --strict
```

Fill the adjudicated columns in
`phase4_human_perceptibility_pilot_disagreements.csv`, then merge consensus
and adjudicated rows:

The adjudication applier treats the disagreement CSV as an integrity-checked
input: every adjudication row must reference a real `blind_id` that actually
disagreed between annotators. Typo IDs and stale rows for consensus examples are rejected before canonical JSONL is written.

```bash
uv run slop-apply-phase4-label-adjudication \
  --annotator-a artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each_annotator_a.csv \
  --annotator-b artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each_annotator_b.csv \
  --blind-map artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each_map.csv \
  --adjudications artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_pilot_disagreements.csv \
  --output artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_adjudicated.jsonl \
  --strict
```

Then summarize the adjudicated pilot:

```bash
uv run slop-summarize-phase4-human-labels \
  --input artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_adjudicated.jsonl \
  --output artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_pilot_summary.csv \
  --summary-output docs/experiments/phase4_human_perceptibility_pilot_summary.md
```

After full-package annotation, run:

```bash
uv run slop-summarize-phase4-human-labels \
  --input artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full_adjudicated.jsonl \
  --output artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_summary.csv \
  --summary-output docs/experiments/phase4_human_perceptibility_summary.md
```

The summary should drive:

- Detector-vs-human-perceived Venn table.
- Tier-3 candidate promotion/demotion decisions.
- Any claim that detector-discovered families are human-perceived slop rather
  than only machine-detectable style markers.
