# Phase 4 Human Annotation Codebook

Date: 2026-06-18

Purpose: give annotators concrete decision rules for the Phase 4
human-perceptibility package. This codebook complements
`docs/experiments/phase4_human_perceptibility_protocol.md`; it does not report
completed human labels.

Primary files:

- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_annotation_package.jsonl`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_pilot_20_each.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_pilot_20_each_map.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_blind_full_map.csv`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/phase4_human_perceptibility_summary.csv`

## Core Question

Label the matched span in context:

> Does this matched span contribute to a recognizable AI slop-style effect in
> this local snippet?

Judge style, not correctness. A wrong answer is not automatically slop. A
correct answer can still contain a formulaic assistant-style marker.

Use `human_perceived_slop=yes` only when the matched text is locally
perceptible as stylistic excess, formulaic assistant prose, unnecessary
framing, generic positivity, or gratuitous closure.

Use `human_perceived_slop=no` when the matched text is natural for the task,
required by the prompt/domain, or not stylistically salient.

Use `human_perceived_slop=context_dependent` when the span could be slop-like
but the snippet is too short to decide whether it is gratuitous.

Use `human_perceived_slop=unclear` only when the annotator cannot interpret
the snippet or matched text.

## Annotator Fill Rules

Annotators should edit only `human_perceived_slop`, `shaib_taxonomy_label`,
and `notes`. Do not edit `blind_id`, `matched_text`, or `snippet`; those fields
are required for import, agreement, adjudication, and unblinding. Leave
`shaib_taxonomy_label` blank only when `human_perceived_slop` is still blank.
For `human_perceived_slop=no`, use `shaib_taxonomy_label=none` unless the row
is better explained as `domain_artifact` or `unclear`.

## Taxonomy Shortcut

Use these labels for `shaib_taxonomy_label`:

| Label | Use when |
|---|---|
| `tone` | The issue is overfriendly, generic, sycophantic, or assistant-persona language. |
| `structure` | The issue is formulaic opening, closing, transition, or scaffolding. |
| `coherence` | The issue is mechanical reasoning flow, over-explaining, or process narration. |
| `relevance` | The issue is unnecessary, off-target, or gratuitous content. |
| `density` | The issue is overpacked adjectives, buzzwords, or generic intensification. |
| `domain_artifact` | The span reflects code, math, recipe, task, or prompt conventions. |
| `none` | Human-perceived slop is `no`. |
| `unclear` | The taxonomy is not decidable from the snippet. |

## Candidate-Specific Rules

| Feature | Yes when | No when | Default taxonomy |
|---|---|---|---|
| `phase4_ig_conversation_greeting` | Greeting or acknowledgment is formulaic, gratuitous, or unrelated to the task. | The task is conversational and the greeting is a normal response move. | `tone` or `structure` |
| `phase4_ig_process_framing` | The model narrates its own plan, steps, or intent without the user needing that framing. | The prompt explicitly asks for process, explanation, or reasoning steps. | `coherence` |
| `phase4_ig_additive_transition` | Transitions such as "also", "however", or "therefore" create repetitive formulaic flow. | The transition is necessary for logic, contrast, or mathematical derivation. | `structure` |
| `phase4_ig_prescriptive_instruction` | Advice imperatives feel generic, moralizing, or over-prescriptive for the task. | The user asks for instructions, safety rules, recipe steps, or code directions. | `relevance` or `tone` |
| `phase4_ig_followup_offer` | A closing offer adds unsolicited assistance or friendly closure after the answer is complete. | The user requested iterative help or the offer is necessary to complete a workflow. | `tone` or `structure` |
| `phase4_ig_response_constraint` | The model repeats constraints, refusal framing, or answer-format limitations unnecessarily. | The matched text is copied from a prompt requirement or needed for safety/compliance. | `relevance` or `domain_artifact` |
| `phase4_ig_code_boilerplate` | Boilerplate framing surrounds code without adding useful task-specific content. | The matched text is actual code syntax, required I/O framing, or conventional comments. | `domain_artifact` |
| `phase4_ig_quantity_time_recipe` | Quantities or timing details are generic filler or falsely precise. | The task is a recipe, schedule, unit conversion, or quantitative problem. | `domain_artifact` or `relevance` |
| `phase4_ig_benefit_intensifier` | Benefit language is generic praise, intensification, or marketing-like uplift. | The matched phrase directly describes a requested benefit or evaluation criterion. | `density` or `tone` |
| `phase4_ig_careful_reasoning` | Caution or reasoning language is formulaic metacommentary rather than needed analysis. | The prompt asks for careful reasoning, proof, verification, or uncertainty handling. | `coherence` |

## Tie-Breakers

- Prefer `no` over `yes` when the matched text is required by the task.
- Prefer `domain_artifact` for code, recipes, math derivations, URLs, and
  structured I/O unless the phrasing is clearly generic assistant filler.
- Prefer `context_dependent` when the snippet lacks the prompt or enough
  surrounding output to judge gratuitousness.
- Do not mark a row `yes` only because the phrase is common in model outputs.
  The span must be locally perceptible as a style problem.
- Keep notes short: name the reason, such as `task-required`, `gratuitous
  closure`, `formulaic transition`, or `prompt constraint`.

## Pilot Procedure

For the 20-per-candidate pilot sheet:

1. Two annotators independently label every pilot row.
2. Prefer the blind pilot CSV for annotator-facing labeling; keep the map file
   separate until after independent labels are collected.
3. Import or summarize the returned blind sheets only after both annotators
   have completed their independent labels.
4. Run `uv run slop-summarize-phase4-label-agreement` to produce per-feature
   agreement rates, Cohen's kappa, and disagreement rows with blank
   adjudication columns.
5. Compare `human_perceived_slop` before taxonomy labels.
6. Discuss rows where one annotator used `yes` and the other used `no`.
7. Fill `adjudicated_human_perceived_slop`,
   `adjudicated_shaib_taxonomy_label`, and `adjudication_notes`, then run
   `uv run slop-apply-phase4-label-adjudication`.
8. Update this codebook if the disagreement reveals a missing rule.
9. Only after the pilot is stable, label the full 1,000-row package.

For the full 1,000-row labeling plan, use the same procedure with
`phase4_human_perceptibility_blind_full.csv` and keep
`phase4_human_perceptibility_blind_full_map.csv` separate until import,
agreement analysis, and adjudication.

The pilot should be reported as a pilot if the full package remains unlabeled.
It should not be used to claim that a detector candidate is human-perceived
slop.
