# Phase 4 Bounded Discovery Report

Phase 4 was originally written as a detector-driven feature discovery loop:
fine-tune a ModernBERT-large human-vs-LLM detector, attribute it with
integrated gradients over at least 10k documents, embed and cluster the
attributed spans, validate clusters with human annotation, then promote the
best clusters into Tier-3 matchers and rerun the earlier pipeline.

That literal plan does not fit the active compute constraint. The training,
attribution, embedding, clustering, and human-validation loop is not a reliable
sub-24h A100 task once setup and validation are included. The completed Phase 4
run is therefore a bounded discovery substitute: it uses no new generation and
no GPU training, reuses existing Phase 2/3 outputs, ranks model-skewed n-gram
spans against SFT references, clusters them with deterministic heuristics, and
emits provisional matcher specs plus a lightweight census over the available
model generations.

## Inputs

The run used:

- OLMo 3 Dolci SFT reference subset: 1,024 documents.
- SmolLM3 SmolTalk2 no_think SFT reference package: 512 documents.
- OLMo 3 base/SFT/DPO/final generated outputs at the existing 512-prompt,
  8-completion, temperature-1 cache, capped to 1,024 rows per source.
- SmolLM3 no_think base/SFT/DPO/final generated outputs at the existing
  512-prompt, 8-completion, temperature-1 production cache, capped to 1,024
  rows per source.

The loaded corpus contains 1,536 reference documents, 8,192 generated documents,
249,363 reference tokens, and 2,846,131 generated tokens. No A100 time was used.

Primary artifacts:

- `artifacts/phase4/bounded_discovery_v1/phase4_scope_decision.md`
- `artifacts/phase4/bounded_discovery_v1/phase4_candidate_spans.csv`
- `artifacts/phase4/bounded_discovery_v1/phase4_candidate_clusters.csv`
- `artifacts/phase4/bounded_discovery_v1/phase4_candidate_examples.jsonl`
- `artifacts/phase4/bounded_discovery_v1/phase4_matcher_specs.json`
- `artifacts/phase4/bounded_discovery_v1/phase4_candidate_census.csv`
- `artifacts/phase4/bounded_discovery_v1/phase4_bounded_discovery_summary.md`

## Method

For each reference and generation document, the bounded detector extracts
1- to 4-gram spans after light normalization. It removes stopword-only spans
and tokenization-only Markdown artifacts, then compares pooled generated text
against pooled SFT reference text.

Each candidate span is ranked by:

- model-side count and document support,
- model per-token rate vs. reference per-token rate,
- smoothed lift vs. reference,
- a score combining log lift and model document support,
- a deterministic cluster label and auto-perceptibility proxy.

This is not a neural detector and it is not attribution from a trained
classifier. It is an interpretable discovery pass designed to find concrete
candidate surface markers that are worth validating.

## Primary Results

The run retained 100 candidate spans from 14,511 candidates passing initial
support filters and grouped them into 38 heuristic clusters. The strongest
readable clusters were:

| Cluster | Plain description | Top candidate | Notes |
|---|---|---|---|
| `offer_examples` | Offer-of-examples framing | `here are` | Very strong in SmolLM3 SFT/DPO/final; weak in OLMo. |
| `politeness_thanks` | Politeness/thanks formula | `thank you` | Mostly a SmolLM3 base spike; not persistent after preference/final. |
| `explanatory_transition` | Explanatory transition phrase | `such as` | Common in both ladders, especially SmolLM3. |
| `followup_question_prompt` | Conversational follow-up prompt | `do you have any` | Strong SmolLM3 base signature, much lower after SFT/DPO/final. |
| `prescriptive_instruction` | Prescriptive/self-instruction style | `i should` | Strong in base generations, almost disappears after alignment. |
| `conversational_disclaimer` | Conversational uncertainty/affirmation | `i'm glad` | Mostly SmolLM3 aligned-model style. |
| `stock_helpful_closure` | Helpful closing formula | `i hope this` | SmolLM3 DPO/final/SFT marker; low in OLMo. |
| `followup_offer` | Follow-up assistance offer | `me know if` | Rises in DPO/final, especially SmolLM3. |
| `summary_closure` | Summary or conclusion closure | `overall` | Present in both ladders; stronger in SmolLM3 aligned outputs. |

The provisional matcher specs select one representative per cluster where
possible. They are intentionally marked as candidates:
`requires_200_hit_manual_validation_before_core_claim`.

## Model Progression Signals

The bounded census is not a teacher-forced amplification factor. It is a
generated-output rate table for newly discovered candidates. Still, it shows
useful progression patterns:

- `here are`: OLMo stays low (`0.035` base to `0.077` final per 1k tokens);
  SmolLM3 rises sharply from base `0.196` to DPO `1.635` and final `1.621`.
- `such as`: OLMo remains moderate (`0.391` base, `0.505` DPO, `0.403` final);
  SmolLM3 is much higher (`0.721` base, `2.335` SFT, `1.710` DPO, `1.809`
  final).
- `do you have any`: almost absent in OLMo; high in SmolLM3 base (`0.988`) and
  much lower by DPO/final (`0.097`/`0.032`).
- `i should`: high in both base models (`0.648` OLMo, `0.393` SmolLM3) and
  nearly eliminated after alignment.
- `i'm glad`: absent in OLMo except a tiny SFT trace; present in SmolLM3, with
  SFT highest (`0.600`) and DPO/final still elevated (`0.395`/`0.353`).
- `i hope this`: low in OLMo; present in SmolLM3 SFT/DPO/final
  (`0.245`/`0.273`/`0.239`).
- `me know if`: rises from OLMo SFT `0.023` to DPO/final `0.092`/`0.080`, and
  from SmolLM3 base `0.064` to DPO/final `0.335`/`0.291`.
- `overall`: present in both ladders, with SmolLM3 SFT/final higher than OLMo.

The most important conclusion is that many discovered candidates are not
universal "post-training slop." They split into ladder-specific signatures:
SmolLM3 has strong conversational examples, thanks, follow-up prompts, and
helpful closers; OLMo has weaker rates for those families in the tested cache.
Some base-only candidates (`i should`, parts of the follow-up-question family)
look like generation/source artifacts or task-register artifacts that alignment
reduces rather than amplifies.

## What Can Be Promoted

Nothing from this bounded pass should be promoted directly into Tier 3 core
claims yet. The credible next promotion candidates are:

- `offer_examples`: validate `here are`, `here are some`, `here are a few`.
- `followup_offer`: validate `let me know`, `me know if`.
- `stock_helpful_closure`: validate `i hope this`, `hope this helps`.
- `conversational_disclaimer`: validate `i'm glad`, `i'm not sure`.
- `summary_closure`: validate `overall`.
- `explanatory_transition`: validate whether `such as` is a real slop marker or
  a neutral explanatory control.

Promotion still requires the original feature contract: a matcher, an
opportunity definition, and a 200-hit precision estimate with target precision
at least 0.9.

## Limits

This run does not complete the literal ModernBERT/integrated-gradients/HDBSCAN
workflow. It also does not perform Shaib-taxonomy human annotation or produce a
human-perceived-vs-detector-only Venn diagram. Those are deferred because they
are outside the sub-24h A100 scope and, for human annotation, outside pure
compute entirely.

The completed bounded Phase 4 result is a practical discovery layer: candidate
features, cluster labels, examples, provisional matcher specs, and a generated
census over the existing OLMo/SmolLM3 caches.
