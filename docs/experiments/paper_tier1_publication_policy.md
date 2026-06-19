# Tier-1 Publication-Use Policy

Date: 2026-06-18

Purpose: separate the current Tier-1 feature uses that are safe for the paper
from uses that remain blocked by manual precision validation. This policy does
not change any measurements; it controls wording and table placement until
more labels are collected.

## Policy Summary

| Feature | Current validation state | Allowed paper use | Disallowed paper use |
|---|---|---|---|
| `contrastive_negation` | Passes interim precision gate: 61 labels, precision 0.934, ambiguous rate 0.000 | Corpus-rate context, paired DPO context, feature-level propensity/output analysis if otherwise supported | Universal claims about contrastive style beyond sampled ladders |
| `stock_openers` | Passes interim precision gate: 68 labels, precision 0.985, ambiguous rate 0.000 | Corpus-rate context and stock-opener examples with bounded sample caveat | Treating all opener/closer behavior as validated |
| `stock_closers` | Passes interim precision gate: 50 labels, precision 0.960, ambiguous rate 0.000 | Corpus-rate context and stock-closer examples with bounded sample caveat | Treating the derived pooled opener/closer metric as independently validated |
| `slop_lexicon` | Passes interim span/item precision gate: 50 labels, precision 1.000, ambiguous rate 0.000 | Historical frozen lexicon index for bounded OLMo/SmolLM3 model-side propensity, generation, compounding, and corpus-rate context with item-level caveats | Broad semantic claims that every lexicon hit is inherently slop-like |
| `rule_of_three_approx` | Demoted for independent claims: 50 labels, 28 exact, 1 partial, 19 false positives, 2 ambiguous; precision 0.560 | Exploratory coordinated-triple surface only, with parser/precision caveat | Publication-grade claims about true rhetorical triples |
| `stock_openers_closers` | Derived pooled view; direct bounded queue has zero retained hits | Derived pooled convenience metric when decomposed into opener/closer components | Independent matcher claims |
| `eqbench_slop_score` | Aggregate benchmark bridge, not a manual-span precision target | Descriptive benchmark comparability and aggregate diagnostic | Causal estimand or substitute for feature-level AF |
| `slop_lexicon_v2_candidate` | Exploratory candidate, unlabeled | Candidate appendix or future-work mention | Main-text evidence |

## Main-Text Rule

The main text can use `contrastive_negation`, `stock_openers`,
`stock_closers`, and `slop_lexicon` as interim precision-supported regex
features. The `slop_lexicon` row remains a frozen historical index rather than
a claim that every word in the pool is semantically diagnostic in every
context; report item-level examples and caveats for broad terms such as
`comprehensive`, `robust`, and `journey`.

## Table And Figure Rule

- Table 4 should continue to show every Tier-1 status row, including demoted,
  derived, and incomplete rows, because validation state is a scope-control
  result.
- Figure 3 can remain the OLMo `slop_lexicon` headline panel if the caption
  keeps the "frozen historical lexicon index" framing and does not imply
  completed corpus-rate precision validation.
- Rule-of-three rows can appear in appendix/control tables as a demoted
  exploratory coordinated-triple surface, but should not become headline
  findings unless a parser-backed replacement passes validation.

## Future Promotion Criteria

A currently incomplete or demoted feature can move into publication-grade
corpus-rate use after:

1. At least 50 labeled rows for the interim gate, with 200 preferred for the
   original validation plan.
2. Precision at or above 0.90.
3. Ambiguous rate at or below 0.10.
4. A short smell-test read confirming that false positives do not concentrate
   in one corpus, prompt type, or generation stage in a way that would bias the
   claimed contrast.
