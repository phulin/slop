# Tier-1 Matchers

## Scope

Worker A owns the focused first-stage matcher component for the Phase 1 census.
The current implementation lives in `src/slop_sftdiv/features/tier1_matchers.py`
and exposes `extract_tier1_features(text)` through `src/slop_sftdiv/features`.

The API returns:

- pooled feature counts for one text
- per-1k-token rates using a simple word-token regex denominator
- per-item counts for each matcher family
- helper data for token count, sentence count, paragraph rhythm, punctuation
  densities, and rule-of-three per-sentence normalization

## Implemented Matcher Families

- Contrastive negation: regex approximations for `not X but Y`,
  `not just/only/merely X but Y`, and `isn't just X, it's Y`.
- Slop lexicon: seed lexical and phrase items from the experiment plan,
  including `delve`, `tapestry`, `testament to`, `important to note`,
  `worth noting`, `nuanced`, `multifaceted`, `seamless`, `landscape`,
  `journey`, `navigate`, `foster`, `unlock`, `comprehensive`, `moreover`,
  and `ultimately`.
- List/header/bold-lead-in: markdown bullet lines, numbered list lines,
  markdown headings, and `**Term**:` style lead-ins.
- Stock openers/closers: generic assistant openings such as `Great question`,
  `Certainly`, `Sure`, `Happy to help`, `Here are/is`, plus endings such as
  `I hope this helps`, `In conclusion`, `Overall`, `To summarize`, and
  `Let me know if...`.
- Punctuation/rhythm: em dash, colon, semicolon, ellipsis, exclamation, and
  question mark counts, plus paragraph token mean, standard deviation, and a
  uniformity score `1 / (1 + coefficient_of_variation)`.
- Rule-of-three approximation: a pragmatic coordinated triple regex for
  short `X, Y, and Z` or `X, Y and Z` spans.

## Task Plan

- [x] Create the Tier-1 matcher module under `src/slop_sftdiv/features`.
- [x] Keep dependencies to the Python standard library.
- [x] Expose a small one-text API with raw counts and normalized helper data.
- [x] Add matcher notes and current limitations here.
- [ ] Precision validation on a hand-labeled sample of 200 hits per feature.
- [ ] Add corpus-level aggregation once the Phase 1 data loader interface is
  settled.
- [ ] Replace the rule-of-three approximation with POS/chunk-aware matching if
  the project adds an NLP dependency.

## Limitations

This is a Tier-1 census pass, not the final validated feature spec. The token
denominator is a simple regex count, so it will not exactly match model
tokenizers. The contrastive-negation and rule-of-three matchers intentionally
favor recall over syntactic precision. The rule-of-three matcher does not yet
split NP, VP, and clause triples. Paragraph rhythm is only meaningful for texts
with paragraph breaks, and single-paragraph responses receive zero standard
deviation by construction.
