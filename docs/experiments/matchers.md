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
  `not just/only/merely X but Y`, `isn't just X, it's Y`, and
  `it's not X, it's Y`.
- Slop lexicon: seed lexical and phrase items from the experiment plan,
  including `delve`, `tapestry`, `testament to`, `important to note`,
  `worth noting`, `nuanced`, `multifaceted`, `seamless`, `landscape`,
  `journey`, `navigate`, `foster`, `unlock`, `comprehensive`, `moreover`,
  and `ultimately`.
- Slop lexicon v2 candidate: exploratory addendum terms from the 2026-06-17
  source scan, including `meticulous`, `commendable`, `pivotal`,
  `burgeoning`, `paradigm`, `transformative`, `tailored`, `harness`,
  `unwavering`, `captivate`, `kaleidoscope`, and `symphony`. This is reported
  separately from the frozen historical `slop_lexicon`.
- EQ-Bench Slop Score: document-level benchmark score using vendored
  EQ-Bench Slop Score word and trigram lists, EQ-Bench leaderboard
  normalization ranges, and the portable Stage-1 not-X-but-Y contrast regexes.
  The feature reports `eqbench_slop_score` plus component helpers for slop word
  matches per 1k words, slop trigram matches per 1k words, and not-X-but-Y
  matches per 1k chars.
- Stock openers/closers: generic assistant openings such as `Great question`,
  `Certainly`, `Sure`, `Happy to help`, `Here are/is`, plus endings such as
  `I hope this helps`, `In conclusion`, sentence-final discourse-frame
  `Overall, ...`, `To summarize`, and `Let me know if...`.
- Punctuation/rhythm: em dash, colon, semicolon, ellipsis, exclamation, and
  question mark counts remain as aggregate density helpers only; they are not
  emitted as validation-sampled hits. Paragraph token mean, standard deviation,
  and a uniformity score `1 / (1 + coefficient_of_variation)` remain available.
- Rule-of-three approximation: a pragmatic coordinated triple regex for
  short `X, Y, and Z` or `X, Y and Z` spans, skipping fenced-code and
  data-like lines.
- Full pybiber: optional spaCy-backed extraction of the 67 pybiber Biber
  features. Use this for full register work when `pybiber` and the requested
  spaCy English model are installed:

```bash
uv run slop-pybiber-full \
  --input artifacts/path/to/generations.jsonl \
  --text-field generation \
  --id-field record_id \
  --output artifacts/path/to/pybiber_full.csv \
  --long-output artifacts/path/to/pybiber_full_long.csv
```

## Task Plan

- [x] Create the Tier-1 matcher module under `src/slop_sftdiv/features`.
- [x] Keep dependencies to the Python standard library.
- [x] Expose a small one-text API with raw counts and normalized helper data.
- [x] Add matcher notes and current limitations here.
- [ ] Precision validation on hand-labeled hit samples per
  `docs/experiments/precision_validation.md`.
- [ ] Add corpus-level aggregation once the Phase 1 data loader interface is
  settled.
- [ ] Replace the rule-of-three approximation with POS/chunk-aware matching if
  the project adds an NLP dependency.

## Precision Validation Notes

Stage 1 hit samples should contain up to 200 hits per feature from sampled
corpora, generated with the frozen matcher version and `seed: 1729`. Labels use
the shared schema: `exact`, `partial`, `false_positive`, and `ambiguous`.
Precision is `exact / total_labeled`; ambiguous and partial labels count
against the gate unless a feature spec explicitly says otherwise.

Features used for Phase 1 claims need precision `>= 0.90`. Demote any feature
below target, any feature with ambiguous rate above `0.10`, or any feature whose
opportunity context or denominator cannot be stated clearly. Demoted features
may remain in exploratory tables but should not support headline Phase 1 claims.

W&B validation runs should use project `slop-stage1`, group
`precision-validation`, and log aggregate metrics such as `matcher/hits`,
`matcher/precision`, `matcher/ambiguous_rate`, label counts, and artifact
references. Do not log unredacted full text or secrets.

Before Phase 1 claims, core matchers must pass precision gates: contrastive
negation, rule-of-three approximation, pooled slop lexicon, and stock
openers/closers. List/header/bold lead-in formatting is retired from the active
feature surface. Punctuation/rhythm remains exploratory and excluded from the
revised Phase 1 core claim set. If any revised core feature fails, block claims
until it is fixed or explicitly removed from the claim set.

`eqbench_slop_score` is a benchmark-comparison feature, not a replacement for
the frozen `slop_lexicon` used in existing Phase 2/3 tables. Its `count` field
is the EQ-Bench 0-100 weighted score summed across documents; use `per_doc` as
the corpus-level mean score, not `per_1k_tokens`.

For benchmark-style generation sets, use the aggregate scorer instead of
averaging row-level matcher outputs:

```bash
uv run slop-eqbench-score \
  --input artifacts/path/to/generations.jsonl \
  --text-field output \
  --output artifacts/path/to/eqbench_slop_score.json \
  --csv-output artifacts/path/to/eqbench_slop_score.csv
```

The CLI filters outputs with `len(output) > 300`, concatenates the valid
outputs, applies the vendored EQ-Bench word/trigram lists and leaderboard
normalization ranges, and writes the 0-100 score with component rates and top
hits. Add `--write-prompts artifacts/path/to/eqbench_prompts.json` to emit the
vendored 300-prompt manifest used for EQ-Bench-style generation.

## Limitations

This is a Tier-1 census pass, not the final validated feature spec. The token
denominator is a simple regex count, so it will not exactly match model
tokenizers. The contrastive-negation and rule-of-three matchers intentionally
favor recall over syntactic precision. The rule-of-three matcher does not yet
split NP, VP, and clause triples. Paragraph rhythm is only meaningful for texts
with paragraph breaks, and single-paragraph responses receive zero standard
deviation by construction.

Full pybiber extraction is available as a separate batched path rather than
inside the default regex census loop. It depends on probabilistic spaCy POS and
dependency parses, so run it as a register feature layer and keep it distinct
from deterministic Tier-1 regex claims.
