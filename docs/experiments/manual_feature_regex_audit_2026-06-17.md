# Manual Feature Regex Audit, 2026-06-17

## Scope

This audit reviews the active Tier-1 manual feature surface after retiring
`list_header_bold_lead_in`. The sample file is:

`artifacts/stage1/validation/hit_samples/manual_feature_smell_test_20_each.csv`

It contains 20 sampled hits each for:

- `contrastive_negation`
- `slop_lexicon`
- `stock_openers`
- `stock_closers`
- `punctuation_rhythm`
- `rule_of_three_approx`

The sample was drawn from the retained OLMo 3 Dolci SFT sample, retained OLMo 3
Dolci DPO sample, and retained Dolma 3 scan sample with seed `314159`.

## Online Source Scan

The current lexicon is broadly aligned with external "GPT-slop" source lists,
but those lists should be treated as candidate generators rather than ground
truth.

- EQ-Bench's Slop Score defines slop as words and patterns over-represented in
  AI text versus human text. Its score weights individual slop words, not-X-but-Y
  patterns, and slop trigrams. This independently supports keeping both
  `slop_lexicon` and `contrastive_negation` in the active surface.
  Source: https://eqbench.com/slop-score.html
- EQ-Bench's benchmark documentation says the slop list is computed from
  over-represented words in generated stories, with extra phrases compiled from
  similar public lists. This is useful for broad candidate discovery, but it is
  creative-writing-skewed.
  Source: https://eqbench.com/about.html
- Matsui's PubMed vocabulary study extracted potentially AI-influenced terms
  from prior literature and reports evidence for terms including `commendable`,
  `meticulous`, `intricate`, `realm`, plus post-ChatGPT excess vocabulary such
  as `delve` and `underscore`. This is stronger evidence for formal/scientific
  style than social-media word lists.
  Source: https://pmejournal.org/articles/10.5334/pme.1929
- FSU's report on Juzek and Ward describes a 2020-vs-2024 abstract comparison
  plus ChatGPT rewrites. It identifies overlap between words that spiked in
  scientific abstracts and words overused by ChatGPT-generated abstracts,
  including `delve`, `realm`, and `intricate`; it also reports that a
  preference-trained Llama variant was less surprised by the buzzword-heavy
  abstracts than the base model. This is directly relevant to our base/SFT/DPO
  progression question.
  Source: https://news.fsu.edu/news/science-technology/2025/02/17/why-does-chatgpt-delve-so-much-fsu-researchers-begin-to-uncover-why-chatgpt-overuses-certain-words/
- Business Insider's coverage of the Paul Graham `delve` discussion is not a
  primary measurement source, but it points to public salience and cites a
  phrase-finder result where `delve`, `explore`, `captivate`, and `tapestry`
  appeared among common ChatGPT terms.
  Source: https://www.businessinsider.com/y-combinator-paul-graham-delve-ai-chatgpt-giveaway-email-pitch-2024-4
- The `sam-paech/slop-score` and `sam-paech/antislop-sampler` lists are useful
  as open candidate inventories. Their raw lists include many creative-writing
  names and genre artifacts, so they should not be imported wholesale.
  Sources: https://github.com/sam-paech/slop-score and
  https://github.com/sam-paech/antislop-sampler

## Candidate Lexicon Additions

Recommended for a versioned `slop_lexicon_v2` candidate set:

- `meticulous` / `meticulously`
- `commendable`
- `pivotal`
- `burgeoning`
- `paradigm` / `paradigms`
- `transformative`
- `tailored`
- `harness`
- `unwavering`
- `captivate` / `captivating` / `captivated`
- `kaleidoscope`
- `symphony`

Do not add broad terms such as `explore`, `essential`, `crucial`, or `vital`
without phrase gating. They appear in public slop lists, but they are common
enough to create many benign technical and expository hits.

Because Phase 2 and Phase 3 already have reported `slop_lexicon` rates under the
frozen current matcher, these additions should be introduced as a versioned
addendum rather than silently changing the historical feature definition.

Implemented follow-up: these terms are now exposed as
`slop_lexicon_v2_candidate`, separate from historical `slop_lexicon`.
`eqbench_slop_score` is also exposed as an exploratory document-level benchmark
feature using the vendored EQ-Bench Slop Score lists and normalization ranges.

## Sample Composition

Feature counts:

| Feature | Sampled Hits |
| --- | ---: |
| `contrastive_negation` | 20 |
| `punctuation_rhythm` | 20 |
| `rule_of_three_approx` | 20 |
| `slop_lexicon` | 20 |
| `stock_closers` | 20 |
| `stock_openers` | 20 |

Subtype counts:

| Feature | Subtypes |
| --- | --- |
| `contrastive_negation` | `not_x_but_y`: 14; `not_just_but`: 5; `not_x_its_y`: 1 |
| `punctuation_rhythm` | `colon`: 10; `em_dash`: 4; `semicolon`: 4; `question`: 2 |
| `rule_of_three_approx` | `approx_triple`: 20 |
| `slop_lexicon` | `robust`: 4; `journey`: 3; `delve`: 2; `underscore`: 2; singletons for `moreover`, `foster`, `testament_to`, `elevate`, `comprehensive`, `tapestry`, `navigate`, `ultimately`, `nuanced` |
| `stock_closers` | `overall`: 10; `let_me_know`: 9; `in_conclusion`: 1 |
| `stock_openers` | `here_are`: 12; `certainly`: 5; `sure`: 2; `great_question`: 1 |

## Smell-Test Findings

### `contrastive_negation`

The matcher is mostly capturing the intended rhetorical contrast pattern. The
cleanest hits are `not only X but also Y`, `not X but Y`, and `it is not X, it
is Y` frames in assistant-style explanations.

Observed weakness: ordinary negation plus `but` can still match. The Dolma hits
included spam/adult-web text and duplicated dating-site text where the syntax
matches but the feature is not a meaningful AI-style rhetorical marker.

Recommendation: keep as active, but report subtype-level rates and keep the
manual precision gate. This remains one of the better-supported features because
EQ-Bench independently treats not-X-but-Y patterns as part of a slop score.

### `slop_lexicon`

The sampled hits include intended markers such as `delve`, `tapestry`,
`testament to`, `journey`, `navigate`, `underscore`, `robust`, and `nuanced`.
The highest-confidence examples are in generated assistant-style prose.

Observed weakness: several entries are semantically ordinary in technical or
domain-specific contexts. `robust`, `comprehensive`, `ultimately`, and
`moreover` can be benign; `journey` can be roleplay-specific rather than a
general assistant tic.

Recommendation: keep the pooled current lexicon for historical continuity, but
interpret item-level rates. Add new items only through a versioned lexicon
addendum with separate reporting.

### `stock_openers`

This category has high face validity. Hits such as `Here is`, `Here are`,
`Certainly`, `Sure`, and `Great question` are exactly the assistant-response
formulae we intended to track.

Observed weakness: one Dolma pretraining web hit begins with `Here are` in a
regular article-like context. That is acceptable for a corpus baseline because
it tells us the phrase is not impossible in pretraining text.

Recommendation: keep as active. This is one of the cleanest manual features.

### `stock_closers`

`Let me know if...` and `In conclusion` are clean. The problem is `overall`.
Half of the sampled closer hits came from `overall`, and many were ordinary
domain usage rather than a closing formula: tensor-shape docstrings,
probability calculations, chemistry explanations, and geographic lists.

Implemented follow-up: `overall` is tightened to final discourse frames such as
`Overall, ...` with a stricter sentence boundary. Keep `let_me_know`,
`in_conclusion`, `to_summarize`, and `hope_this_helps`.

### `punctuation_rhythm`

The per-hit sample is not measuring a meaningful slop feature. It mostly finds
ordinary punctuation, code/CSS colons, markdown table separators, repeated
LaTeX semicolon artifacts, and normal question marks.

Implemented follow-up: `punctuation_rhythm` is no longer emitted by
`iter_tier1_hits`, so it should not consume hit-labeling time. Aggregate
punctuation density helpers remain available for compatibility.

### `rule_of_three_approx`

The regex successfully finds coordinated triples, but that is not the same as
finding AI-style rhetorical triads. It catches real triples in technical prose,
corporate descriptions, historical summaries, adult-web text, and code
descriptions.

Implemented follow-up: the rule-of-three matcher now skips fenced-code and
data-like lines. It remains a surface coordinated-triple feature, not direct
evidence of "AI slop" without caveats.

## Formatting Feature Retirement

`list_header_bold_lead_in` is removed from the active matcher implementation and
from active Stage 1 matcher configuration. Historical reports may still mention
old list/header measurements, but new matcher runs should not emit the feature.

## Bottom Line

The remaining manual features are not equally reliable:

- Strong: `stock_openers`; `let_me_know`/`in_conclusion` closers.
- Good but needs label gating: `contrastive_negation`; current
  `slop_lexicon`.
- Useful only with caveats: `rule_of_three_approx`.
- Demote or redefine: `overall` closer subtype; `punctuation_rhythm`.
