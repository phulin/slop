# Phase 2 Final Conclusion Report

Date: 2026-06-15

Status: final bounded OLMo 3 / Dolci Phase 2 close-out.

Audience: this report is written for a reader who is new to the project. It
explains the question, the measured artifacts, the model progression across
checkpoints, the feature-level results, the Biber-lite style signature, and the
conclusions that should carry forward into Phase 3.

## Start Here

This report is the reader-facing close-out for Phase 2. It is meant to answer
four practical questions:

1. What did Phase 2 measure that Phase 1 did not?
2. How did the OLMo 3 checkpoints change from Base to SFT to DPO to
   Final/RLVR?
3. Which "slop" features are actually amplified, and which are inherited or
   suppressed?
4. What should Phase 3 treat as settled, tentative, or still unmeasured?

The short answer is that Phase 2 found a non-monotonic checkpoint progression.
Base already emits many measured assistant-style markers. SFT often lowers
visible marker rates in sampled output. DPO selectively rebounds, especially
for `slop_lexicon` local propensity and stock phrase emissions. Final/RLVR is
closest to DPO in the assembled output style signature, but it does not
uniformly increase measured slop beyond DPO.

The report is intentionally detailed because the conclusion depends on not
collapsing several different measurements into one number. A word-count rate
in sampled generations, a teacher-forced probability ratio, a chosen-vs-rejected
data comparison, and a self-conditioning window test are all evidence, but they
answer different questions.

## Reader's Guide

This project uses "slop" as an empirical label, not as a moral judgment about
an answer. A measured output can be useful and still contain one of these
markers. The project asks a narrower question:

> When a model produces recognizable assistant-style surface patterns, can we
> tell whether those patterns were inherited from data, amplified by model
> training, or amplified by generation dynamics?

Phase 2 is the model-side part of that question. Phase 1 counted the same
features in data. Phase 2 asks whether checkpoints assign more probability to
those features and whether sampled generations show the same patterns.

The shortest path through this report is:

1. Read the Executive Summary for the main answer.
2. Read "What Phase 2 Actually Measured" to understand the bounded scope.
3. Read "How To Read The Metrics" before interpreting the tables.
4. Read Results 1-8 for the evidence.
5. Read "Final Conclusion" for the claims that should be reused downstream.

The report intentionally separates four things that are easy to conflate:

- **data frequency:** how often a feature appears in corpus samples;
- **local model propensity:** how much probability the model gives a feature
  at a fixed opportunity position;
- **sampled output frequency:** how often the feature appears in generated
  completions;
- **self-conditioning:** whether one hit makes later hits more likely in the
  same answer.

Those four views do not always agree. That disagreement is one of the main
findings.

## Glossary

**Phase 1** measured data. It counted feature rates in corpus samples:
pretraining-like text, SFT targets, DPO chosen responses, and DPO rejected
responses.

**Phase 2** measured models. It asked whether checkpoints assign probability to
the same features under teacher forcing, whether generated completions contain
those features, and whether feature hits compound after they appear once in a
generation.

**Checkpoint ladder** means the sequence of model states being compared:
Base, SFT, DPO, and Final/RLVR.

**Teacher forcing** means the model is scored on a fixed reference prefix
instead of being allowed to sample freely. This isolates local model
probability from generation drift.

**Free-running generation** means the model samples completions normally. This
is closest to user-visible output, but it mixes local model preference,
decoding, sequence length, and self-conditioning.

**Amplification factor**, or AF, is a ratio comparing model probability mass
for a feature to the feature's reference rate under a specific opportunity
definition. An AF above `1` means the model assigns more probability to that
feature than the reference rate would suggest; an AF below `1` means less.

**Neutral-normalized AF** is AF divided by a neutral control basket. The main
`slop_lexicon` teacher-forced result uses this because it reduces the chance
that a general calibration shift is mistaken for slop-specific amplification.

**Prior/no-prior compounding** is the direct test for repetition dynamics. It
compares windows that occur after a feature has already appeared in the same
generation with windows where no earlier hit has appeared.

**Biber-lite** is a regex-based register proxy layer for pronouns, modals,
hedges, amplifiers, verb classes, nominalizations, complements,
subordination, wh-questions, and passive-voice approximations. It is useful
style evidence, but it is not full pybiber extraction.

## Evidence Map

For a new reader, the main evidence can be organized like this:

| Evidence type | Main artifact | What it answers | Strongest use in this report |
|---|---|---|---|
| Phase 1 data rates | Corpus census and preference-pair tables | Was the feature already common in data? | Shows SFT targets and DPO pairs do not explain all model-side changes. |
| Teacher-forced propensity | `olmo3_amplification_spectrum_single_temp_t1_v6.csv` | Which checkpoint locally prefers the feature? | `slop_lexicon` peaks at DPO by neutral-normalized AF. |
| Free-running emissions | `olmo3_generation_stage_grid_target_shape_512prompt_8comp_t1_1024.csv` | What appears in sampled output? | SFT is lowest across retained Tier-1 markers; DPO rebounds selectively. |
| Compounding | `olmo3_generation_compounding_target_shape_512prompt_8comp_t1_1024_tf1024.csv` | Does an earlier hit make later hits more likely? | `slop_lexicon` has a large prior/no-prior signal at every checkpoint. |
| Biber-lite register | `olmo3_biber_lite_generation_vs_corpus_t1.csv` | How does broader output style shift? | Final/RLVR is DPO-like overall but differs from Base and SFT targets in register. |
| Style signature | `olmo3_style_signature_t1.csv` | Which checkpoints are closest in final output style? | Final/RLVR is closest to DPO, then SFT, then Base. |

## Project Context In Plain Language

The project starts from a practical observation: modern instruction-tuned
models often share recognizable surface habits. Some are lexical, such as a
small set of recurring words and phrases. Some are structural, such as
three-part lists or formulaic openings and closings. Some are broader register
choices, such as higher use of pronouns, modal verbs, hedges, or subordinate
clauses.

The experiment does not assume those habits all have one cause. In a
post-training pipeline, a feature can enter or grow in several different ways:

1. It can already be present in pretraining text or in the base model.
2. It can be inherited from SFT target responses.
3. It can be favored by preference data, if chosen responses contain more of
   the feature than rejected responses.
4. It can be amplified by optimization even when the paired preference data do
   not show a clean chosen-over-rejected advantage.
5. It can compound during generation because the model conditions on its own
   earlier wording.

Phase 1 handled the first part of the problem by counting features in data.
Phase 2 handled the second part by measuring models directly. This report is
therefore about the OLMo 3 model ladder, not about whether a feature is common
in every possible training source.

The key idea is that a generated answer is a final product of several forces.
A high feature rate in final output does not automatically mean DPO caused it.
A low final rate does not mean the model lacks the local tendency. That is why
this report keeps data rates, teacher-forced probability, sampled output, and
compounding separate until the conclusion.

## Executive Summary

This project studies where recognizable "LLM-ish" output style enters a
post-training pipeline. The target is not answer quality in general. The target
is a set of measurable surface markers: curated lexical tics, formulaic
openers and closers, three-part rhetorical lists, contrastive "not X but Y"
phrasing, and broader register features such as pronouns, modals, hedges, and
subordination.

Phase 1 measured data. It counted these features in corpus roles such as
pretraining samples, SFT targets, DPO chosen responses, and DPO rejected
responses.

Phase 2 measured models. It asked how the OLMo 3 Instruct checkpoint ladder
actually behaves when scored locally and when sampled freely.

The headline result is:

> OLMo 3 does not follow a simple "later checkpoint means more slop" curve.
> Base already contains substantial measured style. SFT suppresses many
> visible Tier-1 output markers. DPO rebounds on selected features, especially
> local `slop_lexicon` propensity and sampled stock phrase rates. Final/RLVR
> stays closest to DPO in aggregate output style, but it does not broadly
> increase the retained slop markers beyond DPO.

The strongest positive result is `slop_lexicon`:

- DPO has the highest teacher-forced neutral-normalized amplification factor:
  `1.999`, with 95% CI `1.446-2.837`.
- Final/RLVR falls back from the DPO point estimate to `1.659`.
- Free-running sampled output is not a clean DPO-only peak because Base is
  already high: Base `0.233` vs. DPO `0.229` hits per 1,000 generated tokens.
- Once a generated answer has already used a `slop_lexicon` item, later
  windows are about 6x to 9x more likely to contain another hit than windows
  without a prior hit.

The strongest negative result is equally important:

> Phase 2 does not support a broad claim that DPO universally creates slop.
> `rule_of_three_approx` is highest in Base free-running output, its
> teacher-forced proxy peaks at SFT, and Final/RLVR attenuates several DPO
> stage slop rates.

The practical conclusion is that "slop" should be analyzed feature by feature.
There is no single global slop score that accurately describes the pipeline.

For a new reader, the best one-sentence takeaway is:

> Phase 2 found a feature-specific style progression: Base already carries a
> lot of assistant-like surface form, SFT often suppresses visible markers,
> DPO selectively rebounds, and Final/RLVR remains DPO-like without becoming a
> uniformly stronger slop amplifier.

The strongest and weakest claim areas are:

| Claim area | Confidence | Why |
|---|---|---|
| `slop_lexicon` DPO local-propensity peak | High for the bounded OLMo slice | It has Phase 1 data rates, teacher-forced AF, free-running rates, compounding, and prior/no-prior tests. |
| `slop_lexicon` self-conditioning | High for the bounded OLMo slice | Prior-hit windows have much higher later-hit risk at every checkpoint. |
| DPO as universal slop source | Low / not supported | Several features peak at Base or SFT, and Final/RLVR attenuates several DPO rates. |
| `rule_of_three_approx` DPO amplification | Low / not supported | Base has the highest free-running rate and SFT has the highest retained teacher-forced proxy. |
| `contrastive_negation` model-side amplification | Underdetermined | It is visible in output, but retained teacher-forced support is sparse. |
| Full linguistic-register conclusion | Moderate descriptive confidence | Biber-lite is consistent and useful, but it is regex proxy measurement rather than full pybiber. |

## What Phase 2 Actually Measured

The model ladder was the OLMo 3 Instruct path:

| Stage | Report label | Interpretation |
|---|---|---|
| Base | Base | The base OLMo 3 checkpoint in the measured ladder. |
| SFT | SFT | The supervised fine-tuned checkpoint. |
| DPO | DPO | The preference-optimized checkpoint. |
| Final/RLVR | Final/RLVR | The final released checkpoint after the later RLVR-style stage. |

The retained free-running generation grid used one target shape:

| Setting | Value |
|---|---:|
| Prompt source | Held-out Dolci SFT prompt package |
| Prompts per stage | 512 |
| Completions per prompt | 8 |
| Generations per stage | 4,096 |
| Total generations | 16,384 |
| Max new tokens per completion | 1,024 |
| Temperature | 1.0 |
| Top-p | 0.95 |

Supporting Phase 2 analyses included:

- teacher-forced propensity grids for selected feature/opportunity contracts;
- a DPO-only scale check at 1,024 prompts x 8 completions;
- DPO-only temperature sensitivity runs at `0.0`, `0.7`, and `1.0`;
- expected-vs-observed compounding analysis;
- direct prior/no-prior compounding windows;
- Biber-lite register measurements over generated outputs;
- a final assembled style-signature table combining Tier-1 emissions,
  compounding metrics, and Biber-lite rates.

This is a bounded close-out, not the original full production grid. The full
production plan called for 5,000 prompts, 8 completions, 3 temperatures,
multiple model ladders, and full pybiber extraction. Phase 2 closed on a
smaller but internally consistent OLMo 3 / Dolci slice.

## Original Phase 2 Plan vs. Delivered Evidence

The original Phase 2 plan in `EXPERIMENTS.md` had two main measurements:

1. **Teacher-forced propensity:** hold the context fixed and measure how much
   probability each checkpoint assigns to feature-starting continuations.
2. **Free-running emission:** sample completions from each checkpoint and count
   how often the features appear in visible output.

The plan also called for a named result:

> Result B: decompose free-running feature rates into local propensity and
> self-conditioning/compounding.

The bounded Phase 2 close-out delivered that structure for the OLMo 3 / Dolci
slice:

| Planned item | Delivered status | What to trust |
|---|---|---|
| Held-out prompt package | Delivered | 5,000-prompt Dolci SFT package exists; final generation grid uses 512 held-out prompts per stage. |
| Teacher-forced propensity | Delivered for retained contracts | Strongest for `slop_lexicon`; useful but narrower for `rule_of_three_approx`, stock phrases, and denominator-supported features. |
| Free-running emission | Delivered | 512 prompts x 8 completions x 4 checkpoints at temperature `1.0`, max 1,024 new tokens. |
| Temperature grid | Partial | DPO temperature sensitivity exists; final headline report uses single-temperature `t=1.0`. |
| Result B compounding | Delivered | Expected-vs-observed and prior/no-prior window tests are included. |
| Biber/full register style layer | Partial | Biber-lite regex proxies were measured; full pybiber was not run. |
| Cross-ladder replication | Not Phase 2 | SmolLM3 replication belongs to Phase 3. |

The consequence is that this report is suitable as a final bounded Phase 2
conclusion, but it should not be described as the full production-scale
experiment originally scoped.

## What We Were Looking For

The original plan was not just to count words in generated answers. It was to
distinguish several possible stories that can produce the same surface
symptom.

If a feature were mostly **inherited**, we would expect it to appear in the
data and in the base model, with no clean late-stage jump. A high Base
free-running rate or a high pretraining/SFT corpus rate would support that
story.

If a feature were **SFT-amplified**, we would expect a rise at the SFT
checkpoint or high SFT-target prevalence. The feature could then persist into
later checkpoints without DPO being the original source.

If a feature were **preference-amplified**, we would expect a DPO-stage jump
in model propensity or generated output. If DPO chosen responses also had more
of the feature than DPO rejected responses, that would point to preference
data complicity. If the model jumped without a chosen-over-rejected data
advantage, that would point more toward optimization dynamics or an indirect
interaction.

If a feature were **generation-compounded**, we would expect free-running
output to exceed what the local teacher-forced probabilities predict, and we
would expect later windows to become more likely after an earlier hit in the
same answer.

The delivered Phase 2 result contains all four patterns somewhere in the
ladder. That is why the conclusion is an amplification spectrum rather than a
single yes/no answer about whether DPO "causes slop."

## Feature Definitions

This report uses "slop" as shorthand for detector-defined surface features. It
does not mean that an entire answer is bad. A useful answer can contain a
measured marker, and an unhelpful answer can contain none.

The retained Tier-1 feature views were:

| Feature | Meaning |
|---|---|
| `slop_lexicon` | Curated lexical tics and stock multiword expressions. |
| `rule_of_three_approx` | Regex approximation of three-part rhetorical lists like "X, Y, and Z." |
| `contrastive_negation` | Contrastive "not X, but Y" or "not just X" constructions. |
| `stock_openers` | Formulaic response openings. |
| `stock_closers` | Formulaic response closings. |
| `stock_openers_closers` | Pooled opener/closer view. Useful for summaries, but it hides opener-vs-closer differences. |

Phase 2 also measured `biber_lite` register proxies on generated outputs.
These are broader style features such as pronouns, modals, hedges,
amplifiers, nominalizations, complements, subordination, wh-questions, and
passive-voice approximations. They describe final-output register. They are
not teacher-forced amplification measurements.

## How To Read The Metrics

Phase 2 uses several metrics because each answers a different question.

**Corpus rate** is the Phase 1 data measurement. It counts how often a feature
appears in corpus roles such as pretraining samples, SFT targets, DPO chosen
responses, or DPO rejected responses. It answers: how common was this feature
in the data?

**Teacher-forced propensity** is the local, decoding-free model measurement.
The model is shown a fixed reference prefix, and the harness measures
probability mass assigned to feature-bearing continuations at predefined
opportunity positions. It answers: given the same context, how much does this
checkpoint want to start the feature?

**Amplification factor**, or AF, compares model feature probability to the
reference feature rate under the opportunity contract. For the main
`slop_lexicon` result, AF is normalized by `neutral_common_controls`, so a
general shift in model confidence is less likely to masquerade as
slop-specific amplification.

**Free-running emission** counts feature hits in sampled completions. This is
closest to user-visible output, but it is affected by decoding, sequence
length, prompt mix, and the model conditioning on its own prior text.

**Compounding** asks whether a feature becomes more likely after it already
appears in the same generated answer. Phase 2 keeps two views: an
expected-vs-observed join against teacher-forced probability mass, and a
direct prior/no-prior window test over generated text.

**Biber-lite register** describes broader surface style with regex proxies. It
is useful for comparing final output style to Base, SFT, DPO, and corpus
samples. It is descriptive, not causal evidence by itself.

## Phase 1 Baseline Context

Phase 2 results are interpretable only against Phase 1 data baselines. Rates
below are per 1,000 simple tokens:

| Corpus / role | Contrastive negation | Rule of three | Slop lexicon | Stock openers | Stock closers | Openers + closers |
|---|---:|---:|---:|---:|---:|---:|
| Dolma pretrain sample | 0.389 | 3.946 | 0.238 | 0.027 | 0.008 | 0.035 |
| Dolci SFT target | 0.139 | 1.937 | 0.553 | 0.262 | 0.089 | 0.351 |
| Dolci DPO chosen | 0.459 | 3.696 | 0.990 | 0.235 | 0.413 | 0.648 |
| Dolci DPO rejected | 0.369 | 3.391 | 1.022 | 0.277 | 0.292 | 0.569 |

The important baseline facts are:

- pretraining samples are already high on `rule_of_three_approx` and
  `contrastive_negation`;
- SFT targets are higher than the pretraining sample on `slop_lexicon` and
  stock opener/closer phrases;
- DPO chosen responses are higher than rejected responses for contrastive
  negation, rule-of-three, stock closers, and pooled stock phrases;
- DPO chosen responses are not higher than rejected responses for
  `slop_lexicon`.

The last point is important. If DPO model propensity for `slop_lexicon` rises,
the simplest explanation cannot be "the DPO chosen responses just contained
more `slop_lexicon` than rejected responses." Phase 3 should treat that as a
candidate dynamics-driven or mixed effect.

## Result 1: The Checkpoint Progression Is Non-Monotonic

A simple story would be: Base is clean, SFT adds assistant style, DPO adds
more slop, and Final/RLVR adds the most. The bounded OLMo results do not
support that story.

The observed pattern is:

1. Base already emits several measured style markers at substantial rates.
2. SFT is the lowest free-running checkpoint for every retained Tier-1 feature
   in the target-shape grid.
3. DPO rebounds on selected features.
4. Final/RLVR remains closest to DPO in aggregate style, but it does not
   broadly increase slop beyond DPO.

Target-shape free-running rates per 1,000 generated tokens:

| Feature | Base | SFT | DPO | Final/RLVR | Maximum stage |
|---|---:|---:|---:|---:|---|
| `rule_of_three_approx` | 1.019 | 0.488 | 0.861 | 0.802 | Base |
| `slop_lexicon` | 0.233 | 0.171 | 0.229 | 0.193 | Base |
| `contrastive_negation` | 0.136 | 0.041 | 0.141 | 0.142 | Final/RLVR |
| `stock_openers_closers` | 0.092 | 0.072 | 0.108 | 0.091 | DPO |
| `stock_openers` | 0.064 | 0.052 | 0.064 | 0.059 | DPO |
| `stock_closers` | 0.028 | 0.020 | 0.043 | 0.032 | DPO |

This is the cleanest stage-localization result. The visible style features are
feature-specific and non-monotonic.

## Result 2: `slop_lexicon` Is The Strongest DPO Local-Propensity Signal

`slop_lexicon` has the strongest Phase 2 measurement chain: Phase 1 corpus
rates, teacher-forced AF, free-running emission, expected-vs-observed
compounding, and direct prior/no-prior compounding windows.

Teacher-forced neutral-normalized AF:

| Stage | Normalized AF | 95% CI |
|---|---:|---:|
| Base | 1.467 | 1.114-2.046 |
| SFT | 1.695 | 1.315-2.289 |
| DPO | 1.999 | 1.446-2.837 |
| Final/RLVR | 1.659 | 1.189-2.351 |

DPO has the highest point estimate. Final/RLVR falls back toward SFT. The
confidence intervals overlap, so the correct claim is "bounded positive
DPO-stage peak," not "DPO is cleanly separated from every other stage."

Free-running `slop_lexicon` emission is less clean:

| Stage | Free-run /1k generated tokens |
|---|---:|
| Base | 0.233 |
| SFT | 0.171 |
| DPO | 0.229 |
| Final/RLVR | 0.193 |

Base is narrowly above DPO in sampled output. DPO is close to Base and clearly
above SFT. Final/RLVR is below DPO.

Conclusion:

> DPO is the peak local-propensity checkpoint for `slop_lexicon`, but sampled
> output does not become a clean DPO-only peak because Base already emits the
> feature at a comparable rate.

This is why Phase 2 keeps teacher-forced propensity and free-running emission
separate.

## Result 3: `slop_lexicon` Compounds During Generation

The expected-vs-observed compounding analysis compares realized generated
opportunity rates against teacher-forced expectations. For `slop_lexicon`,
observed generated opportunity rates exceed teacher-forced expectations in all
four stages:

| Stage | Observed /1k opportunities | Expected /1k opportunities | Excess /1k | Realized AF | Repeat generations |
|---|---:|---:|---:|---:|---:|
| Base | 0.615 | 0.232 | 0.383 | 1.051 | 192 |
| SFT | 0.697 | 0.336 | 0.361 | 1.192 | 145 |
| DPO | 0.638 | 0.439 | 0.199 | 1.091 | 199 |
| Final/RLVR | 0.582 | 0.445 | 0.137 | 0.994 | 180 |

The direct prior/no-prior window test is even clearer:

| Stage | P(hit after prior) | P(hit without prior) | Risk difference | Risk ratio |
|---|---:|---:|---:|---:|
| Base | 0.0812 | 0.0106 | 0.0707 | 7.68 |
| SFT | 0.1052 | 0.0113 | 0.0939 | 9.35 |
| DPO | 0.0712 | 0.0113 | 0.0599 | 6.29 |
| Final/RLVR | 0.0691 | 0.0107 | 0.0584 | 6.43 |

After a `slop_lexicon` hit appears, later windows are about six to nine times
more likely to contain another hit than windows without a prior hit.

Conclusion:

> `slop_lexicon` has a real self-conditioning signal. The effect is not
> DPO-specific; Base and SFT compound too. Free-running style is not just a
> sequence of independent local token decisions.

## Result 4: `rule_of_three_approx` Is Visible But Not DPO-Amplified Here

`rule_of_three_approx` is the most common retained Tier-1 feature in generated
outputs, but its stage pattern does not support a DPO-specific amplification
claim.

Free-running rate per 1,000 generated tokens:

| Stage | Rate |
|---|---:|
| Base | 1.019 |
| SFT | 0.488 |
| DPO | 0.861 |
| Final/RLVR | 0.802 |

Base is highest. DPO and Final/RLVR rebound from SFT but stay below Base.

The teacher-forced proxy points in the same cautious direction. It measures
third-item extension after a candidate two-item comma/list context. It does
not measure open-vocabulary initiation of an entire three-part construction.

| Stage | Raw AF | 95% CI |
|---|---:|---:|
| Base | 0.721 | 0.698-0.742 |
| SFT | 0.767 | 0.743-0.787 |
| DPO | 0.716 | 0.691-0.738 |
| Final/RLVR | 0.723 | 0.698-0.746 |

SFT has the highest point estimate for this bounded proxy. DPO is slightly
below Base. All stages are below the held-out reference extension rate.

Conclusion:

> `rule_of_three_approx` is a visible output marker, but in this OLMo slice it
> is better described as inherited/base-heavy with SFT suppression and later
> rebound, not as DPO-amplified.

## Result 5: Stock Openers And Closers Split Apart

The pooled `stock_openers_closers` feature is useful for summaries, but it
hides an important opener-vs-closer distinction.

Teacher-forced `stock_openers` raw AF:

| Stage | Raw AF |
|---|---:|
| Base | 0.067 |
| SFT | 0.048 |
| DPO | 0.035 |
| Final/RLVR | 0.035 |

The model assigns much less probability mass to retained stock opener starts
than the held-out references use.

Teacher-forced `stock_closers` raw AF:

| Stage | Raw AF |
|---|---:|
| Base | 73.304 |
| SFT | 65.076 |
| DPO | 68.634 |
| Final/RLVR | 74.032 |

The closer AF is extremely high because the held-out reference denominator is
small: 18 positives in 41,187 opportunities. The direction is informative; the
exact magnitude is fragile.

Free-running stock phrase rates are small and DPO-peaked:

| Feature | Base | SFT | DPO | Final/RLVR |
|---|---:|---:|---:|---:|
| `stock_openers` | 0.064 | 0.052 | 0.064 | 0.059 |
| `stock_closers` | 0.028 | 0.020 | 0.043 | 0.032 |
| `stock_openers_closers` | 0.092 | 0.072 | 0.108 | 0.091 |

Conclusion:

> DPO is the visible sampled-output peak for stock phrases, but opener and
> closer behavior should be analyzed separately. The large teacher-forced
> closer AF is a rare-reference result, not a stable large-effect estimate.

## Result 6: `contrastive_negation` Is Output-Visible, But Teacher-Forced Support Is Sparse

`contrastive_negation` has a clear SFT-suppression and later-rebound pattern
in free-running generation:

| Stage | Free-run /1k generated tokens |
|---|---:|
| Base | 0.136 |
| SFT | 0.041 |
| DPO | 0.141 |
| Final/RLVR | 0.142 |

Base, DPO, and Final/RLVR are close. SFT is much lower.

The retained teacher-forced table does not support a strong AF claim for this
feature because held-out target support is sparse: only 7 reference instances
in the 5,000-prompt denominator audit.

Conclusion:

> The safe Phase 2 claim is output-only: `contrastive_negation` is visible in
> generated text, SFT suppresses it, and later checkpoints rebound to roughly
> Base-like rates. It needs better teacher-forced support before stronger
> causal interpretation.

## Result 7: Biber-Lite Gives A Broader Final-Output Style Signature

Phase 2 measured Biber-lite features on generated outputs and compared them to
Phase 1 corpus samples. This gives a wider style signature beyond the
hand-selected Tier-1 markers.

Selected Biber-lite progression, per 1,000 generated tokens:

| Feature | Base | SFT | DPO | Final/RLVR | Final - Base |
|---|---:|---:|---:|---:|---:|
| Demonstratives | 15.686 | 11.420 | 9.984 | 10.115 | -5.571 |
| Infinitives | 17.426 | 15.233 | 13.009 | 13.623 | -3.803 |
| First-person pronouns | 15.325 | 10.890 | 10.302 | 11.638 | -3.687 |
| Necessity modals | 5.037 | 2.460 | 2.416 | 2.505 | -2.532 |
| Prediction modals | 3.851 | 2.500 | 1.676 | 1.879 | -1.972 |
| Possibility modals | 6.739 | 4.509 | 4.759 | 5.126 | -1.613 |
| Conditional subordinators | 6.133 | 5.307 | 7.865 | 7.659 | +1.525 |
| Second-person pronouns | 4.349 | 4.607 | 5.450 | 5.548 | +1.198 |
| Nominalizations | 25.499 | 28.232 | 27.426 | 27.309 | +1.810 |

Broad pattern:

- Base is high on demonstratives, infinitives, first-person pronouns, and
  modal language.
- DPO and Final/RLVR reduce many Base-heavy markers.
- DPO and Final/RLVR are higher on second-person pronouns, conditional
  subordinators, and nominalizations.

Against SFT targets, Final/RLVR is notably higher on several register proxies:

| Feature | Final /1k | SFT target /1k | log2 ratio |
|---|---:|---:|---:|
| `biber_lite_wh_questions` | 0.627 | 0.133 | 2.239 |
| `biber_lite_hedges` | 1.868 | 0.465 | 2.006 |
| `biber_lite_public_verbs` | 1.073 | 0.392 | 1.452 |
| `biber_lite_private_verbs` | 1.282 | 0.482 | 1.411 |
| `biber_lite_amplifiers` | 0.396 | 0.195 | 1.024 |
| `biber_lite_suasive_verbs` | 0.401 | 0.224 | 0.840 |
| `biber_lite_necessity_modals` | 2.505 | 1.431 | 0.808 |
| `biber_lite_second_person_pronouns` | 5.548 | 3.282 | 0.757 |
| `biber_lite_third_person_pronouns` | 5.407 | 3.575 | 0.597 |
| `biber_lite_conditional_subordinators` | 7.659 | 5.179 | 0.564 |

Conclusion:

> Final output style differs from the SFT target corpus in broader register,
> not only in explicit slop lexicon or stock phrases. Under these proxies, it
> is more question-heavy, hedged, verb-heavy in selected categories,
> pronominal, modal, and subordinating than the SFT target sample.

## Result 8: Final/RLVR Is DPO-Like But Not DPO-Identical

The final output style signature joins Tier-1 free-running slop rates,
empirical compounding metrics, and Biber-lite register rates. It then computes
checkpoint distances over the combined raw vector.

These distances are useful for orientation. They are not formal statistical
tests because high-rate metrics dominate Euclidean distance.

Distances from Final/RLVR:

| Comparison | Shared features | Euclidean distance | Cosine distance |
|---|---:|---:|---:|
| Final/RLVR vs DPO | 43 | 30.567 | 0.000075 |
| Final/RLVR vs SFT | 43 | 35.564 | 0.000287 |
| Final/RLVR vs Base | 43 | 85.139 | 0.001272 |

Under this assembled style signature, Final/RLVR is closest to DPO, then SFT,
then Base.

Largest Final-vs-Base shifts:

| Group | Metric | Feature | Final | Base | Delta |
|---|---|---|---:|---:|---:|
| Compounding | Observed /1k opportunities | `rule_of_three_approx` | 428.106 | 344.007 | +84.098 |
| Compounding | Observed /1k opportunities | `stock_openers` | 1.477 | 8.592 | -7.115 |
| Biber-lite | Generation /1k tokens | Demonstratives | 10.115 | 15.686 | -5.571 |
| Biber-lite | Generation /1k tokens | Infinitives | 13.623 | 17.426 | -3.803 |
| Biber-lite | Generation /1k tokens | First-person pronouns | 11.638 | 15.325 | -3.687 |
| Biber-lite | Generation /1k tokens | Necessity modals | 2.505 | 5.037 | -2.532 |
| Biber-lite | Generation /1k tokens | Prediction modals | 1.879 | 3.851 | -1.972 |
| Biber-lite | Generation /1k tokens | Nominalizations | 27.309 | 25.499 | +1.810 |

Largest Final-vs-DPO shifts:

| Group | Metric | Feature | Final | DPO | Delta |
|---|---|---|---:|---:|---:|
| Compounding | Observed /1k opportunities | `rule_of_three_approx` | 428.106 | 458.417 | -30.311 |
| Compounding | Prior risk ratio | `stock_closers` | 9.399 | 6.198 | +3.201 |
| Biber-lite | Generation /1k tokens | First-person pronouns | 11.638 | 10.302 | +1.336 |
| Biber-lite | Generation /1k tokens | Causal subordinators | 3.831 | 2.856 | +0.975 |
| Biber-lite | Generation /1k tokens | Hedges | 1.868 | 1.109 | +0.759 |
| Biber-lite | Generation /1k tokens | Infinitives | 13.623 | 13.009 | +0.614 |

Conclusion:

> Final/RLVR keeps a DPO-like aggregate style while changing selected register
> markers and reducing several headline slop measurements.

## Feature-By-Feature Conclusions

| Feature | Phase 2 conclusion |
|---|---|
| `slop_lexicon` | Strongest amplification-like signal. DPO peaks in local neutral-normalized AF; free-running output is Base-like/DPO-like; compounding is present at all stages. |
| `rule_of_three_approx` | Strong output marker but not DPO-amplified here. Base has the highest free-running rate; SFT has the highest teacher-forced proxy. |
| `contrastive_negation` | Output-visible with SFT suppression and DPO/Final rebound. Retained teacher-forced support is too sparse for a strong AF claim. |
| `stock_openers` | Low rates. DPO is tied or near peak in free-running output, but teacher-forced raw AF falls through the ladder. |
| `stock_closers` | DPO peaks in free-running output; teacher-forced raw AF is very high because references are rare. Direction useful, magnitude fragile. |
| `stock_openers_closers` | Useful pooled view. DPO is the free-running peak, but pooled interpretation hides the opener/closer split. |
| Biber-lite register proxies | Descriptive style layer. Final/RLVR is less Base-like on many modal/pronoun/demonstrative markers and remains close to DPO overall. |

## Plain-English Answers To The Core Questions

| Question | Phase 2 answer |
|---|---|
| Did later checkpoints simply get more slop-heavy? | No. The progression is non-monotonic. SFT is often lowest in sampled output, DPO rebounds on selected markers, and Final/RLVR does not broadly exceed DPO. |
| Which feature has the strongest model-side amplification signal? | `slop_lexicon`. DPO is the peak checkpoint for teacher-forced neutral-normalized AF, although Base remains high in sampled output. |
| Does DPO explain every measured slop feature? | No. `rule_of_three_approx` is Base-heavy in sampled output and SFT-peaked in its teacher-forced proxy. |
| Did we measure self-conditioning? | Yes. For `slop_lexicon`, later windows are about 6x-9x more likely to contain another hit after a prior hit has already appeared. |
| What is the prior-vs-after-prior result? | For `slop_lexicon`, `P(hit after prior)` is `0.069-0.105` depending on stage, while `P(hit without prior)` is about `0.011`. That is the clearest compounding signal. |
| Did we measure Biber-style features? | Yes, as `biber_lite` regex proxies on generated outputs and Phase 1 corpus samples. This is a register/style comparison layer, not full pybiber. |
| What is Final/RLVR closest to? | DPO, under the assembled style signature. Final/RLVR is DPO-like overall but lower than DPO on several headline slop rates. |
| What should Phase 3 inherit from Phase 2? | A feature-by-feature amplification spectrum, not a single global slop score. |

## One-Page Conclusion

Phase 2 shows that the OLMo 3 Instruct ladder has no single slop trajectory.
The base checkpoint already has substantial measured style. SFT frequently
reduces visible Tier-1 markers in sampled output. DPO brings selected markers
back up, especially `slop_lexicon` local propensity and stock phrase
emissions. Final/RLVR remains closest to DPO in aggregate output style, but it
does not broadly increase the measured slop features beyond DPO.

The most defensible positive claim is about `slop_lexicon`. Its
teacher-forced neutral-normalized amplification factor peaks at DPO, and
generated answers show clear self-conditioning: once a `slop_lexicon` item has
appeared, later windows are much more likely to contain another one. This
means sampled style cannot be explained as independent local token choices.
The model conditions on its own earlier wording.

The most important negative claim is that DPO is not a universal slop
generator. `rule_of_three_approx` is the highest-rate visible feature, but it
is highest in Base sampled output and does not show a DPO peak in the retained
teacher-forced proxy. `contrastive_negation` is output-visible but does not
yet have enough teacher-forced support for a strong AF claim. Stock openers and
closers are DPO-peaked in sampled output, but their teacher-forced behavior is
split and sensitive to rare-reference denominators.

Biber-lite measurements broaden the conclusion. Final/RLVR is not just DPO
with identical wording. It is lower than Base on many modal, demonstrative,
first-person, and infinitive markers, while higher than SFT targets on several
question, hedge, verb, pronoun, modal, and subordination proxies. The assembled
style signature says Final/RLVR is closest to DPO, then SFT, then Base.

The correct carry-forward claim is therefore feature-level and mechanistic:
some markers are inherited, some are SFT-suppressed, some rebound at DPO, some
compound during generation, and broader register shifts do not map cleanly to
any one slop feature. Phase 3 should classify features along that spectrum
instead of collapsing them into one score.

## Checkpoint-By-Checkpoint Conclusions

### Base

Base already contains much of the measured style. It is the target-shape
free-running maximum for `rule_of_three_approx` and `slop_lexicon`, and it is
high on Biber-lite demonstratives, first-person pronouns, infinitives, and
modal markers.

This rules out a naive story where slop begins only at preference
optimization. Some measured behaviors are inherited from pretraining, from the
base model distribution, or from earlier pipeline factors not isolated here.

### SFT

SFT often suppresses visible free-running emissions. In the retained
target-shape grid, SFT is the lowest checkpoint for every Tier-1 free-running
feature.

SFT is not simply "less slop" by every measurement. It raises teacher-forced
`slop_lexicon` normalized AF above Base and has the highest point estimate for
the `rule_of_three_approx` comma-pair extension proxy.

The practical lesson is that local propensity and sampled-output rate can move
differently.

### DPO

DPO is the clearest post-SFT rebound. It is the maximum point estimate for
teacher-forced `slop_lexicon` normalized AF and the free-running maximum for
stock opener/closer rates.

DPO is also close to Base for `slop_lexicon` free-running output. However, DPO
is not the maximum for `rule_of_three_approx`, and Phase 1 did not show a
positive DPO chosen-vs-rejected advantage for `slop_lexicon`.

This makes the DPO result feature-specific rather than a clean "DPO chosen
data contains more of every slop feature" story.

### Final/RLVR

Final/RLVR is closest to DPO in the assembled output signature. It does not
broadly increase slop beyond DPO.

Compared with DPO, Final/RLVR is lower on `slop_lexicon` free-running rate,
`rule_of_three_approx` free-running rate, pooled stock phrase free-running
rate, and `rule_of_three_approx` observed compounding opportunities.

Compared with DPO, Final/RLVR is higher on selected Biber-lite register
markers, including first-person pronouns, causal subordinators, hedges,
infinitives, passive-voice approximation, public verbs, and possibility
modals.

The practical read is that Final/RLVR keeps a DPO-like style signature while
partially smoothing or redirecting several DPO-stage movements.

## Hypothesis Status

| Hypothesis | Phase 2 status | Reason |
|---|---|---|
| H1: inheritance vs. amplification | Partially supported, feature-specific | Base is already high on several visible features, while `slop_lexicon` still shows a DPO local-propensity peak. |
| H2: preference-signal complicity | Not decided by Phase 2 alone | Phase 2 supplies model-side stage effects. Phase 3 must join them to Phase 1 chosen/rejected evidence. |
| H3: compounding | Supported for `slop_lexicon` | Observed generated opportunity rates exceed teacher-forced expectations, and prior-hit windows have much higher later-hit risk. |
| H4: stage localization | Supported as non-monotonic localization | Base-heavy, SFT-suppressed, DPO-rebound, and Final/RLVR-attenuated patterns all appear, depending on feature. |

The most important H2 warning sign is `slop_lexicon`: DPO teacher-forced AF
peaks even though DPO chosen responses are not higher than rejected responses
for that feature. That points away from a simple "chosen data contains more
slop lexicon" explanation.

## What Phase 2 Does Not Prove

Phase 2 does not prove that DPO universally creates slop. The DPO signal is
strongest for `slop_lexicon` local propensity and stock phrase free-running
rates, not every feature.

Phase 2 does not prove that Final/RLVR is worse than DPO on slop. Final/RLVR
is DPO-like in aggregate style, but several headline slop rates are lower than
DPO.

Phase 2 does not prove every feature with equal reliability. `slop_lexicon`
has the strongest measurement chain. `contrastive_negation` is retained mainly
as a free-running result because teacher-forced reference support is sparse.
`rule_of_three_approx` teacher-forced scoring uses a bounded third-item
extension proxy.

Phase 2 does not provide cross-ladder generalization. The SmolLM3 no_think
replication and OLMo-vs-SmolLM3 comparison belong to Phase 3.

Phase 2 does not provide full pybiber extraction. Biber-lite is a useful
surface-register layer, not full Biber category measurement.

## Main Caveats

This is a bounded OLMo 3 / Dolci close-out, not the full production-scale run.

Teacher-forced propensity and free-running generation answer different
questions. A checkpoint can assign high local probability to a feature without
making that feature the highest-rate sampled-output marker.

The retained generation setting is one target shape: 512 prompts, 8
completions per prompt, 1,024 max new tokens, temperature `1.0`, top-p `0.95`.
Other temperatures were explored but are not the final headline grid.

`rule_of_three_approx` is a regex approximation. Its teacher-forced proxy is a
third-item extension test after a candidate two-item list, not full
open-vocabulary rule-of-three initiation.

`contrastive_negation` has sparse held-out reference support in the 5,000
prompt denominator audit, so strong teacher-forced claims are not retained.

`stock_closers` teacher-forced AF has a very large magnitude because the
reference denominator is small. The qualitative direction is useful; the exact
magnitude should be handled cautiously.

Biber-lite features are regex proxies. They are useful for register comparison
but should not be presented as full pybiber extraction.

The style-signature distance uses raw metric scales. It is useful for saying
which checkpoints are closest in the assembled output signature, but it is not
a formal statistical test.

## Final Conclusion

Phase 2 supports a nuanced amplification story.

The cleanest positive claim is:

> `slop_lexicon` has the strongest amplification-like model-side signal. DPO
> is the peak checkpoint for teacher-forced neutral-normalized propensity, and
> generated text shows positive self-conditioning once a `slop_lexicon` hit
> has appeared.

The cleanest negative claim is:

> The OLMo 3 ladder does not show broad monotonic slop amplification. Base is
> already high on several markers, SFT suppresses visible Tier-1 emissions,
> DPO rebounds on selected features, and Final/RLVR does not broadly exceed
> DPO.

The best checkpoint progression summary is:

> Base supplies substantial inherited style. SFT reshapes and often suppresses
> visible emissions. DPO raises selected markers, especially local
> `slop_lexicon` propensity and stock phrase output. Final/RLVR remains
> closest to DPO in aggregate but attenuates several headline slop rates while
> shifting broader register markers.

For someone entering Phase 3, the most important lesson is that features
should be classified one by one. The evidence supports a feature-level
amplification spectrum: inherited/base-heavy features, SFT-suppressed
features, DPO-rebound features, compounding-sensitive features, and broader
register changes that are not themselves causal slop measurements.

## Handoff To Phase 3

Phase 3 should treat this report as the OLMo / Dolci model-side input. The
next job is to classify features by combining:

- Phase 1 corpus rates and preference-pair evidence;
- Phase 2 teacher-forced AF;
- Phase 2 free-running emission;
- Phase 2 compounding evidence;
- Phase 2 Biber-lite style-signature evidence.

Immediate Phase 3 interpretations implied by this report:

- `slop_lexicon` is the main candidate for dynamics-driven or mixed
  amplification because DPO model propensity rises without a clean DPO
  chosen-over-rejected corpus advantage.
- `rule_of_three_approx` should be treated as inherited/base-heavy or
  output-visible rather than DPO-amplified in the OLMo slice.
- Stock phrases need opener/closer separation before making a causal claim.
- `contrastive_negation` needs better teacher-forced support before it can be
  classified beyond output-visible rebound.
- Biber-lite should be used as the broader final-output style signature, not
  as causal slop evidence.
- SmolLM3 no_think replication is required before presenting these as
  cross-model results.

## Primary Artifacts

Final report inputs:

- `artifacts/phase2/analysis/olmo3_amplification_spectrum_single_temp_t1_v6.csv`
- `artifacts/phase2/analysis/olmo3_amplification_spectrum_single_temp_t1_v6_summary.md`
- `artifacts/phase2/analysis/olmo3_generation_stage_grid_target_shape_512prompt_8comp_t1_1024.csv`
- `artifacts/phase2/analysis/olmo3_generation_stage_grid_target_shape_512prompt_8comp_t1_1024_summary.md`
- `artifacts/phase2/analysis/olmo3_generation_compounding_target_shape_512prompt_8comp_t1_1024_tf1024.csv`
- `artifacts/phase2/analysis/olmo3_generation_compounding_target_shape_512prompt_8comp_t1_1024_tf1024_summary.md`
- `artifacts/phase2/analysis/olmo3_biber_lite_generation_vs_corpus_t1.csv`
- `artifacts/phase2/analysis/olmo3_biber_lite_generation_vs_corpus_t1_summary.md`
- `artifacts/phase2/analysis/olmo3_style_signature_t1.csv`
- `artifacts/phase2/analysis/olmo3_style_signature_t1_stage_distances.csv`
- `artifacts/phase2/analysis/olmo3_style_signature_t1_summary.md`
- `artifacts/phase2/analysis/olmo3_phase2_single_temp_t1_final_artifact_manifest.md`

Important W&B runs:

| Artifact family | W&B run |
|---|---|
| Final amplification spectrum | `jxqcadhe` |
| Biber-lite generation-vs-corpus comparison | `eu4glzoq` |
| Final output style signature | `bt0zelup` |
| Final artifact manifest | `5qonr7io` |
