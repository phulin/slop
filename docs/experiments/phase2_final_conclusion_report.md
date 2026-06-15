# Phase 2 Conclusion Report

Date: 2026-06-15

Status: final bounded OLMo 3/Dolci Phase 2 close-out.

## Reader Orientation

This report is written for someone new to the project. It explains what Phase 2
was trying to measure, what was actually run, what the main numbers mean, and
what conclusions are supported by the retained artifacts.

The project studies whether recognizable "LLM slop" style features are merely
present in training data or become amplified as a model moves through
post-training. Phase 1 measured feature rates in data corpora. Phase 2 measured
the model checkpoints themselves: how likely they are to continue into these
features, how often the features appear in sampled generations, and whether a
feature becomes self-reinforcing once it appears.

The short conclusion is:

> In this bounded OLMo 3/Dolci Phase 2 slice, DPO is the clearest checkpoint
> for increased `slop_lexicon` local propensity, but final/RLVR does not
> broadly amplify slop beyond DPO. The final model is DPO-like in aggregate
> output style while partially attenuating several DPO and base-heavy markers.

The important nuance is that there is no single monotonic "post-training adds
slop" curve. The progression is feature-specific. Some measured style is
already strong at the base checkpoint, SFT often suppresses free-running
emissions, DPO raises several features again, and final/RLVR is closest to DPO
in aggregate style but is not simply "more DPO."

## What Phase 2 Was Meant To Answer

The original experiment plan treated Phase 2 as the novel measurement layer.
Phase 1 had counted feature rates in corpora such as pretraining samples, SFT
targets, DPO chosen responses, and DPO rejected responses. Phase 2 asked a
different question:

> Given the same held-out prompts, do checkpoints along the model ladder become
> more likely to produce the target features?

This breaks into three measurement questions.

First, teacher-forced propensity asks whether a model assigns probability mass
to a target continuation when the opportunity is fixed. This is the cleanest
local model-internals measurement in Phase 2. It is not sampled generation; it
is a probability query over controlled opportunities.

Second, free-running emission asks what appears in sampled model outputs. This
is closest to user-visible behavior, but it is affected by decoding settings,
length, prompts, and previous generated tokens.

Third, compounding asks whether a feature becomes more likely after the model
has already used it earlier in the same completion. This tests the
self-conditioning story: a small local propensity can produce a larger
realized style pattern if generated text pulls itself into a repeated mode.

At the end of Phase 2, a fourth descriptive layer was added: a Biber-lite
register/style comparison. Biber-lite features are not slop propensity
measurements. They are surface-register proxies that help describe how the
overall output style changes across checkpoints.

## What Was Actually Run

This report covers a bounded close-out, not the full production grid in
`EXPERIMENTS.md`.

The retained ladder is OLMo 3 Instruct on the Dolci slice:

| Stage | Label in this report | Meaning |
|---|---|---|
| Base | Base | Pre-instruction base checkpoint in the measured ladder. |
| SFT | SFT | Supervised fine-tuned checkpoint. |
| DPO | DPO | Preference-optimized checkpoint. |
| Final/RLVR | Final/RLVR | Final released checkpoint after the later RLVR-style stage. |

The retained free-running generation shape is:

| Setting | Value |
|---|---:|
| Prompt source | Held-out Dolci SFT prompt package |
| Prompts per stage | 512 |
| Completions per prompt | 8 |
| Generations per stage | 4,096 |
| Max new tokens per completion | 1,024 |
| Temperature | 1.0 |
| Top-p | 0.95 |

The target-shape generation grid therefore contains 16,384 completions across
the four checkpoints. Generated-token counts differ slightly by stage because
some completions end before the maximum.

There is also a DPO-only scale check at 1,024 prompts x 8 completions at the
same target shape. That check was used to test stability for DPO, especially
for `slop_lexicon`.

Earlier short generations at temperatures `0.0`, `0.7`, and `1.0` are retained
as sensitivity context, but the final report uses the single-temperature
target-shape grid at `t=1.0`.

This close-out does not include:

- the full 5,000-prompt x 8-completion x 3-temperature production grid,
- the SmolLM3 replication ladder,
- full `pybiber` extraction,
- a complete teacher-forced setup for every possible feature.

## Feature Set

The retained Phase 2 feature set is inherited from the revised Phase 1 close
out.

| Feature | Plain-language meaning | Phase 2 support |
|---|---|---|
| `slop_lexicon` | Curated lexical slop terms and phrases. | Strongest support: teacher-forced normalized AF, free-running rates, and compounding join. |
| `rule_of_three_approx` | Regex approximation of three-part rhetorical list structures. | Strong free-running support; teacher-forced proxy only measures third-item extension after a two-item list. |
| `contrastive_negation` | "Not X, but Y" style contrastive patterns. | Free-running output measured; teacher-forced support is weak because held-out references are sparse. |
| `stock_openers` | Formulaic opening phrases. | Teacher-forced and free-running measured. |
| `stock_closers` | Formulaic closing phrases. | Teacher-forced and free-running measured; teacher-forced magnitude has a small-denominator caveat. |
| `stock_openers_closers` | Pooled opener/closer view. | Useful summary but can hide the split between opener and closer behavior. |

The phrase "slop" in this report should be read as a measurement label for the
above feature detectors, not as a broad claim about output quality.

## How To Read The Metrics

Phase 2 has several metric families. They should not be collapsed into one
score.

### Corpus Rates

Corpus rates come from Phase 1. They count how often a feature appears in data
roles such as Dolci SFT targets, Dolci DPO chosen responses, Dolci DPO rejected
responses, and the bounded Dolma pretrain sample. Rates are usually per 1,000
simple tokens.

These tell us what the data contains. They do not by themselves tell us what
the model has learned to prefer under fixed opportunities.

### Teacher-Forced AF

Teacher-forced amplification factor, or AF, compares model probability mass at
defined opportunities to the reference rate in held-out targets.

For `slop_lexicon`, the headline value is neutral-normalized AF. The model's
slop probability is divided by a neutral-control probability so that a general
increase in continuation confidence is less likely to be mistaken for a slop
specific effect.

For some other features, only raw AF is available. Raw AF can be useful, but it
is more sensitive to denominator definitions and rare references.

### Free-Run Rate

Free-run rate is the count of detected feature hits per 1,000 generated tokens
in sampled completions. It describes what appears in generated text under the
fixed decoding settings above.

Free-run rates can diverge from teacher-forced propensity because generation is
recursive. The model conditions on its own previous text, not only the prompt.

### Compounding

Compounding has two interpretations in this report.

The expected-vs-observed view compares observed generated opportunity rates to
rates predicted from teacher-forced probability mass. If observed is higher
than expected, generation is producing more of the feature than the local
teacher-forced propensity alone would suggest.

The direct prior/no-prior view compares feature-hit risk in windows after a
prior hit with windows without a prior hit. This is the "prior vs after prior"
measurement. For example, for `slop_lexicon`, all four stages have much higher
window hit risk after the feature has already appeared.

### Biber-Lite

Biber-lite features are regex-based proxies for broad register markers:
pronouns, modals, hedges, infinitives, nominalizations, subordination, passive
voice approximations, and related categories.

These values are useful for describing the final output style signature, but
they are not causal measurements and are not full Biber categories.

## Phase 1 Baseline Context

The Phase 1 corpus rates are the baseline for interpreting Phase 2.

| Corpus / role | Contrastive negation | Rule of three | Slop lexicon | Stock openers | Stock closers | Openers + closers |
|---|---:|---:|---:|---:|---:|---:|
| Dolma pretrain sample | 0.389 | 3.946 | 0.238 | 0.027 | 0.008 | 0.035 |
| Dolci SFT target | 0.139 | 1.937 | 0.553 | 0.262 | 0.089 | 0.351 |
| Dolci DPO chosen | 0.459 | 3.696 | 0.990 | 0.235 | 0.413 | 0.648 |
| Dolci DPO rejected | 0.369 | 3.391 | 1.022 | 0.277 | 0.292 | 0.569 |

These Phase 1 rates already complicate a simple story. The pretrain sample is
high on `rule_of_three_approx` and contrastive negation. SFT targets are higher
than pretrain on `slop_lexicon` and stock opener/closer phrases. DPO chosen
responses are higher than rejected responses for contrastive negation,
rule-of-three, stock closers, and pooled openers/closers, but not for
`slop_lexicon`.

That last point matters: if DPO-stage `slop_lexicon` propensity rises, the
rise is not cleanly explained by chosen responses having more slop lexicon than
rejected responses in the Dolci preference data. This becomes important in the
Phase 3 interpretation.

## Main Finding 1: `slop_lexicon` Has A DPO Propensity Peak

`slop_lexicon` is the cleanest Phase 2 result because it has the most complete
measurement chain.

Teacher-forced neutral-normalized AF:

| Stage | Normalized AF | 95% CI |
|---|---:|---:|
| Base | 1.467 | 1.114-2.046 |
| SFT | 1.695 | 1.315-2.289 |
| DPO | 1.999 | 1.446-2.837 |
| Final/RLVR | 1.659 | 1.189-2.351 |

The point estimate is highest at DPO. Final/RLVR falls back toward SFT. The
confidence intervals overlap, so this should be read as a bounded positive
signal, not a precise proof that DPO is statistically isolated from every
other stage.

The free-running result is more mixed:

| Stage | Free-run /1k generated tokens |
|---|---:|
| Base | 0.233 |
| SFT | 0.171 |
| DPO | 0.229 |
| Final/RLVR | 0.193 |

Base is narrowly above DPO in the final target-shape generations. DPO is very
close to base and clearly above SFT. Final/RLVR is below DPO.

The best interpretation is that DPO increases local `slop_lexicon` propensity,
but sampled output does not become a simple DPO-only peak. Base already emits
the feature at a comparable rate under this generation setup.

## Main Finding 2: `slop_lexicon` Compounds During Generation

The expected-vs-observed compounding join for `slop_lexicon` is positive in
all four stages.

| Stage | Observed /1k opportunities | Expected /1k opportunities | Excess /1k | Realized AF | Repeat generations |
|---|---:|---:|---:|---:|---:|
| Base | 0.615 | 0.232 | 0.383 | 1.051 | 192 |
| SFT | 0.697 | 0.336 | 0.361 | 1.192 | 145 |
| DPO | 0.638 | 0.439 | 0.199 | 1.091 | 199 |
| Final/RLVR | 0.582 | 0.445 | 0.137 | 0.994 | 180 |

Observed generated opportunity rates exceed teacher-forced expectations in
every stage. That supports the bounded compounding claim: once generation is
allowed to condition on itself, the feature appears more than the fixed-local
propensity estimate alone would predict.

The stage pattern is not a DPO-only compounding story. Base has the largest
absolute excess. SFT has the highest realized AF. Final/RLVR is near the
teacher-forced reference-rate baseline.

The direct prior/no-prior window test also supports self-conditioning:

| Stage | P(hit after prior) | P(hit without prior) | Risk difference | Risk ratio |
|---|---:|---:|---:|---:|
| Base | 0.0812 | 0.0106 | 0.0706 | 7.69 |
| SFT | 0.1052 | 0.0113 | 0.0939 | 9.35 |
| DPO | 0.0712 | 0.0113 | 0.0599 | 6.29 |
| Final/RLVR | 0.0691 | 0.0107 | 0.0584 | 6.43 |

This is the clearest answer to the "prior vs after prior" question: after a
`slop_lexicon` hit has occurred, nearby later windows are roughly six to nine
times more likely to contain another hit than windows without a prior hit.

## Main Finding 3: SFT Often Suppresses Free-Running Emissions

SFT is the low-emission checkpoint for most retained free-running features.

| Feature | Base | SFT | DPO | Final/RLVR | Max stage |
|---|---:|---:|---:|---:|---|
| `rule_of_three_approx` | 1.019 | 0.488 | 0.861 | 0.802 | Base |
| `slop_lexicon` | 0.233 | 0.171 | 0.229 | 0.193 | Base |
| `contrastive_negation` | 0.136 | 0.041 | 0.141 | 0.142 | Final/RLVR |
| `stock_openers_closers` | 0.092 | 0.072 | 0.108 | 0.091 | DPO |
| `stock_openers` | 0.064 | 0.052 | 0.064 | 0.059 | DPO |
| `stock_closers` | 0.028 | 0.020 | 0.043 | 0.032 | DPO |

This is one of the most important model-progression results. If we only
compared SFT to DPO, we might say DPO adds slop. But compared with base, DPO
often looks like a rebound from SFT suppression rather than a monotonic climb
through the whole ladder.

## Main Finding 4: `rule_of_three_approx` Does Not Support A DPO Amplification Claim

`rule_of_three_approx` is the most common Tier-1 feature in free-running
generation, but its stage pattern points away from a DPO-specific claim.

Free-running rate per 1,000 generated tokens:

| Stage | Rate |
|---|---:|
| Base | 1.019 |
| SFT | 0.488 |
| DPO | 0.861 |
| Final/RLVR | 0.802 |

Base is highest. DPO and final/RLVR are above SFT, but below base.

The teacher-forced proxy tells the same cautionary story. This proxy measures
third-item extension after a candidate two-item comma/list context, not full
open-vocabulary initiation of an entire rule-of-three construction.

| Stage | Raw AF | 95% CI |
|---|---:|---:|
| Base | 0.721 | 0.698-0.742 |
| SFT | 0.767 | 0.743-0.787 |
| DPO | 0.716 | 0.691-0.738 |
| Final/RLVR | 0.723 | 0.698-0.746 |

SFT is the highest point estimate for the proxy. DPO is slightly below base.
All stages are below the held-out reference extension rate. The correct
conclusion is that `rule_of_three_approx` is a real generated-output marker,
but Phase 2 does not support calling it a DPO-amplified feature in this OLMo
slice.

## Main Finding 5: Stock Openers And Stock Closers Behave Differently

The pooled `stock_openers_closers` feature is convenient, but it hides an
important split.

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

The closer-side AF is extremely high because the held-out reference
denominator has only 18 positives in 41,187 opportunities. The direction is
meaningful: stock closer phrases are cheap local continuations for the model.
The magnitude should be read cautiously.

Free-running stock phrase rates are small but DPO-peaked:

| Feature | Base | SFT | DPO | Final/RLVR |
|---|---:|---:|---:|---:|
| `stock_openers` | 0.064 | 0.052 | 0.064 | 0.059 |
| `stock_closers` | 0.028 | 0.020 | 0.043 | 0.032 |
| `stock_openers_closers` | 0.092 | 0.072 | 0.108 | 0.091 |

The practical read is that DPO is the free-running peak for stock phrases, but
the pooled teacher-forced AF is mostly a closer effect, not an opener effect.

## Main Finding 6: Biber-Lite Shows A Broader Register Shift

Yes, Phase 2 measured Biber-lite features on the generated outputs and compared
them with Phase 1 corpus samples. This was added as a style/register layer
after the Tier-1 slop results.

Biber-lite rates are measured per 1,000 generated tokens. They are regex-based
surface proxies, not full Biber annotations.

Selected Biber-lite progression:

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

The broad pattern is that base is high on demonstratives, infinitives,
first-person pronouns, and modal language. DPO/final reduce many of those
base-heavy markers while increasing second-person pronouns, conditional
subordinators, and nominalizations.

Against SFT targets, final/RLVR is notably higher on several register proxies:

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

These results matter because "slop signature" is not only a few explicit
lexical markers. The final output style is also shaped by pronoun use, modals,
subordination, nominal style, and question/hedge patterns.

## Output Style Signature

The final style signature joins three generated-output views:

- Tier-1 slop free-running rates,
- empirical compounding metrics,
- Biber-lite register rates.

It then computes checkpoint distances over the combined raw vector. The
distance matrix is useful as an orientation aid, but not as a formal
inferential statistic, because high-rate metrics dominate Euclidean distance.

Raw-vector distances from final/RLVR:

| Comparison | Shared features | Euclidean distance | Cosine distance |
|---|---:|---:|---:|
| Final/RLVR vs DPO | 43 | 30.567 | 0.000075 |
| Final/RLVR vs SFT | 43 | 35.564 | 0.000287 |
| Final/RLVR vs Base | 43 | 85.139 | 0.001272 |

Under this signature, final/RLVR is closest to DPO, then SFT, then base.

Largest final-vs-base shifts:

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

Largest final-vs-DPO shifts:

| Group | Metric | Feature | Final | DPO | Delta |
|---|---|---|---:|---:|---:|
| Compounding | Observed /1k opportunities | `rule_of_three_approx` | 428.106 | 458.417 | -30.311 |
| Compounding | Prior risk ratio | `stock_closers` | 9.399 | 6.198 | +3.201 |
| Biber-lite | Generation /1k tokens | First-person pronouns | 11.638 | 10.302 | +1.336 |
| Biber-lite | Generation /1k tokens | Causal subordinators | 3.831 | 2.856 | +0.975 |
| Biber-lite | Generation /1k tokens | Hedges | 1.868 | 1.109 | +0.759 |
| Biber-lite | Generation /1k tokens | Infinitives | 13.623 | 13.009 | +0.614 |

The style-signature conclusion is that final/RLVR is DPO-like but not
DPO-identical. It preserves the broad post-DPO style while shifting selected
register markers and attenuating several headline slop metrics.

## Checkpoint-By-Checkpoint Progression

### Base

Base already contains a substantial share of the measured style. It is highest
in final target-shape free-running generation for `rule_of_three_approx` and
`slop_lexicon`. It is also high on demonstratives, first-person pronouns,
infinitives, and modal markers.

This rules out a naive story where slop begins at preference optimization.
Some measured behaviors are inherited from pretraining or earlier stages not
separately isolated by this ladder.

### SFT

SFT often suppresses visible free-running emissions. It is the lowest
checkpoint for every retained Tier-1 free-running row in the final target-shape
grid.

At the same time, SFT is not simply "less slop" under every measurement. It
raises teacher-forced `slop_lexicon` normalized AF above base, and it has the
highest teacher-forced point estimate for the `rule_of_three_approx`
comma-pair extension proxy.

The key lesson is that local probability mass and sampled output rates can
move differently.

### DPO

DPO is the clearest post-SFT rebound. It is the maximum point estimate for
teacher-forced `slop_lexicon` normalized AF and the free-running maximum for
stock opener/closer rates.

DPO is also close to base for `slop_lexicon` free-running output. However, DPO
is not the maximum for `rule_of_three_approx`, and the Phase 1 DPO chosen-vs
rejected data does not show a positive `slop_lexicon` chosen advantage. This
is why the DPO result should be interpreted as feature-specific model
reshaping, not a clean "the chosen data contains more of every slop feature"
story.

### Final/RLVR

Final/RLVR is closest to DPO in aggregate output style. It does not broadly
increase slop beyond DPO.

Compared with DPO, final/RLVR is lower on:

- `slop_lexicon` free-running rate,
- `rule_of_three_approx` free-running rate,
- pooled stock opener/closer free-running rate,
- `rule_of_three_approx` observed compounding opportunities,
- conditional subordinators.

Compared with DPO, final/RLVR is higher on:

- first-person pronouns,
- causal subordinators,
- hedges,
- infinitives,
- passive voice approximation,
- public verbs,
- possibility modals,
- stock closer prior-window risk ratio.

The practical read is that final/RLVR keeps a DPO-like style signature while
partially smoothing or redirecting several DPO-stage movements.

## Original Planned Results: What We Can Claim Now

The original plan had several named hypotheses and results. Phase 2 answers
some of them directly and sets up others for Phase 3.

### H1: Inheritance vs Amplification

Partially supported, but feature-specific.

Base already has high free-running rates for `rule_of_three_approx` and
`slop_lexicon`, so those outputs cannot be explained only as late-stage
preference amplification. `slop_lexicon` still shows a DPO teacher-forced
propensity peak, so it has an amplification-like local signal on top of
inherited/base behavior.

### H2: Preference-Signal Complicity

Phase 2 alone does not decide H2; it provides the model-side measurements that
Phase 3 cross-references with Phase 1 chosen-vs-rejected data.

For `slop_lexicon`, the relevant warning sign is already visible: DPO
teacher-forced AF peaks even though Phase 1 DPO chosen responses are not higher
than rejected responses for the slop lexicon. That points away from a simple
"chosen data contains more slop lexicon" explanation.

### H3: Compounding

Supported for `slop_lexicon` in the bounded sense.

Observed generated `slop_lexicon` opportunity rates exceed teacher-forced
expectations in all four stages, and prior-window risk is much higher than
no-prior-window risk. The effect is not DPO-specific.

### H4: Model Progression

Supported as a non-monotonic progression.

The actual pattern is:

- base has substantial inherited style,
- SFT suppresses many free-running emissions,
- DPO raises selected features again and peaks on `slop_lexicon` local
  propensity,
- final/RLVR remains DPO-like in aggregate but attenuates several slop metrics.

That is a stronger and more accurate conclusion than "later checkpoints are
always sloppier."

## What Phase 2 Does Not Prove

Phase 2 does not prove that DPO universally creates slop. The DPO signal is
strongest for `slop_lexicon` local propensity and stock phrase free-running
rates, not for every feature.

Phase 2 does not prove that final/RLVR is worse than DPO on slop. The final
checkpoint is DPO-like in aggregate style, but many headline slop rates are
lower than DPO.

Phase 2 does not prove that every feature is measured with equal reliability.
`slop_lexicon` has the strongest measurement chain. `contrastive_negation` is
free-running only in the retained headline table because the held-out
teacher-forced reference support is sparse. `rule_of_three_approx`
teacher-forced scoring uses a bounded proxy.

Phase 2 does not provide the full cross-ladder generalization result. That is
the Phase 3/replication work.

## Caveats

This is a bounded OLMo 3/Dolci close-out, not the full production-scale
`EXPERIMENTS.md` run.

Teacher-forced propensity and free-running generation answer different
questions. A model can assign high local probability mass to a feature without
making that feature the most frequent sampled-output marker.

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

The strongest positive result is `slop_lexicon`: DPO is the peak for
teacher-forced neutral-normalized propensity, and generated text shows a
positive self-conditioning/compounding signal. However, free-running
`slop_lexicon` output is narrowly highest at base and very close at DPO, so the
visible generation result is not a clean DPO-only peak.

The strongest negative result is the rejection of a broad monotonic slop
story. `rule_of_three_approx` free-running emissions peak at base, its
teacher-forced proxy peaks at SFT, and final/RLVR generally attenuates from DPO
on several headline slop metrics.

The model progression is best summarized as inherited style plus
stage-specific reshaping: base already carries a substantial style signature;
SFT suppresses many visible slop markers; DPO raises selected markers and
produces the clearest `slop_lexicon` local-propensity peak; final/RLVR remains
closest to DPO overall but does not broadly amplify slop further.

## Primary Artifacts

Final report inputs:

- `artifacts/phase2/analysis/olmo3_amplification_spectrum_single_temp_t1_v6.csv`
- `artifacts/phase2/analysis/olmo3_amplification_spectrum_single_temp_t1_v6_summary.md`
- `artifacts/phase2/analysis/olmo3_generation_stage_grid_target_shape_512prompt_8comp_t1_1024.csv`
- `artifacts/phase2/analysis/olmo3_generation_compounding_target_shape_512prompt_8comp_t1_1024_tf1024.csv`
- `artifacts/phase2/analysis/olmo3_generation_compounding_target_shape_512prompt_8comp_t1_1024_tf1024_summary.md`
- `artifacts/phase2/analysis/olmo3_biber_lite_generation_vs_corpus_t1.csv`
- `artifacts/phase2/analysis/olmo3_biber_lite_generation_vs_corpus_t1_summary.md`
- `artifacts/phase2/analysis/olmo3_style_signature_t1.csv`
- `artifacts/phase2/analysis/olmo3_style_signature_t1_stage_distances.csv`
- `artifacts/phase2/analysis/olmo3_style_signature_t1_summary.md`

Important W&B runs:

| Artifact family | W&B run |
|---|---|
| Final amplification spectrum | `jxqcadhe` |
| Biber-lite generation-vs-corpus comparison | `eu4glzoq` |
| Final output style signature | `bt0zelup` |
| Final artifact manifest | `5qonr7io` |
