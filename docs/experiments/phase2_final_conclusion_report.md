# Phase 2 Final Conclusion Report

Date: 2026-06-15

Status: final bounded OLMo 3/Dolci Phase 2 close-out.

Audience: this document is written for a reader who has not followed the
working thread. It explains the question, the measurements, the evidence, and
the limits of the conclusion.

## Executive Summary

Phase 2 asked whether stylistic features associated with "LLM slop" are merely
present in training data or become more likely as a model moves through
post-training checkpoints.

The short answer is nuanced:

> In the bounded OLMo 3/Dolci Phase 2 slice, DPO is the clearest checkpoint for
> increased local `slop_lexicon` propensity, but the final/RLVR checkpoint does
> not broadly amplify slop beyond DPO. The final model is closest to DPO in the
> assembled output-style signature while partially attenuating several DPO and
> base-heavy markers.

The most important finding is that there is no single monotonic curve where
base is clean, SFT is sloppier, DPO is sloppier still, and final/RLVR is
sloppiest. The actual progression is feature-specific:

- Base already carries substantial measured style.
- SFT suppresses many visible free-running Tier-1 features.
- DPO rebounds on selected features, especially `slop_lexicon` local
  propensity and stock phrase free-running rates.
- Final/RLVR remains DPO-like overall but is not simply "more DPO."

The strongest positive result is `slop_lexicon`: DPO has the highest
teacher-forced neutral-normalized amplification factor, and generated text
shows a self-conditioning signal once a slop-lexicon hit has already appeared.
The strongest negative result is the rejection of a broad DPO-amplifies-everything
story: `rule_of_three_approx` free-running emissions peak at base, its
teacher-forced proxy peaks at SFT, and final/RLVR is below DPO on several
headline slop rates.

## Project Context

The broader project studies whether common LLM output tics can be decomposed
into three sources:

1. Inheritance from pretraining or earlier model behavior.
2. Imitation of SFT or preference-data style.
3. Amplification introduced by model optimization and free-running generation.

Phase 1 measured data. It counted feature rates in corpus roles such as
pretraining samples, SFT targets, DPO chosen responses, and DPO rejected
responses.

Phase 2 measured models. It asked whether checkpoints along the OLMo 3 ladder
become more likely to produce the same features when prompted with the same
held-out SFT prompts.

Phase 3 uses the Phase 1 and Phase 2 outputs together. Phase 2 does not, by
itself, decide whether a feature was caused by the preference signal. It
provides the model-side measurements needed for that later classification.

## What Was Measured

This report covers the bounded final Phase 2 close-out, not the full production
grid originally described in `EXPERIMENTS.md`.

The model ladder is:

| Stage | Report label | Meaning |
|---|---|---|
| Base | Base | OLMo 3 base checkpoint in the measured ladder. |
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

That makes 16,384 sampled completions across the four checkpoints. Generated
token counts differ by stage because some completions stop earlier than the
maximum.

A DPO-only scale check at 1,024 prompts x 8 completions was also retained. It
was used mainly to check whether the DPO `slop_lexicon` and compounding
signals were stable at a larger prompt count.

Earlier short generations at temperatures `0.0`, `0.7`, and `1.0` remain as
sensitivity context. The final conclusion here uses the target-shape
single-temperature grid at `t=1.0`.

This close-out does not include the full 5,000-prompt x 8-completion x
3-temperature grid, the SmolLM3 replication ladder, or full `pybiber`
extraction.

## Feature Glossary

The main Phase 2 features come from the revised Phase 1 feature set.

| Feature | Plain meaning | Phase 2 support |
|---|---|---|
| `slop_lexicon` | Curated lexical slop terms and phrases. | Strongest support: teacher-forced normalized AF, free-running rates, and compounding join. |
| `rule_of_three_approx` | Regex approximation of three-part rhetorical lists. | Strong free-running support; teacher-forced proxy only measures third-item extension after a two-item list. |
| `contrastive_negation` | "Not X, but Y" contrastive patterns. | Free-running measured; retained teacher-forced support is weak because held-out references are sparse. |
| `stock_openers` | Formulaic opening phrases. | Teacher-forced and free-running measured. |
| `stock_closers` | Formulaic closing phrases. | Teacher-forced and free-running measured; teacher-forced magnitude has a small-denominator caveat. |
| `stock_openers_closers` | Pooled opener/closer view. | Useful summary, but it hides opener-vs-closer differences. |

The word "slop" in this report means these operationalized detectors. It is
not a broad quality judgment over whole completions.

Phase 2 also measured Biber-lite register proxies on the generated outputs.
Those are broader style descriptors, not slop-propensity measurements.

## Metric Glossary

Phase 2 has several metric families. They answer different questions and should
not be collapsed into one score.

### Phase 1 Corpus Rates

Corpus rates come from Phase 1 and are usually reported per 1,000 simple
tokens. They answer: how often does a feature appear in a data role?

For example, a feature might be common in DPO chosen responses, rare in DPO
rejected responses, or already common in pretraining samples. That context is
essential, but corpus rates do not directly measure what the model assigns
probability to under fixed opportunities.

### Teacher-Forced Amplification Factor

Teacher-forced amplification factor, or AF, measures local model propensity.
The model is shown a fixed prefix, and the harness asks how much probability
mass the model assigns to feature-bearing continuations at defined
opportunities.

For `slop_lexicon`, the main value is neutral-normalized AF. This divides the
slop continuation probability by a neutral-control continuation probability,
so a generic rise in continuation confidence is less likely to be mistaken for
a slop-specific effect.

For other features, the retained AF is usually raw AF. Raw AF is still useful,
but it is more sensitive to denominator definitions and rare references.

### Free-Running Emission Rate

Free-running rate measures what appears in sampled completions. It is reported
as feature hits per 1,000 generated tokens under the fixed decoding settings.

This is closest to user-visible behavior, but it is affected by prompt mix,
length, decoding temperature, top-p, and the model conditioning on its own
previous tokens.

### Compounding

Compounding asks whether a feature becomes more likely after it has already
appeared earlier in the same completion.

There are two views:

- Expected-vs-observed: compare observed generated opportunity rates to rates
  predicted from teacher-forced probability mass.
- Prior/no-prior windows: compare feature-hit risk in later windows after a
  prior hit with windows where no prior hit has occurred.

The prior/no-prior view is the direct answer to the earlier "prior vs after
prior" question.

### Biber-Lite Register Proxies

Biber-lite measures regex-based surface proxies for broad register categories:
pronouns, modals, hedges, amplifiers, verb classes, nominalizations,
complements, subordination, wh-questions, and passive-voice approximations.

These are descriptive style measurements. They are not causal measurements and
they are not full Biber categories.

## Phase 1 Baseline Context

The Phase 1 corpus rates establish what the data already contained before
looking at model propensity.

| Corpus / role | Contrastive negation | Rule of three | Slop lexicon | Stock openers | Stock closers | Openers + closers |
|---|---:|---:|---:|---:|---:|---:|
| Dolma pretrain sample | 0.389 | 3.946 | 0.238 | 0.027 | 0.008 | 0.035 |
| Dolci SFT target | 0.139 | 1.937 | 0.553 | 0.262 | 0.089 | 0.351 |
| Dolci DPO chosen | 0.459 | 3.696 | 0.990 | 0.235 | 0.413 | 0.648 |
| Dolci DPO rejected | 0.369 | 3.391 | 1.022 | 0.277 | 0.292 | 0.569 |

Several points matter:

- The pretrain sample is already high on `rule_of_three_approx` and
  contrastive negation.
- SFT targets are higher than the pretrain sample on `slop_lexicon` and stock
  opener/closer phrases.
- DPO chosen responses are higher than rejected responses for contrastive
  negation, rule-of-three, stock closers, and pooled stock phrases.
- DPO chosen responses are not higher than rejected responses for
  `slop_lexicon`.

That last point is important. If DPO-stage `slop_lexicon` propensity rises, it
is not cleanly explained by Dolci chosen responses simply containing more
slop-lexicon hits than rejected responses.

## Result 1: `slop_lexicon` Has The Clearest DPO Local-Propensity Peak

`slop_lexicon` is the strongest Phase 2 result because it has the most complete
measurement chain.

Teacher-forced neutral-normalized AF:

| Stage | Normalized AF | 95% CI |
|---|---:|---:|
| Base | 1.467 | 1.114-2.046 |
| SFT | 1.695 | 1.315-2.289 |
| DPO | 1.999 | 1.446-2.837 |
| Final/RLVR | 1.659 | 1.189-2.351 |

The DPO point estimate is highest. Final/RLVR falls back toward SFT. The
confidence intervals overlap, so this should be read as a bounded positive
signal, not as a precise proof that DPO is statistically isolated from every
other stage.

Free-running `slop_lexicon` emission is more mixed:

| Stage | Free-run /1k generated tokens |
|---|---:|
| Base | 0.233 |
| SFT | 0.171 |
| DPO | 0.229 |
| Final/RLVR | 0.193 |

Base is narrowly above DPO in the target-shape generations. DPO is very close
to base and clearly above SFT. Final/RLVR is below DPO.

Conclusion: DPO raises local `slop_lexicon` propensity, but visible sampled
output does not become a clean DPO-only peak. Base already emits the feature at
a comparable rate under this setup.

## Result 2: `slop_lexicon` Self-Conditions During Generation

The expected-vs-observed compounding join for `slop_lexicon` is positive in all
four stages.

| Stage | Observed /1k opportunities | Expected /1k opportunities | Excess /1k | Realized AF | Repeat generations |
|---|---:|---:|---:|---:|---:|
| Base | 0.615 | 0.232 | 0.383 | 1.051 | 192 |
| SFT | 0.697 | 0.336 | 0.361 | 1.192 | 145 |
| DPO | 0.638 | 0.439 | 0.199 | 1.091 | 199 |
| Final/RLVR | 0.582 | 0.445 | 0.137 | 0.994 | 180 |

Observed generated opportunity rates exceed teacher-forced expectations in
every stage. This supports the bounded compounding claim: once generation is
allowed to condition on itself, `slop_lexicon` appears more often than the
fixed-local propensity estimate alone would predict.

The direct prior/no-prior window test is even clearer:

| Stage | P(hit after prior) | P(hit without prior) | Risk difference | Risk ratio |
|---|---:|---:|---:|---:|
| Base | 0.0812 | 0.0106 | 0.0707 | 7.68 |
| SFT | 0.1052 | 0.0113 | 0.0939 | 9.35 |
| DPO | 0.0712 | 0.0113 | 0.0599 | 6.29 |
| Final/RLVR | 0.0691 | 0.0107 | 0.0584 | 6.43 |

After a `slop_lexicon` hit has appeared, nearby later windows are about six to
nine times more likely to contain another hit than windows with no prior hit.

This effect is not DPO-specific. Base and SFT show strong compounding too. The
important claim is about free-running self-conditioning, not only preference
optimization.

## Result 3: SFT Suppresses Many Visible Tier-1 Emissions

SFT is the low-emission checkpoint for every retained Tier-1 free-running row
in the target-shape grid.

| Feature | Base | SFT | DPO | Final/RLVR | Max stage |
|---|---:|---:|---:|---:|---|
| `rule_of_three_approx` | 1.019 | 0.488 | 0.861 | 0.802 | Base |
| `slop_lexicon` | 0.233 | 0.171 | 0.229 | 0.193 | Base |
| `contrastive_negation` | 0.136 | 0.041 | 0.141 | 0.142 | Final/RLVR |
| `stock_openers_closers` | 0.092 | 0.072 | 0.108 | 0.091 | DPO |
| `stock_openers` | 0.064 | 0.052 | 0.064 | 0.059 | DPO |
| `stock_closers` | 0.028 | 0.020 | 0.043 | 0.032 | DPO |

This is one of the most important model-progression results. If we only looked
at SFT -> DPO, we might say DPO adds slop. Looking at Base -> SFT -> DPO ->
Final/RLVR shows a different pattern: SFT suppresses many emissions, and DPO
often rebounds from that suppressed SFT point.

## Result 4: `rule_of_three_approx` Is Real, But Not DPO-Amplified Here

`rule_of_three_approx` is the most common retained Tier-1 feature in sampled
outputs, but the stage pattern does not support a DPO-specific amplification
claim.

Free-running rate per 1,000 generated tokens:

| Stage | Rate |
|---|---:|
| Base | 1.019 |
| SFT | 0.488 |
| DPO | 0.861 |
| Final/RLVR | 0.802 |

Base is highest. DPO and final/RLVR are above SFT but below base.

The teacher-forced proxy points in the same cautious direction. It measures
third-item extension after a candidate two-item comma/list context; it does
not measure open-vocabulary initiation of a whole rule-of-three construction.

| Stage | Raw AF | 95% CI |
|---|---:|---:|
| Base | 0.721 | 0.698-0.742 |
| SFT | 0.767 | 0.743-0.787 |
| DPO | 0.716 | 0.691-0.738 |
| Final/RLVR | 0.723 | 0.698-0.746 |

SFT has the highest point estimate for this proxy. DPO is slightly below base.
All stages are below the held-out reference extension rate.

Conclusion: `rule_of_three_approx` is a useful output marker, but Phase 2 does
not support calling it DPO-amplified in this OLMo slice.

## Result 5: Stock Openers And Closers Should Not Be Collapsed Too Early

The pooled `stock_openers_closers` feature is useful, but it hides a major
opener-vs-closer split.

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
small: 18 positives in 41,187 opportunities. The qualitative direction is
useful, but the exact magnitude should be handled cautiously.

Free-running stock phrase rates are small and DPO-peaked:

| Feature | Base | SFT | DPO | Final/RLVR |
|---|---:|---:|---:|---:|
| `stock_openers` | 0.064 | 0.052 | 0.064 | 0.059 |
| `stock_closers` | 0.028 | 0.020 | 0.043 | 0.032 |
| `stock_openers_closers` | 0.092 | 0.072 | 0.108 | 0.091 |

Conclusion: DPO is the visible sampled-output peak for stock phrases, but the
large teacher-forced AF is mostly a closer effect, not an opener effect.

## Result 6: `contrastive_negation` Is Output-Visible But Teacher-Forced Sparse

`contrastive_negation` has a clear SFT-suppression pattern in free-running
generation:

| Stage | Free-run /1k generated tokens |
|---|---:|
| Base | 0.136 |
| SFT | 0.041 |
| DPO | 0.141 |
| Final/RLVR | 0.142 |

Base, DPO, and final/RLVR are close; SFT is much lower.

The retained teacher-forced table does not make a strong `contrastive_negation`
claim because held-out target support is sparse: 7 reference instances in the
5,000-prompt package. The safe conclusion is therefore output-only: the
feature is visible in generated text, SFT suppresses it, and later checkpoints
rebound to roughly base-like rates.

## Result 7: Biber-Lite Shows A Broader Register Shift

Phase 2 did measure Biber-lite features on generated outputs and compared them
to Phase 1 corpus samples. This was added as a descriptive style layer after
the Tier-1 slop measurements.

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

Conclusion: the final output style is not just a few explicit lexical markers.
It also differs from the SFT target corpus in questions, hedges, verb classes,
pronouns, modals, and subordination proxies.

## Result 8: The Final Output Signature Is DPO-Like But Not DPO-Identical

The final style signature joins:

- Tier-1 free-running slop rates.
- Empirical compounding metrics.
- Biber-lite register rates.

It then computes checkpoint distances over the combined raw vector. The
distance matrix is useful for orientation, but it is not a formal statistical
test because high-rate metrics dominate the Euclidean distance.

Distances from final/RLVR:

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

Conclusion: final/RLVR keeps a DPO-like aggregate style while changing selected
register markers and reducing several headline slop measurements.

## Feature-By-Feature Conclusions

| Feature | Best Phase 2 conclusion |
|---|---|
| `slop_lexicon` | Strongest measured amplification-like signal. DPO peaks in local neutral-normalized AF; free-running output is base-like/DPO-like; compounding is present at all stages. |
| `rule_of_three_approx` | Strong output marker but not DPO-amplified here. Base has highest free-running rate; SFT has highest teacher-forced proxy. |
| `contrastive_negation` | Output-visible with SFT suppression and DPO/final rebound. Retained teacher-forced support is too sparse for a strong AF claim. |
| `stock_openers` | Low rates; DPO is tied/near peak in free-running output, but teacher-forced raw AF falls through the ladder. |
| `stock_closers` | DPO peaks in free-running output; teacher-forced raw AF is very high because references are rare. Direction useful, magnitude fragile. |
| `stock_openers_closers` | Useful pooled view. DPO is free-running peak, but the pooled teacher-forced story is mostly driven by closers. |
| Biber-lite register proxies | Descriptive style layer. Final/RLVR is less base-like on many modal/pronoun/demonstrative markers and remains close to DPO overall. |

## Checkpoint-By-Checkpoint Interpretation

### Base

Base already contains much of the measured style. It is the target-shape
free-running maximum for `rule_of_three_approx` and `slop_lexicon`, and it is
high on Biber-lite demonstratives, first-person pronouns, infinitives, and
modal markers.

This rules out a naive story where slop begins only at preference
optimization. Some measured behaviors are inherited from pretraining or from
earlier parts of the ladder not separately isolated here.

### SFT

SFT often suppresses visible free-running emissions. In the retained
target-shape grid it is the lowest checkpoint for every Tier-1 free-running
feature.

At the same time, SFT is not simply "less slop" by every measurement. It raises
teacher-forced `slop_lexicon` normalized AF above base and has the highest
teacher-forced point estimate for the `rule_of_three_approx` comma-pair
extension proxy.

The lesson is that local model propensity and sampled output rates can move
differently.

### DPO

DPO is the clearest post-SFT rebound. It is the maximum point estimate for
teacher-forced `slop_lexicon` normalized AF and the free-running maximum for
stock opener/closer rates.

DPO is also close to base for `slop_lexicon` free-running output. However, DPO
is not the maximum for `rule_of_three_approx`, and Phase 1 did not show a
positive DPO chosen-vs-rejected advantage for `slop_lexicon`. That makes the
DPO result feature-specific rather than a clean "the chosen data contains more
of every slop feature" story.

### Final/RLVR

Final/RLVR is closest to DPO in the assembled output signature. It does not
broadly increase slop beyond DPO.

Compared with DPO, final/RLVR is lower on `slop_lexicon` free-running rate,
`rule_of_three_approx` free-running rate, pooled stock phrase free-running
rate, and `rule_of_three_approx` observed compounding opportunities.

Compared with DPO, final/RLVR is higher on selected Biber-lite register
markers including first-person pronouns, causal subordinators, hedges,
infinitives, passive voice approximation, public verbs, and possibility
modals.

The practical read is that final/RLVR keeps a DPO-like style signature while
partially smoothing or redirecting several DPO-stage movements.

## Original Planned Results: What Phase 2 Can Claim

The original plan in `EXPERIMENTS.md` named several hypotheses and results.
Phase 2 answers some directly and prepares others for Phase 3.

### H1: Inheritance vs Amplification

Partially supported, with a feature-specific answer.

Base already has high free-running rates for `rule_of_three_approx` and
`slop_lexicon`, so those outputs cannot be explained only as late-stage
preference amplification. `slop_lexicon` still shows a DPO teacher-forced
propensity peak, so it has an amplification-like local signal on top of
inherited/base behavior.

### H2: Preference-Signal Complicity

Phase 2 alone does not decide H2. It supplies the model-side measurements that
Phase 3 cross-references with Phase 1 chosen-vs-rejected data.

For `slop_lexicon`, the warning sign is already visible: DPO teacher-forced AF
peaks even though DPO chosen responses are not higher than rejected responses
for `slop_lexicon`. That points away from a simple "chosen data contains more
slop lexicon" explanation.

### H3: Compounding

Supported for `slop_lexicon` in the bounded sense.

Observed generated `slop_lexicon` opportunity rates exceed teacher-forced
expectations in all four stages, and prior-window risk is much higher than
no-prior-window risk. The effect is not DPO-specific.

### H4: Model Progression

Supported as a non-monotonic progression.

The observed progression is:

1. Base has substantial inherited style.
2. SFT suppresses many free-running emissions.
3. DPO raises selected features again and peaks on `slop_lexicon` local
   propensity.
4. Final/RLVR remains DPO-like in aggregate but attenuates several slop
   metrics.

This is stronger and more accurate than the simpler claim that later
checkpoints are always sloppier.

## What Phase 2 Does Not Prove

Phase 2 does not prove that DPO universally creates slop. The DPO signal is
strongest for `slop_lexicon` local propensity and stock phrase free-running
rates, not for every feature.

Phase 2 does not prove that final/RLVR is worse than DPO on slop. The final
checkpoint is DPO-like in aggregate style, but several headline slop rates are
lower than DPO.

Phase 2 does not prove that every feature is measured with equal reliability.
`slop_lexicon` has the strongest measurement chain. `contrastive_negation` is
free-running only in the retained headline table because teacher-forced
reference support is sparse. `rule_of_three_approx` teacher-forced scoring uses
a bounded proxy.

Phase 2 does not provide the cross-ladder generalization result. The SmolLM3
replication and OLMo-vs-SmolLM3 comparison belong to Phase 3.

## Caveats

This is a bounded OLMo 3/Dolci close-out, not the full production-scale run.

Teacher-forced propensity and free-running generation answer different
questions. A model can assign high local probability mass to a feature without
making that feature the most frequent sampled-output marker.

The retained generation setting is one target shape: 512 prompts, 8 completions
per prompt, 1,024 max new tokens, temperature `1.0`, top-p `0.95`. Other
temperatures were explored but are not the final headline grid.

`rule_of_three_approx` is a regex approximation. Its teacher-forced proxy is a
third-item extension test after a candidate two-item list, not full
open-vocabulary rule-of-three initiation.

`contrastive_negation` has sparse held-out reference support in the 5,000
prompt denominator audit, so strong teacher-forced claims are not retained.

`stock_closers` teacher-forced AF has a very large magnitude because the
reference denominator is small. The qualitative direction is useful; the exact
magnitude should be handled cautiously.

Biber-lite features are regex proxies. They are useful for register comparison
but should not be presented as full `pybiber` extraction.

The style-signature distance uses raw metric scales. It is useful for saying
which checkpoints are closest in the assembled output signature, but it is not
a formal statistical test.

## Final Conclusion

Phase 2 supports a nuanced amplification story.

The strongest positive result is `slop_lexicon`: DPO is the peak for
teacher-forced neutral-normalized propensity, and generated text shows a
positive self-conditioning signal once a slop-lexicon hit appears. However,
free-running `slop_lexicon` output is narrowly highest at base and very close
at DPO, so the visible generation result is not a clean DPO-only peak.

The strongest negative result is the rejection of a broad monotonic slop story.
`rule_of_three_approx` free-running emissions peak at base, its teacher-forced
proxy peaks at SFT, and final/RLVR generally attenuates from DPO on several
headline slop metrics.

The model progression is best summarized as inherited style plus stage-specific
reshaping: base already carries a substantial style signature; SFT suppresses
many visible markers; DPO raises selected markers and produces the clearest
`slop_lexicon` local-propensity peak; final/RLVR remains closest to DPO overall
but does not broadly amplify slop further.

## Handoff To Phase 3

Phase 3 should treat this report as the OLMo/Dolci Phase 2 model-side input.
The Phase 3 job is to classify features by combining:

- Phase 1 corpus rates and preference-pair evidence.
- Phase 2 teacher-forced AF.
- Phase 2 free-running emission.
- Phase 2 compounding and style-signature evidence.

The immediate Phase 3 interpretation implied by this report is:

- `slop_lexicon` is the main candidate for dynamics-driven or mixed
  amplification because DPO model propensity rises without a clean DPO
  chosen-over-rejected corpus advantage.
- `rule_of_three_approx` should be treated as inherited/base-heavy or
  output-visible rather than DPO-amplified in the OLMo slice.
- Stock phrases need opener/closer separation before making a causal claim.
- `contrastive_negation` needs better teacher-forced support before it can be
  classified beyond output-visible rebound.
- Biber-lite is best used as the broader style signature, not as causal slop
  evidence.

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
- `artifacts/phase2/analysis/olmo3_phase2_single_temp_t1_final_artifact_manifest.md`

Important W&B runs:

| Artifact family | W&B run |
|---|---|
| Final amplification spectrum | `jxqcadhe` |
| Biber-lite generation-vs-corpus comparison | `eu4glzoq` |
| Final output style signature | `bt0zelup` |
| Final artifact manifest | `5qonr7io` |
