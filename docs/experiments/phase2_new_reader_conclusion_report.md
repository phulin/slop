# Phase 2 Full Conclusion Report For New Readers

Date: 2026-06-15

Status: bounded OLMo 3 / Dolci Phase 2 conclusion.

This is the readable entry point for Phase 2. It assumes no prior project
context and explains what was measured, why the measurements matter, what was
actually run, how to read the main numbers, how the OLMo 3 checkpoints changed
from Base to SFT to DPO to Final/RLVR, and what conclusions should carry into
Phase 3.

For audit-level detail, see the longer close-out report in
`docs/experiments/phase2_final_conclusion_report.md` and the retained artifact
manifest in
`artifacts/phase2/analysis/olmo3_phase2_single_temp_t1_final_artifact_manifest.md`.

## Quick Orientation

Phase 2 is the model-behavior half of the project. Phase 1 counted style
features in data; Phase 2 asks whether trained checkpoints locally prefer
those features, whether the features show up in generated answers, and whether
one early feature hit makes later hits more likely.

The report has three important reading rules:

1. "Slop" means detector-defined surface style markers, not a judgment that an
   answer is wrong or useless.
2. There is no single global slop score. The result is a feature-by-feature
   spectrum.
3. Teacher-forced propensity, sampled-output frequency, and compounding are
   different measurements. When they disagree, the disagreement is part of the
   result.

## Executive Summary

The project studies where recognizable assistant-style "slop" enters a model
pipeline. Here, "slop" is not a global quality judgment. It means a set of
measurable style markers: stock lexical items, formulaic openers and closers,
contrastive phrasing, three-part rhetorical lists, and broader register
features measured with Biber-lite proxies.

Phase 1 measured the data side: pretraining-like text, Dolci SFT targets, and
Dolci DPO chosen/rejected responses.

Phase 2 measured the model side on the OLMo 3 Instruct checkpoint ladder:
Base, SFT, DPO, and Final/RLVR. It asked whether each checkpoint locally
prefers measured style features under teacher forcing, whether those features
appear in sampled completions, and whether early hits cause later hits in the
same generation.

The main conclusion is:

> Phase 2 does not support a simple story where every later checkpoint becomes
> more slop-heavy. Base already contains substantial measured style, SFT often
> suppresses visible Tier-1 markers, DPO selectively rebounds, and Final/RLVR
> remains closest to DPO in aggregate style without broadly exceeding DPO on
> the headline slop markers.

The strongest positive result is `slop_lexicon`:

- DPO has the highest teacher-forced neutral-normalized amplification factor:
  `1.999`, with 95% CI `1.446-2.837`.
- Final/RLVR falls back to `1.659`, closer to SFT than to a monotonic late-stage
  increase.
- In free-running output, Base and DPO are almost tied: Base `0.233` vs. DPO
  `0.229` hits per 1,000 generated tokens.
- Once a generation contains a `slop_lexicon` hit, later windows are about
  6x-9x more likely to contain another hit than windows with no prior hit.

The strongest negative result is also important:

> DPO does not universally amplify every measured slop feature. For
> `rule_of_three_approx`, Base has the highest sampled-output rate and the
> teacher-forced proxy peaks at SFT, not DPO.

The practical handoff is:

> Treat slop as a feature-level amplification spectrum, not as one global
> score. Some features are inherited or Base-heavy, some are SFT-suppressed,
> some rebound at DPO, some compound during generation, and Biber-lite register
> shifts describe broader final-output style rather than direct causal
> amplification.

## What Phase 2 Was Trying To Measure

The original question was not "are answers good?" A correct, helpful answer can
contain a measured marker, and a bad answer can contain none. The question was
mechanistic:

> When a model produces recognizable assistant-style surface patterns, can we
> tell whether those patterns were inherited from data, amplified by training,
> or amplified by generation dynamics?

Several forces can create the final output style:

1. The base model distribution may already prefer some patterns.
2. SFT targets may teach or suppress those patterns.
3. DPO chosen responses may contain more of a pattern than rejected responses.
4. Preference optimization may shift model probability even when chosen data
   does not have a clean feature advantage.
5. During generation, one hit may make later hits more likely because the model
   conditions on its own words.

Phase 2 measured items 1, 2, 4, and 5 on the model side. It does not by itself
prove preference-data complicity; that requires joining these results back to
Phase 1 chosen/rejected corpus rates in Phase 3.

## Model Ladder And Run Scope

The measured ladder was the OLMo 3 Instruct path:

| Stage | Report label | Interpretation |
|---|---|---|
| Base | Base | The base OLMo 3 checkpoint in the measured ladder. |
| SFT | SFT | The supervised fine-tuned checkpoint. |
| DPO | DPO | The preference-optimized checkpoint. |
| Final/RLVR | Final/RLVR | The later released final checkpoint after the RLVR-style stage. |

The final retained generation grid used:

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

Supporting Phase 2 evidence included teacher-forced propensity grids,
generation counts, DPO scale checks, DPO temperature checks, compounding tests,
Biber-lite register comparisons, and a combined style-signature table.

This was a bounded close-out, not the full production grid in `EXPERIMENTS.md`.
The original plan called for 5,000 prompts, 8 completions, 3 temperatures,
multiple ladders, and full pybiber extraction. The delivered Phase 2 package is
a coherent OLMo 3 / Dolci slice that supports stage-localized conclusions, but
it should not be treated as a universal claim about all preference-trained
models.

## Feature Set

The retained Tier-1 features were:

| Feature | Meaning |
|---|---|
| `slop_lexicon` | Curated lexical tics and stock multiword expressions. |
| `rule_of_three_approx` | Regex approximation of three-part rhetorical lists such as "X, Y, and Z." |
| `contrastive_negation` | Contrastive constructions such as "not X, but Y" or "not just X." |
| `stock_openers` | Formulaic response openings. |
| `stock_closers` | Formulaic response closings. |
| `stock_openers_closers` | Pooled opener/closer view. Useful for summaries, but it hides opener-vs-closer differences. |

Phase 2 also measured 19 `biber_lite` register proxies on generated outputs.
These include pronouns, modals, hedges, amplifiers, nominalizations,
subordination, complement patterns, wh-questions, and passive-voice
approximations. They are regex-style proxies, not full pybiber categories.

## How To Read The Metrics

Phase 2 uses several measurements because they answer different questions:

| Metric | What it asks | Interpretation |
|---|---|---|
| Corpus rate | How often does a feature appear in data? | Data-side availability or preference-pair imbalance. |
| Teacher-forced propensity | Given the same fixed context, how much probability does a checkpoint assign to feature-bearing continuations? | Local model preference independent of sampling drift. |
| Amplification factor, AF | Model feature probability divided by reference feature rate. | A ratio-style measure of local over- or under-weighting. |
| Neutral-normalized AF | Slop AF divided by neutral-control AF. | Reduces the chance that general calibration shifts are mistaken for slop-specific shifts. |
| Free-running emission | How often does the feature appear in sampled completions? | Closest to user-visible behavior. |
| Compounding | Does an earlier feature hit raise the chance of later hits in the same answer? | Tests self-conditioning and path dependence. |
| Biber-lite register | How broader register features move across generated outputs. | A descriptive final-output style layer, not causal AF. |

These measurements can disagree. That disagreement is not a failure; it is one
of the main findings. Teacher forcing is best for local propensity,
free-running emission is best for visible output, and compounding asks whether
style reinforces itself once generation begins.

## Phase 1 Data Context

Phase 2 should be interpreted against the Phase 1 data baselines. Rates below
are per 1,000 simple tokens:

| Corpus / role | Contrastive negation | Rule of three | Slop lexicon | Stock openers | Stock closers | Openers + closers |
|---|---:|---:|---:|---:|---:|---:|
| Dolma pretrain sample | 0.389 | 3.946 | 0.238 | 0.027 | 0.008 | 0.035 |
| Dolci SFT target | 0.139 | 1.937 | 0.553 | 0.262 | 0.089 | 0.351 |
| Dolci DPO chosen | 0.459 | 3.696 | 0.990 | 0.235 | 0.413 | 0.648 |
| Dolci DPO rejected | 0.369 | 3.391 | 1.022 | 0.277 | 0.292 | 0.569 |

The data-side facts are:

- pretraining samples are already high on `rule_of_three_approx` and
  `contrastive_negation`;
- SFT targets are higher than the pretraining sample on `slop_lexicon` and
  stock opener/closer phrases;
- DPO chosen responses are higher than rejected responses for contrastive
  negation, rule-of-three, stock closers, and pooled stock phrases;
- DPO chosen responses are not higher than rejected responses for
  `slop_lexicon`.

That final point is central. If DPO model propensity for `slop_lexicon` rises,
the simplest explanation cannot be "chosen responses just had more
`slop_lexicon` than rejected responses." Phase 3 should treat that as a
candidate dynamics-driven or mixed effect.

## Result 1: The Checkpoint Progression Is Non-Monotonic

A simple story would be: Base is clean, SFT adds assistant style, DPO adds
slop, and Final/RLVR adds the most. The measured OLMo 3 / Dolci slice does not
support that story.

Target-shape free-running rates per 1,000 generated tokens:

| Feature | Base | SFT | DPO | Final/RLVR | Maximum stage |
|---|---:|---:|---:|---:|---|
| `rule_of_three_approx` | 1.019 | 0.488 | 0.861 | 0.802 | Base |
| `slop_lexicon` | 0.233 | 0.171 | 0.229 | 0.193 | Base |
| `contrastive_negation` | 0.136 | 0.041 | 0.141 | 0.142 | Final/RLVR |
| `stock_openers_closers` | 0.092 | 0.072 | 0.108 | 0.091 | DPO |
| `stock_openers` | 0.064 | 0.052 | 0.064 | 0.059 | DPO |
| `stock_closers` | 0.028 | 0.020 | 0.043 | 0.032 | DPO |

The stage progression is feature-specific:

- Base is highest for `rule_of_three_approx` and narrowly highest for
  `slop_lexicon` in sampled output.
- SFT is the lowest checkpoint for every retained Tier-1 free-running feature.
- DPO rebounds on selected markers, especially stock phrases and local
  `slop_lexicon` propensity.
- Final/RLVR stays close to DPO in the combined style signature but does not
  broadly exceed DPO on headline rates.

## Result 2: `slop_lexicon` Is The Strongest Amplification-Like Signal

`slop_lexicon` has the strongest measurement chain: Phase 1 corpus rates,
teacher-forced AF, free-running emission, expected-vs-observed compounding,
and direct prior/no-prior compounding windows.

Teacher-forced neutral-normalized AF:

| Stage | Normalized AF | 95% CI |
|---|---:|---:|
| Base | 1.467 | 1.114-2.046 |
| SFT | 1.695 | 1.315-2.289 |
| DPO | 1.999 | 1.446-2.837 |
| Final/RLVR | 1.659 | 1.189-2.351 |

DPO is the maximum point-estimate stage. The intervals overlap, so the right
claim is "bounded positive DPO-stage peak," not "DPO is cleanly separated from
every other stage."

Free-running `slop_lexicon` emission:

| Stage | Free-run /1k generated tokens |
|---|---:|
| Base | 0.233 |
| SFT | 0.171 |
| DPO | 0.229 |
| Final/RLVR | 0.193 |

Base is narrowly above DPO in sampled output, DPO is clearly above SFT, and
Final/RLVR is below DPO. The conclusion is:

> DPO is the peak local-propensity checkpoint for `slop_lexicon`, but visible
> sampled output does not become a clean DPO-only peak because Base already
> emits the feature at a comparable rate.

## Result 3: `slop_lexicon` Compounds During Generation

The expected-vs-observed compounding analysis compares generated opportunity
rates against teacher-forced expectations. For `slop_lexicon`, observed rates
exceed teacher-forced expectations in all four stages:

| Stage | Observed /1k opportunities | Expected /1k opportunities | Excess /1k | Realized AF | Repeat generations |
|---|---:|---:|---:|---:|---:|
| Base | 0.615 | 0.232 | 0.383 | 1.051 | 192 |
| SFT | 0.697 | 0.336 | 0.361 | 1.192 | 145 |
| DPO | 0.638 | 0.439 | 0.199 | 1.091 | 199 |
| Final/RLVR | 0.582 | 0.445 | 0.137 | 0.994 | 180 |

The direct prior/no-prior window test is the clearest compounding read:

| Stage | P(hit after prior) | P(hit without prior) | Risk difference | Risk ratio |
|---|---:|---:|---:|---:|
| Base | 0.0812 | 0.0106 | 0.0707 | 7.68 |
| SFT | 0.1052 | 0.0113 | 0.0939 | 9.32 |
| DPO | 0.0712 | 0.0113 | 0.0599 | 6.32 |
| Final/RLVR | 0.0691 | 0.0107 | 0.0584 | 6.44 |

After a `slop_lexicon` hit appears, later windows are about six to nine times
more likely to contain another hit than windows without a prior hit.

This effect is not DPO-specific. Base, SFT, DPO, and Final/RLVR all compound.
That means free-running style is not just a sequence of independent local token
choices; generated wording can reinforce later wording.

## Result 4: `rule_of_three_approx` Is Visible But Not DPO-Amplified Here

`rule_of_three_approx` is the highest-rate retained Tier-1 feature in sampled
output, but it does not support a DPO-specific amplification claim in this
bounded slice.

Free-running rate per 1,000 generated tokens:

| Stage | Rate |
|---|---:|
| Base | 1.019 |
| SFT | 0.488 |
| DPO | 0.861 |
| Final/RLVR | 0.802 |

Base is highest. DPO and Final/RLVR rebound from SFT but stay below Base.

The teacher-forced proxy measures third-item extension after a candidate
two-item comma/list context. It does not measure open-vocabulary initiation of
the whole construction:

| Stage | Raw AF | 95% CI |
|---|---:|---:|
| Base | 0.721 | 0.698-0.742 |
| SFT | 0.767 | 0.743-0.787 |
| DPO | 0.716 | 0.691-0.738 |
| Final/RLVR | 0.723 | 0.698-0.746 |

SFT has the highest point estimate for this bounded proxy. DPO is slightly
below Base. All stages are below the held-out reference extension rate.

Conclusion:

> `rule_of_three_approx` is best described as inherited/base-heavy with SFT
> suppression and later rebound, not as DPO-amplified in this OLMo slice.

## Result 5: Stock Openers And Closers Need To Be Split

The pooled `stock_openers_closers` feature is useful for summaries but hides
an opener-vs-closer split.

Teacher-forced `stock_openers` raw AF:

| Stage | Raw AF |
|---|---:|
| Base | 0.067 |
| SFT | 0.048 |
| DPO | 0.035 |
| Final/RLVR | 0.035 |

The model assigns much less probability to retained stock opener starts than
the held-out references use.

Teacher-forced `stock_closers` raw AF:

| Stage | Raw AF |
|---|---:|
| Base | 73.304 |
| SFT | 65.076 |
| DPO | 68.634 |
| Final/RLVR | 74.032 |

The closer AF is extremely high because the held-out reference denominator is
small: 18 positives in 41,187 opportunities. The direction is informative, but
the exact magnitude is fragile.

Free-running stock phrase rates are small and DPO-peaked:

| Feature | Base | SFT | DPO | Final/RLVR |
|---|---:|---:|---:|---:|
| `stock_openers` | 0.064 | 0.052 | 0.064 | 0.059 |
| `stock_closers` | 0.028 | 0.020 | 0.043 | 0.032 |
| `stock_openers_closers` | 0.092 | 0.072 | 0.108 | 0.091 |

Conclusion:

> DPO is the visible sampled-output peak for stock phrases, but opener and
> closer behavior should be analyzed separately. The large teacher-forced
> closer AF is a rare-reference result.

## Result 6: `contrastive_negation` Is Output-Visible But Teacher-Forced Support Is Sparse

`contrastive_negation` has a clear sampled-output pattern:

| Stage | Free-run /1k generated tokens |
|---|---:|
| Base | 0.136 |
| SFT | 0.041 |
| DPO | 0.141 |
| Final/RLVR | 0.142 |

SFT strongly suppresses it; DPO and Final/RLVR rebound to roughly Base-like
rates. The retained teacher-forced table does not support a strong AF claim
because held-out target support is sparse: only 7 reference instances in the
5,000-prompt denominator audit.

Conclusion:

> The safe Phase 2 claim is output-only: `contrastive_negation` is visible in
> generated text, SFT suppresses it, and later checkpoints rebound. It needs
> better teacher-forced support before stronger causal interpretation.

## Result 7: Biber-Lite Adds A Broader Register Signature

The Biber-lite layer asks a different question from Tier-1 slop features. It
does not estimate teacher-forced amplification. It describes broader generated
output style.

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

Against SFT targets, Final/RLVR is higher on several register proxies:

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
> not only in explicit slop lexicon or stock phrases. Under these proxies,
> Final/RLVR is more question-heavy, hedged, selected-verb-heavy, pronominal,
> modal, and subordinating than the SFT target sample.

## Result 8: Final/RLVR Is DPO-Like But Not DPO-Identical

The final style-signature artifact combines Tier-1 free-running rates,
empirical compounding metrics, and Biber-lite register rates. Stage distances
are useful for orientation, not formal statistical testing, because raw metric
scales let high-rate features dominate.

Distances from Final/RLVR:

| Comparison | Shared features | Euclidean distance | Cosine distance |
|---|---:|---:|---:|
| Final/RLVR vs DPO | 43 | 30.567 | 0.000075 |
| Final/RLVR vs SFT | 43 | 35.564 | 0.000287 |
| Final/RLVR vs Base | 43 | 85.139 | 0.001272 |

Under this assembled signature, Final/RLVR is closest to DPO, then SFT, then
Base.

Largest Final-vs-Base movements include:

- `rule_of_three_approx` observed compounding opportunities increase
  (`428.106` vs. `344.007` per 1,000 opportunities), even though the simple
  free-running rate peaks at Base;
- `stock_openers` observed opportunities drop sharply (`1.477` vs. `8.592`
  per 1,000 opportunities);
- Biber-lite demonstratives, infinitives, first-person pronouns, necessity
  modals, prediction modals, and possibility modals all decline;
- Biber-lite nominalizations rise modestly.

Largest Final-vs-DPO movements include:

- `rule_of_three_approx` observed compounding opportunities are lower in Final
  than DPO (`428.106` vs. `458.417`);
- `stock_closers` prior-window risk ratio is higher in Final than DPO
  (`9.399` vs. `6.198`), though stock closer absolute rates remain sparse;
- Final is slightly higher than DPO on first-person pronouns, causal
  subordinators, hedges, infinitives, passive-voice approximation, public
  verbs, and possibility modals.

Conclusion:

> Final/RLVR keeps a DPO-like aggregate style while changing selected register
> markers and reducing several headline slop measurements.

## Checkpoint-By-Checkpoint Conclusions

### Base

Base already contains much of the measured style. It is the sampled-output
maximum for `rule_of_three_approx` and narrowly for `slop_lexicon`, and it is
high on Biber-lite demonstratives, first-person pronouns, infinitives, and
modal markers.

This rules out a naive story where slop begins only at preference
optimization. Some measured behaviors are inherited from pretraining, from the
base model distribution, or from earlier pipeline factors not isolated here.

### SFT

SFT often suppresses visible free-running emissions. In the retained
target-shape grid, SFT is the lowest checkpoint for every Tier-1 free-running
feature.

SFT is not simply "less slop" by every metric. It raises teacher-forced
`slop_lexicon` normalized AF above Base and has the highest point estimate for
the `rule_of_three_approx` comma-pair extension proxy. The practical lesson is
that local propensity and sampled-output rate can move differently.

### DPO

DPO is the clearest post-SFT rebound. It is the maximum point estimate for
teacher-forced `slop_lexicon` normalized AF and the free-running maximum for
stock opener/closer rates.

DPO is also close to Base for `slop_lexicon` sampled output. It is not the
maximum for `rule_of_three_approx`, and Phase 1 did not show a positive DPO
chosen-vs-rejected advantage for `slop_lexicon`. This makes the DPO result
feature-specific rather than a simple "DPO chosen data contains more of every
slop feature" story.

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

## Feature-By-Feature Classification

| Feature | Phase 2 classification | Reason |
|---|---|---|
| `slop_lexicon` | DPO local-propensity peak plus generation compounding | DPO has highest teacher-forced neutral-normalized AF; all stages show positive prior/no-prior compounding; sampled output is Base/DPO-like. |
| `rule_of_three_approx` | Inherited/base-heavy, SFT-suppressed, later rebound | Base has highest free-running rate; teacher-forced proxy peaks at SFT and does not show DPO amplification. |
| `contrastive_negation` | Output-visible rebound, causal status unresolved | SFT suppresses sampled output; DPO/Final rebound; teacher-forced references are sparse. |
| `stock_openers` | Low-rate, not locally amplified under opener start contract | Free-running rates are small; teacher-forced raw AF is far below 1 and falls through the ladder. |
| `stock_closers` | DPO/Final visible support, rare-reference teacher-forced support | DPO peaks in sampled output; teacher-forced AF is high but reference support is small. |
| `stock_openers_closers` | Useful pooled output marker | DPO is the sampled-output peak, but pooled interpretation hides opener/closer differences. |
| Biber-lite proxies | Broader register shift | DPO/Final move away from Base on many modal/pronoun/demonstrative markers and remain close in aggregate style. |

## Hypothesis Status

| Hypothesis | Phase 2 status | Reason |
|---|---|---|
| H1: inheritance vs. amplification | Partially supported, feature-specific | Base is already high on several visible features, while `slop_lexicon` still shows a DPO local-propensity peak. |
| H2: preference-signal complicity | Not decided by Phase 2 alone | Phase 2 supplies model-side stage effects; Phase 3 must join them to chosen/rejected corpus evidence. |
| H3: compounding | Supported for `slop_lexicon` | Observed generated opportunity rates exceed teacher-forced expectations, and prior-hit windows have much higher later-hit risk. |
| H4: stage localization | Supported as non-monotonic localization | Base-heavy, SFT-suppressed, DPO-rebound, and Final/RLVR-attenuated patterns all appear depending on feature. |

The most important H2 warning sign is `slop_lexicon`: DPO teacher-forced AF
peaks even though DPO chosen responses are not higher than rejected responses
for that feature. That points away from a simple chosen-data explanation.

## What Phase 2 Does Not Prove

Phase 2 does not prove that DPO universally creates slop. The DPO signal is
strongest for `slop_lexicon` local propensity and stock phrase sampled output,
not every feature.

Phase 2 does not prove that Final/RLVR is worse than DPO on slop. Final/RLVR is
DPO-like in aggregate style, but several headline slop rates are lower than
DPO.

Phase 2 does not prove every feature with equal reliability. `slop_lexicon`
has the strongest measurement chain. `contrastive_negation` is mainly a
sampled-output result because teacher-forced reference support is sparse.
`rule_of_three_approx` teacher-forced scoring uses a bounded third-item
extension proxy.

Phase 2 does not provide cross-ladder generalization. SmolLM3 no_think
replication and OLMo-vs-SmolLM3 comparison belong to Phase 3.

Phase 2 does not provide full pybiber extraction. Biber-lite is a useful
surface-register layer, not full Biber category measurement.

## Main Caveats

This is a bounded OLMo 3 / Dolci close-out, not the full production-scale run.

Teacher-forced propensity and sampled generation answer different questions. A
checkpoint can assign high local probability to a feature without making that
feature the highest-rate sampled-output marker.

The retained generation setting is one target shape: 512 prompts, 8 completions
per prompt, 1,024 max new tokens, temperature `1.0`, top-p `0.95`. Other
temperatures were explored but are not the headline grid.

`rule_of_three_approx` is a regex approximation. Its teacher-forced proxy is a
third-item extension test after a candidate two-item list, not full
open-vocabulary rule-of-three initiation.

`contrastive_negation` has sparse held-out reference support in the
5,000-prompt denominator audit, so strong teacher-forced claims are not
retained.

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

> `slop_lexicon` has the strongest amplification-like model-side signal. DPO is
> the peak checkpoint for teacher-forced neutral-normalized propensity, and
> generated text shows positive self-conditioning once a `slop_lexicon` hit has
> appeared.

The cleanest negative claim is:

> The OLMo 3 ladder does not show broad monotonic slop amplification. Base is
> already high on several markers, SFT suppresses visible Tier-1 emissions, DPO
> rebounds on selected features, and Final/RLVR does not broadly exceed DPO.

The best checkpoint progression summary is:

> Base supplies substantial inherited style. SFT reshapes and often suppresses
> visible emissions. DPO raises selected markers, especially local
> `slop_lexicon` propensity and stock phrase output. Final/RLVR remains closest
> to DPO in aggregate but attenuates several headline slop rates while shifting
> broader register markers.

For Phase 3, the most important lesson is that features should be classified
one by one. The evidence supports a feature-level amplification spectrum:
inherited/base-heavy features, SFT-suppressed features, DPO-rebound features,
compounding-sensitive features, and broader register changes that are not
themselves causal slop measurements.

## Phase 3 Handoff

Phase 3 should combine:

- Phase 1 corpus rates and preference-pair evidence;
- Phase 2 teacher-forced AF;
- Phase 2 sampled-output emission;
- Phase 2 compounding evidence;
- Phase 2 Biber-lite style-signature evidence.

Immediate interpretations to carry forward:

- `slop_lexicon` is the main candidate for dynamics-driven or mixed
  amplification because DPO model propensity rises without a clean DPO
  chosen-over-rejected corpus advantage.
- `rule_of_three_approx` should be treated as inherited/base-heavy or
  output-visible rather than DPO-amplified in the OLMo slice.
- Stock phrases need opener/closer separation before causal claims.
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
