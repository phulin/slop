# Phase 2 Conclusion Report For New Readers

Date: 2026-06-16

Status: final bounded OLMo 3 / Dolci Phase 2 conclusion report.

This is the main readable entry point for Phase 2. It is written for someone
who has not followed the project and needs to understand what was measured,
what the results mean, what they do not prove, and what should carry forward
into Phase 3.

Phase 2 should be read as a bounded scientific close-out, not as the full
production-scale grid originally imagined in `EXPERIMENTS.md`. The completed
package gives a coherent OLMo 3 / Dolci slice: one model ladder, one prompt
family, one headline generation setting, retained teacher-forced propensity
contracts, generation-side compounding tests, and a broader Biber-lite style
signature.

For audit detail and artifact hashes, see:

- `docs/experiments/phase2_final_conclusion_report.md`
- `artifacts/phase2/analysis/olmo3_phase2_single_temp_t1_final_artifact_manifest.md`

## One-Page Conclusion

This project studies where recognizable assistant-style surface patterns enter
a model pipeline. It does not score answer quality directly. A useful answer
can contain a measured marker, and a poor answer can contain none. The label
"slop" is operational: it means a detector-defined style marker fired.

Phase 1 measured the data side: how often the markers appeared in pretraining
samples, Dolci SFT targets, Dolci DPO chosen responses, and Dolci DPO rejected
responses.

Phase 2 measured the model side: how the OLMo 3 Instruct checkpoint ladder
behaved when the same features were scored under fixed-context teacher
forcing, counted in sampled generations, and tested for within-answer
self-conditioning.

The measured ladder was:

| Stage | Meaning in this report |
|---|---|
| Base | The base OLMo 3 checkpoint in the measured ladder. |
| SFT | The supervised fine-tuned checkpoint. |
| DPO | The preference-optimized checkpoint. |
| Final/RLVR | The later released final checkpoint after the RLVR-style stage. |

The main conclusion is:

> OLMo 3 does not follow a simple "later checkpoint means more slop" curve.
> Base already contains substantial measured style. SFT often suppresses
> visible Tier-1 markers. DPO selectively rebounds, especially for
> `slop_lexicon` local propensity and stock phrase output. Final/RLVR remains
> closest to DPO in aggregate style, but it does not broadly exceed DPO on the
> headline slop markers.

The strongest positive result is `slop_lexicon`:

- DPO has the highest teacher-forced neutral-normalized amplification factor:
  `1.999`, with 95% CI `1.446-2.837`.
- Final/RLVR falls back to `1.659`, closer to SFT than to a monotonic
  late-stage increase.
- In sampled output, Base and DPO are nearly tied: Base `0.233` vs. DPO
  `0.229` hits per 1,000 generated tokens.
- Once a generation contains a `slop_lexicon` hit, later windows are about
  6x-9x more likely to contain another hit than windows without a prior hit.

The strongest negative result is just as important:

> DPO does not universally amplify every measured slop feature. For
> `rule_of_three_approx`, Base has the highest sampled-output rate and the
> teacher-forced proxy peaks at SFT, not DPO.

The practical handoff is:

> Treat "slop" as a feature-level amplification spectrum, not as one global
> score. Some features are inherited or Base-heavy, some are SFT-suppressed,
> some rebound at DPO, some compound during generation, and Biber-lite register
> shifts describe broader final-output style rather than direct causal
> amplification.

## What Phase 2 Was Trying To Answer

The motivating question was not "are model answers good?" Phase 2 asked a
mechanistic question:

> When a model produces recognizable assistant-style surface patterns, can we
> distinguish whether those patterns were inherited from data, amplified by
> post-training, or amplified by generation dynamics?

A final generated answer can reflect several forces:

1. The base model distribution may already prefer some patterns.
2. SFT target responses may teach, reshape, or suppress those patterns.
3. DPO preference data may favor chosen responses that contain those patterns.
4. Preference optimization may shift model probabilities even when chosen data
   does not have a clean feature advantage.
5. During generation, one feature hit may make later hits more likely because
   the model conditions on its own wording.

Phase 2 measured items 1, 2, 4, and 5 on the model side. It does not by itself
prove preference-data complicity. That requires joining model-side stage
effects back to Phase 1 chosen/rejected corpus rates, which is part of Phase 3.

## Plain-Language Glossary

| Term | Meaning |
|---|---|
| Slop | A shorthand label for detector-defined surface style markers. It is not a claim that an answer is wrong or low quality. |
| Feature | One measurable marker, such as `slop_lexicon` or `rule_of_three_approx`. |
| Corpus rate | How often a feature appears in a dataset sample, reported per 1,000 simple tokens. |
| Teacher forcing | Scoring model probabilities on fixed held-out reference text instead of sampling new text. |
| Opportunity | A context position where a feature could begin under a feature's scoring contract. |
| Amplification factor, AF | Model feature probability divided by reference feature rate. Values above 1 mean the model locally assigns more probability than the reference rate would suggest. |
| Neutral-normalized AF | Slop AF divided by a neutral-control AF, used to reduce the risk of confusing general calibration shifts with slop-specific shifts. |
| Free-running generation | Normal sampled model output. In the headline grid, this means 8 completions per prompt at temperature `1.0`. |
| Compounding | A generated-text effect where an earlier feature hit makes later hits more likely in the same answer. |
| Biber-lite | Regex-based proxies for broader register categories, not full pybiber extraction. |

## What Was Actually Run

The final retained generation grid used one target shape:

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

Supporting evidence included:

- teacher-forced propensity grids for retained feature/opportunity contracts;
- a DPO-only scale check at 1,024 prompts x 8 completions;
- DPO-only temperature sensitivity runs at `0.0`, `0.7`, and `1.0`;
- expected-vs-observed compounding analysis;
- direct prior/no-prior compounding window tests;
- Biber-lite register comparisons over generated outputs;
- an assembled final-output style signature over Tier-1 features, compounding
  metrics, and Biber-lite rates.

This differs from the original production plan. The original Phase 2 plan
called for 5,000 prompts, 8 completions, 3 temperatures, multiple model
ladders, and full pybiber extraction. The delivered Phase 2 package is smaller
but internally consistent. Its strongest claims are OLMo 3 / Dolci claims.
They should not be quoted as universal claims about all preference-trained
models until Phase 3 replication is complete.

## Features Measured

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
approximations. They are broader style descriptors, not teacher-forced
amplification measurements.

## How To Read The Measurements

No single number answers the whole Phase 2 question.

| Metric | What it asks | How to interpret it |
|---|---|---|
| Corpus rate | How often did the feature appear in data? | Data-side availability or chosen/rejected imbalance. |
| Teacher-forced propensity | With context fixed, how much probability does a checkpoint assign to feature-bearing continuations? | Local model preference independent of sampling drift. |
| Amplification factor | How large is model feature probability relative to a reference rate? | Ratio-style local over- or under-weighting. |
| Neutral-normalized AF | How large is slop AF after dividing by a neutral-control AF? | Slop-specific shift after a coarse calibration control. |
| Free-running emission | How often does the feature appear in sampled completions? | Closest to user-visible behavior. |
| Compounding | Does an earlier hit raise the chance of later hits in the same answer? | Evidence for self-conditioning and path dependence. |
| Biber-lite register | How broader register proxies move across stages. | Descriptive output style, not causal AF. |

These measurements can disagree. That disagreement is part of the result. A
checkpoint can locally prefer a feature while sampled output does not peak
there, and sampled output can contain a feature because the base model already
emits it often. Teacher forcing is best for local propensity, free-running
generation is best for visible output, and compounding is best for
path-dependent generation behavior.

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

That last point matters. If DPO model propensity for `slop_lexicon` rises, the
simplest explanation cannot be "chosen DPO responses just had more
`slop_lexicon` than rejected responses." Phase 3 should treat that mismatch as
a candidate dynamics-driven or mixed effect.

## Evidence Map

The final conclusion combines several artifact families:

| Evidence type | Main artifact | What it answers |
|---|---|---|
| Amplification spectrum | `olmo3_amplification_spectrum_single_temp_t1_v6.csv` | Joins Phase 1 data rates, teacher-forced AF, free-running rates, and compounding where available. |
| Free-running stage grid | `olmo3_generation_stage_grid_target_shape_512prompt_8comp_t1_1024.csv` | Counts visible Tier-1 feature rates in sampled completions. |
| Compounding | `olmo3_generation_compounding_target_shape_512prompt_8comp_t1_1024_tf1024.csv` | Compares observed generation rates with teacher-forced expectations and prior/no-prior windows. |
| Biber-lite comparison | `olmo3_biber_lite_generation_vs_corpus_t1.csv` | Compares broader generated-output register to Phase 1 corpus samples. |
| Style signature | `olmo3_style_signature_t1.csv` and stage distances | Summarizes which checkpoints are closest in the assembled final-output feature vector. |

## Result 1: Checkpoint Progression Is Non-Monotonic

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

The checkpoint progression is feature-specific:

- Base is highest for `rule_of_three_approx` and narrowly highest for
  `slop_lexicon` in sampled output.
- SFT is the lowest checkpoint for every retained Tier-1 free-running feature.
- DPO rebounds on selected markers, especially stock phrases and local
  `slop_lexicon` propensity.
- Final/RLVR stays close to DPO in aggregate style, but it does not broadly
  exceed DPO on headline rates.

## Result 2: `slop_lexicon` Is The Strongest Amplification-Like Signal

`slop_lexicon` has the strongest measurement chain in Phase 2: Phase 1 corpus
rates, teacher-forced AF, free-running emission, expected-vs-observed
compounding, and direct prior/no-prior compounding windows.

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
Final/RLVR is below DPO.

Conclusion:

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
| SFT | 0.1052 | 0.0113 | 0.0939 | 9.35 |
| DPO | 0.0712 | 0.0113 | 0.0599 | 6.29 |
| Final/RLVR | 0.0691 | 0.0107 | 0.0584 | 6.43 |

After a `slop_lexicon` hit appears, later windows are about six to nine times
more likely to contain another hit than windows without a prior hit.

Conclusion:

> `slop_lexicon` has a real self-conditioning signal. The effect is not
> DPO-specific; Base, SFT, DPO, and Final/RLVR all compound. Free-running style
> is not just a sequence of independent local token decisions.

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

> `rule_of_three_approx` is best described as inherited or Base-heavy with SFT
> suppression and later rebound, not as DPO-amplified in this OLMo slice.

## Result 5: Stock Openers And Closers Need To Be Split

The pooled `stock_openers_closers` feature is useful for summaries, but it
hides an opener-vs-closer distinction.

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

## Result 6: `contrastive_negation` Is Output-Visible But Causally Under-Supported

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

## Plain-English Answers

| Question | Phase 2 answer |
|---|---|
| Did later checkpoints simply get more slop-heavy? | No. The progression is non-monotonic. SFT is often lowest in sampled output, DPO rebounds on selected markers, and Final/RLVR does not broadly exceed DPO. |
| Which feature has the strongest model-side amplification signal? | `slop_lexicon`. DPO is the peak checkpoint for teacher-forced neutral-normalized AF, although Base remains high in sampled output. |
| Does DPO explain every measured slop feature? | No. `rule_of_three_approx` is Base-heavy in sampled output and SFT-peaked in its teacher-forced proxy. |
| Did we measure self-conditioning? | Yes. For `slop_lexicon`, later windows are about 6x-9x more likely to contain another hit after a prior hit has already appeared. |
| What is the prior-vs-after-prior result? | For `slop_lexicon`, `P(hit after prior)` is `0.069-0.105` depending on stage, while `P(hit without prior)` is about `0.011`. |
| Did we measure Biber-style features? | Yes, as `biber_lite` regex proxies on generated outputs and Phase 1 corpus samples. This is a register/style comparison layer, not full pybiber. |
| What is Final/RLVR closest to? | DPO, under the assembled style signature. Final/RLVR is DPO-like overall but lower than DPO on several headline slop rates. |
| What should Phase 3 inherit from Phase 2? | A feature-by-feature amplification spectrum, not a single global slop score. |

## What Phase 2 Does Not Prove

Phase 2 does not prove that DPO universally creates slop. The DPO signal is
strongest for `slop_lexicon` local propensity and stock phrase sampled output,
not every feature.

Phase 2 does not prove that Final/RLVR is worse than DPO on slop. Final/RLVR
is DPO-like in aggregate style, but several headline slop rates are lower than
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
