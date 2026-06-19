# Phase 2 Final Conclusion Report

Date: 2026-06-15

Status: final bounded OLMo 3 / Dolci Phase 2 close-out.

Supersession note: after the full pybiber Phase 1 rerun, the active Tier-2
register surface is corpus-side full pybiber. Any Biber-lite/register-proxy
material in this historical Phase 2 report is retained for provenance only and
should not be promoted into current paper claims.

Audience: this report is written for someone new to the project. It explains
the motivation, the measurement design, the model checkpoints, the resulting
feature-level evidence, the limits of the current run, and the conclusions
that should carry forward into Phase 3.

How to use this report:

- Read the executive summary first if you only need the conclusion.
- Read "How To Read The Metrics" before comparing tables. The same feature can
  move differently under teacher forcing, free-running generation, and
  compounding.
- Treat the feature-by-feature conclusions as the actual handoff to Phase 3.
  Phase 2 did not produce one global "slop score"; it produced a set of
  stage-localized feature stories.
- Use the artifact list at the end when checking or reproducing a number.

## Executive Summary

This project studies where recognizable "LLM-ish" output style enters a
post-training pipeline. The target is not general answer quality. The target
is a set of measurable surface patterns: curated lexical tics, formulaic
openers and closers, three-part rhetorical lists, contrastive "not X but Y"
phrasing, and broader register features such as pronouns, modals, hedges, and
subordination.

Phase 1 measured the data side of the story. It counted these features in
pretraining-like text, Dolci SFT targets, Dolci DPO chosen responses, and
Dolci DPO rejected responses.

Phase 2 measured the model side. It asked how the OLMo 3 Instruct checkpoint
ladder behaves when the same features are scored locally under teacher
forcing, counted in sampled generations, and tested for self-conditioning
during generation.

In plainer terms, Phase 2 asked three questions:

1. If the prompt context is fixed, which checkpoint is most likely to start a
   measured style pattern?
2. When the checkpoints generate full answers, which patterns actually appear
   in the visible output?
3. Once a pattern appears in an answer, does the model become more likely to
   repeat that style later in the same answer?

The main result is:

> OLMo 3 does not follow a simple "later checkpoint means more slop" curve.
> Base already contains substantial measured style. SFT often lowers visible
> Tier-1 marker rates in sampled output. DPO selectively rebounds, especially
> for `slop_lexicon` local propensity and stock phrase emissions. Final/RLVR
> remains closest to DPO in aggregate output style, but it does not broadly
> increase the measured slop markers beyond DPO.

The strongest positive result is `slop_lexicon`:

- DPO has the highest teacher-forced neutral-normalized amplification factor:
  `1.999`, with 95% CI `1.446-2.837`.
- Final/RLVR falls back from DPO to `1.659`, closer to the SFT point estimate
  than to a monotonic late-stage increase.
- In free-running output, Base and DPO are very close: Base `0.233` vs. DPO
  `0.229` hits per 1,000 generated tokens.
- Once a generated answer has already used a `slop_lexicon` item, later
  windows are about 6x to 9x more likely to contain another hit than windows
  without a prior hit.

The strongest negative result is just as important:

> Phase 2 does not support a broad claim that DPO universally creates slop.
> `rule_of_three_approx` is highest in Base free-running output, its
> teacher-forced proxy peaks at SFT, and Final/RLVR attenuates several DPO
> stage slop rates.

The practical conclusion is that "slop" should be treated as a feature-level
amplification spectrum, not as one global score. Some features look inherited
or Base-heavy, some are suppressed by SFT, some rebound at DPO, some compound
during generation, and broader register shifts do not always map onto the
hand-built Tier-1 feature set.

The one-sentence handoff is:

> Phase 2 found a selective DPO-stage `slop_lexicon` propensity peak and a
> strong `slop_lexicon` self-conditioning signal, but it did not find broad
> monotonic slop growth from Base to SFT to DPO to Final/RLVR.

## What This Project Is Asking

The motivating question is narrower than "are model answers good?" A useful
answer can contain a measured marker, and a bad answer can contain none. The
experiment asks a mechanistic question:

> When a model produces recognizable assistant-style surface patterns, can we
> distinguish whether those patterns were inherited from data, amplified by
> model training, or amplified by generation dynamics?

A generated answer is the result of several forces:

1. The base model distribution may already prefer some features.
2. SFT target responses may teach or suppress those features.
3. Preference data may favor chosen responses that contain those features.
4. Preference optimization may shift model probability even without a clean
   chosen-over-rejected data advantage.
5. During generation, one feature hit may make later hits more likely because
   the model conditions on its own wording.

Phase 2 is about items 1, 2, 4, and 5 on the model side. It does not by itself
settle whether DPO preference data are complicit; that requires joining Phase
2 model behavior back to Phase 1 chosen/rejected corpus rates in Phase 3.

## Glossary For New Readers

| Term | Meaning in this report |
|---|---|
| Slop | A shorthand label for detector-defined surface style markers. It is not a claim that an answer is wrong or low quality. |
| Feature | One measurable style marker, such as `slop_lexicon` or `rule_of_three_approx`. |
| Corpus rate | How often a feature appears in a dataset sample, measured per 1,000 simple tokens. |
| Teacher forcing | Scoring model probabilities on fixed held-out reference text instead of sampling new text. |
| Opportunity | A context location where a feature could begin under the feature's scoring contract. |
| Amplification factor, AF | Model feature probability divided by reference feature rate. Values above 1 mean the model locally assigns more probability than the reference rate would suggest. |
| Neutral-normalized AF | A slop AF divided by a neutral-control AF, used to reduce general calibration effects. |
| Free-running generation | Normal sampled model output, here 8 completions per prompt at temperature `1.0`. |
| Compounding | A generated-text effect where an earlier feature hit makes later hits more likely in the same answer. |
| Historical register proxy | Earlier regex-based proxies for broader register categories. These are not full pybiber extraction and are no longer an active paper feature surface. |

## Phase 2 Scope

The measured model ladder was the OLMo 3 Instruct path:

| Stage | Report label | Interpretation |
|---|---|---|
| Base | Base | The base OLMo 3 checkpoint in the measured ladder. |
| SFT | SFT | The supervised fine-tuned checkpoint. |
| DPO | DPO | The preference-optimized checkpoint. |
| Final/RLVR | Final/RLVR | The later released final checkpoint after the RLVR-style stage. |

The final retained free-running generation grid used one target shape:

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

- teacher-forced propensity grids for retained feature/opportunity contracts;
- a DPO-only scale check at 1,024 prompts x 8 completions;
- DPO-only temperature sensitivity runs at `0.0`, `0.7`, and `1.0`;
- expected-vs-observed compounding analysis;
- direct prior/no-prior compounding windows;
- archived generated-output register-proxy diagnostics;
- a final assembled style-signature table combining Tier-1 emissions,
  compounding metrics, and archived register-proxy rates.

This is a bounded close-out, not the full production grid originally described
in `EXPERIMENTS.md`. The original production plan called for 5,000 prompts,
8 completions, 3 temperatures, multiple model ladders, and full pybiber
extraction. Phase 2 closed on a smaller but internally consistent OLMo 3 /
Dolci slice.

The bounded scope is still useful because the retained measurements line up
on the same prompt family and the same four-stage OLMo checkpoint ladder. The
main limitation is generality, not internal consistency: this report is strong
as an OLMo/Dolci close-out, but it should not be quoted as a universal claim
about all preference-trained models until the Phase 3 replication work is
complete.

## Original Plan vs. Delivered Evidence

The original Phase 2 plan had two main measurements:

1. **Teacher-forced propensity:** hold the context fixed and measure how much
   probability each checkpoint assigns to feature-starting continuations.
2. **Free-running emission:** sample completions from each checkpoint and
   count how often the features appear in visible output.

It also named a third result:

> Result B: decompose free-running feature rates into local propensity and
> self-conditioning or compounding.

The bounded Phase 2 close-out delivered that structure for the OLMo 3 / Dolci
slice:

| Planned item | Delivered status | What to trust |
|---|---|---|
| Held-out prompt package | Delivered | 5,000-prompt Dolci SFT package exists; final generation grid uses 512 held-out prompts per stage. |
| Teacher-forced propensity | Delivered for retained contracts | Strongest for `slop_lexicon`; useful but narrower for `rule_of_three_approx`, stock phrases, and denominator-supported features. |
| Free-running emission | Delivered | 512 prompts x 8 completions x 4 checkpoints at temperature `1.0`, max 1,024 new tokens. |
| Temperature grid | Partial | DPO temperature sensitivity exists; headline conclusions use the single-temperature `t=1.0` target-shape grid. |
| Result B compounding | Delivered | Expected-vs-observed and prior/no-prior window tests are included. |
| Register style layer | Partial in Phase 2 | Historical generated-output proxies were measured, but the active full-pybiber result is the Phase 1 corpus-side rerun. |
| Cross-ladder replication | Not Phase 2 | SmolLM3 replication belongs to Phase 3. |

The main deviation from the original plan is scale. The original design was a
production grid: thousands of prompts, three temperatures, multiple ladders,
and richer linguistic features. The delivered Phase 2 package is a bounded
scientific close-out: one primary OLMo ladder, one headline generation
temperature, retained teacher-forced contracts, and a measured final-output
style signature. That changes how strong the claims can be. It is enough to
localize several OLMo-stage effects, but not enough to claim cross-model
universality.

## Feature Set

Phase 2 uses "slop" as shorthand for detector-defined surface features. It is
not a judgment that the entire answer is bad.

The retained Tier-1 feature views were:

| Feature | Meaning |
|---|---|
| `slop_lexicon` | Curated lexical tics and stock multiword expressions. |
| `rule_of_three_approx` | Regex approximation of three-part rhetorical lists like "X, Y, and Z." |
| `contrastive_negation` | Contrastive "not X, but Y" or "not just X" constructions. |
| `stock_openers` | Formulaic response openings. |
| `stock_closers` | Formulaic response closings. |
| `stock_openers_closers` | Pooled opener/closer view. Useful for summaries, but it hides opener-vs-closer differences. |

The earlier Phase 2 pass also produced generated-output register-proxy
artifacts. Those artifacts are now historical diagnostics only. The active
Tier-2 register surface is the full pybiber Phase 1 corpus-side analysis; this
report should not be read as providing full generated-output pybiber register
measurement.

## How To Read The Metrics

Phase 2 uses several metrics because each answers a different question:

| Metric | What it asks | Why it matters |
|---|---|---|
| Corpus rate | How common was this feature in a corpus role? | Tells us whether a feature was already present in SFT or DPO data. |
| Teacher-forced propensity | Given the same fixed context, how much probability does the checkpoint put on feature-bearing continuations? | Isolates local model preference from decoding and generation drift. |
| Amplification factor, or AF | How large is model feature probability relative to the reference feature rate? | Gives a ratio-style measure of local amplification. |
| Neutral-normalized AF | How large is slop AF after dividing by a neutral control basket? | Reduces the chance that general calibration shifts are mistaken for slop-specific effects. |
| Free-running emission | How often does the feature appear in sampled completions? | Closest to user-visible output, but affected by decoding, length, prompt mix, and self-conditioning. |
| Compounding | Does an earlier hit make later hits more likely in the same answer? | Tests whether output style is path-dependent during generation. |
| Historical register-proxy artifacts | How did an earlier proxy layer describe generated-output style? | Archived diagnostic only; not part of the active paper claim surface. |

These metrics can disagree. That disagreement is not noise; it is one of the
main findings. A checkpoint can locally prefer a feature while sampled output
does not peak there, or sampled output can contain a feature because the base
model already emits it often.

The most common interpretation mistake would be to read the free-running rate
table as the whole result. It is the most user-visible table, but it is not the
most causal one. Teacher forcing is better for local model preference;
free-running generation is better for visible product behavior; compounding is
better for path dependence. Phase 2 needs all three because "style" is created
by both model probabilities and the model's own previous words.

## Phase 1 Data Context

Phase 2 should be read against the Phase 1 data baselines. Rates below are per
1,000 simple tokens:

| Corpus / role | Contrastive negation | Rule of three | Slop lexicon | Stock openers | Stock closers | Openers + closers |
|---|---:|---:|---:|---:|---:|---:|
| Dolma pretrain sample | 0.389 | 3.946 | 0.238 | 0.027 | 0.008 | 0.035 |
| Dolci SFT target | 0.139 | 1.937 | 0.553 | 0.262 | 0.089 | 0.351 |
| Dolci DPO chosen | 0.459 | 3.696 | 0.990 | 0.235 | 0.413 | 0.648 |
| Dolci DPO rejected | 0.369 | 3.391 | 1.022 | 0.277 | 0.292 | 0.569 |

The important data-side facts are:

- pretraining samples are already high on `rule_of_three_approx` and
  `contrastive_negation`;
- SFT targets are higher than the pretraining sample on `slop_lexicon` and
  stock opener/closer phrases;
- DPO chosen responses are higher than rejected responses for contrastive
  negation, rule-of-three, stock closers, and pooled stock phrases;
- DPO chosen responses are not higher than rejected responses for
  `slop_lexicon`.

That last point matters. If DPO model propensity for `slop_lexicon` rises, the
simplest explanation cannot be "the chosen DPO responses just had more
`slop_lexicon` than rejected responses." Phase 3 should treat that as a
candidate dynamics-driven or mixed effect.

This is why Phase 2 is not merely a generation-counting exercise. The Phase 1
data side says which features are available in the training and preference
corpora. Phase 2 says how the released checkpoints behave after training. When
those two disagree, the disagreement is exactly where Phase 3 should look for
optimization or decoding dynamics.

## Evidence Map

The final conclusion combines several artifact families:

| Evidence type | Main artifact | What it answers |
|---|---|---|
| Amplification spectrum | `olmo3_amplification_spectrum_single_temp_t1_v6.csv` | Joins Phase 1 data rates, teacher-forced AF, free-running rates, and compounding where available. |
| Free-running stage grid | `olmo3_generation_stage_grid_target_shape_512prompt_8comp_t1_1024.csv` | Counts visible Tier-1 feature rates in sampled completions. |
| Compounding | `olmo3_generation_compounding_target_shape_512prompt_8comp_t1_1024_tf1024.csv` | Compares observed generation rates with teacher-forced expectations and prior/no-prior windows. |
| Archived register-proxy comparison | `olmo3_biber_lite_generation_vs_corpus_t1.csv` | Historical generated-output proxy comparison; superseded in active paper claims by Phase 1 full pybiber. |
| Style signature | `olmo3_style_signature_t1.csv` and stage distances | Summarizes which checkpoints are closest in the assembled final-output feature vector. |

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

This is the cleanest stage-localization result. Visible style features are
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

`rule_of_three_approx` is the highest-rate retained Tier-1 feature in
generated outputs, but its stage pattern does not support a DPO-specific
amplification claim.

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

## Result 7: Archived Register Proxies Give A Broader Historical Style Signature

Phase 2 measured generated-output register proxies and compared them to Phase
1 corpus samples. This gave a wider historical style signature beyond the
hand-selected Tier-1 markers. These proxy results are no longer the active
Tier-2 register surface; the active paper claim uses the full pybiber Phase 1
corpus-side analysis.

Selected historical proxy progression, per 1,000 generated tokens:

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
empirical compounding metrics, and historical register-proxy rates. It then
computes checkpoint distances over the combined raw vector.

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
| Historical proxy | Generation /1k tokens | Demonstratives | 10.115 | 15.686 | -5.571 |
| Historical proxy | Generation /1k tokens | Infinitives | 13.623 | 17.426 | -3.803 |
| Historical proxy | Generation /1k tokens | First-person pronouns | 11.638 | 15.325 | -3.687 |
| Historical proxy | Generation /1k tokens | Necessity modals | 2.505 | 5.037 | -2.532 |
| Historical proxy | Generation /1k tokens | Prediction modals | 1.879 | 3.851 | -1.972 |
| Historical proxy | Generation /1k tokens | Nominalizations | 27.309 | 25.499 | +1.810 |

Largest Final-vs-DPO shifts:

| Group | Metric | Feature | Final | DPO | Delta |
|---|---|---|---:|---:|---:|
| Compounding | Observed /1k opportunities | `rule_of_three_approx` | 428.106 | 458.417 | -30.311 |
| Compounding | Prior risk ratio | `stock_closers` | 9.399 | 6.198 | +3.201 |
| Historical proxy | Generation /1k tokens | First-person pronouns | 11.638 | 10.302 | +1.336 |
| Historical proxy | Generation /1k tokens | Causal subordinators | 3.831 | 2.856 | +0.975 |
| Historical proxy | Generation /1k tokens | Hedges | 1.868 | 1.109 | +0.759 |
| Historical proxy | Generation /1k tokens | Infinitives | 13.623 | 13.009 | +0.614 |

Conclusion:

> Final/RLVR keeps a DPO-like aggregate style while changing selected register
> markers and reducing several headline slop measurements.

## Checkpoint-By-Checkpoint Conclusions

### Base

Base already contains much of the measured style. It is the target-shape
free-running maximum for `rule_of_three_approx` and `slop_lexicon`, and it is
high on historical-proxy demonstratives, first-person pronouns, infinitives, and
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

Compared with DPO, Final/RLVR is higher on selected historical proxy
markers, including first-person pronouns, causal subordinators, hedges,
infinitives, passive-voice approximation, public verbs, and possibility
modals.

The practical read is that Final/RLVR keeps a DPO-like style signature while
partially smoothing or redirecting several DPO-stage movements.

## Feature-By-Feature Conclusions

| Feature | Phase 2 conclusion |
|---|---|
| `slop_lexicon` | Strongest amplification-like signal. DPO peaks in local neutral-normalized AF; free-running output is Base-like/DPO-like; compounding is present at all stages. |
| `rule_of_three_approx` | Strong output marker but not DPO-amplified here. Base has the highest free-running rate; SFT has the highest teacher-forced proxy. |
| `contrastive_negation` | Output-visible with SFT suppression and DPO/Final rebound. Retained teacher-forced support is too sparse for a strong AF claim. |
| `stock_openers` | Low rates. DPO is tied or near peak in free-running output, but teacher-forced raw AF falls through the ladder. |
| `stock_closers` | DPO peaks in free-running output; teacher-forced raw AF is very high because references are rare. Direction useful, magnitude fragile. |
| `stock_openers_closers` | Useful pooled view. DPO is the free-running peak, but pooled interpretation hides the opener/closer split. |
| Historical register proxies | Archived descriptive style layer. Final/RLVR is less Base-like on many modal/pronoun/demonstrative markers and remains close to DPO overall, but this is not active full-pybiber evidence. |

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

## Plain-English Answers

| Question | Phase 2 answer |
|---|---|
| Did later checkpoints simply get more slop-heavy? | No. The progression is non-monotonic. SFT is often lowest in sampled output, DPO rebounds on selected markers, and Final/RLVR does not broadly exceed DPO. |
| Which feature has the strongest model-side amplification signal? | `slop_lexicon`. DPO is the peak checkpoint for teacher-forced neutral-normalized AF, although Base remains high in sampled output. |
| Does DPO explain every measured slop feature? | No. `rule_of_three_approx` is Base-heavy in sampled output and SFT-peaked in its teacher-forced proxy. |
| Did we measure self-conditioning? | Yes. For `slop_lexicon`, later windows are about 6x-9x more likely to contain another hit after a prior hit has already appeared. |
| What is the prior-vs-after-prior result? | For `slop_lexicon`, `P(hit after prior)` is `0.069-0.105` depending on stage, while `P(hit without prior)` is about `0.011`. That is the clearest compounding signal. |
| Did we measure Biber-style features? | Yes for the active paper scope: full pybiber was run over the Phase 1 corpus-side samples. Generated-output full pybiber was not run; older generated-output proxy artifacts are historical diagnostics only. |
| What is Final/RLVR closest to? | DPO, under the assembled style signature. Final/RLVR is DPO-like overall but lower than DPO on several headline slop rates. |
| What should Phase 3 inherit from Phase 2? | A feature-by-feature amplification spectrum, not a single global slop score. |

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

Phase 2 does not provide generated-output full pybiber extraction. The active
full-pybiber result is corpus-side Phase 1 register analysis.

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

Historical generated-output register-proxy features should not be presented as
full pybiber extraction or as part of the current paper's active Tier-2
surface.

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
- archived Phase 2 register-proxy style-signature evidence.

Immediate Phase 3 interpretations implied by this report:

- `slop_lexicon` is the main candidate for dynamics-driven or mixed
  amplification because DPO model propensity rises without a clean DPO
  chosen-over-rejected corpus advantage.
- `rule_of_three_approx` should be treated as inherited/base-heavy or
  output-visible rather than DPO-amplified in the OLMo slice.
- Stock phrases need opener/closer separation before making a causal claim.
- `contrastive_negation` needs better teacher-forced support before it can be
  classified beyond output-visible rebound.
- Historical register-proxy signatures should be treated as archived
  diagnostics, not as active full-pybiber or causal slop evidence.
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
- `artifacts/phase2/analysis/olmo3_style_signature_t1.csv`
- `artifacts/phase2/analysis/olmo3_style_signature_t1_stage_distances.csv`
- `artifacts/phase2/analysis/olmo3_style_signature_t1_summary.md`
- `artifacts/phase2/analysis/olmo3_phase2_single_temp_t1_final_artifact_manifest.md`

Archived generated-output register-proxy artifacts from the earlier Phase 2
pass remain on disk for provenance, but they are no longer final-report inputs
for the active pybiber-augmented paper scope.

Important W&B runs:

| Artifact family | W&B run |
|---|---|
| Final amplification spectrum | `jxqcadhe` |
| Biber-lite generation-vs-corpus comparison | `eu4glzoq` |
| Final output style signature | `bt0zelup` |
| Final artifact manifest | `5qonr7io` |
