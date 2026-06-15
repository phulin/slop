# Phase 2 Final Conclusion Report

Date: 2026-06-15

## Executive Summary

Phase 2 tested whether the "slop" features identified in Phase 1 become more
likely as an OLMo 3 Instruct model moves through the post-training ladder:
base, SFT, DPO, and final/RLVR. The work was run as a bounded measurement
package on the OLMo 3/Dolci slice, not as the full production-scale
EXPERIMENTS.md grid.

The short conclusion is that the model progression is feature-specific. There
is no single monotonic "post-training adds slop" curve. Some slop propensity is
already present in the base model. SFT often suppresses free-running emissions.
DPO is the clearest peak for teacher-forced `slop_lexicon` propensity and for
several stock phrase emissions. Final/RLVR is closest to DPO in aggregate
output style, but it attenuates several DPO/base-heavy markers rather than
amplifying them further.

The strongest positive result is for `slop_lexicon`:

- Teacher-forced neutral-normalized amplification peaks at DPO.
- Free-running target-shape generation puts base and DPO very close, with base
  narrowly highest.
- Generated text shows positive self-conditioning/compounding in every stage,
  but the strongest realized-AF cell is SFT, not DPO or final.
- Final/RLVR is DPO-like overall, but lower than DPO on `slop_lexicon`
  free-running emission and lower than DPO on the teacher-forced normalized AF.

The strongest negative result is for a broad, clean DPO-only slop story.
`rule_of_three_approx` free-running emissions peak at base, its teacher-forced
comma-pair extension proxy peaks at SFT, and the final style signature is best
read as DPO-like but not DPO-identical.

## What Phase 2 Was Trying To Measure

Phase 1 measured feature rates in data corpora. It identified a set of
candidate "slop" or style markers whose rates differed across pretraining,
SFT, and preference data. Phase 2 asked what happens inside and around a model
trained on those distributions.

The central question was:

> When the same prompts are sent through checkpoints along the OLMo 3 Instruct
> ladder, do the target features become more probable, more frequently emitted,
> or more self-reinforcing?

Phase 2 therefore measured three different phenomena:

1. Teacher-forced propensity: given a fixed opportunity in a prompt/reference
   context, how much probability mass does the model assign to a target
   continuation?
2. Free-running emission: when the model generates text, how often does the
   feature actually appear?
3. Compounding: after a feature has appeared once in generated text, does the
   same feature become more likely later in the same completion?

A fourth layer, Biber-lite register comparison, was added at the end to
describe broader output style. It is not a propensity or causal measurement;
it is a generated-output style signature.

## Scope And Dataset

The retained Phase 2 deliverable is a bounded OLMo 3/Dolci measurement package.
It uses a held-out Dolci SFT prompt package with near-duplicate prompt
filtering.

The model ladder is:

| Stage | Meaning |
|---|---|
| Base | The base OLMo 3 checkpoint before the later post-training stages measured here. |
| SFT | The supervised fine-tuned checkpoint. |
| DPO | The preference-optimized checkpoint. |
| Final/RLVR | The final checkpoint after the later RLVR-style stage. |

The final free-running target shape is:

| Setting | Value |
|---|---:|
| Prompts per stage | 512 |
| Completions per prompt | 8 |
| Max new tokens per completion | 1,024 |
| Temperature | 1.0 |
| Top-p | 0.95 |
| Generations per stage | 4,096 |

There is also a DPO-only scale check at 1,024 prompts x 8 completions. The
earlier short generation temperature sweep is retained as sensitivity context,
but the final conclusion uses the single temperature `t=1.0` target-shape
grid.

This is not the full production-scale plan from EXPERIMENTS.md. In particular,
it does not include the full three-temperature, 5,000-prompt, two-ladder
production grid.

## Retained Feature Set

Phase 2 focused on the core feature set retained from the revised Phase 1
close-out:

| Feature | Plain-language description |
|---|---|
| `slop_lexicon` | A curated lexicon of terms and phrases treated as the main slop marker. |
| `rule_of_three_approx` | A proxy for three-part rhetorical list constructions. |
| `contrastive_negation` | Contrastive negation patterns such as "not X, but Y" style forms. |
| `stock_openers` | Repeated or formulaic opening phrases. |
| `stock_closers` | Repeated or formulaic closing phrases. |
| `stock_openers_closers` | Pooled opener/closer view used in some summaries. |

The feature coverage is uneven. `slop_lexicon` has the strongest measurement
support because it has a teacher-forced target/control setup and a compounding
join. `rule_of_three_approx` has a bounded teacher-forced proxy, but that proxy
measures third-item continuation after a candidate two-item list rather than
open-vocabulary initiation of the whole construction. `contrastive_negation`
has sparse held-out reference support, so its teacher-forced claims are weak.

## Measurement Views

### Teacher-Forced Propensity

Teacher-forced scoring asks how much probability mass a checkpoint assigns to
target continuations at fixed opportunities. This is the cleanest way to ask
whether the model locally prefers a target feature, because the prompt and
candidate continuation are controlled.

For `slop_lexicon`, Phase 2 used a neutral control feature
(`neutral_common_controls`) and reports a neutral-normalized amplification
factor. Higher values mean the model assigns relatively more probability mass
to the slop lexicon compared with the neutral control.

### Free-Running Emission

Free-running generation asks how often features actually appear in sampled
model outputs. This is closer to what users see, but it is affected by prompt
mix, generation length, decoding settings, and earlier generated tokens.

The final read uses the target-shape generation grid at `t=1.0`, top-p `0.95`.

### Compounding

Compounding asks whether features become more likely after they have already
appeared in a generated completion. Phase 2 measured this with local windows:
windows after a prior feature hit are compared with windows without a prior
hit.

For `slop_lexicon`, Phase 2 also joins generated opportunities to the
teacher-forced expected rate. This gives an observed-vs-expected comparison and
a realized amplification factor.

### Biber-Lite Register And Style Signature

Biber-lite features are regex-based register proxies measured per 1k generated
tokens. They are useful for describing broad output style, but they are not
teacher-forced propensity measurements.

The final style-signature artifact joins:

- Tier-1 slop free-running rates.
- empirical compounding metrics.
- Biber-lite generated-output register rates.

It then computes a raw-vector distance between checkpoints. That distance is an
orientation aid, not an inferential statistic, because high-rate metrics
dominate raw Euclidean distance.

## Main Result 1: `slop_lexicon` Has A DPO Propensity Peak, But Not A Clean DPO Output Peak

The `slop_lexicon` feature is the best-supported Phase 2 result.

Teacher-forced neutral-normalized amplification:

| Stage | Normalized AF | 95% CI |
|---|---:|---:|
| Base | 1.467 | 1.114-2.046 |
| SFT | 1.695 | 1.315-2.289 |
| DPO | 1.999 | 1.446-2.837 |
| Final/RLVR | 1.659 | 1.189-2.351 |

Interpretation: DPO is the highest point estimate. Final/RLVR attenuates back
toward SFT. The intervals overlap, so this is a bounded positive signal rather
than a precise stage-localization proof.

Target-shape free-running rates per 1k generated tokens:

| Stage | Rate |
|---|---:|
| Base | 0.233 |
| SFT | 0.171 |
| DPO | 0.229 |
| Final/RLVR | 0.193 |

Interpretation: the free-running output result is more mixed than the
teacher-forced result. Base is narrowly above DPO, DPO is very close to base,
SFT is lowest, and final/RLVR is below DPO.

Compounding for `slop_lexicon`:

| Stage | Observed /1k Opp | Expected /1k Opp | Excess /1k | Realized AF | Repeat Gens |
|---|---:|---:|---:|---:|---:|
| Base | 0.615 | 0.232 | 0.383 | 1.051 | 192 |
| SFT | 0.697 | 0.336 | 0.361 | 1.192 | 145 |
| DPO | 0.638 | 0.439 | 0.199 | 1.091 | 199 |
| Final/RLVR | 0.582 | 0.445 | 0.137 | 0.994 | 180 |

Interpretation: observed generated opportunities exceed teacher-forced
expectation in every stage. This supports a bounded compounding signal. It
does not support a clean DPO-only compounding result: SFT has the highest
realized AF, base has the largest absolute excess, and final/RLVR is near the
teacher-forced reference-rate baseline.

## Main Result 2: SFT Often Suppresses Free-Running Feature Emissions

Across the target-shape generation grid, SFT is frequently the low-emission
checkpoint.

Feature rates per 1k generated tokens:

| Feature | Base | SFT | DPO | Final/RLVR | Max Stage |
|---|---:|---:|---:|---:|---|
| `rule_of_three_approx` | 1.019 | 0.488 | 0.861 | 0.802 | Base |
| `slop_lexicon` | 0.233 | 0.171 | 0.229 | 0.193 | Base |
| `contrastive_negation` | 0.136 | 0.041 | 0.141 | 0.142 | Final/RLVR |
| `stock_openers_closers` | 0.092 | 0.072 | 0.108 | 0.091 | DPO |
| `stock_openers` | 0.064 | 0.052 | 0.064 | 0.059 | DPO |
| `stock_closers` | 0.028 | 0.020 | 0.043 | 0.032 | DPO |

Interpretation: SFT lowers nearly every tracked free-running emission relative
to base. DPO then raises several of them again. Final/RLVR usually attenuates
from DPO, with `contrastive_negation` as the main exception.

This matters because an apparent DPO effect can sometimes be read as a rebound
from SFT suppression rather than as a monotonic effect of preference training
from the base model.

## Main Result 3: `rule_of_three_approx` Does Not Support A DPO Amplification Claim

`rule_of_three_approx` is the most frequent Tier-1 free-running feature, but
its stage pattern does not match a clean DPO amplification story.

Free-running rate per 1k generated tokens:

| Stage | Rate |
|---|---:|
| Base | 1.019 |
| SFT | 0.488 |
| DPO | 0.861 |
| Final/RLVR | 0.802 |

The base checkpoint is highest. DPO and final/RLVR are below base, although
they are above SFT.

The teacher-forced comma-pair extension proxy also does not peak at DPO:

| Stage | Raw AF | 95% CI |
|---|---:|---:|
| Base | 0.721 | 0.698-0.742 |
| SFT | 0.767 | 0.743-0.787 |
| DPO | 0.716 | 0.691-0.738 |
| Final/RLVR | 0.723 | 0.698-0.746 |

Interpretation: the teacher-forced proxy is scoreable and well-supported, but
it measures extension after a candidate two-item list. It should not be
over-read as the full open-vocabulary `rule_of_three_approx` behavior. Within
that proxy, SFT is the maximum point estimate and DPO is slightly below base.

## Main Result 4: Stock Phrase Behavior Splits Between Openers And Closers

The pooled `stock_openers_closers` view can be misleading because opener and
closer behavior differ sharply.

Teacher-forced `stock_openers` raw AF:

| Stage | Raw AF |
|---|---:|
| Base | 0.067 |
| SFT | 0.048 |
| DPO | 0.035 |
| Final/RLVR | 0.035 |

The model assigns much less probability to the retained stock opener starts
than the held-out references use.

Teacher-forced `stock_closers` raw AF:

| Stage | Raw AF |
|---|---:|
| Base | 73.304 |
| SFT | 65.076 |
| DPO | 68.634 |
| Final/RLVR | 74.032 |

The closer-side AF is extremely high because the held-out reference denominator
has only 18 positives in 41,187 opportunities, while the model assigns around
3 percent probability mass to closer phrases. The direction is meaningful, but
the magnitude should be read cautiously.

Free-running stock rates are small:

| Feature | Base | SFT | DPO | Final/RLVR |
|---|---:|---:|---:|---:|
| `stock_openers` | 0.064 | 0.052 | 0.064 | 0.059 |
| `stock_closers` | 0.028 | 0.020 | 0.043 | 0.032 |
| `stock_openers_closers` | 0.092 | 0.072 | 0.108 | 0.091 |

Interpretation: DPO is the free-running peak for pooled stock phrase behavior,
but final/RLVR attenuates from DPO. The high teacher-forced pooled AF is mainly
a closer effect, not an opener effect.

## Main Result 5: Final/RLVR Is Closest To DPO In Aggregate Style, But Not Identical

The final style signature combines Tier-1 emissions, compounding metrics, and
Biber-lite register features. It has 172 rows over 25 feature names and four
stages.

Raw-vector stage distances from final:

| Comparison | Shared Features | Euclidean | Cosine Distance |
|---|---:|---:|---:|
| Final/RLVR vs DPO | 43 | 30.567 | 0.000075 |
| Final/RLVR vs SFT | 43 | 35.564 | 0.000287 |
| Final/RLVR vs Base | 43 | 85.139 | 0.001272 |

Interpretation: final/RLVR is closest to DPO, then SFT, then base. This is an
aggregate generated-output style result. Because the vector uses raw metric
scales, high-rate Biber and compounding metrics dominate the Euclidean values.
The feature-level deltas are more interpretable than the distance alone.

Largest final-vs-base movements:

| Group | Metric | Feature | Final | Base | Delta |
|---|---|---|---:|---:|---:|
| compounding | observed /1k opportunities | `rule_of_three_approx` | 428.106 | 344.007 | 84.098 |
| compounding | observed /1k opportunities | `stock_openers` | 1.477 | 8.592 | -7.115 |
| Biber-lite | generation /1k tokens | demonstratives | 10.115 | 15.686 | -5.571 |
| Biber-lite | generation /1k tokens | infinitives | 13.623 | 17.426 | -3.803 |
| Biber-lite | generation /1k tokens | first-person pronouns | 11.638 | 15.325 | -3.687 |
| Biber-lite | generation /1k tokens | necessity modals | 2.505 | 5.037 | -2.532 |
| Biber-lite | generation /1k tokens | prediction modals | 1.879 | 3.851 | -1.972 |
| Biber-lite | generation /1k tokens | nominalizations | 27.309 | 25.499 | 1.810 |

Largest final-vs-DPO movements:

| Group | Metric | Feature | Final | DPO | Delta |
|---|---|---|---:|---:|---:|
| compounding | observed /1k opportunities | `rule_of_three_approx` | 428.106 | 458.417 | -30.311 |
| compounding | prior risk ratio | `stock_closers` | 9.399 | 6.198 | 3.201 |
| Biber-lite | generation /1k tokens | first-person pronouns | 11.638 | 10.302 | 1.336 |
| Biber-lite | generation /1k tokens | causal subordinators | 3.831 | 2.856 | 0.975 |
| Biber-lite | generation /1k tokens | hedges | 1.868 | 1.109 | 0.759 |
| Biber-lite | generation /1k tokens | infinitives | 13.623 | 13.009 | 0.614 |

Interpretation: final/RLVR remains DPO-like, but it is not a pure DPO copy.
It is slightly higher than DPO on some register markers and lower on several
slop or compounding markers.

## Biber-Lite Register Progression

Biber-lite features describe the register of generated text. They are measured
as per-1k-token regex rates over the same `t=1.0` target-shape generation
caches.

The main register pattern is that base is high on several conversational or
modal markers, while DPO/final reduce many of them.

Selected Biber-lite rates per 1k generated tokens:

| Feature | Base | SFT | DPO | Final/RLVR | Final - Base |
|---|---:|---:|---:|---:|---:|
| Demonstratives | 15.686 | 11.420 | 9.984 | 10.115 | -5.571 |
| Infinitives | 17.426 | 15.233 | 13.009 | 13.623 | -3.803 |
| First-person pronouns | 15.325 | 10.890 | 10.302 | 11.638 | -3.687 |
| Necessity modals | 5.037 | 2.460 | 2.416 | 2.505 | -2.532 |
| Prediction modals | 3.851 | 2.500 | 1.676 | 1.879 | -1.972 |
| Possibility modals | 6.739 | 4.509 | 4.759 | 5.126 | -1.613 |
| Conditional subordinators | 6.133 | 5.307 | 7.865 | 7.659 | 1.525 |
| Second-person pronouns | 4.349 | 4.607 | 5.450 | 5.548 | 1.198 |
| Nominalizations | 25.499 | 28.232 | 27.426 | 27.309 | 1.810 |

Interpretation: base has more demonstratives, first-person pronouns,
infinitives, and modal language. DPO/final move many of those downward while
increasing second-person pronouns, conditional subordinators, and
nominalizations. This is a broader register shift, not simply a slop-feature
rate shift.

## Checkpoint-By-Checkpoint Progression

### Base

Base already contains a substantial share of the measured style. It is highest
in target-shape free-running generation for `rule_of_three_approx` and
`slop_lexicon`, and it is high on many Biber-lite conversational or modal
features.

This means the Phase 2 results should not be read as "slop begins at DPO."
Some of the measured behavior is inherited from the base checkpoint or from
earlier training stages not separately isolated in this ladder.

### SFT

SFT often suppresses free-running emissions. It is the lowest stage for
`rule_of_three_approx`, `slop_lexicon`, `contrastive_negation`,
`stock_openers`, `stock_closers`, and the pooled stock phrase view.

At the same time, SFT raises teacher-forced `slop_lexicon` normalized AF above
base and is the highest stage for the `rule_of_three_approx` teacher-forced
comma-pair extension proxy. This split is important: local propensity under a
fixed opportunity and realized free-running emission do not always move
together.

### DPO

DPO is the clearest checkpoint for a post-SFT rebound. It is the teacher-forced
`slop_lexicon` normalized AF peak and the free-running maximum for pooled stock
phrases. It is also very close to base on target-shape `slop_lexicon`
free-running rate.

DPO is not the peak for everything. It is below base for `rule_of_three_approx`
free-running rate, and it is lowest on the `rule_of_three_approx`
teacher-forced extension proxy.

### Final/RLVR

Final/RLVR is closest to DPO in aggregate generated-output style. However, it
usually attenuates from DPO rather than extending the DPO movement.

Compared with DPO, final/RLVR is lower on:

- `slop_lexicon` free-running rate.
- `rule_of_three_approx` free-running rate.
- pooled stock phrase free-running rate.
- `rule_of_three_approx` observed compounding opportunities.
- conditional subordinators.

Compared with DPO, final/RLVR is higher on:

- first-person pronouns.
- causal subordinators.
- hedges.
- infinitives.
- passive voice approximation.
- public verbs.
- possibility modals.
- stock closer prior-window risk ratio.

The practical read is that final/RLVR preserves a DPO-like style signature but
does not act like a broad additional slop amplifier.

## Original Result Questions And Answers

### Did Phase 2 find a DPO-stage slop signal?

Yes, but bounded and feature-specific. The best evidence is the
teacher-forced `slop_lexicon` normalized AF, which peaks at DPO. DPO is also
the maximum for several stock free-running rates.

### Did Phase 2 show that final/RLVR further amplifies slop?

No. Final/RLVR is DPO-like in aggregate, but it attenuates from DPO on several
headline metrics. It is not a broad final-stage slop amplifier.

### Did free-running generation match teacher-forced propensity?

Only partially. Teacher-forced `slop_lexicon` peaks at DPO, while target-shape
free-running `slop_lexicon` is narrowly highest at base and very close at DPO.
This difference is expected: teacher-forced propensity measures local
probability mass under fixed opportunities, while free-running generation
also includes prompt interaction, earlier generated text, and decoding.

### Did Phase 2 find compounding?

Yes for `slop_lexicon`, in the bounded sense that observed generated
opportunity rates exceed teacher-forced expectations in all stages and
prior-window tests are positive. The stage pattern is not a clean DPO peak.

### Did Phase 2 measure the Biber features?

Yes. Biber-lite was added as a generated-output register comparison. It shows
that final/RLVR is much less base-like on demonstratives, infinitives,
first-person pronouns, and modals, while higher than base on nominalizations,
conditional subordinators, and second-person pronouns.

## Caveats

The key caveats are:

- This is a bounded Phase 2 close-out, not the full production-scale plan.
- Teacher-forced and free-running measurements answer different questions and
  should not be collapsed into one score.
- `rule_of_three_approx` teacher-forced scoring uses a third-item extension
  proxy, not full open-vocabulary construction initiation.
- `contrastive_negation` has sparse held-out reference support.
- `stock_closers` teacher-forced AF is very large because the reference
  denominator is small; the direction is informative, but the magnitude should
  be treated cautiously.
- Biber-lite features are regex proxies, not full Biber annotations.
- The style-signature distance is computed on raw metric scales, so it is a
  useful orientation aid but not a formal statistical conclusion.

## Final Conclusion

Phase 2 supports a nuanced version of the slop-amplification hypothesis. The
model ladder does show post-training reshaping of slop and style features, and
`slop_lexicon` has its clearest local propensity peak at DPO. There is also
evidence that generated text can compound `slop_lexicon` once it appears.

However, the results do not support a simple monotonic story where each later
checkpoint becomes more slop-heavy. Base already emits several features at high
rates. SFT often suppresses free-running emissions. DPO restores or increases
some of them and is the main preference-stage signal. Final/RLVR remains
DPO-like in aggregate style but attenuates several headline slop markers.

The best one-sentence conclusion is:

> In this bounded OLMo 3/Dolci Phase 2 slice, DPO is the clearest checkpoint
> for increased `slop_lexicon` propensity, but final/RLVR does not broadly
> amplify slop beyond DPO; the final model is DPO-like in output style while
> partially attenuating several slop and base-heavy register dimensions.

## Primary Artifacts

Final report inputs:

- `artifacts/phase2/analysis/olmo3_amplification_spectrum_single_temp_t1_v6.csv`
- `artifacts/phase2/analysis/olmo3_amplification_spectrum_single_temp_t1_v6_summary.md`
- `artifacts/phase2/analysis/olmo3_generation_stage_grid_target_shape_512prompt_8comp_t1_1024.csv`
- `artifacts/phase2/analysis/olmo3_generation_compounding_target_shape_512prompt_8comp_t1_1024_tf1024.csv`
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
