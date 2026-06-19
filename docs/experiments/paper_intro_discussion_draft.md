# Paper Introduction And Discussion Draft

This draft contains manuscript-facing Introduction, Discussion, Limitations,
and Conclusion prose for the bounded slop-style paper. The claim matrix remains
the authority for paper-safe wording:
`docs/experiments/paper_claim_matrix.md`.

## Introduction Draft

Large language model outputs often share recognizable stylistic habits:
stock phrases, inflated lexical choices, repetitive rhetorical structures,
and a polished assistant voice that can feel detached from the local task.
These features are often grouped under informal labels such as "slop", but the
label hides several different mechanisms. A phrase can appear because it was
common in pretraining data, because supervised fine-tuning taught an assistant
register, because preference optimization changed local propensities, or
because generation self-conditions after the first stylistic marker appears.

This paper treats slop style as a measurement problem. Rather than asking
whether a whole answer is slop, we measure lexical, rhetorical, and register
features across open data and model ladders. The goal is to localize where a
feature enters the pipeline and how it changes across stages. We use open
post-training ladders because they expose the ingredients needed for that
localization: pretraining references, SFT targets, preference pairs, and model
checkpoints before and after post-training steps.

Raw generation counts are not enough for this task. If a model emits a phrase
often, that can reflect inherited data frequency, a general answer-register
shift, a local model preference for initiating the phrase, or a path-dependent
generation effect after a prior hit. We therefore combine corpus measurement,
full pybiber register analysis, teacher-forced propensity, free-running
generation, compounding tests, an EQ-Bench Slop Score bridge, and
detector-driven candidate discovery. Each layer answers a different question.
Corpus rates ask what the data contain. Teacher-forced propensity asks what a
checkpoint wants to do in the same context. Free-running generation asks what
appears in sampled answers. Compounding asks whether one marker makes later
markers more likely.

Our main finding is not that post-training uniformly creates slop. The evidence
supports a more specific amplification-spectrum view. In OLMo/Dolci data,
post-training corpora move sharply away from narrative and personal web prose
toward nominalized, adjectival, answer-like exposition. This broad register
movement is not simply an increase in hedging or intensification. On the model
side, the original slop lexicon shows the clearest OLMo preference-stage
teacher-forced propensity peak, but the corresponding Dolci preference data do
not show chosen-response enrichment for that feature. In sampled output, the
same feature also shows self-conditioning. SmolLM3 no-think exhibits related
style pressure but localizes the main lexical movement earlier, around SFT and
post-SFT stages.

We also show that aggregate slop scores and detector features are useful but
need to be kept in their lane. The EQ-Bench Slop Score provides a public
benchmark bridge, but its stage path does not always match the feature-level
propensity story. Detector attribution finds plausible Tier-3 style families,
and selected candidates have real exact sequence-mass support, but those
families remain provisional until human perceptibility and matcher precision
validation are complete.

The resulting contribution is a bounded but coherent account of slop style as
a spectrum of measurable features. Some features look inherited, some are
amplified by local model propensity, some compound during generation, and some
are better understood as part of the assistant answer register. This
distinction matters because it changes the causal story: a surface tic in final
output is not automatically evidence of preference-data selection, and an
aggregate slop score is not enough to identify where the style came from.

## Discussion Draft

The central lesson is that "slop style" is not one object. It is a family of
features that move differently across the post-training pipeline. Treating it
as a single scalar can be useful for benchmarking, but it obscures the
mechanisms that matter for diagnosis. The same model can have a broad
answer-register shift in its data, a feature-specific DPO propensity peak, and
a different free-running output maximum. Those are not contradictions; they
are different views of the same pipeline.

The full pybiber layer is important because it changes the baseline story. If
we only counted familiar lexical markers, it would be tempting to describe the
phenomenon as "more hedging" or "more ornate wording." The corpus-side
register evidence says something broader: the SFT data are much less narrative
and personal than the Dolma sample and much more nominalized and expository.
Hedges, amplifiers, and emphatics are lower in SFT than in Dolma. The assistant
style we observe is therefore not just a collection of lexical tics. It is a
shift toward a particular answer-register prose, with some conspicuous markers
layered on top.

The OLMo `slop_lexicon` result gives the cleanest example of why fixed-context
propensity is needed. The feature has a DPO-stage teacher-forced peak, yet the
measured Dolci chosen/rejected contrast does not show chosen responses as more
lexicon-heavy than rejected responses. That does not prove a specific optimizer
mechanism by itself. It does, however, rule out the simplest imitation story
for this feature under the measured data contrast. In the analysis taxonomy,
the evidence supports a dynamics-driven label more than a signal-driven label.

The SmolLM3 comparison sharpens the same point. SmolLM3 no-think shows related
style pressure, including high post-SFT EQ-Bench scores and generated
`slop_lexicon` rates, but its stage localization differs from OLMo. That is not
a failed replication. It is evidence that the phenomenon is feature-like and
pipeline-dependent rather than a universal DPO fingerprint. The ladders differ
in scale, data, preference algorithm, and checkpoint construction, so the safe
cross-ladder claim is shared pressure with different localization.

The detector-discovery work expands the feature surface while also showing why
validation cannot be skipped. The detector and attribution pipeline can produce
plausible style families such as process framing, follow-up offers, additive
transitions, and prescriptive instruction. The exact Tier-3 rerun confirms that
some of those families correspond to model propensity shifts rather than only
to free-generation counts. But detector-positive spans are not automatically
human-perceived slop, and sparse reference denominators can make very large AF
values unstable. Human perceptibility labels and matcher precision checks are
therefore part of the scientific claim, not cosmetic cleanup.

A practical implication is that mitigation and evaluation should be
feature-specific. If a style marker is inherited from SFT data, data curation
or target rewriting may be the right intervention. If it is amplified by local
propensity at a preference checkpoint, preference objectives or calibration may
matter more. If it compounds only after an earlier hit, decoding and
self-conditioning controls become relevant. A single aggregate score cannot
distinguish those cases.

The paper also argues for scope discipline in slop measurement. A bounded,
well-instrumented amplification spectrum is more interpretable than an
exhaustive generation grid whose extra cells do not change the estimand. The
completed reduced SGLang panels, exact teacher-forced runs, pybiber corpus
analysis, and detector addendum support the bounded claims. They do not support
a claim that all alignment methods or all LLMs produce the same style
trajectory.

## Limitations Draft

First, the evidence is bounded to specific open ladders and retained samples.
The OLMo corpus-side analysis uses an English-filtered retained Dolma
sample from an 80k scan, not the full Dolma corpus. The SmolLM3 comparison uses
no-think mode and a checkpoint sequence whose preference/final cells are not
structurally identical to OLMo's DPO/final path.

Second, Tier-1 precision validation is a scoped boundary rather than a global
guarantee. The interim gate passes `contrastive_negation`,
`slop_lexicon`, `stock_openers`, and `stock_closers`, while
`rule_of_three_approx` is demoted for publication-grade independent regex
claims after the 50-label interim check. The pooled opener/closer feature
is a derived convenience metric in the current evidence package. The pooled
lexicon still requires item- and context-level interpretation.

Third, several teacher-forced features have sparse reference denominators.
Large AF point estimates for features such as follow-up offers, response
constraints, careful reasoning, and some stock closer variants should be
treated as provisional until denominator support improves. Sparse denominators
are especially important when the feature is rhetorically salient but rare in
the reference package.

Fourth, Dolci DPO chosen/rejected contrasts are not pure human-preference
contrasts. They include synthetic and Delta Learning-style construction
regimes, so chosen-vs-rejected differences can reflect strong-model versus
weak-model style that is distilled into OLMo. This is useful for studying
cross-model style transmission, not direct evidence of human stylistic
preference.

Fifth, full pybiber has been run over the Phase 1 corpus-side samples, but not
over the generation caches. The paper therefore uses pybiber for corpus-side
register interpretation only and does not make generated-output full-register
claims.

Sixth, the EQ-Bench Slop Score is aggregate by design. The scorecard includes
output-bootstrap intervals, but those intervals quantify generation-sample
variability rather than turning the score into the main causal outcome
variable.

Seventh, Phase 4 detector features remain candidates. The detector-discovery
pipeline produced matcher specs and yielded exact sequence-mass support for
selected candidates, but human perceptibility labels are still absent.
The human-facing taxonomy should therefore distinguish detector-discovered
machine features from validated human-perceived slop.

Finally, the completed production scope is intentionally smaller than the
original exhaustive generation plan. The reduced SGLang panels answer the
paper's questions about register shift, local propensity, free-running
emission, and compounding. They are not the same evidence as a
completed 5,000 prompt x 8 completion x 3 temperature OLMo grid.

## Conclusion Draft

Slop style is best understood as a spectrum rather than a single defect. In
open post-training ladders, broad register movement can be separated from
feature-specific propensity amplification and from generation-time
self-conditioning. That separation changes the interpretation of familiar
style markers. Some are inherited from the data, some are amplified at
particular checkpoints, some recur because generation has entered a stylistic
track, and some are artifacts of the broader assistant-answer register.

The evidence supports a bounded thesis: OLMo/Dolci post-training data shift
strongly away from narrative and personal prose and toward nominalized,
adjectival assistant-answer exposition. That movement is not an aggregate
increase in hedging, and the DPO chosen side is not uniformly more slop-like
than the rejected side; pybiber instead shows chosen responses as more
descriptive/expository and rejected responses as more personal,
narrative, and mental-state framed. On top of that broad register shift,
OLMo's original slop lexicon shows a DPO-stage fixed-context propensity peak
that is not explained by measured chosen-response enrichment. SmolLM3 shows
related but differently localized style pressure, and detector attribution can
propose new style families whose promotion requires human validation. This is
a narrower claim than "alignment creates slop", but it is more useful: it
identifies which measurements are needed to localize a style marker in the
post-training pipeline.
