# Paper Reviewer FAQ

Date: 2026-06-18

Purpose: answer likely reviewer and collaborator questions about scope,
terminology, and interpretation for the bounded slop-style paper. The claim
matrix remains the authority for allowed claim language:
`docs/experiments/paper_claim_matrix.md`.

## What Does "Slop Style" Mean Here?

In this paper, "slop style" means measurable stylistic markers and register
features. It does not mean factual error, low answer quality, or uselessness.
A useful answer can contain a measured marker, and a poor answer can contain
none.

The paper separates four localization layers:

- Corpus rates: what appears in pretraining, SFT, and preference data.
- Model propensity: what a checkpoint locally assigns probability to under
  teacher forcing.
- Free-running emission: what appears in sampled model outputs under a
  decoding policy.
- Self-conditioning: whether later markers become more likely after an
  earlier marker appears.

It also reports two auxiliary measurement surfaces. EQ-Bench is an aggregate public benchmark bridge, useful for comparison but not the paper's causal estimand.
Detector-derived Tier 3 is a candidate-feature discovery surface, not a
human-perceived slop result until annotation and matcher precision checks
promote it.

## Is This A Claim That Alignment Creates Slop?

No. The supported thesis is narrower:

> Slop style is feature-specific, stage-specific, and partly a register shift.

The OLMo/Dolci corpus data show a strong shift away from narrative/personal
web prose toward noun-heavy assistant-answer prose. The OLMo model ladder
shows a DPO-stage local-propensity peak for the original `slop_lexicon`, but
not every feature follows that path. SmolLM3 shows related style pressure with
different stage localization.

## Why Use Pybiber?

The pybiber layer prevents a misleading lexical-only story. If we only counted
slop words, it would be tempting to describe the post-training shift as "more
hedging" or "more ornate wording." Full pybiber shows the broader register
movement: SFT data are much less narrative/personal than the Dolma sample and
more nominalized, adjectival, and expository.

It also clarifies the DPO contrast. Chosen responses are more
descriptive/expository on this register layer, while rejected responses are
more personal, narrative, and mental-state framed. That is different from a
claim that chosen responses are uniformly more slop-like.

The pybiber result is corpus-side. Generated-output full pybiber was not run,
so the paper does not make generated-output full-register claims.

## Are Pybiber And The Generated-Output Features Measuring The Same Thing?

No. Pybiber is a corpus-side register measurement over source data: Dolma,
Dolci SFT, and Dolci DPO chosen/rejected text. It supports claims about how the
training and preference-data register changes.

The generated-output results use Tier-1 regex features, EQ-Bench aggregate
scores, teacher-forced propensity, free-running emission, compounding, and
Tier-3 candidate matchers. Those results support model-behavior claims, but
they should not be described as full generated-output pybiber evidence.

## Why Include EQ-Bench Slop Score?

EQ-Bench Slop Score is included as an aggregate benchmark bridge. It helps
connect this paper to a public slop-score vocabulary and gives a useful
lexical density diagnostic.

It is not the causal measurement layer. The causal-style analysis is the
feature-level combination of corpus rates, teacher-forced propensity,
free-running rates, and compounding.

This distinction matters because aggregate EQ-Bench movement can disagree with a feature-specific propensity path.
The paper treats that disagreement as evidence that "slop style" is not one
scalar quantity.

## Why Is DPO Chosen-Vs-Rejected Not Treated As Pure Human Preference?

Dolci DPO includes synthetic and Delta Learning-style construction regimes.
Chosen-vs-rejected differences can reflect strong-model versus weak-model
style being distilled into OLMo, not only direct human taste.

The paper therefore treats the Dolci DPO contrast as preference-data evidence,
not as a universal statement about human preference.

## Does The Paper Need Full Dolma Source Coverage?

No. The OLMo pretraining reference is an English-filtered retained Dolma
sample from a bounded 80k scan, not a claim about every Dolma document or every
pretraining source. That retained sample is enough for the paper's bounded
contrast against Dolci SFT and DPO, especially because the main corpus-side
claim is a large register shift rather than a source-inventory census.

Full pretraining-source coverage would support stronger source-mixture claims,
but it is not required for the current claim matrix. The paper should describe
the Dolma reference as retained-sample evidence and avoid treating it as an
exhaustive pretraining-corpus measurement.

## What Is The Main OLMo Result?

The cleanest OLMo model-side result is the original `slop_lexicon`.
Teacher-forced neutral-normalized AF rises from base and SFT to a DPO peak,
then falls at final/RLVR. The measured Dolci chosen/rejected corpus contrast
does not show chosen-response enrichment for the same feature. That rules out
the simplest imitation account for this measured OLMo feature.

The paper does not claim DPO universally increases all slop markers.

## Which Tier-1 Features Are Publication-Safe?

The paper treats `contrastive_negation`, `stock_openers`, `stock_closers`, and
the frozen historical `slop_lexicon` index as interim precision-supported
features. That does not mean the original validation plan is fully complete,
and it does not mean every lexicon item is semantically diagnostic in every
context.

`rule_of_three_approx` is demoted for independent publication-grade claims,
because manual labels show too many false positives. `stock_openers_closers` is a derived pooled convenience view, not an independently validated matcher.

## Why Is The Generation Grid Smaller Than The Original Plan?

The originally imagined grid was intentionally reduced after benchmarking and
early scale checks. The active evidence package uses bounded SGLang generation
panels, exact teacher-forced runs, compounding analysis, and cross-ladder
comparison rather than the abandoned exhaustive 5,000 prompt x 8 completion x
3 temperature OLMo grid.

The reduced grid is the production estimand. It supports bounded paper claims;
it is not presented as if the abandoned grid had been run.

## What Does SmolLM3 Add?

SmolLM3 no-think gives replication pressure across a different organization,
scale, data mixture, checkpoint construction, and preference algorithm. It
supports the broad conclusion that related style pressure appears beyond OLMo,
but it does not provide a one-to-one DPO-stage replication because its pipeline
is structurally different.

The point of the SmolLM3 panel is not that all open ladders share identical
stage localization. It is that a second ladder shows related style pressure,
with a different localization pattern that helps separate broad style pressure
from an OLMo-specific DPO story.

## How Should Reviewers Read Sparse Tier-3 Effects?

Sparse Tier-3 effects are useful for prioritizing follow-up work, not for
headline taxonomy claims. Follow-up offers have a very large AF point estimate,
but only 11 reference initiations in the exact rerun. That makes the signal
interesting and provisional, not denominator-stable.

The stronger Tier-3 claims are about detector-discovery infrastructure and
denominator-supported candidates such as process framing, additive transitions,
and prescriptive instruction. Human perceptibility labels still decide whether
candidate detector families become human-facing slop categories.

## What Is Still Not Done?

Two items remain submission blockers:

1. Target venue decisions and final template adaptation.
2. Human perceptibility labels for Phase 4 detector-derived candidates.

The Phase 4 detector pipeline is real, and selected candidates have
teacher-forced support. But detector-positive spans are still candidate
machine-detectable style families until humans label whether they are
perceived as slop in context.

## What Should A Reviewer Trust Most?

The strongest bounded claims are:

- OLMo/Dolci post-training data shift register toward assistant-answer prose.
- That broad register movement is not simply more hedging.
- OLMo `slop_lexicon` has a DPO-stage teacher-forced propensity peak.
- The OLMo `slop_lexicon` peak is not directly explained by chosen-response
  enrichment in the measured Dolci DPO data.
- Propensity, sampled output, and compounding answer different questions and
  should not be collapsed into one score.

Claims about detector-discovered human-perceived slop families remain provisional
until annotation is complete.
