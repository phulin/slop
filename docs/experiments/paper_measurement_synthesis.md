# Paper Measurement Synthesis

Date: 2026-06-18

Purpose: give reviewers and future editors a compact map of how the paper's
measurement layers fit together. The claim matrix remains the authority for
allowed claims; this note explains why the layers should not be collapsed into
one global slop score.

## One Thesis, Multiple Measurements

The paper characterizes "slop style" as a family of measurable stylistic
effects. No single measurement layer is treated as the whole phenomenon.
Instead, the paper separates six questions:

| Layer | Question answered | Main evidence | Boundary |
|---|---|---|---|
| Phase 1 corpus rates | What stylistic material is present in source and post-training data? | Tier-1 corpus rates and full pybiber register features | Data-side evidence, not model behavior by itself |
| Full pybiber register | How does broad linguistic register shift from pretraining text to post-training targets? | `phase1_pybiber_register_analysis.md`; pybiber means/deltas/intervals | Corpus-side Phase 1 result; generated-output full pybiber is not claimed |
| EQ-Bench Slop Score | Does a public aggregate slop-style benchmark see similar movement? | Phase 2 EQ-Bench score panels and bootstrap intervals | Benchmark bridge, not the causal estimand |
| Teacher-forced propensity | Does a checkpoint assign more probability mass to a feature in matched contexts? | Phase 3 amplification spectra and AF classifications | Decoding-free local propensity; depends on denominator support |
| Free-running emission and compounding | What appears in sampled generations, and does style self-condition after appearing? | Phase 3 generation rates and compounding tables | Decoding-dependent realized output, not interchangeable with propensity |
| Detector-derived Tier 3 | What additional candidate style families can a trained detector surface? | Phase 4 detector clusters, matcher specs, and exact Tier-3 rerun | Machine-detectable candidates until human labels and precision checks promote them |

## Substantive Interpretation

The pybiber result supplies the broad register backbone. In the OLMo/Dolci
samples, post-training data move away from narrative and personal web prose
toward nominalized, adjectival, answer-like exposition. This is not simply an
increase in hedging or intensification: the pybiber hedge, amplifier, and
emphatic features are lower in SFT than in the retained Dolma sample. That
means the paper should describe a register shift toward polished exposition,
not a generic "alignment adds hedging" story.

Pybiber also blocks a second overread: Dolci DPO chosen responses are not
uniformly "more slop-like" than rejected responses. On the register layer,
chosen responses look more descriptive and expository, while rejected
responses look more personal, narrative, and mental-state framed. This makes
the preference-data contrast useful, but not a pure human-preference or
chosen-equals-slop interpretation.

The Tier-1 and EQ-Bench layers sit on top of that register shift. The original
`slop_lexicon` gives the cleanest OLMo model-side local-propensity result,
peaking at DPO in the bounded teacher-forced spectrum. EQ-Bench gives a useful
public aggregate comparison, but its OLMo trajectory peaks at SFT while the
OLMo `slop_lexicon` propensity result peaks at DPO. That mismatch is a result,
not a defect: aggregate lexical-density scores and feature-specific propensity measure different things.

Generated-output rates add a third view. They show what the model actually
emits under a decoding policy, and the compounding analysis tests whether
features are more likely after the style has already appeared. These sampled
outputs can agree with teacher-forced propensity in broad direction while
differing in exact stage localization or temperature sensitivity.

Phase 4 detector discovery extends the feature surface, but it is deliberately
kept provisional. Detector clusters can identify machine-detectable style
families and exact Tier-3 sequence-mass shifts, but they should not be called
human-perceived slop until the prepared blind-label package is labeled,
adjudicated, and summarized.
They remain candidate-only until human labels are available.

## How To Word The Paper

Use this hierarchy when writing or reviewing the paper:

1. Start with the register result: post-training data shift toward an
   assistant-answer register.
2. Then describe feature-specific amplification: some markers, especially the
   OLMo `slop_lexicon`, localize to particular model stages.
3. Treat EQ-Bench as a public aggregate bridge that can agree or disagree with
   feature-specific AF.
4. Keep propensity, sampled emission, and compounding separate.
5. Keep detector-discovered Tier-3 families candidate-only until human labels
   are available.

Avoid these collapses:

- Do not say pybiber proves generated-output register shift unless generated
  full-pybiber evidence is added.
- Do not say EQ-Bench is the outcome variable for the paper.
- Do not say DPO universally creates slop.
- Do not say chosen preference responses are uniformly more slop-like than
  rejected responses.
- Do not call Phase 4 detector clusters human-perceived slop before manual
  perceptibility labels exist.
