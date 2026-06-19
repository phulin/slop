# Paper Claim Matrix

Date: 2026-06-18

Purpose: freeze the current paper-facing claim language for the bounded
experiments. This is not a replacement for the detailed phase reports; it is a
map from research-paper claims to the artifacts that support them, the
remaining caveats, and the wording that should be used unless later evidence
changes the status.

## Scope Discipline

The current project supports a bounded but coherent paper about stylistic
register and feature-level amplification in open post-training ladders. It does
not yet support a universal claim about all LLMs, all alignment methods, or a
single global "slop score." The safest framing is:

> We measure a set of visible "slop style" markers and broader Biber-register
> features across open pretraining, SFT, preference-data, and checkpoint
> ladders. The main result is feature-specific: post-training data shift toward
> polished answer-register prose, while some surface markers in model
> propensity and sampled output amplify at particular stages. The direction and
> mechanism differ by feature and by ladder.

Tier-1 regex claims must also follow
`docs/experiments/paper_tier1_publication_policy.md`. In particular, the
current paper can use `slop_lexicon` as a frozen historical model-side index
for propensity/generation/compounding claims and bounded corpus-rate context
with item-level caveats; the current validation status separates
precision-supported, demoted, derived, and exploratory rows.

## Evidence Matrix

| Claim ID | Paper claim | Evidence | Strength | Caveat | Allowed wording |
|---|---|---|---|---|---|
| C1 | Alignment data are register-shifted relative to pretraining data. | Phase 1 English-filtered fixed samples: 5,740 Dolma docs, 40,000 SFT targets, 40,000 DPO pairs; full 67-feature pybiber means in `artifacts/stage1/census/phase1_pybiber_register_means.csv`; selected-feature document-bootstrap intervals in `artifacts/stage1/census/phase1_pybiber_register_intervals.csv`; summary in `docs/experiments/phase1_pybiber_register_analysis.md`. | Strong for the sampled OLMo/Dolci data-side comparison. | Dolma is a retained English sample from an 80k scan, not full Dolma; intervals cover selected paper-facing features, not every pybiber feature. | "In the OLMo/Dolci samples, post-training data shift sharply away from narrative/personal web prose toward nominalized, adjectival, answer-like exposition." |
| C2 | The data-side register shift is not just "more hedging." | Full pybiber selected means: hedges, amplifiers, and emphatics are higher in Dolma than SFT; nominalizations and attributive adjectives rise in SFT. | Strong as a correction to an overly broad hedging story. | Applies to pybiber aggregate features on the sampled data; individual generated outputs can still contain hedged formulations. | "The broad register movement is expository and nominalizing, not a simple aggregate increase in hedging or intensification." |
| C3 | DPO chosen and rejected responses differ in register, but chosen is not uniformly more slop-like. | Phase 1 DPO pair deltas and full pybiber chosen-vs-rejected deltas: chosen higher on attributive adjectives, adverbs, prepositions, broad noun density; rejected higher on pronouns, past tense, private verbs, infinitives, nominalizations, analytic negation. | Moderate to strong descriptively. | Dolci DPO includes Delta Learning and synthetic construction regimes; not a pure human-preference result. | "Within Dolci DPO, chosen responses are more descriptive/expository on the Biber layer, while rejected responses are more personal/narrative; this is a preference-data contrast, not direct evidence of human taste." |
| C4 | EQ-Bench style scoring is now implemented as an exploratory benchmark bridge, not as the headline estimand. | `slop-eqbench-score` CLI, vendored EQ-Bench word/trigram lists and prompt manifest, exact/portable contrast backends; Phase 2 score comparison in `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_stage_comparison.md`; bootstrap intervals in `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.md`. | Strong for tooling and exploratory comparison. | Score is aggregate and benchmark-shaped; bootstrap intervals quantify output-sample variability but do not make the score causal. | "We include an EQ-Bench Slop Score bridge for comparability, but analyze it as an aggregate diagnostic rather than the causal amplification measure." |
| C5 | OLMo EQ-Bench generation scores rise from base to SFT, then ease through DPO/final. | Phase 2 EQ-Bench panel with 500-bootstrap output intervals: OLMo base 5.179 [4.689, 5.773], SFT 7.580 [6.863, 8.326], DPO 7.006 [6.460, 7.562], final 6.650 [6.050, 7.196] on 512 prompts x 8 completions at temperature 1.0. | Moderate descriptive output evidence with bootstrap uncertainty. | Prompt/generation panel is bounded; intervals are over valid generated outputs, not over model families or prompt universes. | "On the bounded OLMo generation panel, EQ-Bench Slop Score peaks at SFT rather than DPO/final." |
| C6 | SmolLM3 no-think EQ-Bench generation scores rise sharply after SFT and remain high after preference/final. | Phase 2 EQ-Bench panel with 500-bootstrap output intervals: SmolLM3 no-think base 2.798 [2.528, 3.063], SFT 7.169 [6.657, 7.635], DPO/APO 8.345 [7.815, 8.942], final 8.254 [7.746, 8.812]. | Moderate descriptive output evidence with bootstrap uncertainty. | Same aggregate-score and bounded-panel caveats as C5. DPO and final intervals overlap, so the safe claim is that both remain high. | "In the SmolLM3 no-think panel, EQ-Bench-style lexical density rises strongly after SFT and remains high after preference optimization." |
| C7 | The cleanest OLMo teacher-forced amplification result is the original slop lexicon at the preference stage. | Phase 3 integrated spectrum/classification: OLMo normalized teacher-forced AF for `slop_lexicon` rises base 1.467, SFT 1.695, DPO 1.999, final 1.659. | Strong within the bounded OLMo teacher-forced feature set. | Applies to supported opportunity contract and reduced production scope; not all features have equally strong denominator support. | "For OLMo, the original slop lexicon shows the clearest preference-stage propensity increase, peaking at DPO." |
| C8 | OLMo `slop_lexicon` amplification is not directly explained by chosen-response complicity. | Phase 3 joins Phase 1 paired Dolci tests: `slop_lexicon` chosen-minus-rejected is negative in aggregate and BH q=0.963 while DPO teacher-forced AF rises. | Strong for this feature under current tests. | Delta Learning means the pair contrast is strong-vs-weak model/style signal, not pure human preference. | "For OLMo `slop_lexicon`, the DPO-stage propensity peak appears dynamics-driven rather than a simple imitation of higher-slop chosen responses." |
| C9 | Free-running output rates and teacher-forced propensities do not always tell the same story. | Phase 3: OLMo `slop_lexicon` free-running rates peak at DPO but temperature-panel realized AF does not; compounding excess is positive at all OLMo stages. SmolLM3 shows high realized AF at SFT and post-preference stages. | Strong as a methodological point. | The reduced panels are purpose-specific, not the abandoned exhaustive grid. | "Slop style must be measured both as propensity and as realized generation; the two views can diverge." |
| C10 | Cross-ladder evidence supports shared style pressure but not identical stage localization. | Phase 3 OLMo-vs-SmolLM3 comparison: 24 aligned feature-stage rows, 8 shared non-missing AF values, Spearman AF 0.762, Pearson AF 0.978; OLMo classifies `slop_lexicon` as preference-amplified, SmolLM3 as SFT-amplified. | Moderate. | Shared non-missing AF support is small; ladders differ in algorithms, prompts, and checkpoint construction. | "The two open ladders agree that related style markers amplify, but they localize the main effect to different stages." |
| C11 | Phase 4 detector discovery produced useful Tier-3 candidate features, but human-perceived slop validation remains open. | ModernBERT detector, IG over 10,000 generated docs, GTE/HDBSCAN clusters, matcher specs, candidate census, annotation package, proxy Venn under `artifacts/phase4/modernbert_detector_combined_v2_clean/`; report in `docs/experiments/phase4_detector_discovery_report.md`; labeling protocol in `docs/experiments/phase4_human_perceptibility_protocol.md`; annotation codebook in `docs/experiments/phase4_human_annotation_codebook.md`; readiness audit, blinded pilot/full-package packets, and import path in `docs/experiments/phase4_human_annotation_readiness.md`; blank-label summary in `docs/experiments/phase4_human_perceptibility_summary.md`. | Strong for pipeline execution and annotation readiness; preliminary for human-facing taxonomy. | True human perceptibility labels are not complete: readiness audit reports a balanced unlabeled package, and current summary is 0/100 labeled for each candidate. | "Detector attribution discovers candidate style families; promotion into the main slop taxonomy awaits manual perceptibility and precision validation." |
| C12 | Some Phase 4 detector candidates have real teacher-forced sequence-mass support. | 512-document exact OLMo Tier-3 run: process framing raw AF 1.401 to 1.957 from base to final; additive transition 0.565 to 1.012; prescriptive instruction 0.301 to 0.763; 31/33 paired stage effects BH-FDR significant. | Moderate to strong for denominator-supported candidates. | Sparse candidates remain unstable; neutral-normalized AF can decline when neutral controls rise faster. | "The Tier-3 exact rerun confirms that selected detector-derived patterns correspond to model propensity shifts, especially process framing and alignment-rising instruction/transition patterns." |
| C13 | Follow-up offers are a large but sparse Phase 4 signal. | 512-document exact run: follow-up offer ref inits 11; raw AF base 6.171, SFT 38.393, DPO 39.880, final 39.126. | Strong signal size, weak denominator stability. | Too sparse for a standalone headline claim. | "Follow-up offers show very large model-over-reference propensity, but current denominator support makes them a provisional rather than headline result." |
| C14 | Formatting features should not be part of the active slop-style thesis. | `list_header_bold_lead_in` retired from active feature surface in `EXPERIMENTS.md`; current Phase 1 scope excludes formatting. | Strong project-scope decision. | Earlier artifacts may still mention historical formatting features. | "We exclude markdown/list formatting from the core style claims and focus on lexical, rhetorical, and register features." |

## Claim Strength Bands

Use these bands in the paper draft:

- **Headline-ready, bounded:** C1, C2, C7, C8, C9.
- **Paper-usable with explicit caveat:** C3, C4, C5, C6, C10, C12, C13.
- **Methods/status rather than substantive result:** C11, C14.

## Claims To Avoid

Do not write:

- "DPO universally creates slop."
- "Preference data are uniformly more slop-like than rejected responses."
- "Alignment simply adds hedging."
- "The EQ-Bench Slop Score is the project outcome variable."
- "Phase 4 discovered human-perceived slop features."
- "Generated-output register proxy artifacts are equivalent to full pybiber."
- "The reduced SGLang panels are the same as the originally proposed exhaustive
  5,000 x 8 x 3 OLMo grid."

## Remaining Evidence Gaps

The next paper-facing work should target gaps that would change claim strength,
not just produce more rows:

1. Add manual perceptibility labels for the prepared Phase 4 pilot sheet or
   full annotation package before calling detector clusters human-perceived
   slop.
2. Expand denominator support only for sparse features that matter to the
   story: follow-up offers, response constraints, careful reasoning, and
   stock opener/closer variants.
3. Decide whether full pybiber over generated outputs is worth the cost. The
   corpus-side full pybiber result is already useful; generated-output register
   proxy artifacts are retired from the paper's active feature surface.
4. EQ-Bench bootstrap intervals now exist for Figure 2, but the score remains
   an aggregate benchmark bridge rather than an inferential causal estimand.
