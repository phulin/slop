# Paper Results Draft

This Results draft collects manuscript-facing result prose for the bounded
slop-style paper. The claim matrix
(`docs/experiments/paper_claim_matrix.md`) remains the authority for allowed
wording and caveats. The claim-to-section review trail is
`docs/experiments/paper_claim_evidence_map.md`; figure and table placement is
defined in `docs/experiments/paper_figure_table_manifest.md`.

## Result 1: Post-Training Data Shift Toward Answer-Register Prose

Claim IDs: C1, C2, C3.

The clearest corpus-side result is a broad register shift from pretraining
text to post-training data. In the English-filtered OLMo/Dolci samples, SFT
targets are much less narrative and personal than the Dolma pretraining
reference. In plain terms, they contain far fewer signals of someone recounting
events, describing personal experience, or tracking third-person actors. Past
tense falls from `57.065` [`55.212`, `59.078`] to `4.315` [`4.195`,
`4.460`]; first-person pronouns fall from `44.732` [`42.971`, `46.451`] to
`9.360` [`9.086`, `9.792`]; third-person pronouns fall from `51.809`
[`50.260`, `53.734`] to `4.154` [`4.022`, `4.297`]; and clausal coordination
falls from `12.089` to `3.004`. Document-bootstrap intervals over retained
rows are narrow relative to these large shifts.

The same SFT targets move toward a more nominalized and expository register:
they use more nouns, noun-derived abstractions, descriptive modifiers, and
present-tense general statements. Nominalizations rise from `8.944` to
`24.805`, attributive adjectives rise from `40.742` to `52.458`, present tense
rises from `45.773` to `51.300`, and prepositions rise from `88.294` to
`96.815`.

This is not well described as "alignment adds hedging." Hedges fall from
`0.742` in Dolma to `0.057` in SFT; amplifiers fall from `2.292` to `0.416`;
and emphatics fall from `8.843` to `1.490`. The broad movement is therefore
toward polished, nominalized, answer-like exposition rather than toward more
hedged or intensified language in aggregate.

Dolci DPO chosen and rejected responses differ from each other, but not in a
way that makes chosen uniformly more slop-like. The chosen side is more
descriptive and expository on the pybiber layer: compared with rejected
responses, chosen responses have more broad nouns (`421.637` vs. `409.725`),
attributive adjectives (`63.965` vs. `56.861`), adverbs (`42.427` vs.
`37.954`), and prepositions (`94.630` vs. `91.819`). The rejected side is more
personal, narrative, and mental-state framed: it has more first-person pronouns
(`11.934` vs. `9.223`), third-person pronouns (`8.671` vs. `6.138`), past
tense (`8.805` vs. `6.421`), and private verbs (`13.088` vs. `11.109`).

The substantive pybiber result is therefore a broad register result, not
a new single slop score. Post-training data are shifted toward a generic
assistant-answer voice, while narrower lexical and rhetorical markers vary on
top of that voice. This matters because a feature such as `slop_lexicon` can
be amplified locally even when the broader register evidence is not simply
"more hedging" or "chosen responses are always more slop-like."

Figure 1 shows the selected-feature pybiber register shift. Table 2 gives the
corresponding token-weighted means, with selected-feature intervals available
for table notes or appendix uncertainty reporting.

Evidence artifacts:

- `docs/experiments/phase1_pybiber_register_analysis.md`
- `artifacts/stage1/census/phase1_pybiber_register_means.csv`
- `artifacts/stage1/census/phase1_pybiber_register_deltas.csv`
- `artifacts/stage1/census/phase1_pybiber_register_intervals.csv`

## Result 2: Aggregate EQ-Bench Scores Are Useful But Insufficient

Claim IDs: C4, C5, C6.

The EQ-Bench Slop Score gives a useful benchmark-shaped diagnostic, but it
does not replace the feature-level amplification analysis. On the bounded
OLMo 512-prompt/8-completion temperature-1 panel, the score rises from base
`5.179` [95% bootstrap CI `4.689`, `5.773`] to SFT `7.580`
[`6.863`, `8.326`], then eases through DPO `7.006` [`6.460`, `7.562`] and
final `6.650` [`6.050`, `7.196`].
That trajectory differs from the strongest OLMo teacher-forced
`slop_lexicon` result, where DPO is the propensity peak.

SmolLM3 no-think shows a different aggregate score path. Its score rises from
base `2.798` [`2.528`, `3.063`] to SFT `7.169` [`6.657`, `7.635`], then
remains high at DPO/APO `8.345` [`7.815`, `8.942`] and final `8.254`
[`7.746`, `8.812`]. The DPO and final intervals overlap, so the safe claim is
that both post-preference cells remain high. The score components show that
most of this movement is driven by EQ-Bench slop-list word density rather than
by contrastive-negation density.

The corpus reference points make the same caution visible. OLMo Dolci DPO
chosen and rejected data are both high under EQ-Bench scoring, with rejected
slightly higher than chosen (`12.766` vs. `12.318`). Thus this aggregate slice
does not support a simple chosen-over-rejected preference-selection story for
slop. SmolLM3 sampled pretraining and no-think SFT corpus references are near
`3-4`, while SFT/DPO/final generations are near `7-8`, pointing to
generation-time amplification relative to those sampled references.

EQ-Bench is valuable for public comparability and for capturing a familiar
lexical slop signature. It is presented as an aggregate diagnostic, not as the
causal measurement layer.

Figure 2 shows the OLMo and SmolLM3 EQ-Bench score trajectories with
output-bootstrap intervals. Its caption frames the scorecard as descriptive
and as an aggregate benchmark bridge.

Evidence artifacts:

- `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_stage_comparison.md`
- `artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.md`

## Result 3: OLMo Slop-Lexicon Propensity Peaks At DPO

Claim IDs: C7, C8, C9.

The cleanest OLMo feature-level model-side result is the original
`slop_lexicon`. In the bounded teacher-forced spectrum, its normalized AF
rises from base `1.467` and SFT `1.695` to DPO `1.999`, then falls at
final/RLVR to `1.659`. This local-propensity peak is stage-specific; it should
not be described as a monotonic "later checkpoint equals more slop" curve.

The DPO peak is not directly explained by measured Dolci chosen-response
complicity. The paired Dolci chosen-vs-rejected evidence for `slop_lexicon`
does not show chosen-greater-rejected support: the aggregate chosen-minus-
rejected direction is negative and the BH-adjusted q-value is `0.963`. Under
the analysis taxonomy, this supports a dynamics-driven label for this OLMo
feature: the DPO-stage propensity peak is not a simple imitation of higher
slop rates in chosen preference responses.

Free-running output gives a related but not identical view. In the OLMo
reduced long-output panel, `slop_lexicon` rates peak at DPO: base `0.278`,
SFT `0.317`, DPO `0.381`, and final/RLVR `0.366` hits per 1,000 generated
tokens. But the temperature-dependent realized-AF table does not make DPO the
largest point everywhere; at temperature `1.0`, SFT has the largest realized
AF (`1.359`) while DPO is `1.107`. This is the core methodological reason to
separate fixed-context propensity from sampled-output rates.

Compounding is visible for the same feature. Observed minus teacher-forced
expected rates are positive at every OLMo stage in the reduced long-output
panel: base `0.214`, SFT `0.223`, DPO `0.261`, and final/RLVR `0.198` excess
hits per 1,000 opportunities. This means that once the model has begun using
the lexicon, later output is more likely to continue in the same style than
a simple independent local-propensity model would predict.

The strongest OLMo slop-lexicon result is a preference-stage local-propensity
peak plus generation-side self-conditioning, not a general claim that DPO
worsens every slop marker.

Figure 3 shows the OLMo `slop_lexicon` teacher-forced, generated-rate, and
compounding views. It is framed as a headline feature view rather than the
full amplification spectrum.

Evidence artifacts:

- `docs/experiments/phase3_integrated_conclusion_report.md`
- `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv`
- `artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_feature_classification_t07_long1024.csv`

## Result 4: SmolLM3 Replicates Style Pressure With Different Stage Localization

Claim ID: C10.

SmolLM3 no-think provides replication pressure but not identical stage
localization. In the production `t=1.0` panel, `slop_lexicon` generated rates
rise from base `0.118` to SFT `0.310`, DPO/APO `0.379`, and final `0.390`
hits per 1,000 generated tokens. The no-think EQ-Bench score shows the same
general movement from low base to high post-SFT scores.

The OLMo-vs-SmolLM3 comparison aligns 24 feature-stage rows, with 8 shared
non-missing AF values. The AF rank correlation is positive: Spearman `0.762`
and Pearson `0.978`. Because the overlap is small and the feature-stage rows
are not fully independent, this is suggestive cross-ladder pressure rather
than robust universal replication.

The main substantive point is that the ladders agree on style pressure while
disagreeing on exact stage localization. OLMo classifies `slop_lexicon` as
preference-amplified and dynamics-driven in the bounded spectrum. SmolLM3
classifies the same feature as SFT-amplified under the no-think production
scope. That difference is plausible given the different scale, training
recipe, preference algorithm, and final checkpoint construction.

Figure 4 shows the matched OLMo-vs-SmolLM3 AF comparison, with the
small-overlap caveat kept visible.

Evidence artifacts:

- `docs/experiments/phase3_integrated_conclusion_report.md`
- `docs/experiments/phase3_dpo_vs_apo_variation_report.md`
- `artifacts/phase3/analysis/smollm3_no_think_amplification_spectrum_512prompt_tf_generation_compounding_baselines_data_rates_slop_neutral_rule3_production.csv`

## Result 5: Detector Discovery Produces Candidate Tier-3 Style Families

Claim IDs: C11, C12, C13.

Phase 4 demonstrates that detector-driven discovery can find new candidate
style families beyond the original Tier-1 seed set. The combined
ModernBERT-large detector was attributed over 10,000 generated documents,
producing 77,013 document-span rows and 52,109 unique spans. The attribution
spans were embedded and clustered, then converted into 10 provisional matcher
specs.

The detector pipeline is real, but its human-facing interpretation is
provisional. The detector transfers well to SmolLM3 generations but poorly to
OLMo generations, and many OLMo SFT references are detector-positive. That
means the clusters are candidate machine-detectable style families until
manual perceptibility labels and matcher precision checks promote them.

The 512-document exact Tier-3 OLMo rerun confirms that some candidates are not
merely free-generation counting artifacts. Process framing has the most stable
denominator-supported signal: 126 reference initiations, raw AF above one at
every stage, and raw AF rising from base `1.401` to final `1.957`. Additive
transitions have stronger denominator support, with 218 reference initiations,
and move from below reference rate at base (`0.565`) to near reference-matched
by final (`1.012`). Prescriptive instruction has 74 reference initiations and
rises from base `0.301` to final `0.763`.

Follow-up offers are the largest Tier-3 signal but remain sparse: 11 reference
initiations and raw AF jumping from base `6.171` to about `39` after SFT. This
is a strong provisional signal, not a standalone headline claim. Response
constraint similarly rises from base `0.740` to final `2.017`, but has only
11 reference initiations. Careful reasoning has only one reference initiation,
so its very large AF values are not stable rate estimates.

Detector-derived features include real fixed-context propensity shifts,
especially process framing and selected instruction/transition patterns.
Promotion into the main slop taxonomy awaits manual human-perceptibility and
precision validation.

Figure 5 shows the Tier-3 exact teacher-forced AF rerun with denominator
support. Sparse features remain visually and textually marked as provisional.

Evidence artifacts:

- `docs/experiments/phase4_detector_discovery_report.md`
- `docs/experiments/phase4_tier3_teacher_forced_addendum.md`
- `artifacts/phase4/modernbert_detector_combined_v2_clean/`

## Result 6: The Current Paper Claim Is Bounded But Coherent

The combined evidence supports a narrower and stronger claim than the original
simple story. Post-training data shift sharply away from narrative/personal
prose toward nominalized, adjectival answer-register exposition. The original
slop lexicon shows a clean OLMo DPO-stage propensity peak that is not directly
explained by chosen-response enrichment. Free-running generation adds
self-conditioning. SmolLM3 shows related style pressure with different stage
localization. Detector attribution expands the feature set, but the new
families remain candidates until validation.

The paper should not claim that DPO universally creates slop, that preference
data are uniformly more slop-like, or that a single aggregate score captures
the phenomenon. The safe thesis is that "slop style" is a spectrum of
measurable lexical, rhetorical, and register features whose sources differ by
feature, ladder, and measurement mode.

Table 3 summarizes claim-strength bands, and Table 5 summarizes paper-safe
negative claims when the venue allows explicit scope-control tables.
