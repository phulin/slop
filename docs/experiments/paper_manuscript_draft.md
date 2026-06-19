# Localizing "Slop Style" In Open Post-Training Ladders

## Abstract

Large language model outputs often share recognizable stylistic markers, but
raw output counts cannot distinguish inherited training-data style from
model-side amplification. We study this problem in open model/data ladders
where pretraining references, supervised fine-tuning data, preference pairs,
and checkpoints can be compared directly. We combine corpus censuses, full
pybiber register measurement, teacher-forced propensity, free-running
generation, compounding tests, EQ-Bench Slop Score diagnostics, and
detector-driven feature discovery. In OLMo/Dolci data, post-training corpora
move away from narrative and personal web prose toward a more noun-heavy,
descriptive assistant-answer register; this is not simply an aggregate
increase in hedging. In the OLMo
checkpoint ladder, the original slop lexicon shows the cleanest preference-stage
propensity peak, but the corresponding preference data do not show
chosen-response enrichment, pointing away from a simple imitation account.
SmolLM3 no-think shows related style pressure with different stage
localization. Detector attribution identifies candidate Tier-3 style families,
several of which have real teacher-forced sequence-mass support, but human
perceptibility validation remains open. The result is an amplification
spectrum: slop style is feature-specific, stage-specific, and partly a
register shift rather than a single global score.

## 1. Introduction

Large language model outputs often share recognizable stylistic habits: stock
phrases, inflated lexical choices, repetitive rhetorical structures, and a
polished assistant voice that can feel detached from the local task. These
features are often grouped under informal labels such as "slop", but the label
hides several different mechanisms. A phrase can appear because it was common
in pretraining data, because supervised fine-tuning taught an assistant
register, because preference optimization changed local propensities, or
because generation self-conditions after the first stylistic marker appears.

This paper treats slop style as a measurement problem. Rather than asking
whether a whole answer is slop, we measure lexical, rhetorical, and register
features across open data and model ladders. The goal is to localize where a
feature enters the training-and-generation process and how it changes across stages. We use open
post-training ladders because they expose the ingredients needed for that
localization: pretraining references, SFT targets, preference pairs, and model
checkpoints before and after post-training steps [@team_olmo_2025_olmo3;
@allenai_olmo3_sft_card; @allenai_olmo3_dpo_card;
@allenai_dolci_sft_card; @allenai_dolci_dpo_card;
@huggingface_2025_smollm3_blog; @huggingface_smollm3_card].

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
toward nominalized, adjectival, answer-like exposition: a more noun-heavy,
descriptive assistant-answer register. This broad register movement is not
simply an increase in hedging or intensification. On the model
side, the original slop lexicon shows the clearest OLMo preference-stage
teacher-forced propensity peak, but the corresponding Dolci preference data do
not show chosen-response enrichment for that feature. In sampled output, the
same feature also shows self-conditioning. SmolLM3 no-think exhibits related
style pressure but localizes the main lexical movement earlier, around SFT and
post-SFT stages.

We also show that aggregate slop scores and detector features are useful but
must be interpreted separately from the causal measurement layers. The
EQ-Bench Slop Score provides a public benchmark bridge [@eqbench_slop_score],
but its stage path does not always match the feature-level propensity story.
Detector attribution finds plausible
Tier-3 style families, and selected candidates have real exact sequence-mass
support, but those families remain provisional until human perceptibility and
matcher precision validation are complete.

### Contributions

1. A measurement design that separates corpus inheritance, fixed-context model
   propensity, and free-running self-conditioning for stylistic features.
2. A full pybiber register read over English-filtered OLMo Phase 1 samples,
   showing that alignment data shift toward noun-heavy, descriptive
   assistant-answer prose rather than simply increasing hedges or intensifiers.
3. A bounded OLMo and SmolLM3 amplification spectrum showing that
   `slop_lexicon` amplifies in both ladders but localizes to different stages.
4. An EQ-Bench Slop Score bridge that supports benchmark comparison while
   keeping aggregate scores separate from causal propensity claims.
5. A detector-driven Tier-3 discovery pass that turns attribution clusters
   into provisional features and validates selected candidates with exact
   teacher-forced sequence mass.

## 2. Related Work

This work sits between style measurement, open post-training analysis, and
feature discovery. AI-generated text detection is often framed as deciding
whether a passage was produced by a model, with detectors spanning
watermarking, statistical, neural, and human-assisted methods
[@wu_2023_llm_text_detection_survey]. Prior human/AI parallel-corpus work shows
that generated text differs from human writing in grammatical and rhetorical
style, and provides a useful basis for separating broad register movement from
narrow lexical tics [@reinhart_2024_hape_style; @brown_hape_dataset_card].
Recent work also proposes an explicit human-facing taxonomy for AI "slop" in
text [@shaib_2025_slop]. We use that framing cautiously: this paper does not
treat "AI-written" or "slop" as a single class. Instead, it asks where
individual style markers enter or amplify across data and checkpoint stages.

Register analysis is the second background thread. Biber-style
multidimensional register analysis is a natural fit for distinguishing
narrative/personal prose from nominalized expository prose
[@biber_1988_variation]. The Phase 1 pybiber layer uses a Biber-style feature
inventory via the `pybiber` package [@pybiber_pypi]. The role of this layer is
descriptive and diagnostic: it shows whether post-training corpora move across
broad register dimensions. The substantive register claims in this paper come
from the Phase 1 measurements, while the literature and package citations
establish method and implementation provenance.

Preference optimization is the third background thread. Direct Preference Optimization
provides the general method context for the OLMo DPO checkpoint
[@rafailov_2023_dpo], and Anchored Preference Optimization provides method
context for the SmolLM3-side preference-stage discussion
[@doosterlinck_2024_apo]. Preference and evaluator pipelines can also carry
style-related biases such as verbosity preference [@saito_2023_verbosity_bias].
Our claims are therefore not generic claims about DPO, APO, or preference
optimization as a class. The OLMo/Dolci evidence is tied to the released OLMo
3 and Dolci artifacts [@team_olmo_2025_olmo3; @allenai_dolci_dpo_card], while
SmolLM3 is used as a different open ladder with different training and
final-checkpoint caveats [@huggingface_2025_smollm3_blog;
@huggingface_smollm3_card].

Two auxiliary tooling layers should not be confused with the main estimand.
EQ-Bench Slop Score is a public aggregate diagnostic and lexicon source
[@eqbench_slop_score], not the causal measurement layer. The
detector-discovery path uses ModernBERT, integrated gradients, GTE-large
embedding, and HDBSCAN clustering to propose candidate Tier-3 features
[@warner_2024_modernbert; @answerai_modernbert_card;
@sundararajan_2017_integrated_gradients; @alibaba_gte_large_en_v15_card;
@campello_2013_hdbscan; @mcinnes_2017_hdbscan]. Those candidates require
human perceptibility labels and matcher precision validation before they
become paper-grade slop features.

Generation infrastructure is a methods detail rather than a substantive
claim: SGLang is used for the bounded pure-generation panels because it makes
the reduced production scope practical under fixed compute constraints
[@zheng_2023_sglang].

The compounding part of the paper is related to prior work on generation-time
degeneration and repetition. Nucleus-sampling and unlikelihood-training work
show that decoding and likelihood objectives can produce repetitive or bland
text even when the underlying language model is otherwise useful
[@holtzman_2019_neural_text_degeneration; @welleck_2019_unlikelihood]. Our
measurement is narrower: it tests whether already-emitted slop markers predict
later markers within the same sampled output.

## 3. Methods

### 3.1 Study Design

The study measures slop style as a set of observable stylistic markers and
register features, not as factual error or answer quality. The central
measurement problem is source attribution: a style marker can be inherited from
pretraining data, introduced by supervised fine-tuning data, selected by
preference data, amplified by model-side optimization, or increased during
free-running generation after an earlier hit changes the local context.

We separate four localization layers:

1. Corpus inheritance: feature rates in pretraining, SFT, and preference data.
2. Fixed-context propensity: teacher-forced probability of initiating a feature
   in a held-out reference context.
3. Free-running emission: realized feature rates in sampled generations.
4. Self-conditioning: whether later feature hits are more likely after a
   generation has already produced an earlier hit.

Two additional measurement surfaces are kept separate from those localization
layers. The EQ-Bench Slop Score is an aggregate public benchmark bridge, not a
causal estimand. Detector-derived Tier-3 features are a discovery surface for
candidate style families, not human-perceived slop claims until manual labels
and matcher precision checks promote them.

The main estimand is an amplification spectrum: for each feature, we compare its
data-side frequency with model-side propensity and generated-output behavior
across checkpoints. The completed experiment set is bounded; it is not the
abandoned exhaustive 5,000 prompt x 8 completion x 3 temperature OLMo grid.

### 3.2 Model And Data Ladders

The primary ladder is OLMo 3 7B Instruct [@team_olmo_2025_olmo3]: base
`allenai/Olmo-3-1025-7B`, SFT `allenai/Olmo-3-7B-Instruct-SFT`, DPO
`allenai/Olmo-3-7B-Instruct-DPO`, and final/RLVR
`allenai/Olmo-3-7B-Instruct` [@allenai_olmo3_sft_card;
@allenai_olmo3_dpo_card].

The Phase 1 corpus package uses English-filtered retained samples:

- Dolma 3 pretraining reference: 5,740 retained documents from a bounded 80k
  scan, 8,689,472 tokens.
- Dolci SFT: 40,000 retained target responses, 8,279,115 tokens.
- Dolci DPO: 40,000 retained preference pairs, expanded to 80,000 chosen and
  rejected rows, 27,448,016 tokens.

Dolci SFT and DPO source provenance is from the official dataset cards; the
sample sizes above are measured after English filtering and retention
[@allenai_dolci_sft_card; @allenai_dolci_dpo_card].

Language filtering uses source language metadata when available and Lingua
language detection otherwise. The English filter is part of the sample
definition for the retained Phase 1 sample.

The replication ladder is SmolLM3-3B in no-think mode
[@huggingface_2025_smollm3_blog; @huggingface_smollm3_card]. It differs from
OLMo in scale, organization, data mixture, and preference algorithm. Its
preference cell is APO rather than OLMo's DPO, and the final model involves
checkpoint averaging plus long-context merging. For that reason, SmolLM3 is
taken as replication pressure on broad style amplification, not as a clean
one-to-one stage attribution match to OLMo.

Table 1 summarizes the bounded data/model scope used throughout the paper.

### 3.3 Feature Surface

Tier 1 contains seed slop markers: `contrastive_negation`, the frozen
historical `slop_lexicon`, `rule_of_three_approx`, `stock_openers`,
`stock_closers`, and the derived pooled `stock_openers_closers` view.
Formatting-only features such as list/header/bold lead-ins are retired from
the active feature surface. Punctuation/rhythm and generic hedging are
exploratory rather than core Phase 1 claims.

Manual precision validation is scoped. In the validation table,
`contrastive_negation`, `slop_lexicon`, `stock_openers`, and
`stock_closers` pass the interim span/item precision gate, while
`rule_of_three_approx` is demoted for publication-grade independent regex
claims after 50 labels showed 0.560 precision. The `slop_lexicon` row remains
a pooled historical index whose broad items need item-level interpretation.
Table 4 reports the validation counts. The bounded direct queue for
`stock_openers_closers` produced zero retained hits, so the pooled feature is
treated as a derived convenience metric rather than an independent corpus-rate
claim.
This validation boundary applies to publication-grade corpus-rate claims. The
paper's `slop_lexicon` result is framed separately as a frozen historical
lexicon index in the model-side propensity and generation analyses.

Tier 2 uses full `pybiber` extraction for Phase 1 corpus-side analysis
[@pybiber_pypi]. Pybiber completed with full retained-row coverage on the
English-filtered samples:
Dolma 5,740 returned rows x 67 features, Dolci SFT 40,000 returned rows x 67
features, and Dolci DPO 80,000 returned rows x 67 features. Generated-output
full pybiber was not run, so generated-output register claims are excluded
from the main evidence package.

Tier 3 features come from the Phase 4 detector-discovery pass. A
ModernBERT-large detector was trained on HAP-E and OLMo/SmolLM3 generation/reference
slices, attributed with integrated gradients over 10,000 generated documents,
and clustered after GTE-large embedding. Cluster summaries were converted into
10 provisional matcher specifications [@reinhart_2024_hape_style;
@brown_hape_dataset_card; @warner_2024_modernbert;
@answerai_modernbert_card; @sundararajan_2017_integrated_gradients;
@alibaba_gte_large_en_v15_card; @campello_2013_hdbscan;
@mcinnes_2017_hdbscan]. These features are candidates until human
perceptibility labels and matcher precision validation are complete; the
Shaib et al. taxonomy provides the external framing for that
annotation layer [@shaib_2025_slop].

Table 4 summarizes the Tier-1 precision-validation status and the
claim boundaries for demoted, derived, and exploratory rows.

### 3.4 Corpus Rates, EQ-Bench, Propensity, And Generation

Corpus rates are computed over fixed retained samples. Tier-1 rates are
reported per 1,000 simple tokens and, where the feature contract supports it,
per opportunity. DPO pair deltas are measured within retained preference pairs,
which controls prompt/topic by construction. Dolci DPO chosen/rejected
differences should not be interpreted as pure human-preference effects because
the dataset includes synthetic and Delta Learning-style construction regimes.

The EQ-Bench Slop Score is implemented as an exploratory benchmark bridge, not
as the paper's causal outcome variable. Outputs are filtered, concatenated
into the EQ-Bench aggregate shape, and scored as a document or panel-level
0-100 diagnostic [@eqbench_slop_score].

Teacher-forced propensity estimates local model preference for initiating a
feature in a fixed reference context. For each feature, the harness defines an
opportunity set, an initiating token set or exact candidate sequence, and a
reference indicator. For each checkpoint and feature, the model is evaluated
under teacher forcing on held-out reference text. The basic amplification
factor is:

`AF = mean model initiation probability / mean reference initiation indicator`.

Neutral-normalized AF divides a feature AF by the AF of matched neutral control
phrases for the same stage. This reduces, but does not eliminate, the risk
that a broad calibration shift is mistaken for a slop-specific shift;
calibration remains a general concern when interpreting neural probability
estimates [@guo_2017_calibration].

Generation uses SGLang for pure generation tasks [@zheng_2023_sglang]. The
bounded generation scope is:

- OLMo reduced main style panel: 8,192 generations.
- OLMo reduced temperature panel: 6,144 generations.
- OLMo reduced long-output compounding panel: 2,048 generations.
- SmolLM3 no-think production-shape panel: 49,152 generations.
- OLMo Instruct/Think/RL-Zero stretch panel: 5,120 final-answer-mode
  generations.

Compounding tests whether style markers self-condition during generation,
using expected-versus-observed comparisons and prior-hit windows. The
strongest compounding result in the retained feature set is for
`slop_lexicon`.

### 3.5 Statistics And Scope

The analysis uses paired designs wherever possible: DPO corpus deltas compare
chosen and rejected responses within the same preference pair; teacher-forced
stage effects use paired opportunity-level tests where the opportunity
contract supports pairing; free-running stage effects use paired prompt or
prompt-cluster units when comparing adjacent stages. Benjamini-Hochberg FDR
correction is applied within the relevant feature set for multi-feature
stage-effect summaries. Bootstrap intervals are clustered by document, prompt,
or prompt cluster depending on the measurement layer. Sparse denominators are
reported explicitly; a large AF with few reference initiations is treated as
provisional even if the point estimate is large.

## 4. Results

### 4.1 Post-Training Data Shift Toward Answer-Register Prose

The clearest corpus-side result is a broad register shift from pretraining text
to post-training data. In the English-filtered OLMo/Dolci samples, SFT targets
are much less narrative and personal than the Dolma pretraining reference.
In plain terms, they contain far fewer signals of someone recounting events,
describing personal experience, or tracking third-person actors. Past tense
falls from `57.065` [`55.212`, `59.078`] to `4.315` [`4.195`, `4.460`];
first-person pronouns fall from `44.732` [`42.971`, `46.451`] to `9.360`
[`9.086`, `9.792`]; third-person pronouns fall from `51.809` [`50.260`,
`53.734`] to `4.154` [`4.022`, `4.297`]; and clausal coordination falls from
`12.089` to `3.004`. Document-bootstrap intervals over retained rows are
narrow relative to these large shifts.

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
way that makes chosen uniformly more slop-like. On this register layer, chosen
responses are more descriptive/expository: compared with rejected responses,
chosen responses have more broad nouns (`421.637` vs. `409.725`), attributive
adjectives (`63.965` vs. `56.861`), adverbs (`42.427` vs. `37.954`), and
prepositions (`94.630` vs. `91.819`). The rejected side is more
personal, narrative, and mental-state framed: it has more first-person pronouns
(`11.934` vs. `9.223`), third-person pronouns (`8.671` vs. `6.138`), past
tense (`8.805` vs. `6.421`), and private verbs (`13.088` vs. `11.109`).

The substantive pybiber result is therefore a broad register result, not
a new single slop score. Post-training data are shifted toward a generic
assistant-answer voice, while narrower lexical and rhetorical markers vary on
top of that voice. This matters because a feature such as `slop_lexicon` can
be amplified locally even when the broader register evidence is not simply
"more hedging" or "chosen responses are always more slop-like."

Figure 1 visualizes the selected pybiber register shift. Table 2 gives the
corresponding token-weighted means, with selected-feature intervals reported
in the reproducibility appendix.

### 4.2 Aggregate EQ-Bench Scores Are Useful But Insufficient

The EQ-Bench Slop Score gives a useful benchmark-shaped diagnostic, but it
does not replace the feature-level amplification analysis. On the bounded OLMo
512-prompt/8-completion temperature-1 panel, the score rises from base `5.179`
[95% bootstrap CI `4.689`, `5.773`] to SFT `7.580` [`6.863`, `8.326`], then
eases through DPO `7.006` [`6.460`, `7.562`] and final `6.650` [`6.050`,
`7.196`]. That
trajectory differs from the strongest OLMo teacher-forced `slop_lexicon`
result, where DPO is the propensity peak.

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

Figure 2 plots the EQ-Bench score by checkpoint for the bounded OLMo and
SmolLM3 panels, with output-bootstrap intervals over valid generated outputs.

### 4.3 OLMo Slop-Lexicon Propensity Peaks At DPO

The cleanest OLMo feature-level model-side result is the original
`slop_lexicon`. In the bounded teacher-forced spectrum, its normalized AF
rises from base `1.467` and SFT `1.695` to DPO `1.999`, then falls at
final/RLVR to `1.659`. This local-propensity peak is stage-specific; it should
not be described as a monotonic "later checkpoint equals more slop" curve.

The DPO peak is not directly explained by measured Dolci chosen-response
complicity. The paired Dolci chosen-vs-rejected evidence for `slop_lexicon`
does not show chosen-greater-rejected support: the aggregate chosen-minus-
rejected direction is negative and the BH-adjusted q-value is `0.963`. Under
this taxonomy, this supports a dynamics-driven label for this OLMo
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
the lexicon, later output is more likely to continue in the same style than a
simple independent local-propensity model would predict.

Figure 3 shows the OLMo `slop_lexicon` teacher-forced, generated-output, and
compounding views.

### 4.4 SmolLM3 Replicates Style Pressure With Different Stage Localization

SmolLM3 no-think provides replication pressure but not identical stage
localization. In the production `t=1.0` panel, `slop_lexicon` generated rates
rise from base `0.118` to SFT `0.310`, DPO/APO `0.379`, and final `0.390`
hits per 1,000 generated tokens. The no-think EQ-Bench score shows the same
general movement from low base to high post-SFT scores.

The OLMo-vs-SmolLM3 comparison aligns 24 feature-stage rows, with 8 shared
non-missing AF values. The AF rank correlation is positive: Spearman `0.762`
and Pearson `0.978`. Because the overlap is small and the feature-stage rows
are not fully independent, we interpret this as suggestive cross-ladder
pressure rather than robust universal replication.

The main substantive point is that the ladders agree on style pressure while
disagreeing on exact stage localization. OLMo classifies `slop_lexicon` as
preference-amplified and dynamics-driven in the bounded spectrum. SmolLM3
classifies the same feature as SFT-amplified under the no-think production
scope.

Figure 4 shows the matched OLMo-vs-SmolLM3 AF comparison.

### 4.5 Detector Discovery Produces Candidate Tier-3 Style Families

Phase 4 demonstrates that detector-driven discovery can find new candidate
style families beyond the original Tier-1 seed set. The combined
ModernBERT-large detector was attributed over 10,000 generated documents,
producing 77,013 document-span rows and 52,109 unique spans. The attribution
spans were embedded and clustered, then converted into 10 provisional matcher
specs.

The detector run is real, but its human-facing interpretation is
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
constraint similarly rises from base `0.740` to final `2.017`, but has only 11
reference initiations. Careful reasoning has only one reference initiation, so
its very large AF values are not stable rate estimates.

Figure 5 shows the Tier-3 exact teacher-forced AF rerun with denominator
support.

## 5. Discussion

The central lesson is that slop style is not one object. It is a family of
features that move differently across data, optimization, and decoding.
Treating it as a single scalar can be useful for benchmarking, but it obscures the
mechanisms that matter for diagnosis. The same model family can have a broad
answer-register shift in its data, a feature-specific DPO propensity peak, and
a different free-running output maximum. Those are not contradictions; they
are different views of the same system.

The full pybiber layer is important because it changes the baseline story. If
we only counted familiar lexical markers, it would be tempting to describe the
phenomenon as "more hedging" or "more ornate wording." The corpus-side register
evidence says something broader: the SFT data are much less narrative and
personal than the Dolma sample and much more nominalized and expository.
Hedges, amplifiers, and emphatics are lower in SFT than in Dolma. The assistant
style we observe is therefore not just a collection of lexical tics. It is a
shift toward a particular assistant-answer prose, with some conspicuous markers
layered on top.

The OLMo `slop_lexicon` result gives the cleanest example of why fixed-context
propensity is needed. The feature has a DPO-stage teacher-forced peak, yet the
measured Dolci chosen/rejected contrast does not show chosen responses as more
lexicon-heavy than rejected responses. That does not prove a specific
optimizer mechanism by itself. It does, however, rule out the simplest
imitation story for this feature under the measured data contrast.

The SmolLM3 comparison sharpens the same point. SmolLM3 no-think shows related
style pressure, including high post-SFT EQ-Bench scores and generated
`slop_lexicon` rates, but its stage localization differs from OLMo. That is
not a failed replication. It is evidence that the phenomenon is feature-like
and pipeline-dependent rather than a universal DPO fingerprint.

The detector-discovery work expands the feature surface while also showing why
validation cannot be skipped. The detector and attribution workflow can produce
plausible style families such as process framing, follow-up offers, additive
transitions, and prescriptive instruction. The exact Tier-3 rerun confirms that
some of those families correspond to model propensity shifts rather than only
to free-generation counts. But detector-positive spans are not automatically
human-perceived slop, and sparse reference denominators can make very large AF
values unstable.

A practical implication is that mitigation and evaluation should be
feature-specific. If a style marker is inherited from SFT data, data curation
or target rewriting may be the right intervention. If it is amplified by local
propensity at a preference checkpoint, preference objectives or calibration may
matter more. If it compounds only after an earlier hit, decoding and
self-conditioning controls become relevant. A single aggregate score cannot
distinguish those cases.

## 6. Limitations

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
is a derived convenience metric in this analysis; the pooled lexicon still
requires item- and context-level interpretation.

Third, several teacher-forced features have sparse reference denominators.
Large AF point estimates for features such as follow-up offers, response
constraints, careful reasoning, and some stock closer variants should be
treated as provisional until denominator support improves.

Fourth, Dolci DPO chosen/rejected contrasts are not pure human-preference
contrasts. They include synthetic and Delta Learning-style construction
regimes, so chosen-vs-rejected differences can reflect strong-model versus
weak-model style that is distilled into OLMo.

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

Finally, the completed production scope is intentionally smaller than the
original exhaustive generation plan. The reduced SGLang panels answer the
paper's questions about register shift, local propensity, free-running
emission, and compounding. They are not the same evidence as a
completed 5,000 prompt x 8 completion x 3 temperature OLMo grid.

## 7. Conclusion

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

## Figure And Table Captions

**Figure 1. Phase 1 register shift in selected pybiber features.**
Token-weighted pybiber rates for retained English-filtered Dolma, Dolci SFT,
and Dolci DPO chosen/rejected samples. Post-training data move sharply away
from narrative/personal prose markers such as past tense and personal
pronouns, and toward nominalized, adjectival, answer-register exposition. This
is a corpus-side register result; generated-output full pybiber is not claimed
here. Error bars show nonparametric document-bootstrap intervals over retained
rows for the plotted features.

**Figure 2. EQ-Bench Slop Score by checkpoint.**
Aggregate EQ-Bench Slop Score for bounded OLMo and SmolLM3 generation panels.
The score provides a public benchmark bridge and visible lexical slop
diagnostic, but it is not the causal measurement layer. Error bars show
nonparametric 95% bootstrap intervals over valid generated outputs.

**Figure 3. OLMo `slop_lexicon` amplification views.**
Teacher-forced normalized amplification factor, generated-output rate, and
compounding evidence for the original `slop_lexicon` in the bounded OLMo
ladder. The teacher-forced propensity peak occurs at DPO, while the measured
Dolci chosen/rejected contrast does not show chosen-response enrichment for
the same feature.

**Figure 4. Cross-ladder AF comparison.**
Matched OLMo and SmolLM3 amplification-factor coordinates where both ladders
have non-missing support. The positive rank correlation is suggestive evidence
for related style pressure, but the small overlap and different checkpoint
recipes make this a replication-pressure result rather than a universal
stage-localization claim.

**Figure 5. Phase 4 Tier-3 exact AF with denominators.**
Exact sequence-mass teacher-forced amplification factors for
detector-discovered candidate features over the 512-reference OLMo rerun.
Denominator labels are part of the figure because sparse candidates with large
point estimates remain provisional until human perceptibility and precision
validation are complete.

**Table 1. Data and measurement scope.**
Bounded corpus, generation, detector, and teacher-forced rerun sizes used in
the study. The scope is the reduced production estimand, not the originally
proposed exhaustive OLMo generation grid.

**Table 2. Selected full-pybiber register means.**
Token-weighted means for selected pybiber features across retained Dolma,
Dolci SFT, and Dolci DPO chosen/rejected samples. The selected features
summarize the main register movement; the full 67-feature outputs are reported
in the reproducibility appendix, along with selected-feature bootstrap
intervals.

**Table 3. Claim strength bands.**
Paper claim IDs grouped by evidence strength and required caveat language.
This table can appear in the appendix when the venue allows explicit
claim-strength controls.

**Table 4. Tier-1 precision-validation status.**
Manual-validation state for the Tier-1 regex features and exploratory addenda.
Passing, demoted, derived, and unlabeled rows define the paper's claim
boundaries rather than serving as headline empirical results.

**Table 5. Paper-safe negative claims.**
Claims the paper should avoid and the bounded replacement wording supported by
the evidence. This table functions as an appendix scope-control table.

## Reproducibility Appendices

Appendix A records claim and scope controls, including the claim matrix and
claim-to-evidence map used to keep caveats colocated with claims. Appendix B
reports Phase 1 corpus and pybiber materials, including retained sample sizes,
full 67-feature outputs, token-weighted register summaries, and selected-
feature bootstrap intervals. Appendix C reports Tier-1 precision-validation
queues and labels. Appendix D documents the EQ-Bench Slop Score bridge and
scorecard. Appendix E records the Phase 3 amplification spectrum and SGLang
generation panels. Appendix F records the Phase 4 detector-discovery outputs,
annotation package, and Tier-3 exact teacher-forced rerun. Appendix G contains
figure/table provenance and reference-source metadata.
