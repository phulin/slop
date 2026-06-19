# Paper Methods Draft

This Methods draft describes the bounded experiment package in manuscript
form. It is aligned with the integrated manuscript and the paper claim matrix:
`docs/experiments/paper_manuscript_draft.md` and
`docs/experiments/paper_claim_matrix.md`. Source provenance for cited external
work is tracked in `docs/experiments/paper_reference_sources.md`.

## Study Design

The study measures "slop style" as a set of observable stylistic markers and
register features, not as factual error or answer quality. The central
measurement problem is source attribution: a style marker can be inherited from
pretraining data, introduced by supervised fine-tuning data, selected by
preference data, amplified by model-side optimization, or increased during
free-running generation after an earlier hit changes the local context.

We therefore separate four localization layers:

1. **Corpus inheritance:** feature rates in pretraining, SFT, and preference
   data.
2. **Fixed-context propensity:** teacher-forced probability of initiating a
   feature in a held-out reference context.
3. **Free-running emission:** realized feature rates in sampled generations.
4. **Self-conditioning:** whether later feature hits are more likely after a
   generation has already produced an earlier hit.

Two additional measurement surfaces are interpreted separately. The EQ-Bench
Slop Score is an aggregate public benchmark bridge, not a causal estimand.
Detector-derived Tier-3 features are a discovery surface for candidate style
families, not human-perceived slop claims until manual labels and matcher
precision checks promote them.

The main estimand is an amplification spectrum: for each feature, compare its
data-side frequency with model-side propensity and generated-output behavior
across checkpoints. The completed production scope is bounded; it is not the
abandoned exhaustive 5,000 prompt x 8 completion x 3 temperature OLMo grid.

## Model And Data Ladders

### OLMo 3 / Dolci Primary Ladder

The primary ladder is OLMo 3 7B Instruct [@team_olmo_2025_olmo3]:

| Stage | Checkpoint | Data role |
|---|---|---|
| Base | `allenai/Olmo-3-1025-7B` | Pretraining reference via Dolma 3 sample |
| SFT | `allenai/Olmo-3-7B-Instruct-SFT` | Dolci SFT targets |
| DPO | `allenai/Olmo-3-7B-Instruct-DPO` | Dolci DPO chosen/rejected pairs |
| Final | `allenai/Olmo-3-7B-Instruct` | Released RLVR/final instruct model |

The released SFT and DPO checkpoint provenance is taken from the official
model cards [@allenai_olmo3_sft_card; @allenai_olmo3_dpo_card].

The Phase 1 corpus package uses English-filtered retained samples:

- Dolma 3 pretraining reference: 5,740 retained documents from a bounded 80k
  scan, 8,689,472 tokens.
- Dolci SFT: 40,000 retained target responses, 8,279,115 tokens.
- Dolci DPO: 40,000 retained preference pairs, expanded to 80,000 chosen and
  rejected rows, 27,448,016 tokens.

Dolci SFT and DPO dataset provenance is taken from the official dataset cards;
the row and token counts above are measured from the retained sample artifacts
[@allenai_dolci_sft_card; @allenai_dolci_dpo_card].

Language filtering uses source language metadata when available and Lingua
language detection otherwise. The English filter is part of the retained
Phase 1 sample definition.

### SmolLM3 Replication Ladder

The replication ladder is SmolLM3-3B in no-think mode
[@huggingface_2025_smollm3_blog; @huggingface_smollm3_card]. It differs from
OLMo in scale, organization, data mixture, and preference algorithm. Its
preference cell is APO rather than OLMo's DPO, and the final model involves
checkpoint averaging plus long-context merging. APO is cited as method
background rather than as evidence about the exact measured checkpoint
sequence
[@doosterlinck_2024_apo]. For that reason, SmolLM3 is treated as replication
pressure on broad style amplification, not as a clean one-to-one stage
attribution match to OLMo.

All SmolLM3 generated outputs used for the production comparison are collected
in no-think mode, with explicit thinking markers audited out of the completed
production slices.

## Feature Surface

### Tier 1: Seed Slop Markers

The active Tier-1 feature surface contains:

- `contrastive_negation`: "not X, but Y" / "not just X, but Y" style
  contrastive constructions.
- `slop_lexicon`: the frozen historical lexical/trigram slop list used for
  comparability.
- `rule_of_three_approx`: a regex approximation to coordinated triples.
- `stock_openers` and `stock_closers`: stock assistant opening and closing
  phrases.
- `stock_openers_closers`: a pooled derived convenience view over openers and
  closers.

Formatting-only features such as list/header/bold lead-ins are retired from
the active feature surface. Punctuation/rhythm and generic hedging are
exploratory rather than core Phase 1 claims.

Manual precision validation is scoped. The current validation tables are
`docs/experiments/precision_validation_status.md` and
`artifacts/stage1/validation/precision_validation_status.csv`.
`contrastive_negation`, `slop_lexicon`, `stock_openers`, and
`stock_closers` pass the interim span/item precision gate, while
`rule_of_three_approx` is demoted for publication-grade independent regex
claims after 50 labels showed 0.560 precision. The `slop_lexicon` result still
requires item-level interpretation because broad entries such as
`comprehensive`, `robust`, and `journey` can be ordinary in some contexts.
The bounded direct queue for `stock_openers_closers` produced zero retained
hits, so the pooled feature is treated as a derived convenience metric rather
than an independent corpus-rate claim.

### Tier 2: Full Pybiber Register Features

The broad register layer uses full `pybiber` extraction for Phase 1
corpus-side analysis. Earlier generated-output register proxies are retired
from the paper's active feature surface. Biber-style register analysis is used
as background for interpreting broad
narrative/personal versus expository/nominalized movement
[@biber_1988_variation]. The implementation uses the `pybiber` package
[@pybiber_pypi] to produce wide per-document feature tables and long
feature-value tables for the 67 pybiber features.

Pybiber completed with full retained-row coverage on the English-filtered
Phase 1 samples:

- Dolma: 5,740 returned rows x 67 features.
- Dolci SFT: 40,000 returned rows x 67 features.
- Dolci DPO: 80,000 returned rows x 67 features.

Token-weighted means and deltas are stored in:

- `artifacts/stage1/census/phase1_pybiber_register_means.csv`
- `artifacts/stage1/census/phase1_pybiber_register_deltas.csv`
- `artifacts/stage1/census/phase1_pybiber_register_intervals.csv`
- `docs/experiments/phase1_pybiber_register_analysis.md`

Generated-output full pybiber was not run and is not used for main-paper
generated-output register claims.

### Tier 3: Detector-Discovered Candidates

Tier-3 features come from the Phase 4 detector-discovery pass. A
ModernBERT-large detector was trained on HAP-E plus OLMo/SmolLM3
generation/reference slices, attributed with integrated gradients over 10,000 generated documents,
and clustered after GTE-large embedding. The method provenance is HAP-E for
the parallel human/AI style data, ModernBERT for the detector backbone,
Integrated Gradients for attribution, GTE-large for embeddings, and HDBSCAN
for clustering [@reinhart_2024_hape_style; @brown_hape_dataset_card;
@warner_2024_modernbert; @answerai_modernbert_card;
@sundararajan_2017_integrated_gradients; @alibaba_gte_large_en_v15_card;
@campello_2013_hdbscan; @mcinnes_2017_hdbscan]. Cluster summaries were
converted into 10 provisional matcher specifications. These features are
candidates until human perceptibility labels and matcher precision validation
are complete; the planned human-perceived slop framing uses the Shaib et al.
taxonomy [@shaib_2025_slop].

The Tier-3 candidate evidence is recorded in the Phase 4 detector-discovery
artifact set under `artifacts/phase4/modernbert_detector_combined_v2_clean/`.

## Corpus Rate Measurement

Corpus rates are computed over fixed retained samples. Tier-1 rates are
reported per 1,000 simple tokens and, where the feature contract supports it,
per opportunity. DPO pair deltas are measured within retained preference pairs,
which controls prompt/topic by construction.

The Phase 1 close-out artifacts are retained for reproducibility:

- `artifacts/stage1/census/feature_rates_by_corpus.parquet`
- `artifacts/stage1/census/feature_rates_by_stratum.parquet`
- `artifacts/stage1/census/preference_pair_deltas.parquet`
- `artifacts/stage1/census/census_summary.md`

Dolci DPO chosen/rejected differences should not be interpreted as a pure
human-preference effect. Dolci includes synthetic and Delta Learning-style
construction regimes, so chosen-vs-rejected differences often measure
strong-model versus weak-model style that may be distilled into OLMo.
Preference/evaluator pipelines can also carry style biases such as verbosity
preference [@saito_2023_verbosity_bias].

## EQ-Bench Slop Score Bridge

The EQ-Bench Slop Score is implemented as an exploratory benchmark bridge, not
as the paper's causal outcome variable. The implementation uses the EQ-Bench
word and trigram lists, leaderboard normalization ranges, and portable
contrastive-negation backends. Generated outputs are filtered, concatenated
into the EQ-Bench aggregate shape, and scored as document- or panel-level
0-100 diagnostics [@eqbench_slop_score].

The bounded scorecard is:
`artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_stage_comparison.md`.

Use the EQ-Bench score for comparability with a public benchmark. Use
feature-level corpus rates, teacher-forced propensity, and free-running
compounding for mechanism claims.

## Teacher-Forced Propensity

Teacher-forced propensity estimates local model preference for initiating a
feature in a fixed reference context. For each feature, the harness defines:

- an opportunity set: positions or spans where the feature could begin;
- an initiating token set or exact candidate sequence: continuations that
  commit to the feature;
- a reference indicator: whether the held-out reference actually initiates the
  feature at that opportunity.

For each checkpoint and feature, the model is evaluated under teacher forcing
on held-out reference text. The harness records the probability mass assigned
to feature-initiating continuations and compares it with the empirical
reference initiation rate.

Let \(O_f\) be the opportunity set for feature \(f\). For opportunity
\(o \in O_f\), let \(p_m(I_f \mid x_o)\) be model \(m\)'s probability mass on
feature-initiating continuations in context \(x_o\), and let \(y_o\) indicate
whether the reference text actually initiates the feature. The basic
amplification factor is:

```text
AF_f(m) =
  mean_{o in O_f} p_m(I_f | x_o)
  --------------------------------
  mean_{o in O_f} y_o
```

Neutral-normalized AF divides a feature AF by the AF of matched neutral control
phrases for the same stage. This is used to reduce, but not eliminate, the risk
that a broad calibration shift is mistaken for a slop-specific shift:

```text
normalized_AF_f(m) = AF_f(m) / AF_neutral(m)
```

Calibration remains a general concern when interpreting neural probability
estimates [@guo_2017_calibration].

Teacher-forced claims are only strong when reference denominators are adequate.
Sparse features such as contrastive negation, stock closers, and selected
Tier-3 features are flagged when reference initiations are too few for stable
headline claims.

## Free-Running Generation

Generation uses SGLang for pure generation tasks in the completed production
path [@zheng_2023_sglang]. The bounded generation scope is:

- OLMo reduced main style panel: 8,192 generations.
- OLMo reduced temperature panel: 6,144 generations.
- OLMo reduced long-output compounding panel: 2,048 generations.
- SmolLM3 no-think production-shape panel: 49,152 generations.
- OLMo Instruct/Think/RL-Zero stretch panel: 5,120 final-answer-mode
  generations.

The original large OLMo grid remains an optional addendum. It is not required
for the bounded paper estimand.

Free-running outputs are scored with the same feature matchers after final
answer extraction and mode-specific cleanup. Rates are reported per 1,000
generated tokens or per opportunity, depending on feature contract.

## Compounding Analysis

Compounding tests whether style markers self-condition during generation. The
analysis uses two complementary views:

1. **Expected versus observed:** compare generated feature rates with rates
   expected from teacher-forced local propensities under a simple independence
   assumption.
2. **Prior-hit windows:** compare later-window hit risk after a generation has
   already produced a feature hit against later-window risk when no prior hit
   occurred.

The strongest compounding result is for `slop_lexicon`: prior-hit
windows are much more likely to contain later hits than no-prior windows in the
measured panels. This supports a path-dependent generation story rather than a
pure fixed-context propensity story.

This part of the method is related to prior work on neural text degeneration
and repetition, but the estimand here is narrower: whether an already-emitted
style marker predicts later same-family markers within the same output
[@holtzman_2019_neural_text_degeneration; @welleck_2019_unlikelihood].

## Amplification Spectrum Assembly

The amplification spectrum joins, by feature and stage:

- Phase 1 corpus rates and DPO chosen/rejected deltas.
- Teacher-forced AF and neutral-normalized AF where denominator support exists.
- Free-running generation rates.
- Compounding metrics and bootstrap intervals where available.
- Feature classifications such as inherited, SFT-amplified,
  preference-amplified, and compounding-dominant.

The OLMo spectrum and SmolLM3 no-think spectrum are compared as a
cross-ladder check. Because only a small set of features has non-missing AF in
both ladders, cross-ladder correlations should be described as suggestive
replication pressure rather than a robust universal law.

## Detector Discovery And Tier-3 Rerun

Phase 4 trains and evaluates a detector, then uses attribution to propose new
features:

1. Fine-tune ModernBERT-large on HAP-E plus OLMo/SmolLM3 generation/reference
   slices [@reinhart_2024_hape_style; @brown_hape_dataset_card;
   @warner_2024_modernbert; @answerai_modernbert_card].
2. Evaluate transfer to held-out HAP-E and OLMo/SmolLM3 generation slices.
3. Attribute detector decisions with embedding-level integrated gradients over
   10,000 generated documents [@sundararajan_2017_integrated_gradients].
4. Embed high-attribution spans with GTE-large and cluster with HDBSCAN
   [@alibaba_gte_large_en_v15_card; @campello_2013_hdbscan;
   @mcinnes_2017_hdbscan].
5. Convert cluster summaries into provisional matcher specs.
6. Run a bounded exact sequence-mass teacher-forced OLMo rerun for the 10
   candidate features plus neutral controls.

The primary Tier-3 exact rerun uses 512 reference documents and all four OLMo
stages. Its outputs are retained under
`artifacts/phase4/modernbert_detector_combined_v2_clean/tier3_teacher_forced_exact_512/`.

Tier-3 features are not promoted into the core taxonomy until manual
perceptibility labels and matcher precision checks are complete.

## Statistical Treatment

The analysis uses paired designs wherever possible:

- DPO corpus deltas compare chosen and rejected responses within the same
  preference pair.
- Teacher-forced stage effects use paired opportunity-level tests where the
  opportunity contract supports pairing.
- Free-running stage effects use paired prompt or prompt-cluster units when
  comparing adjacent stages.

Benjamini-Hochberg FDR correction is applied within the relevant feature set
for multi-feature stage-effect summaries. Bootstrap intervals are clustered by
document, prompt, or prompt cluster depending on the measurement layer.

Sparse denominators are reported explicitly. A large AF with few reference
initiations is treated as provisional even if the point estimate is large.

## Reproducibility Pointers

Primary review artifacts:

- Claim matrix: `docs/experiments/paper_claim_matrix.md`
- Paper scaffold: `docs/experiments/paper_scaffold.md`
- Draft tables: `docs/experiments/paper_tables.md`
- Phase 1 gap audit: `docs/experiments/phase1_gap_audit.md`
- Pybiber register analysis: `docs/experiments/phase1_pybiber_register_analysis.md`
- Precision status: `docs/experiments/precision_validation_status.md`
- Phase 3 integrated conclusion: `docs/experiments/phase3_integrated_conclusion_report.md`
- Phase 4 detector report: `docs/experiments/phase4_detector_discovery_report.md`
- Phase 4 exact Tier-3 addendum:
  `docs/experiments/phase4_tier3_teacher_forced_addendum.md`

Core reproducibility commands:

- `uv run slop-pybiber-full`
- `uv run slop-eqbench-score`
- `uv run slop-summarize-precision-validation`
- `uv run slop-render-paper-figures`

## Methods Caveats To Preserve In The Paper

- The estimand is bounded production scope, not the original exhaustive
  generation grid.
- Tier-1 precision validation is scoped: four independent regex rows pass,
  `rule_of_three_approx` is demoted, and `stock_openers_closers` is derived.
- Dolci DPO chosen/rejected contrasts are not pure human-preference contrasts.
- Generated-output full pybiber was not run; do not make main-paper generated
  register claims from retired proxy artifacts.
- EQ-Bench Slop Score is an aggregate diagnostic bridge, not a causal
  amplification measure.
- Detector-discovered Tier-3 features remain candidates until human
  perceptibility labels exist.
- Sparse reference denominators make several large AF point estimates
  provisional.
