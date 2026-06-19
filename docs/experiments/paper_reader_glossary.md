# Paper Reader Glossary

Date: 2026-06-18

Purpose: give new readers a compact map of the paper's recurring terms before
they read the manuscript, claim matrix, or evidence reports. This glossary is
not a claim authority; use `docs/experiments/paper_claim_matrix.md` for allowed
claim wording and caveats.

| Term | Meaning in this paper | Boundary |
|---|---|---|
| Slop style | A set of measurable stylistic markers and register features in model data or outputs. | Not factual error, low answer quality, or a single global class. |
| Register | Broad linguistic style measured through features such as tense, pronouns, nominalizations, adjectives, adverbs, and coordination. | Corpus-side register evidence is separate from generated-output marker rates. |
| Pybiber | The full 67-feature Biber-style extractor used for Phase 1 corpus-side register analysis. | Generated-output full pybiber is not claimed. |
| Tier 1 | Hand-defined lexical/rhetorical regex features such as `contrastive_negation`, `slop_lexicon`, `stock_openers`, and `stock_closers`. | Only precision-supported rows carry publication-grade corpus-rate claims. |
| Tier 2 | Full pybiber register features. | Active paper use is Phase 1 corpus-side only. |
| Tier 3 | Detector-discovered candidate style families from the Phase 4 ModernBERT attribution and clustering run. | Candidate-only until human perceptibility labels and matcher precision validation exist. |
| EQ-Bench Slop Score | A public aggregate slop-score diagnostic used as a benchmark bridge. | Not the causal measurement layer and not an amplification factor; its stage path can disagree with feature-specific propensity. |
| Corpus rate | How often a feature appears in retained pretraining, SFT, or preference data. | Measures data contents, not model propensity. |
| Teacher-forced propensity | A fixed-context model probability measurement for initiating a feature at defined opportunities. | Decoding-free; it asks what the checkpoint locally assigns probability to. |
| Free-running emission | Feature rates observed in sampled model generations. | Depends on prompts, decoding settings, and previous generated text. |
| Compounding | A generation-time test asking whether later feature hits become more likely after an earlier hit appears. | Distinct from teacher-forced local propensity. |
| Amplification factor (AF) | Mean model initiation probability divided by mean reference initiation indicator for a feature/opportunity set. | Sparse denominators can make large point estimates provisional. |
| Neutral-normalized AF | Feature AF divided by an AF for matched neutral control phrases. | Reduces broad calibration confounds but does not eliminate them. |
| OLMo ladder | The open OLMo 3 7B checkpoint sequence: base, SFT, DPO, and final/RLVR. | Main fine-grained stage-localization ladder. |
| SmolLM3 ladder | The open SmolLM3 no-think comparison sequence: base, SFT, DPO/APO-labeled preference cell, and final. | Replication pressure, not a one-to-one OLMo stage clone. |
| Dolci SFT | The OLMo supervised fine-tuning target-response dataset sample. | English-filtered retained sample, not all possible Dolci rows. |
| Dolci DPO | The OLMo preference-pair dataset sample with chosen and rejected responses. | Mixed synthetic/Delta Learning construction; not pure human preference; pybiber reads chosen as more descriptive/expository and rejected as more personal/narrative. |
| Dolma reference | The retained OLMo pretraining reference sample. | English-filtered sample from a bounded scan, not full Dolma. |
| Preference-data contrast | Chosen-versus-rejected pair comparison within Dolci DPO or similar preference data. | Interpreted through dataset construction and model provenance. |
| Candidate human-perceived slop | A detector-derived style family that might be perceptible to humans. | Requires Phase 4 human labels before being promoted. |

## How To Read The Evidence

- Use corpus rates and full pybiber for data-side inheritance and register
  shift.
- Use teacher-forced propensity and AF for fixed-context model-side
  localization.
- Use free-running emission and compounding for sampled-output behavior.
- Use EQ-Bench as an aggregate public bridge, not as the mechanism estimate.
- Use Phase 4 detector features as candidate families until the handoff
  package is labeled and summarized.
