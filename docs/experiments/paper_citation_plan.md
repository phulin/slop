# Paper Citation Plan

Date: 2026-06-18

Purpose: define the citation slots needed to keep the manuscript draft
citation-ready. This file deliberately avoids unverified bibliographic
details. Before adding a citation to a `.bib` file or final manuscript, verify
title, authors, year, venue, and URL/DOI from the primary source.

Checked reference artifacts:

- Source inventory: `docs/experiments/paper_reference_sources.md`
- BibTeX file: `docs/experiments/paper_references.bib`
- Integrated draft with citation keys:
  `docs/experiments/paper_manuscript_draft.md`

Status note: the integrated manuscript and standalone section drafts now pass
the paper-package citation gate. Every cited key is present in
`paper_references.bib`, and every BibTeX key has a verified row in
`paper_reference_sources.md`. The remaining work is optional expansion: add a
new source only if final Methods or Related Work prose actually needs that
background.

## Citation Policy

- Prefer primary sources: papers, official dataset/model cards, official docs,
  or benchmark repositories.
- Do not cite secondary blog summaries when a paper or official repository
  exists.
- For model/data ladders, cite both the model card or technical report and the
  dataset card/source.
- For software methods, cite the original method paper where appropriate and
  the implementation docs/repository when the exact tool behavior matters.
- Keep empirical claims sourced to this repo's artifacts; external citations
  support background, method provenance, and dataset/model provenance.

## Related-Work Buckets

### Stylistic Artifacts And AI-Writing Detection

Needed to motivate the paper's problem statement and Phase 4 detector framing.

Citation slots:

| Slot | Need | Preferred source type | Used in draft |
|---|---|---|---|
| `ai_writing_detection_survey` | Survey or representative work on detecting AI-generated writing style | Peer-reviewed survey or benchmark paper | Related Work; covered by `wu_2023_llm_text_detection_survey` |
| `hap_e` | HAP-E human/AI parallel corpus provenance | Dataset paper/card | Methods, Phase 4 |
| `shaib_slop_taxonomy` | Human-perceived slop taxonomy used for Phase 4 annotation framing | Original paper or dataset card | Phase 4, Limitations |

Verification notes:

- Confirm HAP-E field names and whether pre-extracted Biber features are part
  of the released artifact or a companion artifact.
- Shaib et al. is now cited for taxonomy framing. Confirm exact taxonomy labels
  before claiming alignment with our proxy Venn.

### Register Analysis And Biber Features

Needed to justify the pybiber register layer and the interpretation of
nominalized/expository versus narrative/personal prose.

Citation slots:

| Slot | Need | Preferred source type | Used in draft |
|---|---|---|---|
| `biber_register_classic` | Foundational multidimensional/register analysis | Original Biber book/paper | Methods, Result 1, Discussion |
| `computational_register_features` | Modern computational use of Biber-style features | Paper or package docs | Optional; current draft cites Biber 1988 plus pybiber |
| `pybiber` | Exact package/tool used for 67-feature extraction | Package docs/repository plus any associated paper | Methods, Artifact appendix |

Verification notes:

- Verify whether `pybiber` exposes exactly 67 features by package docs or by
  our code/artifact, and cite package docs only for implementation provenance.
- Keep the substantive pybiber result sourced to our artifacts, not to Biber
  literature.
- Biber 1988 is now verified through Cambridge for classic register-analysis
  background.

### Preference Optimization And Post-Training

Needed to frame SFT/DPO/APO/RLVR stage localization and avoid a simplistic
"DPO creates slop" story.

Citation slots:

| Slot | Need | Preferred source type | Used in draft |
|---|---|---|---|
| `dpo_original` | Direct Preference Optimization method | Original DPO paper | Methods, Discussion |
| `apo_or_tulu_preference` | APO or SmolLM3 preference-stage method provenance | Original APO paper and/or SmolLM3 technical report | Methods, Result 4 |
| `rlvr_background` | RLVR/final-stage training context for OLMo final checkpoint | OLMo model card/report or RLVR method paper | Covered by OLMo model/card provenance unless final text expands RLVR method claims |
| `reward_model_style_bias` | Background on preference/reward models selecting for style or verbosity | Primary paper | Related Work, Discussion |

Verification notes:

- For OLMo, cite model/data cards for what the released checkpoints are, not
  just general DPO/RLVR methods.
- For SmolLM3, verify the exact meaning of the "DPO" label in our internal
  artifacts versus APO/model-soup/long-context merge in the public source.
- APO method background and verbosity-bias background are now cited. The exact
  SmolLM3 preference-stage implementation should still be phrased from
  SmolLM3 sources/local artifacts, not inferred from the APO paper alone.

### Open Model And Dataset Ladders

Needed for provenance of OLMo, Dolma, Dolci, SmolLM3, SmolTalk2, and Tulu-3
references.

Citation slots:

| Slot | Need | Preferred source type | Used in draft |
|---|---|---|---|
| `olmo3_model_card` | OLMo 3 checkpoint ladder and released stages | Official model card/report | Methods |
| `dolma3_dataset` | Pretraining reference corpus provenance | Official dataset card/report | Optional; current draft keeps Dolma sample sizes tied to retained artifacts and OLMo/Dolci provenance |
| `dolci_sft_dataset` | SFT data provenance | Official dataset card | Methods |
| `dolci_dpo_dataset` | DPO pair provenance and construction caveats | Official dataset card | Methods, Limitations |
| `smollm3_model_card` | SmolLM3 no-think/final-stage caveats | Official model card/report | Methods, Result 4 |
| `smoltalk2_dataset` | SmolLM3 SFT reference provenance | Official dataset card/report | Optional; current draft does not make SmolTalk2-specific claims |
| `tulu3_preference` | Preference-data background for SmolLM3 APO stage if applicable | Official dataset/model report | Optional; current draft avoids detailed Tulu-3 provenance claims |

Verification notes:

- Check exact checkpoint names and whether the public docs describe OLMo SFT
  checkpoint merging and SmolLM3 model soup/long-context merge.
- Keep our measured sample sizes sourced to `paper_tables.md` and local
  artifacts rather than external cards.

### Slop Lexicons And EQ-Bench

Needed for the EQ-Bench score bridge and public slop lexicon framing.

Citation slots:

| Slot | Need | Preferred source type | Used in draft |
|---|---|---|---|
| `eq_bench_slop_score` | EQ-Bench Slop Score definition, word/trigram lists, normalization | Official EQ-Bench repository/docs | Methods, Result 2 |
| `public_slop_lexicon` | Any public slop-word/trigram list source used by EQ-Bench or our lexicon | Official repository/docs | Covered by EQ-Bench for the benchmark bridge; do not use to validate our historical lexicon |

Verification notes:

- Cite EQ-Bench for benchmark comparability only. Do not cite it as evidence
  for our amplification claims.
- Confirm license/redistribution status for any vendored lists before
  release-facing writeup.

### Teacher-Forced Propensity, Calibration, And Generation Evaluation

Needed to situate the fixed-context probability measurement and generation
panels.

Citation slots:

| Slot | Need | Preferred source type | Used in draft |
|---|---|---|---|
| `teacher_forcing_lm_eval` | Background on teacher forcing / scoring reference continuations | Method paper or standard LM evaluation reference | Optional; current draft defines AF directly and cites calibration context |
| `calibration_lm` | Background on probability calibration if discussing neutral-normalized AF | Primary paper/survey | Methods |
| `self_conditioning_generation` | Prior work on repetition/self-conditioning/exposure effects in generation | Primary paper | Compounding discussion |
| `sglang` | SGLang generation runtime provenance | Official paper/docs/repository | Methods |

Verification notes:

- Our AF definition is project-specific; cite external work for context, then
  define AF from our method.
- SGLang citation is software provenance, not evidence for results.
- Calibration and generation-degeneration background are now cited. A
  teacher-forcing/perplexity citation remains optional unless final Methods
  prose needs broader LM-scoring context.

### Attribution, Embeddings, And Clustering

Needed for Phase 4 detector-discovery methods.

Citation slots:

| Slot | Need | Preferred source type | Used in draft |
|---|---|---|---|
| `modernbert` | ModernBERT model architecture/checkpoint provenance | Official paper/model card | Phase 4 Methods |
| `integrated_gradients` | Attribution method | Original Integrated Gradients paper | Phase 4 Methods |
| `gte_large` | Embedding model provenance | Official model card/report | Phase 4 Methods |
| `hdbscan` | Clustering algorithm | Original HDBSCAN paper/software docs | Phase 4 Methods |

Verification notes:

- Distinguish detector-training data provenance from attribution/clustering
  method provenance.
- Keep human-perceptibility claims uncited until actual labels exist; cite
  taxonomy background only.
- GTE-large model-card provenance is now verified for the exact embedding
  checkpoint named in the Phase 4 pipeline.

## Citation Coverage Audit

The package checker enforces three citation invariants:

1. Every citation key in the integrated manuscript and standalone Methods,
   Results, and Intro/Discussion drafts must exist in `paper_references.bib`.
2. Every BibTeX key must have a verified source row in
   `paper_reference_sources.md`.
3. Every verified source row must have a matching BibTeX entry.

The current package passes these checks with `uv run slop-check-paper-package
--root /home/user/slop`.

## Manuscript Citation Map

| Manuscript section | Citation slots needed |
|---|---|
| Introduction | `eq_bench_slop_score`; model/data provenance keys |
| Related Work | All buckets, in compressed form, including `ai_writing_detection_survey` |
| Methods: Data/model ladders | `olmo3_model_card`, `dolci_sft_dataset`, `dolci_dpo_dataset`, `smollm3_model_card`; optional Dolma/SmolTalk/Tulu detail only if final prose expands |
| Methods: Features | `biber_register_classic`, `pybiber`, `eq_bench_slop_score`, `shaib_slop_taxonomy` |
| Methods: Propensity/generation | `calibration_lm`, `self_conditioning_generation`, `sglang`; optional teacher-forcing background only if needed |
| Methods: Detector discovery | `hap_e`, `modernbert`, `integrated_gradients`, `gte_large`, `hdbscan` |
| Results | Mostly internal artifacts; cite method/dataset sources only as context |
| Discussion | `reward_model_style_bias`, `self_conditioning_generation`, `biber_register_classic` |

## Next Citation Work

1. Add new references only when final prose introduces a claim that needs an
   external source beyond the verified set.
2. Expand `docs/experiments/paper_references.bib` only after metadata is
   verified in `docs/experiments/paper_reference_sources.md`.
3. Keep empirical result claims tied to experiment artifacts rather than to
   external background citations.
4. Revisit Related Work after final venue selection; the current compact
   section is citation-complete for the bounded draft.
