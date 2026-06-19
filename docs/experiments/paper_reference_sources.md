# Paper Reference Sources

Date: 2026-06-18

Purpose: record primary/official sources verified for the current paper draft.
This supports `docs/experiments/paper_references.bib`. Empirical claims in the
paper remain sourced to local artifacts; these references support background,
method provenance, dataset/model provenance, and benchmark provenance.

## Verified Sources Added To BibTeX

| Key | Source | Verified from | Notes for use |
|---|---|---|---|
| `team_olmo_2025_olmo3` | Olmo 3 technical report | `https://arxiv.org/abs/2512.13961` | Use for OLMo 3 model family and checkpoint-ladder provenance. |
| `allenai_olmo3_sft_card` | OLMo 3 7B Instruct SFT model card | `https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT` | Use for released SFT checkpoint and Dolma/Dolci provenance. |
| `allenai_olmo3_dpo_card` | OLMo 3 7B Instruct DPO model card | `https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO` | Use for released DPO checkpoint provenance. |
| `allenai_dolci_sft_card` | Dolci Instruct SFT dataset card | `https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT` | Use for Dolci SFT mixture provenance. |
| `allenai_dolci_dpo_card` | Dolci Instruct DPO dataset card | `https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO` | Use for Dolci DPO pair provenance and size. |
| `huggingface_2025_smollm3_blog` | SmolLM3 official blog | `https://huggingface.co/blog/smollm3` | Use for SmolLM3 training/provenance overview. |
| `huggingface_smollm3_card` | SmolLM3-3B model card | `https://huggingface.co/HuggingFaceTB/SmolLM3-3B` | Use for no-think/think model provenance and capability caveats. |
| `eqbench_slop_score` | EQ-Bench Slop Score page | `https://eqbench.com/slop-score.html` | Use for score components, word/trigram list provenance, and benchmark bridge. |
| `rafailov_2023_dpo` | Direct Preference Optimization paper | `https://arxiv.org/abs/2305.18290` | Use for DPO method background. |
| `warner_2024_modernbert` | ModernBERT paper | `https://arxiv.org/abs/2412.13663` | Use for ModernBERT architecture/model provenance. |
| `answerai_modernbert_card` | ModernBERT model card | `https://huggingface.co/answerdotai/ModernBERT-base` | Use for exact detector backbone provenance if needed. |
| `zheng_2023_sglang` | SGLang paper | `https://arxiv.org/abs/2312.07104` | Use for SGLang runtime/software provenance. |
| `reinhart_2024_hape_style` | HAP-E / LLM rhetorical style paper | `https://arxiv.org/abs/2410.16107` and `https://www.pnas.org/doi/10.1073/pnas.2422455122` | Use for HAP-E, Biber-style LLM-vs-human style framing. |
| `brown_hape_dataset_card` | HAP-E dataset card | `https://huggingface.co/datasets/browndw/human-ai-parallel-corpus/blob/main/README.md` | Use for dataset construction details. |
| `wu_2023_llm_text_detection_survey` | Survey on LLM-generated text detection | `https://arxiv.org/abs/2310.14724` | Use for broad AI-generated-text detection framing; do not use as evidence for this paper's feature-level slop measurements. |
| `pybiber_pypi` | pybiber package page | `https://pypi.org/project/pybiber/` | Use for package provenance and 67-feature extraction claim. |
| `biber_1988_variation` | Foundational Biber register-analysis book | `https://www.cambridge.org/core/books/variation-across-speech-and-writing/A546CF5ED8F8E62F1432CB2F369CF356` | Use for Biber-style multidimensional/register analysis background. |
| `shaib_2025_slop` | AI slop taxonomy and annotation framework paper | `https://arxiv.org/abs/2509.19163` | Use for human-perceived slop taxonomy background and Phase 4 annotation framing. |
| `doosterlinck_2024_apo` | Anchored Preference Optimization and CLAIR paper | `https://arxiv.org/abs/2408.06266` and `https://huggingface.co/papers/2408.06266` | Use for APO method background, not as evidence about our SmolLM3 artifacts. |
| `saito_2023_verbosity_bias` | Verbosity bias in LLM preference labeling | `https://arxiv.org/abs/2310.10076` | Use for preference/evaluator style-bias background. |
| `guo_2017_calibration` | Neural-network calibration paper | `https://arxiv.org/abs/1706.04599` | Use for calibration background when discussing neutral-normalized AF caveats. |
| `holtzman_2019_neural_text_degeneration` | Neural text degeneration and nucleus sampling paper | `https://arxiv.org/abs/1904.09751` | Use for generation/decoding and repetition-degeneration background. |
| `welleck_2019_unlikelihood` | Unlikelihood training paper for repetitive/dull text generation | `https://arxiv.org/abs/1908.04319` | Use for repetition/self-conditioning background, not as evidence for our compounding result. |
| `alibaba_gte_large_en_v15_card` | GTE-large-en-v1.5 model card | `https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5` | Use for exact embedding-model provenance in Phase 4. |
| `sundararajan_2017_integrated_gradients` | Integrated Gradients paper | `https://arxiv.org/abs/1703.01365` | Use for attribution method background. |
| `mcinnes_2017_hdbscan` | HDBSCAN JOSS software paper | `https://joss.theoj.org/papers/10.21105/joss.00205` | Use for HDBSCAN implementation/provenance. |
| `campello_2013_hdbscan` | HDBSCAN algorithm paper | `https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14` | Use for density-based hierarchical clustering algorithm background. |

## Still Needs Verification

These remain citation-plan slots, not BibTeX entries:

- Exact SmolLM3 preference-stage implementation details. The current source set
  supports general APO background and SmolLM3 model provenance, but not a
  stronger claim that the public SmolLM3 checkpoint exactly implements one
  cited APO variant in isolation.
- Teacher-forcing / LM-evaluation background. The project-specific AF
  definition is already described locally; add an external reference only if
  the final Methods section needs broader methodological context.
