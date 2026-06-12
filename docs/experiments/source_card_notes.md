# Source Card Notes

Date: 2026-06-12.

These notes record primary-source dataset-card facts and local metadata probes
needed before interpreting Phase 1 measurements. They are not a substitute for
manual precision validation.

## OLMo 3 Dolci SFT

Primary source: Hugging Face dataset card
`https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT`.

Verified card facts:

- Canonical dataset ID is `allenai/Dolci-Instruct-SFT`.
- The card says this mixture was used to train `Olmo 3 7B Instruct SFT`.
- Dataset card metadata lists train split size as 2,152,112 examples.
- Visible schema fields are `id`, `messages`, `source_dataset`, and `domain`.
- `messages` is a chat-style list with `content`, function metadata, and
  `role`.
- The card lists a mixture of existing and new prompt sources, including
  OpenThoughts 3, CoCoNot, FLAN, Tulu 3 Persona variants, WildGuardMix,
  WildJailbreak, Aya, TableGPT, SciRIFF, Evol CodeAlpaca, Dolci Python
  Algorithms, Logic Puzzles, Verifiable Reasoning, and Dolci Tool Use.

Local retained-prefix probe:

- W&B run:
  `https://wandb.ai/phulin-self/slop-stage1/runs/xieewrhy`.
- Probe file:
  `artifacts/stage1/corpora/olmo3_dolci_10k_source_metadata_probe.json`.
- First 10,000 streamed rows contained 18 `source_dataset` values and 8
  `domain` values.
- Top SFT source counts in the retained prefix were Dolci Instruct Python
  Algorithms (1,425), Logic Puzzles (1,240), Tulu 3 Persona MATH (1,133),
  Dolci Instruct Precise IF (1,083), and Evol CodeAlpaca (853).
- Domain counts in the retained prefix were Coding (2,533), Math (1,983),
  Other (1,978), Precise IF (1,083), Safety (818), Science (806),
  Multilingual (744), and Chat (55).

Current interpretation:

- The current normalizer correctly treats assistant-role message content as
  the measured SFT target response for corpus-frequency work.
- `source_dataset` and `domain` support a provenance-like retained-prefix
  split, but they are not yet a clean human-vs-synthetic provenance label.
  Phase 1 claims should describe them as source/domain strata until a stronger
  provenance mapping is added.

## OLMo 3 Dolci DPO

Primary source: Hugging Face dataset card
`https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO`.

Verified card facts:

- Canonical dataset ID is `allenai/Dolci-Instruct-DPO`.
- The card says this mixture was used to preference tune Olmo 3 Instruct 7B.
- The card says it contains 260,000 preference pairs total.
- The card decomposes the mixture into 125,000 Delta Learning heuristic pairs,
  125,000 delta-aware Ultrafeedback-esque GPT-judge pairs, and 10,000
  multiturn preference pairs.
- Dataset preview/schema exposes `chosen`, `rejected`, `chosen_model`,
  `rejected_model`, `prompt_id`, and `preference_type`.
- The dataset has 259,922 rows in the Hugging Face card UI.

Local retained-prefix probe:

- W&B run:
  `https://wandb.ai/phulin-self/slop-stage1/runs/xieewrhy`.
- Probe file:
  `artifacts/stage1/corpora/olmo3_dolci_10k_source_metadata_probe.json`.
- First 10,000 streamed rows contained 4 preference types:
  `llm_judged` (4,856), `delta_learning` (4,730),
  `multiturn_self_talk` (240), and `multiturn_synthetic_context` (174).
- First 10,000 streamed rows contained 24 chosen-model values and 24
  rejected-model values.
- Top chosen model was `qwen3-no_reasoning-32b` with 5,168 rows.
- Top rejected model was `qwen3-no_reasoning-0.6b` with 5,669 rows.

Local retained metadata-aware pair analysis:

- W&B run:
  `https://wandb.ai/phulin-self/slop-stage1/runs/sgszvu4u`.
- Input pair-delta file:
  `artifacts/stage1/census/olmo3_dolci_dpo_10k_metadata_pair_deltas.csv`.
- Output analysis file:
  `artifacts/stage1/census/olmo3_dolci_dpo_10k_metadata_pair_analysis.csv`.
- The analysis grouped 79,960 pair-delta rows by
  `source/subset/preference_type/chosen_model/rejected_model/feature`.
- It produced 3,352 grouped feature rows and 72 BH-FDR-significant rows before
  manual precision validation.
- Grouped row counts by `preference_type` were `llm_judged` 3,312,
  `multiturn_self_talk` 16, `multiturn_synthetic_context` 16, and
  `delta_learning` 8.
- Top significant examples include `delta_learning`
  `qwen3-no_reasoning-32b` vs `qwen3-no_reasoning-0.6b` for `stock_closers`
  (`n=4,730`, `mean_delta=0.169827...`, `p=2.628e-15`, `q=8.809e-12`,
  `chosen_gt_rejected`) and `punctuation_rhythm` (`n=4,730`,
  `mean_delta=4.271586...`, `p=5.253e-11`, `q=8.804e-08`), plus
  `llm_judged` `gpt-120b` vs `olmo2-1b` `punctuation_rhythm` (`n=64`,
  `mean_delta=95.202...`, `p=7.672e-10`, `q=8.572e-07`).

Current interpretation:

- Result A on Dolci DPO should not be described as a pure human-preference
  effect. In this dataset it measures the stylistic signal induced by a mixed
  preference construction: Delta Learning heuristic pairs, LLM-judged pairs,
  and multiturn pairs.
- The retained 10k metadata-aware pair analysis now stratifies by
  `preference_type` and model pair, narrowing the DPO interpretation blocker.
  It still does not close Phase 1 because manual precision validation, final
  chosen/rejected construction semantics, and SmolLM3/Tulu work remain.
- The current normalizer correctly extracts assistant-only chosen/rejected
  responses and preserves stable pair identity through `prompt_id` when that
  field is selected by the caller.

## OLMo 3 Dolma 3 Pretraining Reference

Primary source: Hugging Face dataset card
`https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B`.

Verified card facts:

- The card describes the dataset as the full mix of documents used to train
  OLMo 3 7B.
- The mix is approximately 5.93T tokens and 3.87B documents.
- Listed source/doc-type categories are Common Crawl web pages,
  olmOCR science PDFs, Stack Edu code, FineMath, RPJ Proofpile ArXiv, and
  English Wikipedia.
- The card reports mix proportions: Common Crawl 76.07%, olmOCR science PDFs
  13.57%, Stack Edu 6.89%, FineMath 2.56%, RPJ Proofpile ArXiv 0.86%, and
  Wikipedia 0.04%.
- License is ODC-BY.

Current interpretation:

- The retained Phase 1 pretraining reference should sample separate strata
  from the available `source` or metadata fields rather than treating Dolma 3
  as one homogeneous corpus.
- Code should remain excluded from the main pretraining baseline unless a code
  audit bucket is intentionally reported.
- Forums/Q&A is still not verified as a populated Dolma 3 stratum from the
  public card evidence above; it remains a desired assistant-register baseline,
  not a proven source bucket in the current implementation.

## Open Source-Verification Tasks

- Add a retained Dolma 3 stratification probe over a bounded sample and log
  `source` counts to W&B.
- Use DPO pair-analysis grouping by `preference_type`, `chosen_model`, and
  `rejected_model` in any Result A writeup, and keep aggregate Dolci DPO
  interpretations explicitly qualified as mixed-construction signal.
- Map Dolci SFT `source_dataset` and `domain` fields to a defensible
  human/synthetic/mixed provenance taxonomy, or avoid human-vs-synthetic
  claims in Phase 1.
- Identify and verify the SmolLM3 pretraining, SmolTalk2 SFT, and Tulu/APO
  preference sources separately before starting replication-ladder Phase 1.
