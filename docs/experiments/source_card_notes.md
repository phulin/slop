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
- Historical all-feature diagnostic examples include `delta_learning`
  `qwen3-no_reasoning-32b` vs `qwen3-no_reasoning-0.6b` for `stock_closers`
  (`n=4,730`, `mean_delta=0.169827...`, `p=2.628e-15`, `q=8.809e-12`,
  `chosen_gt_rejected`) and `punctuation_rhythm` (`n=4,730`,
  `mean_delta=4.271586...`, `p=5.253e-11`, `q=8.804e-08`), plus
  `llm_judged` `gpt-120b` vs `olmo2-1b` `punctuation_rhythm` (`n=64`,
  `mean_delta=95.202...`, `p=7.672e-10`, `q=8.572e-07`). Punctuation results
  are excluded from the revised Phase 1 core scope.

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

## SmolLM3 Replication Sources

Primary sources inspected:

- SmolLM3 model card:
  `https://huggingface.co/HuggingFaceTB/SmolLM3-3B`.
- SmolLM3 pretraining dataset collection:
  `https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets`.
- SmolTalk2 dataset card:
  `https://huggingface.co/datasets/HuggingFaceTB/smoltalk2`.

Current primary-source facts:

- The SmolLM3 model card describes a 3B model pretrained on about 11T tokens
  with a staged curriculum over web, code, math, and reasoning data, followed
  by mid-training, SFT, and APO.
- The model card says pretraining datasets are published in the SmolLM3
  pretraining collection and that training configs/code are in
  `huggingface/smollm`.
- The pretraining collection lists candidate exact public sources including
  `HuggingFaceFW/fineweb-edu`, `mlfoundations/dclm-baseline-1.0`,
  `epfml/FineWeb2-HQ`, `HuggingFaceFW/fineweb-2`,
  `HuggingFaceTB/finemath`, `bigcode/the-stack-v2`,
  `HuggingFaceTB/issues-kaggle-notebooks`,
  `HuggingFaceTB/stackexchange_2025_md`, `LLM360/MegaMath`,
  `HuggingFaceTB/stack-edu`, `HuggingFaceTB/smollm-corpus`,
  `allenai/dolmino-mix-1124`, `nvidia/OpenMathReasoning`,
  `nvidia/OpenCodeReasoning`, and `facebook/natural_reasoning`.
- The collection notes Stage 1 proportions as 85% web, 12% code, and 3% math;
  it also labels additional Stage 2 and Stage 3 decay sources. Exact stage
  weights still need to be resolved from the training recipe before sampling.
- The SmolTalk2 card says `HuggingFaceTB/smoltalk2` contains the Mid, SFT, and
  Preference subsets corresponding to SmolLM3 post-training phases.
- SmolTalk2 SFT contains 25 datasets, split into 10 `think` datasets with
  about 1.5M rows and 15 `no_think` datasets with about 1.9M rows. The card
  describes decontamination, emoji filtering, and a `chat_template_kwargs`
  column created from system messages or tool descriptions.
- SmolTalk2 preview/schema exposes chat-style `messages` and a dataset/source
  identifier column in the preview; the normalizer still needs a bounded schema
  probe to confirm the exact field names for each SFT config before sampling.
- SmolTalk2 Preference contains two APO sources totaling 446,886 rows:
  `llama_3.1_tulu_3_8b_preference_mixture_no_think` with 230,501 examples and
  `tulu_3_8b_pref_mix_Qwen3_32B_Qwen3_0.6B_think` with 216,385 examples.
- The SmolTalk2 card states that the `think` preference equivalent uses the
  no-think prompts, with chosen responses generated by Qwen3-32B and rejected
  responses generated by Qwen3-0.6B.

Live bounded source-identification probe:

- W&B run:
  `https://wandb.ai/phulin-self/slop-stage1/runs/wiedc2oj`.
- Run name:
  `stage1-smollm3-tulu-source-identification-probe`.
- Local probe file:
  `artifacts/stage1/corpora/smollm3_tulu_source_probe.json`.
- Probe coverage: 5 dataset probes, 2 model probes, 3 dataset searches, 1
  model search, 7 sample rows, and 80 search-result rows.
- `HuggingFaceTB/smoltalk2` exposes configs `Mid`, `Preference`, and `SFT`.
- `Preference` splits include
  `llama_3.1_tulu_3_8b_preference_mixture_no_think` and
  `tulu_3_8b_pref_mix_Qwen3_32B_Qwen3_0.6B_think`.
- `SFT` splits include many source-labeled `think` and `no_think` splits,
  including `OpenThoughts3_1.2M_think`,
  `smoltalk_smollm3_everyday_conversations_no_think`,
  `smoltalk_smollm3_smol_magpie_ultra_no_think`, and
  `tulu_3_sft_personas_instruction_following_no_think`.
- `allenai/llama-3.1-tulu-3-8b-preference-mixture` and
  `allenai/llama-3.1-tulu-3-70b-preference-mixture` both load with the
  default `train` split and row shape `id`, `source`, `prompt`, `chosen`,
  and `rejected`; `chosen` and `rejected` are message lists.
- `HuggingFaceTB/smollm3-configs` is useful as a metadata/config-card source
  but did not load as dataset files in this probe.
- `HuggingFaceTB/smollm3-blueprint` is documentation/PDF-oriented; sample
  loading currently needs `pdfplumber`.
- Generic `HuggingFaceTB/smoltalk2` sample loading without an explicit config
  fails. The next implementation step is config/split-aware bounded probing.

Live config-specific SmolTalk2 probe:

- W&B run:
  `https://wandb.ai/phulin-self/slop-stage1/runs/wo6496wj`.
- Run name:
  `stage1-smollm3-smoltalk2-config-source-probe`.
- Local probe file:
  `artifacts/stage1/corpora/smollm3_tulu_config_source_probe.json`.
- Probe inputs were five exact SmolTalk2 dataset specs plus one model probe:
  `HuggingFaceTB/smoltalk2::Preference::llama_3.1_tulu_3_8b_preference_mixture_no_think`,
  `HuggingFaceTB/smoltalk2::Preference::tulu_3_8b_pref_mix_Qwen3_32B_Qwen3_0.6B_think`,
  `HuggingFaceTB/smoltalk2::SFT::smoltalk_smollm3_everyday_conversations_no_think`,
  `HuggingFaceTB/smoltalk2::SFT::OpenThoughts3_1.2M_think`,
  `HuggingFaceTB/smoltalk2::Mid::OpenThoughts3_1.2M`, and
  model `HuggingFaceTB/SmolLM3-3B`.
- Each dataset spec loaded successfully with 2 shape-only sample rows.
- Preference columns are `chosen`, `rejected`, `prompt`,
  `chat_template_kwargs`, and `source`. `chosen` and `rejected` are message
  lists; `chat_template_kwargs` contains `custom_instructions`,
  `enable_thinking`, `python_tools`, and `xml_tools`.
- SFT columns are `messages`, `chat_template_kwargs`, and `source`;
  `messages` are role/content message lists.
- Mid `OpenThoughts3_1.2M` columns are `messages` and `source`.
- This closes the generic-config loader issue and proves config/split-aware
  bounded probing works. It still does not finish Phase 1.

Live bounded SmolLM3 baseline samples:

- The pretraining collection URL is a Hugging Face collection, not a loadable
  dataset ID. Exact production pretraining feature rates therefore need
  per-source sampling plus recipe weighting rather than one direct dataset
  load.
- Bounded source-stratified proxy samples now exist:
  `artifacts/stage1/corpora/smollm3_pretrain_fineweb_edu_2k.jsonl`
  from `HuggingFaceFW/fineweb-edu`,
  `artifacts/stage1/corpora/smollm3_pretrain_dclm_2k.jsonl`
  from `mlfoundations/dclm-baseline-1.0` config `default`,
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_deu_2k.jsonl`
  from `HuggingFaceFW/fineweb-2` config `deu_Latn`,
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_spa_2k.jsonl`
  from `HuggingFaceFW/fineweb-2` config `spa_Latn`,
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_fra_2k.jsonl`
  from `HuggingFaceFW/fineweb-2` config `fra_Latn`,
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_ita_2k.jsonl`
  from `HuggingFaceFW/fineweb-2` config `ita_Latn`,
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_cmn_2k.jsonl`
  from `HuggingFaceFW/fineweb-2` config `cmn_Hani`,
  `artifacts/stage1/corpora/smollm3_pretrain_finemath_2k.jsonl`
  from `HuggingFaceTB/finemath` config `finemath-3plus`,
  `artifacts/stage1/corpora/smollm3_pretrain_pes2o_2k.jsonl`
  from `allenai/dolmino-mix-1124` config `pes2o`,
  `artifacts/stage1/corpora/smollm3_pretrain_stackexchange_apple_2k.jsonl`
  from `HuggingFaceTB/stackexchange_2025_md` config
  `apple.stackexchange.com`, and
  `artifacts/stage1/corpora/smollm3_smoltalk2_mid_llama_nemotron_reasoning_2k.jsonl`
  from `HuggingFaceTB/smoltalk2` config `Mid` split
  `Llama_Nemotron_Post_Training_Dataset_reasoning_r1`.
- The retained Tier-1 census is
  `artifacts/stage1/census/smollm3_pretrain_mid_baselines_2k_tier1_feature_rates.csv`.
  These rows are suitable as bounded baseline context in Phase 3, but they are
  not the exact weighted SmolLM3 11T pretraining mix.

Live SmolLM3 recipe-weight extraction:

- Added and ran `slop-extract-smollm3-config-weights` against the published
  `HuggingFaceTB/smollm3-configs` YAMLs:
  `stage3_9T_11T.yaml`, `long_context_4k_to_32k.yaml`, and
  `long_context_32k_to_64.yaml`.
- The extractor treats `stage3_9T_11T.yaml` as the cumulative 4k pretraining
  recipe, derives stage spans from `start_training_step`, computes tokens per
  step from the config batch geometry, and normalizes each segment's
  `dataset_weights`.
- Outputs:
  `artifacts/phase3/analysis/smollm3_config_source_weights_detail.csv`,
  `artifacts/phase3/analysis/smollm3_config_source_weights_aggregate.csv`,
  `artifacts/phase3/analysis/smollm3_config_source_weights_groups.csv`, and
  `artifacts/phase3/analysis/smollm3_config_source_weights_summary.md`.
- The extracted config-implied total is `11.234968T` tokens when the two
  long-context extension configs are included. Top aggregate sources are
  `dclm` (`35.486%`), `fineweb-edu` (`31.140%`), `fw2-deu` (`2.209%`),
  `fw2-spa` (`2.003%`), `stack-edu-Python` (`1.811%`), and `pes2o`
  (`1.724%`).
- The current bounded pretraining feature-rate samples cover `dclm`
  (`35.486%` exact config share), `fineweb-edu` (`31.140%`), `fw2-deu`
  (`2.209%`), `fw2-spa` (`2.003%`), `pes2o` (`1.724%`), `fw2-fra`
  (`1.607%`), `finemath` (`1.410%`), `fw2-ita` (`1.062%`), and
  `fw2-cmn` (`0.991%`), `fw2-rus` (`0.991%`), `fw2-por` (`0.931%`),
  `finemath-4plus` (`0.606%`), and `stackexchange` (`0.333%`). The remaining
  production baseline blocker is feature-rate coverage for the other recipe
  sources, not source-weight discovery.
- Added and ran `slop-assemble-weighted-pretrain-baseline` to join current
  sampled pretraining feature rates to the extracted recipe weights with
  explicit source maps. Outputs:
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy.csv`
  and
  `artifacts/phase3/analysis/smollm3_weighted_pretrain_baseline_coverage_proxy_summary.md`.
- Added a bounded `dclm` source sample from
  `mlfoundations/dclm-baseline-1.0`, config `default`, split `train`:
  `artifacts/stage1/corpora/smollm3_pretrain_dclm_2k.jsonl` and
  `artifacts/stage1/census/smollm3_pretrain_dclm_2k_tier1_feature_rates.csv`.
  The sample retained 2,000 rows and 1,780,498 simple tokens from a 20,000-row
  hash-reservoir scan.
- Added a bounded `fw2-deu` source sample from `HuggingFaceFW/fineweb-2`,
  config `deu_Latn`, split `train`:
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_deu_2k.jsonl` and
  `artifacts/stage1/census/smollm3_pretrain_fw2_deu_2k_tier1_feature_rates.csv`.
  The sample retained 2,000 rows and 986,326 simple tokens from a 20,000-row
  hash-reservoir scan.
- Added a bounded `fw2-spa` source sample from `HuggingFaceFW/fineweb-2`,
  config `spa_Latn`, split `train`:
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_spa_2k.jsonl` and
  `artifacts/stage1/census/smollm3_pretrain_fw2_spa_2k_tier1_feature_rates.csv`.
  The sample retained 2,000 rows and 1,271,133 simple tokens from a 20,000-row
  hash-reservoir scan.
- Added a bounded `pes2o` source sample from `allenai/dolmino-mix-1124`,
  config `pes2o`, split `train`:
  `artifacts/stage1/corpora/smollm3_pretrain_pes2o_2k.jsonl` and
  `artifacts/stage1/census/smollm3_pretrain_pes2o_2k_tier1_feature_rates.csv`.
  The sample retained 2,000 rows and 399,454 simple tokens from a 20,000-row
  hash-reservoir scan.
- Added a bounded `fw2-fra` source sample from `HuggingFaceFW/fineweb-2`,
  config `fra_Latn`, split `train`:
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_fra_2k.jsonl` and
  `artifacts/stage1/census/smollm3_pretrain_fw2_fra_2k_tier1_feature_rates.csv`.
  The sample retained 2,000 rows and 1,152,313 simple tokens from a 20,000-row
  hash-reservoir scan.
- Added a bounded `finemath` source sample from `HuggingFaceTB/finemath`,
  config `finemath-3plus`, split `train`:
  `artifacts/stage1/corpora/smollm3_pretrain_finemath_2k.jsonl` and
  `artifacts/stage1/census/smollm3_pretrain_finemath_2k_tier1_feature_rates.csv`.
  The sample retained 2,000 rows and 2,302,447 simple tokens from a
  20,000-row hash-reservoir scan.
- Added a bounded `fw2-ita` source sample from `HuggingFaceFW/fineweb-2`,
  config `ita_Latn`, split `train`:
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_ita_2k.jsonl` and
  `artifacts/stage1/census/smollm3_pretrain_fw2_ita_2k_tier1_feature_rates.csv`.
  The sample retained 2,000 rows and 1,053,366 simple tokens from a
  20,000-row hash-reservoir scan.
- Added a bounded `fw2-cmn` source sample from `HuggingFaceFW/fineweb-2`,
  config `cmn_Hani`, split `train`:
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_cmn_2k.jsonl` and
  `artifacts/stage1/census/smollm3_pretrain_fw2_cmn_2k_tier1_feature_rates.csv`.
  The sample retained 2,000 rows and 269,299 simple tokens from a 20,000-row
  hash-reservoir scan.
- Added a bounded `fw2-rus` source sample from `HuggingFaceFW/fineweb-2`,
  config `rus_Cyrl`, split `train`:
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_rus_2k.jsonl` and
  `artifacts/stage1/census/smollm3_pretrain_fw2_rus_2k_tier1_feature_rates.csv`.
  The sample retained 2,000 rows and 1,587,847 simple tokens from a
  20,000-row hash-reservoir scan.
- Added a bounded `fw2-por` source sample from `HuggingFaceFW/fineweb-2`,
  config `por_Latn`, split `train`:
  `artifacts/stage1/corpora/smollm3_pretrain_fw2_por_2k.jsonl` and
  `artifacts/stage1/census/smollm3_pretrain_fw2_por_2k_tier1_feature_rates.csv`.
  The sample retained 2,000 rows and 1,344,559 simple tokens from a
  20,000-row hash-reservoir scan.
- Added a bounded `finemath-4plus` source sample from
  `HuggingFaceTB/finemath`, config `finemath-4plus`, split `train`:
  `artifacts/stage1/corpora/smollm3_pretrain_finemath_4plus_2k.jsonl` and
  `artifacts/stage1/census/smollm3_pretrain_finemath_4plus_2k_tier1_feature_rates.csv`.
  The sample retained 2,000 rows and 1,774,598 simple tokens from a
  20,000-row hash-reservoir scan.
- Added a bounded `infiwebmath-4plus` source sample from
  `HuggingFaceTB/finemath`, config `infiwebmath-4plus`, split `train`:
  `artifacts/stage1/corpora/smollm3_pretrain_infiwebmath_4plus_2k.jsonl` and
  `artifacts/stage1/census/smollm3_pretrain_infiwebmath_4plus_2k_tier1_feature_rates.csv`.
  The sample retained 2,000 rows and 1,868,874 simple tokens from a
  20,000-row hash-reservoir scan, with retained strata `web_cc`, `forums_qa`,
  `wiki`, `code`, and `scientific`.
- Added bounded `megamath-text-code-block` and `megamath-web-pro` source
  samples from exact `LLM360/MegaMath` parquet subdirectories:
  `megamath-text-code-block/*.parquet` and `megamath-web-pro/*.parquet`.
  The text-code sample retained 2,000 rows and 844,480 simple tokens from a
  19,150-row hash-reservoir scan. The web-pro sample retained 2,000 rows and
  1,182,841 simple tokens from a 20,000-row hash-reservoir scan. The default
  `LLM360/MegaMath` dataset loader exposes metadata-like rows for the first
  data directory, so sampling uses the generic `parquet` loader plus
  `--hf-data-files` to target the exact source paths.
- With DCLM, FineWeb2 German, FineWeb2 Spanish, FineWeb2 French, FineWeb2
  Italian, FineWeb2 Chinese, FineWeb2 Russian, FineWeb2 Portuguese, FineMath,
  FineMath-4plus, InfiWebMath-4plus, MegaMath text-code-block, MegaMath
  web-pro, and PES2O included, the current retained Tier-1 proxy covers
  `82.259%` of the extracted recipe and leaves `17.741%`
  unsampled, so its covered-only rates are majority-coverage diagnostics
  rather than full-mixture estimates.

In-progress source-identification plan:

1. Use the extracted SmolLM3 recipe weights to decide the next source-sampling
   strata for a production weighted feature-rate baseline. Map each usable
   source to Phase 1 strata: web/CC, forums/Q&A, wiki/science, math/reasoning,
   and code-excluded audit bucket. Do not treat the collection listing alone
   as final mixture weights.
2. Run only bounded metadata probes before sampling: no full dataset downloads,
   fixed scan caps, streaming reads where available, and W&B logging of source
   counts, schema fields, split/config names, scan caps, seed, and failures.
3. Verify SmolTalk2 SFT config names, split names, chat schema, response role
   extraction, `think` versus `no_think` separation, and source/dataset
   provenance columns. For Phase 1 comparability, default the replication SFT
   target to assistant responses and keep reasoning traces explicitly tagged.
4. Verify the exact SmolLM3 APO preference source using SmolTalk2 `Preference`
   plus any linked Tulu-3 preference card/config. Keep
   `llama_3.1_tulu_3_8b_preference_mixture_no_think` separate from
   `tulu_3_8b_pref_mix_Qwen3_32B_Qwen3_0.6B_think`.
5. Treat the no-think Tulu preference mixture as the candidate RM/human-ranked
   preference source only after its original Tulu card/config confirms the
   construction. Treat the Qwen3-32B/Qwen3-0.6B `think` source as synthetic
   strong-vs-weak model-pair data, not human preference.
6. Block replication-ladder Result A claims until pair fields, chosen/rejected
   role semantics, pair IDs, source construction, and response extraction are
   all verified from primary cards or bounded schema probes.

Current interpretation:

- SmolLM3 exact recipe source weights are now extracted from the published
  configs. Bounded source-stratified proxy samples exist for Phase 3 context,
  but production pretraining-mixture feature-rate claims should wait for
  feature-rate coverage across the relevant weighted sources. The coverage
  proxy quantifies the current gap; DCLM, FineWeb2 German, FineWeb2 Spanish,
  FineWeb2 French, FineWeb2 Italian, FineWeb2 Chinese, FineWeb2 Russian,
  FineWeb2 Portuguese, FineMath, FineMath-4plus, InfiWebMath-4plus,
  MegaMath text-code-block, MegaMath web-pro, and PES2O are now sampled.
  `stack-edu-Python` (`1.811%`) is the largest unresolved source, but
  `HuggingFaceTB/stack-edu` config `Python` exposes blob metadata rather than
  code text, so measuring it needs a blob-hydration path or another
  text-bearing mirror. `stack-edu-Cpp` (`1.304%`) carries the same caveat; the
  next directly sampleable sources are smaller FineWeb2 language shards and
  other small exact recipe sources. The `infiwebmath` source row (`0.903%`)
  remains unmapped to an exact public `HuggingFaceTB/finemath` config; the
  public configs expose `infiwebmath-3plus` and `infiwebmath-4plus`, with
  `infiwebmath-4plus` now sampled for its exact `0.383%` source row.
- The live probes narrow SmolTalk2/Tulu source identification to specific
  configs and splits and verify config/split-aware loading. Remaining blockers
  are broader split/source count census, target response extraction and
  normalization, Tulu construction semantics, and weighted pretraining
  feature-rate coverage.
- SmolTalk2 provides the likely SFT and APO preference substrate for the
  replication ladder, but its SFT and Preference configs must be handled as
  multiple source/config strata rather than one homogeneous dataset.
- The synthetic Qwen3-32B/Qwen3-0.6B preference source should be analyzed as
  cross-model transmission, parallel to Delta-Learning interpretation, while
  the Tulu no-think preference mixture remains the candidate human/RM-ranked
  comparison once its construction is verified.

## Open Source-Verification Tasks

- Add a retained Dolma 3 stratification probe over a bounded sample and log
  `source` counts to W&B.
- Use DPO pair-analysis grouping by `preference_type`, `chosen_model`, and
  `rejected_model` in any Result A writeup, and keep aggregate Dolci DPO
  interpretations explicitly qualified as mixed-construction signal.
- Map Dolci SFT `source_dataset` and `domain` fields to a defensible
  human/synthetic/mixed provenance taxonomy, or avoid human-vs-synthetic
  claims in Phase 1.
- Complete bounded, W&B-logged SmolLM3 probes for weighted pretraining
  feature-rate coverage, SmolTalk2 SFT schema/configs, and Tulu/APO preference construction.
  Use primary Hugging Face metadata, dataset cards, and linked training recipes;
  avoid secondary summaries for source semantics.
