# Pangram EditLens Llama 3.2 SAE Smoke Results

Date: 2026-06-20

## Status

We completed an end-to-end smoke run for the Pangram/EditLens Llama SAE path:

1. Wrote the execution plan: `docs/experiments/pangram_llama_sae_plan.md`.
2. Found a usable public Transformers-format Llama-3.2-3B base mirror.
3. Loaded the Pangram `pangram/editlens_Llama-3.2-3B` PEFT adapter against that base.
4. Built an activation cache with document/token provenance.
5. Trained a BatchTopK SAE.
6. Scored and exported source-enriched latents plus distinct-document examples.

Primary output directory:

`artifacts/pangram_llama_sae/no_robots_cap640_smoke_unsloth_base/`

## Base Model Resolution

The configured `HF_TOKEN` loads correctly and can see the Meta model metadata, but the account is not authorized to download gated files from:

- `meta-llama/Llama-3.2-3B`
- `meta-llama/Llama-3.2-3B-Instruct`

The working replacement base is:

- `unsloth/Llama-3.2-3B`

This repo is public, non-gated, and provides standard Transformers safetensors:

- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`
- `config.json`

The Pangram adapter declares `base_model_name_or_path = meta-llama/Llama-3.2-3B`, and the Unsloth mirror has the matching Llama-3.2-3B architecture.

## Loader Fix

The Pangram adapter does not use the default `LlamaForSequenceClassification.score.weight` head. It saves a custom 4-way head:

- `score.norm.weight`
- `score.norm.bias`
- `score.linear.weight`, shape `(4, 3072)`

The loader now reconstructs this head before applying the PEFT adapter. Future bfloat16 runs also cast the restored score head and adapter consistently.

## Run Configuration

Command basis:

```bash
uv run slop-pangram-llama-sae \
  --base-model unsloth/Llama-3.2-3B \
  --local-model-dir artifacts/models/pangram_editlens_Llama-3.2-3B_unsloth_base \
  --local-files-only \
  --dataset-parquet artifacts/no_robots_batch_generations/categorical_openai_gemini_triplet_cap_640/data/train.parquet \
  --output-dir artifacts/pangram_llama_sae/no_robots_cap640_smoke_unsloth_base \
  --max-docs-per-source 256 \
  --max-length 512 \
  --collect-batch-size 8 \
  --max-activations 200000 \
  --latent-dim 4096 \
  --k 32 \
  --train-batch-size 4096 \
  --epochs 3 \
  --learning-rate 0.001 \
  --lr-schedule wsd \
  --torch-dtype float32 \
  --score-top-latents 50 \
  --max-examples-per-latent 20
```

Note: the actual run used `float32` to get past the adapter-head dtype mismatch before the loader fix. The code now supports the intended bfloat16 path.

## Artifacts

- Activation cache: `pangram_llama_sae_activation_cache.pt`
- SAE checkpoint: `pangram_llama_batchtopk_sae.pt`
- Train log: `pangram_llama_sae_train_log.csv`
- Eval log: `pangram_llama_sae_eval_log.csv`
- Ranked latents: `pangram_llama_sae_latents.csv`
- Max-activation examples: `pangram_llama_sae_examples.csv`
- Summary: `pangram_llama_sae_summary.json`

## Run Metrics

- Documents: `768`
  - human: `256`
  - GPT-5.5: `256`
  - Gemini 3.5 Flash: `256`
- Activation vectors: `120,714`
- Hidden size: `3072`
- SAE: `latent_dim=4096`, `k=32`
- Epochs: `3`
- Eval MSE:
  - epoch 1: `1.6358`
  - epoch 2: `1.3056`
  - epoch 3: `1.1801`
- Final train MSE: `1.1825`
- Latents exported: `50`
- Examples exported: `1,000`, 20 distinct-document examples per latent

Compile note: `torch.compile` fell back to eager because `libcuda.so` was not visible to Inductor. The run still completed. Future compiled runs should set `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH` or the repo's known CUDA library path before launch.

## Feature Families Observed

This is a smoke run, so these labels should be treated as preliminary. The SAE is small-cache and short-trained, and the ranking is simple AI-vs-human source enrichment. Still, several interpretable families are visible.

### 1. Polished Fiction / Poetic Imagery

Latents: `3603`, `1038`, `402`, `3299`, `776`

Examples include glowing memories, night imagery, dramatic scene transitions, fantasy/children's-story moments, and sensory phrases such as light, smoke, stars, wet walls, cobalt blue, and emotionally heightened motion. These look similar to the ModernBERT SAE's fiction/style latents, but are noisier.

Interpretation: the Pangram detector has directions that respond to smooth generated narrative prose and elevated sensory description.

### 2. Template Closings and Promotional Calls to Action

Latents: `1294`, `722`, partly `449`

Examples include business-letter placeholders, `Sincerely, [Your Name]`, branded CTA language, free-trial prompts, ticket/social hashtags, and polished relationship/greeting text.

Interpretation: the detector/SAE is picking up formulaic generated-output endings and templated marketing/correspondence scaffolds.

### 3. Educational / Procedural Scaffolding

Latents: `1680`, `3146`, partly `1223`

Examples include lesson-plan structure, "Part 1", "Goal:", "Pre-Reading & Vocabulary", timed activities, explanation of benefits, and structured prompts for students or users.

Interpretation: a clear generated-instruction feature family: explicit teaching/procedure scaffolds with labeled sections and purpose statements.

### 4. Explanatory Advice / Diagnostic Prose

Latents: `354`, `1223`, partly `590`

Examples include dental filling symptoms, sensitivity explanations, supplement/fitness explanations, and self-care/trigger-point advice.

Interpretation: detector features fire on polished explanatory advice, especially when it turns symptoms or practical guidance into confident, structured cause-effect prose.

### 5. QA, Trivia, and Enumerated Factual Snippets

Latents: `2093`, `782`, `1855`, `2770`, `1340`

Examples include quiz-style Q/A, named entities, years, awards, "who/which" factual snippets, and list fragments. This family is interpretable but less stylistic.

Interpretation: these are probably task-format artifacts in the no-robots sample rather than universal AI style.

## Cautions

The top-ranked latents are heavily Gemini-skewed in the examples:

- Gemini examples: `627`
- GPT-5.5 examples: `321`
- Human examples: `52`

That means the current scoring is finding AI-enriched directions, but some are probably source/model or task artifacts. The very top rows also include low-total-mass latents, so future ranking should combine log-odds with minimum mass, source balance, and probably ablation/first-order detector-score effect.

The run is useful as a proof of path, not as the final SAE analysis. It shows that the Pangram/EditLens Llama detector can be loaded, cached, SAE-trained, and feature-inspected. The most convincing feature classes are generated fiction/imagery, templated CTA/correspondence, educational/procedural scaffolding, and explanatory advice prose.

## Recommended Next Run

Run a larger bfloat16 cache now that the loader is fixed:

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
uv run slop-pangram-llama-sae \
  --base-model unsloth/Llama-3.2-3B \
  --local-model-dir artifacts/models/pangram_editlens_Llama-3.2-3B_unsloth_base \
  --local-files-only \
  --dataset-parquet artifacts/no_robots_batch_generations/categorical_openai_gemini_triplet_cap_640/data/train.parquet \
  --output-dir artifacts/pangram_llama_sae/no_robots_cap640_k64_bf16 \
  --max-docs-per-source 1024 \
  --max-length 512 \
  --collect-batch-size 16 \
  --max-activations 750000 \
  --latent-dim 4096 \
  --k 64 \
  --init-mode tied \
  --train-batch-size 8192 \
  --epochs 10 \
  --learning-rate 0.0005 \
  --lr-schedule wsd \
  --torch-dtype bfloat16 \
  --score-top-latents 100 \
  --max-examples-per-latent 50
```

