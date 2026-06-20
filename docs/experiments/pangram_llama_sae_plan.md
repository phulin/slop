# Pangram EditLens Llama 3.2 SAE Plan

Goal: train a BatchTopK SAE on Pangram's `pangram/editlens_Llama-3.2-3B` adapter over Llama-3.2-3B hidden states, then classify high-signal SAE features by source enrichment and max-activation examples.

## Target

- Base model: `meta-llama/Llama-3.2-3B`
- Adapter: `pangram/editlens_Llama-3.2-3B`
- Loader: `AutoModelForSequenceClassification` plus PEFT adapter
- Tokenizer/model auth: `HF_TOKEN` loaded from `.env` or process environment
- Activation site: final hidden state for non-special, non-padding tokens
- Primary dataset: `artifacts/no_robots_batch_generations/categorical_openai_gemini_triplet_cap_640/data/train.parquet`
- Initial scope: capped no-robots human/GPT/Gemini continuation slice, balanced by source via `--max-docs-per-source`

## Rationale

The Pangram/EditLens model is itself a detector-style Llama finetune. Instead of training the SAE on ModernBERT detector activations, this stretch path asks whether a detector based on a small Llama finetune exposes interpretable hidden-state directions for AI-vs-human style. The useful output is not just reconstruction quality; it is whether high-activation latents can be labeled with recognizable feature families.

## Execution Plan

1. Verify the command stack.
   - Use `uv run pytest tests/test_pangram_llama_sae.py tests/test_phase4_batchtopk_sae.py`.
   - Confirm `peft` is available with the project `models` extra.
   - Confirm `HF_TOKEN` can be loaded without printing it.

2. Build a first activation cache.
   - Run `slop-pangram-llama-sae` against the capped no-robots parquet.
   - Start with a small but useful cache: around 200k-500k token activations.
   - Use `--max-length 512` or `640`, `--collect-batch-size` as high as the A100 permits, and `--torch-dtype bfloat16`.
   - Save the cache with provenance so each activation maps back to `(doc_id, source, token_index)`.

3. Train the SAE.
   - Use BatchTopK with `latent_dim=4096`.
   - Start with `k=32` or `k=64`.
   - Use tied init if collapse appears; otherwise the initial smoke run can use the default.
   - Train a short smoke run first, then increase epochs if active-latent coverage and eval MSE look sane.

4. Score and classify features.
   - Rank latents by dense ReLU source enrichment: AI mass per AI token vs. human mass per human token.
   - Save `pangram_llama_sae_latents.csv` with AI/human log-odds, mass share, total mass, and max activation.
   - Save `pangram_llama_sae_examples.csv` with max-activation token windows for each ranked latent.
   - Interpret the top features by reading examples and grouping them into feature families.

5. Decide whether the path is promising.
   - Positive evidence: top latents have coherent examples, strong AI enrichment, and reusable families such as polished modifier bundles, catalog/list structure, procedural explanation, or synthetic fiction voice.
   - Negative evidence: top latents are mostly dataset artifacts, punctuation, single-token quirks, or source-specific topics with no stable stylistic interpretation.

## Initial Command

```bash
set -a
source .env
set +a
uv sync --extra models
uv run slop-pangram-llama-sae \
  --dataset-parquet artifacts/no_robots_batch_generations/categorical_openai_gemini_triplet_cap_640/data/train.parquet \
  --output-dir artifacts/pangram_llama_sae/no_robots_cap640_smoke \
  --max-docs-per-source 256 \
  --max-length 512 \
  --collect-batch-size 4 \
  --max-activations 200000 \
  --latent-dim 4096 \
  --k 32 \
  --train-batch-size 4096 \
  --epochs 3 \
  --learning-rate 0.001 \
  --lr-schedule wsd \
  --torch-dtype bfloat16 \
  --score-top-latents 50 \
  --max-examples-per-latent 20
```

## Expected Outputs

- `pangram_llama_sae_activation_cache.pt`
- `pangram_llama_batchtopk_sae.pt`
- `pangram_llama_sae_train_log.csv`
- `pangram_llama_sae_eval_log.csv`
- `pangram_llama_sae_latents.csv`
- `pangram_llama_sae_examples.csv`
- `pangram_llama_sae_summary.json`

