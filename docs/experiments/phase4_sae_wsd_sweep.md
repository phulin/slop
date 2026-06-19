# Phase 4 WSD Detector BatchTopK SAE Sweep

Date: 2026-06-19

## Setup

This sweep trained BatchTopK sparse autoencoders on penultimate-token activations from the capped no_robots categorical detector.

- Detector checkpoint: `artifacts/no_robots_batch_generations/lr_bs_sweep_cap640/lr2x_bs64_e2_wsd_checkpoint/checkpoint`
- Detector eval accuracy: 95.04% three-way author classification
- Activation corpus: capped no_robots triplets, 3,000 continuations per source (`human`, `gpt-5.5`, `gemini-3.5-flash`)
- Activation cache: `artifacts/phase4/sae_wsd_sweep/activation_cache_300k.pt`
- Activation vectors: 300,000 penultimate hidden states, dimension 1024
- SAE training split: 90% train / 10% held-out activation vectors
- Compile mode: `torch.compile(mode="reduce-overhead")`
- Sweep summary: `artifacts/phase4/sae_wsd_sweep/sweep_results.csv`

The first compiled detector attempt failed until `LD_LIBRARY_PATH` included `/usr/lib/x86_64-linux-gnu`, where `libcuda.so.1` is installed. After that, detector activation collection and SAE training both compiled successfully with no fallback.

## Best Results

| Run | Latents | k | LR | Epochs | Held-out MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ld2048_k768_lr2e3_e8` | 2048 | 768 | 0.002 | 8 | 9.604 |
| `ld2048_k1024_lr2e3_e4` | 2048 | 1024 | 0.002 | 4 | 11.801 |
| `ld2048_k512_lr2e3_e8` | 2048 | 512 | 0.002 | 8 | 12.704 |
| `ld2048_k768_lr2e3_e4` | 2048 | 768 | 0.002 | 4 | 13.648 |
| `ld2048_k512_lr2e3_e4` | 2048 | 512 | 0.002 | 4 | 16.744 |
| `ld2048_k256_lr2e3_e4` | 2048 | 256 | 0.002 | 4 | 20.972 |
| `ld2048_k128_lr2e3_e4` | 2048 | 128 | 0.002 | 4 | 27.166 |

The pure reconstruction winner is `2048` latents with `k=768`, LR `2e-3`, trained for 8 epochs. This is a high-active setting: 768 active latents out of 2048 per token, so it should be treated as a reconstruction-oriented upper bound rather than the most interpretable sparse configuration.

For a more conservative sparse setting, `2048/k512/lr2e-3/e8` is the best current compromise: held-out MSE 12.704 with 25% of latents active per token. The `2048/k1024/e4` boundary was tested as a half-active reconstruction lower-bound check; it did not beat `k768/e8`.

## Scored SAE Output

Scored runs were produced for the reconstruction winner and the more conservative sparse candidate:

- Reconstruction winner output: `artifacts/phase4/sae_wsd_sweep/ld2048_k768_lr2e3_e8_scored`
- Conservative candidate output: `artifacts/phase4/sae_wsd_sweep/ld2048_k512_lr2e3_e8_scored`
- Ranked latents: 16 per scored run
- Example rows: 48 for k768, 64 for k512
- Scoring docs: 200 per source, with ablations scored on the 400 AI continuations
- Target detector classes: `gemini-3.5-flash` and `gpt-5.5`

Top positive AI-target latent effects for the reconstruction winner:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 862 | 0.0337 | 1.000 |
| 139 | 0.0318 | 0.995 |
| 1530 | 0.0168 | 0.985 |
| 208 | 0.0151 | 1.000 |
| 1227 | 0.0150 | 0.935 |

Top positive AI-target latent effects for the conservative `k512` candidate:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 862 | 0.0321 | 1.000 |
| 208 | 0.0147 | 1.000 |
| 1530 | 0.0138 | 0.978 |
| 1520 | 0.0132 | 0.875 |
| 1227 | 0.0127 | 0.940 |

## Conclusions

The main reconstruction driver was `k`, not latent width. At fixed or similar training budgets, smaller 2048-latent models often beat wider 4096/8192 models once `k` was raised. Longer training also helped substantially: `2048/k512/lr2e-3` improved from 24.351 MSE at 2 epochs to 12.704 at 8 epochs.

LR `2e-3` was the best tested learning rate in the useful region. LR `5e-4` undertrained badly; LR `3e-3` was worse than `2e-3` for the same 2048/k512/e4 setting.

The scored runs have similar top positive detector-relevant latents, especially latent 862. The high-k winner improves reconstruction, but the top detector-effect signal is not obviously stronger than the k512 model on this small scoring pass.

The current recommended follow-up depends on the goal:

- For lowest reconstruction error: continue from the `2048/k768/lr2e-3/e8` region, but treat it as weakly sparse.
- For a better sparsity/reconstruction tradeoff: use `2048/k512/lr2e-3/e8` as the next production candidate.
- For interpretability: run larger latent scoring on the `k512` and `k768` candidates and compare whether high-k latents remain coherent enough to use.
