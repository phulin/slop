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
- Initial sweep summary: `artifacts/phase4/sae_wsd_sweep/sweep_results.csv`
- Larger-cache confirmation summary: `artifacts/phase4/sae_wsd_sweep_1m/sweep_results.csv`

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

## Larger-Cache Confirmation

The best two 300k-cache settings were rerun on a 1M-token activation cache built from up to 5,000 documents per source.

| Run | Activations | Latents | k | LR | Epochs | Held-out MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ld2048_k768_lr1e3_e8` | 1,000,000 | 2048 | 768 | 0.001 | 8 | 4.239 |
| `ld2048_k768_lr2e3_e8` | 1,000,000 | 2048 | 768 | 0.002 | 8 | 4.932 |
| `ld2048_k768_lr3e3_e8` | 1,000,000 | 2048 | 768 | 0.003 | 8 | 5.746 |
| `ld2048_k768_lr2e3_e4` | 1,000,000 | 2048 | 768 | 0.002 | 4 | 6.627 |
| `ld2048_k512_lr1e3_e8` | 1,000,000 | 2048 | 512 | 0.001 | 8 | 8.181 |
| `ld2048_k512_lr2e3_e8` | 1,000,000 | 2048 | 512 | 0.002 | 8 | 9.083 |
| `ld2048_k512_lr2e3_e4` | 1,000,000 | 2048 | 512 | 0.002 | 4 | 10.945 |

The larger cache preserves the same ranking and lowers measured MSE substantially. The 1M-cache pure reconstruction winner is now `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k768_lr1e3_e8`. The more conservative 1M-cache candidate is `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k512_lr1e3_e8`.

Scored 1M-cache outputs were also produced:

- `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k768_lr2e3_e8_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k768_lr1e3_e8_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k512_lr2e3_e8_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k512_lr1e3_e8_scored`

Top 1M-cache k768/lr1e-3 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 1464 | 0.1986 | 1.000 |
| 1733 | 0.1421 | 1.000 |
| 679 | 0.0695 | 1.000 |
| 717 | 0.0492 | 0.938 |
| 862 | 0.0264 | 1.000 |

Top 1M-cache k768/lr2e-3 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 1130 | 0.1199 | 0.965 |
| 1341 | 0.1095 | 0.998 |
| 862 | 0.0287 | 1.000 |
| 1189 | 0.0098 | 0.723 |
| 1520 | 0.0095 | 0.823 |

Top 1M-cache k512 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 1793 | 0.1943 | 1.000 |
| 96 | 0.1323 | 0.988 |
| 862 | 0.0281 | 1.000 |
| 183 | 0.0151 | 0.593 |
| 208 | 0.0113 | 0.983 |

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

The main reconstruction driver was `k`, not latent width. At fixed or similar training budgets, smaller 2048-latent models often beat wider 4096/8192 models once `k` was raised. Longer training also helped substantially: `2048/k512` improved from 24.351 MSE at 2 epochs to 12.704 at 8 epochs on the 300k cache, and the 1M-cache k512 run improved further to 8.181 after the LR was lowered to `1e-3`.

LR `2e-3` was the best tested learning rate in the initial useful region. On the 1M-cache k768/e8 and k512/e8 follow-ups, `1e-3` beat `2e-3`; `3e-3` was unstable and worse by the final epoch. LR `5e-4` undertrained badly in the initial k32 check.

The 300k-cache scored runs have similar top positive detector-relevant latents, especially latent 862. On the 1M-cache scored runs, the k512 candidate produces the largest top ablation effects while the k768 candidate keeps the best reconstruction loss. That makes k512 more attractive for interpretability despite weaker reconstruction.

The current recommended follow-up depends on the goal:

- For lowest reconstruction error: continue from the 1M-cache `2048/k768/lr1e-3/e8` region, but treat it as weakly sparse.
- For a better sparsity/reconstruction tradeoff: use the 1M-cache `2048/k512/lr1e-3/e8` run as the next production candidate.
- For interpretability: run larger latent scoring on the `k512` and `k768` candidates and compare whether high-k latents remain coherent enough to use.
