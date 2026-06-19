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
| `ld2048_k1280_lr1e3_e16` | 1,000,000 | 2048 | 1280 | 0.001 | 16 | 0.474 |
| `ld2048_k1280_lr1e3_e8` | 1,000,000 | 2048 | 1280 | 0.001 | 8 | 0.878 |
| `ld2048_k1536_lr1e3_e8` | 1,000,000 | 2048 | 1536 | 0.001 | 8 | 0.878 |
| `ld2048_k1024_lr1e3_e8` | 1,000,000 | 2048 | 1024 | 0.001 | 8 | 1.596 |
| `ld4096_k768_lr1e3_e16` | 1,000,000 | 4096 | 768 | 0.001 | 16 | 2.350 |
| `ld2048_k768_lr1e3_e16` | 1,000,000 | 2048 | 768 | 0.001 | 16 | 3.237 |
| `ld2048_k768_lr1e3_e8` | 1,000,000 | 2048 | 768 | 0.001 | 8 | 4.239 |
| `ld2048_k768_lr2e3_e8` | 1,000,000 | 2048 | 768 | 0.002 | 8 | 4.932 |
| `ld2048_k768_lr3e3_e8` | 1,000,000 | 2048 | 768 | 0.003 | 8 | 5.746 |
| `ld2048_k768_lr2e3_e4` | 1,000,000 | 2048 | 768 | 0.002 | 4 | 6.627 |
| `ld4096_k512_lr1e3_wsd_e16` | 1,000,000 | 4096 | 512 | 0.001 | 16 | 5.264 |
| `ld2048_k512_lr1e3_wsd_e16` | 1,000,000 | 2048 | 512 | 0.001 | 16 | 6.630 |
| `ld2048_k512_lr1e3_wsd_e8` | 1,000,000 | 2048 | 512 | 0.001 | 8 | 7.932 |
| `ld2048_k512_lr1e3_e8` | 1,000,000 | 2048 | 512 | 0.001 | 8 | 8.181 |
| `ld2048_k384_lr1e3_wsd_e16` | 1,000,000 | 2048 | 384 | 0.001 | 16 | 8.863 |
| `ld2048_k512_lr2e3_e8` | 1,000,000 | 2048 | 512 | 0.002 | 8 | 9.083 |
| `ld2048_k512_lr2e3_e4` | 1,000,000 | 2048 | 512 | 0.002 | 4 | 10.945 |
| `ld2048_k256_lr1e3_wsd_e16` | 1,000,000 | 2048 | 256 | 0.001 | 16 | 12.050 |

The larger cache lowers measured MSE substantially. The 1M-cache pure reconstruction boundary is now `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k1280_lr1e3_e16`, but this is a saturated high-k setting: increasing `k` to 1536 produced the same 8-epoch loss and similar active counts because the learned ReLU codes only kept about 1,250 positive entries per vector. Wider 4096-latent dictionaries helped at the more usable sparse settings: `4096/k768/e16` improved held-out MSE from 3.237 to 2.350 versus `2048/k768/e16`, and `4096/k512/WSD/e16` improved from 6.630 to 5.264 versus `2048/k512/WSD/e16`. The best weakly sparse reconstruction candidate is now `artifacts/phase4/sae_wsd_sweep_1m/ld4096_k768_lr1e3_e16`. The more conservative 1M-cache candidate is now `artifacts/phase4/sae_wsd_sweep_1m/ld4096_k512_lr1e3_wsd_e16`.

Lower-k frontier checks show the reconstruction cost of stricter sparsity: `k384` lands at 8.863 MSE and `k256` at 12.050 MSE after 16 epochs with WSD. These are useful reference points, but the current tradeoff knee remains k512.

Scored 1M-cache outputs were also produced:

- `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k768_lr2e3_e8_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k768_lr1e3_e8_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k768_lr1e3_e16_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k512_lr2e3_e8_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k512_lr1e3_e8_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k512_lr1e3_wsd_e8_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k512_lr1e3_wsd_e16_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k1280_lr1e3_e8_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld2048_k1280_lr1e3_e16_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld4096_k768_lr1e3_e16_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld4096_k512_lr1e3_wsd_e16_scored`

Top 1M-cache k1280/lr1e-3/e16 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 1485 | 0.1803 | 1.000 |
| 1733 | 0.1573 | 1.000 |
| 1644 | 0.0724 | 0.985 |
| 1922 | 0.0513 | 0.968 |
| 72 | 0.0448 | 0.893 |

Top 1M-cache k1280/lr1e-3/e8 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 1733 | 0.1535 | 1.000 |
| 461 | 0.0641 | 1.000 |
| 72 | 0.0444 | 0.883 |
| 139 | 0.0347 | 0.995 |
| 867 | 0.0301 | 0.973 |

Top 1M-cache k768/lr1e-3/e16 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 1130 | 0.2569 | 1.000 |
| 1464 | 0.2367 | 1.000 |
| 1438 | 0.1809 | 1.000 |
| 979 | 0.1428 | 1.000 |
| 1733 | 0.1421 | 1.000 |

Top 1M-cache 4096-latent k768/lr1e-3/e16 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 3501 | 0.2887 | 1.000 |
| 3658 | 0.1680 | 1.000 |
| 3065 | 0.1259 | 0.985 |
| 817 | 0.0139 | 0.765 |
| 3636 | 0.0123 | 0.640 |

Top 1M-cache k768/lr1e-3/e8 AI-target latent effects:

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

Top 1M-cache k512/WSD/e16 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 381 | 0.2649 | 1.000 |
| 750 | 0.0690 | 0.968 |
| 1512 | 0.0664 | 0.925 |
| 448 | 0.0419 | 0.743 |
| 77 | 0.0298 | 0.960 |

Top 1M-cache 4096-latent k512/WSD/e16 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 1939 | 0.1796 | 0.963 |
| 2841 | 0.1560 | 1.000 |
| 247 | 0.0710 | 0.940 |
| 13 | 0.0638 | 1.000 |
| 2089 | 0.0435 | 0.813 |

Top 1M-cache k512/WSD/e8 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 1171 | 0.0459 | 1.000 |
| 862 | 0.0255 | 1.000 |
| 507 | 0.0248 | 0.828 |
| 1841 | 0.0224 | 0.978 |
| 807 | 0.0134 | 0.988 |

Top 1M-cache k512/constant-lr AI-target latent effects:

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

The main reconstruction driver in the first sweep was `k`, but the 1M-cache follow-ups show that latent width matters once the run is trained long enough at a usable sparse `k`. Wider 4096-latent dictionaries improved both current production-shaped candidates: k768 fell from 3.237 to 2.350 MSE, and k512/WSD fell from 6.630 to 5.264 MSE. Longer training also helped substantially: `2048/k512` improved from 24.351 MSE at 2 epochs to 12.704 at 8 epochs on the 300k cache, and the 1M-cache k512/WSD run improved further to 6.630 after extending to 16 epochs. Pushing `k` past 768 continues to lower reconstruction loss, but by k1280 the SAE is no longer meaningfully sparse. Extending k1280 from 8 to 16 epochs lowered held-out MSE from 0.878 to 0.474, although the eval curve remained spiky.

LR `2e-3` was the best tested learning rate in the initial useful region. On the 1M-cache k768/e8 and k512/e8 follow-ups, `1e-3` beat `2e-3`; `3e-3` was unstable and worse by the final epoch. LR `5e-4` undertrained badly in the initial k32 check. A WSD schedule with 10% warmup and 10% decay slightly improved the k512 compromise from 8.181 to 7.932 MSE, but worsened k768 from 4.239 to 4.544 MSE.

The 300k-cache scored runs have similar top positive detector-relevant latents, especially latent 862. On the 1M-cache scored runs, the k512 candidate produces the largest top ablation effects while the k768 candidate keeps the best reconstruction loss. That makes k512 more attractive for interpretability despite weaker reconstruction.

The current recommended follow-up depends on the goal:

- For lowest reconstruction error: use the 1M-cache `2048/k1280/lr1e-3/e16` run as the current reconstruction boundary, but do not treat it as sparse or likely interpretable.
- For weakly sparse reconstruction: continue from the 1M-cache `4096/k768/lr1e-3/e16` region.
- For a better sparsity/reconstruction tradeoff: use the 1M-cache `4096/k512/lr1e-3/wsd/e16` run as the next production candidate.
- For interpretability: run larger latent scoring on the `k512` and `k768` candidates and compare whether high-k latents remain coherent enough to use.
