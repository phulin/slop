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

| Run | Activations | Latents | k | LR | Schedule | Epochs | Held-out MSE |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `ld4096_k3072_lr1e3_wsd_e16` | 1,000,000 | 4096 | 3072 | 0.001 | WSD | 16 | 0.210 |
| `ld4096_k2048_lr1e3_wsd_e16` | 1,000,000 | 4096 | 2048 | 0.001 | WSD | 16 | 0.222 |
| `ld4096_k1536_lr1e3_wsd_e16` | 1,000,000 | 4096 | 1536 | 0.001 | WSD | 16 | 0.236 |
| `ld4096_k1280_lr1e3_wsd_e16` | 1,000,000 | 4096 | 1280 | 0.001 | WSD | 16 | 0.241 |
| `ld8192_k1536_lr1e3_wsd_e16` | 1,000,000 | 8192 | 1536 | 0.001 | WSD | 16 | 0.288 |
| `ld4096_k1280_lr1e3_e16` | 1,000,000 | 4096 | 1280 | 0.001 | constant | 16 | 0.456 |
| `ld2048_k1280_lr1e3_e16` | 1,000,000 | 2048 | 1280 | 0.001 | constant | 16 | 0.474 |
| `ld2048_k1280_lr1e3_e8` | 1,000,000 | 2048 | 1280 | 0.001 | constant | 8 | 0.878 |
| `ld2048_k1536_lr1e3_e8` | 1,000,000 | 2048 | 1536 | 0.001 | constant | 8 | 0.878 |
| `ld8192_k768_lr4e3_wsd_e32` | 1,000,000 | 8192 | 768 | 0.004 | WSD | 32 | 1.474 |
| `ld8192_k768_lr4e3_wsd_e24` | 1,000,000 | 8192 | 768 | 0.004 | WSD | 24 | 1.538 |
| `ld2048_k1024_lr1e3_e8` | 1,000,000 | 2048 | 1024 | 0.001 | constant | 8 | 1.596 |
| `ld8192_k768_lr4e3_wsd_e16` | 1,000,000 | 8192 | 768 | 0.004 | WSD | 16 | 1.697 |
| `ld4096_k768_lr4e3_wsd_e16` | 1,000,000 | 4096 | 768 | 0.004 | WSD | 16 | 1.754 |
| `ld4096_k768_lr3e3_wsd_e16` | 1,000,000 | 4096 | 768 | 0.003 | WSD | 16 | 1.763 |
| `ld4096_k768_lr5e3_wsd_e16` | 1,000,000 | 4096 | 768 | 0.005 | WSD | 16 | 1.779 |
| `ld8192_k768_lr4e3_wsd_e48` | 1,000,000 | 8192 | 768 | 0.004 | WSD | 48 | 1.787 |
| `ld8192_k768_lr4e3_wsd_e40` | 1,000,000 | 8192 | 768 | 0.004 | WSD | 40 | 2.025 |
| `ld8192_k768_lr5e3_wsd_e16` | 1,000,000 | 8192 | 768 | 0.005 | WSD | 16 | 2.138 |
| `ld4096_k768_lr1e3_e16` | 1,000,000 | 4096 | 768 | 0.001 | constant | 16 | 2.350 |
| `ld2048_k768_lr1e3_e16` | 1,000,000 | 2048 | 768 | 0.001 | constant | 16 | 3.237 |
| `ld4096_k512_lr5e3_wsd_e64` | 1,000,000 | 4096 | 512 | 0.005 | WSD | 64 | 4.175 |
| `ld2048_k768_lr1e3_e8` | 1,000,000 | 2048 | 768 | 0.001 | constant | 8 | 4.239 |
| `ld4096_k512_lr5e3_wsd_e48` | 1,000,000 | 4096 | 512 | 0.005 | WSD | 48 | 4.263 |
| `ld4096_k512_lr5e3_wsd_e32` | 1,000,000 | 4096 | 512 | 0.005 | WSD | 32 | 4.342 |
| `ld4096_k512_lr5e3_wsd_e24` | 1,000,000 | 4096 | 512 | 0.005 | WSD | 24 | 4.380 |
| `ld2048_k768_lr1e3_wsd_e8` | 1,000,000 | 2048 | 768 | 0.001 | WSD | 8 | 4.544 |
| `ld4096_k512_lr5e3_wsd_e16` | 1,000,000 | 4096 | 512 | 0.005 | WSD | 16 | 4.614 |
| `ld8192_k512_lr5e3_wsd_e16` | 1,000,000 | 8192 | 512 | 0.005 | WSD | 16 | 4.632 |
| `ld4096_k512_lr4e3_wsd_e16` | 1,000,000 | 4096 | 512 | 0.004 | WSD | 16 | 4.651 |
| `ld4096_k512_lr6e3_wsd_e16` | 1,000,000 | 4096 | 512 | 0.006 | WSD | 16 | 4.657 |
| `ld4096_k512_lr3e3_wsd_e16` | 1,000,000 | 4096 | 512 | 0.003 | WSD | 16 | 4.731 |
| `ld4096_k512_lr2e3_wsd_e16` | 1,000,000 | 4096 | 512 | 0.002 | WSD | 16 | 4.904 |
| `ld2048_k768_lr2e3_e8` | 1,000,000 | 2048 | 768 | 0.002 | constant | 8 | 4.932 |
| `ld4096_k512_lr1e3_wsd_e16` | 1,000,000 | 4096 | 512 | 0.001 | WSD | 16 | 5.264 |
| `ld8192_k512_lr1e3_wsd_e16` | 1,000,000 | 8192 | 512 | 0.001 | WSD | 16 | 5.376 |
| `ld2048_k768_lr3e3_e8` | 1,000,000 | 2048 | 768 | 0.003 | constant | 8 | 5.746 |
| `ld4096_k512_lr5e4_wsd_e16` | 1,000,000 | 4096 | 512 | 0.0005 | WSD | 16 | 5.840 |
| `ld2048_k768_lr2e3_e4` | 1,000,000 | 2048 | 768 | 0.002 | constant | 4 | 6.627 |
| `ld2048_k512_lr1e3_wsd_e16` | 1,000,000 | 2048 | 512 | 0.001 | WSD | 16 | 6.630 |
| `ld4096_k384_lr1e3_wsd_e16` | 1,000,000 | 4096 | 384 | 0.001 | WSD | 16 | 7.675 |
| `ld2048_k512_lr1e3_wsd_e8` | 1,000,000 | 2048 | 512 | 0.001 | WSD | 8 | 7.932 |
| `ld2048_k512_lr1e3_e8` | 1,000,000 | 2048 | 512 | 0.001 | constant | 8 | 8.181 |
| `ld2048_k384_lr1e3_wsd_e16` | 1,000,000 | 2048 | 384 | 0.001 | WSD | 16 | 8.863 |
| `ld2048_k512_lr2e3_e8` | 1,000,000 | 2048 | 512 | 0.002 | constant | 8 | 9.083 |
| `ld2048_k512_lr2e3_e4` | 1,000,000 | 2048 | 512 | 0.002 | constant | 4 | 10.945 |
| `ld2048_k256_lr1e3_wsd_e16` | 1,000,000 | 2048 | 256 | 0.001 | WSD | 16 | 12.050 |

The larger cache lowers measured MSE substantially. The 1M-cache pure reconstruction boundary is now `artifacts/phase4/sae_wsd_sweep_1m/ld4096_k3072_lr1e3_wsd_e16`, but this is a dense upper-bound setting rather than an interpretable sparse model: `k=3072` allows 75% of the 4096-latent dictionary to be active, and the learned ReLU codes still kept roughly 2,600 positive entries per vector late in training. WSD improved high-k reconstruction: `4096/k1280` dropped from 0.456 with constant LR to 0.241 with WSD, then k1536, k2048, and k3072 edged the boundary down to 0.236, 0.222, and 0.210. Going wider to `8192/k1536/WSD` did not help and landed at 0.288.

Wider 4096-latent dictionaries helped at the more usable sparse settings: `4096/k768/e16` improved held-out MSE from 3.237 to 2.350 versus `2048/k768/e16`, and `4096/k512/WSD/e16` improved from 6.630 to 5.264 versus `2048/k512/WSD/e16` before LR tuning. Tuning WSD LR improved those further: `4096/k768/lr4e-3/WSD/e16` reached 1.754, and `4096/k512/lr5e-3/WSD/e16` reached 4.614. Going wider again to 8192 latents helped k768 modestly at the tuned LR, reaching 1.697, but did not help k512: `8192/k512/lr5e-3/WSD/e16` landed at 4.632 versus 4.614 for 4096 latents. Extending the current usable candidates continued to help up to different horizons: `8192/k768/lr4e-3/WSD` reached 1.538 at 24 epochs and 1.474 at 32 epochs, then regressed to 2.025 at 40 epochs and 1.787 at 48 epochs after mid-run instability; `4096/k512/lr5e-3/WSD` reached 4.380 at 24 epochs, 4.342 at 32 epochs, 4.263 at 48 epochs, and 4.175 at 64 epochs. The best weakly sparse reconstruction candidate remains `artifacts/phase4/sae_wsd_sweep_1m/ld8192_k768_lr4e3_wsd_e32`. The more conservative 1M-cache candidate is now `artifacts/phase4/sae_wsd_sweep_1m/ld4096_k512_lr5e3_wsd_e64`.

Lower-k frontier checks show the reconstruction cost of stricter sparsity: `2048/k384` lands at 8.863 MSE and `2048/k256` at 12.050 MSE after 16 epochs with WSD. A wider `4096/k384/WSD` run improves that lower-k frontier to 7.675. These are useful reference points, but the current tradeoff knee remains k512.

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
- `artifacts/phase4/sae_wsd_sweep_1m/ld4096_k512_lr5e3_wsd_e16_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld4096_k768_lr4e3_wsd_e16_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld8192_k768_lr4e3_wsd_e16_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld8192_k768_lr4e3_wsd_e24_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld4096_k512_lr5e3_wsd_e24_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld8192_k768_lr4e3_wsd_e32_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld4096_k512_lr5e3_wsd_e32_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld4096_k512_lr5e3_wsd_e48_scored`
- `artifacts/phase4/sae_wsd_sweep_1m/ld4096_k512_lr5e3_wsd_e64_scored`

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

Top 1M-cache 4096-latent k768/lr4e-3/WSD/e16 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 2154 | 0.5502 | 1.000 |
| 1334 | 0.2749 | 1.000 |
| 3065 | 0.2207 | 1.000 |
| 69 | 0.2151 | 1.000 |
| 883 | 0.1983 | 1.000 |

Top 1M-cache 8192-latent k768/lr4e-3/WSD/e16 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 5336 | 0.4418 | 1.000 |
| 7204 | 0.3726 | 1.000 |
| 2441 | 0.1989 | 1.000 |
| 7304 | 0.1647 | 0.978 |
| 2785 | 0.1360 | 0.995 |

Top 1M-cache 8192-latent k768/lr4e-3/WSD/e24 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 7204 | 0.4641 | 1.000 |
| 5974 | 0.2737 | 1.000 |
| 7304 | 0.2637 | 1.000 |
| 4771 | 0.2409 | 0.995 |
| 5298 | 0.1671 | 1.000 |

Top 1M-cache 8192-latent k768/lr4e-3/WSD/e32 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 7204 | 0.7105 | 1.000 |
| 5336 | 0.4874 | 1.000 |
| 2876 | 0.2619 | 1.000 |
| 4771 | 0.2495 | 1.000 |
| 7304 | 0.2236 | 1.000 |

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

Top 1M-cache 4096-latent k512/lr5e-3/WSD/e16 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 65 | 0.5728 | 1.000 |
| 3358 | 0.4489 | 1.000 |
| 2301 | 0.2192 | 1.000 |
| 1939 | 0.0896 | 0.878 |
| 2201 | 0.0436 | 0.938 |

Top 1M-cache 4096-latent k512/lr5e-3/WSD/e24 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 69 | 0.5399 | 0.995 |
| 3795 | 0.2858 | 1.000 |
| 132 | 0.2213 | 0.988 |
| 1295 | 0.1231 | 0.990 |
| 1939 | 0.1105 | 0.938 |

Top 1M-cache 4096-latent k512/lr5e-3/WSD/e32 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 3358 | 0.6268 | 1.000 |
| 1939 | 0.2910 | 1.000 |
| 2004 | 0.2534 | 1.000 |
| 2841 | 0.1870 | 1.000 |
| 3265 | 0.1182 | 0.953 |

Top 1M-cache 4096-latent k512/lr5e-3/WSD/e48 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 3581 | 0.5185 | 1.000 |
| 1346 | 0.4992 | 1.000 |
| 4030 | 0.2047 | 0.978 |
| 1939 | 0.1154 | 0.948 |
| 3936 | 0.0636 | 0.875 |

Top 1M-cache 4096-latent k512/lr5e-3/WSD/e64 AI-target latent effects:

| Latent | Mean target-logit drop when ablated | Positive effect rate |
| ---: | ---: | ---: |
| 911 | 0.5366 | 1.000 |
| 2089 | 0.2392 | 1.000 |
| 2301 | 0.1776 | 1.000 |
| 1346 | 0.1011 | 0.883 |
| 13 | 0.0728 | 0.998 |

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

The main reconstruction driver in the first sweep was `k`, but the 1M-cache follow-ups show that latent width matters once the run is trained long enough at a usable sparse `k`. Wider 4096-latent dictionaries improved both current production-shaped candidates: k768 fell from 3.237 to 2.350 MSE before LR tuning and to 1.754 after WSD LR tuning, while k512/WSD fell from 6.630 to 5.264 before LR tuning and to 4.614 after LR tuning. A further 8192-latent k768 run improved the weakly sparse frontier to 1.697 at 16 epochs, 1.538 at 24 epochs, and 1.474 at 32 epochs, but regressed at 40 and 48 epochs after mid-run instability. The analogous k512 run was slightly worse at 8192 latents, but extending the 4096/k512 candidate improved it to 4.380 at 24 epochs, 4.342 at 32 epochs, 4.263 at 48 epochs, and 4.175 at 64 epochs. Longer training helps, but the useful horizon is architecture-dependent. Pushing `k` past 768 continues to lower reconstruction loss, but by k1280 the SAE is already no longer meaningfully sparse; k2048 and k3072 should be treated only as dense reconstruction upper bounds.

LR `2e-3` was the best tested learning rate in the initial useful region. On the 1M-cache k768/e8 and k512/e8 follow-ups, `1e-3` beat `2e-3`; `3e-3` was unstable and worse by the final epoch. That conclusion did not hold after moving to 4096 latents, 16 epochs, and WSD: for `4096/k512/WSD`, LR `5e-4` undertrained at 5.840 MSE, `1e-3` reached 5.264, `2e-3` reached 4.904, `3e-3` reached 4.731, `4e-3` reached 4.651, `5e-3` reached 4.614, and `6e-3` backed off to 4.657. For `4096/k768/WSD`, LR `3e-3` reached 1.763, `4e-3` reached 1.754, and `5e-3` backed off to 1.779. The current best k768 setting is `8192/k768/lr4e-3/WSD/e32`; the current best k512 setting is `4096/k512/lr5e-3/WSD/e64`.

The 300k-cache scored runs have similar top positive detector-relevant latents, especially latent 862. On the 1M-cache scored runs, the tuned 4096 and 8192 candidates produce much larger top ablation effects than the earlier k512 and k768 candidates. The 32-epoch 8192/k768 run's top latents have mean target-logit drops of 0.710, 0.487, 0.262, 0.250, and 0.224, all with 1.000 positive effect rate. The 64-epoch 4096/k512 run's top latents drop the detector's AI-target logits by 0.537, 0.239, 0.178, 0.101, and 0.073 on average, with positive effect rates from 0.883 to 1.000. Tuned k768 now has both stronger reconstruction and strong detector-relevant latents, while tuned k512 remains the more conservative sparsity tradeoff.

The current recommended follow-up depends on the goal:

- For lowest reconstruction error: use the 1M-cache `4096/k3072/lr1e-3/wsd/e16` run as the current dense reconstruction boundary, but do not treat it as sparse or likely interpretable.
- For weakly sparse reconstruction: continue from the 1M-cache `8192/k768/lr4e-3/wsd/e32` region.
- For a better sparsity/reconstruction tradeoff: use the 1M-cache `4096/k512/lr5e-3/wsd/e64` run as the next production candidate.
- For interpretability: run larger latent scoring on the `k512` and `k768` candidates and compare whether high-k latents remain coherent enough to use.
