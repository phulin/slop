# Pangram SAE Lexical Context Analysis

This compares corpus-derived target rows against short independent minimal pairs for the two lexical SAE paths.

## Group Summary

| feature | group | variant | words_mean | feature_hits_mean | feature_hits_per_100_words_mean | upstream_baseline_mass_mean | downstream_baseline_mass_mean | downstream_mass_delta_mean | target_logit_drop_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| discourse_marker | combined_long_stress | marked | 374.000 | 16.000 | 4.278 | 0.592 | 0.000 | 0.000 | 0.008 |
| discourse_marker | combined_long_stress | plain | 352.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| discourse_marker | counterfactual | edited | 377.062 | 0.000 | 0.000 | 12.591 | 14.454 | -11.504 | 0.037 |
| discourse_marker | counterfactual | original | 378.562 | 4.625 | 1.286 | 21.769 | 27.880 | -22.649 | 0.043 |
| discourse_marker | heldout |  | 387.375 | 4.417 | 1.201 | 22.828 | 28.867 | -23.800 | 0.047 |
| discourse_marker | long_stress | marked | 124.667 | 5.333 | 4.345 | 1.302 | 0.000 | 0.000 | 0.005 |
| discourse_marker | long_stress | plain | 117.333 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| discourse_marker | minimal_pair | marked | 28.500 | 1.000 | 3.572 | 0.266 | 0.000 | 0.000 | -0.000 |
| discourse_marker | minimal_pair | plain | 27.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| discourse_marker | top_export |  | 325.053 | 1.526 | 0.532 | 29.101 | 39.094 | -31.164 | 0.049 |
| negation_fragment | combined_long_stress | marked | 357.000 | 33.000 | 9.244 | 67.078 | 89.912 | -71.782 | 0.031 |
| negation_fragment | combined_long_stress | plain | 392.000 | 0.000 | 0.000 | 30.742 | 32.629 | -25.980 | 0.000 |
| negation_fragment | counterfactual | edited | 416.562 | 0.000 | 0.000 | 16.457 | 17.023 | -15.187 | 0.093 |
| negation_fragment | counterfactual | original | 408.500 | 8.062 | 2.043 | 28.062 | 33.826 | -28.748 | 0.116 |
| negation_fragment | heldout |  | 419.708 | 7.667 | 1.882 | 28.055 | 33.394 | -28.674 | 0.114 |
| negation_fragment | long_stress | marked | 119.000 | 11.000 | 9.273 | 15.533 | 22.938 | -15.858 | 0.052 |
| negation_fragment | long_stress | plain | 130.667 | 0.000 | 0.000 | 3.934 | 5.413 | -3.487 | 0.031 |
| negation_fragment | minimal_pair | marked | 20.750 | 2.000 | 10.006 | 0.175 | 0.000 | 0.000 | -0.000 |
| negation_fragment | minimal_pair | plain | 22.750 | 0.000 | 0.000 | 0.066 | 0.000 | 0.000 | 0.000 |
| negation_fragment | top_export |  | 403.400 | 3.350 | 0.873 | 33.935 | 39.626 | -35.184 | 0.198 |

## Highest Downstream Activation Examples

### discourse_marker

- `top_export` `` `af34df515a680fa9` model `human` downstream `85.06` upstream `60.94` words `391` hits `2`
  - Dear graduates, distinguished faculty members, and esteemed guests, it is an incredible honor to be here today to address this talented group of graduates. Congratulations on your significant achievement, and thank yo...
- `heldout` `` `d84183a3e51649a1` model `qwen3.6-35b` downstream `70.43` upstream `49.11` words `368` hits `5`
  - The use of standardized assessments in primary education remains a subject of intense debate among educators, policymakers, and parents. As young learners transition into formal schooling, these testing instruments ar...
- `counterfactual` `original` `d84183a3e51649a1` model `qwen3.6-35b` downstream `70.43` upstream `49.11` words `368` hits `5`
  - The use of standardized assessments in primary education remains a subject of intense debate among educators, policymakers, and parents. As young learners transition into formal schooling, these testing instruments ar...
- `top_export` `` `0403673ff6e55bea` model `qwen3.6-35b` downstream `62.68` upstream `48.79` words `419` hits `4`
  - Three-dimensional printing, commonly referred to as additive manufacturing, has emerged as a transformative technology for aerospace engineers, redefining how aircraft and spacecraft are designed, prototyped, and prod...
- `top_export` `` `d0917480d3aa31da` model `gpt-5.5` downstream `62.37` upstream `44.11` words `349` hits `1`
  - Why NASA Was Founded—and What It Has Accomplished NASA, the National Aeronautics and Space Administration, was founded in 1958 during a tense moment in world history. The Soviet Union had launched Sputnik, the first a...
- `counterfactual` `original` `e079aec935477190` model `gpt-5.5` downstream `61.59` upstream `44.81` words `344` hits `5`
  - Pollution: Causes, Consequences, and Solutions Pollution is one of the most serious problems facing the world today. It occurs when harmful substances enter the environment and damage air, water, soil, and living orga...
- `heldout` `` `e079aec935477190` model `gpt-5.5` downstream `61.59` upstream `44.81` words `344` hits `5`
  - Pollution: Causes, Consequences, and Solutions Pollution is one of the most serious problems facing the world today. It occurs when harmful substances enter the environment and damage air, water, soil, and living orga...
- `top_export` `` `9d45422dc29eb8a1` model `glm-5.2` downstream `55.04` upstream `43.07` words `599` hits `1`
  - Mountain climbing is an incredibly rewarding pursuit that offers breathtaking views, a profound sense of accomplishment, and a unique way to experience the natural world. However, venturing into high altitudes require...

### negation_fragment

- `combined_long_stress` `marked` `combined_negatio` model `combined_long_stress_pair` downstream `89.91` upstream `67.08` words `357` hits `33`
  - Lena didn't trust the inventory report, and she didn't hide that fact when the warehouse manager asked why she was still at her desk. The totals didn't match the pallets by the loading door, the return bin wasn't labe...
- `heldout` `` `71a3c948431c1d48` model `gpt-5.5` downstream `68.41` upstream `54.34` words `455` hits `7`
  - Rachel kept her smile pinned in place as Mr. Hale slid the folder across his desk. “The Vanders’ estate?” she asked, already hearing the dull thud of society pages and ribbon-cutting ceremonies. “You want me to cover...
- `top_export` `` `f5b3db16770918aa` model `qwen3.6-35b` downstream `58.76` upstream `45.09` words `433` hits `8`
  - The heavy iron gates of Wisteria groaned as Dan swung them open. Sylvie stood on the other side, mud caked on her boots, eyes sharp as broken glass. She didn’t smile. Didn’t nod. Just stared at him like he was a ghost...
- `top_export` `` `df5fb92fe06685a8` model `qwen3.6-35b` downstream `57.39` upstream `49.29` words `488` hits `5`
  - The alarm went off at 6:00 AM, screaming like it had personal grievances against me. I slapped it off, hoping for just five more minutes, but when I opened my eyes, it was already 6:15. Great. Just perfect. I stumbled...
- `top_export` `` `eacb0cf544de2210` model `qwen3.6-35b` downstream `53.05` upstream `43.42` words `568` hits `3`
  - The neon hum of the Cyber-Sip Bistro buzzed like an angry hornet trapped in glass. Rain lashed against the panoramic window, blurring the sprawling, acid-green sprawl of Sector 4 outside. Elara sat alone at a corner t...
- `counterfactual` `original` `473ee77815b33268` model `qwen3.6-35b` downstream `52.25` upstream `43.11` words `458` hits `9`
  - The rain on Lothalo didn’t wash things clean; it just made the grime slicker, reflecting the neon glow of the slums in oily puddles. I, Goreth, a Chagrian with four horns and skin the color of old parchment, stood hud...
- `heldout` `` `473ee77815b33268` model `qwen3.6-35b` downstream `52.25` upstream `43.11` words `458` hits `9`
  - The rain on Lothalo didn’t wash things clean; it just made the grime slicker, reflecting the neon glow of the slums in oily puddles. I, Goreth, a Chagrian with four horns and skin the color of old parchment, stood hud...
- `top_export` `` `f18bba3d96f77b75` model `gpt-5.5` downstream `49.40` upstream `38.89` words `438` hits `2`
  - The first week of sophomore year smells like rain on hot pavement, burnt coffee, and the sharp bite of new textbooks. You tell yourself you’re different now. You are not the same girl who cried in the third-floor bath...

## Negation Density Sweep

I built `artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_targets.csv` from one fixed narrative frame and introduced the first N contractions in order. This keeps plot and wording mostly fixed while varying contraction count.

| Variant | Contractions | Words | Upstream mass | Downstream mass | Downstream mass delta | Class-3 logit drop |
|---|---:|---:|---:|---:|---:|---:|
| `contract_00` | 0 | 392 | `30.69` | `32.41` | `-25.86` | `0.0000` |
| `contract_02` | 2 | 390 | `27.85` | `29.29` | `-22.58` | `0.0625` |
| `contract_04` | 4 | 388 | `26.30` | `29.28` | `-23.98` | `0.0625` |
| `contract_08` | 8 | 384 | `25.96` | `31.26` | `-24.59` | `0.0000` |
| `contract_16` | 16 | 376 | `39.14` | `52.78` | `-42.16` | `0.0938` |
| `contract_32` | 32 | 360 | `60.02` | `78.91` | `-63.22` | `0.0312` |

Correlations with contraction count:

- `upstream_baseline_mass`: `0.94`
- `downstream_baseline_mass`: `0.97`
- `downstream_mass_delta`: `-0.97`
- `target_logit_drop`: `0.13`

Interpretation:

- `L19:913 -> L24:1310` shows a clean activation-level dose response to contraction density after a residual baseline from expanded negation syntax.
- The detector target score does not show a similarly clean dose response in this synthetic frame. This supports the same conclusion as the score-control panel: the path is a real latent circuit candidate, but it is not by itself a simple scalar readout circuit for the class-3 score.

## Discourse Token Loci

I ran the same token-locus inspection for `L19:3450 -> L24:3469` on held-out discourse-marker rows, the local counterfactual rows, and the synthetic long-stress rows.

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/heldout_discourse_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/heldout_discourse_token_loci_summary.csv`
- `docs/experiments/pangram_sae_heldout_discourse_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_discourse_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_discourse_token_loci_summary.csv`
- `docs/experiments/pangram_sae_counterfactual_discourse_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/long_stress_discourse_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/combined_long_stress_discourse_token_loci_details.csv`

Held-out real documents:

| Node | Docs | Active tokens | Mean mass/doc | Max activation | Top-token class mix |
|---|---:|---:|---:|---:|---|
| `L19:3450` | 24 | 173 | `22.81` | `10.10` | word `57.8%`, discourse marker `30.6%`, sentence boundary `9.2%` |
| `L24:3469` | 24 | 173 | `28.81` | `13.19` | word `59.0%`, discourse marker `30.1%`, sentence boundary `7.5%` |

Counterfactual rows:

| Node | Docs | Active tokens | Mean mass/doc | Max activation | Top-token class mix |
|---|---:|---:|---:|---:|---|
| `L19:3450` | 32 | 207 | `17.17` | `10.10` | word `57.7%`, discourse marker `21.4%`, sentence boundary `10.7%`, style/function word `6.6%` |
| `L24:3469` | 32 | 200 | `21.12` | `13.19` | word `62.6%`, discourse marker `21.1%`, sentence boundary `11.1%` |

I then split the held-out counterfactuals into targeted edits:

| Edit | Base docs | Upstream mass delta | Downstream mass delta | Downstream ablation delta change | Target-logit-drop change |
|---|---:|---:|---:|---:|---:|
| remove transition markers only | 22 | `-7.34` | `-9.01` | `+8.05` | `-0.0075` |
| replace `such as` with `including` | 21 | `-4.39` | `-6.53` | `+5.02` | `+0.0039` |
| remove transitions and replace `such as` | 22 | `-11.46` | `-15.00` | `+12.86` | `-0.0099` |

Here `downstream ablation delta change` is edited minus original for the `L19:3450` ablation effect on `L24:3469`. Positive values mean the downstream latent had less remaining mass to suppress after the edit.

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_discourse_transition_removed_targets.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_discourse_transition_removed_intervention_L19_3450_to_L24_3469.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_discourse_such_as_replaced_targets.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_discourse_such_as_replaced_intervention_L19_3450_to_L24_3469.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_discourse_transition_and_such_targets.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_discourse_transition_and_such_intervention_L19_3450_to_L24_3469.csv`

Synthetic long-stress rows:

| Set | Node | Docs | Active tokens | Mean mass/doc | Max activation |
|---|---|---:|---:|---:|---:|
| long stress | `L19:3450` | 6 | 4 | `0.65` | `2.01` |
| long stress | `L24:3469` | 6 | 0 | `0.00` | `0.00` |
| combined long stress | `L19:3450` | 2 | 1 | `0.30` | `0.59` |
| combined long stress | `L24:3469` | 2 | 0 | `0.00` | `0.00` |

Interpretation:

- The discourse path is marker-associated but not a pure marker counter. In real held-out documents, roughly 30% of top token peaks are explicit markers such as `However`, `Furthermore`, `Additionally`, and `Moreover`, but the majority are ordinary content words in the same polished explanatory scaffold, especially `such`, `amounts`, `type`, `associated`, and section-closing periods.
- Local counterfactual edits lower the path's activation mass, so the markers are part of the feature. The split edits show additive evidence from both sentence-level transition markers and `such as` example scaffolding. However, marker-dense synthetic stress text does not recover the downstream `L24:3469` activation at all. This means the downstream node likely requires corpus-realized expository context: paragraph role, generic explanatory semantics, and local marker use together.
- Next test should be generated expository minimal pairs that preserve the full corpus style while varying only markers and `such as` constructions. Short independent minimal pairs and hand-written stress passages are too far from the activation manifold for this path.

## Modal / Function-Word Candidate

I inspected `L19:262 -> L24:1664`, previously labeled modal/simple-explanatory function wording.

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_modal_function_L19_262_to_L24_1664.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_modal_function_L19_262_to_L24_1664.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/modal_function_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/modal_function_token_loci_summary.csv`
- `docs/experiments/pangram_sae_modal_function_token_loci.md`

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Max activation | Top-token class mix |
|---|---:|---:|---:|---:|---:|---|
| `L19:262` | 14 | 5,618 | `401.29` | `1879.47` | `12.19` | word `78.6%`, style/function word `12.1%`, expanded-negation/function auxiliaries `3.9%` |
| `L24:1664` | 14 | 5,616 | `401.14` | `3208.64` | `18.47` | word `79.3%`, style/function word `13.2%`, expanded-negation/function auxiliaries `3.2%` |

The direct intervention is large: ablating `L19:262` drops `L24:1664` downstream mass by `-2965.80` on average and drops the class-3 logit by `0.6024`. But token loci show that both nodes are active on almost every non-special token in these documents. Top tokens include `can`, `is`, `to`, `it`, `are`, `feel`, `they`, `still`, `people`, `water`, and punctuation-adjacent ordinary words.

Interpretation:

- This is probably a broad detector-supporting residual direction for simple explanatory or classroom-style prose, not a crisp lexical/style circuit.
- It should remain in the registry as an entangled positive-control or bookkeeping path, but it should be lower priority than `LEX-1` and `LEX-2` for human-readable slop indicators.
- Any future claim about this path needs same-mass controls and token-position controls, because the raw downstream mass effect is dominated by document-wide activation density.
