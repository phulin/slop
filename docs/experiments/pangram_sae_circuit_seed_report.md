# Pangram SAE Circuit Seed Report

This is an initial, low-cost circuit-discovery pass over existing latent explorer exports. It does not claim mechanistic circuits yet; it identifies cross-layer SAE latent chains worth deeper intervention tests.

For the compact current queue, see `docs/experiments/pangram_sae_circuit_candidate_registry.md`.

## Inputs

- Layer 16: `apps/latent-explorer/public/data/layer16/pangram_llama_sae_browser_index.json`
- Layer 19: `apps/latent-explorer/public/data/layer19/pangram_llama_sae_browser_index.json`
- Layer 20: `apps/latent-explorer/public/data/layer20/pangram_llama_sae_browser_index.json`
- Layer 24: `apps/latent-explorer/public/data/layer24/pangram_llama_sae_browser_index.json`

## Method

- Keep the top causal/reranked latents per layer.
- For each latent, use its displayed high-impact prompt IDs, model mix, and signed prompt-contrast logit drops.
- Connect adjacent-layer latents when they fire on overlapping high-impact prompts.
- Rank edges by shared prompts, prompt-set Jaccard, model-impact profile cosine, and signed-direction agreement.
- Greedily assemble layer 16 -> 19 -> 20 -> 24 chains from high-scoring adjacent edges.

## Layer Inventory

### Layer 16

- `L16:1983` rank 1, max abs impact 4.359, mean signed impact -2.242, prompts 50, models accounts/fireworks/models/glm-5p2:8, gemini-3.5-flash:10, gpt-5.5:29, human:3.
- `L16:2296` rank 2, max abs impact 1.473, mean signed impact 0.330, prompts 50, models accounts/fireworks/models/glm-5p2:3, gpt-5.5:1, human:46.
- `L16:2120` rank 3, max abs impact 1.133, mean signed impact 0.349, prompts 50, models accounts/fireworks/models/glm-5p2:8, gemini-3.5-flash:4, gpt-5.5:6, human:32.
- `L16:2943` rank 4, max abs impact 1.438, mean signed impact -0.614, prompts 50, models accounts/fireworks/models/glm-5p2:15, gemini-3.5-flash:10, gpt-5.5:20, human:5.
- `L16:3074` rank 5, max abs impact 1.000, mean signed impact -0.561, prompts 50, models human:50.

### Layer 19

- `L19:2943` rank 1, max abs impact 3.227, mean signed impact -1.669, prompts 50, models gemini-3.5-flash:2, gpt-5.5:2, qwen3.6-35b:46.
- `L19:2687` rank 2, max abs impact 3.072, mean signed impact -1.917, prompts 50, models gemini-3.5-flash:3, glm-5.2:4, human:1, qwen3.6-35b:42.
- `L19:2346` rank 3, max abs impact 1.656, mean signed impact -0.757, prompts 50, models gemini-3.5-flash:1, glm-5.2:3, gpt-5.5:2, human:41, qwen3.6-35b:3.
- `L19:3990` rank 4, max abs impact 1.510, mean signed impact 0.469, prompts 50, models gpt-5.5:8, human:42.
- `L19:2301` rank 5, max abs impact 1.289, mean signed impact 0.334, prompts 50, models gemini-3.5-flash:6, glm-5.2:11, gpt-5.5:18, human:4, qwen3.6-35b:11.

### Layer 20

- `L20:3553` rank 1, max abs impact 3.934, mean signed impact -2.389, prompts 50, models gemini-3.5-flash:12, glm-5.2:5, gpt-5.5:1, qwen3.6-35b:32.
- `L20:3795` rank 2, max abs impact 3.391, mean signed impact -2.031, prompts 50, models gemini-3.5-flash:6, glm-5.2:3, gpt-5.5:4, qwen3.6-35b:37.
- `L20:3702` rank 3, max abs impact 1.367, mean signed impact 0.310, prompts 50, models gemini-3.5-flash:9, glm-5.2:9, gpt-5.5:6, human:21, qwen3.6-35b:5.
- `L20:741` rank 4, max abs impact 1.281, mean signed impact 0.569, prompts 50, models gemini-3.5-flash:6, glm-5.2:3, gpt-5.5:5, human:35, qwen3.6-35b:1.
- `L20:1210` rank 5, max abs impact 2.252, mean signed impact 0.696, prompts 50, models gpt-5.5:7, human:43.

### Layer 24

- `L24:741` rank 1, max abs impact 1.533, mean signed impact 0.739, prompts 50, models gemini-3.5-flash:3, glm-5.2:1, human:46.
- `L24:4086` rank 2, max abs impact 1.727, mean signed impact -1.089, prompts 50, models gemini-3.5-flash:5, glm-5.2:6, human:3, qwen3.6-35b:36.
- `L24:2919` rank 3, max abs impact 4.117, mean signed impact -2.475, prompts 50, models gemini-3.5-flash:2, glm-5.2:2, gpt-5.5:1, qwen3.6-35b:45.
- `L24:2940` rank 4, max abs impact 0.961, mean signed impact 0.103, prompts 50, models gemini-3.5-flash:17, glm-5.2:10, gpt-5.5:4, human:11, qwen3.6-35b:8.
- `L24:3889` rank 5, max abs impact 1.369, mean signed impact -0.582, prompts 50, models gemini-3.5-flash:3, glm-5.2:6, gpt-5.5:7, human:31, qwen3.6-35b:3.

## Top Cross-Layer Edges

- `L19:491` -> `L20:3178`: score 27.839, shared prompts 30, Jaccard 0.429, model cosine 1.000, signed cosine 1.000.
- `L19:1895` -> `L20:2864`: score 27.481, shared prompts 30, Jaccard 0.429, model cosine 0.991, signed cosine 0.990.
- `L19:1066` -> `L20:590`: score 26.248, shared prompts 29, Jaccard 0.408, model cosine 0.997, signed cosine 0.998.
- `L20:1210` -> `L24:1120`: score 24.564, shared prompts 28, Jaccard 0.389, model cosine 0.991, signed cosine 0.991.
- `L19:2355` -> `L20:3219`: score 22.554, shared prompts 27, Jaccard 0.370, model cosine 0.987, signed cosine 0.946.
- `L19:2943` -> `L20:3795`: score 22.313, shared prompts 28, Jaccard 0.389, model cosine 0.929, signed cosine 0.929.
- `L19:1634` -> `L20:3514`: score 22.301, shared prompts 27, Jaccard 0.370, model cosine 0.967, signed cosine 0.964.
- `L19:3600` -> `L20:2433`: score 21.767, shared prompts 28, Jaccard 0.389, model cosine 0.929, signed cosine 0.882.
- `L19:3009` -> `L20:2077`: score 20.754, shared prompts 25, Jaccard 0.333, model cosine 0.998, signed cosine 0.995.
- `L19:4063` -> `L20:1373`: score 19.579, shared prompts 24, Jaccard 0.316, model cosine 1.000, signed cosine 1.000.
- `L19:3990` -> `L20:1210`: score 19.289, shared prompts 24, Jaccard 0.316, model cosine 0.990, signed cosine 0.990.
- `L19:240` -> `L20:433`: score 18.173, shared prompts 23, Jaccard 0.299, model cosine 0.993, signed cosine 0.992.
- `L19:3899` -> `L20:3913`: score 16.529, shared prompts 22, Jaccard 0.282, model cosine 0.987, signed cosine 0.947.
- `L20:3795` -> `L24:2919`: score 16.328, shared prompts 22, Jaccard 0.282, model cosine 0.966, signed cosine 0.966.
- `L16:62` -> `L19:2041`: score 16.082, shared prompts 21, Jaccard 0.266, model cosine 1.000, signed cosine 1.000.
- `L20:2864` -> `L24:3251`: score 15.953, shared prompts 21, Jaccard 0.266, model cosine 0.995, signed cosine 0.993.
- `L20:2077` -> `L24:3642`: score 15.897, shared prompts 21, Jaccard 0.266, model cosine 0.994, signed cosine 0.989.
- `L19:1325` -> `L20:2290`: score 15.454, shared prompts 22, Jaccard 0.282, model cosine 0.963, signed cosine 0.866.
- `L19:3074` -> `L20:2162`: score 14.475, shared prompts 21, Jaccard 0.266, model cosine 0.944, signed cosine 0.907.
- `L19:442` -> `L20:1490`: score 14.012, shared prompts 20, Jaccard 0.250, model cosine 0.985, signed cosine 0.898.

## Candidate Chains

- `L16:3660 -> L19:3990 -> L20:1210 -> L24:1120`: geomean score 17.802, min shared prompts 18, edge scores 11.906|19.289|24.564.
- `L16:2351 -> L19:3990 -> L20:1210 -> L24:1120`: geomean score 15.694, min shared prompts 14, edge scores 8.158|19.289|24.564.
- `L16:1677 -> L19:3009 -> L20:2077 -> L24:3642`: geomean score 14.346, min shared prompts 21, edge scores 8.948|20.754|15.897.
- `L16:642 -> L19:4063 -> L20:1373 -> L24:3645`: geomean score 13.950, min shared prompts 15, edge scores 13.922|19.579|9.960.
- `L16:240 -> L19:1895 -> L20:2864 -> L24:3251`: geomean score 13.659, min shared prompts 20, edge scores 5.812|27.481|15.953.
- `L16:3660 -> L19:3990 -> L20:1210 -> L24:664`: geomean score 13.572, min shared prompts 18, edge scores 11.906|19.289|10.887.
- `L16:1959 -> L19:2355 -> L20:3219 -> L24:61`: geomean score 13.038, min shared prompts 17, edge scores 10.132|22.554|9.699.
- `L16:2351 -> L19:3990 -> L20:1210 -> L24:664`: geomean score 11.966, min shared prompts 14, edge scores 8.158|19.289|10.887.
- `L16:2351 -> L19:3615 -> L20:1210 -> L24:1120`: geomean score 11.639, min shared prompts 13, edge scores 7.579|8.468|24.564.
- `L16:689 -> L19:3990 -> L20:1210 -> L24:1120`: geomean score 11.618, min shared prompts 7, edge scores 3.309|19.289|24.564.
- `L16:2989 -> L19:2834 -> L20:3750 -> L24:1511`: geomean score 11.411, min shared prompts 15, edge scores 11.982|10.147|12.221.
- `L16:2943 -> L19:2943 -> L20:3795 -> L24:2919`: geomean score 11.307, min shared prompts 11, edge scores 3.968|22.313|16.328.
- `L16:1227 -> L19:2355 -> L20:3219 -> L24:61`: geomean score 10.944, min shared prompts 16, edge scores 5.992|22.554|9.699.
- `L16:1983 -> L19:2943 -> L20:3795 -> L24:2919`: geomean score 10.820, min shared prompts 9, edge scores 3.476|22.313|16.328.
- `L16:2989 -> L19:2834 -> L20:3750 -> L24:207`: geomean score 10.725, min shared prompts 15, edge scores 11.982|10.147|10.147.
- `L16:2351 -> L19:3615 -> L20:246 -> L24:664`: geomean score 10.449, min shared prompts 13, edge scores 7.579|11.444|13.153.
- `L16:3660 -> L19:3615 -> L20:1210 -> L24:1120`: geomean score 10.348, min shared prompts 10, edge scores 5.326|8.468|24.564.
- `L16:3660 -> L19:3990 -> L20:246 -> L24:664`: geomean score 10.315, min shared prompts 14, edge scores 11.906|7.009|13.153.
- `L16:2943 -> L19:2943 -> L20:3795 -> L24:1076`: geomean score 10.168, min shared prompts 11, edge scores 3.968|22.313|11.873.
- `L16:2346 -> L19:2834 -> L20:3750 -> L24:1511`: geomean score 10.154, min shared prompts 13, edge scores 8.443|10.147|12.221.

## Initial Manual Triage

These labels are first-pass hypotheses from the top activating contexts, not final interpretations.

### Candidate A: broad human boilerplate / simple continuation path

Chain: `L16:3660 -> L19:3990 -> L20:1210 -> L24:1120`

Evidence:

- The path is highly human-heavy at every node.
- Examples include simple expository continuations, future-letter prompts, classroom-style stories, how-to prose, and generic business/service copy.
- Exact prompt-contrast signs are mostly positive, meaning the path is not a straightforward AI-slop feature. It looks more like a detector-supporting direction for fluent but generic human-written no-robots continuations, or a false-positive bridge where human boilerplate resembles generated text.

Priority:

- High as a control circuit. It can test whether our method distinguishes human-signifying/supporting directions from AI-signifying style features.

Next intervention:

- On shared prompts such as the future-year story and "Dear Younger Self", ablate the layer-16 node and measure whether layer-19/20/24 activations in the same chain drop at the corresponding peak positions.

### Candidate B: generated literary scene-setting path

Chain: `L16:1677 -> L19:3009 -> L20:2077 -> L24:3642`

Evidence:

- Examples are dominated by GPT-5.5 literary fiction: named characters, atmospheric scene openings, rain/fog/alley/station/cabin settings, and polished sensory narration.
- Signed effects are mostly negative, so this is closer to an AI-signifying direction under the prompt-contrast score.
- The semantic family is coherent across all four layers, making it a good first target for a genuine style circuit.

Priority:

- Very high as the first stylistic circuit attempt.

Next intervention:

- Use 10 shared high-overlap narrative prompts. Ablate `L16:1677`, then rescan `L19:3009`, `L20:2077`, and `L24:3642` activation mass. Compare against matched-rank non-overlapping fiction latents.

### Candidate C: rhymed/stanzaic verse and heightened imagery path

Chain: `L16:1959 -> L19:2355 -> L20:3219 -> L24:61`

Evidence:

- Examples are poems or poem-like passages with line breaks, rhyme, elevated adjectives, and compact scenic imagery.
- Model mix includes GLM, Gemini, and Qwen examples rather than only GPT-5.5, so it may generalize across AI sources.
- The sign is mixed, so this may be a format/style path rather than a clean AI-vs-human separator.

Priority:

- Medium-high. It is semantically clean and easy to control with held-out poem prompts.

Next intervention:

- Test whether line-break/rhyme controls drive the same path independently of model source, and whether prose rewrites suppress it.

### Candidate D: informal/noisy human prose path

Chain: `L16:642 -> L19:4063 -> L20:1373 -> L24:3645`

Evidence:

- Examples include informal human writing, colloquial phrasing, imperfect grammar, first-person anecdotes, and children's-story text.
- This looks like a human-naturalness or low-polish path rather than a slop feature.
- It is useful as an anti-slop/human-signifying candidate.

Priority:

- Medium. Useful for controls and for understanding what detector evidence pushes away from polished AI style.

Next intervention:

- Compare ablation effects on noisy human text vs. polished AI text with similar topic.

### Candidate E: AI product/advice/list scaffold path

Chain seed: `L16:2943 -> L19:2943 -> L20:3795 -> L24:2919`

Evidence:

- This chain is not top-ranked overall but includes high-causal layer-19/20/24 nodes.
- Examples around layer 19/20/24 include product recommendations, skill lists, real-estate routines, business systems, and practical advice.
- It is likely closer to the operational "slop" signature than the top human-heavy chain.

Priority:

- High after Candidate B. It should be evaluated because it may capture polished recommendation/procedural scaffolding, one of the recurring manual feature families.

Next intervention:

- Pull shared examples for this chain, assign token-level labels in the explorer, then test exact ablation against human-written product/advice examples.

## Immediate Follow-Ups

1. Open the top chains in the latent explorer and assign semantic labels to each node.
2. For a labeled chain, run activation patching or latent clamping at each layer on the same prompt group.
3. Test whether upstream-latent ablation reduces downstream-latent activation on the same tokens/prompts.
4. Split candidates into likely stylistic circuits, task-format artifacts, and detector bookkeeping directions.

## Direct Layer-19-to-Layer-24 Seed Scan

I added a direct edge scan over layer 19 and layer 24 as a fast screen and fallback. It uses the same prompt-overlap, model-impact cosine, and signed-effect cosine heuristics as the original seed pass, but it does not require layer 20. The earlier note that layer-20 testing was blocked by checkpoint incompatibility was wrong; the available layer-20 checkpoints are interchangeable for this workflow.

Artifacts:

- `scripts/discover_pangram_sae_direct_edges.py`
- `artifacts/pangram_llama_sae/circuit_discovery/direct_edges_L19_to_L24.csv`
- `docs/experiments/pangram_sae_direct_edges_L19_to_L24.md`

Top direct edges:

| Rank | Edge | Score | Shared prompts | Notes |
|---:|---|---:|---:|---|
| 1 | `L19:3009 -> L24:3642` | `34.44` | 34 | recovers BOUNDARY-1 |
| 2 | `L19:2834 -> L24:207` | `20.83` | 25 | human-heavy control candidate |
| 3 | `L19:2041 -> L24:2988` | `20.20` | 30 | human-heavy narrative/dialogue candidate |
| 4 | `L19:1895 -> L24:3251` | `19.56` | 24 | GPT/GLM narrative candidate |
| 5 | `L19:3990 -> L24:1120` | `18.35` | 23 | recovers CONTROL-1 |
| 12 | `L19:2687 -> L24:4086` | `14.87` | 20 | new AI-heavy factual/procedural candidate |
| 24 | `L19:3450 -> L24:3469` | `9.52` | 18 | recovers LEX-1 |
| 25 | `L19:913 -> L24:1310` | `9.44` | 15 | recovers LEX-2 |

### CONTROL-2: human practical/expository continuation direction

Target: `L19:2834 -> L24:207`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_human_practical_expository_L19_2834_to_L24_207.csv`
- `docs/experiments/pangram_sae_circuit_target_human_practical_expository_L19_2834_to_L24_207.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_practical_expository_L19_2834_to_L24_207.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_practical_expository_L19_2834_to_L24_207.json`
- `artifacts/pangram_llama_sae/circuit_discovery/human_practical_expository_token_loci_details.csv`
- `docs/experiments/pangram_sae_human_practical_expository_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_human_practical_expository.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_practical_expository_summary.csv`

Intervention result:

| Docs | Source | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 25 | human | `1106.19` | `1673.94` | `-1578.20` | `-8.22` | `-0.179` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean class-3 logit drop |
|---|---:|---:|---:|
| different upstream, same downstream | `+4.72` | `5.10` | `-0.011` |
| same upstream, different downstream | `+0.10` | `1.44` | `-0.179` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Top-token class mix |
|---|---:|---:|---:|---:|---|
| `L19:2834` | 25 | 8,529 | `341.16` | `1106.19` | word `77.6%`, style/function word `16.0%`, expanded-negation word `4.6%` |
| `L24:207` | 25 | 8,454 | `338.16` | `1673.94` | word `88.2%`, style/function word `8.4%`, expanded-negation word `2.2%` |

Interpretation:

- This is a strong path-specific downstream dependency: controls that keep `L24:207` but change the upstream latent barely move downstream mass, while the anchor ablation drops almost all downstream mass.
- It is not an AI-slop path. The class-3 logit drop is negative, so ablating the upstream latent raises the AI score.
- The score movement follows `L19:2834`, not uniquely the full path: same-upstream/different-downstream controls preserve the same negative class-3 logit effect.
- The semantics are broad and human-practical rather than local lexical: examples are human no-robots continuations about home inspection, college choice, vacation planning, finance advice, TikTok, travel, essays, and simple fiction; active tokens are ordinary content and function words across hundreds of positions per document.

### CONTROL-2B: weaker sibling content feed into human practical/expository downstream

Target: `L19:3534 -> L24:207`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_human_practical_alt_L19_3534_to_L24_207.csv`
- `docs/experiments/pangram_sae_circuit_target_human_practical_alt_L19_3534_to_L24_207.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_practical_alt_L19_3534_to_L24_207.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_practical_alt_L19_3534_to_L24_207.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_human_practical_alt.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_practical_alt_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/human_practical_alt_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/human_practical_alt_token_loci_summary.csv`
- `docs/experiments/pangram_sae_human_practical_alt_token_loci.md`

Intervention result:

| Docs | Source | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 15 | human | `212.83` | `1647.98` | `-111.24` | `-0.38` | `-0.032` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.33` | `2.56` | `-0.012` | `-0.000` |
| same upstream, different downstream | `+0.64` | `0.80` | `+0.005` | `-0.032` |

Token-locus result:

| Node | Docs | Mean active tokens/doc | Mean mass/doc | Mean max/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:3534` | 15 | `107.87` | `212.83` | `4.61` | word `90.0%`, punctuation `4.1%`, sentence boundary `2.8%`, style/function word `2.2%` |
| `L24:207` | 15 | `322.93` | `1647.98` | `10.39` | word `80.7%`, style/function word `5.3%`, punctuation `5.1%`, sentence boundary `3.5%`, expanded-negation word `2.6%` |

Token-position overlap:

| Pair | Mean peak-token distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|
| `L19:3534` vs `L24:207` | `72.47` | `0.0%` | `0.065` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:3534` | `options`, `face`, `years`, `game`, `life`, `market`, `progress`, `planet`, `tree`, `expenses`, `finances`, `app`, `mind`, `surroundings` | sparse human content words in practical, explanatory, and simple narrative continuations |
| `L24:207` | `a`, `be`, `is`, `good`, `the`, `in`, `to`, `it`, `more`, `working`, `any`, `major`, `an`, `are`, `at` | broad human practical/expository continuation direction already characterized by `CONTROL-2` |

Interpretation:

- This is a real but much weaker sibling upstream into the same `L24:207` terminal used by `CONTROL-2`. The anchor ablation drops downstream mass by `-111.24`, while same-downstream controls average only `-0.33`.
- It is human-signifying, not AI-signifying: the class-3 logit movement is negative, so ablating the upstream slightly raises the AI score.
- It is not a duplicate of `CONTROL-2`. `L19:3534` is far sparser than `L19:2834`, almost entirely content-word driven, and has no same-peak overlap with `L24:207`.
- Treat it as a weaker semantic/content feed into the broad human practical/expository terminal. It is useful for decomposing human-side controls and as a guardrail against overinterpreting shared downstream latents.

### CONTROL-3: human narrative/dialogue scene continuation direction

Target: `L19:2041 -> L24:2988`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_human_narrative_dialogue_L19_2041_to_L24_2988.csv`
- `docs/experiments/pangram_sae_circuit_target_human_narrative_dialogue_L19_2041_to_L24_2988.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_narrative_dialogue_L19_2041_to_L24_2988.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_narrative_dialogue_L19_2041_to_L24_2988.json`
- `artifacts/pangram_llama_sae/circuit_discovery/human_narrative_dialogue_token_loci_details.csv`
- `docs/experiments/pangram_sae_human_narrative_dialogue_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_human_narrative_dialogue.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_narrative_dialogue_summary.csv`

Intervention result:

| Docs | Source | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 30 | human | `479.47` | `761.87` | `-486.55` | `-4.56` | `-0.060` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean class-3 logit drop |
|---|---:|---:|---:|
| different upstream, same downstream | `+0.76` | `0.85` | `+0.003` |
| same upstream, different downstream | `+0.01` | `1.02` | `-0.060` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Top-token class mix |
|---|---:|---:|---:|---:|---|
| `L19:2041` | 30 | 7,633 | `254.43` | `479.47` | word `79.4%`, punctuation `7.2%`, sentence boundary `4.8%` |
| `L24:2988` | 30 | 7,676 | `255.87` | `761.87` | word `77.9%`, punctuation `8.2%`, sentence boundary `5.2%` |

Interpretation:

- This is a path-specific human narrative/dialogue dependency. Alternate upstreams barely move `L24:2988`; alternate downstreams barely move under `L19:2041` ablation.
- It is human-signifying rather than AI-signifying: the class-3 logit drop is negative, so ablating `L19:2041` raises the AI score.
- The examples are no-robots human stories with dialogue, scene action, named characters, and simple narrative continuation. The active loci are broad ordinary narrative tokens rather than a small local punctuation or phrase family.
- This should be used as a contrast for AI boundary circuits: it helps separate human story-continuation evidence from generated scene-beat or boundary-density evidence.

### CONTROL-3B: human narrative boundary feed into broad narrative downstream

Target: `L19:3457 -> L24:2988`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_human_narrative_sibling_L19_3457_to_L24_2988.csv`
- `docs/experiments/pangram_sae_circuit_target_human_narrative_sibling_L19_3457_to_L24_2988.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_narrative_sibling_L19_3457_to_L24_2988.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_narrative_sibling_L19_3457_to_L24_2988.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_human_narrative_sibling.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_narrative_sibling_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_narrative_sibling_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/human_narrative_sibling_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/human_narrative_sibling_token_loci_summary.csv`
- `docs/experiments/pangram_sae_human_narrative_sibling_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 17 | human/GLM | `202.82` | `886.36` | `-53.36` | `-0.16` | `-0.530` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-1.51` | `1.78` | `-0.006` | `-0.003` |
| same upstream, different downstream | `-0.38` | `0.99` | `+0.038` | `-0.530` |

Token-locus result:

| Node | Docs | Mean active tokens/doc | Mean mass/doc | Mean max/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:3457` | 17 | `95.24` | `202.82` | `5.86` | word `38.2%`, sentence boundary `25.4%`, punctuation `24.6%`, style/function word `4.3%`, line break `4.3%` |
| `L24:2988` | 17 | `294.41` | `886.36` | `8.09` | word `77.3%`, punctuation `8.3%`, sentence boundary `6.1%`, style/function word `3.1%`, expanded-negation word `2.8%` |

Token-position overlap:

| Pair | Mean peak-token distance | Median peak-token distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:3457` vs `L24:2988` | `76.82` | `69.0` | `0.0%` | `0.000` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:3457` | period 137, paragraph-final period 21, `!"` 10, `."` 9, quote-period paragraph break 6, `?"` 5 | human narrative sentence and dialogue-boundary punctuation |
| `L24:2988` | `up`, comma, `and`, `he`, `thought`, `says`, `looked`, `walks`, `said`, `down`, `whispered`, `she`, `stood`, `eyes`, `out` | broad human narrative/action/dialogue words |

Interpretation:

- This is a real but weak sibling feed into the `CONTROL-3` downstream. Ablating `L19:3457` drops `L24:2988` mass by `-53.36`, while different-upstream same-downstream controls average only `-1.51`.
- It is not a local same-token path. The upstream is boundary punctuation in human narrative, while the downstream is broad narrative/action wording; peak overlap and top-token Jaccard are both zero.
- The downstream max barely moves (`-0.16`), unlike the main `CONTROL-3` path (`-4.56`), so the effect is diffuse downstream mass rather than collapse of the strongest activation sites.
- The classifier movement follows the upstream and is human-signifying (`-0.530` class-3 logit drop). Treat this as human narrative boundary context feeding a broad narrative downstream, useful for decomposing human controls and for comparison against AI narrative-boundary paths.

### CONTROL-4: human expository/comparison document-wide direction

Target: `L19:442 -> L24:2175`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_human_expository_compare_L19_442_to_L24_2175.csv`
- `docs/experiments/pangram_sae_circuit_target_human_expository_compare_L19_442_to_L24_2175.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_expository_compare_L19_442_to_L24_2175.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_expository_compare_L19_442_to_L24_2175.json`
- `artifacts/pangram_llama_sae/circuit_discovery/human_expository_compare_token_loci_details.csv`
- `docs/experiments/pangram_sae_human_expository_compare_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_human_expository_compare.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_expository_compare_summary.csv`

Intervention result:

| Docs | Source | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 24 | human | `616.98` | `1000.44` | `-934.39` | `-5.06` | `-0.154` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean class-3 logit drop |
|---|---:|---:|---:|
| different upstream, same downstream | `-2.47` | `2.88` | `-0.009` |
| same upstream, different downstream | `-0.12` | `1.42` | `-0.154` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:442` | 24 | 6,827 | `284.46` | `616.98` | word `80.2%`, style/function word `5.1%`, punctuation `5.0%`, sentence boundary `4.6%` |
| `L24:2175` | 24 | 6,826 | `284.42` | `1000.44` | word `80.3%`, style/function word `5.1%`, punctuation `4.8%`, sentence boundary `4.6%` |

Token-position overlap:

| Pair | Peak-token mean distance | Peak-token median distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:442` vs `L24:2175` | `24.4` | `9.0` | `20.8%` | `0.646` |

Interpretation:

- This is a strong human-signifying path-specific dependency. Alternate upstreams barely move `L24:2175`, while `L19:442` ablation collapses most of its mass.
- It is broad and document-wide rather than a local phrase or boundary circuit. Both nodes activate on almost exactly the same number of tokens per document and have nearly identical word-heavy token-class distributions.
- The examples are human no-robots expository/comparison continuations: SEO/business authority, CPA vs accountant, standardized testing, travel, diets, speeches, and human-written short fiction.
- This is useful as a high-mass human control against AI expository terminal circuits like BOUNDARY-2 and broad AI-register paths like BROAD-1. It should not be interpreted as an AI-slop feature.

### CONTROL-5: human practical/advice document-wide direction

Target: `L19:2650 -> L24:3790`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_human_practical_advice_L19_2650_to_L24_3790.csv`
- `docs/experiments/pangram_sae_circuit_target_human_practical_advice_L19_2650_to_L24_3790.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_practical_advice_L19_2650_to_L24_3790.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_practical_advice_L19_2650_to_L24_3790.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_human_practical_advice.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_practical_advice_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_practical_advice_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/human_practical_advice_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/human_practical_advice_token_loci_summary.csv`
- `docs/experiments/pangram_sae_human_practical_advice_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 19 | mostly human | `590.99` | `1034.10` | `-1017.01` | `-5.35` | `-0.081` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `+1.57` | `2.07` | `+0.004` | `+0.005` |
| same upstream, different downstream | `+0.23` | `0.33` | `+0.023` | `-0.081` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:2650` | 19 | 5,077 | `267.21` | `590.99` | word `78.5%`, style/function word `8.3%`, punctuation `5.7%`, expanded-negation word `4.4%` |
| `L24:3790` | 19 | 5,398 | `284.11` | `1034.10` | word `75.4%`, style/function word `11.0%`, expanded-negation word `7.0%`, punctuation `4.4%` |

Token-position overlap:

| Pair | Peak-token mean distance | Peak-token median distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:2650` vs `L24:3790` | `52.9` | `20.0` | `21.1%` | `0.377` |

Top-token context inspection:

| Node | Top tokens | Context pattern counts |
|---|---|---|
| `L19:2650` | `,` 13, `is` 9, `will` 6, `a` 6, `to` 5, `you` 5, `and` 4, `have` 4 | advice contexts 36; practical-topic contexts 21; list-like contexts 40 |
| `L24:3790` | `to` 15, `is` 14, `,` 10, `are` 9, `will` 8, `you` 7, `I` 6, `and` 6 | advice contexts 36; practical-topic contexts 15; list-like contexts 41 |

Interpretation:

- This is a high-mass path-specific human-side dependency. Alternate upstreams into `L24:3790` barely move downstream mass compared with the anchor's `-1017.01` collapse.
- It is human-signifying rather than AI-signifying: the class-3 logit drop is negative, so ablating the upstream raises the AI score.
- It is not a local phrase or boundary circuit. Both nodes activate on hundreds of ordinary tokens per document, and peak overlap is low. The target documents are mostly human no-robots practical/advice/expository prose about environmental policy, planned obsolescence, saving money, cats, travel, and everyday decisions.
- This should be used as a broad human control against AI broad-register paths like `BROAD-1` and `BROAD-2`, not as a compact style feature.

### CONTROL-6: human informal/mixed-prose document-wide direction

Target: `L19:4063 -> L24:3645`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_human_informal_noisy_L19_4063_to_L24_3645.csv`
- `docs/experiments/pangram_sae_circuit_target_human_informal_noisy_L19_4063_to_L24_3645.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_informal_noisy_L19_4063_to_L24_3645.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_informal_noisy_L19_4063_to_L24_3645.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_human_informal_noisy.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_informal_noisy_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_informal_noisy_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/human_informal_noisy_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/human_informal_noisy_token_loci_summary.csv`
- `docs/experiments/pangram_sae_human_informal_noisy_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 14 | human only | `494.57` | `487.78` | `-343.25` | `-4.83` | `-0.201` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.28` | `0.83` | `-0.009` | `+0.013` |
| same upstream, different downstream | `+0.28` | `1.19` | `+0.026` | `-0.201` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:4063` | 14 | 3,586 | `256.14` | `494.57` | word `81.0%`, punctuation `4.6%`, style/function word `4.0%`, sentence boundary `4.0%`, expanded-negation word `3.0%` |
| `L24:3645` | 14 | 1,670 | `119.29` | `487.78` | word `89.9%`, style/function word `2.9%`, punctuation `2.9%`, sentence boundary `2.1%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:4063` vs `L24:3645` | `7.1%` | `0.375` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:4063` | `day` 4, `me` 4, `it` 3, `Love` 2, `there` 2, `time` 2, `group` 2, `sign` 2, `street` 2, `weather` 2, `part` 2, `said` 2 | broad human prose across poem-like love writing, stage dialogue, informal stories, ghost lists, market explanations, travel advice |
| `L24:3645` | `day` 4, `it` 4, `me` 3, `markets` 2, `rise` 2, `market` 2, `part` 2, `here` 2, `down` 2, `morning` 2, `woods` 2, `ground` 2 | same broad human-prose family but with lower active-token count and different peak positions |

Interpretation:

- This is a strong human-signifying path-specific dependency. The anchor ablation drops L24 mass by `-343.25`, while different-upstream controls into `L24:3645` average only `-0.28`.
- It is human-signifying rather than AI-signifying: the class-3 logit drop is negative, so ablating `L19:4063` raises the AI score.
- It is broad and document-wide rather than local. L19 fires on about 256 tokens/doc and L24 on about 119 tokens/doc, mostly ordinary words, with low same-peak overlap. The examples are all human no-robots continuations spanning poems, dramatic scripts, ghost lists, folk/adventure stories, bored-science exposition, travel advice, and market explanations.
- This should be treated as a broad human-control path. It helps calibrate the detector's human side but should not be used as a compact local style circuit.

### CONTROL-7: human narrative sentence-boundary / scene-progression direction

Target: `L19:3457 -> L24:1511`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_human_broad_L19_3457_to_L24_1511.csv`
- `docs/experiments/pangram_sae_circuit_target_human_broad_L19_3457_to_L24_1511.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_broad_L19_3457_to_L24_1511.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_broad_L19_3457_to_L24_1511.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_human_broad.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_broad_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_broad_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/human_broad_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/human_broad_token_loci_summary.csv`
- `docs/experiments/pangram_sae_human_broad_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 14 | human only | `162.12` | `297.24` | `-84.08` | `-2.20` | `-0.493` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.76` | `0.82` | `-0.010` | `-0.006` |
| same upstream, different downstream | `-13.13` | `18.62` | `-0.023` | `-0.493` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:3457` | 14 | 1,057 | `75.50` | `162.12` | word `45.4%`, sentence boundary `24.0%`, punctuation `18.7%`, style/function word `4.4%`, line break `4.1%` |
| `L24:1511` | 14 | 1,036 | `74.00` | `297.24` | word `39.1%`, sentence boundary `24.9%`, punctuation `22.3%`, style/function word `6.1%`, line break `4.2%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:3457` vs `L24:1511` | `21.4%` | `0.451` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:3457` | `.` 132, `.\n\n` 12, `.\n` 5, `."\n` 4, `".` 3, `."` 3, comma 2 | human narrative/story sentence endings, dialogue closures, and scene beats |
| `L24:1511` | `.` 127, comma 30, `".` 3, `."` 3, opening quote 2 | same broad sentence-boundary/dialogue-closure family with more comma mass |

Interpretation:

- This is a path-specific human-signifying narrative boundary direction. The anchor ablation drops L24 mass by `-84.08`, while different-upstream controls into `L24:1511` average only `-0.76`.
- It has the largest human-side score movement in the current documented control set: ablating `L19:3457` raises the AI score by about `0.493` class-3 logit units.
- Unlike `CONTROL-6`, this is not purely document-wide ordinary-word mass. Both nodes fire heavily on periods, quote-punctuation, and sentence endings in human story continuations, with examples involving haunted houses, missing children, fantasy villages, healers, family scenes, and dialogue.
- It is still a human-control path, not an AI-slop circuit. Its value is as a contrast case: human narratives can have strong boundary/scene-progression directions that move the classifier away from the AI class.

### CONTROL-7B: broad human practical/expository feed into narrative-boundary terminal

Target: `L19:2834 -> L24:1511`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_human_practical_to_narrative_boundary_L19_2834_to_L24_1511.csv`
- `docs/experiments/pangram_sae_circuit_target_human_practical_to_narrative_boundary_L19_2834_to_L24_1511.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_practical_to_narrative_boundary_L19_2834_to_L24_1511.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_practical_to_narrative_boundary_L19_2834_to_L24_1511.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_human_practical_to_narrative_boundary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_practical_to_narrative_boundary_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_practical_to_narrative_boundary_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/human_practical_to_narrative_boundary_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/human_practical_to_narrative_boundary_token_loci_summary.csv`
- `docs/experiments/pangram_sae_human_practical_to_narrative_boundary_token_loci.md`

Intervention result:

| Docs | Source | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 13 | human | `1072.21` | `253.02` | `-68.15` | `-1.52` | `-0.166` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.22` | `0.38` | `-0.007` | `-0.001` |
| same upstream, different downstream | `+5.51` | `92.68` | `-0.113` | `-0.166` |

Token-locus result:

| Node | Docs | Mean active tokens/doc | Mean mass/doc | Mean max/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:2834` | 13 | `314.46` | `1072.21` | `6.10` | word `79.7%`, style/function word `4.6%`, sentence boundary `4.6%`, punctuation `4.6%`, expanded-negation word `3.9%` |
| `L24:1511` | 13 | `64.38` | `253.02` | `9.01` | word `37.9%`, sentence boundary `23.1%`, punctuation `22.5%`, style/function word `8.0%`, line break `3.7%` |

Token-position overlap:

| Pair | Mean peak-token distance | Median peak-token distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:2834` vs `L24:1511` | `39.92` | `27.0` | `0.0%` | `0.017` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:2834` | `to`, `would`, `a`, `be`, `is`, `the`, `you`, `but`, `will`, `help`, `an`, `it`, `was`, `have` | broad human practical/expository word and function-word mass |
| `L24:1511` | period 119, comma 25, dash 3, period-newline, question mark, quote-period tokens | sparse human narrative/practical boundary terminal |

Interpretation:

- This is a path-specific sibling into the `CONTROL-7` downstream: `L19:2834` ablation drops `L24:1511` by `-68.15` mass and `-1.52` max, while different-upstream same-downstream controls average only `-0.22`.
- It is not a same-token local boundary circuit. The upstream is broad practical/expository word mass, while the downstream is sparse punctuation and sentence-boundary mass; peak overlap is zero.
- The class-score movement follows `L19:2834` and is human-signifying (`-0.166` class-3 logit drop). Same-upstream/different-downstream controls preserve the score movement but not the specific `L24:1511` collapse.
- Treat this as broad human practical/expository context feeding a narrative/practical boundary terminal. It is useful for decomposing human controls and for separating broad context feeds from same-token boundary circuits.

### CONTROL-8: mixed high-register document-wide humanlike prose direction

Target: `L19:2301 -> L24:229`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_positive_L19_2301_to_L24_229.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_positive_L19_2301_to_L24_229.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_positive_L19_2301_to_L24_229.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_positive_L19_2301_to_L24_229.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_positive.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_positive_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_positive_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_positive_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_positive_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_positive_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 14 | Gemini/GLM/GPT/human/Qwen | `611.25` | `986.91` | `-844.45` | `-7.38` | `-0.618` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `+3.58` | `4.15` | `+0.004` | `+0.279` |
| same upstream, different downstream | `-3.38` | `5.35` | `+0.183` | `-0.618` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:2301` | 14 | 2,861 | `204.36` | `611.25` | word `78.3%`, punctuation `7.8%`, line break `4.5%`, sentence boundary `3.6%`, style/function word `2.9%` |
| `L24:229` | 14 | 2,901 | `207.21` | `986.91` | word `78.5%`, punctuation `7.9%`, line break `4.6%`, sentence boundary `3.5%`, style/function word `2.8%` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:2301` | comma 21, period 20, `Judaism` 4, `It` 3, `of` 3, `brain` 3, plus scattered ordinary content words | broad polished/high-register prose across architecture biography, legal critique, astronomy exposition, time-travel dialogue, religious taxonomy, aircraft design, esoteric biography, body-proportion exposition, cipher explanation, mythic history, film review, verse-like strength lesson, and sailing prose |
| `L24:229` | comma 29, `Judaism` 4, `of` 4, period 3, `the` 3, `a` 3, `it` 3, `watch` 2, `not` 2, `are` 2 | same broad document-wide lexical/prose-state feature with stronger downstream mass and nearly identical token-class distribution |

Interpretation:

- This is a very strong path-specific dependency: anchor ablation drops L24 mass by `-844.45`, while different-upstream same-downstream controls average only `+3.58`.
- It is human-signifying under the current classifier. The negative class-3 movement means ablating the upstream raises the AI score by about `0.618` logit units.
- The target examples are mixed-source, not human-only. The shared feature appears to be high-register, polished, or literary humanlike prose, which can be produced by GLM/Qwen/GPT/Gemini as well as human text.
- It is not a compact local style circuit. Both nodes fire on about 200 ordinary-word tokens per document with nearly identical token-class distributions. Use it as a broad control and decompose further only if we need to separate this high-register humanlike prose state from local slop circuits.

### CONTROL-9: formal factual/report-style document-wide direction

Target: `L19:2390 -> L24:3169`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown6_L19_2390_to_L24_3169.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_unknown6_L19_2390_to_L24_3169.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown6_L19_2390_to_L24_3169.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown6_L19_2390_to_L24_3169.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_unknown6.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown6_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown6_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown6_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown6_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_unknown6_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 14 | human/Qwen/GLM/Gemini | `501.13` | `397.26` | `-364.29` | `-2.18` | `-0.288` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.746` | `0.879` | `-0.001` | `+0.033` |
| same upstream, different downstream | `+0.957` | `1.058` | `+0.014` | `-0.288` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:2390` | 14 | 3,905 | `278.93` | `501.13` | word `80.8%`, punctuation `8.0%`, style/function word `3.7%`, sentence boundary `2.9%` |
| `L24:3169` | 14 | 3,005 | `214.64` | `397.26` | word `79.5%`, punctuation `8.4%`, style/function word `4.0%`, sentence boundary `3.3%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:2390` vs `L24:3169` | `14.3%` | `0.391` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:2390` | comma 33, `is` 11, period 7, `to` 5, `also` 4, `Rocket` 4, `was` 3, `by` 3 | formal factual/news/report-style prose about gold, Saudi astronauts on the ISS, 9/11 victim demographics, UAE asteroid exploration, Hiroshima, urgent antibody testing, space companies, company history, and finance Q&A |
| `L24:3169` | comma 37, `is` 5, `to` 4, `Rocket` 4, period 4, `would` 3, `was` 3, `AA` 3 | same broad document-wide formal factual/reporting direction, with downstream mass spread over ordinary words, punctuation, proper-noun fragments, and function words |

Interpretation:

- This is a strong path-specific dependency: anchor ablation drops `L24:3169` mass by `-364.29`, while both matched-control families average around one unit of downstream movement.
- It is human-signifying, not AI-signifying. The negative class-3 movement means ablating `L19:2390` raises the AI score by about `0.288` logit units.
- The feature is broad and document-wide rather than local. Both nodes activate over hundreds of mostly ordinary word tokens, with only modest top-token overlap.
- This is a useful contrast for polished factual AI-register paths such as `BROAD-1` and `BROAD-3`: surface formality and report-like prose alone are not enough to imply an AI-slop circuit.

### CONTROL-10: plain human advice/story continuation direction

Target: `L19:1459 -> L24:1599`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown7_L19_1459_to_L24_1599.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_unknown7_L19_1459_to_L24_1599.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown7_L19_1459_to_L24_1599.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown7_L19_1459_to_L24_1599.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_unknown7.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown7_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown7_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown7_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown7_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_unknown7_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 13 | human/GLM/Qwen | `489.96` | `811.74` | `-741.30` | `-5.80` | `-0.111` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.180` | `1.138` | `+0.002` | `+0.000` |
| same upstream, different downstream | `+0.023` | `2.939` | `-0.001` | `-0.111` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:1459` | 13 | 2,894 | `222.62` | `489.96` | word `81.0%`, style/function word `4.6%`, punctuation `4.3%`, sentence boundary `3.5%` |
| `L24:1599` | 13 | 2,940 | `226.15` | `811.74` | word `80.8%`, style/function word `4.6%`, punctuation `4.5%`, sentence boundary `3.8%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:1459` vs `L24:1599` | `53.8%` | `0.615` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:1459` | `and` 6, `will` 5, `are` 4, `the` 4, `a` 3, `is` 3, `to` 3, `it` 3, `more` 3 | plain human continuations: community involvement advice, employee-retention tips, simple folk/story prose, robot/domestic stories, retail-business advice, home fragrance anecdote, comedy recommendations, snow poem, park/family narrative, and baseball-news-like prose |
| `L24:1599` | `will` 7, `and` 7, `was` 5, `the` 5, `can` 4, `are` 4, `to` 4, `it` 4, `he` 3 | same broad plain-continuation direction, with stronger downstream mass and high overlap in the same ordinary token positions |

Interpretation:

- This is a very strong path-specific dependency: `L19:1459` ablation drops `L24:1599` mass by `-741.30`, while matched controls are near zero.
- It is human-signifying. The negative class-3 movement means removing the feature raises the AI score by about `0.111` logit units.
- Compared with `CONTROL-9`, this path has much stronger token-position overlap and a simpler prose profile. It is still broad and document-wide rather than a compact lexical or formatting circuit.
- Use it as a plain human-continuation control for advice and simple narrative examples. It should not be treated as local slop evidence.

### ORNATE-1: archaic/formal voice into boundary-format terminal

Target: `L19:3912 -> L24:1697`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_ornate_dialogue_L19_3912_to_L24_1697.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_ornate_dialogue_L19_3912_to_L24_1697.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_ornate_dialogue_L19_3912_to_L24_1697.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_ornate_dialogue_L19_3912_to_L24_1697.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_ornate_dialogue.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_ornate_dialogue_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_ornate_dialogue_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_ornate_dialogue_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_ornate_dialogue_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_ornate_dialogue_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 23 | Gemini/GLM/GPT/human/Qwen | `819.81` | `220.95` | `-105.97` | `-3.63` | `-0.017` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.40` | `0.55` | `-0.022` | `+0.005` |
| same upstream, different downstream | `+0.39` | `1.26` | `+0.015` | `-0.017` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:3912` | 23 | 7,844 | `341.04` | `819.81` | word `78.9%`, punctuation `6.8%`, sentence boundary `3.9%`, line break `3.6%`, style/function word `3.6%` |
| `L24:1697` | 23 | 1,284 | `55.83` | `220.95` | punctuation `27.1%`, sentence boundary `25.5%`, line break `23.2%`, word `19.2%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:3912` vs `L24:1697` | `0.0%` | `0.031` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:3912` | `I`/` I` 38, comma 20, `you` 11, `have` 8, `not` 7, `You` 7, `who` 6, `shall` 5 | broad archaic/formal voice, first/second-person address, modal/negation-heavy stylized prose |
| `L24:1697` | `.` 150, `.\n` 17, `.\n\n` 15, newline 15, opening quote 12, comma 11, `!` 6, `But` 5, `?\n` 5 | boundary/format terminal in stylized dialogue, verse, formal letters, and archaic/fantasy prose |

Interpretation:

- This is a clean path-specific dependency. The anchor ablation drops L24 mass by `-105.97`, while matched controls are close to zero.
- It is not score-relevant under the current classifier: class-3 movement is only `-0.017`.
- This is not a same-token circuit. The upstream is broad register/diction, the downstream is a boundary/format terminal, and overlap is almost absent.
- Treat it as a register-to-boundary-format circuit useful for decomposing stylized/archaic prose, not as an AI-slop path.

### MIXED-1: sparse scene/dialogue boundary and line-break path

Target: `L19:1483 -> L24:2644`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_scene_dialogue_L19_1483_to_L24_2644.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_scene_dialogue_L19_1483_to_L24_2644.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_scene_dialogue_L19_1483_to_L24_2644.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_scene_dialogue_L19_1483_to_L24_2644.json`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_scene_dialogue_token_loci_details.csv`
- `docs/experiments/pangram_sae_mixed_scene_dialogue_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_scene_dialogue.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_scene_dialogue_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_mixed_scene_dialogue_linebreak_joined_targets.csv`
- `docs/experiments/pangram_sae_counterfactual_mixed_scene_dialogue_linebreak_joined_targets.md`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_mixed_scene_dialogue_linebreak_joined_intervention_L19_1483_to_L24_2644.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_mixed_scene_dialogue_linebreak_joined_token_loci_summary.csv`
- `docs/experiments/pangram_sae_counterfactual_mixed_scene_dialogue_linebreak_joined_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_mixed_scene_dialogue_sentence_softened_targets.csv`
- `docs/experiments/pangram_sae_counterfactual_mixed_scene_dialogue_sentence_softened_targets.md`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_mixed_scene_dialogue_sentence_softened_intervention_L19_1483_to_L24_2644.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_mixed_scene_dialogue_sentence_softened_token_loci_summary.csv`
- `docs/experiments/pangram_sae_counterfactual_mixed_scene_dialogue_sentence_softened_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 32 | Gemini/GLM/GPT/human/Qwen | `80.74` | `128.55` | `-68.60` | `-5.20` | `-0.003` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean class-3 logit drop |
|---|---:|---:|---:|
| different upstream, same downstream | `+0.23` | `0.31` | `+0.003` |
| same upstream, different downstream | `+0.23` | `0.53` | `-0.003` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Top-token class mix |
|---|---:|---:|---:|---:|---|
| `L19:1483` | 32 | 1,426 | `44.56` | `80.74` | sentence boundary `31.0%`, punctuation `26.4%`, word `22.3%`, line break `17.2%` |
| `L24:2644` | 32 | 1,522 | `47.56` | `128.55` | word `32.2%`, sentence boundary `27.2%`, punctuation `23.2%`, line break `15.0%` |

Counterfactual result:

| Edit | Variant | L19 mass | L24 mass | L24 mass delta after L19 ablation | L24 max | Mean class-3 logit drop |
|---|---|---:|---:|---:|---:|---:|
| join line breaks | original | `82.73` | `131.14` | `-70.39` | `9.81` | `-0.005` |
| join line breaks | edited | `84.57` | `138.71` | `-75.14` | `9.02` | `-0.011` |
| terminal `.?!` to commas | original | `80.74` | `128.55` | `-68.60` | `9.82` | `-0.003` |
| terminal `.?!` to commas | edited | `72.14` | `128.72` | `-64.44` | `8.38` | `+0.004` |

Counterfactual token-locus shift:

| Edit | Node | Variant | Main active token classes |
|---|---|---|---|
| join line breaks | `L19:1483` | original | sentence boundary `31.0%`, punctuation `26.5%`, word `22.1%`, line break `17.3%` |
| join line breaks | `L19:1483` | edited | sentence boundary `44.7%`, punctuation `32.9%`, word `19.9%` |
| join line breaks | `L24:2644` | original | word `31.7%`, sentence boundary `27.3%`, punctuation `23.4%`, line break `15.2%` |
| join line breaks | `L24:2644` | edited | sentence boundary `38.3%`, word `31.0%`, punctuation `28.2%` |
| terminal `.?!` to commas | `L19:1483` | edited | punctuation `59.0%`, word `23.7%`, line break `14.8%` |
| terminal `.?!` to commas | `L24:2644` | edited | punctuation `48.6%`, word `38.2%`, line break `10.3%` |

Interpretation:

- This is a real path-specific downstream dependency under the matched-control test, but it is not a useful detector-score path: mean class-3 movement is essentially zero.
- The direct-edge heuristic found a shared genre/format pathway despite weak signed-direction agreement. That makes this a useful calibration case for separating "overlapping scene/dialogue style" from classifier-relevant slop or human evidence.
- The loci are sparse compared with the broad human controls and concentrate on segment-boundary-like tokens. The counterfactuals rule out a narrow interpretation as a literal newline or period detector: joining line breaks shifts mass onto sentence-boundary punctuation and increases activation, while replacing `.?!` with commas mostly preserves downstream mass and shifts loci onto comma punctuation.
- The current best label is sparse scene/dialogue segment-boundary accumulator. It should remain a weak/mixed format-path control rather than a slop-style feature.

### BOUNDARY-1B: broad polished narrative upstream into BOUNDARY-1 terminal

Target: `L19:1066 -> L24:3642`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_boundary_alt_upstream_L19_1066_to_L24_3642.csv`
- `docs/experiments/pangram_sae_circuit_target_boundary_alt_upstream_L19_1066_to_L24_3642.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_boundary_alt_upstream_L19_1066_to_L24_3642.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_boundary_alt_upstream_L19_1066_to_L24_3642.json`
- `artifacts/pangram_llama_sae/circuit_discovery/boundary_alt_upstream_token_loci_details.csv`
- `docs/experiments/pangram_sae_boundary_alt_upstream_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_boundary_alt_upstream.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_boundary_alt_upstream_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/targets_boundary_terminal_dual_upstream_L19_3009_L19_1066_to_L24_3642.csv`
- `docs/experiments/pangram_sae_circuit_target_boundary_terminal_dual_upstream_L19_3009_L19_1066_to_L24_3642.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_boundary_terminal_dual_target_L19_3009_to_L24_3642.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_boundary_terminal_dual_target_L19_1066_to_L24_3642.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_boundary_terminal_dual_target_L19_3009_plus_L19_1066_to_L24_3642.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/boundary_terminal_dual_upstream_token_loci_details.csv`
- `docs/experiments/pangram_sae_boundary_terminal_dual_upstream_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 24 | mostly GPT-5.5 | `1533.55` | `361.35` | `-93.77` | `-1.14` | `0.167` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean class-3 logit drop |
|---|---:|---:|---:|
| different upstream, same downstream | `+0.09` | `0.19` | `+0.003` |
| same upstream, different downstream | `+1.54` | `2.18` | `+0.167` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Top-token class mix |
|---|---:|---:|---:|---:|---|
| `L19:1066` | 24 | 10,668 | `444.50` | `1533.55` | word `80.6%`, punctuation `7.5%`, sentence boundary `4.1%` |
| `L24:3642` | 24 | 2,094 | `87.25` | `361.35` | punctuation `31.0%`, word `23.9%`, sentence boundary `22.1%`, line break `15.5%` |

Additivity result on the `L19:3009`, `L19:1066`, `L24:3642` shared target set:

| Ablated upstream | Docs | Mean L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| `L19:3009` | 16 | `-251.41` | `-6.64` | `0.127` |
| `L19:1066` | 16 | `-100.36` | `-1.07` | `0.147` |
| `L19:3009 + L19:1066` | 16 | `-343.09` | `-7.82` | `0.286` |

Token-position overlap on the same shared target set:

| Pair | Peak-token mean distance | Peak-token median distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:3009` vs `L19:1066` | `118.2` | `105.0` | `0.0%` | `0.030` |
| `L19:3009` vs `L24:3642` | `45.8` | `0.0` | `62.5%` | `0.748` |
| `L19:1066` vs `L24:3642` | `123.6` | `110.0` | `0.0%` | `0.046` |

Shared-target token-locus summary:

| Node | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---|
| `L19:3009` | `76.50` | `237.24` | sentence boundary `27.6%`, word `24.5%`, line break `20.7%`, punctuation `20.3%` |
| `L19:1066` | `471.19` | `1619.17` | word `80.5%`, punctuation `7.2%`, sentence boundary `4.4%` |
| `L24:3642` | `94.06` | `409.64` | punctuation `29.2%`, word `24.3%`, sentence boundary `22.7%`, line break `16.8%` |

Interpretation:

- `L24:3642` is not fed only by the local BOUNDARY-1 upstream `L19:3009`. `L19:1066` is another specific upstream path into the same terminal: alternate upstreams into `L24:3642` barely move downstream mass, while this one drops it by `-93.77`.
- The upstream is qualitatively different from `L19:3009`: it is broad, word-heavy, and document-wide over polished narrative prose. The downstream remains sparse and boundary-heavy.
- The score movement follows the broad upstream: same-upstream/different-downstream controls preserve the class-3 logit drop. The specific circuit evidence is the selective downstream collapse of `L24:3642`, not unique final-score mediation through that downstream latent.
- The dual-upstream panel suggests mostly additive inputs into the same terminal. The summed single-upstream L24 mass drop is `-351.76`, while the joint drop is `-343.09`; this is close to additive with mild subadditivity. The score drop is also roughly additive (`0.127 + 0.147` vs joint `0.286`).
- The position-overlap result splits the inputs: `L19:3009` is token-local to the terminal, sharing the same peak token with `L24:3642` in 62.5% of documents and a top-token Jaccard of `0.748`; `L19:1066` is a broad context feed with almost no top-token overlap against either node. The next stronger test should be mediation or activation patching rather than another plain ablation.

### BOUNDARY-2: AI expository end-of-window terminal accumulator

Target: `L19:2943 -> L24:2919`

This direct layer-19-to-layer-24 test complements the old seed `L19:2943 -> L20:3795 -> L24:2919`; the layer-20 hops should be tested with the available interchangeable layer-20 checkpoint.

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_capstone_boundary_L19_2943_to_L24_2919.csv`
- `docs/experiments/pangram_sae_circuit_target_ai_capstone_boundary_L19_2943_to_L24_2919.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_capstone_boundary_L19_2943_to_L24_2919.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_capstone_boundary_L19_2943_to_L24_2919.json`
- `artifacts/pangram_llama_sae/circuit_discovery/ai_capstone_boundary_token_loci_details.csv`
- `docs/experiments/pangram_sae_ai_capstone_boundary_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_ai_capstone_boundary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_ai_capstone_boundary_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_capstone_boundary_L19_2943_to_L24_2919_max640.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/ai_capstone_boundary_token_loci_max640_details.csv`
- `docs/experiments/pangram_sae_ai_capstone_boundary_token_loci_max640.md`
- `scripts/build_pangram_sae_prefix_targets.py`
- `artifacts/pangram_llama_sae/circuit_discovery/prefix_ai_capstone_boundary_targets.csv`
- `docs/experiments/pangram_sae_prefix_ai_capstone_boundary_targets.md`
- `artifacts/pangram_llama_sae/circuit_discovery/prefix_ai_capstone_boundary_intervention_L19_2943_to_L24_2919.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/prefix_ai_capstone_boundary_token_loci_details.csv`
- `docs/experiments/pangram_sae_prefix_ai_capstone_boundary_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 14 | Qwen-heavy | `105.19` | `151.48` | `-117.00` | `-11.04` | `2.054` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean class-3 logit drop |
|---|---:|---:|---:|
| different upstream, same downstream | `-4.04` | `7.63` | `0.349` |
| same upstream, different downstream | `+18.34` | `18.67` | `2.054` |

The different-upstream controls are mostly near zero, but one broad AI upstream (`L19:2687 -> L24:2919`) still drops L24 mass by `-28.78` and class-3 logit by `1.525`. This means `L24:2919` is not exclusively driven by `L19:2943`; the anchor is much stronger, but the terminal also responds to broader AI-register upstreams.

Token-locus and max-length result:

| Max length | Node | Mean active tokens/doc | Mean mass/doc | Peak median index | Same L19/L24 peak token | Main active token classes |
|---:|---|---:|---:|---:|---:|---|
| 512 | `L19:2943` | `53.71` | `105.19` | `511` | `92.9%` | word `48.8%`, sentence boundary `20.9%`, punctuation `18.8%` |
| 512 | `L24:2919` | `44.07` | `151.48` | `511` | `92.9%` | word `50.2%`, sentence boundary `24.6%`, punctuation `14.9%` |
| 640 | `L19:2943` | `55.14` | `108.07` | `639` | `92.9%` | word `49.2%`, sentence boundary `20.7%`, punctuation `18.8%` |
| 640 | `L24:2919` | `45.36` | `156.95` | `639` | `92.9%` | word `50.6%`, sentence boundary `24.4%`, punctuation `15.1%` |

Max-length intervention comparison:

| Max length | Mean L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---:|---:|---:|---:|
| 512 | `-117.00` | `-11.04` | `2.054` |
| 640 | `-120.95` | `-11.03` | `2.181` |

Prefix-counterfactual result at max length 512:

| Variant | Rows | L19 mass | L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|---:|---:|
| original | 14 | `105.19` | `151.62` | `-117.03` | `-11.04` | `2.067` |
| first 160 words | 14 | `53.55` | `73.98` | `-56.87` | `-7.26` | `1.535` |
| first 240 words | 6 | `68.76` | `106.83` | `-77.73` | `-6.94` | `1.512` |

Prefix token-locus result:

| Variant | Node | Peak median index | Same L19/L24 peak token | Mean mass/doc | Main active token classes |
|---|---|---:|---:|---:|---|
| original | `L19:2943` | `511` | `100.0%` | `105.19` | word `48.9%`, sentence boundary `20.9%`, punctuation `18.8%` |
| original | `L24:2919` | `511` | `100.0%` | `151.62` | word `50.4%`, sentence boundary `24.6%`, punctuation `14.9%` |
| first 160 words | `L19:2943` | `486.5` | `92.9%` | `53.55` | word `39.3%`, sentence boundary `30.9%`, punctuation `21.2%` |
| first 160 words | `L24:2919` | `486.5` | `92.9%` | `73.98` | word `41.3%`, sentence boundary `36.5%`, punctuation `16.0%` |
| first 240 words | `L19:2943` | `418.5` | `66.7%` | `68.76` | word `43.4%`, sentence boundary `27.9%`, punctuation `20.5%` |
| first 240 words | `L24:2919` | `451.5` | `66.7%` | `106.83` | word `50.0%`, sentence boundary `31.1%`, punctuation `12.2%` |

Interpretation:

- This is a strong detector-relevant path: ablating `L19:2943` substantially collapses `L24:2919` and drops the AI class logit by about two points.
- The original "final-summary/capstone" label was too semantic. The max-length test shows that the peak follows the context-window boundary: when the pass changes from 512 to 640 tokens, both nodes move their peak from token 511 to token 639 with essentially unchanged mass and effect.
- Prefix counterfactuals show the path is not only a fixed 512/640 cutoff artifact. Shorter visible prefixes still activate the pair and remain causal, though with reduced mass. The pair tends to peak near the current visible ending and usually on the same token.
- The current best label is AI expository visible-prefix terminal accumulator. It is probably capturing accumulated polished expository/advice register at the current final visible segment, not necessarily a semantic final-summary sentence.
- Before treating this as a rhetorical capstone circuit, test on documents known to fit fully under the context cap and compare true endings against same-length prefixes that end mid-argument.

### BOUNDARY-2B: weak broad sibling of the BOUNDARY-2 upstream

Target: `L19:2943 -> L24:1076`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_capstone_sibling_L19_2943_to_L24_1076.csv`
- `docs/experiments/pangram_sae_circuit_target_ai_capstone_sibling_L19_2943_to_L24_1076.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_capstone_sibling_L19_2943_to_L24_1076.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_capstone_sibling_L19_2943_to_L24_1076.json`
- `artifacts/pangram_llama_sae/circuit_discovery/ai_capstone_sibling_token_loci_details.csv`
- `docs/experiments/pangram_sae_ai_capstone_sibling_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 21 | mostly Qwen | `120.72` | `371.48` | `-24.91` | `-0.22` | `2.176` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Peak median index | Same L19/L24 peak token | Main active token classes |
|---|---:|---:|---:|---:|---:|---:|---|
| `L19:2943` | 21 | 1,314 | `62.57` | `120.72` | `490` | `0.0%` | word `53.2%`, sentence boundary `18.3%`, punctuation `17.9%` |
| `L24:1076` | 21 | 3,110 | `148.10` | `371.48` | `497` | `0.0%` | word `79.3%`, punctuation `8.9%`, style/function word `4.8%` |

Interpretation:

- `L24:1076` has higher prompt-overlap edge score than `L24:2919`, but weaker mechanistic evidence. It is a broad word-heavy latent with many active tokens per document and no same-token peak overlap with `L19:2943`.
- The large class-3 logit drop is the upstream `L19:2943` score direction, not evidence that the score is mediated through `L24:1076`.
- This is useful as a sibling/control: prompt overlap alone can rank broad downstream latents highly, while token locality and downstream collapse distinguish the main terminal readout `L24:2919`.

### SCENE-1: tense scene sentence/line-break boundary circuit

Target: `L19:240 -> L24:3473`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_tense_scene_L19_240_to_L24_3473.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_tense_scene_L19_240_to_L24_3473.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_tense_scene_L19_240_to_L24_3473.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_tense_scene_L19_240_to_L24_3473.json`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_tense_scene_token_loci_details.csv`
- `docs/experiments/pangram_sae_mixed_tense_scene_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_tense_scene.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_tense_scene_summary.csv`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 32 | Gemini/GLM/Qwen/human | `127.05` | `199.30` | `-122.10` | `-6.52` | `-0.064` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean class-3 logit drop |
|---|---:|---:|---:|
| different upstream, same downstream | `-0.37` | `0.52` | `-0.004` |
| same upstream, different downstream | `-1.17` | `2.07` | `-0.064` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:240` | 32 | 1,629 | `50.91` | `127.05` | sentence boundary `41.0%`, word `20.4%`, line break `18.3%`, punctuation `15.9%` |
| `L24:3473` | 32 | 1,365 | `42.66` | `199.30` | sentence boundary `49.7%`, line break `21.8%`, punctuation `14.5%`, word `11.7%` |

Token-position overlap:

| Pair | Peak-token mean distance | Peak-token median distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:240` vs `L24:3473` | `22.4` | `0.0` | `75.0%` | `0.748` |

Interpretation:

- This is a clean local path-specific boundary circuit in tense scene prose. The downstream collapse is large and matched controls are near zero.
- It is not an AI-score circuit under the current classifier readout. The mean class-3 logit drop is slightly negative, so ablating this path marginally raises the AI score.
- The examples are dramatic scene/action passages with sensory pressure, fear, pursuit, doors, vehicles, cabins, weather, and bodily tension. The mechanism itself appears more local than semantic: it fires heavily on sentence boundaries and line breaks, with strong L19/L24 token-position overlap.
- This should be grouped with format/rhythm circuits (`MIXED-1`, `VERSE-1`, BOUNDARY variants), not with lexical slop circuits. A useful next check would compare paragraph joining and terminal-punctuation softening to determine whether it behaves like `MIXED-1` or is more strictly sentence-boundary local.

### PUNCT-1: sparse dialogue quote-punctuation circuit

Target: `L19:402 -> L24:1434`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_general_scene_L19_402_to_L24_1434.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_general_scene_L19_402_to_L24_1434.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_general_scene_L19_402_to_L24_1434.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_general_scene_L19_402_to_L24_1434.json`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_general_scene_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_general_scene_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_general_scene_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_general_scene.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_general_scene_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_mixed_general_scene_quote_removed_targets.csv`
- `docs/experiments/pangram_sae_counterfactual_mixed_general_scene_quote_removed_targets.md`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_mixed_general_scene_quote_removed_intervention_L19_402_to_L24_1434.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_mixed_general_scene_quote_removed_intervention_L19_402_to_L24_1434.json`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_mixed_general_scene_quote_removed_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_mixed_general_scene_quote_removed_token_loci_summary.csv`
- `docs/experiments/pangram_sae_counterfactual_mixed_general_scene_quote_removed_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 29 | Gemini/GLM/GPT/human/Qwen | `45.36` | `59.75` | `-54.37` | `-14.50` | `-0.092` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.03` | `0.06` | `-0.01` | `+0.035` |
| same upstream, different downstream | `+0.02` | `0.23` | `+0.02` | `-0.092` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:402` | 29 | 180 | `6.21` | `45.36` | punctuation `82.2%`, word `15.0%`, line break `2.2%` |
| `L24:1434` | 29 | 181 | `6.24` | `59.75` | punctuation `81.8%`, word `16.0%`, style/function word `1.7%` |

Token-position overlap:

| Pair | Peak-token mean distance | Peak-token median distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:402` vs `L24:1434` | `12.4` | `0.0` | `89.7%` | `0.825` |

Source mix:

| Source | Docs |
|---|---:|
| glm-5.2 | 11 |
| gemini-3.5-flash | 7 |
| gpt-5.5 | 7 |
| human | 2 |
| qwen3.6-35b | 2 |

Top-token context inspection:

| Node | Rows | Top tokens | Context pattern counts |
|---|---:|---|---|
| `L19:402` | 180 | `,"` 66, `,”` 24, `."` 17, `?"` 11, `!"` 7 | quote character in 174 rows; comma-quote pattern in 120; explicit dialogue verb in 74 |
| `L24:1434` | 181 | `,"` 66, `,”` 24, `."` 16, `?"` 11, `!"` 7 | quote character in 171 rows; comma-quote pattern in 119; explicit dialogue verb in 75 |

Quote-removal counterfactual:

The edit removes the closing quote mark immediately after comma, period, question-mark, or exclamation punctuation: regex `([,.!?])(["”]) => \1`. This preserves the surrounding sentence text but turns combined quote-punctuation tokens into bare punctuation.

| Variant | L19 mass | L19 max | L24 mass | L24 max | L24 mass delta after L19 ablation | L24 max delta after L19 ablation | Mean class-3 logit drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | `45.34` | `12.30` | `59.75` | `16.25` | `-54.38` | `-14.50` | `-0.091` |
| edited | `15.01` | `4.15` | `23.62` | `6.42` | `-15.87` | `-4.33` | `-0.006` |
| edited - original | `-30.33` | `-8.15` | `-36.14` | `-9.83` | `+38.51` | `+10.17` | `+0.085` |

Counterfactual token-locus shift:

| Node | Variant | Mean active tokens/doc | Mean mass/doc | Main active token classes | Top residual tokens |
|---|---|---:|---:|---|---|
| `L19:402` | original | `6.17` | `45.34` | punctuation `83.6%`, word `14.1%` | `,"`, `,”`, `."`, `?"`, `!"` |
| `L19:402` | edited | `7.00` | `15.01` | punctuation `51.5%`, word `36.1%`, sentence boundary `11.9%` | `,`, `?`, `!`, `she`, `he`, `said` |
| `L24:1434` | original | `6.24` | `59.75` | punctuation `81.8%`, word `16.0%` | `,"`, `,”`, `."`, `?"`, `!"` |
| `L24:1434` | edited | `7.55` | `23.62` | punctuation `49.0%`, word `34.1%`, sentence boundary `14.4%` | `,`, `?`, `!`, `.`, `she`, `Leo` |

Interpretation:

- This is a clean, path-specific local punctuation circuit. The anchor ablation removes almost all downstream mass and the downstream peak, while matched controls are effectively zero.
- It is more punctuation-specific and sparser than `MIXED-1` or `SCENE-1`: both nodes activate on only about six tokens per document, more than 80% of active loci are punctuation, and L19/L24 peaks are the same token in nearly 90% of documents.
- It should not be interpreted as a detector slop-score path. The class-3 logit movement is slightly negative, so ablating the path marginally raises the AI score.
- The counterfactual narrows the mechanism from generic punctuation to dialogue quote punctuation. Removing just the quote mark after punctuation removes about 60% of the downstream mass and peak activation, while residual activation follows the remaining bare punctuation and nearby dialogue words. The next control should distinguish whether dialogue tags add anything beyond quote-punctuation tokens themselves.

### CODE-1: programming-explanation boundary circuit

Target: `L19:731 -> L24:2110`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_code_explanation_L19_731_to_L24_2110.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_code_explanation_L19_731_to_L24_2110.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_code_explanation_L19_731_to_L24_2110.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_code_explanation_L19_731_to_L24_2110.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_code_explanation.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_code_explanation_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_code_explanation_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_code_explanation_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_code_explanation_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_code_explanation_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 32 | Gemini/GLM/GPT/human/Qwen | `56.34` | `92.47` | `-51.63` | `-5.32` | `-0.101` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.03` | `0.15` | `+0.004` | `+0.014` |
| same upstream, different downstream | `-0.48` | `0.71` | `-0.049` | `-0.101` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:731` | 32 | 882 | `27.56` | `56.34` | sentence boundary `33.8%`, punctuation `27.9%`, line break `20.2%`, word `11.2%` |
| `L24:2110` | 32 | 924 | `28.88` | `92.47` | sentence boundary `34.2%`, punctuation `28.9%`, line break `22.8%`, word `8.0%` |

Token-position overlap:

| Pair | Peak-token mean distance | Peak-token median distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:731` vs `L24:2110` | `10.1` | `0.0` | `90.6%` | `0.788` |

Top-token context inspection:

| Node | Top tokens | Context pattern counts |
|---|---|---|
| `L19:731` | `.` 127, `,` 73, `.\n\n` 39, `\n\n` 8, `//` 6, `}\n\n` 5, `;\n` 5 | code-term contexts 203; Python contexts 29; TypeScript contexts 20; explanatory-marker contexts 72 |
| `L24:2110` | `.` 128, `,` 73, `.\n\n` 43, `\n\n` 9, `//` 7, `:\n\n` 6, `}\n\n` 5 | code-term contexts 204; Python contexts 32; TypeScript contexts 18; explanatory-marker contexts 76 |

Interpretation:

- This is a clean path-specific dependency: alternate upstreams into `L24:2110` barely move downstream mass, while ablating `L19:731` collapses roughly half the downstream mass and max activation.
- The score movement follows the upstream rather than the downstream path. Same-upstream/different-downstream controls preserve the `-0.101` class-3 logit movement, while downstream collapse is specific to `L24:2110`.
- The mechanism is local and boundary-like, but the domain is programming explanations rather than narrative. Peaks are usually periods, commas, paragraph breaks, or code punctuation in passages explaining functions, types, arrays, binary search, TypeScript mapped types, and similar code concepts.
- Because the class-3 movement is negative, this is a non-slop or human/non-AI-style control path. It is useful for contrasting code/explanation formatting circuits against AI expository slop circuits.

### EXPO-1: expository sentence/paragraph period boundary circuit

Target: `L19:3615 -> L24:664`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_general_expository_L19_3615_to_L24_664.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_general_expository_L19_3615_to_L24_664.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_general_expository_L19_3615_to_L24_664.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_general_expository_L19_3615_to_L24_664.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_general_expository.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_general_expository_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_general_expository_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_general_expository_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_general_expository_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_general_expository_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 17 | human/GPT-heavy mixed | `63.42` | `121.78` | `-56.83` | `-4.82` | `0.271` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.08` | `0.10` | `-0.003` | `-0.023` |
| same upstream, different downstream | `+0.33` | `0.98` | `+0.103` | `+0.271` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:3615` | 17 | 584 | `34.35` | `63.42` | sentence boundary `49.2%`, line break `24.0%`, word `16.9%`, punctuation `8.2%` |
| `L24:664` | 17 | 677 | `39.82` | `121.78` | sentence boundary `56.7%`, line break `25.0%`, punctuation `10.4%`, word `9.9%` |

Token-position overlap:

| Pair | Peak-token mean distance | Peak-token median distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:3615` vs `L24:664` | `12.9` | `0.0` | `76.5%` | `0.641` |

Top-token context inspection:

| Node | Top tokens | Context pattern counts |
|---|---|---|
| `L19:3615` | `.` 90, `.\n\n` 35, `,` 14, `.\n` 4 | general-expository contexts 106; polished-AI contexts 4; format contexts 8 |
| `L24:664` | `.` 108, `.\n\n` 36, `,` 16, `,\n` 5 | general-expository contexts 116; polished-AI contexts 6; format contexts 11 |

Interpretation:

- This is a clean path-specific local boundary circuit. Alternate upstreams into `L24:664` have near-zero downstream effect, while the anchor drops downstream mass by `-56.83`.
- The score movement follows the upstream: same-upstream/different-downstream controls preserve the `0.271` class-3 logit drop without reproducing the L24 collapse.
- The local feature is mostly sentence/paragraph boundary punctuation in general expository prose. The examples include human praise/exposition, educational explanations, GPT informational prose, and poem-like passages, so the current evidence supports a boundary-mechanism label more strongly than a semantic register label.
- The next test should edit paragraph-final periods or join sentence boundaries to see whether the path behaves like `BOUNDARY-1`/`BOUNDARY-3` or is a separate expository-period feature.

### EXPO-1B: broad simple-continuation feed into EXPO-1 boundary terminal

Target: `L19:3990 -> L24:664`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_human_simple_to_expo_boundary_L19_3990_to_L24_664.csv`
- `docs/experiments/pangram_sae_circuit_target_human_simple_to_expo_boundary_L19_3990_to_L24_664.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_simple_to_expo_boundary_L19_3990_to_L24_664.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_simple_to_expo_boundary_L19_3990_to_L24_664.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_human_simple_to_expo_boundary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_simple_to_expo_boundary_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_simple_to_expo_boundary_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/human_simple_to_expo_boundary_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/human_simple_to_expo_boundary_token_loci_summary.csv`
- `docs/experiments/pangram_sae_human_simple_to_expo_boundary_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 16 | human/GPT | `1272.22` | `144.91` | `-75.40` | `-2.87` | `+0.641` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `+5.21` | `5.77` | `+0.221` | `+0.223` |
| same upstream, different downstream | `+1.06` | `1.23` | `+0.086` | `+0.641` |

Token-locus result:

| Node | Docs | Mean active tokens/doc | Mean mass/doc | Mean max/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:3990` | 16 | `283.81` | `1272.22` | `8.02` | word `78.7%`, punctuation `6.9%`, sentence boundary `4.1%`, style/function word `4.1%`, line break `2.4%` |
| `L24:664` | 16 | `48.19` | `144.91` | `9.69` | word `37.2%`, sentence boundary `25.2%`, punctuation `17.6%`, line break `14.1%`, style/function word `2.5%` |

Token-position overlap:

| Pair | Mean peak-token distance | Median peak-token distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:3990` vs `L24:664` | `34.88` | `22.0` | `0.0%` | `0.059` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:3990` | `and`, comma, `He`, `for`, `the`, `is`, `to`, `They`, `he`, period, `his`, `they`, `their`, `It` | broad simple-continuation / connective prose mass |
| `L24:664` | period 98, paragraph-final period 40, comma 14, colon 9, comma-newline 5 | sparse sentence/paragraph boundary terminal already seen in `EXPO-1` |

Interpretation:

- This is a real cross-feed into the `EXPO-1` downstream. Ablating `L19:3990` drops `L24:664` by `-75.40` mass, while matched different-upstream/same-downstream controls average only `+5.21`.
- It is not a second same-token expository period circuit. The upstream is broad and word/connective-heavy, the downstream is sparse and punctuation/boundary-heavy, and the peak tokens never match.
- The classifier-score movement follows `L19:3990`: same-upstream/different-downstream controls preserve the `+0.641` class-3 logit drop without reproducing the `L24:664` collapse.
- Treat this as broad simple-continuation context feeding an expository boundary terminal. It helps decompose `L24:664` into local boundary input (`EXPO-1`) versus broader context input (`EXPO-1B`).

### EXPO-2: definitional/explanatory sentence-boundary circuit

Target: `L19:75 -> L24:2341`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown4_L19_75_to_L24_2341.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_unknown4_L19_75_to_L24_2341.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown4_L19_75_to_L24_2341.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown4_L19_75_to_L24_2341.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_unknown4.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown4_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown4_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown4_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown4_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_unknown4_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 19 | Gemini/GLM/GPT/human/Qwen | `33.90` | `60.83` | `-27.27` | `-3.77` | `-0.010` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.286` | `0.288` | `-0.001` | `+0.007` |
| same upstream, different downstream | `+0.067` | `0.077` | `+0.001` | `-0.010` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:75` | 19 | 368 | `19.37` | `33.90` | word `35.1%`, sentence boundary `33.2%`, punctuation `14.1%`, style/function word `8.4%`, line break `7.1%` |
| `L24:2341` | 19 | 344 | `18.11` | `60.83` | sentence boundary `37.2%`, word `27.9%`, punctuation `16.6%`, line break `8.4%`, style/function word `6.1%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:75` vs `L24:2341` | `89.5%` | `0.753` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:75` | period 109, comma 30, `.\n\n` 12, `makes` 4, `.\n` 4, `The` 3, `and` 3, `because` 3, `When` 3 | definitional/explanatory paragraph boundaries in short factual answers about hair loss, cornrows, coat types, nanoparticles, bees, milk, ceramics, hydrogen bonds, tomatoes, ticks, plants, plastic, and taste |
| `L24:2341` | period 119, comma 36, `.\n\n` 15, `In` 4, `.\n` 4, `and` 3, `because` 3, `such` 2 | same explanatory-period family, with stronger downstream period and discourse-word mass |

Interpretation:

- This is a clean, small expository boundary circuit. The anchor ablation drops L24 mass by `-27.27`, while matched controls are close to zero.
- It is not a detector-score path: class-3 movement is essentially zero.
- The local feature is sentence-boundary punctuation and nearby explanatory discourse in definitional paragraphs. It is more token-local than `EXPO-1`, with `89.5%` same-peak overlap.
- Treat it as a sibling of `EXPO-1` and `DESCR-1`, useful for decomposing period-boundary behavior in explanatory prose rather than identifying slop style directly.

### CHILD-1: child/simple educational sentence-boundary circuit

Target: `L19:755 -> L24:2595`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_child_educational_L19_755_to_L24_2595.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_child_educational_L19_755_to_L24_2595.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_child_educational_L19_755_to_L24_2595.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_child_educational_L19_755_to_L24_2595.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_child_educational.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_child_educational_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_child_educational_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_child_educational_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_child_educational_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_child_educational_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 20 | human-heavy mixed | `143.27` | `249.45` | `-129.78` | `-4.37` | `0.218` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-1.23` | `1.28` | `-0.032` | `+0.041` |
| same upstream, different downstream | `-0.13` | `1.49` | `+0.001` | `+0.218` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:755` | 20 | 1,387 | `69.35` | `143.27` | sentence boundary `77.1%`, line break `13.3%`, punctuation `6.7%` |
| `L24:2595` | 20 | 1,313 | `65.65` | `249.45` | sentence boundary `75.4%`, line break `17.1%`, punctuation `5.4%` |

Token-position overlap:

| Pair | Peak-token mean distance | Peak-token median distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:755` vs `L24:2595` | `21.3` | `0.0` | `75.0%` | `0.702` |

Top-token context inspection:

| Node | Top tokens | Context pattern counts |
|---|---|---|
| `L19:755` | `.` 170, `.\n\n` 23, `!` 13, `,` 8, opening quote 6 | child-story contexts 57; simple-explanation contexts 40; line/list contexts 8 |
| `L24:2595` | `.` 166, `.\n\n` 28, `!` 12, `,` 7, closing-quote paragraph break 5 | child-story contexts 55; simple-explanation contexts 43; line/list contexts 10 |

Interpretation:

- This is a clean, path-specific local boundary circuit. The anchor ablation drops L24 mass by `-129.78`, while matched controls are close to zero.
- The score movement follows the upstream feature: same-upstream/different-downstream controls preserve the `0.218` class-3 logit drop without reproducing the L24 collapse.
- The local feature is sentence/paragraph boundary punctuation in simple child stories and child-facing explanations. Examples include little-boy/little-girl stories, baby-sister narratives, classroom/space explanations, opal/dinosaur explanations, and simple religious/educational stories.
- This looks like a sibling of `EXPO-1`: both are score-positive boundary circuits, but `CHILD-1` is more specialized to simple/child prose and has higher downstream mass. Compare both against BOUNDARY-1/3 before claiming a general period-circuit family.

### DESCR-1: elevated descriptive/expository sentence-boundary circuit

Target: `L19:3992 -> L24:943`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_fantasy_descriptive_L19_3992_to_L24_943.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_fantasy_descriptive_L19_3992_to_L24_943.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_fantasy_descriptive_L19_3992_to_L24_943.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_fantasy_descriptive_L19_3992_to_L24_943.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_fantasy_descriptive.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_fantasy_descriptive_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_fantasy_descriptive_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_fantasy_descriptive_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_fantasy_descriptive_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_fantasy_descriptive_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 21 | Gemini/GLM/GPT/human/Qwen | `49.93` | `78.83` | `-48.08` | `-5.74` | `-0.055` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean class-3 logit drop |
|---|---:|---:|---:|
| different upstream, same downstream | `-0.25` | `0.29` | `-0.010` |
| same upstream, different downstream | `-0.69` | `0.93` | `-0.055` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:3992` | 21 | 613 | `29.19` | `49.93` | sentence boundary `36.1%`, punctuation `25.4%`, word `23.5%`, line break `9.6%` |
| `L24:943` | 21 | 522 | `24.86` | `78.83` | sentence boundary `43.3%`, punctuation `26.1%`, word `15.3%`, line break `11.1%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:3992` vs `L24:943` | `95.2%` | `0.730` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:3992` | `.` 153, `,` 43, `.\n\n` 15, `;` 4, line break 4 | elevated descriptive or expository sentences; character descriptions, cosmic/space exposition, historical exposition, child/fantasy stories |
| `L24:943` | `.` 177, `,` 26, `.\n\n` 15, `;` 4, `but` 4 | same sentence-boundary loci with stronger period mass; occasional transition words immediately after periods |

Interpretation:

- This is a clean path-specific local boundary circuit. The anchor ablation drops L24 mass by `-48.08`, while different-upstream controls into `L24:943` average only `-0.25`.
- The score movement is small and negative: ablating the upstream slightly raises the AI score rather than lowering it. Same-upstream/different-downstream controls preserve that small score movement without reproducing the L24 collapse.
- The target examples look like fantasy, science-fiction, historical, review, and polished educational prose, but the local evidence points to sentence-boundary punctuation and paragraph rhythm rather than genre words. Same-token peak overlap is unusually high at `95.2%`.
- This should be grouped with the local boundary family (`SCENE-1`, `EXPO-1`, `CHILD-1`) as a descriptive/expository subtype. It is not currently evidence for a compact AI-slop style feature.

### HUMOR-1: humor/joke/laughter and social-scene word path

Target: `L19:390 -> L24:2926`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_story_style_L19_390_to_L24_2926.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_story_style_L19_390_to_L24_2926.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_story_style_L19_390_to_L24_2926.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_story_style_L19_390_to_L24_2926.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_story_style.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_story_style_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_story_style_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_story_style_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_story_style_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_story_style_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 15 | all five sources | `144.13` | `187.99` | `-158.90` | `-7.03` | `-0.018` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.154` | `0.216` | `-0.002` | `+0.001` |
| same upstream, different downstream | `+0.057` | `0.183` | `+0.001` | `-0.018` |

Token-locus result:

| Node | Docs | Mean active tokens/doc | Mean mass/doc | Mean max/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:390` | 15 | `68.53` | `144.13` | `7.89` | word `79.4%`, punctuation `10.6%`, sentence boundary `3.3%`, style/function word `2.5%`, line break `1.8%` |
| `L24:2926` | 15 | `68.60` | `187.99` | `10.15` | word `77.2%`, punctuation `10.0%`, sentence boundary `4.8%`, style/function word `2.8%`, line break `2.6%` |

Token-position overlap:

| Pair | Mean peak-token distance | Median peak-token distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:390` vs `L24:2926` | `14.80` | `0.0` | `66.7%` | `0.544` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:390` | `a`, comma, `and`, `joke`, `at`, `laughed`, `the`, `is`, `of`, `jokes`, `made`, `cruel`, `Theo`, `laugh` | humor, joking, laughter, social embarrassment, and social-scene wording |
| `L24:2926` | comma, `a`, `the`, `at`, `and`, period, `joke`, `of`, `with`, `jokes`, `made`, `cruel`, `George`, `fun`, `else` | same humor/social-scene lexical family with strong token alignment |

Interpretation:

- This is a clean path-specific lexical/style circuit. Ablating `L19:390` drops `L24:2926` mass by `-158.90` and max by `-7.03`, while both matched-control families average near zero downstream movement.
- It is not a detector-score path. The class-3 logit movement is essentially zero (`-0.018`), despite the strong downstream collapse.
- The examples are mixed-source and include comedy recommendations, explanations of jokes and shaggy-dog stories, social scenes with laughing or joking, jokester siblings, comedy/YouTube advice, and some broader narrative/social contexts. The active tokens are mostly ordinary words, but they are much more local and token-aligned than the broad-register paths.
- Treat this as a useful non-slop lexical/style circuit. It is a good contrast case for future slop-signature work: causal SAE paths can be clean and human-readable without being AI-detection evidence.

### STEM-1: `st`/`St` lexical-piece path

Target: `L19:191 -> L24:779`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_expository_general_L19_191_to_L24_779.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_expository_general_L19_191_to_L24_779.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_expository_general_L19_191_to_L24_779.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_expository_general_L19_191_to_L24_779.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_expository_general.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_expository_general_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_expository_general_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_expository_general_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_expository_general_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_expository_general_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 24 | Gemini/GLM/GPT/human/Qwen | `31.24` | `37.22` | `-36.60` | `-8.64` | `-0.005` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `+0.06` | `0.08` | `-0.000` | `+0.000` |
| same upstream, different downstream | `-0.02` | `0.05` | `-0.006` | `-0.005` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:191` | 24 | 212 | `8.83` | `31.24` | word `99.5%`, line break `0.5%` |
| `L24:779` | 24 | 210 | `8.75` | `37.22` | word `98.4%`, line break/punctuation/discourse `1.6%` |

Token-position overlap:

| Pair | Peak-token mean distance | Peak-token median distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:191` vs `L24:779` | `0.0` | `0.0` | `100.0%` | `0.859` |

Top-token context inspection:

| Node | Top tokens |
|---|---|
| `L19:191` | `storm` 15, `St` 16 total across space/no-space variants, `stars` 4, `stake` 4, `holder` 4, `stood` 3, `storms` 3, `established` 3 |
| `L24:779` | `storm` 15, `St` 16 total across space/no-space variants, `stars` 4, `stake` 4, `holder` 4, `stood` 3, `storms` 3, `station`/`stargazer` examples |

Interpretation:

- This is a very clean path-specific dependency, but it is not a style circuit. The nodes share exact peak positions in every target document, controls are near zero, and the active tokens are almost entirely a lexical/subword family around `st`/`St`.
- The target-export examples initially looked mixed/expository because the shared prompts included code, fantasy, letters, storms, and list-like prose. Token-locus inspection resolves the ambiguity: the active feature is the lexical piece, not the document genre.
- The near-zero class-3 movement confirms that this path should be treated as a lexical control. It is useful for calibrating the discovery pipeline and for filtering future candidates where token identity explains the edge.

### S-LEX-1: `s`/`S` lexical-word path

Target: `L19:627 -> L24:4066`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_weak_L19_627_to_L24_4066.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_weak_L19_627_to_L24_4066.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_weak_L19_627_to_L24_4066.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_weak_L19_627_to_L24_4066.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_weak.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_weak_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_weak_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_weak_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_weak_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_weak_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 17 | Gemini/GLM/GPT/human/Qwen | `51.18` | `54.13` | `-53.79` | `-6.28` | `-0.005` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.019` | `0.134` | `-0.000` | `+0.002` |
| same upstream, different downstream | `-0.031` | `0.131` | `-0.005` | `-0.005` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:627` | 17 | 362 | `21.29` | `51.18` | word `97.0%`, discourse marker `1.4%`, expanded-negation/punctuation/style/whitespace `1.6%` |
| `L24:4066` | 17 | 315 | `18.53` | `54.13` | word `96.8%`, discourse marker `1.6%`, expanded-negation/punctuation/style `1.6%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:627` vs `L24:4066` | `100.0%` | `0.803` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:627` | `silence` 6, `sleep` 3, `specific` 3, `sound` 3, `social` 3, `so` 3, `sun` 3, `some` 3, `Sunday` 3, `sustainable` 2, `sustainability` 2 | mixed prose; the shared property is `s`/`S` lexical form, not document genre |
| `L24:4066` | `silence` 6, `sleep` 4, `small` 4, `specific` 3, `social` 3, `so` 3, `see` 3, `saw` 3, `Sunday` 3, `sustainable` 2, `sustainability` 2 | same lexical family with slightly different high-frequency members |

Interpretation:

- This is a clean path-specific dependency, but it is lexical rather than stylistic. The anchor ablation drops downstream mass by `-53.79`, while matched controls are essentially zero.
- Both nodes activate almost exclusively on word tokens, and the strongest shared property is an `s`/`S` initial or `s`-heavy lexical pattern: `sustainable`, `sleep`, `silence`, `sauce`, `social`, `Sunday`, `specific`, `saw`, `see`, and related words.
- The class-3 movement is effectively zero, so this should be treated as a positive lexical/subword control. Together with `STEM-1`, it shows that the direct-edge scan can find exact token-identity paths that should be filtered before claiming human-readable style circuits.

### BROAD-2: GPT recommendation/review register

Target: `L19:991 -> L24:3466`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_gpt_recommendation_review_L19_991_to_L24_3466.csv`
- `docs/experiments/pangram_sae_circuit_target_ai_gpt_recommendation_review_L19_991_to_L24_3466.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_gpt_recommendation_review_L19_991_to_L24_3466.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_gpt_recommendation_review_L19_991_to_L24_3466.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_ai_gpt_recommendation_review.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_ai_gpt_recommendation_review_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_ai_gpt_recommendation_review_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/ai_gpt_recommendation_review_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/ai_gpt_recommendation_review_token_loci_summary.csv`
- `docs/experiments/pangram_sae_ai_gpt_recommendation_review_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 20 | GPT-5.5 only | `655.10` | `840.94` | `-807.03` | `-6.59` | `0.105` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.73` | `1.25` | `-0.006` | `-0.006` |
| same upstream, different downstream | `+1.12` | `1.75` | `+0.043` | `+0.105` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:991` | 20 | 5,465 | `273.25` | `655.10` | word `74.6%`, punctuation `15.0%`, style/function word `7.1%` |
| `L24:3466` | 20 | 5,132 | `256.60` | `840.94` | word `68.8%`, punctuation `17.5%`, style/function word `7.5%`, sentence boundary `3.3%` |

Token-position overlap:

| Pair | Peak-token mean distance | Peak-token median distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:991` vs `L24:3466` | `15.8` | `0.0` | `60.0%` | `0.613` |

Top-token context inspection:

| Node | Top tokens | Context pattern counts |
|---|---|---|
| `L19:991` | `,` 26, `is` 20, `and` 15, `are` 12, `a` 10, `it`/`It` 10, `you` 4, `but` 4, `can` 4, `also` 4 | recommendation/review contexts 29; list-marker contexts 41 |
| `L24:3466` | `,` 29, `is` 20, `and` 17, `a` 13, `are` 10, `.` 8, en-dash 6, `be` 5 | recommendation/review contexts 30; list-marker contexts 36 |

Interpretation:

- This is a strong, path-specific broad GPT-5.5 register path. The anchor ablation collapses almost all `L24:3466` mass, while different-upstream same-downstream controls barely move the downstream latent.
- The final score movement follows the upstream feature: same-upstream/different-downstream controls preserve the `0.105` class-3 logit drop while not reproducing the `L24:3466` collapse.
- The feature is not compact or token-local in the way `PUNCT-1`, `CODE-1`, or `STEM-1` are. It activates across hundreds of ordinary tokens per document in practical recommendation, review, option-list, and advice prose.
- The right interpretation is broad GPT recommendation/review register. It is a plausible high-level slop-style family, but the next step should be decomposition into smaller local features rather than claiming the full broad latent as one human-readable circuit.

### BROAD-3: polished assistant/expository register

Target: `L19:2236 -> L24:2070`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown2_L19_2236_to_L24_2070.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_unknown2_L19_2236_to_L24_2070.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown2_L19_2236_to_L24_2070.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown2_L19_2236_to_L24_2070.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_unknown2.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown2_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown2_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown2_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown2_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_unknown2_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 13 | GLM/Gemini/Qwen | `729.58` | `909.30` | `-844.14` | `-4.33` | `+0.046` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-1.83` | `2.36` | `-0.008` | `+0.005` |
| same upstream, different downstream | `-7.54` | `8.57` | `-0.001` | `+0.046` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:2236` | 13 | 4,219 | `324.54` | `729.58` | word `82.2%`, punctuation `7.4%`, sentence boundary `3.0%`, style/function word `3.0%`, line break `2.0%` |
| `L24:2070` | 13 | 4,173 | `321.00` | `909.30` | word `82.6%`, punctuation `7.3%`, style/function word `3.0%`, sentence boundary `2.9%`, line break `1.9%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:2236` vs `L24:2070` | `23.1%` | `0.385` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:2236` | comma 24, period 8, `a` 8, `is` 6, `the` 5, `and` 5, `to` 4, `]:` 4, `with` 3, `are` 3, `was` 3 | polished assistant/expository prose: customer-service self-description, Rio travel copy, survey-question design, fantasy name suggestions, resume advice, Python explanation, narrative scene, corporate profile, list entries, LHC exposition, Antarctica travel |
| `L24:2070` | comma 25, `is` 6, `and` 5, `to` 4, `was` 4, `the` 3, `a` 3, `you` 3, `It` 3, `have` 2, `code` 2 | same broad assistant/expository register with stronger downstream mass but low peak-token alignment |

Interpretation:

- This is a strong path-specific broad-register edge. The anchor ablation drops `L24:2070` mass by `-844.14`, while matched-control families are one to two orders of magnitude smaller.
- It is mildly score-positive under the current classifier, but the score effect is small compared with `BROAD-1`.
- This is not a local style circuit. Both nodes fire on more than 300 tokens per document, mostly ordinary words, and peak overlap is low.
- Treat it as a third broad model-produced assistant-register path alongside `BROAD-1` and `BROAD-2`. It should be decomposed into smaller local circuits before assigning a readable slop signature.

### BROAD-4: Qwen polished explanatory/opener register

Target: `L19:838 -> L24:3422`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown11_L19_838_to_L24_3422.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_unknown11_L19_838_to_L24_3422.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown11_L19_838_to_L24_3422.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown11_L19_838_to_L24_3422.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_unknown11.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown11_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown11_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown11_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown11_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_unknown11_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 14 | mostly Qwen, plus Gemini/GLM | `352.15` | `612.66` | `-378.35` | `-5.22` | `+0.047` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `+1.80` | `2.32` | `+0.020` | `+0.001` |
| same upstream, different downstream | `+0.31` | `0.39` | `+0.045` | `+0.047` |

Token-locus result:

| Node | Docs | Mean active tokens/doc | Mean mass/doc | Mean max/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:838` | 14 | `168.21` | `352.15` | `5.53` | word `80.8%`, punctuation `8.5%`, sentence boundary `3.9%`, style/function word `3.6%`, discourse marker `1.4%` |
| `L24:3422` | 14 | `183.07` | `612.66` | `9.08` | word `82.4%`, punctuation `8.1%`, style/function word `3.7%`, sentence boundary `3.2%`, discourse marker `1.4%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:838` vs `L24:3422` | `64.3%` | `0.483` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:838` | comma 19, `This` 4, `a` 4, `to` 4, `the` 3, `and` 2, `line` 2, `landscape` 2, `around` 2, `with` 2 | polished explanatory openings and setup clauses, mostly Qwen |
| `L24:3422` | comma 17, `line` 3, `landscape` 3, `This` 2, `that` 2, `effects` 2, `accountant` 2, `issy` 2 | same broad explanatory/opening register with moderate token-position alignment |

Interpretation:

- This is a strong path-specific broad AI-register edge. The anchor ablation drops `L24:3422` mass by `-378.35` and max activation by `-5.22`, while both matched-control families stay near zero on downstream mass.
- The target set is mostly Qwen and consists of polished explanatory openings or setup-heavy prose: WFH post-COVID, emoji narrative explanation, threat assessment, invented sport, glass-half-full metaphor, dangerous hikes, horror films, professional tripods, CPA/accountant, health anxiety, basketball tactics, UFO road trip, Gilligan/Chrissy, and climate-change title framing.
- The feature is document-wide rather than compact. Both nodes activate on more than 160 tokens per document and mostly on ordinary words, though they are more token-aligned than `BROAD-3`.
- Treat this as a fourth broad model-produced register path. It is useful for mapping the model's broad AI-style directions, but should be decomposed before turning it into a human-readable slop signature.

### BROAD-5: GPT/Qwen sentimental reflective narrative closure register

Target: `L19:3600 -> L24:4083`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_model_story_L19_3600_to_L24_4083.csv`
- `docs/experiments/pangram_sae_circuit_target_ai_model_story_L19_3600_to_L24_4083.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_model_story_L19_3600_to_L24_4083.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_model_story_L19_3600_to_L24_4083.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_ai_model_story.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_ai_model_story_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_ai_model_story_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/ai_model_story_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/ai_model_story_token_loci_summary.csv`
- `docs/experiments/pangram_sae_ai_model_story_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 15 | GPT/Qwen | `260.33` | `396.93` | `-130.53` | `-1.53` | `+0.003` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.076` | `0.241` | `+0.000` | `+0.000` |
| same upstream, different downstream | `+0.484` | `0.816` | `+0.011` | `+0.003` |

Token-locus result:

| Node | Docs | Mean active tokens/doc | Mean mass/doc | Mean max/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:3600` | 15 | `162.93` | `260.33` | `4.48` | word `74.9%`, punctuation `7.9%`, expanded-negation word `4.7%`, sentence boundary `4.3%`, style/function word `4.3%` |
| `L24:4083` | 15 | `181.00` | `396.93` | `5.55` | word `74.0%`, punctuation `9.0%`, sentence boundary `5.8%`, style/function word `3.8%`, expanded-negation word `3.7%` |

Token-position overlap:

| Pair | Mean peak-token distance | Median peak-token distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:3600` vs `L24:4083` | `10.27` | `7.0` | `13.3%` | `0.390` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:3600` | `was`, `to`, `the`, `that`, comma, `in`, `had`, `he`, `finally`, `of`, `I`, `never`, `quiet`, `peace` | sentimental reflective narrative closure, inner transformation, gratitude, loneliness, and resolution beats |
| `L24:4083` | comma, `was`, `that`, period, `to`, opening quote, `for`, `and`, `now`, `but`, `me`, `finally`, `Maybe`, `he` | same GPT/Qwen reflective narrative register, with stronger punctuation and dialogue-adjacent mass |

Interpretation:

- This is a strong path-specific broad model-output dependency. Ablating `L19:3600` drops `L24:4083` mass by `-130.53`, while both matched-control families stay near zero.
- It is not a detector-score path: mean class-3 movement is only `+0.003`.
- The target set is GPT/Qwen only and concentrates on sentimental or reflective narrative closures: farewell letters, quiet transformation arcs, graduation anxiety, chalk-boundary metaphors, falling-apple perspective, martial-arts self-mastery, Bible-translation documentary scenes, isolated villages, escape from family conflict, romantic coffee-shop framing, and lonely neighborly resolution.
- Treat this as a fifth broad model-produced register path and a decomposition target. It is more semantically coherent than generic broad word mass, but still too document-wide to call a compact slop marker.

### PRACT-1: practical advice/list closure and paragraph-rhythm path

Target: `L19:2905 -> L24:3226`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_practical_craft_seo_L19_2905_to_L24_3226.csv`
- `docs/experiments/pangram_sae_circuit_target_ai_practical_craft_seo_L19_2905_to_L24_3226.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_practical_craft_seo_L19_2905_to_L24_3226.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_practical_craft_seo_L19_2905_to_L24_3226.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_ai_practical_craft_seo.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_ai_practical_craft_seo_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_ai_practical_craft_seo_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/ai_practical_craft_seo_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/ai_practical_craft_seo_token_loci_summary.csv`
- `docs/experiments/pangram_sae_ai_practical_craft_seo_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 19 | Gemini/GLM/GPT/human/Qwen | `35.63` | `46.61` | `-27.38` | `-5.25` | `-0.141` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `+0.004` | `0.031` | `+0.001` | `-0.018` |
| same upstream, different downstream | `-0.094` | `0.498` | `-0.074` | `-0.141` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:2905` | 19 | 394 | `20.74` | `35.63` | word `36.3%`, sentence boundary `24.1%`, line break `21.1%`, punctuation `11.9%` |
| `L24:3226` | 19 | 344 | `18.11` | `46.61` | word `39.2%`, sentence boundary `24.4%`, line break `23.0%`, punctuation `7.0%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:2905` vs `L24:3226` | `100.0%` | `0.693` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:2905` | `.` 52, `.\n\n` 48, `,` 18, line break 6, `!` 5, `These` 2, `Lastly` 2 | practical/list advice endings: meal prep, recruiting, cabinets, cat care, SEO, journaling, Christmas decorations, retirement, local dining |
| `L24:3226` | `.` 56, `.\n\n` 54, `,` 15, line break 6, `!` 5, `the` 3, `These` 2, `Lastly` 2 | same closure/paragraph rhythm with stronger paragraph-break and final-period mass |

Interpretation:

- This is a path-specific practical/list closure feature. The anchor ablation drops downstream mass by `-27.38`, while different-upstream controls into `L24:3226` average only `+0.004`.
- The classifier direction is negative: ablating `L19:2905` raises the AI score. Same-upstream/different-downstream controls preserve that negative score movement without reproducing the L24 collapse.
- The surface genre looks close to generated advice slop: meal prep ideas, recruiting tips, kitchen cabinet updates, cat-care guidance, SEO advice, journaling approaches, decoration ideas, and retirement suggestions. The local feature, however, is paragraph/list closure rhythm: paragraph-final periods, `.\n\n`, `These`, `With`, `Lastly`, and short practical conclusion sentences.
- This should be used as a human/non-AI practical-list control when decomposing broad advice-register paths such as `BROAD-2`, not as a positive AI-slop circuit.

### SCORE-ONLY-1: letter/email upstream without downstream edge

Target screened: `L19:1049 -> L24:3971`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_letter_email_L19_1049_to_L24_3971.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_letter_email_L19_1049_to_L24_3971.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_letter_email_L19_1049_to_L24_3971.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_letter_email_L19_1049_to_L24_3971.json`

Intervention result:

| Docs | Mean L24 mass delta after L19 ablation | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---:|---:|---:|---:|---:|
| 24 | `+0.03` | `0.21` | `-0.001` | `+0.259` |

Interpretation:

- This is a score-only upstream result, not a circuit edge. The classifier score moves strongly when `L19:1049` is ablated, but the nominated downstream latent `L24:3971` barely changes.
- Do not spend matched-control or token-locus time on this pair unless a later scan identifies a different downstream latent for `L19:1049`.
- Keep `L19:1049` in mind as a possible standalone letter/email or register latent, but remove `L19:1049 -> L24:3971` from the circuit queue.

### WEAK-EDGE-1: alternate letter/email upstream with weak downstream edge

Target screened: `L19:3424 -> L24:3971`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_letter_email_alt_L19_3424_to_L24_3971.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_letter_email_alt_L19_3424_to_L24_3971.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_letter_email_alt_L19_3424_to_L24_3971.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_letter_email_alt_L19_3424_to_L24_3971.json`

Intervention result:

| Docs | Mean L24 mass delta after L19 ablation | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---:|---:|---:|---:|---:|
| 21 | `-1.91` | `1.91` | `-0.099` | `+0.053` |

Interpretation:

- This is not a convincing circuit edge. Unlike the confirmed path-specific circuits, downstream mass movement is only about two activation units and max movement is about one tenth of an activation unit.
- This second weak result suggests `L24:3971` is a poor downstream target in the current direct-edge queue. Deprioritize additional `L24:3971` pairings unless their ablation effect is much larger.

### WEAK-EDGE-2: alternate code/explanation upstream with weak downstream edge

Target screened: `L19:1878 -> L24:2110`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_code_explanation_alt_L19_1878_to_L24_2110.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_code_explanation_alt_L19_1878_to_L24_2110.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_code_explanation_alt_L19_1878_to_L24_2110.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_code_explanation_alt_L19_1878_to_L24_2110.json`

Intervention result:

| Docs | Mean L24 mass delta after L19 ablation | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---:|---:|---:|---:|---:|
| 31 | `-1.29` | `1.29` | `-0.046` | `-0.139` |

Interpretation:

- This is not a convincing alternate upstream for the `CODE-1` downstream. The classifier score moves somewhat through `L19:1878`, but `L24:2110` barely moves.
- The confirmed `CODE-1` path `L19:731 -> L24:2110` drops the same downstream by `-51.63` mass, so this `-1.29` effect is too small to warrant matched controls or token-locus inspection.
- Keep `L24:2110` anchored to `L19:731` for now. This is a useful reminder that downstream siblings must pass the same causal screen as fresh direct-edge candidates.

### WEAK-EDGE-3: alternate code/explanation downstream with weak edge

Target screened: `L19:731 -> L24:3070`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown10_L19_731_to_L24_3070.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_unknown10_L19_731_to_L24_3070.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown10_L19_731_to_L24_3070.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown10_L19_731_to_L24_3070.json`

Intervention result:

| Docs | Mean L24 mass delta after L19 ablation | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---:|---:|---:|---:|---:|
| 25 | `-7.70` | `7.70` | `-0.275` | `-0.087` |

Interpretation:

- This is a weak alternate downstream for the confirmed `CODE-1` upstream. `L19:731` clearly matters for code/explanation style, but the strong downstream readout remains `L24:2110`.
- The effect on `L24:3070` is larger than the weakest screened-out edges, but still far below the confirmed `L19:731 -> L24:2110` effect (`-51.63` mass, `-5.32` max).
- Do not spend matched-control or token-locus time on this pair unless a future scan specifically suggests that `L24:3070` captures a distinct code/explanation subfeature.

### BROAD-1: polished factual/procedural AI register

Target: `L19:2687 -> L24:4086`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_qwen_polished_factual_L19_2687_to_L24_4086.csv`
- `docs/experiments/pangram_sae_circuit_target_ai_qwen_polished_factual_L19_2687_to_L24_4086.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_qwen_polished_factual_L19_2687_to_L24_4086.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/ai_qwen_polished_factual_token_loci_details.csv`
- `docs/experiments/pangram_sae_ai_qwen_polished_factual_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_ai_qwen_polished_factual.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_ai_qwen_polished_factual_summary.csv`

Intervention result:

| Docs | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean class-3 logit drop |
|---:|---:|---:|---:|---:|
| 20 | `1037.03` | `1327.31` | `-1294.69` | `2.9914` |

Source split:

| Source | Docs | L19 mass | L24 mass | L24 mass delta | Class-3 logit drop |
|---|---:|---:|---:|---:|---:|
| qwen3.6-35b | 16 | `999.92` | `1282.57` | `-1273.59` | `3.022` |
| gemini-3.5-flash | 2 | `907.76` | `1068.51` | `-1067.53` | `2.781` |
| glm-5.2 | 1 | `1406.53` | `1758.72` | `-1752.60` | `3.781` |
| human | 1 | `1273.92` | `1631.56` | `-1628.82` | `2.125` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Top-token class mix |
|---|---:|---:|---:|---:|---|
| `L19:2687` | 20 | 3,986 | `199.30` | `1024.73` | word `62.7%`, punctuation `23.5%`, sentence boundary `9.2%` |
| `L24:4086` | 20 | 3,925 | `196.25` | `1302.42` | word `76.2%`, punctuation `18.2%`, style/function word `4.5%` |

Matched-control result:

| Ablation family | Controls | Mean class-3 logit drop | Mean downstream mass delta | Mean absolute downstream mass delta |
|---|---:|---:|---:|---:|
| anchor `L19:2687 -> L24:4086` | 1 | `2.9914` | `-1294.69` | `1294.69` |
| different upstream, same downstream | 5 | `0.4128` | `0.87` | `4.53` |
| same upstream, different downstream | 5 | `2.9914` | `72.89` | `80.28` |

Interpretation:

- This is a real high-impact path by intervention, but it is not yet a clean local feature. It fires on hundreds of tokens per document.
- Peak contexts are polished factual/procedural passages: topic summaries, product/factual descriptions, explanatory list prose, and public-information style paragraphs.
- Top local peaks include comma punctuation and abstract nouns such as `framework`, `metrics`, `success`, `contracts`, `efforts`, `protagonist`, and `phenomenon`.
- Matched controls separate the two claims: the downstream activation collapse is specific to `L19:2687 -> L24:4086`, but the detector score movement is mostly carried by the upstream `L19:2687` direction and not by a unique final readout through `L24:4086`.
- Treat this as a broad AI-register circuit candidate. It needs decomposition into smaller local features before it can support a human-readable "slop indicator" claim.

### VERSE-1: rhymed/stanzaic line-ending accumulator

Target: `L19:2355 -> L24:61`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_verse_rhyme_L19_2355_to_L24_61.csv`
- `docs/experiments/pangram_sae_circuit_target_ai_verse_rhyme_L19_2355_to_L24_61.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_verse_rhyme_L19_2355_to_L24_61.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/ai_verse_rhyme_token_loci_details.csv`
- `docs/experiments/pangram_sae_ai_verse_rhyme_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_ai_verse_rhyme.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_ai_verse_rhyme_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_ai_verse_rhyme_linebreak_removed_targets.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_ai_verse_rhyme_linebreak_removed_intervention_L19_2355_to_L24_61.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_ai_verse_rhyme_linebreak_removed_token_loci_details.csv`
- `docs/experiments/pangram_sae_counterfactual_ai_verse_rhyme_linebreak_removed_token_loci.md`

Intervention result:

| Docs | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---:|---:|---:|---:|---:|
| 23 | `13.24` | `19.38` | `-10.10` | `-8.61` | `-0.0242` |

Matched-control result:

| Ablation family | Controls | Mean class-3 logit drop | Mean downstream mass delta | Mean absolute downstream mass delta |
|---|---:|---:|---:|---:|
| anchor `L19:2355 -> L24:61` | 1 | `-0.0242` | `-10.10` | `10.10` |
| different upstream, same downstream | 5 | `-0.0141` | `0.047` | `0.084` |
| same upstream, different downstream | 5 | `-0.0242` | `0.058` | `0.421` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Top-token class mix |
|---|---:|---:|---:|---:|---|
| `L19:2355` | 23 | 138 | `6.00` | `13.23` | word `65.9%`, sentence boundary `16.7%`, line break `16.7%` |
| `L24:61` | 23 | 80 | `3.48` | `19.38` | word `37.5%`, sentence boundary `28.7%`, line break `27.5%`, punctuation `6.2%` |

Line-break-removal counterfactual:

| Variant | Docs | L19 mass/doc | L24 mass/doc | L24 active tokens/doc | L19 ablation effect on L24 |
|---|---:|---:|---:|---:|---:|
| original | 23 | `13.24` | `19.38` | `3.48` | `-10.10` |
| line breaks removed | 23 | `2.90` | `3.38` | `2.22` | `-1.41` |
| edited - original | 23 | `-10.34` | `-16.00` | `-1.26` | `+8.69` |

Token loci after removing line breaks:

| Node | Original top-token mix | Edited top-token mix |
|---|---|---|
| `L19:2355` | word `65.9%`, sentence boundary `16.7%`, line break `16.7%` | sentence boundary `73.6%`, word `25.3%` |
| `L24:61` | word `37.5%`, sentence boundary `28.7%`, line break `27.5%` | sentence boundary `98.0%`, punctuation `2.0%` |

Interpretation:

- This is a clean format/style circuit candidate, not a detector-score path.
- The local peaks are line-ending periods, line breaks, and rhyme/end-line words such as `land`, `skies`, `away`, `spark`, `bride`, `flew`, `light`, and `sky`.
- Matched controls are near zero, so the downstream `L24:61` collapse is specific to `L19:2355` on these verse targets.
- The line-break-removal counterfactual shows that most of the path depends on stanzaic formatting. Rhyme/end-line words alone do not preserve the original L24 activation once line breaks are removed.
- A later prose rewrite could test whether poetic diction without lineation carries any residual part of the feature, but the main circuit is now best described as a lineated verse format circuit.

### VERSE-2: rhymed-verse end-word lexical path

Target: `L19:996 -> L24:1568`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown_L19_996_to_L24_1568.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_unknown_L19_996_to_L24_1568.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown_L19_996_to_L24_1568.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown_L19_996_to_L24_1568.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_unknown.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_unknown_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 17 | Gemini/GLM/GPT/human/Qwen | `85.78` | `107.64` | `-81.78` | `-10.00` | `-0.091` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.010` | `0.094` | `+0.004` | `-0.007` |
| same upstream, different downstream | `+0.024` | `0.082` | `+0.001` | `-0.091` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:996` | 17 | 304 | `17.88` | `85.78` | word `94.7%`, line break `2.3%`, sentence boundary `1.6%` |
| `L24:1568` | 17 | 299 | `17.59` | `107.64` | word `91.0%`, sentence boundary `5.7%`, line break `2.0%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:996` vs `L24:1568` | `94.1%` | `0.830` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:996` | `light` 3, period 3, `delight` 3, `bright` 3, `face` 2, `fair` 2, `race` 2, `door` 2, `stars` 2, `day` 2, `night` 2, `gold` 2, `life` 2 | rhymed verse and verse-like endings in poems about moonlight, pets, Hollywood, invisible strength, parks, tsunamis, butterflies, dragons, Costco, desert birds, Caesar/Cleopatra, wealth, and travel |
| `L24:1568` | period 8, `light` 3, `delight` 3, `bright` 3, `fair` 2, `face` 2, `race` 2, `stars` 2, `night` 2, `day` 2, `fame` 2, `pride` 2, `strife` 2 | same rhyming/end-word set, with slightly more sentence-boundary mass downstream |

Interpretation:

- This is a clean path-specific verse circuit. The anchor ablation drops L24 mass by `-81.78`, while both matched-control families are effectively zero.
- It is distinct from `VERSE-1`: `VERSE-1` is mostly lineated/stanzaic format, while `VERSE-2` is a sparse lexical path over rhyming/end-line words.
- It is not a slop-score path. The class-3 movement is mildly negative, so ablating it raises the AI score slightly.
- The high same-peak overlap and top-token Jaccard make this one of the more token-local paths in the current registry. Treat it as a rhymed-verse lexical/end-word circuit and use it as a control when filtering poetic-format features out of slop-style analysis.

### VERSE-3: broad poetic/lyrical imagery register path

Target: `L19:2130 -> L24:1478`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown5_L19_2130_to_L24_1478.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_unknown5_L19_2130_to_L24_1478.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown5_L19_2130_to_L24_1478.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown5_L19_2130_to_L24_1478.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_unknown5.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown5_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown5_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown5_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown5_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_unknown5_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 21 | Gemini/GLM/GPT/Qwen | `138.98` | `293.71` | `-87.28` | `-1.44` | `-0.099` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.086` | `0.131` | `-0.004` | `+0.000` |
| same upstream, different downstream | `-0.821` | `0.872` | `-0.036` | `-0.099` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:2130` | 21 | 1,798 | `85.62` | `138.98` | word `78.9%`, line break `12.6%`, punctuation `4.7%` |
| `L24:1478` | 21 | 1,954 | `93.05` | `293.71` | word `80.5%`, line break `11.7%`, punctuation `4.5%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:2130` vs `L24:1478` | `0.0%` | `0.444` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:2130` | `,\n` 87, comma 42, `.\n` 13, `And` 6, `the` 4, `A` 3, `To` 3, `;\n` 3, `light` 2, `guide` 2, `Yet` 2, `While` 2 | broad lyrical passages with line breaks and poetic imagery about storms, light, shadows, stars, mountains, anxiety, butterflies, eclipses, and tsunamis |
| `L24:1478` | `,\n` 71, comma 32, `.\n` 21, `And` 8, `;\n` 5, `through` 4, `drift` 3, `chase` 3, `As` 3, `Yet` 3, `shadows` 3 | same broad poetic/lyrical register, with more downstream mass on ordinary imagery words and line-break-adjacent punctuation |

Interpretation:

- This is a confirmed path-specific dependency: ablating `L19:2130` drops `L24:1478` mass by `-87.28`, while alternate upstreams into `L24:1478` average only `0.13` absolute mass movement.
- It is a broad verse/lyrical-register circuit, not a compact lexical feature. Both nodes fire across dozens of word and line-break positions per document, and the peak token does not usually coincide even though the active-token sets overlap.
- It is distinct from the earlier verse circuits. `VERSE-1` is primarily stanza/lineation format, `VERSE-2` is a sparse rhyme/end-word lexical path, and `VERSE-3` is a wider poetic imagery/register path.
- It is not an AI-slop score path. The class-3 movement is negative, so removing the upstream feature slightly raises the AI score.

### OUTLINE-1: outline/list/report line-break structure

Target: `L19:279 -> L24:508`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown3_L19_279_to_L24_508.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_unknown3_L19_279_to_L24_508.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown3_L19_279_to_L24_508.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown3_L19_279_to_L24_508.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_unknown3.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown3_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown3_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown3_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown3_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_unknown3_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 24 | Gemini/GLM/GPT/human | `93.42` | `179.17` | `-104.19` | `-5.32` | `+0.037` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `+0.098` | `0.171` | `-0.001` | `-0.007` |
| same upstream, different downstream | `-0.208` | `0.530` | `+0.021` | `+0.037` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:279` | 24 | 1,168 | `48.67` | `93.42` | line break `44.1%`, word `39.6%`, sentence boundary `11.1%`, punctuation `4.6%` |
| `L24:508` | 24 | 1,463 | `60.96` | `179.17` | word `43.2%`, line break `37.2%`, sentence boundary `13.2%`, punctuation `5.7%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:279` vs `L24:508` | `62.5%` | `0.780` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:279` | newline 84, `.\n\n` 64, blank line 38, `.\n` 23, period 13, `)\n` 8, `:\n\n` 8, `?\n` 5, bullet 2, `Key` 2 | outline/list/report structure: essay outlines, lesson plans, business summaries, job posts, resumes, activity materials, recipes, reports, classified listings, and instruction blocks |
| `L24:508` | newline 88, `.\n\n` 62, blank line 38, `.\n` 23, period 13, `:\n\n` 9, `)\n` 7, `?\n` 5, bullet 3, `Key` 2 | same structural-format family with stronger downstream mass and slightly more word spillover around headings/list items |

Interpretation:

- This is a clean structural-format circuit. The anchor ablation drops L24 mass by `-104.19`, while matched-control families are near zero.
- It is not a lexical slop marker. The active loci are line breaks, blank lines, heading breaks, list separators, and nearby outline/list words.
- The mild positive score movement means this structure can support the AI class somewhat, but many target rows are human. The stronger claim is format-level: outline/list/report structure.
- This should be a good candidate for a local counterfactual that joins list items into paragraphs or removes outline lineation while preserving content.

### OUTLINE-2: structured outline/report/code domain-token path

Target: `L19:2107 -> L24:11`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown9_L19_2107_to_L24_11.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_unknown9_L19_2107_to_L24_11.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown9_L19_2107_to_L24_11.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown9_L19_2107_to_L24_11.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_unknown9.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown9_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown9_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown9_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown9_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_unknown9_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 14 | human/GPT | `162.07` | `335.61` | `-193.77` | `-5.46` | `+0.016` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.096` | `0.458` | `+0.001` | `+0.012` |
| same upstream, different downstream | `-0.146` | `0.429` | `-0.005` | `+0.016` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:2107` | 14 | 1,453 | `103.79` | `162.07` | word `72.9%`, line break `10.1%`, punctuation `8.1%`, sentence boundary `6.5%` |
| `L24:11` | 14 | 1,181 | `84.36` | `335.61` | word `86.1%`, line break `7.2%`, punctuation `4.2%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:2107` vs `L24:11` | `57.1%` | `0.558` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:2107` | `apps` 4, `display` 4, newline 3, `sex` 3, `section` 3, comma 3, `position` 3, `bitmap` 3 | structured outline/report/code content: NASA/Gemini outline, character bullet bio, Dogecoin/social protocol outline, Q&A format, suitcase poem/list, 9/11 demographic report, jewelry post sections, crypto/remittance outline, spell ingredients, Terraform, OLED bitmap code, and lab report |
| `L24:11` | `bitmap` 5, `sex` 4, `display` 4, `apps` 3, `group` 3, `.id` 3, `spacecraft` 2, `VA` 2, `cryptocurrencies` 2 | same structured-domain token family, with stronger downstream mass and many exact or near-exact token overlaps |

Interpretation:

- This is a very clean path-specific dependency. The anchor ablation drops `L24:11` mass by `-193.77`, while both matched-control families average less than half a mass unit.
- It is not meaningfully a detector-score path. The mean class-3 movement is only `+0.016`.
- It is distinct from `OUTLINE-1`: `OUTLINE-1` is mostly line/paragraph boundary structure, while `OUTLINE-2` is word-heavy and follows structured-domain tokens inside outlines, reports, Q&A, code, and list-like artifacts.
- Treat it as a structured-content/list-domain path and as a strong non-score control. A useful next counterfactual would remove outline/list scaffolding while preserving the same domain terms, and separately replace the domain terms while preserving the scaffold.

### LETTER-1: correspondence signoff / formal letter closing format

Target: `L19:3424 -> L24:3666`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_letter_formal_L19_3424_to_L24_3666.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_letter_formal_L19_3424_to_L24_3666.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_letter_formal_L19_3424_to_L24_3666.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_letter_formal_L19_3424_to_L24_3666.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_letter_formal.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_letter_formal_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_letter_formal_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_letter_formal_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_letter_formal_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_letter_formal_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 23 | human/Gemini/GLM/GPT | `42.01` | `64.99` | `-45.75` | `-10.85` | `+0.067` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.001` | `0.226` | `-0.006` | `+0.022` |
| same upstream, different downstream | `+0.095` | `0.116` | `+0.014` | `+0.067` |

Token-locus result:

| Node | Docs | Mean active tokens/doc | Mean mass/doc | Mean max/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:3424` | 23 | `17.74` | `42.01` | `9.11` | word `51.2%`, line break `30.9%`, sentence boundary `9.3%`, punctuation `5.6%` |
| `L24:3666` | 23 | `19.35` | `64.99` | `14.44` | word `50.3%`, line break `31.9%`, sentence boundary `9.2%`, punctuation `5.6%` |

Token-position overlap:

| Pair | Mean peak-token distance | Median peak-token distance | Same peak token | Top-token set Jaccard |
|---|---:|---:|---:|---:|
| `L19:3424` vs `L24:3666` | `0.91` | `0.0` | `82.6%` | `0.711` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:3424` | paragraph-final period 75, period 24, comma paragraph break 11, `I` 10, `Thank` 9, `regards` 7, `you` 7, `Your` 5, `Best` 5, exclamation paragraph break 5, `Sincerely`, `Warm`, `All`, `soon` | letter/email closing paragraphs, signoffs, polite requests, and correspondence formatting |
| `L24:3666` | paragraph-final period 73, period 26, comma paragraph break 16, `I` 11, `Thank` 7, `you` 7, comma 6, `Best` 5, `look` 5, `Your` 4, `Warm`, `All`, `Please`, `request` | same signoff and formal-correspondence format with stronger downstream max activations |

Interpretation:

- This is a clean correspondence-format circuit. Ablating `L19:3424` drops `L24:3666` max activation by `-10.85`, and matched controls are essentially zero.
- It is compact and highly token-aligned. Both nodes fire on about 18-19 tokens per document, same-peak overlap is `82.6%`, and top-token Jaccard is `0.711`.
- The feature is not merely broad politeness. The strongest loci are paragraph breaks and signoff/closing tokens such as `Thank`, `Best`, `regards`, `Sincerely`, `Your`, `Warm`, and `All`.
- Treat it as a format circuit for emails/letters and polite correspondence closings. It should be separated from broad assistant-register or slop-style claims unless future counterfactuals show score effects from signoff format itself.

### BOUNDARY-3: narrative paragraph / scene-break accumulator

Target: `L19:1895 -> L24:3251`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_narrative_scene_L19_1895_to_L24_3251.csv`
- `docs/experiments/pangram_sae_circuit_target_ai_narrative_scene_L19_1895_to_L24_3251.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_narrative_scene_L19_1895_to_L24_3251.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/ai_narrative_scene_token_loci_details.csv`
- `docs/experiments/pangram_sae_ai_narrative_scene_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_ai_narrative_scene_break.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_ai_narrative_scene_break_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_ai_narrative_scene_paragraph_joined_targets.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_ai_narrative_scene_paragraph_joined_intervention_L19_1895_to_L24_3251.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_ai_narrative_scene_paragraph_joined_token_loci_details.csv`
- `docs/experiments/pangram_sae_counterfactual_ai_narrative_scene_paragraph_joined_token_loci.md`

Intervention result:

| Docs | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---:|---:|---:|---:|---:|
| 26 | `99.26` | `176.75` | `-121.83` | `-11.57` | `0.7230` |

Source split:

| Source | Docs | L19 mass | L24 mass | L24 mass delta | Class-3 logit drop |
|---|---:|---:|---:|---:|---:|
| gpt-5.5 | 20 | `107.49` | `191.02` | `-131.80` | `0.775` |
| glm-5.2 | 6 | `71.83` | `129.18` | `-88.59` | `0.549` |

Matched-control result:

| Ablation family | Controls | Mean class-3 logit drop | Mean downstream mass delta | Mean absolute downstream mass delta |
|---|---:|---:|---:|---:|
| anchor `L19:1895 -> L24:3251` | 1 | `0.7230` | `-121.83` | `121.83` |
| different upstream, same downstream | 5 | `-0.0391` | `2.84` | `3.32` |
| same upstream, different downstream | 5 | `0.7230` | `-1.03` | `1.22` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Top-token class mix |
|---|---:|---:|---:|---:|---|
| `L19:1895` | 26 | 954 | `36.69` | `99.26` | line break `65.3%`, word `18.3%`, sentence boundary `9.3%` |
| `L24:3251` | 26 | 984 | `37.85` | `176.75` | line break `72.5%`, word `15.3%`, sentence boundary `7.5%` |

Paragraph-join counterfactual:

| Variant | Docs | L19 mass/doc | L24 mass/doc | L24 active tokens/doc | L19 ablation effect on L24 | Class-3 logit drop |
|---|---:|---:|---:|---:|---:|---:|
| original | 26 | `99.43` | `176.72` | `37.69` | `-121.73` | `0.7225` |
| paragraph joined | 26 | `141.09` | `236.90` | `56.62` | `-177.52` | `1.0047` |
| edited - original | 26 | `+41.67` | `+60.18` | `+18.92` | `-55.79` | `+0.2822` |

Token loci after paragraph joining:

| Node | Original top-token mix | Edited top-token mix |
|---|---|---|
| `L19:1895` | line break `65.3%`, word `18.3%`, sentence boundary `9.3%` | sentence boundary `72.5%`, punctuation `16.9%`, word `8.5%` |
| `L24:3251` | line break `72.5%`, word `15.1%`, sentence boundary `7.6%` | sentence boundary `74.2%`, punctuation `17.3%`, word `7.9%` |

Interpretation:

- This is not a generic scene-word latent. Original top activations are paragraph-final punctuation plus blank-line tokens such as `.\\n\\n`, `.”\\n\\n`, and `?”\\n\\n`.
- The paragraph-join counterfactual shows it is not a strict blank-line detector. Removing blank lines shifts the path onto ordinary sentence-boundary punctuation and increases total mass, probably because many more sentence-boundary tokens remain eligible in the compressed format.
- It differs from BOUNDARY-1: BOUNDARY-1 is a narrower canonical sentence/period scene-beat path, while BOUNDARY-3 is a broader narrative boundary-density / scene-rhythm path.
- Matched controls support path specificity for downstream activation. As with BROAD-1, the class-score movement follows the upstream node across downstream controls, so the detector-score evidence should be described as upstream-driven rather than uniquely mediated through `L24:3251`.
- Next test should control sentence-boundary density directly, e.g. replacing some periods with semicolons or joining short paragraphs without preserving every sentence boundary.

### BOUNDARY-4: rhetorical/monologue prose into sentence-boundary terminal

Target: `L19:1099 -> L24:3970`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown8_L19_1099_to_L24_3970.csv`
- `docs/experiments/pangram_sae_circuit_target_mixed_unknown8_L19_1099_to_L24_3970.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown8_L19_1099_to_L24_3970.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown8_L19_1099_to_L24_3970.json`
- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_mixed_unknown8.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown8_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown8_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown8_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/mixed_unknown8_token_loci_summary.csv`
- `docs/experiments/pangram_sae_mixed_unknown8_token_loci.md`

Intervention result:

| Docs | Source mix | Mean L19 mass | Mean L24 mass | Mean L24 mass delta after L19 ablation | Mean L24 max delta | Mean class-3 logit drop |
|---:|---|---:|---:|---:|---:|---:|
| 15 | mostly Qwen | `488.08` | `265.71` | `-37.56` | `-0.48` | `+0.074` |

Control result:

| Control type | Mean L24 mass delta | Mean abs L24 mass delta | Mean L24 max delta | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | `-0.266` | `0.484` | `-0.006` | `-0.005` |
| same upstream, different downstream | `-3.239` | `4.316` | `+0.029` | `+0.074` |

Token-locus result:

| Node | Docs | Active tokens | Mean active tokens/doc | Mean mass/doc | Main active token classes |
|---|---:|---:|---:|---:|---|
| `L19:1099` | 15 | 5,778 | `385.20` | `488.08` | word `75.5%`, punctuation `8.2%`, sentence boundary `6.9%`, style/function word `4.8%` |
| `L24:3970` | 15 | 1,311 | `87.40` | `265.71` | sentence boundary `34.1%`, punctuation `26.2%`, word `25.5%`, style/function word `6.6%` |

Token-position overlap:

| Pair | Same peak token | Top-token set Jaccard |
|---|---:|---:|
| `L19:1099` vs `L24:3970` | `0.0%` | `0.003` |

Top-token context inspection:

| Node | Top tokens | Context pattern |
|---|---|---|
| `L19:1099` | comma 13, `They` 11, `I` 10, `are` 9, `You` 8, `We` 8, `am` 8, `to` 7, `is` 7, `It` 6 | broad rhetorical/monologue register, especially Qwen outputs: war meditation, motivational assassin speech, Basquiat first-person voice, environmental/community speech, existential self-reflection, archaeology grandeur, interview dialogue, diary/letter, sandwich satire, and graduation speech |
| `L24:3970` | period 153, question mark 9, comma 5, exclamation mark 3, `.\n\n` 3 | sentence-boundary terminal for the same rhetorical prose; downstream mass concentrates on periods and paragraph/sentence endings rather than the upstream pronoun tokens |

Interpretation:

- This is a real but smaller path-specific dependency. The anchor ablation drops `L24:3970` mass by `-37.56`, while alternate upstreams into `L24:3970` average only `0.48` absolute mass movement.
- It is score-positive under the detector, but the score effect is modest compared with the major Qwen/expository paths.
- The upstream and downstream are not same-token features. `L19:1099` is a broad rhetorical/pronoun-heavy prose state, while `L24:3970` is a punctuation-heavy sentence-boundary terminal with almost zero token-set overlap.
- This should be treated as a Qwen-heavy rhetorical-register-to-boundary path. A useful next counterfactual would reduce repeated first-person/second-person rhetorical framing while keeping sentence-boundary density fixed, and separately vary punctuation density while preserving the same monologue text.

## First Intervention Results

I added `scripts/measure_pangram_sae_chain_intervention.py` and ran the first pairwise intervention tests on two exported target bundles. The script ablates an upstream SAE latent contribution in the model residual stream, then measures both the detector class-3 logit change and the downstream SAE latent activation change on the same documents.

Layer-20 pairwise tests were initially deferred because the current layer-20 explorer sparse export points to `artifacts/pangram_llama_sae/layer20_full_neuralwatt_cap640_k64_e50`, while the `.pt` checkpoint present on disk is under `artifacts/pangram_llama_sae/layer20_full_glm52_cap640_k64_e50`. That deferral was unnecessary: these SAE checkpoints are interchangeable for this workflow, so the layer-20 pairwise tests should be run using the available GLM52 layer-20 checkpoint.

### Literary scene chain

Target: `L16:1677 -> L19:3009 -> L20:2077 -> L24:3642`

Available tested hops:

| Hop | Docs | Mean class-3 logit drop | Mean downstream mass delta | Mean downstream max delta |
|---|---:|---:|---:|---:|
| `L16:1677 -> L19:3009` | 12 | `0.1274` | `-118.63` | `-3.34` |
| `L19:3009 -> L24:3642` | 12 | `0.1756` | `-249.58` | `-6.76` |

Interpretation:

- This is the best current circuit seed for an AI-style path.
- The shared prompts are all GPT-5.5 literary scene openings with polished sensory narration and named characters.
- Ablating the earlier node reduces the later node's activation mass on every tested document, which is stronger evidence than prompt overlap alone.
- The logit effect is moderate but consistently in the expected direction for the target detector class.

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_literary_scene_chain.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_literary_L16_1677_to_L19_3009.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_literary_L19_3009_to_L24_3642.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/literary_scene_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/literary_scene_token_loci_summary.csv`
- `docs/experiments/pangram_sae_literary_scene_token_loci.md`

Token-locus update:

| Node | Docs | Active tokens | Mean mass/doc | Top-token class mix |
|---|---:|---:|---:|---|
| `L19:3009` | 12 | 952 | `234.79` | sentence boundary `89.2%`, punctuation `10.0%`, word `0.8%` |
| `L24:3642` | 12 | 1,122 | `394.42` | sentence boundary `88.3%`, punctuation `10.8%`, word `0.8%` |

This reframes the candidate as a narrative sentence-boundary / scene-beat accumulator. The semantic coherence comes from where the periods occur: polished narrative scene beats, dialogue closures, and short atmospheric turns. The local latent itself is almost entirely boundary-local.

Boundary counterfactual update:

I built a paired edit set that replaces sentence-final `. ` boundaries with `; ` in the same 12 literary target documents. This preserves most content and local word order while removing many canonical period boundaries.

| Variant | Docs | L19 mass/doc | L24 mass/doc | L24 active tokens/doc | L19 ablation effect on L24 |
|---|---:|---:|---:|---:|---:|
| original | 12 | `234.85` | `394.48` | `93.67` | `-249.83` |
| joined-boundary edit | 12 | `167.84` | `265.11` | `81.25` | `-174.45` |
| edited - original | 12 | `-67.01` | `-129.37` | `-12.42` | `+75.38` |

Token loci after editing shift from period tokens to semicolon punctuation:

| Node | Original top-token mix | Edited top-token mix |
|---|---|---|
| `L19:3009` | sentence boundary `89.2%`, punctuation `10.0%` | punctuation `98.8%`, sentence boundary `0.8%` |
| `L24:3642` | sentence boundary `87.9%`, punctuation `11.2%` | punctuation `97.5%`, sentence boundary `1.7%` |

Interpretation:

- The path is not period-only; it is a boundary-punctuation / scene-beat accumulator.
- Canonical period boundaries carry more mass than semicolon-joined boundaries in these documents, but the edited text still activates on semicolons.
- This is strong evidence that BOUNDARY-1 is a real local boundary circuit candidate, distinct from the lexical slop features.

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_literary_sentence_boundary_joined_targets.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_literary_sentence_boundary_joined_intervention_L19_3009_to_L24_3642.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_literary_sentence_boundary_joined_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_literary_sentence_boundary_joined_token_loci_summary.csv`
- `docs/experiments/pangram_sae_counterfactual_literary_sentence_boundary_joined_token_loci.md`

### Human boilerplate/control chain

Target: `L16:3660 -> L19:3990 -> L20:1210 -> L24:1120`

Available tested hops:

| Hop | Docs | Mean class-3 logit drop | Mean downstream mass delta | Mean downstream max delta |
|---|---:|---:|---:|---:|
| `L16:3660 -> L19:3990` | 11 | `1.4631` | `-1330.84` | `-6.31` |
| `L19:3990 -> L24:1120` | 11 | `0.8171` | `-2177.07` | `-10.23` |

Interpretation:

- This path is not an AI-slop path. It is mostly human-labeled simple story, generic essay, poem, or boilerplate continuation text.
- It is still circuit-relevant: the activation coupling is much larger than in the literary AI-style path, and it provides a strong control for detector evidence that class-3 is not simply "AI text".
- Because the target class-3 logit drops when these nodes are ablated, the path may be a broad detector-supporting direction for generated-looking form even when the corpus label is human.

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/targets_human_boilerplate_chain.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_L16_3660_to_L19_3990.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_L19_3990_to_L24_1120.csv`

### Immediate Next Tests

1. Run the layer-20 hops for the two full four-node chains using the available interchangeable layer-20 checkpoint.
2. Add matched controls: same upstream layer, same downstream layer, random same-rank latents, and high-mass non-overlap latents.
3. For the literary chain, run held-out narrative prompts not used in candidate selection.
4. For the human/control chain, split by source label and check whether the large effect is driven by human examples alone or by generic simple-continuation form.

## First Matched Controls

I added a first set of matched controls using the same intervention script, the same target document bundles, and the same available layer pairs. These controls do not yet replace random same-rank controls over a larger panel, but they answer the immediate failure mode: whether upstream ablation merely perturbs the model enough to suppress any downstream SAE latent.

### Literary target bundle controls

Target bundle: `targets_literary_scene_chain.csv`

| Test | Docs | Mean class-3 logit drop | Mean downstream mass delta | Mean downstream max delta |
|---|---:|---:|---:|---:|
| chain `L16:1677 -> L19:3009` | 12 | `0.1274` | `-118.63` | `-3.34` |
| control, same upstream `L16:1677 -> L19:3990` | 12 | `0.1284` | `4.21` | `0.02` |
| control, human upstream `L16:3660 -> L19:3009` | 12 | `0.0760` | `0.81` | `0.02` |
| chain `L19:3009 -> L24:3642` | 12 | `0.1756` | `-249.58` | `-6.76` |
| control, same upstream `L19:3009 -> L24:1120` | 12 | `0.1792` | `6.50` | `0.05` |
| control, human upstream `L19:3990 -> L24:3642` | 12 | `0.0470` | `-0.20` | `0.00` |

Interpretation:

- The class-3 logit can move under both chain and non-chain ablations, so the classifier-score effect alone is not enough to claim a circuit.
- The downstream activation effect is much more specific. The chain hops sharply suppress their downstream latents, while same-upstream and reciprocal non-chain controls leave those downstream latents nearly unchanged.
- This makes Candidate B a credible activation-coupled style path for polished AI literary scene-setting, though it still needs held-out prompts and broader random controls.

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/control_literary_same_upstream_L16_1677_to_L19_3990.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_literary_human_upstream_L16_3660_to_L19_3009.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_literary_same_upstream_L19_3009_to_L24_1120.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_literary_human_upstream_L19_3990_to_L24_3642.csv`

### Human/control target bundle controls

Target bundle: `targets_human_boilerplate_chain.csv`

| Test | Docs | Mean class-3 logit drop | Mean downstream mass delta | Mean downstream max delta |
|---|---:|---:|---:|---:|
| chain `L16:3660 -> L19:3990` | 11 | `1.4631` | `-1330.84` | `-6.31` |
| control, same upstream `L16:3660 -> L19:3009` | 11 | `1.4691` | `1.04` | `0.17` |
| chain `L19:3990 -> L24:1120` | 11 | `0.8171` | `-2177.07` | `-10.23` |
| control, same upstream `L19:3990 -> L24:3642` | 11 | `0.8189` | `0.80` | `0.11` |

Interpretation:

- The human-heavy chain has very large downstream coupling, but the logit effect is also reproduced when measuring non-chain downstream latents. That means the detector score movement is a broad effect of removing a strong upstream feature, not evidence that every downstream latent participates in the same circuit.
- The downstream activation controls are decisive: the chain downstream latents collapse, while the literary downstream controls barely move.
- Candidate A remains useful as a strong human/form-control circuit, not as an AI-slop circuit.

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/control_human_same_upstream_L16_3660_to_L19_3009.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_human_same_upstream_L19_3990_to_L24_3642.csv`

### Updated Conclusion

The strongest evidence so far is not that these latents causally move the classifier score; many upstream ablations do that. The stronger claim is narrower and better supported: selected upstream latents causally support specific downstream latents on the same prompt bundle, and that support is absent for matched non-chain downstream latents. That gives us two intervention-backed circuit seeds:

1. `L16:1677 -> L19:3009 -> L24:3642`: polished AI literary scene-setting.
2. `L16:3660 -> L19:3990 -> L24:1120`: human-heavy simple-continuation/form-control path.

The next step is to run a larger automatic control panel: same-layer random controls, same-rank controls, high-activation non-overlap controls, and held-out prompts for Candidate B. Layer-20 hops should now be resumed with the available layer-20 checkpoint.

## Reproducible Control Panel Queue

I added `scripts/select_pangram_sae_circuit_controls.py` to make the next control round reproducible. It reads the existing latent explorer browser indexes, then for each available tested hop picks:

- same-upstream, different-downstream controls
- different-upstream, same-downstream controls

The selector prefers controls with similar causal rank but low overlap with the anchor latent's high-impact prompts. The first control run used compressed chains matching the `16 -> 19 -> 24` intervention hops already tested:

- `L16:1677 -> L19:3009 -> L24:3642`
- `L16:3660 -> L19:3990 -> L24:1120`

Output:

- `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_available_hops.csv`

Coverage:

- 40 selected controls total.
- For each target chain and each available hop, 5 same-upstream/different-downstream controls and 5 different-upstream/same-downstream controls.
- This panel should be the next batch of exact interventions before making stronger claims about circuit specificity.

I then ran the full 40-row panel with `scripts/run_pangram_sae_intervention_panel.py`, which loads the model once and caches baselines per downstream latent. It produced 460 document-level rows.

Outputs:

- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_available_hops_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_available_hops_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_panel_available_hops_summary.json`

Panel summary:

| Target | Chain hop | Control type | Controls | Mean downstream mass delta | Range | Mean class-3 logit drop |
|---|---|---|---:|---:|---:|---:|
| human boilerplate | `L16:3660 -> L19:3990` | different upstream, same downstream | 5 | `-9.63` | `-32.04` to `0.50` | `0.011` |
| human boilerplate | `L16:3660 -> L19:3990` | same upstream, different downstream | 5 | `119.08` | `-38.19` to `483.85` | `1.469` |
| human boilerplate | `L19:3990 -> L24:1120` | different upstream, same downstream | 5 | `53.58` | `-9.19` to `276.29` | `0.072` |
| human boilerplate | `L19:3990 -> L24:1120` | same upstream, different downstream | 5 | `1.80` | `0.33` to `2.64` | `0.819` |
| literary scene | `L16:1677 -> L19:3009` | different upstream, same downstream | 5 | `-0.81` | `-4.05` to `0.45` | `0.002` |
| literary scene | `L16:1677 -> L19:3009` | same upstream, different downstream | 5 | `0.83` | `-0.06` to `2.44` | `0.128` |
| literary scene | `L19:3009 -> L24:3642` | different upstream, same downstream | 5 | `0.01` | `-0.36` to `0.50` | `-0.004` |
| literary scene | `L19:3009 -> L24:3642` | same upstream, different downstream | 5 | `0.70` | `-0.78` to `3.03` | `0.179` |

Interpretation:

- Candidate B now has much stronger specificity evidence. Its chain-hop downstream mass drops are `-118.63` and `-249.58`, while the matched control means are within about `-0.81` to `0.83` for the layer-16-to-19 hop and `0.01` to `0.70` for the layer-19-to-24 hop.
- Candidate A remains a powerful non-AI control path. Its chain-hop downstream mass drops are `-1330.84` and `-2177.07`; selected controls can move other features, but they do not approach the chain-specific downstream collapse.
- Same-upstream controls often reproduce the class-logit drop because the upstream latent itself is causally important to the detector. They generally do not reproduce the downstream-latent collapse. This is the key distinction the next circuit work should preserve.

## Peak-Token Evidence

I added `scripts/export_pangram_sae_peak_windows.py` to join target rows against tokenized documents and export the local text around each peak token without rerunning the model.

Outputs:

- `docs/experiments/pangram_sae_peak_windows_literary_scene.md`
- `docs/experiments/pangram_sae_peak_windows_human_boilerplate.md`
- `artifacts/pangram_llama_sae/circuit_discovery/peak_windows_literary_scene.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/peak_windows_human_boilerplate.csv`

This changes the interpretation slightly:

- Candidate B should not be described as simply "literary imagery" at the token level. In the 12 exported target prompts, every node in the chain peaks on a period token: `L16:1677`, `L19:3009`, `L20:2077`, and `L24:3642` are all `12/12` period peaks. The surrounding documents are polished GPT-5.5 literary scene-setting, often at a transition into the next beat of the scene, but the SAE path appears to be a sentence-boundary or narrative-beat accumulator inside that style.
- Candidate A peaks more often on high-frequency function words and local connective structure in simple/repetitive human prose: examples include `and`, `his`, `it`, `of`, `for`, plus occasional sentence-final periods. It looks less like one lexical feature and more like a broad simple-continuation/form direction.

Updated labels:

1. `L16:1677 -> L19:3009 -> L24:3642`: AI narrative sentence-boundary / scene-beat accumulator.
2. `L16:3660 -> L19:3990 -> L24:1120`: human/simple-continuation connective-form accumulator.

### Candidate E Peak Check

I also exported the high-priority AI advice/product scaffold seed:

- Target export: `docs/experiments/pangram_sae_circuit_target_ai_advice_scaffold.md`
- Peak windows: `docs/experiments/pangram_sae_peak_windows_ai_advice_scaffold.md`
- CSVs: `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_advice_scaffold_chain.csv`, `artifacts/pangram_llama_sae/circuit_discovery/peak_windows_ai_advice_scaffold.csv`

Only 4 prompts are shared across all four nodes. The peak-window join covers 14 of 16 rows because two older Fireworks GLM doc IDs from the layer-16 export are not present in the current layer-19 token parquet. Among the joined rows, the pattern is again period-centered:

- `L19:2943`: `4/4` peaks are periods.
- `L20:3795`: `4/4` peaks are periods.
- `L24:2919`: `4/4` peaks are periods.
- `L16:2943`: `2/2` joined rows peak on sentence-final punctuation.

The surrounding contexts are generic capstone sentences in advice, wellness, technology, and real-estate prose, often near the end of the document: examples include durable phrases like "supports long-term vitality", "digital age", and "professional integrity and market expertise throughout every phase of the sales cycle." Candidate E is therefore better described as an AI final-summary/capstone sentence-boundary path, not merely a list/advice scaffold.

## Peak-Token Screen For Next Candidates

I added `scripts/summarize_pangram_sae_latent_peak_tokens.py` to summarize peak-token distributions for top browser latents. It joins the browser peak token indexes against the tokenized document parquet and computes punctuation, period, and word fractions.

Outputs:

- `artifacts/pangram_llama_sae/circuit_discovery/latent_peak_tokens_layer19.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/latent_peak_tokens_layer24.csv`

This screen confirms that several of the highest causal-impact latents are punctuation or sentence-boundary features:

- `L19:2943`: `50/50` joined peaks are periods; rank 1; strongly AI/Qwen-heavy.
- `L24:2919`: `39/39` joined peaks are periods; rank 3; strongly AI/Qwen-heavy.
- `L24:741`: `38/38` joined peaks are periods; rank 1; strongly human-heavy.
- `L19:1895`, `L19:2355`, `L24:3251`, and `L24:61` are also mostly period/punctuation peaks.

High-impact mostly-word candidates worth checking next:

- `L24:4086`: word fraction `0.921`, max abs impact `1.727`, Qwen-heavy; top peak tokens are diverse content words and comma. This is a better candidate for the earlier "human lexicon minus AI lexicon" hypothesis than the period chains.
- `L19:3990 -> L24:1120`: human/simple connective path; peaks on `and`, `to`, `his`, `of`, `it`, `you`, etc. This remains useful as a human/form control.
- `L19:1054`: word fraction `0.980`; peaks on call-to-action/web words such as `today`, `now`, `website`, `Visit`, `us`, `link`, and `tickets`.
- `L19:913`: word fraction `1.000`; peaks on contractions/negation fragments such as `didn`, `wasn`, `hadn`, `doesn`, `don`.
- `L24:776`: word fraction `0.977`; human-heavy; peaks on pronouns/connectives like `this`, `that`, `you`, `also`, `down`.

Next candidate-selection rule:

- Keep period/boundary chains as their own class of detector circuits.
- For slop-style semantics, prioritize candidates with high word fraction, low punctuation fraction, and source/model enrichment. This should avoid repeatedly selecting end-of-sentence accumulator paths when the research question is about stylistic content.

## Lexical/Style Candidate Queue

I added `scripts/select_pangram_sae_lexical_chains.py` to rank chains after filtering out punctuation-heavy nodes. The selector uses:

- peak-token summaries for layers 16, 19, 20, and 24;
- minimum word fraction `0.70`;
- maximum punctuation fraction `0.25`;
- maximum period fraction `0.25`;
- existing cross-layer edge scores and prompt overlap.

Outputs:

- General lexical/style queue: `docs/experiments/pangram_sae_lexical_style_candidate_chains.md`
- AI-enriched lexical/style queue: `docs/experiments/pangram_sae_lexical_style_ai_enriched_candidate_chains.md`
- CSVs: `artifacts/pangram_llama_sae/circuit_discovery/lexical_style_candidate_chains.csv`, `artifacts/pangram_llama_sae/circuit_discovery/lexical_style_ai_enriched_candidate_chains.csv`

The general lexical queue is dominated by human/simple-connective chains, especially variants of `L19:3990 -> L20:1210 -> L24:1120`. The AI-enriched queue surfaces three more useful lexical families:

1. Modal/simple-explanatory wording: `L16:3800 -> L19:262 -> L20:1382 -> L24:1664`, with peaks such as `can`, `feel`, `still`, `might`, and `to`. Full-chain shared prompts are sparse (`3`), so this is semantically plausible but not yet a strong intervention target.
2. Negation/contraction fragments: `L16:2177 -> L19:913 -> L20:354 -> L24:1310`, with peaks such as `didn`, `wasn`, `don`, and related subword fragments. It is source-cleaner and AI-heavy, but full-chain shared prompts are only `2` and some peaks are subword pieces inside unrelated words.
3. Discourse markers / contrastive transition prose: `L16:3450 -> L19:3450 -> L20:1780 -> L24:3469`, with peaks around `However`, `Additionally`, `Furthermore`, and `such`. It has zero prompts shared across all four nodes but strong adjacent-hop overlap, especially `L19:3450 -> L20:1780` and `L20:1780 -> L24:3469`.

### First Non-Period Lexical Intervention

As a fast first pass, I tested the direct layer-19-to-layer-24 hop for the discourse-marker path:

- Target: `L19:3450 -> L24:3469`
- Shared prompts: `18`
- Target export: `docs/experiments/pangram_sae_circuit_target_ai_discourse_marker_L19_3450_to_L24_3469.md`
- Peak windows: `docs/experiments/pangram_sae_peak_windows_ai_discourse_marker_L19_3450_to_L24_3469.md`
- Intervention artifact: `artifacts/pangram_llama_sae/circuit_discovery/intervention_discourse_marker_L19_3450_to_L24_3469.json`

Result:

| Test | Docs | Mean class-3 logit drop | Mean downstream mass delta | Mean downstream max delta |
|---|---:|---:|---:|---:|
| chain `L19:3450 -> L24:3469` | 19 | `0.0492` | `-31.16` | `-11.91` |

Matched controls:

- Control panel: `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_discourse_marker.csv`
- Details: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_discourse_marker_details.csv`
- Summary: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_discourse_marker_summary.csv`

| Control type | Controls | Mean downstream mass delta | Range | Mean class-3 logit drop |
|---|---:|---:|---:|---:|
| different upstream, same downstream | 5 | `0.004` | `-0.032` to `0.020` | `-0.012` |
| same upstream, different downstream | 5 | `1.036` | `-11.170` to `9.466` | `0.049` |

Interpretation:

- This is the best non-period lexical circuit seed so far.
- The downstream effect is specific: unrelated upstream latents do not move `L24:3469`, while `L19:3450` sharply suppresses it.
- The class-logit effect is small, so this path is likely a localized style/transition feature rather than a dominant detector-score path by itself.
- The semantic label is "discourse-marker / contrastive-transition prose", not generic AI advice.

### Additional Direct Lexical Interventions

I then checked the AI-enriched lexical queue for direct `L19 -> L24` prompt overlap. The strongest direct candidates were:

- `L19:913 -> L24:1310`: negation/contraction fragments, 15 shared prompts.
- `L19:262 -> L24:1664`: modal/simple-explanatory function words, 13 shared prompts.
- `L19:3450 -> L24:3469`: discourse markers, 18 shared prompts, already tested above.

Exports:

- `docs/experiments/pangram_sae_circuit_target_ai_negation_fragment_L19_913_to_L24_1310.md`
- `docs/experiments/pangram_sae_peak_windows_ai_negation_fragment_L19_913_to_L24_1310.md`
- `docs/experiments/pangram_sae_circuit_target_ai_modal_function_L19_262_to_L24_1664.md`
- `docs/experiments/pangram_sae_peak_windows_ai_modal_function_L19_262_to_L24_1664.md`

Exact intervention results:

| Target | Docs | Mean class-3 logit drop | Mean downstream mass delta | Mean downstream max delta |
|---|---:|---:|---:|---:|
| negation fragments `L19:913 -> L24:1310` | 20 | `0.1984` | `-35.18` | `-9.92` |
| modal/function wording `L19:262 -> L24:1664` | 14 | `0.6024` | `-2965.80` | `-13.10` |
| discourse markers `L19:3450 -> L24:3469` | 19 | `0.0492` | `-31.16` | `-11.91` |

Matched controls for negation and modal/function:

- Control panel: `artifacts/pangram_llama_sae/circuit_discovery/selected_control_panel_lexical_l19_l24.csv`
- Details: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_lexical_l19_l24_details.csv`
- Summary: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_lexical_l19_l24_summary.csv`

| Target | Control type | Controls | Mean downstream mass delta | Range | Mean class-3 logit drop |
|---|---|---:|---:|---:|---:|
| negation fragments | different upstream, same downstream | 5 | `-0.083` | `-0.435` to `0.018` | `0.001` |
| negation fragments | same upstream, different downstream | 5 | `0.992` | `-7.206` to `13.325` | `0.198` |
| modal/function wording | different upstream, same downstream | 5 | `-17.983` | `-87.840` to `-0.046` | `-0.067` |
| modal/function wording | same upstream, different downstream | 5 | `-45.829` | `-240.825` to `6.635` | `0.602` |

Interpretation:

- `L19:913 -> L24:1310` is a clean lexical circuit seed. Peak windows repeatedly show contraction/negation fragments such as `didn`, `doesn`, `wasn`, and `don`, often in generated narrative prose. Controls leave the downstream latent essentially unchanged, so the coupling is specific.
- `L19:262 -> L24:1664` is much stronger in raw downstream mass effect, but it looks broader and more grammar-like: `to`, `can`, `feel`, `still`, `are`, `has`, and similar simple explanatory-function tokens. Controls do move other features more than in the negation/discourse cases, so this is a powerful but less isolated path.
- Together with the discourse-marker path, we now have three non-period lexical/direct circuit seeds. They are not yet full four-layer circuits because the layer-20 hops have not been tested, but the available layer-20 checkpoint can be used for those tests.

### Held-Out Pattern Generalization

I added `scripts/select_pangram_sae_pattern_targets.py` to select held-out documents by text regex rather than latent-top-example overlap. It excludes the documents used in the direct target exports and writes the same target CSV shape expected by the intervention script.

Held-out bundles:

- Discourse markers: `artifacts/pangram_llama_sae/circuit_discovery/heldout_discourse_marker_targets.csv`
- Negation fragments: `artifacts/pangram_llama_sae/circuit_discovery/heldout_negation_fragment_targets.csv`
- Audits: `docs/experiments/pangram_sae_heldout_discourse_marker_targets.md`, `docs/experiments/pangram_sae_heldout_negation_fragment_targets.md`

Selection patterns:

- Discourse markers: `however|furthermore|additionally|moreover|for example|in addition|such as`
- Negation fragments: `didn't`, `doesn't`, `wasn't`, `don't`, `hadn't`, `couldn't`, `wouldn't`, including curly apostrophe variants.

Held-out intervention results:

| Target | Held-out docs | Mean class-3 logit drop | Mean downstream mass delta | Mean downstream max delta |
|---|---:|---:|---:|---:|
| discourse markers `L19:3450 -> L24:3469` | 24 | `0.0465` | `-23.80` | `-5.93` |
| negation fragments `L19:913 -> L24:1310` | 24 | `0.1136` | `-28.67` | `-6.63` |

Interpretation:

- Both lexical paths generalize beyond the latent-selected target rows.
- The held-out effects are smaller than the top-example effects but still strongly directional.
- This is now stronger evidence that the discourse-marker and negation-fragment paths are genuine style/lexical circuits rather than only artifacts of the explorer's top-document selection.

### Counterfactual Surface-Form Edits

I added `scripts/build_pangram_sae_counterfactual_targets.py` and patched the intervention loader so target CSVs with explicit `text` columns can be measured directly. This lets us test original/edited pairs without adding edited documents to the parquet.

Counterfactual bundles:

- Discourse markers: `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_discourse_marker_targets.csv`
- Negation fragments: `artifacts/pangram_llama_sae/circuit_discovery/counterfactual_negation_fragment_targets.csv`
- Audits: `docs/experiments/pangram_sae_counterfactual_discourse_marker_targets.md`, `docs/experiments/pangram_sae_counterfactual_negation_fragment_targets.md`

Edits:

- Discourse: replace markers such as `However`, `Furthermore`, `Additionally`, `Moreover`, `In addition`, and `such as` with plainer alternatives like `Still`, `Also`, or `including`.
- Negation: expand contractions such as `didn't`, `doesn't`, `wasn't`, `don't`, `hadn't`, `couldn't`, and `wouldn't` into `did not`, `does not`, etc.

Original-vs-edited summary:

| Target | Variant | Docs | Mean downstream baseline mass | Mean downstream baseline max | Mean ablation mass delta | Mean class-3 logit drop |
|---|---|---:|---:|---:|---:|---:|
| discourse markers | original | 16 | `27.88` | `6.87` | `-22.65` | `0.043` |
| discourse markers | edited | 16 | `14.45` | `5.55` | `-11.50` | `0.037` |
| negation fragments | original | 16 | `33.83` | `7.92` | `-28.75` | `0.116` |
| negation fragments | edited | 16 | `17.02` | `6.49` | `-15.19` | `0.093` |

Paired edited-minus-original effects:

| Target | Baseline mass change | Baseline max change | Ablation mass-delta change |
|---|---:|---:|---:|
| discourse markers | `-13.43` | `-1.32` | `+11.14` |
| negation fragments | `-16.80` | `-1.43` | `+13.56` |

Interpretation:

- Replacing discourse markers and expanding contractions both reduce downstream latent baseline activation by about half.
- The upstream-ablation effect also shrinks on edited text, consistent with these being surface-form-sensitive lexical paths.
- This is stronger than held-out matching alone: it shows local text edits in the same documents move the downstream latent in the expected direction.

### Score-Path Ablations

I added `scripts/measure_pangram_sae_score_ablation.py` to ablate one or more SAE nodes and measure the detector target-score effect directly. For the two cleanest lexical paths, I ablated the upstream node alone, downstream node alone, and both nodes together.

| Target set | Nodes ablated | Docs | Mean class-3 logit drop | Mean class-3 probability drop |
|---|---|---:|---:|---:|
| held-out discourse markers | `L19:3450` | 24 | `0.0465` | `0.0017` |
| held-out discourse markers | `L24:3469` | 24 | `0.0368` | `0.0018` |
| held-out discourse markers | `L19:3450 + L24:3469` | 24 | `0.0531` | `0.0024` |
| counterfactual discourse rows | `L19:3450` | 32 | `0.0402` | `0.0009` |
| counterfactual discourse rows | `L24:3469` | 32 | `0.0234` | `0.0010` |
| counterfactual discourse rows | `L19:3450 + L24:3469` | 32 | `0.0421` | `0.0012` |
| held-out negation fragments | `L19:913` | 24 | `0.1136` | `0.0144` |
| held-out negation fragments | `L24:1310` | 24 | `0.0325` | `0.0053` |
| held-out negation fragments | `L19:913 + L24:1310` | 24 | `0.1178` | `0.0150` |
| counterfactual negation rows | `L19:913` | 32 | `0.1049` | `0.0137` |
| counterfactual negation rows | `L24:1310` | 32 | `0.0273` | `0.0045` |
| counterfactual negation rows | `L19:913 + L24:1310` | 32 | `0.1085` | `0.0143` |

Interpretation:

- Both lexical paths are positive contributors to the detector's target AI-style score on held-out and counterfactual rows.
- The joint ablations are subadditive: ablating both nodes moves the score less than the sum of the single-node effects. That is consistent with the nodes lying on an overlapping path rather than independently contributing unrelated score evidence.
- The negation/contraction path has a larger detector-readout effect than the discourse-marker path. Most of that effect is carried by the layer-19 node, while the layer-24 node remains useful as downstream feature evidence.
- The counterfactual rows have weaker effects than the held-out pattern rows, consistent with the local-edit result: removing or expanding the target surface form reduces path-relevant evidence.

### Held-Out Score Controls

I added `scripts/build_pangram_sae_score_control_panel.py` and `scripts/run_pangram_sae_score_ablation_panel.py` to evaluate many score ablations with one model load. The held-out control panel reuses the existing matched control selections, but applies them to the held-out discourse-marker and negation-fragment target sets.

Artifacts:

- Panel: `artifacts/pangram_llama_sae/circuit_discovery/score_control_panel_heldout_lexical_l19_l24.csv`
- Detail rows: `artifacts/pangram_llama_sae/circuit_discovery/score_control_panel_heldout_lexical_l19_l24_details.csv`
- Per-ablation summary: `artifacts/pangram_llama_sae/circuit_discovery/score_control_panel_heldout_lexical_l19_l24_summary.csv`
- Grouped summary: `artifacts/pangram_llama_sae/circuit_discovery/score_control_panel_heldout_lexical_l19_l24_grouped_summary.csv`

Grouped mean class-3 logit drops:

| Target | Ablation group | Rows | Mean logit drop | Range |
|---|---|---:|---:|---:|
| discourse markers | anchor upstream only | 1 | `0.0465` | `0.0465` to `0.0465` |
| discourse markers | anchor downstream only | 1 | `0.0368` | `0.0368` to `0.0368` |
| discourse markers | anchor joint | 1 | `0.0531` | `0.0531` to `0.0531` |
| discourse markers | control upstream only | 5 | `-0.0059` | `-0.0150` to `0.0039` |
| discourse markers | control upstream + anchor downstream | 5 | `0.0264` | `0.0166` to `0.0339` |
| discourse markers | control downstream only | 5 | `0.0031` | `-0.0195` to `0.0221` |
| discourse markers | anchor upstream + control downstream | 5 | `0.0478` | `0.0270` to `0.0684` |
| negation fragments | anchor upstream only | 1 | `0.1136` | `0.1136` to `0.1136` |
| negation fragments | anchor downstream only | 1 | `0.0325` | `0.0325` to `0.0325` |
| negation fragments | anchor joint | 1 | `0.1178` | `0.1178` to `0.1178` |
| negation fragments | control upstream only | 5 | `0.0018` | `-0.0042` to `0.0147` |
| negation fragments | control upstream + anchor downstream | 5 | `0.0321` | `0.0278` to `0.0401` |
| negation fragments | control downstream only | 5 | `0.0090` | `-0.0107` to `0.0205` |
| negation fragments | anchor upstream + control downstream | 5 | `0.1210` | `0.1055` to `0.1351` |

Interpretation:

- The layer-19 anchor nodes are strongly score-specific. Matched layer-19 upstream controls are near zero for both discourse (`-0.0059`) and negation (`0.0018`), while the anchors are `0.0465` and `0.1136`.
- The layer-24 anchor nodes also beat matched downstream controls on average, but less decisively: discourse `0.0368` vs control mean `0.0031`; negation `0.0325` vs control mean `0.0090`.
- The joint score readout is not path-specific in the strictest sense. Pairing the anchor upstream with other downstream controls can match or exceed the anchor joint effect, especially for negation (`0.1210` control mean vs `0.1178` anchor joint).
- Therefore the current evidence should be phrased carefully: these are strong layer-19 score features and specific `L19 -> L24` downstream-activation couplings. The final detector score is mostly carried by the layer-19 node, while the layer-24 node provides better evidence about path structure than about unique readout contribution.

### Independent Minimal-Pair Check

I added `scripts/build_pangram_sae_minimal_pair_targets.py` to create short, hand-written pairs independent of the corpus rows:

- Discourse marker pairs: `artifacts/pangram_llama_sae/circuit_discovery/minimal_pair_discourse_marker_targets.csv`
- Negation/contraction pairs: `artifacts/pangram_llama_sae/circuit_discovery/minimal_pair_negation_fragment_targets.csv`

Intervention artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/minimal_pair_intervention_discourse_marker_L19_3450_to_L24_3469.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/minimal_pair_intervention_negation_fragment_L19_913_to_L24_1310.csv`

Result:

| Target | Variant | Mean upstream baseline mass | Mean downstream baseline mass | Mean downstream mass delta | Mean class-3 logit drop |
|---|---|---:|---:|---:|---:|
| discourse markers | marked | `0.266` | `0.000` | `0.000` | `-0.0002` |
| discourse markers | plain | `0.000` | `0.000` | `0.000` | `0.0000` |
| negation fragments | marked | `0.175` | `0.000` | `0.000` | `-0.0005` |
| negation fragments | plain | `0.066` | `0.000` | `0.000` | `0.0000` |

Interpretation:

- This is a negative result for short hand-written minimal pairs. They sometimes weakly activate the layer-19 upstream feature, but they do not activate the layer-24 downstream feature at all.
- That does not falsify the held-out/counterfactual corpus result, because those rows are much more in-distribution for the detector and the SAE latents.
- It does constrain the interpretation: these layer-24 lexical latents are not simple word detectors that fire on any isolated occurrence. They appear to require generated-corpus context, longer surrounding style, stronger feature density, or a conjunction of local lexical and broader distributional evidence.
- The next minimal-pair test should use in-distribution generated completions with controlled edits, rather than short hand-written snippets.

### Long Synthetic Stress Checks

I added two lightweight follow-ups to separate "too short" from "wrong context":

- `scripts/build_pangram_sae_long_stress_targets.py`: creates longer marked/plain stress pairs and combined corpus-length marked/plain stress pairs.
- `scripts/analyze_pangram_sae_lexical_contexts.py`: joins target texts to intervention rows and summarizes length, feature-hit density, upstream mass, downstream mass, and score movement.

Artifacts:

- `docs/experiments/pangram_sae_lexical_context_analysis.md`
- `artifacts/pangram_llama_sae/circuit_discovery/lexical_context_analysis_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/long_stress_intervention_discourse_marker_L19_3450_to_L24_3469.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/long_stress_intervention_negation_fragment_L19_913_to_L24_1310.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/combined_long_stress_intervention_discourse_marker_L19_3450_to_L24_3469.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/combined_long_stress_intervention_negation_fragment_L19_913_to_L24_1310.csv`

Key summary:

| Target | Group | Variant | Mean words | Mean feature hits | Mean upstream mass | Mean downstream mass | Mean downstream mass delta |
|---|---|---|---:|---:|---:|---:|---:|
| discourse | corpus held-out | original | `387` | `4.4` | `22.83` | `28.87` | `-23.80` |
| discourse | combined synthetic | marked | `374` | `16.0` | `0.59` | `0.00` | `0.00` |
| discourse | combined synthetic | plain | `352` | `0.0` | `0.00` | `0.00` | `0.00` |
| negation | corpus held-out | original | `420` | `7.7` | `28.06` | `33.39` | `-28.67` |
| negation | combined synthetic | marked | `357` | `33.0` | `67.08` | `89.91` | `-71.78` |
| negation | combined synthetic | plain | `392` | `0.0` | `30.74` | `32.63` | `-25.98` |

Interpretation:

- The discourse-marker path is not a simple marker-density feature. Even a corpus-length synthetic document with sixteen discourse markers did not recruit `L24:3469`. The layer-24 discourse latent appears to depend on a broader expository/generated-corpus style state, not merely the presence of `However`, `Additionally`, or similar markers.
- The negation/contraction path is much closer to a surface-form-density feature, but not purely so. Corpus-length contraction-heavy synthetic text strongly activates `L24:1310`, and `L19:913` ablation collapses it. Expanded/plain synthetic text still has substantial residual activation, suggesting the path also responds to surrounding narrative negation structure or related syntax.
- This changes the next experiment design: discourse needs generated or corpus-derived minimal pairs that preserve the expository style manifold; negation can be tested with controlled contraction-density sweeps.

### Negation Dose-Response Sweep

I added `scripts/build_pangram_sae_negation_density_sweep.py` to create a controlled narrative where the same expanded base text receives the first N contractions in order. I then ran `L19:913 -> L24:1310` intervention on the six sweep rows.

Artifact summary:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_targets.csv`
- Intervention rows: `artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_intervention_L19_913_to_L24_1310.csv`
- Compact summary: `artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_summary.csv`

| Variant | Contractions | Upstream mass | Downstream mass | Downstream mass delta | Class-3 logit drop |
|---|---:|---:|---:|---:|---:|
| `contract_00` | 0 | `30.69` | `32.41` | `-25.86` | `0.0000` |
| `contract_02` | 2 | `27.85` | `29.29` | `-22.58` | `0.0625` |
| `contract_04` | 4 | `26.30` | `29.28` | `-23.98` | `0.0625` |
| `contract_08` | 8 | `25.96` | `31.26` | `-24.59` | `0.0000` |
| `contract_16` | 16 | `39.14` | `52.78` | `-42.16` | `0.0938` |
| `contract_32` | 32 | `60.02` | `78.91` | `-63.22` | `0.0312` |

Interpretation:

- This is the cleanest dose-response evidence so far. Contraction count correlates strongly with upstream mass (`0.94`), downstream mass (`0.97`), and downstream ablation effect (`-0.97`).
- There is a large residual baseline at zero contractions, so the feature is not only the apostrophe form. It also responds to expanded negation syntax or the broader narrative negation frame.
- The detector class-3 logit drop does not track contraction count cleanly (`0.13` correlation), so this path is a stronger latent-circuit result than final-score-readout result.

Token-locus inspection now clarifies what the residual baseline means. I added `scripts/inspect_pangram_sae_token_loci.py` and ran it on the density sweep for both `L19:913` and `L24:1310`.

Artifacts:

- `docs/experiments/pangram_sae_negation_density_sweep_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_token_loci_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_token_loci_class_summary.csv`

Token-class transition:

| Node | Contractions | Contraction-piece fraction | Expanded-negation fraction | Sentence-boundary fraction |
|---|---:|---:|---:|---:|
| `L19:913` | 0 | `0.00` | `0.61` | `0.11` |
| `L19:913` | 8 | `0.33` | `0.27` | `0.13` |
| `L19:913` | 16 | `0.75` | `0.00` | `0.06` |
| `L19:913` | 32 | `0.89` | `0.00` | `0.04` |
| `L24:1310` | 0 | `0.00` | `0.38` | `0.31` |
| `L24:1310` | 8 | `0.38` | `0.12` | `0.25` |
| `L24:1310` | 16 | `0.63` | `0.00` | `0.21` |
| `L24:1310` | 32 | `0.79` | `0.00` | `0.10` |

Interpretation:

- At low contraction counts, both nodes fire on expanded negation auxiliaries such as `did` and `do`, with `L24:1310` also picking up nearby sentence boundaries.
- At high contraction counts, both nodes shift to contraction subword pieces such as `didn`, `wouldn`, `couldn`, `wasn`, and `shouldn`.
- This supports a two-component interpretation: broad negation/narrative structure at low density, plus a direct contraction-piece path at high density.

Held-out corpus token loci validate the high-density side of this interpretation. I ran the same token-locus inspector on the 24 held-out negation-fragment documents:

Artifacts:

- `docs/experiments/pangram_sae_heldout_negation_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/heldout_negation_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/heldout_negation_token_loci_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/heldout_negation_token_loci_class_summary.csv`

Mean token-class fractions across held-out documents:

| Node | Mean active tokens | Mean mass | Contraction-piece fraction | Expanded-negation fraction | Sentence-boundary fraction |
|---|---:|---:|---:|---:|---:|
| `L19:913` | `9.21` | `28.01` | `0.575` | `0.000` | `0.154` |
| `L24:1310` | `7.96` | `33.27` | `0.651` | `0.000` | `0.077` |

Top held-out peak tokens are contraction pieces: `couldn`, `didn`, `wasn`, `hadn`, `don`, and `wouldn`. This means the corpus examples are not mainly using the expanded-negation baseline seen in the synthetic `contract_00` row; they are predominantly recruiting the direct contraction-piece component of the path.

### Contraction Sibling Latents

I compared `L19:913 -> L24:1310` against the nearest contraction-heavy sibling latents from the peak-token summaries:

- Layer 19: `L19:2787`, `L19:1650`
- Layer 24: `L24:62`

Artifacts:

- `docs/experiments/pangram_sae_negation_density_sweep_sibling_token_loci.md`
- `artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_sibling_token_loci_details.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_sibling_token_loci_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_sibling_token_loci_class_summary.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_intervention_sibling_L19_2787_to_L24_62.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_intervention_cross_L19_913_to_L24_62.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/negation_density_sweep_intervention_cross_L19_2787_to_L24_1310.csv`

Sibling token-locus summary:

| Node | Mass correlation with contractions | Max mass | Contraction-piece fraction at 32 contractions |
|---|---:|---:|---:|
| `L19:913` | `0.940` | `60.02` | `0.893` |
| `L19:2787` | `0.994` | `53.68` | `1.000` |
| `L19:1650` | `0.504` | `3.38` | `0.400` |
| `L24:1310` | `0.972` | `78.91` | `0.793` |
| `L24:62` | `0.944` | `72.26` | `0.957` |

Cross-pair intervention matrix on the density sweep:

| Upstream | Downstream | Mean downstream mass delta | Mass-delta correlation with contractions | Mean class-3 logit drop |
|---|---|---:|---:|---:|
| `L19:913` | `L24:1310` | `-33.73` | `-0.971` | `0.042` |
| `L19:2787` | `L24:62` | `-15.73` | `-0.969` | `0.000` |
| `L19:913` | `L24:62` | `-5.50` | `-0.786` | `0.042` |
| `L19:2787` | `L24:1310` | `-8.43` | `-0.991` | `0.000` |

Interpretation:

- `L19:913 -> L24:1310` is not the only contraction circuit. `L19:2787 -> L24:62` is a real sibling contraction path with strong dose response and causal downstream suppression.
- The anchor path is more score-relevant: `L19:913` ablations move the class-3 score, while `L19:2787` ablations do not in this sweep.
- Cross-pairs are weaker than matched pairs. This suggests partial modularity inside a broader contraction family rather than one fully shared contraction direction.
- `L19:1650` is not a strong member of the synthetic contraction family despite having some contraction-heavy browser peaks; it has very low mass on the sweep and appears mixed with human/function-word or boundary behavior.

### BROAD-6: short practical-answer / advice-and-options register

I screened `L19:401 -> L24:3218` from the direct layer-19-to-layer-24 queue as a GPT-heavy practical-answer candidate.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_short_advice_L19_401_to_L24_3218.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_ai_short_advice_L19_401_to_L24_3218.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_short_advice_L19_401_to_L24_3218.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_ai_short_advice_summary.csv`
- Token loci: `docs/experiments/pangram_sae_ai_short_advice_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `13` | `-243.27` | `-5.59` | `0.048` |

Matched controls:

| Control family | Largest absolute downstream mass delta |
|---|---:|
| different upstream, same downstream | `0.58` |
| same upstream, different downstream | `3.33` |

Token-locus summary:

| Metric | Value |
|---|---:|
| Same peak token | `9/13` (`69.2%`) |
| Mean top-token Jaccard | `0.666` |
| Mean active tokens, `L19:401` | `152.8` |
| Mean active tokens, `L24:3218` | `115.6` |

Interpretation:

- This is a real path-specific downstream dependency: ablating `L19:401` nearly erases `L24:3218`, while matched controls barely move it.
- Unlike some broad register paths, the upstream and downstream token loci are strongly aligned. This makes it look like a replicated practical-answer register feature across layers rather than a broad-context-to-boundary mediation.
- Examples include concise advice, recommendations, product/garden/travel lists, brief explainers, and compressed narrative advice.
- It is still broad and word-heavy, so the right next step is decomposition: test whether the feature is driven by advice imperatives, recommendation/list vocabulary, answer-compression structure, or some mixture of those.

### EXPO-1C: period/line-break upstream into broad simple-prose word accumulator

I screened `L19:3615 -> L24:1120` because `L19:3615` was already a known expository-boundary upstream for `EXPO-1`, while `L24:1120` looked like a different broad downstream.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_expo_boundary_alt_L19_3615_to_L24_1120.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_expo_boundary_alt_L19_3615_to_L24_1120.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_expo_boundary_alt_L19_3615_to_L24_1120.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_expo_boundary_alt_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_expo_boundary_alt_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `15` | `-68.92` | `-0.34` | `0.364` |

Matched controls:

| Control family | Mean abs downstream mass delta | Mean class-3 logit drop |
|---|---:|---:|
| different upstream, same downstream | `0.64` | `-0.062` |
| same upstream, different downstream | `0.99` | `0.364` |

Token-locus summary:

| Metric | `L19:3615` | `L24:1120` |
|---|---:|---:|
| Mean active tokens/doc | `38.1` | `251.2` |
| Mean mass/doc | `69.65` | `1825.11` |
| Dominant token classes | sentence-boundary / line-break | word / function-word |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `0/15` (`0.0%`) |
| Mean top-token Jaccard | `0.029` |

Interpretation:

- This is a real downstream dependency: ablating `L19:3615` drops `L24:1120` by much more than matched controls.
- It is not a local same-token boundary path. `L19:3615` fires mostly on periods and paragraph/line breaks, while `L24:1120` is a broad word-level accumulator.
- The class-score movement follows the upstream: same-upstream/different-downstream controls preserve the `0.364` class-3 logit drop.
- Examples are mostly simple, formulaic human prose and poetry, with a few GPT/Qwen passages. This should be treated as a boundary/rhythm-to-simple-prose mediation path, not as a compact slop marker.

### SIMPLE-1: broad simple/formulaic prose same-family path

I screened `L19:3990 -> L24:1120` after finding that both nodes recur around simple/formulaic prose: `L19:3990` was already a broad upstream into the `EXPO-1` boundary terminal, and `L24:1120` appeared in `EXPO-1C` as a broad simple-prose downstream.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_human_simple_to_simple_prose_L19_3990_to_L24_1120.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_human_simple_to_simple_prose_L19_3990_to_L24_1120.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_simple_to_simple_prose_L19_3990_to_L24_1120.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_simple_to_simple_prose_summary.csv`
- Token loci: `docs/experiments/pangram_sae_human_simple_to_simple_prose_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `23` | `-1918.34` | `-10.17` | `0.779` |

Matched controls:

| Control family | Mean abs downstream mass delta | Notes |
|---|---:|---|
| different upstream, same downstream | `64.17` | mostly tiny relative to target; one control increases L24 mass by `+291.58` rather than collapsing it |
| same upstream, different downstream | `1.63` | score movement follows upstream, but downstream collapse is specific to `L24:1120` |

Token-locus summary:

| Metric | `L19:3990` | `L24:1120` |
|---|---:|---:|
| Mean active tokens/doc | `274.3` | `271.4` |
| Mean mass/doc | `1363.81` | `2222.81` |
| Dominant token classes | word / punctuation / sentence-boundary | word / function-word |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `6/23` (`26.1%`) |
| Mean top-token Jaccard | `0.376` |

Interpretation:

- This is the dominant path into `L24:1120`. Its downstream collapse is much larger than the `EXPO-1C` boundary/rhythm feed into the same terminal.
- Both nodes are broad word-heavy features with moderate top-token overlap. This is closer to a same-family simple/formulaic prose transfer than a local punctuation circuit.
- The examples are mostly human rows, plus a few GPT rows: simple first-person stories, formulaic scene/dialogue prose, movie/list answers, childlike poems, broad explanatory paragraphs, and simple advice/idea lists.
- The direction is AI-score-positive despite the mostly human source mix. The safer interpretation is that the detector treats this formulaic/simple answer style as AI-like, not that the latent directly tracks source identity.

### VERSE-4: lineated poem structure into ornate verse word accumulator

I screened `L19:3894 -> L24:953` as a possible alternate upstream into the `L24:953` family. Earlier `DESCR-1` used `L19:3992 -> L24:943`; this candidate instead lands on `L24:953` and is much more verse-specific.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_poetry_to_descr_boundary_L19_3894_to_L24_953.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_poetry_to_descr_boundary_L19_3894_to_L24_953.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_poetry_to_descr_boundary_L19_3894_to_L24_953.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_poetry_to_descr_boundary_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_poetry_to_descr_boundary_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `21` | `-51.00` | `-0.25` | `0.151` |

Matched controls:

| Control family | Mean abs downstream mass delta | Mean class-3 logit drop |
|---|---:|---:|
| different upstream, same downstream | `0.45` | `-0.001` |
| same upstream, different downstream | `0.23` | `0.151` |

Token-locus summary:

| Metric | `L19:3894` | `L24:953` |
|---|---:|---:|
| Mean active tokens/doc | `23.5` | `110.6` |
| Mean mass/doc | `56.69` | `840.98` |
| Dominant token classes | line-break / sentence-boundary | word / punctuation / line-break |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `0/21` (`0.0%`) |
| Mean top-token Jaccard | `0.007` |

Interpretation:

- This is a real alternate feed: same-downstream controls barely move `L24:953`, while `L19:3894` ablation drops it by `-51.00` mass.
- It is a mediation path, not a local same-token path. The upstream fires on line endings and line breaks (`.\n`, `.\n\n`, `;\n`, `,\n`), while the downstream fires broadly across ornate/rhymed verse wording.
- The source set is entirely model-generated and consists of lineated/rhymed poems about brains, stars, summer, pirates, ice cream, apocalypse, poppies, mirrors, bread, cats, and Costco.
- This should be grouped with verse-format circuits, not broad assistant slop. It is useful because it shows a structural lineation feature feeding a broad word-register latent.

### BOUNDARY-4B: sentimental/rhetorical model-prose feed into BOUNDARY-4 terminal

I screened `L19:3600 -> L24:3970` as an alternate downstream from the already confirmed `BROAD-5` upstream and an alternate upstream into the known `BOUNDARY-4` downstream.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_model_story_to_boundary_L19_3600_to_L24_3970.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_ai_model_story_to_boundary_L19_3600_to_L24_3970.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_model_story_to_boundary_L19_3600_to_L24_3970.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_ai_model_story_to_boundary_summary.csv`
- Token loci: `docs/experiments/pangram_sae_ai_model_story_to_boundary_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-19.82` | `-0.24` | `0.017` |

Matched controls:

| Control family | Mean abs downstream mass delta |
|---|---:|
| different upstream, same downstream | `0.26` |
| same upstream, different downstream | `1.95` |

Token-locus summary:

| Metric | `L19:3600` | `L24:3970` |
|---|---:|---:|
| Mean active tokens/doc | `293.4` | `72.9` |
| Mean mass/doc | `564.30` | `227.98` |
| Dominant token classes | word / function-word | sentence-boundary / punctuation / line-break |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `0/14` (`0.0%`) |
| Mean top-token Jaccard | `0.000` |

Interpretation:

- This is weaker than the main `BROAD-5` edge and weaker than the main `BOUNDARY-4` edge, but it is not noise: same-downstream controls barely move `L24:3970`.
- The mechanism is broad-context-to-boundary mediation. `L19:3600` fires across sentimental, reflective, graduation-speech, letter, and narrative model prose; `L24:3970` fires on periods and paragraph-final boundaries.
- The result suggests `L24:3970` is a shared downstream boundary terminal for multiple broad generated-prose states rather than a single-source rhetorical feature.

### LETTER-2: correspondence/email body punctuation and paragraph rhythm

I screened `L19:1496 -> L24:3971` after earlier `L24:3971` nominations looked weak. `L19:1049 -> L24:3971` was score-only, and `L19:3424 -> L24:3971` barely moved the downstream latent. This candidate is different: it gives a large, path-specific downstream collapse and a compact token-locus match.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_letter_to_weak3971_L19_1496_to_L24_3971.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_letter_to_weak3971_L19_1496_to_L24_3971.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_letter_to_weak3971_L19_1496_to_L24_3971.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_letter_to_weak3971_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_letter_to_weak3971_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `22` | `-70.47` | `-4.76` | `0.013` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `0.17` | `0.41` |
| same upstream, different downstream | `0.70` | `1.09` |

Token-locus summary:

| Metric | `L19:1496` | `L24:3971` |
|---|---:|---:|
| Mean active tokens/doc | `36.6` | `34.5` |
| Mean mass/doc | `71.20` | `105.46` |
| Mean max activation | `5.44` | `8.22` |
| Dominant token classes | sentence-boundary / line-break / punctuation | sentence-boundary / line-break / punctuation |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `10/22` (`45.5%`) |
| Mean top-token Jaccard | `0.790` |

Interpretation:

- This is the confirmed upstream for `L24:3971`. The controls are two orders of magnitude smaller than the target collapse, so the edge should no longer be treated as a weak or generic `L24:3971` nomination.
- The circuit is correspondence-format evidence, but not the same as `LETTER-1`. `LETTER-1` is a closing/signoff path; `LETTER-2` is body-level punctuation and paragraph rhythm inside email/letter-like documents.
- The strongest active tokens are periods, paragraph-final periods, commas, paragraph breaks, bracket placeholders, and occasional correspondence words such as `Thank` and `Please`.
- Examples include PTO requests, marketing outreach, apology/rest messages, project issue letters, prayers, performance reviews, city-council letters, project status emails, recruiting outreach, birthday-gift messages, business offering letters, salary-review requests, mayor letters, and layoff emails.
- The classifier movement is nearly zero (`+0.013`), so the immediate claim is mechanistic/style-format rather than a detector-score path.

### CONTROL-11: broad plain human explanation/story/advice continuation direction

I screened `L19:2346 -> L24:3889` because both nodes were near the top of the human-signifying direct-edge queue and shared examples that looked like plain explanatory, advice, and simple story prose. The first-pass effect was large enough to justify controls and token loci.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_human_practical_plain_L19_2346_to_L24_3889.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_human_practical_plain_L19_2346_to_L24_3889.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_practical_plain_L19_2346_to_L24_3889.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_practical_plain_summary.csv`
- Token loci: `docs/experiments/pangram_sae_human_practical_plain_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `12` | `-1257.63` | `-9.08` | `-0.746` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `6.76` | `13.18` |
| same upstream, different downstream | `10.88` | `30.06` |

Token-locus summary:

| Metric | `L19:2346` | `L24:3889` |
|---|---:|---:|
| Mean active tokens/doc | `214.1` | `216.5` |
| Mean mass/doc | `862.66` | `1445.29` |
| Mean max activation | `7.96` | `13.43` |
| Dominant token classes | word / sentence-boundary / function-word | word / sentence-boundary / punctuation |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `6/12` (`50.0%`) |
| Mean top-token Jaccard | `0.633` |

Interpretation:

- This is a strong path-specific human-side dependency. Alternate upstreams and alternate downstreams are far smaller than the target collapse.
- The target-score movement is negative (`-0.746` class-3 logit drop), meaning ablating the upstream raises the model's AI score. This belongs with human/form controls, not AI slop circuits.
- The loci are broad and mostly word-level, with many active tokens per document. Top examples include plain Rome/history explanation, task-prioritization advice, food-choice advice, chalk-step introspection, everyday dating-story narration, and bug/troubleshooting explanation.
- Because both nodes are broad and highly overlapping, this is best treated as a same-family plain-human-prose continuation path rather than a compact feature like a punctuation, signoff, or lexical phrase circuit.

### VOICE-1: playful/childlike narrative dialogue and interjection punctuation

I screened `L19:1183 -> L24:3186` because the target examples clustered around childlike stories, first-person animal/character narration, dialogue, and playful interjections. The effect is smaller than the broad human controls, but it is path-specific and has a clear token-locus profile.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_narrative_voice_L19_1183_to_L24_3186.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_narrative_voice_L19_1183_to_L24_3186.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_narrative_voice_L19_1183_to_L24_3186.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_narrative_voice_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_narrative_voice_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `19` | `-129.24` | `-6.03` | `-0.054` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `1.60` | `7.12` |
| same upstream, different downstream | `1.35` | `4.79` |

Token-locus summary:

| Metric | `L19:1183` | `L24:3186` |
|---|---:|---:|
| Mean active tokens/doc | `64.9` | `70.3` |
| Mean mass/doc | `143.58` | `273.12` |
| Mean max activation | `7.01` | `13.36` |
| Dominant token classes | sentence-boundary / punctuation / word / line-break | sentence-boundary / word / punctuation / line-break |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `12/19` (`63.2%`) |
| Mean top-token Jaccard | `0.601` |

Interpretation:

- This is a real path-specific dependency: matched controls are small compared with the `-129.24` target mass collapse.
- The feature is not just closing quote punctuation. It fires on sentence boundaries, quote marks, colons, exclamation/question marks, paragraph breaks, and interjection-like tokens such as `Ow`, `Ah`, elongated cries, and quoted child/story dialogue.
- Examples include scared animal inner monologue, Timmy/Goldilocks-style children's stories, talking animals, first-person dog narration, Star Wars log-entry parody, kitchen/smoke-alarm monologue, and domestic mystery/dialogue scenes.
- The score movement is small and human-leaning, so this belongs with format/style circuits and controls rather than detector-positive slop markers.

### CODE-2: code-block boundary / code-to-explanation structure circuit

I screened `L19:1878 -> L24:3070` after two related code-family cross-pairs had looked weak: `L19:1878 -> L24:2110` and `L19:731 -> L24:3070`. The matched pair is strong, which suggests the code/explanation family has at least two specific upstream/downstream pairings rather than freely interchangeable code latents.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_code_alt_to_unknown_L19_1878_to_L24_3070.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_code_alt_to_unknown_L19_1878_to_L24_3070.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_code_alt_to_unknown_L19_1878_to_L24_3070.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_code_alt_to_unknown_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_code_alt_to_unknown_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `28` | `-81.96` | `-8.49` | `-0.160` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `0.24` | `0.58` |
| same upstream, different downstream | `2.05` | `9.82` |

Token-locus summary:

| Metric | `L19:1878` | `L24:3070` |
|---|---:|---:|
| Mean active tokens/doc | `42.0` | `49.5` |
| Mean mass/doc | `79.68` | `135.19` |
| Mean max activation | `7.92` | `12.79` |
| Dominant token classes | word / line-break / punctuation | word / line-break / punctuation |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `23/28` (`82.1%`) |
| Mean top-token Jaccard | `0.656` |

Interpretation:

- This is a clean path-specific dependency: same-downstream controls are effectively zero, and same-upstream alternate downstreams are much smaller than the target edge.
- The active loci are code-block and code-to-explanation boundaries: `}\n\n`, blank lines, `}\n`, `)\n`, `);\n`, `();\n`, `{\n`, `:\n\n`, periods after code/explanation transitions, `display.display();`, `print(...)`, array/dictionary snippets, and comments.
- Examples span Arduino/OLED display code, Scrabble scoring, C `absmax`, complex arctangent code, string parsing, matplotlib, Roman numerals, running sums, pandas DataFrames, React Router, TypeScript async map, and book-list organizers.
- The score movement is negative, so this is another human/non-AI-style code/explanation control rather than an AI-slop circuit.
- The result updates the earlier weak-edge interpretation: `L24:3070` was weak for `L19:731`, and `L19:1878` was weak for `L24:2110`, but `L19:1878 -> L24:3070` is the matching code-block boundary circuit.

### M-LEX-1: `m`/`M` lexical-word path

I screened `L19:834 -> L24:2349` because the shared examples initially looked like a mixed rhythm/description/practical-prose cluster. The intervention and controls were strong, but token loci showed the actual feature is lexical rather than stylistic.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_rhythm_description_L19_834_to_L24_2349.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_rhythm_description_L19_834_to_L24_2349.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_rhythm_description_L19_834_to_L24_2349.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_rhythm_description_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_rhythm_description_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `12` | `-47.58` | `-5.95` | `-0.002` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `0.095` | `0.196` |
| same upstream, different downstream | `0.101` | `0.179` |

Token-locus summary:

| Metric | `L19:834` | `L24:2349` |
|---|---:|---:|
| Mean active tokens/doc | `12.4` | `13.0` |
| Mean mass/doc | `38.63` | `49.36` |
| Mean max activation | `5.78` | `7.05` |
| Dominant token classes | word only | word only |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `12/12` (`100.0%`) |
| Mean top-token Jaccard | `0.849` |

Interpretation:

- This is a very clean direct path, but it is not a prose-style circuit. The active loci are almost entirely `m`/`M` word pieces.
- Top tokens include `mouth`, `milk`, `music`, `meal`, `market`, `Marco`, `mind`, `meals`, `melody`, `mountain`, `Michael`, `mimic`, `microwave`, and `morning`.
- The examples are mixed because those lexical pieces appear in different contexts: music/rhythm story, whole-food diet, cooking, ice-cream elegy, Penn/Teller dialogue, meal-prep advice, and descriptive narrative.
- This should be kept as a lexical/subword control and filtered out before making style or slop claims.

### E-LEX-1: `e`/`E` lexical-word path

I screened `L19:1094 -> L24:271` as a possible mixed story/verse/factual candidate. Like `M-LEX-1`, it is strongly causal and controlled, but token loci show a lexical-word-piece family.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_story_verse_L19_1094_to_L24_271.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_story_verse_L19_1094_to_L24_271.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_story_verse_L19_1094_to_L24_271.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_story_verse_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_story_verse_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-28.59` | `-6.64` | `0.0002` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `0.073` | `0.125` |
| same upstream, different downstream | `0.149` | `0.252` |

Token-locus summary:

| Metric | `L19:1094` | `L24:271` |
|---|---:|---:|
| Mean active tokens/doc | `9.3` | `7.6` |
| Mean mass/doc | `28.28` | `29.17` |
| Mean max activation | `6.11` | `7.21` |
| Dominant token classes | word only | word only |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `12/14` (`85.7%`) |
| Mean top-token Jaccard | `0.834` |

Interpretation:

- The direct-edge signal is real and path-specific, but the feature is lexical/subword rather than story or verse style.
- Top tokens include `earnings`, `ethnic`, `expectations`, `Echo`, `echo`, `empty`, `Eggs`, `Estate`, `education`, `every`, `entity`, and related `e`/`E` pieces.
- The surface examples span poetry, breakfast prose, whole-food diet, financial/news report, estate scene, philosophical mist text, demographics report, and anxiety/verse passages because the common factor is the token family.
- This is another positive lexical-control family. It strengthens the rule that direct-edge candidates need token-locus inspection before semantic labels.

### CONTROL-12: broad reflective/explanatory ordinary-word direction

I screened `L19:1223 -> L24:3506` because the target examples looked like a mixed reflective/philosophical/explanatory cluster. The causal edge is real, but the token loci make it a broad ordinary-word direction rather than a compact style marker.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_confusion_philosophy_L19_1223_to_L24_3506.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_confusion_philosophy_L19_1223_to_L24_3506.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_confusion_philosophy_L19_1223_to_L24_3506.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_confusion_philosophy_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_confusion_philosophy_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `20` | `-141.56` | `-3.05` | `-0.014` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `0.381` | `0.598` |
| same upstream, different downstream | `0.463` | `0.951` |

Token-locus summary:

| Metric | `L19:1223` | `L24:3506` |
|---|---:|---:|
| Mean active tokens/doc | `90.1` | `95.3` |
| Mean mass/doc | `152.77` | `213.85` |
| Mean max activation | `3.74` | `5.23` |
| Dominant token classes | word / punctuation / function words | word / punctuation / function words |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `6/20` (`30.0%`) |
| Mean top-token Jaccard | `0.550` |

Interpretation:

- This is a specific upstream/downstream dependency: matched controls are hundreds of times smaller than the target mass drop.
- The feature is broad and ordinary-token-heavy, with top loci such as commas, `to`, `and`, `is`, `the`, `you`, `that`, `I`, `have`, `but`, and `not`.
- The target set is mixed-source and human-heavy, and the score movement is slightly negative. It should be kept as a broad reflective/explanatory control, not a compact AI-slop circuit.

### LETTER-3: correspondence signoff/placeholders and paragraph-break format

I screened `L19:1049 -> L24:2242` because the same upstream had previously moved the detector score in the failed `L24:3971` nomination. This pass found the matching downstream: a compact correspondence-format path.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_letter_body_alt_L19_1049_to_L24_2242.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_letter_body_alt_L19_1049_to_L24_2242.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_letter_body_alt_L19_1049_to_L24_2242.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_letter_body_alt_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_letter_body_alt_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `21` | `-44.60` | `-7.64` | `0.238` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `0.113` | `0.182` |
| same upstream, different downstream | `0.098` | `0.231` |

Token-locus summary:

| Metric | `L19:1049` | `L24:2242` |
|---|---:|---:|
| Mean active tokens/doc | `19.2` | `20.9` |
| Mean mass/doc | `42.21` | `66.59` |
| Mean max activation | `7.41` | `10.38` |
| Dominant token classes | word / line-break / punctuation | word / line-break / punctuation |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `20/21` (`95.2%`) |
| Mean top-token Jaccard | `0.826` |

Interpretation:

- This is a clean, compact correspondence-format circuit.
- Top loci include `.\n\n`, `,\n\n`, bracket placeholders, `Your`, `Name`, `Dear`, `Hi`, `Sincerely`, and `regards`.
- This revises the earlier `SCORE-ONLY-1` interpretation: `L19:1049` is a real score-moving correspondence latent, but `L24:3971` was not its matching downstream. `L24:2242` is.

### CONTROL-13: broad expressive/comedic prose ordinary-word direction

I screened `L19:2238 -> L24:605` because the examples looked like expressive, comic, or sarcastic model prose. The edge is very high-mass and path-specific, but the score direction and loci argue for a broad expressive control rather than slop evidence.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_expressive_misc_L19_2238_to_L24_605.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_ai_expressive_misc_L19_2238_to_L24_605.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_expressive_misc_L19_2238_to_L24_605.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_ai_expressive_misc_summary.csv`
- Token loci: `docs/experiments/pangram_sae_ai_expressive_misc_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-560.95` | `-2.75` | `-0.114` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `0.827` | `2.503` |
| same upstream, different downstream | `3.886` | `17.652` |

Token-locus summary:

| Metric | `L19:2238` | `L24:605` |
|---|---:|---:|
| Mean active tokens/doc | `369.9` | `307.8` |
| Mean mass/doc | `569.67` | `603.21` |
| Mean max activation | `4.05` | `4.41` |
| Dominant token classes | word / punctuation / function words | word / punctuation / function words |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `4/14` (`28.6%`) |
| Mean top-token Jaccard | `0.383` |

Interpretation:

- The path is real: target mass drop is far larger than matched controls.
- The active loci are document-wide and ordinary-token-heavy, with common loci like commas, `to`, `is`, `and`, `a`, `that`, `It`, `they`, `absolutely`, and `must`.
- The examples are model-heavy, but ablating the path moves the classifier away from AI (`-0.114` class-3 logit drop). This is a good reminder that source mix alone is not enough; score direction and token breadth matter.

### WEAK-EDGE-10: weak verse-family alternate

I screened `L19:33 -> L24:2278` because the direct-edge examples looked verse-heavy and model-generated.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_verse_alt_L19_33_to_L24_2278.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_ai_verse_alt_L19_33_to_L24_2278.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_verse_alt_L19_33_to_L24_2278.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `22` | `-11.42` | `-0.041` | `-0.018` |

Interpretation:

- This is not strong enough to promote beside the confirmed verse circuits. The mass drop is modest and the peak activation barely moves.
- I did not spend controls or token-locus work on it. It should remain screened out unless a future pass finds stronger evidence for `L24:2278` with a different upstream.

### CONTROL-14: human practical/list sentence-boundary and separator circuit

I screened `L19:46 -> L24:3195` as the strongest remaining human-heavy direct edge not already represented in the registry. It is a real, controlled circuit, but it belongs on the human/practical-list side rather than the AI-slop side.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_human_unknown12_L19_46_to_L24_3195.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_human_unknown12_L19_46_to_L24_3195.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_unknown12_L19_46_to_L24_3195.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_unknown12_summary.csv`
- Token loci: `docs/experiments/pangram_sae_human_unknown12_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `9` | `-93.85` | `-3.58` | `-0.244` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `0.621` | `1.330` |
| same upstream, different downstream | `3.921` | `17.157` |

Token-locus summary:

| Metric | `L19:46` | `L24:3195` |
|---|---:|---:|
| Mean active tokens/doc | `104.3` | `42.4` |
| Mean mass/doc | `203.05` | `144.16` |
| Mean max activation | `6.12` | `8.00` |
| Dominant token classes | sentence boundary / punctuation | sentence boundary / line-break / punctuation |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `2/9` (`22.2%`) |
| Mean top-token Jaccard | `0.667` |

Interpretation:

- The target examples are all human practical/list advice: streaming setup, medical anxiety, recruiting, dietitian tech, saving money, weight loss, meal planning, and Instagram growth.
- The active loci are extremely boundary-heavy. `L19:46` fires mostly on periods and hyphen separators (` -`); `L24:3195` fires mostly on periods, paragraph-final periods, and hyphen separators.
- This is a useful control against over-reading practical/list surface genre. The text looks like the kind of advice/list prose that generated systems often produce, but the detector direction is human/non-AI and the mechanism is local boundary/separator rhythm.

### WEAK-EDGE-11 and WEAK-EDGE-12: human-side sibling screens

I screened two additional high-ranked human-side alternate edges after `CONTROL-14`.

`L19:3534 -> L24:1511`:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `13` | `-3.77` | `-0.069` | `-0.026` |

`L19:3457 -> L24:3645`:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `12` | `-43.51` | `-0.267` | `-0.499` |

Interpretation:

- `L19:3534 -> L24:1511` is too weak to promote: both mass and peak movement are tiny compared with confirmed `L24:1511` paths.
- `L19:3457 -> L24:3645` has a visible mass effect and a strong human-side score movement, but its peak movement is small relative to the confirmed `CONTROL-6` path into the same downstream (`-343.25` mass, `-4.83` max).
- I did not spend controls/loci on either. They remain screened-out sibling candidates unless a future pass specifically focuses on human-side mediation into `L24:1511` or `L24:3645`.

### CONTROL-15: broad elevated/ornate prose article-scaffold direction

I screened `L19:1146 -> L24:2940` because both nodes were high-rank mixed/humanlike latents and the first-pass intervention produced an unusually large downstream collapse.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown13_L19_1146_to_L24_2940.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown13_L19_1146_to_L24_2940.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown13_L19_1146_to_L24_2940.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown13_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown13_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `12` | `-2427.20` | `-10.69` | `-0.192` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `8.142` | `25.394` |
| same upstream, different downstream | `49.137` | `203.041` |

Token-locus summary:

| Metric | `L19:1146` | `L24:2940` |
|---|---:|---:|
| Mean active tokens/doc | `260.7` | `261.0` |
| Mean mass/doc | `2814.75` | `5329.45` |
| Mean max activation | `16.88` | `31.37` |
| Dominant token classes | word / sentence boundary / punctuation | word / punctuation |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `5/12` (`41.7%`) |
| Mean top-token Jaccard | `0.505` |

Interpretation:

- This is the largest same-feature dependency found in this latest pass. The target mass drop is roughly two orders of magnitude larger than same-downstream controls.
- The examples are elevated, ornate, poetic, or stylized narrative prose across human and model outputs.
- Token loci show a broad article/scaffold-word direction rather than a compact semantic marker: top tokens include `the`, `The`, `a`, `of`, commas, and sentence boundaries.
- The score movement is negative, so this should be treated as a mixed/humanlike ornate-prose control and future decomposition target, not as AI-slop evidence.

### WEAK-EDGE-13 and WEAK-EDGE-14: verse/ornate sibling screens

I screened two more verse/ornate-family alternate edges after `CONTROL-15`.

`L19:3733 -> L24:953`:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `29` | `+2.26` | `+0.043` | `-0.002` |

`L19:996 -> L24:4095`:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `20` | `-3.39` | `-0.234` | `-0.048` |

Interpretation:

- `L19:3733 -> L24:953` is not a meaningful alternate into the known ornate-verse downstream: the downstream mass slightly increases and max movement is tiny.
- `L19:996 -> L24:4095` is a weak alternate downstream for the confirmed rhymed end-word upstream. The confirmed `VERSE-2` path remains `L19:996 -> L24:1568`.
- I did not spend controls/loci on either weak screen.

### CONTROL-16: human practical/informal content-word mediation path

I screened `L19:2834 -> L24:3645` because `L19:2834` is already a strong human practical/expository upstream and `L24:3645` is a known human informal/mixed-prose downstream. This candidate is real and very specific, but it is a context-to-content mediation path rather than a local same-token circuit.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_human_practical_to_informal_alt_L19_2834_to_L24_3645.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_human_practical_to_informal_alt_L19_2834_to_L24_3645.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_practical_to_informal_alt_L19_2834_to_L24_3645.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_practical_to_informal_alt_summary.csv`
- Token loci: `docs/experiments/pangram_sae_human_practical_to_informal_alt_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `10` | `-105.73` | `-1.00` | `-0.151` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `0.158` | `0.310` |
| same upstream, different downstream | `1.115` | `2.004` |

Token-locus summary:

| Metric | `L19:2834` | `L24:3645` |
|---|---:|---:|
| Mean active tokens/doc | `311.3` | `131.0` |
| Mean mass/doc | `1062.55` | `530.87` |
| Mean max activation | `6.04` | `8.68` |
| Dominant token classes | word / function words | word |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `0/10` (`0.0%`) |
| Mean top-token Jaccard | `0.004` |

Interpretation:

- The target examples are all human and span practical advice, simple stories, reflective prose, and informal narrative.
- The downstream collapse is highly path-specific: matched controls are near zero.
- Token loci show almost no position overlap. The upstream is a broad human practical/expository state over ordinary function/content words, while the downstream is a narrower content-word state with tokens such as `life`, `people`, `finances`, `market`, `dream`, `mountain`, `happy`, and `box`.
- This belongs with the human/form controls. It is not local slop evidence, but it is useful for studying broad-to-content mediation on the human side.

### BROAD-7: sentimental/lyrical model-story word-register path

I screened `L19:3600 -> L24:591` because `L19:3600` already appears in model-story and sentimental narrative paths, while `L24:591` was an unscreened model-heavy downstream. The edge is controlled and specific, but score-neutral.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_ai_model_story_to_unknown_L19_3600_to_L24_591.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_ai_model_story_to_unknown_L19_3600_to_L24_591.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_ai_model_story_to_unknown_L19_3600_to_L24_591.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_ai_model_story_to_unknown_summary.csv`
- Token loci: `docs/experiments/pangram_sae_ai_model_story_to_unknown_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `17` | `-79.51` | `-1.08` | `0.000` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `0.881` | `1.949` |
| same upstream, different downstream | `0.595` | `1.462` |

Token-locus summary:

| Metric | `L19:3600` | `L24:591` |
|---|---:|---:|
| Mean active tokens/doc | `255.5` | `111.5` |
| Mean mass/doc | `475.79` | `273.10` |
| Mean max activation | `4.57` | `5.88` |
| Dominant token classes | word / function words | word |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `0/17` (`0.0%`) |
| Mean top-token Jaccard | `0.031` |

Interpretation:

- The examples are model-heavy sentimental, lyrical, reflective, and transformation-story prose.
- The path is very specific under controls, but it has no mean classifier-score movement.
- The upstream active loci are broad scaffold/voice words such as `to`, `the`, `in`, `that`, `I`, `was`, `am`, and `you`. The downstream loci are word-register tokens such as `peace`, `connection`, `light`, `journey`, `yourself`, `chaos`, `shelter`, `solitude`, and `strength`.
- This belongs with the broad model-produced register family, not with compact local slop markers.

### WEAK-EDGE-15: weak letter-family alternate

I screened `L19:3424 -> L24:2242` because `L19:3424` is the confirmed `LETTER-1` signoff upstream and `L24:2242` is the confirmed `LETTER-3` downstream.

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `22` | `-4.82` | `-0.576` | `0.056` |

Interpretation:

- This is too weak to promote. The confirmed `LETTER-3` pair `L19:1049 -> L24:2242` drops the same downstream by `-44.60` mass and `-7.64` max.
- The result helps separate correspondence-format subfamilies: `L19:3424` remains the signoff/closing upstream, while `L19:1049` is the correct upstream for `L24:2242`.

### CONTROL-17: broad human advice/list and recommendation prose direction

I screened `L19:1756 -> L24:776` because both nodes were all-human and high-rank. It is a very strong human-side path in advice, list, and recommendation prose.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_human_unknown14_L19_1756_to_L24_776.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_human_unknown14_L19_1756_to_L24_776.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_unknown14_L19_1756_to_L24_776.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_unknown14_summary.csv`
- Token loci: `docs/experiments/pangram_sae_human_unknown14_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `8` | `-954.69` | `-3.85` | `-0.330` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `2.485` | `10.232` |
| same upstream, different downstream | `49.390` | `185.219` |

Token-locus summary:

| Metric | `L19:1756` | `L24:776` |
|---|---:|---:|
| Mean active tokens/doc | `380.4` | `393.9` |
| Mean mass/doc | `1004.73` | `1718.02` |
| Mean max activation | `4.95` | `7.76` |
| Dominant token classes | word / function words / punctuation | word / function words / punctuation |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `1/8` (`12.5%`) |
| Mean top-token Jaccard | `0.175` |

Interpretation:

- The examples are all human advice/list/recommendation documents: 5K preparation, cryptocurrency alternatives, supernatural creatures, camping food, marketing audiences, travel destinations, martial arts, and RPG classes.
- The effect is highly path-specific but broad over hundreds of tokens. It is not a local list-marker circuit.
- The classifier movement is negative, so it is another strong human-side guardrail against interpreting advice/list surface form as AI-slop evidence.

### CONTROL-18: human narrative/adventure broad word-register path

I screened `L19:320 -> L24:848` because it was a high-ranked human-heavy edge with mixed-source downstream examples. The controls confirm a strong path-specific dependency, but the score direction is human/non-AI.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_human_mixed_unknown15_L19_320_to_L24_848.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_human_mixed_unknown15_L19_320_to_L24_848.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_mixed_unknown15_L19_320_to_L24_848.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_mixed_unknown15_summary.csv`
- Token loci: `docs/experiments/pangram_sae_human_mixed_unknown15_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `12` | `-435.21` | `-1.74` | `-0.098` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `4.851` | `19.773` |
| same upstream, different downstream | `0.497` | `1.598` |

Token-locus summary:

| Metric | `L19:320` | `L24:848` |
|---|---:|---:|
| Mean active tokens/doc | `322.8` | `383.4` |
| Mean mass/doc | `468.48` | `928.80` |
| Mean max activation | `2.81` | `5.07` |
| Dominant token classes | word / sentence boundary / punctuation | word / function words / punctuation |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `1/12` (`8.3%`) |
| Mean top-token Jaccard | `0.183` |

Interpretation:

- The examples are mostly human adventure, fantasy/RPG, detective, dialogue, and action scenes, with a couple model variants.
- The upstream has more sentence-boundary and punctuation support, while the downstream is broader and word-heavy (`was`, `to`, `It`, `that`, `had`, `she`, `but`, `who`).
- This is a broad narrative/adventure control, not a local boundary circuit and not slop evidence.

### WEAK-EDGE-16: weak alternate into verse lineation

I screened `L19:1325 -> L24:61` as another alternate upstream into the confirmed `VERSE-1` lineation downstream.

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `18` | `-3.37` | `-1.10` | `-0.017` |

Interpretation:

- The peak movement is nonzero, but the mass movement is tiny relative to confirmed verse circuits.
- This should stay screened out. `L24:61` remains anchored to `L19:2355` for the current lineated/rhymed verse path.

### CONTROL-19: human practical/simple-narrative content-word sibling into `L24:3645`

I screened `L19:3534 -> L24:3645` because it shares the human practical/informal downstream seen in `CONTROL-6` and `CONTROL-16`. The edge is real and controlled, but it is score-neutral to human-side rather than AI-slop evidence.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_human_practical_sibling_to_informal_L19_3534_to_L24_3645.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_human_practical_sibling_to_informal_L19_3534_to_L24_3645.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_practical_sibling_to_informal_L19_3534_to_L24_3645.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_practical_sibling_to_informal_summary.csv`
- Token loci: `docs/experiments/pangram_sae_human_practical_sibling_to_informal_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `10` | `-109.32` | `-1.58` | `-0.013` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `1.720` | `2.619` |
| same upstream, different downstream | `0.942` | `2.307` |

Token-locus summary:

| Metric | `L19:3534` | `L24:3645` |
|---|---:|---:|
| Mean active tokens/doc | `112.1` | `148.8` |
| Mean mass/doc | `220.40` | `586.29` |
| Mean max activation | `4.44` | `8.48` |
| Dominant token classes | word / sparse function words | word / function words |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `1/10` (`10.0%`) |
| Mean top-token Jaccard | `0.375` |

Interpretation:

- The target set is all human: roof/buying advice, a samurai scene, class hierarchy explanation, hiking advice, finance advice, reflective prose, a road/release scene, and fantasy rescue/adoption.
- The active tokens are broad practical/simple-narrative content words such as `years`, `market`, `expenses`, `finances`, `fight`, `life`, `dream`, `mountain`, and `system`.
- This gives a third controlled input into `L24:3645`. `CONTROL-6` is a broader informal/mixed-prose path, `CONTROL-16` is a broad practical/informal context-to-content path, and `CONTROL-19` is a narrower content-word sibling. None of these should be treated as slop evidence.

### SCENE-2: everyday narrative comma/quote/punctuation scene path

I screened `L19:947 -> L24:1834` as a mixed human-heavy narrative-style edge. This one is real, controlled, and much more token-local than the broad controls, but it has essentially no detector-score movement.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_human_unknown17_L19_947_to_L24_1834.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_human_unknown17_L19_947_to_L24_1834.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_human_unknown17_L19_947_to_L24_1834.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_human_unknown17_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_human_unknown17_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `20` | `-156.66` | `-6.23` | `+0.005` |

Matched controls:

| Control family | Mean abs downstream mass delta | Max abs downstream mass delta |
|---|---:|---:|
| different upstream, same downstream | `0.315` | `0.772` |
| same upstream, different downstream | `0.673` | `1.272` |

Token-locus summary:

| Metric | `L19:947` | `L24:1834` |
|---|---:|---:|
| Mean active tokens/doc | `77.7` | `66.2` |
| Mean mass/doc | `135.14` | `224.71` |
| Mean max activation | `5.54` | `10.29` |
| Dominant token classes | punctuation / word / sentence boundary | word / punctuation / sentence boundary |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `13/20` (`65.0%`) |
| Mean top-token Jaccard | `0.596` |

Interpretation:

- The source mix is mostly human (`13/20`) with GPT-5.5, Gemini, and GLM variants.
- The top loci are commas, periods, quote-punctuation, and ordinary narrative scaffold words such as `and`, `at`, `when`, `to`, `was`, and `for`.
- This sits between `PUNCT-1` and `SCENE-1`: it is less sparse than pure quote punctuation, but more token-local and peak-aligned than broad scene rhythm. It is a narrative format/style circuit, not a slop-score path.

### WEAK-EDGE-17: weak mixed verse/unknown edge

I screened `L19:2284 -> L24:4095` as a mixed verse-like nomination.

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `16` | `-2.06` | `-0.052` | `+0.005` |

Interpretation:

- The downstream movement is too small to justify controls or token-locus work.
- This should stay screened out unless a different upstream produces a much larger first-pass effect for `L24:4095`.

### WEAK-EDGE-18: weak alternate into mixed verse/unknown downstream

I screened `L19:1218 -> L24:4095` because it was the next unregistered high-ranked direct edge into the same weak `L24:4095` family.

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `19` | `-0.14` | `+0.001` | `+0.004` |

Interpretation:

- This is weaker than `WEAK-EDGE-17` and has no meaningful downstream peak movement.
- `L24:4095` should remain deprioritized until a first-pass ablation finds a much stronger upstream.

### EXPO-3: sparse factual/cultural sentence-boundary and quote-period path

I screened `L19:2280 -> L24:3080` as a mixed-source factual/cultural edge. It is low-mass but controlled, sparse, and highly peak-aligned.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown19_L19_2280_to_L24_3080.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown19_L19_2280_to_L24_3080.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown19_L19_2280_to_L24_3080.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown19_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown19_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `10` | `-6.26` | `-1.81` | `-0.041` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `0.257` | `0.029` |
| same upstream, different downstream | `0.671` | `0.149` |

Token-locus summary:

| Metric | `L19:2280` | `L24:3080` |
|---|---:|---:|
| Mean active tokens/doc | `5.0` | `14.2` |
| Mean mass/doc | `10.68` | `36.99` |
| Mean max activation | `5.96` | `8.44` |
| Dominant token classes | sentence boundary / punctuation | word / sentence boundary / punctuation |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `9/10` (`90.0%`) |
| Mean top-token Jaccard | `0.336` |

Interpretation:

- The examples are mixed Qwen/GPT/human/GLM factual or cultural explanations: art analysis, film summaries, AI ethics, hotel/film trivia, mineral discovery, local recommendations, and pop-culture continuation.
- `L19:2280` is almost a pure period/quote-period trigger. `L24:3080` keeps those boundary tokens but adds local factual/content words around them.
- The path is real but low-mass and detector-human-side. It belongs with local boundary controls, not slop markers.

### CTA-1: promotional call-to-action ending into punctuation/paragraph-boundary path

I screened `L19:1054 -> L24:1472` as a model-heavy edge. It is controlled and interpretable, but its score movement is negative, so it is not positive AI-slop evidence.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown20_L19_1054_to_L24_1472.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown20_L19_1054_to_L24_1472.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown20_L19_1054_to_L24_1472.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown20_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown20_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `15` | `-18.63` | `-0.90` | `-0.192` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `1.809` | `0.275` |
| same upstream, different downstream | `1.237` | `0.100` |

Token-locus summary:

| Metric | `L19:1054` | `L24:1472` |
|---|---:|---:|
| Mean active tokens/doc | `43.7` | `41.2` |
| Mean mass/doc | `60.87` | `129.11` |
| Mean max activation | `4.18` | `8.91` |
| Dominant token classes | word / sentence boundary / line break | word / sentence boundary / punctuation / line break |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `0/15` (`0.0%`) |
| Mean top-token Jaccard | `0.117` |

Interpretation:

- The source mix is mostly model output, but with a detector-human-side score movement.
- `L19:1054` fires on promotional/call-to-action words near endings: `Visit`, `Contact`, `today`, `now`, `Enroll`, `Get`, `tour`, `schedule`, and `discount`.
- `L24:1472` fires more on periods, exclamation marks, paragraph breaks, commas, and ending punctuation around those same closing passages.
- This is best treated as a context-to-boundary promotional-ending path. It is useful for separating promotional formatting from positive slop evidence.

### CONTROL-20: broad conversational/copular prose into sentence-boundary terminal

I screened `L19:3074 -> L24:1278` as the next mixed-source direct edge. It is controlled, but the active loci show a broad upstream feeding a compact boundary downstream, with negative score movement.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown21_L19_3074_to_L24_1278.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown21_L19_3074_to_L24_1278.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown21_L19_3074_to_L24_1278.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown21_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown21_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `13` | `-47.10` | `-1.04` | `-0.163` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `1.29` | `0.013` |
| same upstream, different downstream | `13.47` | `0.139` |

Token-locus summary:

| Metric | `L19:3074` | `L24:1278` |
|---|---:|---:|
| Mean active tokens/doc | `354.8` | `64.8` |
| Mean mass/doc | `671.35` | `253.28` |
| Mean max activation | `4.09` | `10.18` |
| Dominant token classes | word / punctuation / function words | sentence boundary / line break / punctuation |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `0/13` (`0.0%`) |
| Mean top-token Jaccard | `0.007` |

Interpretation:

- The examples are conversational or comedic/review rewrites about fashion rules, old internet tech, Mars tourism, occult biography, Facebook ads, mosquitoes, Saturn, and missing pets.
- The upstream is broad over copular/pronoun/scaffold words such as `is`, `are`, `they`, `was`, `and`, and `the`.
- The downstream is a compact punctuation terminal, mostly periods, paragraph-final periods, question marks, and exclamation marks.
- This is a context-to-boundary control path, not a local same-token boundary circuit and not slop evidence.

### CONTROL-21: human-heavy broad informational/list prose direction

I screened `L19:280 -> L24:185` because it was a high-ranked human-heavy direct edge. It is extremely path-specific, and unusually it moves the detector score in the AI-positive direction despite being mostly human examples.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_human_unknown22_L19_280_to_L24_185.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_human_unknown22_L19_280_to_L24_185.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_unknown22_L19_280_to_L24_185.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_unknown22_summary.csv`
- Token loci: `docs/experiments/pangram_sae_human_unknown22_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `11` | `-263.34` | `-1.90` | `+0.067` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `6.76` | `0.028` |
| same upstream, different downstream | `1.41` | `0.061` |

Token-locus summary:

| Metric | `L19:280` | `L24:185` |
|---|---:|---:|
| Mean active tokens/doc | `181.0` | `179.7` |
| Mean mass/doc | `255.19` | `381.78` |
| Mean max activation | `2.67` | `4.17` |
| Dominant token classes | word / punctuation / function words | word / punctuation / function words |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `4/11` (`36.4%`) |
| Mean top-token Jaccard | `0.337` |

Interpretation:

- The target set is 10 human rows and 1 GLM row: timelines/history, TV show summaries, workday advice, comic/absurd pseudo-scripture, a butterfly story, geopolitics/grain factors, book recommendations, materials explanation, and a business letter.
- Both nodes are broad word/function-word features, with top tokens such as `and`, `is`, `can`, `that`, `to`, `out`, `have`, `book`, `material`, and `Internet`.
- This is a useful caution: the detector can assign positive AI-score support to a mostly human broad prose direction. Source mix and token loci must stay part of the interpretation, not only score sign.

### ARA-LEX-1: `ara`/proper-name and miscellaneous subword path

I screened `L19:176 -> L24:2360` because it was a GPT/mixed direct edge with strong first-pass effects. Token loci show it is a clean lexical/subword path.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown23_L19_176_to_L24_2360.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown23_L19_176_to_L24_2360.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown23_L19_176_to_L24_2360.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown23_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown23_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `17` | `-49.47` | `-3.45` | `+0.010` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `0.31` | `0.004` |
| same upstream, different downstream | `0.69` | `0.102` |

Token-locus summary:

| Metric | `L19:176` | `L24:2360` |
|---|---:|---:|
| Mean active tokens/doc | `32.5` | `27.4` |
| Mean mass/doc | `46.66` | `51.15` |
| Mean max activation | `3.72` | `4.19` |
| Dominant token classes | word pieces | word pieces |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `16/17` (`94.1%`) |
| Mean top-token Jaccard | `0.743` |

Interpretation:

- The examples span narrative, Python/package discussion, tactical-game advice, fantasy, travel, screenplays, and practical advice.
- The actual locus is lexical/subword identity, not those surface genres. Top tokens include `ara`, `Mara`, `Sara`, `Jenna`, `tactical`, `bike`, `flick`, `MIME`, and similar word pieces.
- This should be filtered like the existing lexical/subword controls before making semantic style claims.

### WEAK-EDGE-19: weak alternate downstream from VERSE-1 lineation upstream

I screened `L19:2355 -> L24:1661` because `L19:2355` is the confirmed `VERSE-1` lineation upstream and `L24:1661` looked like an alternate line/paragraph downstream.

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-0.46` | `-0.017` | `+0.045` |

Interpretation:

- The score movement follows the known upstream, but the nominated downstream barely moves.
- This is not the matching downstream for `L19:2355`; no controls or loci are needed.

### PARA-1: sparse paragraph/line-break boundary path

I screened `L19:2284 -> L24:1661` as a second possible upstream into `L24:1661`. Unlike `L19:2355`, this edge is controlled and token-local.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_verse_unknown24_L19_2284_to_L24_1661.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_verse_unknown24_L19_2284_to_L24_1661.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_verse_unknown24_L19_2284_to_L24_1661.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_verse_unknown24_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_verse_unknown24_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-7.40` | `-4.48` | `+0.001` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `0.046` | `0.008` |
| same upstream, different downstream | `3.059` | `0.125` |

Token-locus summary:

| Metric | `L19:2284` | `L24:1661` |
|---|---:|---:|
| Mean active tokens/doc | `3.0` | `5.8` |
| Mean mass/doc | `11.00` | `23.31` |
| Mean max activation | `7.12` | `10.72` |
| Dominant token classes | line break | line break / sentence boundary |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `14/14` (`100.0%`) |
| Mean top-token Jaccard | `0.606` |

Interpretation:

- The active loci are almost entirely paragraph or line-boundary tokens: `.\n\n`, `\n\n`, `\n`, `.\n`, and occasional punctuation-newline variants.
- This is a clean paragraph/line-break format circuit with essentially no detector-score movement.
- It should be separated from `VERSE-1`: both involve line breaks, but `L19:2355` does not feed `L24:1661`, while `L19:2284` does.

### WEAK-EDGE-20: weak alternate downstream from VERSE-1 lineation upstream

I screened `L19:2355 -> L24:4095` as another possible downstream for the confirmed `VERSE-1` upstream.

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `19` | `-0.96` | `-0.016` | `+0.053` |

Interpretation:

- Like `WEAK-EDGE-19`, this mostly shows upstream score movement without meaningful downstream collapse.
- `L24:4095` remains deprioritized after several weak nominations.

### CONTROL-22: broad human/GLM ordinary-word continuation direction

I screened `L19:1191 -> L24:3009` because it was the next high-ranked remaining direct edge. It is a very strong and controlled broad path, but the score direction is human/non-AI.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown25_L19_1191_to_L24_3009.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown25_L19_1191_to_L24_3009.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown25_L19_1191_to_L24_3009.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown25_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown25_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `9` | `-323.62` | `-3.21` | `-0.229` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `5.54` | `0.019` |
| same upstream, different downstream | `3.35` | `0.072` |

Token-locus summary:

| Metric | `L19:1191` | `L24:3009` |
|---|---:|---:|
| Mean active tokens/doc | `174.9` | `198.4` |
| Mean mass/doc | `209.25` | `396.83` |
| Mean max activation | `3.55` | `5.18` |
| Dominant token classes | word / punctuation / sentence boundary | word / punctuation / sentence boundary |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `4/9` (`44.4%`) |
| Mean top-token Jaccard | `0.424` |

Interpretation:

- The target set is human/GLM. The path is broad over ordinary words and punctuation, with top tokens around periods, commas, `This`, `math`, `skills`, `engineers`, `wolves`, `message`, and `error`.
- This is a high-mass broad control path, not a local style circuit.
- The `3009` downstream here is layer 24 and should not be confused with the previously studied layer-19 `L19:3009` boundary upstream.

### CONTROL-23: sparse scaffold/boundary upstream into broad modal/function-word terminal

I screened `L19:1437 -> L24:1664` because it was a high-ranked edge into the known broad `L24:1664` terminal. The path is strong, but the downstream remains broad and somewhat entangled.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_gpt_unknown26_L19_1437_to_L24_1664.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_gpt_unknown26_L19_1437_to_L24_1664.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_gpt_unknown26_L19_1437_to_L24_1664.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_gpt_unknown26_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_gpt_unknown26_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `12` | `-132.78` | `-0.66` | `+0.174` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `35.50` | `0.083` |
| same upstream, different downstream | `1.14` | `0.109` |

Token-locus summary:

| Metric | `L19:1437` | `L24:1664` |
|---|---:|---:|
| Mean active tokens/doc | `16.3` | `432.8` |
| Mean mass/doc | `34.89` | `3158.20` |
| Mean max activation | `7.52` | `14.90` |
| Dominant token classes | word / line break / sentence boundary | word / punctuation / function words |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `0/12` (`0.0%`) |
| Mean top-token Jaccard | `0.011` |

Interpretation:

- The target set is mostly GPT-5.5 plus a few human/GLM rows.
- `L19:1437` is sparse over periods, paragraph breaks, and scaffold tokens such as `according`, `don`, `such`, and `However`.
- `L24:1664` is very broad over modal/function words such as `is`, `can`, `to`, `a`, `are`, and `be`.
- The target path is stronger than controls, but `L19:913 -> L24:1664` also moves the downstream by `-35.50` mass. This supports treating `L24:1664` as a broad entangled terminal with multiple inputs rather than a compact local feature.

### VERSE-2B: rhymed/poetic end-word sibling path

I screened `L19:996 -> L24:457` because `L19:996` is the confirmed upstream for the `VERSE-2` rhymed end-word path. This is another controlled sibling downstream in the same poetic/rhyme family.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown27_L19_996_to_L24_457.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown27_L19_996_to_L24_457.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown27_L19_996_to_L24_457.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown27_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown27_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `15` | `-40.44` | `-0.14` | `-0.088` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `0.78` | `0.017` |
| same upstream, different downstream | `0.26` | `0.016` |

Token-locus summary:

| Metric | `L19:996` | `L24:457` |
|---|---:|---:|
| Mean active tokens/doc | `24.0` | `45.4` |
| Mean mass/doc | `109.51` | `398.72` |
| Mean max activation | `10.01` | `18.18` |
| Dominant token classes | word | word |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `0/15` (`0.0%`) |
| Mean top-token Jaccard | `0.273` |

Interpretation:

- The active tokens are poetic/rhyme words rather than line breaks. Upstream examples include `light`, `land`, `fate`, `space`, `stars`, `night`, and `delight`; downstream examples include `bright`, `low`, `high`, `day`, `night`, `breeze`, `embrace`, and `frame`.
- This is a sibling to `VERSE-2`, not a lineation circuit.
- Score movement is negative, so this should not be counted as positive AI-slop evidence.

### I-LEX-1: long-vowel / `i`-bearing word-piece lexical path

I screened `L19:1984 -> L24:1995` because it appeared as another high-overlap mixed-source direct edge. The intervention and loci show a clean lexical/subword path, not a semantic style circuit.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown28_L19_1984_to_L24_1995.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown28_L19_1984_to_L24_1995.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown28_L19_1984_to_L24_1995.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown28_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown28_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `9` | `-70.53` | `-4.51` | `+0.0005` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `0.48` | `0.035` |
| same upstream, different downstream | `0.65` | `0.009` |

Token-locus summary:

| Metric | `L19:1984` | `L24:1995` |
|---|---:|---:|
| Mean active tokens/doc | `30.0` | `29.0` |
| Mean mass/doc | `48.56` | `75.93` |
| Mean max activation | `4.45` | `6.25` |
| Dominant token classes | word | word |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `7/9` (`77.8%`) |
| Mean top-token Jaccard | `0.725` |

Interpretation:

- The target set spans Gemini, GPT, human, GLM, and Qwen examples, but the active loci are mostly word pieces such as `slice`, `line`, `ice`, `pace`, `race`, `course`, `resource`, `grease`, `cheese`, and `white`.
- Controls are tiny relative to the target edge, so this is a real path-specific dependency.
- The near-zero score movement and lexical loci make this a lexical/subword control, similar to `STEM-1`, `S-LEX-1`, `M-LEX-1`, `E-LEX-1`, and `ARA-LEX-1`.

### CONTROL-24: human sentence-boundary period accumulator

I screened `L19:1441 -> L24:741` because `L24:741` was a high-ranked layer-24 latent with large score impact. The path is strong and controlled, but the evidence is all-human sentence-boundary punctuation.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_human_unknown30_L19_1441_to_L24_741.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_human_unknown30_L19_1441_to_L24_741.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_unknown30_L19_1441_to_L24_741.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_human_unknown30_summary.csv`
- Token loci: `docs/experiments/pangram_sae_human_unknown30_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `8` | `-256.55` | `-14.99` | `+0.775` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `10.09` | `0.497` |
| same upstream, different downstream | `33.57` | `0.755` |

Token-locus summary:

| Metric | `L19:1441` | `L24:741` |
|---|---:|---:|
| Mean active tokens/doc | `19.0` | `18.8` |
| Mean mass/doc | `135.09` | `259.72` |
| Mean max activation | `8.60` | `16.57` |
| Dominant token classes | sentence boundary | sentence boundary |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `2/8` (`25.0%`) |
| Mean top-token Jaccard | `0.651` |

Interpretation:

- All eight evaluated documents are human.
- Both nodes are almost entirely period/bullet/question-mark loci: the top detail tokens are `.` plus a few `•` and `?` tokens.
- The large positive class-3 movement is therefore a score-sign caution. It shows that detector-score movement by itself is not enough to call a path AI-slop evidence; source mix and token loci are decisive here.

### INVERSE-EDGE-1: inverse effect into `L24:2940`

I screened `L19:3529 -> L24:2940` because it shared the high-mass ornate-prose downstream already studied in `CONTROL-15`. It does not behave like a feed-forward edge under the current upstream-ablation criterion.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_human_mixed_unknown29_L19_3529_to_L24_2940.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_human_mixed_unknown29_L19_3529_to_L24_2940.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_human_mixed_unknown29_L19_3529_to_L24_2940.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Mean abs downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|---:|
| `14` | `+39.35` | `+0.009` | `0.053` | `+0.049` |

Interpretation:

- Ablating `L19:3529` increases rather than suppresses `L24:2940` mass.
- This is the opposite sign from the normal edge criterion and should be tracked separately from confirmed causal feeds.
- If revisited, the right hypothesis is competition/disinhibition around the `CONTROL-15` downstream, not a standard upstream-to-downstream style circuit.

### OUTLINE-3: paragraph-boundary and enumerated-transition circuit

I screened `L19:21 -> L24:2537` because it was the next unscreened direct edge after already-known candidates. The target examples initially look like Qwen/GLM polished marketing and practical-advice prose, but token loci show a sparse paragraph/list-transition feature.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown31_L19_21_to_L24_2537.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown31_L19_21_to_L24_2537.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown31_L19_21_to_L24_2537.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown31_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown31_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `13` | `-10.54` | `-6.14` | `+0.017` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `0.068` | `0.040` |
| same upstream, different downstream | `0.625` | `0.105` |

Token-locus summary:

| Metric | `L19:21` | `L24:2537` |
|---|---:|---:|
| Mean active tokens/doc | `5.4` | `7.8` |
| Mean mass/doc | `11.69` | `32.77` |
| Mean max activation | `7.82` | `11.31` |
| Dominant token classes | line break / word | line break / sentence boundary / word |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `13/13` (`100.0%`) |
| Mean top-token Jaccard | `0.461` |

Interpretation:

- The active tokens are dominated by `.\n\n`, with smaller support from `First`, `Second`, `Next`, and `such`.
- Matched controls are tiny, so the pair is path-specific.
- The right label is a paragraph/list-transition format circuit in polished practical prose, not generic marketing semantics.

### WEAK-EDGE-21: weak alternate into `L24:2278`

I screened `L19:3894 -> L24:2278` because it is a verse-family sibling candidate. It does not have enough downstream peak movement to promote beside the confirmed verse circuits.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown32_L19_3894_to_L24_2278.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown32_L19_3894_to_L24_2278.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown32_L19_3894_to_L24_2278.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Mean abs downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|---:|
| `16` | `-7.22` | `-0.061` | `0.066` | `+0.128` |

Interpretation:

- The target examples are verse/poetic outputs, but the nominated downstream barely moves at the peak level.
- The class-3 movement likely follows `L19:3894` itself, which remains anchored to the confirmed `VERSE-4` downstream `L24:953`.
- No controls or token loci are warranted unless a future first-pass ablation produces a much larger `L24:2278` effect.

### STRUCT-1: mixed structured-code/list/verse boundary-token path

I screened `L19:2040 -> L24:2324` because it had a mixed target set with code, formatted lists, verse, and human prose. The path is controlled and real, but its loci are structural tokens rather than prose semantics.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown33_L19_2040_to_L24_2324.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown33_L19_2040_to_L24_2324.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown33_L19_2040_to_L24_2324.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown33_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown33_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-41.08` | `-3.95` | `-0.013` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `0.816` | `0.037` |
| same upstream, different downstream | `0.564` | `0.009` |

Token-locus summary:

| Metric | `L19:2040` | `L24:2324` |
|---|---:|---:|
| Mean active tokens/doc | `46.1` | `36.3` |
| Mean mass/doc | `63.37` | `94.60` |
| Mean max activation | `4.46` | `8.93` |
| Dominant token classes | word / line break / punctuation | word / line break / punctuation |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `10/14` (`71.4%`) |
| Mean top-token Jaccard | `0.503` |

Interpretation:

- Top tokens include newlines, `}\n`, `)\n`, `;\n`, `.id`, numeric labels, `plan`, `Blue`/`blue`, section labels, and code-return fragments.
- The controlled downstream collapse is real, but the feature is structural/code/list/verse formatting.
- Score movement is slightly negative, so this is a format-control path rather than detector-positive slop evidence.

### WEAK-EDGE-22: weak alternate into `L24:4095`

I screened `L19:33 -> L24:4095` because `L24:4095` has several verse-family nominations. This is another weak result for that downstream.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown34_L19_33_to_L24_4095.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown34_L19_33_to_L24_4095.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown34_L19_33_to_L24_4095.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `15` | `-4.50` | `-0.120` | `+0.025` |

Interpretation:

- This is not enough downstream movement to promote the edge.
- `L24:4095` has now failed multiple upstream nominations, so it should stay out of the near-term confirmed-circuit queue.

### WEAK-EDGE-23: lineation upstream does not feed broad lyrical terminal

I screened `L19:2355 -> L24:1478` to test whether the confirmed `VERSE-1` lineation upstream also feeds the broad lyrical `VERSE-3` downstream. It does not.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown35_L19_2355_to_L24_1478.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown35_L19_2355_to_L24_1478.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown35_L19_2355_to_L24_1478.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Mean abs downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|---:|
| `15` | `+0.47` | `-0.001` | `0.002` | `-0.121` |

Interpretation:

- The downstream does not collapse; mass slightly increases and max movement is effectively zero.
- The score movement is upstream-owned and should not be attributed to `L24:1478`.
- `L24:1478` remains better explained by the confirmed `VERSE-3` path from `L19:2130`.

### LEX-1B: discourse-marker upstream into broad practical-topic word terminal

I screened `L19:3450 -> L24:2898` because it was a high-overlap sibling of the known discourse-marker upstream. It is controlled and real, but the loci split between sparse discourse/boundary tokens upstream and broad practical-topic words downstream.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown36_L19_3450_to_L24_2898.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown36_L19_3450_to_L24_2898.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown36_L19_3450_to_L24_2898.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown36_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown36_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `12` | `-89.93` | `-1.13` | `+0.039` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `2.88` | `0.016` |
| same upstream, different downstream | `1.56` | `0.125` |

Token-locus summary:

| Metric | `L19:3450` | `L24:2898` |
|---|---:|---:|
| Mean active tokens/doc | `11.1` | `127.7` |
| Mean mass/doc | `35.66` | `415.80` |
| Mean max activation | `10.19` | `6.91` |
| Dominant token classes | word / discourse marker / sentence boundary | word |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `0/12` (`0.0%`) |
| Mean top-token Jaccard | `0.004` |

Interpretation:

- `L19:3450` fires on periods and discourse/scaffold words: `such`, `Additionally`, `However`, `Furthermore`, `Second`, and `Finally`.
- `L24:2898` is broad and word-only in the top loci, with tokens such as `rates`, `options`, `emissions`, `transportation`, `needs`, `projects`, `events`, and `culture`.
- This should be treated as a sibling context-to-content mediation path from the `LEX-1` upstream, not as a local same-token discourse-marker circuit.

### SCENE-3: mixed narrative/sensory sentence-boundary path

I screened `L19:3664 -> L24:3803` because it was a high-ranked mixed-source narrative edge. It is controlled and real, and token loci show a sentence-boundary path rather than broad narrative word mass.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown37_L19_3664_to_L24_3803.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown37_L19_3664_to_L24_3803.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown37_L19_3664_to_L24_3803.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown37_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown37_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `18` | `-44.81` | `-3.30` | `+0.067` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `0.139` | `0.006` |
| same upstream, different downstream | `16.93` | `0.721` |

Token-locus summary:

| Metric | `L19:3664` | `L24:3803` |
|---|---:|---:|
| Mean active tokens/doc | `47.2` | `20.7` |
| Mean mass/doc | `74.85` | `86.55` |
| Mean max activation | `4.84` | `7.73` |
| Dominant token classes | sentence boundary / punctuation / word | sentence boundary / punctuation |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `9/18` (`50.0%`) |
| Mean top-token Jaccard | `0.596` |

Interpretation:

- The target examples are narrative/sensory scenes across all sources, mostly Qwen with human/GPT/GLM examples.
- Top loci are overwhelmingly periods, with smaller support from question/exclamation punctuation and contractions.
- This is a sentence-boundary circuit in narrative prose, not a broad story-noun or sensory-adjective path.

### WEAK-EDGE-24: weak verse-family alternate into `L24:43`

I screened `L19:33 -> L24:43` because it looked like another verse-family same-upstream candidate. It does not have meaningful downstream movement.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown38_L19_33_to_L24_43.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown38_L19_33_to_L24_43.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown38_L19_33_to_L24_43.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Mean abs downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|---:|
| `17` | `+0.42` | `-0.009` | `0.046` | `-0.003` |

Interpretation:

- The examples are verse-like and the downstream often has high terminal activation, but upstream ablation does not move it.
- This is another negative result showing that prompt/source overlap and surface genre are not enough.

### WEAK-EDGE-25: weak alternate downstream from `CODE-1` upstream

I screened `L19:731 -> L24:3165` because it shares the confirmed `CODE-1` upstream. This candidate fails the downstream-ablation screen.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown39_L19_731_to_L24_3165.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown39_L19_731_to_L24_3165.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown39_L19_731_to_L24_3165.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `23` | `-0.09` | `-0.003` | `-0.070` |

Interpretation:

- Despite code/explanation examples and score movement from `L19:731`, the nominated downstream barely moves.
- `L19:731` should remain anchored to the confirmed `CODE-1` downstream `L24:2110`; `L24:3165` is not a confirmed terminal for this upstream.

### WEAK-EDGE-26: paragraph/line-break upstream does not feed rhymed end-word terminal

I screened `L19:2284 -> L24:1568` to test whether the paragraph/line-break upstream could feed the confirmed rhymed end-word downstream. It does not.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown40_L19_2284_to_L24_1568.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown40_L19_2284_to_L24_1568.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown40_L19_2284_to_L24_1568.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Mean abs downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|---:|
| `16` | `+0.51` | `+0.011` | `0.036` | `+0.010` |

Interpretation:

- Downstream mass increases slightly and max movement is tiny.
- `L24:1568` remains anchored to `VERSE-2` upstream `L19:996`, not paragraph/line-break upstream `L19:2284`.

### BROAD-8: GPT-heavy lyrical/narrative comma-rich register path

I screened `L19:1066 -> L24:2225` as a sibling of the generated narrative/rhetorical upstream family. The path is strong, controlled, and detector-positive, but very broad.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown41_L19_1066_to_L24_2225.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown41_L19_1066_to_L24_2225.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown41_L19_1066_to_L24_2225.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown41_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown41_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-445.83` | `-3.38` | `+0.169` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `0.87` | `0.008` |
| same upstream, different downstream | `2.96` | `0.202` |

Token-locus summary:

| Metric | `L19:1066` | `L24:2225` |
|---|---:|---:|
| Mean active tokens/doc | `227.1` | `226.9` |
| Mean mass/doc | `776.85` | `900.64` |
| Mean max activation | `6.42` | `8.86` |
| Dominant token classes | word / line break / punctuation | word / line break / punctuation |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `2/14` (`14.3%`) |
| Mean top-token Jaccard | `0.254` |

Interpretation:

- The target set is mostly GPT-5.5, with a small number of Gemini/GLM examples.
- Both nodes are broad, with top tokens including commas, line breaks, `the`, `a`, `sky`, `soft`, `quiet`, `bright`, `rain`, and `dark`.
- This is a strong generated lyrical/narrative register path and useful decomposition target, but it is too document-wide to be a compact slop marker.

### WEAK-EDGE-27: broad lyrical upstream does not feed rhymed end-word terminal

I screened `L19:2130 -> L24:1568` to test whether broad lyrical register feeds the confirmed rhymed end-word downstream. It does not.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown42_L19_2130_to_L24_1568.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown42_L19_2130_to_L24_1568.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown42_L19_2130_to_L24_1568.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Mean abs downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|---:|
| `15` | `+0.81` | `+0.003` | `0.038` | `-0.083` |

Interpretation:

- The downstream does not collapse; mass and max both move in the wrong or negligible direction.
- This separates broad lyrical register (`VERSE-3`) from the sparse rhymed end-word terminal (`VERSE-2`).

### COMMA-1: comma/content-word same-token mixed prose path

I screened `L19:1660 -> L24:3524` because it was a high-ranked mixed-source direct edge. It is a strong, controlled, same-token punctuation/content-word path with slightly human/non-AI score movement.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown43_L19_1660_to_L24_3524.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown43_L19_1660_to_L24_3524.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown43_L19_1660_to_L24_3524.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown43_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown43_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-111.91` | `-8.47` | `-0.016` |

Matched controls:

| Control family | Max mean abs downstream mass delta | Max mean abs downstream max delta |
|---|---:|---:|
| different upstream, same downstream | `0.514` | `0.015` |
| same upstream, different downstream | `0.535` | `0.006` |

Token-locus summary:

| Metric | `L19:1660` | `L24:3524` |
|---|---:|---:|
| Mean active tokens/doc | `58.1` | `60.7` |
| Mean mass/doc | `87.00` | `147.92` |
| Mean max activation | `6.01` | `11.67` |
| Dominant token classes | word / punctuation | word / punctuation |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `13/14` (`92.9%`) |
| Mean top-token Jaccard | `0.632` |

Interpretation:

- Top tokens are commas plus local content/function words: `either`, `head`, `loose`, `foolish`, `Whether`, `her`, `member`, and nearby scene/advice words.
- The path is mechanistically crisp and highly same-token, but it is not detector-positive.
- Keep it as a punctuation/content-word control and possible comparison point for `PUNCT-1` and `SCENE-2`.

### WEAK-EDGE-28: weak alternate downstream from discourse-marker upstream

I screened `L19:3450 -> L24:3160` as another sibling of the discourse-marker upstream. It fails the downstream movement screen.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown44_L19_3450_to_L24_3160.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown44_L19_3450_to_L24_3160.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown44_L19_3450_to_L24_3160.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-2.35` | `-0.015` | `+0.053` |

Interpretation:

- Score moves through `L19:3450`, but the nominated downstream barely moves.
- Keep `L19:3450` anchored to `LEX-1` and `LEX-1B`.

### WEAK-EDGE-29: weak correspondence-body to signoff cross-pair

I screened `L19:1496 -> L24:3666` to test whether the correspondence-body upstream also feeds the signoff/closing downstream. The effect is weak.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown45_L19_1496_to_L24_3666.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown45_L19_1496_to_L24_3666.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown45_L19_1496_to_L24_3666.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `16` | `-5.02` | `-0.588` | `+0.009` |

Interpretation:

- This is far smaller than confirmed correspondence/signoff circuits.
- Keep `L19:1496` anchored to `LETTER-2` and `L24:3666` anchored to `LETTER-1`.

### WEAK-EDGE-30: weak expressive/mixed downstream movement

I screened `L19:3539 -> L24:3867` as a mixed expressive/prose candidate. The downstream max movement is too small to promote.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown46_L19_3539_to_L24_3867.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown46_L19_3539_to_L24_3867.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown46_L19_3539_to_L24_3867.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `16` | `-10.87` | `-0.157` | `-0.131` |

Interpretation:

- The score movement is sizeable and negative, but the nominated downstream barely moves at the peak level.
- Screen out under the current edge criterion.

### WEAK-EDGE-31: paragraph/line-break upstream does not feed `VERSE-1`

I screened `L19:2284 -> L24:61` to test whether the paragraph/line-break upstream feeds the confirmed `VERSE-1` lineation downstream. It does not.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown47_L19_2284_to_L24_61.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown47_L19_2284_to_L24_61.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown47_L19_2284_to_L24_61.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Mean abs downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|---:|
| `14` | `+1.06` | `+0.004` | `0.054` | `+0.006` |

Interpretation:

- Downstream mass increases slightly and max movement is negligible.
- `L24:61` remains anchored to `VERSE-1` upstream `L19:2355`.

### WEAK-EDGE-32: weak alternate into `L24:4095`

I screened `L19:1325 -> L24:4095` as another attempt to find a real upstream for the repeatedly nominated `L24:4095` downstream. It remains weak.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown48_L19_1325_to_L24_4095.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown48_L19_1325_to_L24_4095.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown48_L19_1325_to_L24_4095.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `11` | `-5.37` | `-0.203` | `-0.012` |

Interpretation:

- This is another weak nomination for `L24:4095`.
- Keep `L24:4095` deprioritized until a first-pass ablation shows order-of-magnitude larger movement.

### WEAK-EDGE-33: weak alternate from the outline/list-transition upstream

I screened `L19:21 -> L24:2470` as a sibling downstream for the confirmed `OUTLINE-3` upstream. The downstream movement is too small to promote.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown49_L19_21_to_L24_2470.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown49_L19_21_to_L24_2470.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown49_L19_21_to_L24_2470.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `12` | `-1.14` | `-0.004` | `+0.023` |

Interpretation:

- This is far below the confirmed `OUTLINE-3` downstream movement.
- Keep `L19:21` anchored to `L24:2537`.

### WEAK-EDGE-34: weak code-family alternate

I screened `L19:731 -> L24:4024` as a sibling downstream for the confirmed `CODE-1` upstream. It fails the downstream movement screen.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown50_L19_731_to_L24_4024.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown50_L19_731_to_L24_4024.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown50_L19_731_to_L24_4024.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `+0.16` | `+0.008` | `-0.084` |

Interpretation:

- Downstream mass increases slightly and peak movement is negligible.
- Keep `L19:731` anchored to the confirmed code/explanation downstream `L24:2110`.

### RESOLUTION-1: whimsical story-resolution prose into paragraph/sentence-boundary terminal

I promoted `L19:1670 -> L24:3072` after the control panel showed the target downstream movement was specific. The examples are whimsical story completions across human, Gemini, GPT, GLM, and Qwen rows: lonely old men and magical cats, enchanted bicycles, baristas, garden ornaments, unusual hedgehogs, toy-shop dolls, forest discoveries, caves of wonder, and pirates.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown51_L19_1670_to_L24_3072.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown51_L19_1670_to_L24_3072.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown51_L19_1670_to_L24_3072.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown51_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown51_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-40.89` | `-0.590` | `-0.111` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `3.20` | `0.030` |
| same upstream, different downstream | `2.95` | `0.071` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:1670` | `12.0` | word / punctuation / sentence boundary | comma, `.`, `He`, `and`, `She`, `he`, `was`, `aby` |
| `L24:3072` | `12.0` | sentence boundary / line break | `.`, `.\n\n`, comma, quote-period paragraph breaks, `The` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `0/14` (`0.0%`) |
| Mean top-token Jaccard | `0.108` |

Interpretation:

- This is a controlled context-to-boundary path, not a same-token punctuation path.
- `L19:1670` peaks late in story-resolution prose around pronoun-led closing actions and emotional/plot resolution beats.
- `L24:3072` is mostly paragraph and sentence boundary activation.
- The score movement is negative, so this should be treated as a human/non-AI-side whimsical-story control rather than positive AI-slop evidence.

### WEAK-EDGE-35: weak mixed/content downstream movement

I screened `L19:2572 -> L24:2898` as the next direct-edge nomination. It fails the first-pass downstream movement screen.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown52_L19_2572_to_L24_2898.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown52_L19_2572_to_L24_2898.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown52_L19_2572_to_L24_2898.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `23` | `-2.28` | `-0.015` | `-0.037` |

Interpretation:

- The nominated downstream barely moves.
- Screen out under the current edge criterion.

### CHILD-1B: broad simple-explanatory/function-word feed into child/simple boundary terminal

I promoted `L19:262 -> L24:2595` as a controlled sibling feed into the known `CHILD-1` terminal. The upstream `L19:262` was already known from the modal/simple-explanatory `L19:262 -> L24:1664` path; this run shows it can also feed the child/simple boundary terminal.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown53_L19_262_to_L24_2595.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown53_L19_262_to_L24_2595.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown53_L19_262_to_L24_2595.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown53_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown53_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `12` | `-92.58` | `-2.22` | `+0.639` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta | Note |
|---|---:|---:|---|
| different upstream, same downstream | `53.89` | `1.45` | `L19:1441 -> L24:2595` also moves this terminal |
| same upstream, different downstream | `7.12` | `0.399` | score movement follows `L19:262`, but downstream collapse is much smaller |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:262` | `12.0` | word / function word | `a`, `to`, `were`, `the`, `was`, `Kevin`, `had`, `It`, `they` |
| `L24:2595` | `12.0` | sentence boundary / line break | `.`, `.\n\n`, `!`, quote-exclamation paragraph break |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `0/12` (`0.0%`) |
| Mean top-token Jaccard | `0.000` |

Interpretation:

- This is strong enough to keep, but it is not a same-token boundary circuit.
- `L19:262` looks like broad simple/child explanatory prose and function-word context.
- `L24:2595` remains the child/simple sentence-boundary terminal previously identified in `CHILD-1`.
- The control panel indicates the terminal can be driven by more than one upstream, so `CHILD-1B` should be treated as a sibling/context feed rather than a replacement for `CHILD-1`.

### WEAK-EDGE-36: weak/inverse poetic sibling movement

I screened `L19:1218 -> L24:457` as a poetic/rhymed sibling nomination. It does not show a feed-forward downstream dependency.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown54_L19_1218_to_L24_457.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown54_L19_1218_to_L24_457.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown54_L19_1218_to_L24_457.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `+1.13` | `+0.054` | `-0.006` |

Interpretation:

- Downstream activation increases slightly, and the score barely moves.
- Screen out as a weak/inverse sibling rather than a causal feed-forward path.

### WEAK-EDGE-37: weak rhymed/end-word upstream sibling

I screened `L19:996 -> L24:3223` as another sibling downstream for the confirmed rhymed/end-word upstream. It fails the first-pass movement screen.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown55_L19_996_to_L24_3223.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown55_L19_996_to_L24_3223.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown55_L19_996_to_L24_3223.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `15` | `+0.06` | `+0.012` | `-0.042` |

Interpretation:

- Downstream activity slightly increases rather than collapsing.
- Keep `L19:996` anchored to `VERSE-2` and `VERSE-2B`; this nominated downstream is not a confirmed terminal.

### CONTROL-25: human-heavy list/recommendation/correspondence same-token path

I promoted `L19:2629 -> L24:3849` as a controlled human-heavy same-token path. The examples include human list/recommendation answers, freelancing advice, content-organization advice, mountain recommendations, and two Gemini correspondence/letter rows.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown56_L19_2629_to_L24_3849.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown56_L19_2629_to_L24_3849.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown56_L19_2629_to_L24_3849.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown56_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown56_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `9` | `-95.03` | `-7.27` | `+0.017` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `1.06` | `0.009` |
| same upstream, different downstream | `1.26` | `0.010` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:2629` | `11.7` | word / punctuation / sentence boundary | comma, `.`, `-`, `of`, `reading`, `for`, `use`, `on`, `Manager` |
| `L24:3849` | `11.7` | word / punctuation / sentence boundary | `.`, comma, `-`, `of`, `for`, `reading`, `on`, `to`, `Manager` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `9/9` (`100.0%`) |
| Mean top-token Jaccard | `0.708` |

Interpretation:

- This is a crisp same-token path with excellent control separation.
- The score movement is effectively neutral, and the source mix is human-heavy.
- Keep it as a positive control for clean non-slop causal edges in list/recommendation/correspondence prose.

### CONTROL-26: second-person practical/reflective word-function path

I promoted `L19:1556 -> L24:310` as a controlled mixed-source same-token word/function path. The examples include advice to a younger self, mistakes/self-improvement, Ben 10 explanation, bathroom refresh tips, diary/reflection prose, sustainable weight loss, gratitude, narrative reflection, and eco-friendly yard advice.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown57_L19_1556_to_L24_310.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown57_L19_1556_to_L24_310.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown57_L19_1556_to_L24_310.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown57_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown57_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `13` | `-41.02` | `-2.67` | `-0.031` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `0.81` | `0.017` |
| same upstream, different downstream | `0.47` | `0.012` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:1556` | `12.0` | word / style-function word | `to`, `you`, `the`, `that`, comma, `this`, `.`, `want`, `of`, `way` |
| `L24:310` | `12.0` | word / style-function word | `you`, `that`, `to`, `is`, `great`, `of`, `the`, `want`, `You`, `way` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `13/13` (`100.0%`) |
| Mean top-token Jaccard | `0.449` |

Interpretation:

- This is another controlled same-token path, but broader and more ordinary-word/function-word dominated than `CONTROL-25`.
- It is slightly detector-human-side and should be kept as a practical/reflection control rather than a compact slop marker.

### VOICE-1B: child/simple sentence-boundary feed into playful/dialogue punctuation terminal

I promoted `L19:755 -> L24:3186` as a controlled sibling feed from the `CHILD-1` upstream into the `VOICE-1` downstream. The examples are child/simple stories, parent-child dialogue, first-person child/dog narration, and classroom/pet stories across human, GLM, GPT, and Gemini rows.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown58_L19_755_to_L24_3186.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown58_L19_755_to_L24_3186.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown58_L19_755_to_L24_3186.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown58_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown58_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `17` | `-25.59` | `-0.595` | `+0.187` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `2.26` | `0.026` |
| same upstream, different downstream | `6.22` | `0.043` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:755` | `12.0` | sentence boundary / line break / punctuation | `.`, `.\n\n`, `:`, `!`, quote-open tokens |
| `L24:3186` | `11.6` | punctuation / sentence boundary | `.`, quote-open tokens, `:`, `!`, quotes |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `0/17` (`0.0%`) |
| Mean top-token Jaccard | `0.176` |

Interpretation:

- This is a controlled but weaker sibling feed into the known `VOICE-1` terminal.
- The upstream is mostly simple child-story sentence/paragraph boundaries, while the downstream is more dialogue/interjection punctuation.
- The zero same-peak overlap means this should be described as boundary-context-to-voice-punctuation mediation, not as a local same-token punctuation circuit.

### WEAK-EDGE-38: weak/inverse broad-poetic upstream into rhymed downstream

I screened `L19:1218 -> L24:1568` as a possible broad-poetic sibling into the confirmed `VERSE-2` rhymed/end-word downstream. It does not move the downstream in the expected direction.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown59_L19_1218_to_L24_1568.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown59_L19_1218_to_L24_1568.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown59_L19_1218_to_L24_1568.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `16` | `+0.10` | `+0.004` | `-0.008` |

Interpretation:

- Downstream activity slightly increases, and score movement is near zero.
- Keep `L24:1568` anchored to `VERSE-2` upstream `L19:996`.

### WEAK-EDGE-39: weak paragraph/line-break upstream into poetic end-word sibling

I screened `L19:2284 -> L24:457` as another paragraph/line-break to poetic-word sibling. The effect is weak.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown60_L19_2284_to_L24_457.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown60_L19_2284_to_L24_457.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown60_L19_2284_to_L24_457.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `16` | `-1.58` | `-0.122` | `+0.010` |

Interpretation:

- The downstream movement is far below confirmed verse/poetic paths.
- Screen out; paragraph/line-break upstreams do not automatically feed poetic word terminals.

### CONTROL-27: mixed practical/expository content-word same-token path

I promoted `L19:2255 -> L24:850` as a controlled, score-neutral same-token content-word path. The examples span color symbolism, solar-system outlines, course marketing, literary analysis, affirmations, marathon training, finance pros/cons, menus, work-boundary advice, and joke explanation.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown61_L19_2255_to_L24_850.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown61_L19_2255_to_L24_850.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown61_L19_2255_to_L24_850.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown61_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown61_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `13` | `-74.79` | `-3.90` | `+0.0005` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `0.76` | `0.010` |
| same upstream, different downstream | `0.49` | `0.078` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:2255` | `12.0` | word / punctuation | `week`, comma, `.`, `:`, `miles`, `giants`, `warmth`, `atmosphere`, `Thesis` |
| `L24:850` | `12.0` | word | `week`, `miles`, `warmth`, `atmosphere`, `giants`, `Giants`, `life`, `organizing`, `moment` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `9/13` (`69.2%`) |
| Mean top-token Jaccard | `0.521` |

Interpretation:

- This is a strong, controlled, mostly same-token content-word path.
- It has essentially no detector-score movement, so it is a positive control for causal SAE edges that are not slop features.

### WEAK-EDGE-40: weak lineation upstream into rhymed/end-word downstream on this export

I screened `L19:2355 -> L24:1568` as a cross-pair between the `VERSE-1` lineation upstream and the `VERSE-2` rhymed/end-word downstream. Despite direct-edge overlap, the intervention is weak on this target export.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown62_L19_2355_to_L24_1568.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown62_L19_2355_to_L24_1568.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown62_L19_2355_to_L24_1568.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-0.98` | `-0.033` | `+0.002` |

Interpretation:

- This is not a meaningful downstream collapse.
- Keep `L19:2355` anchored to `VERSE-1`; keep `L24:1568` anchored to `VERSE-2` upstream `L19:996`.

### WEAK-EDGE-41: weak outline/list cross-pair with tiny peak movement

I screened `L19:2107 -> L24:508` as an outline/list cross-pair. The mass movement is nonzero, but peak movement is too small for a confirmed edge.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown63_L19_2107_to_L24_508.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown63_L19_2107_to_L24_508.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown63_L19_2107_to_L24_508.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `9` | `-14.34` | `-0.016` | `+0.008` |

Interpretation:

- Peak movement is effectively absent despite the aggregate mass delta.
- Keep this screened out unless a later target set shows much stronger peak collapse.

### WEAK-EDGE-42: upstream-owned score movement without downstream edge

I screened `L19:3161 -> L24:2644` as the next direct-edge nomination. The upstream affects the detector score, but the nominated downstream does not collapse.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown64_L19_3161_to_L24_2644.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown64_L19_3161_to_L24_2644.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown64_L19_3161_to_L24_2644.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `+0.65` | `-0.002` | `-0.081` |

Interpretation:

- The score movement is upstream-owned.
- The nominated downstream is not a feed-forward dependency under ablation.

### BROAD-9: model-only broad ordinary-word/function-word register path

I promoted `L19:2007 -> L24:1561` as a controlled, model-only broad word/function-word path. It is one of the largest activation effects in this queue, but it moves the detector score negative, so it is not positive slop evidence.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown65_L19_2007_to_L24_1561.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown65_L19_2007_to_L24_1561.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown65_L19_2007_to_L24_1561.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown65_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown65_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `23` | `-403.66` | `-5.23` | `-0.036` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `2.10` | `0.010` |
| same upstream, different downstream | `1.10` | `0.069` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:2007` | `12.0` | word / punctuation / style-function word | `a`, `the`, comma, `and`, `is`, `for`, `your`, `to`, `with`, `are` |
| `L24:1561` | `12.0` | word / punctuation / style-function word | comma, `and`, `a`, `the`, `is`, `your`, `to`, `are`, `with`, `in` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `20/23` (`87.0%`) |
| Mean top-token Jaccard | `0.599` |

Interpretation:

- This is a strong and controlled same-token broad-register path in model outputs.
- The negative score movement makes it a control/caution: source mix alone is not enough to call something detector-positive slop.

### PUNCT-2: exclamation/dialogue punctuation circuit

I promoted `L19:133 -> L24:3638` as a compact exclamation/quote punctuation circuit. It is mostly human examples plus GLM/Gemini rows, and it fires on exclamation-heavy dialogue, chants, games, and instructions.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown66_L19_133_to_L24_3638.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown66_L19_133_to_L24_3638.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown66_L19_133_to_L24_3638.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown66_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown66_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `13` | `-27.54` | `-12.38` | `-0.075` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `0.14` | `0.037` |
| same upstream, different downstream | `0.83` | `0.089` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:133` | `4.0` | sentence boundary / line break / punctuation | `!`, `!\n`, `!"`, `!\n\n`, `...\n`, `?` |
| `L24:3638` | `3.9` | sentence boundary / line break / punctuation | `!`, `!\n`, `."`, `!"`, `!\n\n`, `?` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `12/13` (`92.3%`) |
| Mean top-token Jaccard | `0.865` |

Interpretation:

- This is a clean punctuation circuit, sibling to `PUNCT-1` but focused on exclamation marks and exclamation-line endings.
- It is score-negative and human-heavy, so it is not positive AI-slop evidence.

### WEAK-EDGE-43: correspondence placeholder/signoff upstream weakly moves signoff terminal

I screened `L19:1049 -> L24:3666` as a cross-pair between the `LETTER-3` placeholder/signoff upstream and the `LETTER-1` signoff terminal. The upstream score movement is real, but the nominated downstream movement is weak.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown67_L19_1049_to_L24_3666.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown67_L19_1049_to_L24_3666.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown67_L19_1049_to_L24_3666.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `21` | `-0.87` | `-0.249` | `+0.265` |

Interpretation:

- The score movement follows `L19:1049`, but `L24:3666` barely moves.
- Keep `L19:1049` anchored to `LETTER-3` and `L24:3666` anchored to `LETTER-1`.

### WEAK-EDGE-44: lineation upstream does not feed rhymed/poetic sibling

I screened `L19:2355 -> L24:457` as another verse-family sibling. It does not show a downstream collapse.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown68_L19_2355_to_L24_457.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown68_L19_2355_to_L24_457.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown68_L19_2355_to_L24_457.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `21` | `+0.64` | `+0.004` | `-0.021` |

Interpretation:

- Downstream mass increases slightly and peak movement is negligible.
- Keep `L19:2355` anchored to `VERSE-1`; this is not a confirmed terminal.

### B-LEX-1: `b`/`bc` lexical-wordpiece path

I promoted `L19:654 -> L24:919` as a controlled lexical/subword path. The examples span botanical prose, fantasy/action text, binary-search/code explanations, history, food classification, fiction, and LaTeX/Pandas conversion, but the active token family is lexical rather than semantic.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/targets_mixed_unknown69_L19_654_to_L24_919.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_mixed_unknown69_L19_654_to_L24_919.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_mixed_unknown69_L19_654_to_L24_919.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_panel_mixed_unknown69_summary.csv`
- Token loci: `docs/experiments/pangram_sae_mixed_unknown69_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `11` | `-39.71` | `-6.78` | `-0.005` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `0.13` | `0.008` |
| same upstream, different downstream | `0.75` | `0.018` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:654` | `9.0` | word / style-function word | `binary`, `bc`, `battle`, `because`, `but`, `search`, `ect`, `brain`, `body`, `base` |
| `L24:919` | `9.0` | word / style-function word | `bc`, `binary`, `battle`, `because`, `ect`, `brain`, `body`, `search`, `<`, `base` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `10/11` (`90.9%`) |
| Mean top-token Jaccard | `0.838` |

Interpretation:

- This is a clean lexical/subword path, not a style circuit.
- It should be filtered alongside the other lexical controls before assigning semantic labels to direct edges.

### VERSE-5: model-generated poetic/rhymed word path

I promoted `L19:1325 -> L24:457` as a controlled generated-poetic word path. This resolves one of the repeated `L24:457` sibling ambiguities: lineation and broad-poetic upstreams did not feed this downstream, but `L19:1325` does.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_1325_to_L24_457.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_1325_to_L24_457.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_1325_to_L24_457.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_1325_to_L24_457.csv`
- Token loci: `docs/experiments/pangram_sae_direct_L19_1325_token_loci.md`, `docs/experiments/pangram_sae_direct_L24_457_for_L19_1325_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-192.09` | `-8.28` | `-0.002` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `1.70` | `0.085` |
| same upstream, different downstream | `1.43` | `0.199` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:1325` | `51.4` | word-heavy | `grace`, `night`, `light`, `away`, `sleep`, `day`, `sky`, `spark`, `bright`, `doorway` |
| `L24:457` | `41.1` | word-heavy | `grace`, `night`, `bright`, `light`, `away`, `day`, `sleep`, `way`, `sky`, `tune` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `1/14` (`7.1%`) |
| Mean top-token Jaccard | `0.668` |

Interpretation:

- This is a strong generated-poetic/rhymed-word circuit, not a line-break circuit.
- The effect is highly path-specific under controls, but the score movement is near zero, so it is not positive detector-slop evidence.

### BOUNDARY-9: model practical-answer sentence-boundary circuit

I promoted `L19:3539 -> L24:2470` as a controlled sentence-boundary circuit in model practical/recommendation prose. The path is local and mechanically specific, but the classifier movement is negative.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3539_to_L24_2470.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3539_to_L24_2470.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3539_to_L24_2470.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_3539_to_L24_2470.csv`
- Token loci: `docs/experiments/pangram_sae_direct_L19_3539_token_loci.md`, `docs/experiments/pangram_sae_direct_L24_2470_for_L19_3539_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `19` | `-35.31` | `-2.79` | `-0.099` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `0.43` | `0.012` |
| same upstream, different downstream | `3.62` | `0.046` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:3539` | `34.8` | sentence boundary / punctuation / line break | `.`, `,`, `.\n\n`, `:`, newline, `."`, `can`, `You`, `and` |
| `L24:2470` | `33.9` | sentence boundary / punctuation / line break | `.`, `:`, `.\n\n`, newline, `,`, `."`, `but` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `7/19` (`36.8%`) |
| Mean top-token Jaccard | `0.690` |

Interpretation:

- This is a clean local sentence-boundary circuit in model practical/recommendation text.
- It is score-negative, so it belongs with boundary/format controls rather than detector-positive AI-slop circuits.

### BOUNDARY-10: sparse period / short-sentence boundary circuit

I promoted `L19:612 -> L24:2765` as a controlled sparse period circuit. The source mix is not purely model-generated, and the score movement is small, but the local circuit evidence is strong.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_612_to_L24_2765.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_612_to_L24_2765.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_612_to_L24_2765.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_612_to_L24_2765.csv`
- Token loci: `docs/experiments/pangram_sae_direct_L19_612_token_loci.md`, `docs/experiments/pangram_sae_direct_L24_2765_for_L19_612_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `12` | `-59.12` | `-3.32` | `+0.025` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `0.83` | `0.020` |
| same upstream, different downstream | `1.28` | `0.025` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:612` | `40.5` | sentence boundary / line break / punctuation | `.`, `?`, `!`, `.\n`, `.\n\n`, `But` |
| `L24:2765` | `30.6` | sentence boundary / line break | `.`, `!`, `.\n\n`, `?`, `.”`, `.\n` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `9/12` (`75.0%`) |
| Mean top-token Jaccard | `0.662` |

Interpretation:

- This is a clean local sentence-boundary path, mostly periods in short narrative or practical continuations.
- The source mix and small score movement make it a boundary/control circuit, not a positive slop marker.

### WEAK-EDGE-45: weak alternate into broad lyrical-register downstream

I screened `L19:33 -> L24:1478`. Although the target set has a small positive score movement, the nominated downstream does not show meaningful peak collapse.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_33_to_L24_1478.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_33_to_L24_1478.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_33_to_L24_1478.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `16` | `-5.03` | `+0.003` | `+0.037` |

Interpretation:

- This is below the downstream-peak threshold for a confirmed edge.
- Keep `L24:1478` anchored to the stronger `VERSE-3` broad lyrical-register upstream.

### CONTROL-28: broad ornate / high-register word-and-comma direction

I promoted `L19:3451 -> L24:351` as a controlled broad high-register path. It is one of the largest direct-edge effects in the current queue, but token loci show a document-wide word/comma direction rather than a compact local phrase or boundary.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3451_to_L24_351.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3451_to_L24_351.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3451_to_L24_351.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_3451_to_L24_351.csv`
- Token loci: `docs/experiments/pangram_sae_direct_L19_3451_token_loci.md`, `docs/experiments/pangram_sae_direct_L24_351_for_L19_3451_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `16` | `-1400.42` | `-8.19` | `-0.178` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `2.00` | `0.006` |
| same upstream, different downstream | `4.46` | `0.313` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:3451` | `335.5` | word / punctuation | comma, `.`, `the`, `to`, `a`, `I`, `was`, `It`, `we`, `that` |
| `L24:351` | `359.4` | word / punctuation | comma, `the`, `a`, `to`, `an`, `.`, `is`, `my`, `this`, `its` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `9/16` (`56.2%`) |
| Mean top-token Jaccard | `0.493` |

Interpretation:

- This is a very high-mass broad-control path over ornate/high-register prose, not a compact slop-style feature.
- The negative score movement and mixed source mix make it a guardrail against interpreting high causal mass as AI evidence.

### WEAK-EDGE-46: weak alternate from outline/list-transition upstream

I screened `L19:21 -> L24:3226` as another alternate from the `OUTLINE-3` upstream family. The downstream effect is too small for a confirmed edge.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_21_to_L24_3226.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_21_to_L24_3226.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_21_to_L24_3226.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `16` | `-0.49` | `-0.094` | `+0.028` |

Interpretation:

- This is below the downstream-edge threshold.
- Keep `L19:21` anchored to `OUTLINE-3`; do not promote this downstream.

### BOUNDARY-11: Qwen/GLM factual-expository period/comma circuit

I promoted `L19:2120 -> L24:3750` as a controlled factual-expository boundary circuit. It is model-only in this target export, but the score movement is negative, so it is a model-format control rather than positive slop evidence.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_2120_to_L24_3750.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_2120_to_L24_3750.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_2120_to_L24_3750.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_2120_to_L24_3750.csv`
- Token loci: `docs/experiments/pangram_sae_direct_L19_2120_token_loci.md`, `docs/experiments/pangram_sae_direct_L24_3750_for_L19_2120_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `11` | `-51.08` | `-5.01` | `-0.372` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `1.14` | `0.109` |
| same upstream, different downstream | `3.60` | `0.203` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:2120` | `28.2` | sentence boundary / word / punctuation | `.`, comma, `.\n\n`, `Consequently`, `The`, `While` |
| `L24:3750` | `24.7` | sentence boundary / word / punctuation | `.`, comma, `.\n\n`, `These`, `While`, `States`, `Additionally` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `6/11` (`54.5%`) |
| Mean top-token Jaccard | `0.745` |

Interpretation:

- This is a compact factual-expository period/comma circuit in Qwen/GLM outputs.
- The detector score movement is negative, so it is a model-format control rather than a positive AI-slop marker.

### CONTROL-29: broad ordinary-word explanatory/content direction

I promoted `L19:1198 -> L24:1571` as a controlled broad ordinary-word path. It has strong downstream collapse, but its loci are spread over ordinary content and function words, and it barely moves the detector score.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_1198_to_L24_1571.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_1198_to_L24_1571.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_1198_to_L24_1571.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_1198_to_L24_1571.csv`
- Token loci: `docs/experiments/pangram_sae_direct_L19_1198_token_loci.md`, `docs/experiments/pangram_sae_direct_L24_1571_for_L19_1198_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-244.40` | `-6.07` | `-0.005` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `4.49` | `0.005` |
| same upstream, different downstream | `0.24` | `0.069` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:1198` | `68.3` | word / punctuation / function word | `the`, `to`, comma, `is`, `a`, `.`, `that`, `material`, `has`, `and` |
| `L24:1571` | `111.6` | word / punctuation / function word | `the`, `a`, comma, `to`, `is`, `that`, `an`, `up`, `material`, `has` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `2/14` (`14.3%`) |
| Mean top-token Jaccard | `0.566` |

Interpretation:

- This is a broad same-family content/explanatory path, not a local boundary, lexical, or slop phrase circuit.
- It is useful as another broad-control guardrail because downstream collapse is strong while score movement is near zero.

### CONTROL-30: broad human advice/argument content direction

I promoted `L19:3001 -> L24:2603` as an all-human broad control. It has large path-specific downstream collapse, but loci show a context-to-content relationship rather than same-token local style.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3001_to_L24_2603.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3001_to_L24_2603.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3001_to_L24_2603.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_3001_to_L24_2603.csv`
- Token loci: `docs/experiments/pangram_sae_direct_L19_3001_token_loci.md`, `docs/experiments/pangram_sae_direct_L24_2603_for_L19_3001_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `11` | `-628.83` | `-2.86` | `-0.045` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `40.22` | `0.215` |
| same upstream, different downstream | `0.81` | `0.095` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:3001` | `300.6` | word / function word / boundary | `you`, `.`, comma, `essential`, `didn`, `don`, `way`, `it`, `college` |
| `L24:2603` | `397.3` | word / function word / boundary | `character`, `content`, `college`, `years`, `loans`, `thing`, `here`, `something` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `0/11` (`0.0%`) |
| Mean top-token Jaccard | `0.140` |

Interpretation:

- This is an all-human broad context-to-content path, not a local circuit.
- It is useful as a high-mass human-side control and as evidence that broad paths should be decomposed before semantic labeling.

### BOUNDARY-12: human/GLM narrative sentence and paragraph boundary circuit

I promoted `L19:3173 -> L24:3605` as a controlled narrative boundary circuit. It is mostly human, has negative score movement, and is highly local to sentence and paragraph boundaries.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3173_to_L24_3605.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3173_to_L24_3605.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3173_to_L24_3605.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_3173_to_L24_3605.csv`
- Token loci: `docs/experiments/pangram_sae_direct_L19_3173_token_loci.md`, `docs/experiments/pangram_sae_direct_L24_3605_for_L19_3173_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `19` | `-81.76` | `-4.13` | `-0.059` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `4.88` | `0.048` |
| same upstream, different downstream | `1.17` | `0.124` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:3173` | `48.4` | sentence boundary / word / punctuation / line break | `.`, `.\n\n`, `.\n`, `."\n\n`, `."`, `.”\n\n`, comma |
| `L24:3605` | `46.7` | sentence boundary / word / punctuation / line break | `.`, `.\n\n`, `.\n`, `."\n\n`, comma, `."`, `but` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `9/19` (`47.4%`) |
| Mean top-token Jaccard | `0.780` |

Interpretation:

- This is a clean human-side narrative sentence/paragraph-boundary circuit.
- It is a useful contrast against AI narrative boundary circuits because the surface form is similar but score direction and source mix differ.

### WEAK-EDGE-47: weak lineation upstream into alternate verse downstream

I screened `L19:2355 -> L24:2278` as another alternate downstream from the `VERSE-1` lineation upstream. The score movement follows the upstream, but the nominated downstream barely moves.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_2355_to_L24_2278.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_2355_to_L24_2278.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_2355_to_L24_2278.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-0.49` | `+0.003` | `-0.077` |

Interpretation:

- This is below the downstream-edge threshold.
- Keep `L19:2355` anchored to `VERSE-1`; do not promote this alternate downstream.

### CONTROL-14B: weaker human practical/list boundary sibling

I promoted `L19:46 -> L24:3285` as a controlled sibling of the existing human practical/list boundary path. It is real and specific, but lower-mass and less token-aligned than `CONTROL-14`.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_46_to_L24_3285.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_46_to_L24_3285.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_46_to_L24_3285.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_46_to_L24_3285.csv`
- Token loci: `docs/experiments/pangram_sae_direct_L19_46_for_L24_3285_token_loci.md`, `docs/experiments/pangram_sae_direct_L24_3285_for_L19_46_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `16` | `-17.77` | `-0.559` | `-0.100` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `0.48` | `0.006` |
| same upstream, different downstream | `1.05` | `0.115` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:46` | `60.0` | word / sentence boundary / punctuation | `.`, `:`, comma, ` -`, `.\n\n`, `.\n`, `but` |
| `L24:3285` | `32.9` | sentence boundary / word / punctuation / line break | `.`, comma, `:`, ` -`, `and`, `.\n`, `.\n\n`, `but` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `5/15` (`33.3%`) |
| Mean top-token Jaccard | `0.283` |

Interpretation:

- This is a controlled practical/list boundary sibling, but it is not the canonical `L19:46` downstream.
- Use it as secondary human practical/list boundary evidence; `CONTROL-14` remains the cleaner path.

### WEAK-EDGE-48: weak code-family alternate downstream

I screened `L19:1878 -> L24:3165` as a code-family alternate. The upstream carries score movement, but the nominated downstream does not meaningfully move.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_1878_to_L24_3165.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_1878_to_L24_3165.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_1878_to_L24_3165.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `24` | `-0.79` | `-0.061` | `-0.129` |

Interpretation:

- This is below the downstream-edge threshold.
- Keep `L19:1878` anchored to `CODE-2`; do not promote this alternate downstream.

### THE-LEX-1: `the` article lexical/function-word path

I promoted `L19:1988 -> L24:3585` as a controlled lexical/function-word path. It is one of the cleanest sanity checks in the direct-edge queue: the path is highly specific and causal, but the feature is simply the article `the`.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_1988_to_L24_3585.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_1988_to_L24_3585.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_1988_to_L24_3585.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_1988_to_L24_3585.csv`
- Token loci: `docs/experiments/pangram_sae_direct_L19_1988_token_loci.md`, `docs/experiments/pangram_sae_direct_L24_3585_for_L19_1988_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `10` | `-133.60` | `-9.29` | `+0.010` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `0.15` | `0.007` |
| same upstream, different downstream | `0.12` | `0.002` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:1988` | `17.1` | word | `the`, `The` |
| `L24:3585` | `17.7` | word | `the`, `a`, `The` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `7/10` (`70.0%`) |
| Mean top-token Jaccard | `0.874` |

Interpretation:

- This is a clean article/function-word lexical path.
- It should be filtered like the other lexical controls before assigning semantic style labels to direct-edge candidates.

### EXPO-4: factual/entity-history sentence-boundary circuit

I promoted `L19:3239 -> L24:3991` as a controlled factual/entity-history period circuit. The examples are mostly factual summaries about institutions, companies, books, libraries, musicians, and films.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3239_to_L24_3991.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3239_to_L24_3991.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3239_to_L24_3991.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_3239_to_L24_3991.csv`
- Token loci: `docs/experiments/pangram_sae_direct_L19_3239_token_loci.md`, `docs/experiments/pangram_sae_direct_L24_3991_for_L19_3239_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `11` | `-26.95` | `-5.20` | `-0.130` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `0.066` | `0.003` |
| same upstream, different downstream | `1.07` | `0.060` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:3239` | `17.0` | sentence boundary / word / punctuation | `.`, comma, `.\n\n`, `The`, `of`, `and`, `):` |
| `L24:3991` | `13.3` | sentence boundary / punctuation / word | `.`, `.\n\n`, comma, `and`, `of`, `):` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `10/11` (`90.9%`) |
| Mean top-token Jaccard | `0.681` |

Interpretation:

- This is a clean local factual/entity-history sentence-boundary circuit.
- Its score movement is negative, so it is another detector-human-side format control, not positive slop evidence.

### WEAK-EDGE-49: negation/contraction upstream does not feed BOUNDARY-1 terminal

I screened `L19:913 -> L24:3642` as a cross-family nomination from the negation/contraction upstream into the `BOUNDARY-1` terminal. The score movement is real but downstream activation moves in the wrong direction.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_913_to_L24_3642.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_913_to_L24_3642.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_913_to_L24_3642.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `16` | `+0.85` | `+0.044` | `+0.269` |

Interpretation:

- This is not a feed-forward edge into `L24:3642`.
- Keep `L19:913` anchored to the negation/contraction path; the score movement is upstream-owned here.

### BOUNDARY-13: Qwen narrative sentence-boundary circuit

I promoted `L19:3903 -> L24:3913` as a controlled Qwen-only narrative sentence-boundary circuit. It is local to periods and score-positive in this target export.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3903_to_L24_3913.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3903_to_L24_3913.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3903_to_L24_3913.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_3903_to_L24_3913.csv`
- Token loci: `docs/experiments/pangram_sae_direct_L19_3903_token_loci.md`, `docs/experiments/pangram_sae_direct_L24_3913_for_L19_3903_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `8` | `-84.24` | `-3.92` | `+0.260` |

Control summary:

| Control family | Largest abs mass delta | Largest abs max delta |
|---|---:|---:|
| different upstream, same downstream | `0.61` | `0.010` |
| same upstream, different downstream | `25.29` | `0.387` |

Token-locus summary:

| Node | Mean active tokens/doc | Dominant token classes | Top tokens |
|---|---:|---|---|
| `L19:3903` | `53.5` | sentence boundary | `.`, `.\n\n` |
| `L24:3913` | `49.0` | sentence boundary | `.` |

Cross-node overlap:

| Metric | Value |
|---|---:|
| Same peak token | `3/8` (`37.5%`) |
| Mean top-token Jaccard | `0.702` |

Interpretation:

- This is a score-positive Qwen narrative period circuit.
- It should be compared directly against the score-negative human/GLM narrative boundary `BOUNDARY-12` rather than collapsed into a generic period feature.

### WEAK-EDGE-50: practical-boundary upstream does not feed alternate downstream

I screened `L19:3539 -> L24:3818` as an alternate downstream from the `BOUNDARY-9` upstream. The downstream movement is inverse/tiny while score movement follows the upstream.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3539_to_L24_3818.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3539_to_L24_3818.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3539_to_L24_3818.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `18` | `+1.14` | `+0.010` | `-0.110` |

Interpretation:

- This is not a feed-forward downstream edge.
- Keep `L19:3539` anchored to `BOUNDARY-9`.

### WEAK-EDGE-51: weak verse/rhymed-end-word alternate

I screened `L19:33 -> L24:1568` as a verse/rhymed-end-word alternate. It is far weaker than the confirmed `VERSE-2` path into `L24:1568`.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_33_to_L24_1568.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_33_to_L24_1568.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_33_to_L24_1568.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `15` | `-5.45` | `-0.183` | `+0.018` |

Interpretation:

- This is below the downstream-edge threshold for the rhymed/end-word family.
- Keep `L24:1568` anchored to `VERSE-2`.

### WEAK-EDGE-52: correspondence body upstream does not feed LETTER-3 downstream

I screened `L19:1496 -> L24:2242` as a correspondence-family cross-pair. It does not move the `LETTER-3` downstream.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_1496_to_L24_2242.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_1496_to_L24_2242.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_1496_to_L24_2242.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `22` | `+0.91` | `+0.003` | `+0.005` |

Interpretation:

- This is not a feed-forward edge.
- Keep `L19:1496` anchored to `LETTER-2`; keep `L24:2242` anchored to `LETTER-3`.

### WEAK-EDGE-53: practical-boundary upstream inverse on practical/list sibling

I screened `L19:3539 -> L24:3285` as a practical-boundary sibling. The downstream moves in the wrong direction.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3539_to_L24_3285.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3539_to_L24_3285.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3539_to_L24_3285.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `11` | `+2.73` | `+0.059` | `-0.097` |

Interpretation:

- This is inverse/tiny downstream movement, not a confirmed edge.
- Keep `L19:3539` anchored to `BOUNDARY-9`.

### WEAK-EDGE-54: practical/craft upstream weakly moves practical/list sibling

I screened `L19:2905 -> L24:3285` as another practical/list sibling. The downstream peak barely moves.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_2905_to_L24_3285.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_2905_to_L24_3285.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_2905_to_L24_3285.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-1.42` | `-0.005` | `-0.108` |

Interpretation:

- This is below the downstream-edge threshold.
- Keep `L19:2905` anchored to its stronger practical/craft path; do not promote this downstream.

### WEAK-EDGE-55: dialogue-punctuation upstream weak/inverse on mixed scene downstream

I screened `L19:402 -> L24:2644` as a dialogue/scene-family sibling. The downstream does not collapse under upstream ablation.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_402_to_L24_2644.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_402_to_L24_2644.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_402_to_L24_2644.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `15` | `+0.26` | `+0.006` | `-0.043` |

Interpretation:

- This is weak/inverse downstream movement, not a feed-forward edge.
- Keep `L19:402` anchored to the confirmed dialogue-punctuation path; do not promote this mixed-scene downstream.

### WEAK-EDGE-56: negation/contraction upstream weakly moves mixed scene downstream

I screened `L19:913 -> L24:2644` as another mixed scene-family sibling. The score moves through the upstream, but the nominated downstream barely moves.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_913_to_L24_2644.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_913_to_L24_2644.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_913_to_L24_2644.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `15` | `-1.58` | `-0.021` | `+0.178` |

Interpretation:

- This is upstream-owned score movement, not a downstream circuit.
- Keep `L19:913` anchored to the negation/contraction family.

### BOUNDARY-14: polished practical/promotional sentence and paragraph boundary circuit

I promoted `L19:3028 -> L24:1600` after first-pass ablation, matched controls, and token-locus inspection.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3028_to_L24_1600.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3028_to_L24_1600.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3028_to_L24_1600.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_3028_to_L24_1600.csv`
- Upstream loci: `docs/experiments/pangram_sae_direct_L19_3028_token_loci.md`
- Downstream loci: `docs/experiments/pangram_sae_direct_L24_1600_for_L19_3028_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `13` | `-72.02` | `-5.06` | `-0.072` |

Control summary:

- Different-upstream same-downstream controls are tiny: largest mean mass movement is `+1.40` and largest max movement is `+0.086`.
- Same-upstream different-downstream controls are also tiny relative to target: largest mean absolute mass movement is `1.87` and largest max movement is `0.035`.

Token-locus summary:

- `L19:3028`: 156 inspected active tokens; 98 sentence-boundary, 42 line-break, 12 punctuation, 3 word, 1 style/function.
- `L24:1600`: 156 inspected active tokens; 93 sentence-boundary, 44 line-break, 15 punctuation, 3 word, 1 style/function.
- The two nodes share the same peak token in `8/13` documents and have mean top-token Jaccard `0.842`.
- Top tokens are overwhelmingly `.`, `.\n\n`, and comma tokens in polished practical/travel/promotional passages.

Interpretation:

- This is a controlled model-side boundary/rhythm circuit in polished practical/promotional prose.
- It should not be described as a compact lexical slop feature: the local evidence is period and paragraph-boundary rhythm, with only a small negative classifier-score movement.

### WEAK-EDGE-57: weak mixed-source downstream movement

I screened `L19:2572 -> L24:716` from the direct-edge queue. It does not have enough downstream peak movement to justify controls.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_2572_to_L24_716.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_2572_to_L24_716.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_2572_to_L24_716.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `19` | `-2.61` | `-0.009` | `-0.016` |

Interpretation:

- This is below the downstream-peak threshold.
- Screen it out rather than spending controls/loci on a weak mixed-source edge.

### VERSE-6: broad poem-word upstream into rhymed poetic content downstream

I promoted `L19:33 -> L24:457` after first-pass ablation, matched controls, and token-locus inspection. This is a real sibling feed into the same `L24:457` downstream as `VERSE-5`, but it is mediated rather than same-token.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_33_to_L24_457.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_33_to_L24_457.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_33_to_L24_457.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_33_to_L24_457.csv`
- Upstream loci: `docs/experiments/pangram_sae_direct_L19_33_for_L24_457_token_loci.md`
- Downstream loci: `docs/experiments/pangram_sae_direct_L24_457_for_L19_33_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `21` | `-43.96` | `-1.32` | `+0.011` |

Control summary:

- Different-upstream same-downstream controls are tiny: largest mean mass movement is `0.67` and largest max movement is `0.034`.
- Same-upstream different-downstream controls are also much smaller than target: largest mean mass movement is `1.39` and largest max movement is `0.161`.

Token-locus summary:

- `L19:33`: top inspected tokens are broad poem-word/function-word mass such as `and`, `the`, `to`, `was`, `her`, `steady`, `happy`, `brave`, and `with`.
- `L24:457`: top inspected tokens are rhymed/poetic content words such as `night`, `grace`, `bright`, `light`, `sky`, `tall`, `blue`, `wide`, `eyes`, `sight`, and `day`.
- The two nodes have no same-token alignment on the inspected rows: same peak `0/21`, mean top-token Jaccard `0.000`.

Interpretation:

- This is a controlled generated-verse mediation path: broad poem-register wording upstream feeds a rhymed/content-word downstream.
- It differs from `VERSE-5`, where both nodes are word-dominant and moderately overlapping, and from `VERSE-1`, where the mechanism is lineation/stanza format.
- It is not detector-positive slop evidence; the score movement is near zero.

### WEAK-EDGE-58: weak downstream movement with near-zero score effect

I screened `L19:2878 -> L24:1591` from the direct-edge queue.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_2878_to_L24_1591.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_2878_to_L24_1591.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_2878_to_L24_1591.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `18` | `-0.19` | `-0.011` | `+0.002` |

Interpretation:

- This is below the downstream-edge threshold.

### WEAK-EDGE-59: weak/inverse downstream movement

I screened `L19:2023 -> L24:2537` from the direct-edge queue.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_2023_to_L24_2537.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_2023_to_L24_2537.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_2023_to_L24_2537.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `12` | `+0.17` | `+0.048` | `+0.003` |

Interpretation:

- This is inverse/tiny downstream movement, not a feed-forward edge.

### BOUNDARY-15: practical/advice connector scaffold into sentence-boundary terminal

I promoted `L19:1908 -> L24:2470` after first-pass ablation, matched controls, and token-locus inspection. It is a sibling feed into the same downstream terminal as `BOUNDARY-9`, but the upstream is less same-token-local.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_1908_to_L24_2470.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_1908_to_L24_2470.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_1908_to_L24_2470.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_1908_to_L24_2470.csv`
- Upstream loci: `docs/experiments/pangram_sae_direct_L19_1908_for_L24_2470_token_loci.md`
- Downstream loci: `docs/experiments/pangram_sae_direct_L24_2470_for_L19_1908_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `18` | `-32.32` | `-1.22` | `-0.011` |

Control summary:

- Different-upstream same-downstream controls are tiny: largest mean mass movement is `0.32`, largest max movement is `0.016`.
- Same-upstream different-downstream controls are also much smaller than target: largest mean mass movement is `1.26`, largest max movement is `0.057`.

Token-locus summary:

- `L19:1908`: 216 inspected active tokens; 101 sentence-boundary, 66 punctuation, 37 word, 9 style/function, 2 line-break, 1 discourse-marker.
- `L24:2470`: 216 inspected active tokens; 181 sentence-boundary, 26 punctuation, 9 line-break.
- Top upstream tokens are `.`, comma, `but`, `-`, `so`, `;`, and `:`.
- Top downstream tokens are almost entirely periods plus sentence-boundary variants.
- Same peak `0/18`; mean top-token Jaccard `0.318`.

Interpretation:

- This is a controlled practical/advice context-to-boundary path.
- It should be grouped with `BOUNDARY-9` as another input to `L24:2470`, but not collapsed into it: `BOUNDARY-9` is more compact and same-token boundary-like, while this path carries connector/punctuation scaffold into the boundary terminal.

### WEAK-EDGE-60: broad lyrical upstream does not feed VERSE-1 lineation terminal

I screened `L19:2130 -> L24:61` as a verse-family sibling.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_2130_to_L24_61.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_2130_to_L24_61.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_2130_to_L24_61.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `15` | `-0.34` | `-0.130` | `-0.087` |

Interpretation:

- The score movement follows the broad lyrical upstream, but the lineation downstream barely moves.
- Keep `L19:2130` anchored to `VERSE-3` and `L24:61` anchored to `VERSE-1`.

### WEAK-EDGE-61: upstream-owned score movement without downstream edge

I screened `L19:3161 -> L24:1112` from the direct-edge queue.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3161_to_L24_1112.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3161_to_L24_1112.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3161_to_L24_1112.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-0.12` | `-0.011` | `-0.081` |

Interpretation:

- This is upstream-owned score movement, not a downstream circuit.

### WEAK-EDGE-62: upstream-owned score movement with tiny downstream peak change

I screened `L19:3161 -> L24:848` as a same-upstream sibling after `WEAK-EDGE-61`.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3161_to_L24_848.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3161_to_L24_848.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3161_to_L24_848.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `15` | `-5.07` | `-0.009` | `-0.065` |

Interpretation:

- The mass movement is still far below confirmed edges and the peak barely moves.
- Treat `L19:3161` as upstream-owned score evidence in this batch, not as a confirmed feed into either `L24:1112` or `L24:848`.

### CONTROL-31: human/no-robots broad prose context into document-wide word accumulator

I promoted `L19:1650 -> L24:627` after first-pass ablation, matched controls, and token-locus inspection. This is a very high-mass path, but it is mostly a human/no-robots broad-control path rather than a compact local style circuit.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_1650_to_L24_627.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_1650_to_L24_627.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_1650_to_L24_627.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_1650_to_L24_627.csv`
- Upstream loci: `docs/experiments/pangram_sae_direct_L19_1650_for_L24_627_token_loci.md`
- Downstream loci: `docs/experiments/pangram_sae_direct_L24_627_for_L19_1650_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `22` | `-665.56` | `-5.64` | `+0.094` |

Control summary:

- The largest different-upstream same-downstream control is `L19:3287 -> L24:627` at `-37.94` mass and `-0.35` max, far below the target.
- Same-upstream different-downstream controls are much smaller: largest mean mass movement is `1.88`.

Token-locus summary:

- `L19:1650`: 242 inspected active tokens; 99 word, 62 sentence-boundary, 53 line-break, 16 punctuation, 10 contraction pieces.
- `L24:627`: 264 inspected top tokens in the detailed export, almost all ordinary words (`245/264`), but summary rows show many documents have 150-550 active downstream tokens.
- Same peak `0/22`; mean top-token Jaccard `0.044`.

Interpretation:

- This is a controlled human/no-robots broad prose path, not a local same-token circuit.
- The positive score movement is important as a guardrail: source mix and loci are necessary before interpreting score-positive movement as AI slop.

### VERSE-7: lyrical comma-line / poetic register path

I promoted `L19:2130 -> L24:2278` after first-pass ablation, matched controls, and token-locus inspection.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_2130_to_L24_2278.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_2130_to_L24_2278.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_2130_to_L24_2278.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_2130_to_L24_2278.csv`
- Upstream loci: `docs/experiments/pangram_sae_direct_L19_2130_for_L24_2278_token_loci.md`
- Downstream loci: `docs/experiments/pangram_sae_direct_L24_2278_for_L19_2130_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `29` | `-139.77` | `-2.18` | `-0.133` |

Control summary:

- Same-downstream controls are tiny: largest alternate upstream mass movement is `0.91`, with max movement `0.041`.
- Same-upstream alternate downstream controls are also small relative to the target: largest mass movement is `3.55`, with max movement `0.143`.

Token-locus summary:

- `L19:2130`: 348 inspected active tokens; 170 line-break, 90 word, 81 punctuation, 6 style/function-word, 1 sentence-boundary.
- `L24:2278`: 348 inspected active tokens; 160 line-break, 110 word, 70 punctuation, 5 style/function-word, 3 sentence-boundary.
- High-frequency active tokens include comma-newline, comma, period-newline, `love`, `And`, `Together`, `hearts`, and `forever`.
- Same peak `13/29`; mean top-token Jaccard `0.530`.

Interpretation:

- This is a controlled lyrical comma-line / poetic-register circuit.
- It is separate from the `L24:61` lineation terminal; `WEAK-EDGE-60` showed that `L19:2130` does not materially feed `L24:61`.
- The score movement is negative, so this is register/format evidence rather than AI-slop evidence.

### RESOLUTION-2: sentence-boundary accumulation into story-resolution / moral wording

I promoted `L19:3173 -> L24:3072` after first-pass ablation, matched controls, and token-locus inspection. This is a second controlled input to the same downstream terminal as `RESOLUTION-1`, but its upstream is more boundary-accumulator-like.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3173_to_L24_3072.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3173_to_L24_3072.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3173_to_L24_3072.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_3173_to_L24_3072.csv`
- Upstream loci: `docs/experiments/pangram_sae_direct_L19_3173_for_L24_3072_token_loci.md`
- Downstream loci: `docs/experiments/pangram_sae_direct_L24_3072_for_L19_3173_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `16` | `-19.74` | `-1.03` | `-0.084` |

Control summary:

- Same-downstream alternate upstream controls are tiny: mean mass movement ranges from `0.07` to `0.19`.
- Same-upstream alternate downstream controls are also much smaller than the target: largest mean mass movement is `1.15`.

Token-locus summary:

- `L19:3173`: 192 inspected active tokens; top raw tokens are `.` (`131`) and paragraph-period (`38`), with active-token summaries dominated by sentence-boundary and punctuation mass.
- `L24:3072`: 192 inspected active tokens; top raw tokens are `.` (`67`), paragraph-period (`45`), comma (`7`), and story-resolution words such as `saving`, `guided`, `village`, `valuable`, and `taught`.
- Same peak `0/16`; mean top-token Jaccard `0.218`.

Interpretation:

- This is a controlled context-to-resolution terminal, not a same-token local period circuit.
- The downstream appears to combine story-ending boundary rhythm with explicit resolution/moral wording.
- The score movement is negative, so it is useful as detector-human-side narrative-closure evidence rather than positive AI-slop evidence.

### CONTROL-32: human-heavy broad prose word/punctuation accumulator

I promoted `L19:338 -> L24:1577` after first-pass ablation, matched controls, and token-locus inspection. The edge is highly causal and specific, but the feature is broad and human-heavy rather than compact AI-style evidence.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_338_to_L24_1577.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_338_to_L24_1577.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_338_to_L24_1577.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_338_to_L24_1577.csv`
- Upstream loci: `docs/experiments/pangram_sae_direct_L19_338_for_L24_1577_token_loci.md`
- Downstream loci: `docs/experiments/pangram_sae_direct_L24_1577_for_L19_338_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `17` | `-174.86` | `-2.25` | `-0.030` |

Control summary:

- Same-downstream alternate upstream controls are tiny: all are below `1.0` mean mass movement.
- Same-upstream alternate downstream controls are also small: all are below `0.53` mean mass movement.

Token-locus summary:

- Source mix is human-heavy: `10` human, `3` GLM, `2` GPT, and `2` Gemini rows.
- `L19:338`: mean `96.9` active tokens/doc; inspected top tokens are mostly ordinary words plus punctuation (`.`, comma, `and`, `of`, `in`, `to`, `on`, `as`).
- `L24:1577`: mean `152.6` active tokens/doc; inspected top tokens are similar but broader (`.`, comma, `and`, `when`, `but`, `on`, `as`).
- Same peak `7/17`; mean top-token Jaccard `0.437`.

Interpretation:

- This is a broad prose accumulator path, not a compact local style circuit.
- Its specificity under controls makes it useful as a positive causal-control path.
- Its source mix and negative score movement keep it out of the positive AI-slop queue.

### VERSE-8: broad poem-word upstream into ornate/generated verse accumulator

I promoted `L19:33 -> L24:953` after first-pass ablation, matched controls, and token-locus inspection. This is a sibling of `VERSE-6` on the upstream side and a sibling of `VERSE-4` on the downstream side.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_33_to_L24_953.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_33_to_L24_953.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_33_to_L24_953.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_33_to_L24_953.csv`
- Upstream loci: `docs/experiments/pangram_sae_direct_L19_33_for_L24_953_token_loci.md`
- Downstream loci: `docs/experiments/pangram_sae_direct_L24_953_for_L19_33_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `21` | `-158.50` | `-1.49` | `+0.025` |

Control summary:

- Same-downstream alternate upstream controls are tiny relative to target: largest mean mass movement is `2.09`.
- Same-upstream alternate downstream controls are also tiny: all are below `0.52` mean mass movement.

Token-locus summary:

- Source mix is generated-verse-heavy: `8` GPT, `4` GLM, `4` Gemini, `3` Qwen, and `2` human rows.
- `L19:33`: mean `120.6` active tokens/doc; broad poem-word/function-word mass with top raw tokens `and`, `the`, `to`, `her`, and `in`.
- `L24:953`: mean `123.4` active tokens/doc and mean mass `931.2`; stronger ornate/generated verse word-and-line accumulator with top raw tokens including comma, comma-newline, `heart`, `away`, `beneath`, `world`, and `warmth`.
- Same peak `0/21`; mean top-token Jaccard `0.023`.

Interpretation:

- This is a controlled mediated verse path, not a local same-token rhyme or lineation circuit.
- `L19:33` appears to provide broad poem-register word mass into several downstream verse terminals.
- The small positive score movement makes it relevant to model-verse style, but it is still too broad to call a compact slop marker.

### WEAK-EDGE-63: weak/inverse downstream movement with upstream-owned score movement

I screened `L19:2572 -> L24:3160` from the direct-edge queue.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_2572_to_L24_3160.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_2572_to_L24_3160.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_2572_to_L24_3160.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `18` | `+0.18` | `-0.0001` | `-0.011` |

Interpretation:

- The nominated downstream does not move under upstream ablation.
- The small score movement is upstream-owned, so this is not a confirmed `L24:3160` circuit.

### P-LEX-1: `p`/`P`-initial lexical-wordpiece path

I promoted `L19:1781 -> L24:890` after first-pass ablation, matched controls, and token-locus inspection. Control selection required widening the latent index to `--top-latents-per-layer 512` because `L24:890` is outside the selector's default top-160 layer index.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_1781_to_L24_890.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_1781_to_L24_890.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_1781_to_L24_890.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_1781_to_L24_890.csv`
- Upstream loci: `docs/experiments/pangram_sae_direct_L19_1781_for_L24_890_token_loci.md`
- Downstream loci: `docs/experiments/pangram_sae_direct_L24_890_for_L19_1781_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `-37.14` | `-6.57` | `-0.007` |

Control summary:

- Same-downstream alternate upstream controls are essentially zero: largest mean mass movement is `0.04`.
- Same-upstream alternate downstream controls are also tiny: largest mean mass movement is `0.26`.

Token-locus summary:

- Source mix: `6` human, `3` GLM, `2` GPT, `2` Qwen, and `1` Gemini rows.
- `L19:1781`: mean `10.8` active tokens/doc; mostly `p`/`P` word pieces and words.
- `L24:890`: mean `10.2` active tokens/doc; the same token family.
- High-frequency active tokens include `Peter`, `people`, `place`, `past`, `parents`, `patch`, `panic`, `population`, `point`, and `pale`.
- Same peak `14/14`; mean top-token Jaccard `0.927`.

Interpretation:

- This is a clean lexical/subword path, not prose style.
- It is useful as a positive control for high-specificity, high-overlap causal edges that should be filtered before semantic slop labeling.

### INVERSE-EDGE-2: inverse/upstream-owned score movement into `L24:848`

I screened `L19:3009 -> L24:848` from the direct-edge queue. This is not a feed-forward edge: ablating the upstream increases the nominated downstream mass.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3009_to_L24_848.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3009_to_L24_848.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3009_to_L24_848.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `17` | `+25.09` | `+0.038` | `+0.222` |

Interpretation:

- The effect is inverse or release-like: downstream mass increases under upstream ablation.
- Peak movement is tiny, so this is not a normal feed-forward dependency.
- The score movement is large and positive, but it should be treated as upstream-owned or competitive evidence until inverse effects get a separate control plan.

### CONTROL-33: mixed concrete content-word same-token path

I promoted `L19:3076 -> L24:585` after first-pass ablation, matched controls, and token-locus inspection. This is a strong content-word path with near-zero detector-score movement.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3076_to_L24_585.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3076_to_L24_585.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3076_to_L24_585.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_3076_to_L24_585.csv`
- Upstream loci: `docs/experiments/pangram_sae_direct_L19_3076_for_L24_585_token_loci.md`
- Downstream loci: `docs/experiments/pangram_sae_direct_L24_585_for_L19_3076_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `13` | `-122.14` | `-4.13` | `-0.0002` |

Control summary:

- Same-downstream alternate upstream controls are tiny relative to target: largest mean mass movement is `0.70`.
- Same-upstream alternate downstream controls are also tiny: largest mean mass movement is `0.29`.

Token-locus summary:

- Source mix: `7` human, `3` GPT, `2` Qwen, and `1` GLM rows.
- `L19:3076`: mean `55.8` active tokens/doc; word-only content tokens.
- `L24:585`: mean `73.5` active tokens/doc; same content-token family, somewhat broader.
- High-frequency active tokens include `coolant`, `daycare`, `numeral`, `folklore`, `caravan`, `terrain`, `tether`, `beneath`, `humanoid`, and `serpent`.
- Same peak `10/13`; mean top-token Jaccard `0.512`.

Interpretation:

- This is a controlled same-token content-word path, not a reusable style marker.
- Its near-zero score movement makes it useful as a content/topic control for strong causal edges.

### LINE-1: human expressive line-ending punctuation/newline circuit

I promoted `L19:3529 -> L24:3058` after first-pass ablation, matched controls, and token-locus inspection. This is a true feed-forward path for line-ending punctuation/newline tokens, despite the same upstream having an inverse sibling elsewhere.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3529_to_L24_3058.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_3529_to_L24_3058.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3529_to_L24_3058.csv`
- Controls: `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_3529_to_L24_3058.csv`
- Upstream loci: `docs/experiments/pangram_sae_direct_L19_3529_for_L24_3058_token_loci.md`
- Downstream loci: `docs/experiments/pangram_sae_direct_L24_3058_for_L19_3529_token_loci.md`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `13` | `-184.03` | `-8.16` | `+0.028` |

Control summary:

- Same-upstream alternate downstream controls are tiny relative to target: largest mean mass movement is `5.36`, with max movement below `0.19`.
- Same-downstream alternate upstream controls are mostly tiny, except `L19:3899 -> L24:3058`, which moves mass (`-33.84`) but not peak activation (`-0.12`), far below the target peak collapse.

Token-locus summary:

- Source mix: `12` human and `1` GPT row.
- `L19:3529`: mean `14.2` active tokens/doc, all line-break/punctuation-newline tokens.
- `L24:3058`: mean `17.7` active tokens/doc, same line-ending family.
- Top raw tokens are `.\n`, bare newline, `,\n`, `?\n`, `!\n`, and paragraph-period.
- Same peak `5/13`; mean top-token Jaccard `0.714`.

Interpretation:

- This is a human-side expressive line-ending circuit.
- It is related to verse/lineation controls but should not be collapsed into generated-verse evidence: the source mix is overwhelmingly human and the local feature is punctuation/newline format.

### WEAK-EDGE-64: weak alternate upstream into `PARA-1` downstream

I screened `L19:5 -> L24:1661` from the direct-edge queue.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_5_to_L24_1661.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_5_to_L24_1661.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_5_to_L24_1661.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `11` | `-0.06` | `-0.007` | `-0.004` |

Interpretation:

- The nominated downstream barely moves.
- Keep `L24:1661` anchored to the existing `PARA-1` paragraph/line-break path.

### WEAK-EDGE-65: weak/inverse-tiny downstream movement

I screened `L19:532 -> L24:3204` from the direct-edge queue.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_532_to_L24_3204.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_532_to_L24_3204.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_532_to_L24_3204.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `13` | `+0.24` | `+0.002` | `-0.016` |

Interpretation:

- The nominated downstream barely moves and moves in the wrong direction for a feed-forward edge.
- Screen out under the downstream-peak criterion.

### WEAK-EDGE-66: weak alternate upstream into `SCENE-2` downstream

I screened `L19:2041 -> L24:1834` from the direct-edge queue.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_2041_to_L24_1834.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_2041_to_L24_1834.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_2041_to_L24_1834.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `11` | `-1.36` | `-0.093` | `-0.062` |

Interpretation:

- The nominated downstream moves only weakly.
- Keep `L24:1834` anchored to the stronger `SCENE-2` upstream rather than promoting this alternate.

### WEAK-EDGE-67: upstream-owned score movement without downstream edge

I screened `L19:1878 -> L24:3759` from the direct-edge queue.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_1878_to_L24_3759.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_1878_to_L24_3759.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_1878_to_L24_3759.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `14` | `+0.28` | `-0.001` | `-0.142` |

Interpretation:

- The score movement is substantial, but the nominated downstream barely changes.
- Screen out as upstream-owned score movement rather than a confirmed downstream circuit.

### WEAK-EDGE-68: weak alternate into ornate/generated verse downstream

I screened `L19:1325 -> L24:953` from the direct-edge queue.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_1325_to_L24_953.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_1325_to_L24_953.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_1325_to_L24_953.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `21` | `-11.16` | `+0.019` | `-0.006` |

Interpretation:

- Downstream mass moves modestly, but the peak does not collapse.
- Keep `L19:1325` anchored to `VERSE-5`; keep `L24:953` anchored to the stronger ornate/generated verse paths rather than promoting this cross-edge.

### BOUNDARY-16: practical/advice period and paragraph-boundary accumulator

Target: `L19:1415 -> L24:3743`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_1415_to_L24_3743.csv`
- `docs/experiments/pangram_sae_circuit_target_direct_L19_1415_to_L24_3743.md`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_1415_to_L24_3743.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_1415_to_L24_3743.json`
- `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_1415_to_L24_3743.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/loci_details_direct_L19_1415_for_L24_3743.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/loci_details_direct_L24_3743_for_L19_1415.csv`
- `docs/experiments/pangram_sae_direct_L19_1415_for_L24_3743_token_loci.md`
- `docs/experiments/pangram_sae_direct_L24_3743_for_L19_1415_token_loci.md`

Intervention summary:

| Docs | Source mix | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---|---:|---:|---:|
| `18` | human/GPT/GLM | `-78.47` | `-1.92` | `+0.012` |

Control summary:

| Control family | Mean L24 mass delta range | Mean L24 max delta range |
|---|---:|---:|
| same downstream, alternate upstream | `-0.18` to `+0.37` | `-0.004` to `+0.009` |
| same upstream, alternate downstream | `-0.003` to `+0.39` | `-0.002` to `+0.058` |

Token-locus summary:

| Node | Active tokens/doc | Mean mass/doc | Top tokens | Overlap |
|---|---:|---:|---|---|
| `L19:1415` | `66.1` | `157.9` | `.`, comma, `.\n`, `?`, ` so` | same-peak with L24: `0/18`; top-token Jaccard `0.351` |
| `L24:3743` | `29.0` | `160.5` | `.`, `.\n`, `.\n\n`, bullet, `?` | same-peak with L19: `0/18`; top-token Jaccard `0.351` |

Interpretation:

- The edge is real and specific: target downstream collapse is two orders of magnitude larger than matched controls.
- The local mechanism is punctuation-boundary rhythm in practical/advice prose, not a compact lexical or semantic slop marker.
- The zero same-peak overlap means this is better read as a mediated period/paragraph-boundary accumulator than a same-token handoff.

### WEAK-EDGE-69: weak sibling into practical-boundary downstream

I screened `L19:1233 -> L24:2470` from the direct-edge queue.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_1233_to_L24_2470.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_1233_to_L24_2470.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_1233_to_L24_2470.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `16` | `-9.64` | `-0.184` | `+0.013` |

Interpretation:

- The nominated downstream moves slightly, but not enough at the peak-token level to treat this as a confirmed feed into `L24:2470`.
- Keep `L24:2470` anchored to the stronger `BOUNDARY-9` and `BOUNDARY-15` upstreams.

### WEAK-EDGE-70: upstream-owned BOUNDARY-2 score movement without downstream collapse

I screened `L19:2943 -> L24:2898` from the direct-edge queue.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_2943_to_L24_2898.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_2943_to_L24_2898.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_2943_to_L24_2898.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `10` | `-7.72` | `-0.158` | `+2.075` |

Interpretation:

- The class-score movement is large because `L19:2943` is the already-known `BOUNDARY-2` upstream.
- The nominated alternate downstream barely collapses, so this is not a confirmed feed-forward edge.
- Keep the local `BOUNDARY-2` terminal as `L24:2919`.

### WEAK-EDGE-71: weak punctuation-to-scene sibling edge

I screened `L19:402 -> L24:1834` from the direct-edge queue.

Artifacts:

- Targets: `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_402_to_L24_1834.csv`
- Target examples: `docs/experiments/pangram_sae_circuit_target_direct_L19_402_to_L24_1834.md`
- Intervention: `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_402_to_L24_1834.csv`

Intervention summary:

| Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---:|---:|---:|---:|
| `16` | `-4.14` | `-0.054` | `-0.024` |

Interpretation:

- The sparse dialogue-punctuation upstream barely moves the `SCENE-2` downstream.
- Keep quote/dialogue punctuation and scene-boundary families separated unless a different upstream produces a stronger peak collapse.

## Layer-20 Restoration: First Confirmed `19 -> 20 -> 24` Circuits

The earlier deferral of layer-20 interventions was based on a bad checkpoint-compatibility assumption. Using the available interchangeable layer-20 checkpoint at `artifacts/pangram_llama_sae/layer20_full_glm52_cap640_k64_e50/pangram_llama_batchtopk_sae.pt`, I tested the old middle hops for the two strongest boundary seeds.

### BOUNDARY-2 full chain: `L19:2943 -> L20:3795 -> L24:2919`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_2943_to_L20_3795.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_2943_to_L20_3795.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_2943_to_L20_3795.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L20_3795_to_L24_2919.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L20_3795_to_L24_2919.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L20_3795_to_L24_2919.csv`
- `docs/experiments/pangram_sae_direct_L19_2943_for_L20_3795_token_loci.md`
- `docs/experiments/pangram_sae_direct_L20_3795_for_L19_2943_token_loci.md`
- `docs/experiments/pangram_sae_direct_L20_3795_for_L24_2919_token_loci.md`
- `docs/experiments/pangram_sae_direct_L24_2919_for_L20_3795_token_loci.md`

Intervention summary:

| Hop | Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---|---:|---:|---:|---:|
| `L19:2943 -> L20:3795` | `24` | `-98.49` | `-8.13` | `+2.326` |
| `L20:3795 -> L24:2919` | `24` | `-106.52` | `-11.54` | `+2.824` |

Control summary:

| Hop | Control family | Mean abs mass delta | Mean abs max delta | Largest abs max delta |
|---|---|---:|---:|---:|
| `L19:2943 -> L20:3795` | same downstream, alternate upstream | `7.96` | `0.178` | `0.708` |
| `L19:2943 -> L20:3795` | same upstream, alternate downstream | `8.26` | `0.024` | `0.056` |
| `L20:3795 -> L24:2919` | same downstream, alternate upstream | `12.53` | `0.546` | `1.228` |
| `L20:3795 -> L24:2919` | same upstream, alternate downstream | `13.59` | `0.012` | `0.043` |

Token-locus summary:

- `L19:2943` and `L20:3795` share the same peak token in `24/24` target docs; top-token Jaccard is `0.878`.
- `L20:3795` and `L24:2919` share the same peak token in `21/24` target docs; top-token Jaccard is `0.814`.
- Active loci are mostly periods, commas, and paragraph-final periods. The target docs are Qwen-heavy polished expository/advice/procedural continuations.

Interpretation:

- This is now a confirmed `19 -> 20 -> 24` boundary/window circuit, not merely a direct `19 -> 24` shortcut.
- The strong detector-score movement travels with the upstream/middle boundary feature, but the mechanistic local object is still punctuation and context-window boundary rhythm, not a semantic "final summary" feature.

### BOUNDARY-1 full chain: `L19:3009 -> L20:2077 -> L24:3642`

Artifacts:

- `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L19_3009_to_L20_2077.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L19_3009_to_L20_2077.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L19_3009_to_L20_2077.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/target_direct_L20_2077_to_L24_3642.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/intervention_direct_L20_2077_to_L24_3642.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/control_summary_direct_L20_2077_to_L24_3642.csv`
- `docs/experiments/pangram_sae_direct_L20_2077_for_L19_3009_token_loci.md`

Intervention summary:

| Hop | Docs | Downstream mass delta | Downstream max delta | Class-3 logit drop |
|---|---:|---:|---:|---:|
| `L19:3009 -> L20:2077` | `25` | `-209.04` | `-4.91` | `+0.261` |
| `L20:2077 -> L24:3642` | `23` | `-236.66` | `-6.09` | `+0.311` |

Control summary:

| Hop | Control family | Mean abs mass delta | Mean abs max delta | Largest abs max delta |
|---|---|---:|---:|---:|
| `L19:3009 -> L20:2077` | same downstream, alternate upstream | `0.121` | `0.000` | `0.001` |
| `L19:3009 -> L20:2077` | same upstream, alternate downstream | `0.371` | `0.051` | `0.065` |
| `L20:2077 -> L24:3642` | same downstream, alternate upstream | `0.097` | `0.002` | `0.006` |
| `L20:2077 -> L24:3642` | same upstream, alternate downstream | `0.565` | `0.061` | `0.174` |

Token-locus summary:

- `L20:2077` target examples are mostly GPT-5.5 (`23/25`) with two GLM examples.
- Active loci are almost entirely sentence/paragraph boundaries: top tokens are `.` (`174`), `.\n\n` (`114`), and dialogue paragraph-final periods (`12`).
- Mean `L20:2077` activation mass is `250.79` over `82.16` active tokens/doc.

Interpretation:

- This is the cleanest controlled `19 -> 20 -> 24` path so far. Matched controls are essentially zero on both hops.
- The semantic wrapper is GPT-heavy polished narrative/scene prose, but the local circuit is a period/paragraph-boundary scene-beat accumulator.
