# Pangram SAE Circuit Discovery Plan

Date: 2026-06-23

## Goal

Use the trained Pangram/EditLens Llama SAEs to move from isolated latent labels to small, testable detector circuits: sets of SAE latents across layers whose activations, causal effects, and prompt-level behavior jointly explain a detector score movement.

The immediate target is not a full mechanistic proof. The target is a ranked queue of circuit hypotheses with enough evidence that the next intervention is specific: which prompts, which layer/latent nodes, which token positions, and what score or downstream activation should change.

## Available State

We currently have browser-ready SAEs for layers 16, 19, 20, and 24. Layer 19 is the strongest new run:

- Dataset: Qwen-inclusive no-robots slice, 59,090 docs, balanced across human, GPT-5.5, Gemini 3.5 Flash, GLM-5.2, and Qwen3.6-35B.
- SAE: BatchTopK, `latent_dim=4096`, `k=64`, tied init, 50 epochs.
- Layer 19 reconstruction: best eval MSE `0.026288862`, final train MSE `0.026280778`.
- Latent usage: final training reached full batch coverage (`4096` unique latents, `dead=0` in the late schedule).
- Causal/rerank export: 200 selected latents, 13,735 gradient-scored docs, 82,955 exact generation-impact rows.
- Explorer run: `apps/latent-explorer/public/data/layer19`.

The layer-19 reranker targets detector class `3`, the class with the largest AI-minus-human logit difference in the sampled explorer set. Exact scores are prompt contrasts: human target-logit drop minus mean AI target-logit drop after ablating the latent.

## Working Definition of a Circuit

A circuit hypothesis is a small directed graph:

- Nodes: `(layer, SAE latent)` plus token/prompt activation loci.
- Edges: evidence that an upstream node helps produce or preserve a downstream node, a detector-logit direction, or both.
- Readout: a detector score movement, currently class-3 logit or human-vs-AI prompt contrast.

A hypothesis is not yet a circuit if it is only a high-activation latent, only a source-enriched latent, or only a one-layer ablation result. It becomes a circuit candidate when we can say:

1. The same prompt/token family recruits multiple layer-specific latents.
2. The latents have compatible causal signs on the detector score.
3. Upstream intervention changes either downstream latent activation or final detector score on held-out prompts.
4. The semantic interpretation is stable across more than one model source.

## Discovery Stages

### Stage 1: Seed Candidate Graphs From Existing Exports

Use no new model forwards. Read the browser index/rerank exports for layers 16, 19, 20, and 24.

For each latent:

- collect top high-impact prompt IDs;
- summarize model/source mix;
- summarize signed and absolute prompt-contrast impacts;
- keep a few high-activation contexts for manual labeling.

For adjacent layers, create edges when latent pairs share high-impact prompts. Rank edges by:

- number of shared prompts;
- prompt-set Jaccard overlap;
- model-impact profile cosine;
- signed-impact profile agreement.

Then greedily assemble layer `16 -> 19 -> 20 -> 24` chains. These are circuit seeds, not confirmed circuits.

Deliverables:

- `artifacts/pangram_llama_sae/circuit_discovery/latent_summaries.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/cross_layer_edges.csv`
- `artifacts/pangram_llama_sae/circuit_discovery/candidate_chains.csv`
- `docs/experiments/pangram_sae_circuit_seed_report.md`

### Stage 2: Human Semantic Labeling

For the top 20 seeds:

- open each latent in the explorer;
- label the observable feature family;
- mark whether it is stylistic, task-format, topic/entity, punctuation/position, or detector bookkeeping;
- record whether the feature is AI-signifying or human-signifying under exact ablation.

The useful output is a short list of labeled hypotheses such as:

- polished product-advice accumulator;
- formulaic narrative-emotion imagery;
- factual list/entity scaffold;
- period/end-of-sentence accumulator;
- human lexical/locality accumulator.

### Stage 3: Downstream-Activation Intervention

For each labeled seed, test whether upstream latent intervention changes downstream latent activity.

Primary interventions:

- ablate one upstream latent at a selected layer and rescan downstream SAE activations on the same prompt group;
- clamp an upstream latent contribution along its decoder direction and measure downstream changes;
- compare effect at peak-token positions and over full document activation mass.

Evidence to record:

- downstream latent max activation delta;
- downstream latent total activation mass delta;
- detector class-3 logit delta;
- whether the effect survives on held-out prompts from the same semantic category.

### Stage 4: Score-Path Intervention

Once a seed has upstream/downstream coupling evidence, test path specificity:

- ablate each node alone;
- ablate all nodes together;
- ablate unrelated matched-mass latents as controls;
- compare additive vs. subadditive score effects.

A stronger circuit should show a larger effect for the matched path than for random or merely high-mass latents, and the combined path should explain a meaningful share of the target score movement.

### Stage 5: Negative Controls and Generalization

Use controls that prevent over-reading:

- prompts with similar topic but different style;
- prompts from sources not dominant in the top examples;
- latents with similar activation mass but weak prompt overlap;
- same latent on non-peak positions;
- model-source swaps within the same prompt group.

The result should separate genuine style/detector circuits from source artifacts, topical shortcuts, and final-token accounting directions.

## Initial Candidate Families to Prioritize

1. High causal-impact AI directions in layer 19: start with the exact-max-abs top latents.
2. Latents with cross-layer prompt overlap: these are most likely to become multi-layer circuit seeds.
3. Human-signifying directions: include negative or human-favoring prompt contrasts so the analysis is not only "AI slop" features.
4. Previously suspicious accumulator latents: period/end-sentence and human-lexicon-like directions need path-specific tests because they may be intermediate bookkeeping rather than surface style features.

## Success Criteria

A first circuit discovery pass is successful if it produces:

- at least 10 cross-layer candidate chains with concrete prompt examples;
- at least 5 manually labeled hypotheses with plausible semantic descriptions;
- at least 3 intervention-ready hypotheses with exact layer/latent/token targets;
- one runnable intervention script design, or a first implementation, for upstream-latent ablation to downstream-SAE activation deltas.

## Current First Step

I added `scripts/discover_pangram_sae_circuit_seeds.py`, which implements Stage 1 from existing exports. It should be run before manual labeling, and its report should become the queue for the next explorer pass.

## Current Progress

Stage 1 is complete for the available browser exports. The seed pass produced:

- 480 latent summaries across layers 16, 19, 20, and 24.
- 2,763 adjacent-layer candidate edges.
- 50 greedy `16 -> 19 -> 20 -> 24` chains.
- Seed report: `docs/experiments/pangram_sae_circuit_seed_report.md`.
- Candidate registry: `docs/experiments/pangram_sae_circuit_candidate_registry.md`.

Stage 2 has started with first-pass labels for five candidate chains:

- Candidate A: human boilerplate / simple continuation form.
- Candidate B: polished AI literary scene-setting.
- Candidate C: rhymed or stanzaic verse imagery.
- Candidate D: informal/noisy human prose.
- Candidate E: AI product/advice/list scaffold.

Stage 3 has started with exact upstream-latent ablations for two compressed `16 -> 19 -> 24` chains. These were prioritized before we clarified that the available layer-20 SAE checkpoints can be used interchangeably:

- `L16:1677 -> L19:3009 -> L24:3642`
- `L16:3660 -> L19:3990 -> L24:1120`

The first interventions show strong downstream-latent suppression on both chains. The first matched controls show that classifier-logit movement is broad, but downstream-latent suppression is specific to the selected chain hops. That is the key current evidence: not a completed circuit proof, but two intervention-backed circuit seeds.

A larger control panel has also been generated and run for the currently available `16 -> 19 -> 24` topology:

- 40 selected controls.
- 460 document-level exact intervention rows.
- Literary scene controls stay near zero downstream-mass change, while the chain hops drop downstream mass by `-118.63` and `-249.58`.
- Human/form controls can perturb other features, but still do not reproduce the chain-specific downstream collapses of `-1330.84` and `-2177.07`.

Peak-token evidence changes the candidate-selection strategy. The strongest AI-scored chains examined so far are often period or sentence-boundary accumulators rather than lexical/content features. Going forward, candidate selection should explicitly stratify:

- boundary circuits: high punctuation or period fraction;
- lexical/style circuits: high word fraction, low punctuation fraction, source/model enrichment;
- human/form controls: human-heavy function-word or connective paths.

That stratified pass is now started. The first non-period lexical circuit seed is `L19:3450 -> L24:3469`, a discourse-marker / contrastive-transition path with examples around `However`, `such`, `Additionally`, and `Furthermore`. Ablating `L19:3450` on 19 shared-prompt documents reduces `L24:3469` downstream mass by `-31.16` on average, while five different-upstream/same-downstream controls have mean downstream mass delta `0.004`. The detector class-logit effect is small (`0.0492`), so this is currently evidence for a localized style feature rather than a dominant readout path.

Two additional direct `L19 -> L24` lexical seeds now have intervention evidence:

- `L19:913 -> L24:1310`: negation/contraction fragments such as `didn`, `doesn`, `wasn`, and `don`; downstream mass delta `-35.18`; different-upstream/same-downstream controls average `-0.083`.
- `L19:262 -> L24:1664`: simple modal/explanatory function wording such as `to`, `can`, `feel`, `still`, `are`, and `has`; downstream mass delta `-2965.80`; controls move some related features but remain much smaller than the chain effect.

This gives us three non-period lexical/direct seeds, plus the earlier sentence-boundary and human/form-control paths. The next full-chain step is to test the corresponding `L19 -> L20` and `L20 -> L24` hops using the available layer-20 checkpoint rather than treating layer 20 as blocked.

Held-out pattern tests have started for two lexical seeds. I selected documents by regex rather than latent-top-example overlap, excluding the direct target-export docs:

- `L19:3450 -> L24:3469` on 24 held-out discourse-marker documents: downstream mass delta `-23.80`, class-3 logit drop `0.0465`.
- `L19:913 -> L24:1310` on 24 held-out negation/contraction documents: downstream mass delta `-28.67`, class-3 logit drop `0.1136`.

These effects are smaller than the top-example effects but still strongly directional, which upgrades the discourse-marker and negation-fragment paths from top-export coupling to held-out lexical/style candidates.

Counterfactual text edits now support the same two paths:

- Discourse-marker edits reduce `L24:3469` baseline mass by `-13.43` on average in paired original/edited documents, and shrink the `L19:3450` ablation effect from `-22.65` to `-11.50`.
- Negation-contraction expansion reduces `L24:1310` baseline mass by `-16.80` on average, and shrink the `L19:913` ablation effect from `-28.75` to `-15.19`.

These edits are not a full causal path proof, but they show that the downstream latents are sensitive to the intended local surface forms in the same documents.

Stage 4 has started for the two strongest lexical seeds. Upstream-only, downstream-only, and joint node ablations all reduce the detector target score on held-out and counterfactual rows:

- Discourse marker path `L19:3450 -> L24:3469`: held-out mean class-3 logit drops are `0.0465` for `L19`, `0.0368` for `L24`, and `0.0531` jointly.
- Negation/contraction path `L19:913 -> L24:1310`: held-out mean class-3 logit drops are `0.1136` for `L19`, `0.0325` for `L24`, and `0.1178` jointly.

The joint effects are subadditive, which is consistent with shared-path structure. Matched held-out score controls sharpen the interpretation: the layer-19 anchor nodes are strongly score-specific relative to matched layer-19 controls, and layer-24 anchors are above matched layer-24 controls on average, but the final score readout is mostly carried by the layer-19 node. Anchor-upstream plus non-anchor downstream controls can match the anchor joint score effect, so the best current circuit evidence remains the combination of score-specific layer-19 features plus specific downstream-activation collapse at layer 24.

An independent short minimal-pair check was negative. Hand-written discourse-marker and contraction pairs sometimes weakly activated the layer-19 upstream nodes, but did not activate the corresponding layer-24 downstream nodes. This suggests the layer-24 lexical/style nodes are not simple isolated word detectors; they likely need generated-corpus context, longer surrounding style, stronger feature density, or other distributional evidence. Future minimal-pair tests should therefore use generated completions or corpus-derived edits rather than short hand-written snippets.

Longer synthetic stress checks split the two lexical paths:

- Discourse marker path: even a corpus-length synthetic document with many discourse markers did not activate `L24:3469`. This path is probably not a simple discourse-marker counter; it likely requires the broader expository/generated-corpus style state.
- Negation/contraction path: corpus-length contraction-heavy synthetic narrative strongly activated `L24:1310`, and `L19:913` ablation collapsed it. Expanded/plain synthetic text retained weaker residual activation, so the path appears to combine contraction density with narrative negation/syntax.

Near-term follow-up should therefore branch:

- For discourse, generate or select matched expository completions that preserve the corpus style while editing marker usage.
- For negation, run a contraction-density sweep in controlled narrative completions and measure dose-response in `L19:913`, `L24:1310`, and class-3 score.

The first negation density sweep is complete. In a fixed narrative frame with 0, 2, 4, 8, 16, and 32 contractions, contraction count correlates strongly with `L19:913` mass (`0.94`), `L24:1310` mass (`0.97`), and downstream ablation effect (`-0.97`). The class-3 logit drop correlation is weak (`0.13`). This upgrades the negation path as an activation-level circuit candidate, but it remains a weak standalone final-score readout.

Layer-20 restoration has now started and produced two confirmed `19 -> 20 -> 24` boundary circuits:

- `BOUNDARY-2`: `L19:2943 -> L20:3795 -> L24:2919`, a Qwen-heavy polished expository/advice boundary/window circuit. Hop effects are `-98.49` L20 mass / `-8.13` L20 max and `-106.52` L24 mass / `-11.54` L24 max, with much smaller matched controls.
- `BOUNDARY-1`: `L19:3009 -> L20:2077 -> L24:3642`, a GPT-heavy narrative scene-boundary circuit. Hop effects are `-209.04` L20 mass / `-4.91` L20 max and `-236.66` L24 mass / `-6.09` L24 max, with near-zero matched controls.

These are the first confirmed middle-layer circuits in this pass. Both are strong boundary/rhythm circuits rather than compact lexical slop markers.

Token-level locus inspection across the sweep is also complete. At low contraction density, peaks land mostly on expanded negation auxiliaries such as `did` and `do`, with `L24:1310` also activating on sentence boundaries. At high contraction density, both nodes shift to contraction subword pieces such as `didn`, `wouldn`, `couldn`, `wasn`, and `shouldn`. This supports a two-component path: broad negation/narrative structure at low density, direct contraction-piece activation at high density.

Held-out corpus token loci validate the direct contraction-piece component. Across the 24 held-out negation documents, contraction pieces account for `0.575` of active `L19:913` tokens and `0.651` of active `L24:1310` tokens, while expanded-negation words account for `0.000` in both nodes. The corpus path is therefore mostly contraction-piece driven, with the expanded-negation baseline showing up mainly in synthetic zero-contraction controls.

Sibling-latent checks show that negation/contraction is a small family, not a single isolated path. `L19:2787` and `L24:62` also show strong contraction dose response, and the matched sibling path `L19:2787 -> L24:62` causally suppresses downstream activation. However, `L19:913 -> L24:1310` is the score-relevant branch: it has the largest downstream collapse on the sweep (`-33.73`) and nonzero class-3 logit drop, while `L19:2787 -> L24:62` has smaller downstream collapse (`-15.73`) and no score drop. Cross-pairs are weaker than matched pairs, suggesting partial modularity within the contraction family.

New helper scripts:

- `scripts/export_pangram_sae_circuit_targets.py`: exports shared prompt/document bundles for a chain.
- `scripts/measure_pangram_sae_chain_intervention.py`: ablates an upstream SAE decoder contribution and measures downstream SAE activity plus detector logit movement.
- `scripts/select_pangram_sae_circuit_controls.py`: selects reproducible low-overlap controls for the next intervention batch.
- `scripts/run_pangram_sae_intervention_panel.py`: executes a selected control panel with a single model load and cached baselines.
- `scripts/export_pangram_sae_peak_windows.py`: exports local text windows around browser peak tokens.
- `scripts/summarize_pangram_sae_latent_peak_tokens.py`: screens browser latents by peak-token punctuation/word distribution.
- `scripts/select_pangram_sae_lexical_chains.py`: ranks cross-layer chains after filtering for word-heavy, low-punctuation peak-token distributions and optional AI enrichment.
- `scripts/select_pangram_sae_pattern_targets.py`: selects held-out target documents by text regex while excluding existing target-export document IDs.
- `scripts/build_pangram_sae_counterfactual_targets.py`: builds explicit-text original/edited target rows for local counterfactual surface-form tests.
- `scripts/measure_pangram_sae_score_ablation.py`: ablates one or more SAE nodes and measures detector target-logit/probability changes for score-path tests.
- `scripts/build_pangram_sae_score_control_panel.py`: converts matched intervention controls into score-ablation panels.
- `scripts/run_pangram_sae_score_ablation_panel.py`: runs many score ablations with cached baselines and one model load.
- `scripts/build_pangram_sae_minimal_pair_targets.py`: creates short independent lexical minimal-pair target CSVs for out-of-distribution checks.
- `scripts/build_pangram_sae_long_stress_targets.py`: creates longer and combined marked/plain synthetic stress targets.
- `scripts/analyze_pangram_sae_lexical_contexts.py`: summarizes text length, feature-hit density, activation mass, and score effects across corpus, counterfactual, minimal-pair, and stress targets.
- `scripts/build_pangram_sae_negation_density_sweep.py`: creates a fixed-frame contraction-density sweep for dose-response tests.
- `scripts/inspect_pangram_sae_token_loci.py`: runs selected SAE nodes on explicit target rows and exports top activating token loci and token-class summaries.

Current layer-20 note:

- The available layer-20 SAE checkpoints are interchangeable for this intervention workflow. The older note treating the GLM52 layer-20 checkpoint as incompatible was wrong. Layer-20 intervention hops should be tested directly using the available `layer20_full_glm52_cap640_k64_e50/pangram_llama_batchtopk_sae.pt` checkpoint when the matching neuralwatt `.pt` is absent.
