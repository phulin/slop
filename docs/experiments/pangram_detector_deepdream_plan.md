# Pangram Detector DeepDream Plan

Goal: optimize a fixed-length continuation so Pangram/EditLens assigns it maximal AI-detector score, while a base-LM KL term keeps the relaxed text near a plausible continuation distribution for the prompt.

## Objective

Represent each completion position as a learned categorical distribution `q_t` over the tokenizer vocabulary. The detector sees the expected token embedding:

`e_t = q_t E`

For prompt embeddings `p` and completion embeddings `e`, optimize:

`loss = -editlens_score(p, e) + beta * mean_t KL(q_t || p0_t)`

where `p0_t` is the base Llama next-token distribution under the original hard prompt-plus-initial-completion prefix. This is a static KL anchor: it is cheaper and more stable than differentiating through a second autoregressive LM at every step. It tests whether the detector has nearby high-AI directions in token space before attempting a heavier dynamic-KL variant.

Pangram/EditLens uses four ordinal buckets trained from `cosine_score`, not a
binary label. The public config uses `n_buckets=4`, `lo_threshold=0.03`, and
`hi_threshold=0.15`; bucket 0 is near-human and bucket 3 is the high-AI-edit
bucket. Public inference computes the continuous EditLens score as the expected
bucket index divided by `n_buckets - 1`. Optimizing raw bucket-3 logit is easier
to exploit than optimizing this scalar score, so the CLI default is
`--target-objective score`.

## Controls

- Run with `--kl-anchor none` to expose the unconstrained detector optimum. Expect degenerate tokens.
- Sweep `--kl-coeff` over `0, 0.01, 0.05, 0.1, 0.5`.
- Repeat for multiple prompts and initial completions.
- Record the target detector probability, target logit, final decoded text, and mean KL.
- Compare final texts with integrated-gradient spans and Pangram SAE latents to see whether the optimized directions match previously observed detector features.

## Command Sketch

```bash
uv run slop-pangram-detector-deepdream \
  --base-model unsloth/Llama-3.2-3B \
  --local-model-dir artifacts/models/pangram_editlens_Llama-3.2-3B_unsloth_base \
  --local-files-only \
  --prompt "Write a concise answer about why cities build public libraries." \
  --completion-tokens 96 \
  --kl-coeff 0.05 \
  --steps 80 \
  --output-dir artifacts/pangram_deepdream/library_beta005
```

## First-Pass Interpretation

This is DeepDream-like, but the object being optimized is a relaxed token distribution rather than pixels. A high detector score with low KL means the detector has an accessible AI-looking direction near normal model text. A high detector score only at huge KL means the detector can be driven, but the optimum is off-manifold. Read decoded outputs as probes, not as normal samples; the real evidence is the trajectory plus follow-up attribution on the final hard decode.

## Autoregressive Search Follow-Up

The relaxed-token optimizer found detector directions, but hard decoding those
soft optima exposed an off-manifold exploit: the detector score was high in the
soft path and low when the decoded text was rescored normally. The better
generation framework is to keep the original 50-token free-running completion
fixed, then append tokens autoregressively from the base Llama next-token
distribution.

At each step:

- get a large top-k candidate pool from the base LM;
- filter invalid decoded tokens, non-ASCII tokens, overlong single tokens, and
  alphabetic all-caps tokens;
- in prose mode, also filter code/URL symbols, camelCase/code substrings,
  isolated uppercase initials, bare subword starts after punctuation, and
  multi-character punctuation-only tokens;
- score each candidate continuation with Pangram/EditLens;
- rank candidates by `editlens_score + lm_logprob_coeff * mean_lm_logprob -
  repetition_penalty - sequence_quality_penalty_coeff *
  sequence_quality_penalty`;
- keep a small beam and continue.

Recommended prompt-0 setting so far:

```bash
uv run slop-pangram-detector-ar-search \
  --base-model unsloth/Llama-3.2-3B \
  --local-model-dir artifacts/models/pangram_editlens_Llama-3.2-3B_unsloth_base \
  --local-files-only \
  --prompt-file artifacts/pangram_deepdream/random5_seed1729_beta005/prompt_00.txt \
  --prefix-tokens 50 \
  --continuation-tokens 50 \
  --candidate-top-k 48 \
  --candidate-pool-factor 16 \
  --beam-size 4 \
  --beam-branch-k 8 \
  --ascii-candidates \
  --reject-allcaps-tokens \
  --max-token-chars 24 \
  --reject-symbol-tokens \
  --reject-single-uppercase-tokens \
  --reject-camelcase-tokens \
  --reject-code-substrings \
  --reject-bare-subword-tokens \
  --reject-punctuation-run-tokens \
  --min-lm-logprob -5.0 \
  --lm-logprob-coeff 0.1 \
  --repetition-penalty 0.08 \
  --sequence-quality-penalty-coeff 0.0 \
  --device cuda \
  --torch-dtype bfloat16 \
  --output-dir artifacts/pangram_deepdream/prompt00_ar_beam4_top48_minlp50_lm01_rep008_shape
```

Prompt 0 result:

- fixed prefix score: `0.1214`;
- final score: `0.5960`;
- final bucket probabilities: `[0.1225, 0.3152, 0.2141, 0.3482]`;
- final mean LM log probability: `-2.1568`;
- continuation:

```text
est people in the world.
You have to work hard to learn all you can. You must practice, read, and write. You have to do these things. You won't know you have learned something until someone else asks. This happens in real
```

For comparison, the same beam setting without token-shape filters reached
`0.6183` but exploited the detector with all-caps rare-word fragments. A
stricter filtered setting, `lm_logprob_coeff=0.15` and
`repetition_penalty=0.12`, was more fluent but dropped to `0.3938`, so it is
too conservative for the current objective.

## Five-Prompt Status

The current report command is:

```bash
uv run slop-pangram-detector-ar-report \
  artifacts/pangram_deepdream/prompt00_ar_beam4_top48_minlp50_lm01_rep008_shape \
  artifacts/pangram_deepdream/random5_seed1729_ar_beam4_shape \
  artifacts/pangram_deepdream/random5_seed1729_ar_beam4_prose \
  artifacts/pangram_deepdream/random5_seed1729_ar_beam4_prose_boundary_v4 \
  --output-csv artifacts/pangram_deepdream/ar_search_comparison.csv \
  --output-json artifacts/pangram_deepdream/ar_search_comparison.json
```

Best current readable results:

- prompt 0: `0.1214 -> 0.5960`, readable generic advice;
- prompt 2: `0.2379 -> 0.4478`, mostly grammatical but repetitive;
- prompt 3: `0.1127 -> 0.6513`, high-scoring essay-rubric prose;
- prompt 4: `0.4095 -> 0.8658`, high-scoring and readable after prose filters.

Prompt 1 is the hard case. The detector can be driven to `0.6770`, but the
unfiltered route is URL/RDF-like markup. Prose filters block that, then the
search shifts through code errors, all-caps fragments, numeric fragments, and
punctuation. With strict prose filters the score falls below the original
prefix (`0.4351 -> 0.0765`). This is evidence that token filters are useful
guardrails, but not enough as the final quality control layer.

The current CLI now includes a full-continuation heuristic reranker,
`--sequence-quality-penalty-coeff`. The report recomputes the current quality
penalty for old and new runs. On prompt 1 it correctly assigns high penalties to
the URL/RDF route (`5.34`) and code-error route (`3.55`). A relaxed beam with
`--sequence-quality-penalty-coeff 0.15` reaches `0.8242`, but still finds an
awkward titlecase technical-list pattern with current computed penalty `1.62`.
Increasing or combining the penalty with hard code/symbol filters lowers the
score into the `0.50-0.58` range and still does not produce clean prose.

Current framework status: detector-guided AR beam search plus base-LM logprob,
token guardrails, and full-continuation quality reporting is good enough to
generate and diagnose high-AI-ness completions. It has not fully solved prompt
1. The next improvement should be a learned or model-based naturalness reranker
over complete candidate continuations, not more token-level filters.

## Sample-Rerank Baseline

The framework now also has a non-adversarial proposer:

```bash
uv run slop-pangram-detector-sample-search \
  --base-model unsloth/Llama-3.2-3B \
  --local-model-dir artifacts/models/pangram_editlens_Llama-3.2-3B_unsloth_base \
  --local-files-only \
  --prompt-file artifacts/pangram_deepdream/random5_seed1729_beta005/prompt_01.txt \
  --prefix-tokens 50 \
  --continuation-tokens 50 \
  --num-samples 128 \
  --sample-batch-size 4 \
  --sample-temperature 1.1 \
  --sample-top-p 0.97 \
  --sequence-quality-penalty-coeff 0.25 \
  --device cuda \
  --torch-dtype bfloat16 \
  --output-dir artifacts/pangram_deepdream/random5_seed1729_sample128_t11_q025/run_01
```

This keeps continuations on the base-LM manifold and then scores/reranks them.
On prompt 1 it avoids the adversarial URL/code/titlecase routes, but only gives
a modest score gain:

- 64 samples at temperature `0.95`: `0.4351 -> 0.4731`, but the top sample is
  SEO/list-like and receives quality penalty `0.55`;
- 128 samples at temperature `1.1`: `0.4351 -> 0.4599`, zero quality penalty,
  natural game-dev prose.

This sample-rerank arm is useful as a floor: if AR beam produces a high score
with a large quality penalty, sample search gives a lower-scoring but
on-manifold alternative. The report command now includes both AR-beam and
sample-search summaries.

## Candidate-Level Reports

Summary-level reports only show the selected winner from each run. For framework
work, the more useful artifact is candidate-level reporting:

```bash
uv run slop-pangram-detector-candidate-report \
  artifacts/pangram_deepdream/prompt00_ar_beam4_candidate_export_smoke \
  artifacts/pangram_deepdream/random5_seed1729_sample64_q025 \
  artifacts/pangram_deepdream/random5_seed1729_sample128_t11_q025 \
  --top-k 8 \
  --output-csv artifacts/pangram_deepdream/candidate_comparison.csv \
  --output-json artifacts/pangram_deepdream/candidate_comparison.json
```

AR beam now writes `pangram_detector_ar_search_final_beams.csv`, while sample
search writes `pangram_detector_sample_search_samples.csv`. The candidate report
combines both formats and sorts by objective, score, or quality. This matters
because prompt 1 has high-score but high-penalty candidates and lower-score
zero-penalty candidates. Candidate-level reporting lets us pick the best
high-AI/naturalness tradeoff after the run rather than baking one scalar
objective into every experiment.
