# Phase 2 Generation Backend Benchmark Notes

Date: 2026-06-13

## Task Plan

1. Keep teacher-forced propensity on the existing Torch/Transformers scorer.
2. Test vLLM only for Phase 2 free-running generation, in isolated `uv`
   environments outside the project `.venv`.
3. Compare backend load/runtime viability before scaling generation shards.
4. Keep all real benchmark attempts W&B-logged and avoid raw prompt/output text
   in W&B tables.

## Current Decision

Do not replace the Torch/Transformers free-running CLI yet. Current vLLM is a
plausible generation backend, but it needs a CUDA-12.8-compatible sidecar stack
on this host. SGLang remains a second backend to test only if vLLM stays blocked
or underperforms after a successful cu128 smoke.

Teacher-forced AF scoring should stay on the existing Torch/Transformers path.
The scorer needs arbitrary continuation probability mass at internal corpus
offsets with exact reference alignment, shared-prefix caching, and bootstrap
summaries. Serving backends should not be used there until they can reproduce
that estimator exactly.

## vLLM Attempts

### Current vLLM Release

Install attempted in a sidecar env:

```bash
uv venv --python 3.12 /tmp/slop-vllm-bench
uv pip install --python /tmp/slop-vllm-bench/bin/python -e . vllm --torch-backend=auto
```

Resolved stack:

- `vllm==0.22.1`
- `torch==2.11.0+cu129`
- `transformers==5.12.0`

Import smoke passed only after adding the sidecar CUDA runtime path:

```bash
LD_LIBRARY_PATH=/tmp/slop-vllm-bench/lib/python3.12/site-packages/nvidia/cu13/lib:/tmp/slop-vllm-bench/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/lib/x86_64-linux-gnu
```

W&B attempts:

- `r962btj8`: failed during FlashInfer sampler JIT because no CUDA toolkit/nvcc
  was available at `/usr/local/cuda`.
- `c6i9dqr9`: retried with `VLLM_USE_FLASHINFER_SAMPLER=0`; failed during CUDA
  graph profiling with `CUDA driver version is insufficient for CUDA runtime
  version`.

Conclusion: current vLLM/cu129 wheels are not the efficient path on this
driver (`570.124.06`) unless the driver is upgraded or an official compatibility
container is validated.

### CUDA 12.8 vLLM Candidate

Dry-run and install command:

```bash
uv venv --python 3.12 /tmp/slop-vllm-cu128-bench
uv pip install --python /tmp/slop-vllm-cu128-bench/bin/python -e . 'vllm==0.11.1' --torch-backend=cu128
```

Resolved/imported stack:

- `vllm==0.11.1`
- `torch==2.9.0+cu128`
- `transformers==4.57.6`
- CUDA runtime 12.8, A100 visible

The first OLMo benchmark attempt (`6h1s5tq4`) initialized the engine but was
stopped after the parent process idled with no GPU work while an engine
subprocess remained alive. No generation artifact was produced. Before trying a
larger generation run, rerun a smaller smoke with one prompt, `max_new_tokens`
8, `VLLM_USE_FLASHINFER_SAMPLER=0`, and a short external timeout.

## Repeatable Benchmark Script

The repo now includes `scripts/benchmark_vllm_generation.py`. It reads Phase 2
prompt-package JSONL, runs vLLM offline generation, writes local JSONL/CSV
artifacts, logs aggregate feature-rate tables to W&B, and redacts generation
text from W&B sample rows.

Use the cu128 sidecar env first:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
/tmp/slop-vllm-cu128-bench/bin/python scripts/benchmark_vllm_generation.py \
  --model allenai/Olmo-3-7B-Instruct-SFT \
  --input artifacts/phase2/prompts/olmo3_dolci_sft_phase2_prompt_package_512.jsonl \
  --sample-size 1 \
  --temperature 0.0 \
  --top-p 0.95 \
  --max-new-tokens 8 \
  --max-model-len 1024 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --generations-output artifacts/phase2/generations/olmo3_sft_vllm0111_cu128_smoke.jsonl \
  --summary-output artifacts/phase2/generations/olmo3_sft_vllm0111_cu128_smoke_summary.csv \
  --wandb-run-name stage2-phase2-olmo3-sft-vllm0111-cu128-smoke \
  --wandb-mode online
```

If this smoke does not complete promptly, keep the next Phase 2 free-running
shard on `slop-free-running-emission` and revisit vLLM through a container or a
driver upgrade.
