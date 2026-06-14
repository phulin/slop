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

Do not replace the Torch/Transformers free-running CLI yet. vLLM remains
blocked for OLMo-3 offline generation on this host. SGLang is now viable as a
high-throughput fixed-length generation candidate in an isolated CUDA 12.8
sidecar, but only when `ignore_eos` is enabled. Without `ignore_eos`, SGLang
stops on OLMo's additional chat stop token (`<|im_end|>`, token id `100265`),
which does not match the current Torch free-running measurement contract.

Before using SGLang for science shards, run a contract check on the same prompt
sample against the Torch CLI and decide explicitly whether the fixed-length
Phase 2 generation contract should ignore additional chat stop tokens.

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

The rerun after tightening the benchmark script contract was
`stage2-phase2-olmo3-dpo-vllm0111-cu128-1prompt-smoke`
(`https://wandb.ai/phulin-self/slop-stage1/runs/dibzdwyd`). It again reached
OLMo engine initialization and then made no generation progress before the
external 300-second timeout. Exit code was `124`, no JSONL/CSV artifact was
written, and no vLLM/GPU process remained afterward. Treat vLLM 0.11.1 cu128 as
blocked on this host for OLMo-3 offline generation until a container, backend
flag change, or version change is validated.

## Repeatable Benchmark Script

The repo includes `scripts/benchmark_vllm_generation.py`. It reads Phase 2
prompt-package JSONL, runs vLLM offline generation, writes local JSONL/CSV
artifacts, logs aggregate feature-rate tables to W&B, and redacts generation
text from W&B sample rows. It now mirrors the Torch free-running contract more
closely by using the revised six-feature default set, adding `source` to
summary rows, recording prompt token counts, and truncating prompts to
`--max-prompt-tokens` token IDs before sending them to vLLM.

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

Current status: this smoke did not complete. Do not launch a larger vLLM shard
from this environment.

## SGLang CUDA 12.8 Candidate

Sidecar setup:

```bash
uv venv --python 3.12 /tmp/slop-sglang-bench
uv pip install --python /tmp/slop-sglang-bench/bin/python -e .
uv pip install --python /tmp/slop-sglang-bench/bin/python \
  'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128' 'torchaudio==2.8.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /tmp/slop-sglang-bench/bin/python \
  'sglang[srt]==0.5.2' 'transformers==4.57.6' \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
sudo apt-get install -y libnuma1 cuda-nvcc-12-8 ninja-build gcc-14 g++-14
```

Runtime environment:

```bash
CC=/usr/bin/gcc-14
CXX=/usr/bin/g++-14
CUDAHOSTCXX=/usr/bin/g++-14
CUDA_HOME=/usr/local/cuda-12.8
PATH=/usr/local/cuda-12.8/bin:$PATH
TORCH_CUDA_ARCH_LIST=8.0
CUDA_VISIBLE_DEVICES=0
```

Resolved stack:

- `sglang==0.5.2`
- `torch==2.8.0+cu128`
- `transformers==4.57.6`
- CUDA toolkit/nvcc 12.8
- GCC/G++ 14 as the CUDA host compiler

Early setup blockers resolved:

- The first install pulled CUDA 13 Torch wheels; reinstalling explicit cu128
  Torch fixed the driver/runtime mismatch.
- `sglang` needed the `[srt]` extras for `uvloop` and runtime dependencies.
- `transformers==4.56.1` did not recognize `model_type=olmo3`; 4.57.6 does.
- `libnuma1`, nvcc 12.8, `ninja-build`, and GCC/G++ 14 were required for
  FlashInfer/Triton first-run JIT on this host. CUDA 12.8 rejects the default
  GCC 15.

The repo now includes `scripts/benchmark_sglang_generation.py`. It mirrors the
vLLM benchmark contract: prompt-package JSONL input, prompt-token truncation,
local JSONL/CSV artifacts, W&B aggregate tables, redacted W&B sample rows, and
the revised Phase 2 feature set. It also creates an asyncio event loop before
calling `engine.generate`, which is required under this SGLang/uvloop stack.

W&B runs:

- `29i0m563`: one-prompt smoke, clean W&B sync. It generated one token because
  the sampled prompt immediately hit a stop token.
- `hkdjxvmm`: 64 prompts x 128 max tokens, default stop behavior. It generated
  only 1,117 tokens total because 61/64 prompts stopped on `<|im_end|>`.
  Decode throughput was `46.5` generated tokens/sec, dominated by early stops
  and not comparable to fixed-length Torch shards.
- `qija648z`: 64 prompts x 128 max tokens with `--ignore-eos`. It generated
  the full 8,192 tokens. Decode throughput was `1,969.4` generated tokens/sec;
  wall throughput including model load and CUDA graph capture was `257.9`
  generated tokens/sec.

The current Torch/Transformers target-shape OLMo shards run at about
`356-359` generated tokens/sec including load/compile for much larger
512-prompt x 8-completion x 1,024-token shards. SGLang's decode path is much
faster once loaded, but the 64-prompt wall-clock comparison is not yet better
because model load and graph capture dominate the small benchmark. For larger
fixed-length generation shards, SGLang is likely worth a controlled pilot,
provided the stop-token contract is accepted.
