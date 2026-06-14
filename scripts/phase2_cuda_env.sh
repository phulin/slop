#!/usr/bin/env bash
# Source this before Phase 2 GPU/status commands after a shell or host reset.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source scripts/phase2_cuda_env.sh from the repository root" >&2
  exit 2
fi

SLOP_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLOP_NVIDIA_LIBS="$(find "${SLOP_REPO_ROOT}/.venv/lib" -path '*/site-packages/nvidia/*/lib' -type d 2>/dev/null | sort | paste -sd: -)"

if [[ -z "${SLOP_NVIDIA_LIBS}" ]]; then
  echo "No NVIDIA wheel libraries found under ${SLOP_REPO_ROOT}/.venv/lib" >&2
  return 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLOP_REPO_ROOT}/.uv-cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${SLOP_NVIDIA_LIBS}:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${SLOP_NVIDIA_LIBS}"
fi

echo "Phase 2 CUDA env ready:"
echo "  UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "  LD_LIBRARY_PATH includes $(tr ':' '\n' <<< "${SLOP_NVIDIA_LIBS}" | wc -l) NVIDIA wheel library directories"
