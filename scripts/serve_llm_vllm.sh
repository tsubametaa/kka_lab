#!/usr/bin/env bash
# Jalankan server vLLM continuous batching untuk model chat Qwen (+ LoRA opsional / merged BF16) di port 8002.
# Dioptimalkan untuk RTX 5090 (32 GB VRAM) dengan prefix caching agar puluhan siswa terlayani paralel < 3 detik.
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
LLM_SERVE_MODEL="${LLM_SERVE_MODEL:-${MODEL_ID}}"
LLM_MODEL_NAME="${LLM_MODEL_NAME:-kka-lora}"
MODEL_ADAPTER="${MODEL_ADAPTER:-}"
LLM_HOST="${LLM_HOST:-127.0.0.1}"
LLM_PORT="${LLM_PORT:-8002}"
LLM_GPU_MEMORY_UTILIZATION="${LLM_GPU_MEMORY_UTILIZATION:-0.82}"
LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN:-16384}"
LLM_MAX_NUM_SEQS="${LLM_MAX_NUM_SEQS:-64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/.venv-vllm/bin/vllm}"
if [[ ! -x "${VLLM_BIN}" ]]; then
  VLLM_BIN="vllm"
fi

args=(
  "${LLM_SERVE_MODEL}"
  --host "${LLM_HOST}"
  --port "${LLM_PORT}"
  --dtype bfloat16
  --gpu-memory-utilization "${LLM_GPU_MEMORY_UTILIZATION}"
  --max-model-len "${LLM_MAX_MODEL_LEN}"
  --max-num-seqs "${LLM_MAX_NUM_SEQS}"
  --enable-prefix-caching
)

# Jika melayani model hasil merge_and_unload() (folder lokal), ekspos langsung sebagai nama model target tanpa overhead LoRA runtime.
if [[ -d "${LLM_SERVE_MODEL}" ]]; then
  args+=(--served-model-name "${LLM_MODEL_NAME}")
elif [[ -n "${MODEL_ADAPTER}" && -d "${MODEL_ADAPTER}" ]]; then
  args+=(--enable-lora --lora-modules "${LLM_MODEL_NAME}=${MODEL_ADAPTER}")
elif [[ -n "${MODEL_ADAPTER}" ]]; then
  echo "WARNING: MODEL_ADAPTER='${MODEL_ADAPTER}' tidak ditemukan; menjalankan base model '${MODEL_ID}'." >&2
fi

exec "${VLLM_BIN}" serve "${args[@]}"
