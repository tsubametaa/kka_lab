#!/usr/bin/env bash
# Jalankan server vLLM untuk model chat Qwen2.5-3B (+ LoRA opsional) di port lokal 8002.
# Cocok untuk NVIDIA L40S (48 GB VRAM) bersamaan dengan serve_ocr_vllm.sh di port 8001.
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_ADAPTER="${MODEL_ADAPTER:-}"
LLM_HOST="${LLM_HOST:-127.0.0.1}"
LLM_PORT="${LLM_PORT:-8002}"
LLM_GPU_MEMORY_UTILIZATION="${LLM_GPU_MEMORY_UTILIZATION:-0.48}"
LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN:-16384}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/.venv-vllm/bin/vllm}"
if [[ ! -x "${VLLM_BIN}" ]]; then
  VLLM_BIN="vllm"
fi

args=(
  "${MODEL_ID}"
  --host "${LLM_HOST}"
  --port "${LLM_PORT}"
  --dtype bfloat16
  --gpu-memory-utilization "${LLM_GPU_MEMORY_UTILIZATION}"
  --max-model-len "${LLM_MAX_MODEL_LEN}"
)

if [[ -n "${MODEL_ADAPTER}" && -d "${MODEL_ADAPTER}" ]]; then
  args+=(--enable-lora --lora-modules "kka-lora=${MODEL_ADAPTER}")
elif [[ -n "${MODEL_ADAPTER}" ]]; then
  echo "WARNING: MODEL_ADAPTER='${MODEL_ADAPTER}' tidak ditemukan; menjalankan base model '${MODEL_ID}'." >&2
fi

exec "${VLLM_BIN}" serve "${args[@]}"
