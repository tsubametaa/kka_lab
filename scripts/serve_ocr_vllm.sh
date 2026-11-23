#!/usr/bin/env bash
# Jalankan server internal Baidu Unlimited-OCR via vLLM di port lokal 8001.
# Menyisakan ~55% VRAM GPU (mis. di RTX 5090 32 GB) untuk model chat Qwen di app/main.py.
set -euo pipefail

OCR_MODEL_ID="${OCR_MODEL_ID:-baidu/Unlimited-OCR}"
OCR_HOST="${OCR_HOST:-127.0.0.1}"
OCR_PORT="${OCR_PORT:-8001}"
OCR_GPU_MEMORY_UTILIZATION="${OCR_GPU_MEMORY_UTILIZATION:-0.38}"
OCR_MOE_BACKEND="${OCR_MOE_BACKEND:-triton}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/.venv-vllm/bin/vllm}"
if [[ ! -x "${VLLM_BIN}" ]]; then
  VLLM_BIN="vllm"
fi
PY_BIN="${REPO_ROOT}/.venv-vllm/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python3"
fi

args=(
  "${OCR_MODEL_ID}"
  --host "${OCR_HOST}"
  --port "${OCR_PORT}"
  --trust-remote-code
  --no-enable-prefix-caching
  --mm-processor-cache-gb 0
  --moe-backend "${OCR_MOE_BACKEND}"
  --gpu-memory-utilization "${OCR_GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size 1
)

if "${PY_BIN}" -c "from vllm.model_executor.models.unlimited_ocr import NGramPerReqLogitsProcessor" >/dev/null 2>&1; then
  args+=(--logits-processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor)
elif "${PY_BIN}" -c "from vllm.model_executor.models.ming_v3_asr_logits_processor import NoRepeatNGramLogitsProcessor" >/dev/null 2>&1; then
  args+=(--logits-processors vllm.model_executor.models.ming_v3_asr_logits_processor:NoRepeatNGramLogitsProcessor)
fi

exec "${VLLM_BIN}" serve "${args[@]}"

