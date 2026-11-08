#!/usr/bin/env bash
# Jalankan server internal Baidu Unlimited-OCR via vLLM di port lokal 8001.
# Menyisakan ~55% VRAM GPU (mis. di RTX 5090 32 GB) untuk model chat Qwen di app/main.py.
set -euo pipefail

OCR_MODEL_ID="${OCR_MODEL_ID:-baidu/Unlimited-OCR}"
OCR_HOST="${OCR_HOST:-127.0.0.1}"
OCR_PORT="${OCR_PORT:-8001}"
OCR_GPU_MEMORY_UTILIZATION="${OCR_GPU_MEMORY_UTILIZATION:-0.45}"

exec vllm serve "${OCR_MODEL_ID}" \
  --host "${OCR_HOST}" \
  --port "${OCR_PORT}" \
  --trust-remote-code \
  --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization "${OCR_GPU_MEMORY_UTILIZATION}" \
  --tensor-parallel-size 1

