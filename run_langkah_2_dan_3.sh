#!/usr/bin/env bash
set -euo pipefail

echo ""
echo "=================================================================="
echo ">>> [LANGKAH 2.1] Mengunduh Qwen/Qwen2.5-3B-Instruct & kunci model_record.json..."
echo "=================================================================="
python scripts/download_model.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --record model_record.json \
  --force

echo ""
echo "=================================================================="
echo ">>> [LANGKAH 2.2] Training QLoRA Dataset v3 pada Qwen/Qwen2.5-3B-Instruct..."
echo "=================================================================="
python training/train_lora.py \
  --data data/lora/lora_train.jsonl \
  --val-data data/lora/lora_val.jsonl \
  --record model_record.json \
  --out adapters/qwen-kka-lora-v3-qwen25 \
  --epochs 4 \
  --eval-strategy epoch \
  --early-stopping-patience 2 \
  --precision bf16 \
  --batch-size 4 \
  --grad-accum 2 \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --max-grad-norm 1.0 \
  --neftune-noise-alpha 5.0

echo ""
echo "=================================================================="
echo ">>> [LANGKAH 2.3] Evaluasi 40 kasus dengan adaptor v3 Qwen2.5-3B..."
echo "=================================================================="
python scripts/evaluate_assistant.py \
  --local-adapter adapters/qwen-kka-lora-v3-qwen25/final \
  --save-report exports/eval_v3_qwen25_adapter.json \
  --save-responses exports/eval_v3_qwen25_responses.jsonl

echo ""
echo "=================================================================="
echo ">>> [LANGKAH 3.1] Mengunduh Qwen/Qwen3-4B-Instruct-2507 & salin model_record_qwen3_4b.json..."
echo "=================================================================="
python scripts/download_model.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --record model_record.json \
  --force
cp model_record.json model_record_qwen3_4b.json

echo ""
echo "=================================================================="
echo ">>> [LANGKAH 3.2] Training QLoRA Dataset v3 pada Qwen/Qwen3-4B-Instruct-2507..."
echo "=================================================================="
python training/train_lora.py \
  --data data/lora/lora_train.jsonl \
  --val-data data/lora/lora_val.jsonl \
  --record model_record.json \
  --out adapters/qwen-kka-lora-v3-qwen3-4b \
  --epochs 4 \
  --eval-strategy epoch \
  --early-stopping-patience 2 \
  --precision bf16 \
  --batch-size 4 \
  --grad-accum 2 \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --max-grad-norm 1.0 \
  --neftune-noise-alpha 5.0

echo ""
echo "=================================================================="
echo ">>> [LANGKAH 3.3] Evaluasi 40 kasus dengan adaptor v3 Qwen3-4B-Instruct-2507..."
echo "=================================================================="
python scripts/evaluate_assistant.py \
  --local-adapter adapters/qwen-kka-lora-v3-qwen3-4b/final \
  --save-report exports/eval_v3_qwen3_4b_adapter.json \
  --save-responses exports/eval_v3_qwen3_4b_responses.jsonl

echo ""
echo "=================================================================="
echo ">>> [SELESAI] Langkah 1 s/d Langkah 3 berhasil diselesaikan!"
echo "=================================================================="
