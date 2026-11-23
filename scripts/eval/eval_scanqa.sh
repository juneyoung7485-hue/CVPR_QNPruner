#!/bin/bash
# ScanQA val: predictions + EM / BLEU / METEOR / ROUGE / CIDEr.
# Usage: bash scripts/eval/eval_scanqa.sh <method> [voxel_size] [K]
#   method  qnpruner | native (LLaVA-3D without token selection)
#   e.g.    bash scripts/eval/eval_scanqa.sh qnpruner 0.2 64
#           bash scripts/eval/eval_scanqa.sh native
# EVAL_LIMIT=N runs only the first N questions (prediction file gets a _subset suffix).
cd "$(dirname "$0")/../.." || exit 1
source scripts/eval/common.sh "$@" || exit 1
PRED_FILE=./pred/scanqa_${LABEL}${SUFFIX}.json
echo "=== ScanQA: POOLING=${POOLING} VOXEL_SIZE=${VOXEL_SIZE} TARGET_TOKENS=${TARGET_TOKENS} -> ${PRED_FILE} ==="
python llava/eval/model_scanqa.py \
    --model-path ChaimZhu/LLaVA-3D-7B \
    --question-file playground/data/annotations/llava-3d-scanqa_val_question.json \
    --answers-file "${PRED_FILE}" "${LIMIT_ARGS[@]}"
python llava/eval/scanqa_evaluator.py \
    --pred-file "${PRED_FILE}" \
    --gt-file playground/data/annotations/llava3d_scanqa_val_answer.json
