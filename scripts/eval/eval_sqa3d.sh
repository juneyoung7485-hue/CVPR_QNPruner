#!/bin/bash
# SQA3D test: predictions + EM.
# Usage: bash scripts/eval/eval_sqa3d.sh <method> [voxel_size] [K]   (method: qnpruner | native)
# EVAL_LIMIT=N runs only the first N questions (prediction file gets a _subset suffix).
cd "$(dirname "$0")/../.." || exit 1
source scripts/eval/common.sh "$@" || exit 1
PRED_FILE=./pred/sqa3d_${LABEL}${SUFFIX}.json
echo "=== SQA3D: POOLING=${POOLING} VOXEL_SIZE=${VOXEL_SIZE} TARGET_TOKENS=${TARGET_TOKENS} -> ${PRED_FILE} ==="
python llava/eval/model_sqa3d.py \
    --model-path ChaimZhu/LLaVA-3D-7B \
    --question-file playground/data/annotations/llava-3d-sqa3d_test_question.json \
    --answers-file "${PRED_FILE}" "${LIMIT_ARGS[@]}"
python llava/eval/sqa3d_evaluator.py \
    --pred-file "${PRED_FILE}" \
    --gt-file playground/data/annotations/llava3d_sqa3d_test_answer.json
