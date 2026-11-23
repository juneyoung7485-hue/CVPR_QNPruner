#!/bin/bash
# OpenEQA (ScanNet part): predictions, converted to the official results format
# ([{question_id, answer}], <pred>_official.json). Score them with scripts/eval/score_openeqa.py (LLM-Match).
# Usage: OPENEQA_VIDEO_FOLDER=/data/LLaVA-3D-Pretrain/openeqa_ep600 \
#        bash scripts/eval/eval_openeqa.sh <method> [voxel_size] [K]   (method: qnpruner | native)
# EVAL_LIMIT=N runs only the first N questions (prediction file gets a _subset suffix).
cd "$(dirname "$0")/../.." || exit 1
source scripts/eval/common.sh "$@" || exit 1
VIDEO_FOLDER=${OPENEQA_VIDEO_FOLDER:-/data/LLaVA-3D-Pretrain}
PRED_FILE=./pred/openeqa_${LABEL}${SUFFIX}.json
echo "=== OpenEQA: VIDEO_FOLDER=${VIDEO_FOLDER} POOLING=${POOLING} VOXEL_SIZE=${VOXEL_SIZE} TARGET_TOKENS=${TARGET_TOKENS} -> ${PRED_FILE} ==="
python llava/eval/model_openeqa.py \
    --model-path ChaimZhu/LLaVA-3D-7B \
    --video-folder "${VIDEO_FOLDER}" \
    --question-file playground/data/annotations/llava-3d-openeqa_scannet.json \
    --answers-file "${PRED_FILE}" "${LIMIT_ARGS[@]}" || exit 1
python - "${PRED_FILE}" <<'PY'
import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p) if l.strip()]
out = p[:-len('.json')] + '_official.json'
json.dump([{'question_id': r['question_id'], 'answer': r['text']} for r in rows], open(out, 'w'), indent=1)
print(f'[openeqa] {len(rows)} predictions -> {out}')
PY
