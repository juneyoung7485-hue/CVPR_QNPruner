#!/bin/bash
# Usage: bash scripts/eval/eval_scanqa.sh <method> <voxel_size> <target_tokens>
#
#   method       native | 3dzip | dtc | vispruner | divprune | segpruner
#                | visionzip | tosa | fastv | sparsevlm
#   voxel_size   LLaVA-3D's voxelization pooling size. Every method starts from
#                the voxel sequence this builds, so it has to match across a
#                comparison -- 0.2 is the paper setting.
#   target_tokens  K. For the vision-path methods it is the token count they
#                compress to; for fastv/sparsevlm it is the decoder-side budget
#                (see LLM_PRUNE_BUDGET below).
#
# Example (baseline):   bash scripts/eval/eval_scanqa.sh native
# Example (3DZip):      bash scripts/eval/eval_scanqa.sh 3dzip 0.2 64
# Example (VisPruner):  bash scripts/eval/eval_scanqa.sh vispruner 0.2 128
# Example (FastV):      bash scripts/eval/eval_scanqa.sh fastv 0.2 128
# Example (SparseVLM):  bash scripts/eval/eval_scanqa.sh sparsevlm 0.2 128

# Run from the repo root and import *this* tree's llava (the container also has
# an older editable install at /workspace/3dzip) and its method packages.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

POOLING=${1:-native}
VOXEL_SIZE=${2:-0.2}
TARGET_TOKENS=${3:-0}

case "${POOLING}" in
    native|voxelize|3dzip|3dzip_gt_oracle|dtc|vispruner|divprune|segpruner|visionzip|tosa|fastv|sparsevlm) ;;
    # Any other name must be a method package directory at the repo root
    # (one per version, e.g. query_token_compression_v1/); see
    # llava/model/multimodal_encoder/method_registry.py.
    *) if [ -f "${POOLING}/__init__.py" ]; then :; else
       echo "Unknown method '${POOLING}'. Use: native | 3dzip | dtc | vispruner |" \
            "divprune | segpruner | visionzip | tosa | fastv | sparsevlm" >&2; exit 1; fi ;;
esac

# native = baseline 
if [ "${POOLING}" == "native" ]; then
    POOLING=voxelize
    VOXEL_SIZE=0.2
    TARGET_TOKENS=0
fi

# FastV and SparseVLM are query-based and prune inside the LLM decoder, not in
# the vision path: the vision tower stays on the uncompressed LLaVA-3D
# baseline and the budget is handed to the decoder-side pruner instead.
# LABEL keeps the output files named after the method the user asked for.
LABEL=${POOLING}
export LLM_PRUNE=${LLM_PRUNE:-none}
if [ "${POOLING}" == "fastv" ] || [ "${POOLING}" == "sparsevlm" ]; then
    export LLM_PRUNE=${POOLING}
    export LLM_PRUNE_TOKENS=${LLM_PRUNE_TOKENS:-${TARGET_TOKENS}}
    # The vision path stays on the LLaVA-3D baseline: voxelization pooling at
    # the requested size and no token budget, because K is spent in the decoder
    # instead. VOXEL_SIZE is whatever was passed, so these run on the same voxel
    # sequence as the vision-path methods at the same argument.
    POOLING=voxelize
    ADAPTIVE_RATIO_OVERRIDE=0
fi
export LLM_PRUNE_TOKENS=${LLM_PRUNE_TOKENS:-${TARGET_TOKENS}}
# Each decoder-side method is measured at the budget its own paper's "retain K"
# parameter denotes, which is not the same quantity for the two:
#
#   FastV      K tokens survive from the pruning layer onward. The layer-average
#              reading is not even available here -- layers 0-2 see every token,
#              which alone forces an average of ~135 on this data, so a target of
#              128 has no solution and clamps to a single token.
#   SparseVLM  RETAIN_TOKN is a layer average in the released code: its own
#              comment prices the 192 schedule as `2*576 4*300 10*200 16*110`,
#              which averages 191 over 32 layers. At K=128 on ~1427 voxel tokens
#              that is ~15 tokens at the end, and the same FLOPs per layer as a
#              vision-path method keeping 128 everywhere.
#
# Both reproduce the 3DZip table (FastV 50.90 vs 50.9, SparseVLM 50.7 vs 50.3),
# and a comparison table has to say which quantity each row reports.
if [ "${LABEL}" == "sparsevlm" ]; then
    export LLM_PRUNE_BUDGET=${LLM_PRUNE_BUDGET:-average}
else
    export LLM_PRUNE_BUDGET=${LLM_PRUNE_BUDGET:-retained}
fi
export LLM_PRUNE_DEBUG=${LLM_PRUNE_DEBUG:-0}
export FASTV_LAYER=${FASTV_LAYER:-2}
# 'fixed_budget' solves upstream's stage profile against the requested budget,
# which is what the 3DZip comparison row needs. 'official_v1' instead runs
# upstream V1's absolute sparse_token_list as written and reports whatever token
# count it lands on -- on LLaVA-3D that is ~39 effective tokens at RETAIN_TOKN
# 128, because the table was tuned against 576 image tokens. See
# docs/sparsevlm_v1_llava3d_audit.md.
export SPARSEVLM_MODE=${SPARSEVLM_MODE:-fixed_budget}
export SPARSEVLM_LAYERS=${SPARSEVLM_LAYERS:-2,6,15}
export SPARSEVLM_RECYCLE=${SPARSEVLM_RECYCLE:-1}
# Paper 3.3 reconstructs a cluster by element-wise SUM; upstream's released
# code divides by the cluster size. 'sum' follows the paper.
export SPARSEVLM_RECONSTRUCT=${SPARSEVLM_RECONSTRUCT:-mean}
# V1 truncates position_ids after a prune (survivors renumbered 0..L-1);
# keeping the original ids is what V2 added as "idea1".
export SPARSEVLM_POSITION_IDS=${SPARSEVLM_POSITION_IDS:-renumber}

export POOLING
export VOXEL_SIZE
export ADAPTIVE_RATIO=${ADAPTIVE_RATIO_OVERRIDE:-${TARGET_TOKENS}}
# Where every compression method is applied: `voxel` = after LLaVA-3D's
# voxelization pooling, which is the 3DZip paper's comparison setting;
# `patch` = on the raw CLIP patch grid, the ablation.
export COMPRESS_UNITS=${COMPRESS_UNITS:-voxel}
# How a voxel inherits the CLS attention of the patches inside it. `mean`
# over-selects thinly-observed voxels, so `sum` (total attention mass) is
# the default; see _compression_inputs in video_encoder.py.
export COMPRESS_ATTN_POOL=${COMPRESS_ATTN_POOL:-sum}
export MERGE_CUTOFF=${MERGE_CUTOFF:-5}
export DTC_SEED=${DTC_SEED:-0}
export DTC_DEBUG=${DTC_DEBUG:-0}
export DTC_EXACT_BUDGET=${DTC_EXACT_BUDGET:-1}
export VISPRUNER_TOKENS=${VISPRUNER_TOKENS:-${TARGET_TOKENS}}
export VISPRUNER_IMPORTANT_RATIO=${VISPRUNER_IMPORTANT_RATIO:-0.5}
export VISPRUNER_DEBUG=${VISPRUNER_DEBUG:-0}
export DIVPRUNE_TOKENS=${DIVPRUNE_TOKENS:-${TARGET_TOKENS}}
# The reproduction spec's names; DIVPRUNE_SPACE keeps working as an alias.
#   post_projector = 'projected' (DivPrune upstream's own measurement point)
#   pre_projector  = 'geo'       (where 3DZip's own DPP selection runs)
export DIVPRUNE_FEATURE_SPACE=${DIVPRUNE_FEATURE_SPACE:-${DIVPRUNE_SPACE:-post_projector}}
export DIVPRUNE_SPACE=${DIVPRUNE_SPACE:-projected}
# fp16 reproduces upstream's arithmetic, where 1-cos_sim quantises hard and the
# greedy argmax has to break ties float32 would have resolved. Ablation only.
export DIVPRUNE_PRECISION=${DIVPRUNE_PRECISION:-fp32}
# Upstream restores the input sequence order (keep_indexs.sort().values);
# 'selection' keeps the MMDP pick order and is an audit knob only.
export DIVPRUNE_ORDER=${DIVPRUNE_ORDER:-original}
export DIVPRUNE_DEBUG=${DIVPRUNE_DEBUG:-0}
export SEGPRUNER_TOKENS=${SEGPRUNER_TOKENS:-${TARGET_TOKENS}}
export SEGPRUNER_IMPORTANT_RATIO=${SEGPRUNER_IMPORTANT_RATIO:-0.5}
export SEGPRUNER_LAMBDA=${SEGPRUNER_LAMBDA:-0.5}
export SEGPRUNER_DEBUG=${SEGPRUNER_DEBUG:-0}
export VISIONZIP_TOKENS=${VISIONZIP_TOKENS:-${TARGET_TOKENS}}
export VISIONZIP_CONTEXTUAL_RATIO=${VISIONZIP_CONTEXTUAL_RATIO:-0.15625}
export VISIONZIP_MERGE_SPACE=${VISIONZIP_MERGE_SPACE:-clip}
export VISIONZIP_DEBUG=${VISIONZIP_DEBUG:-0}
export TOSA_TOKENS=${TOSA_TOKENS:-${TARGET_TOKENS}}
export TOSA_ALPHA=${TOSA_ALPHA:-0.5}
export TOSA_ALPHA_STYLE=${TOSA_ALPHA_STYLE:-uniform}
export TOSA_PE_BINS=${TOSA_PE_BINS:-27}
export TOSA_PE_CHANNELS=${TOSA_PE_CHANNELS:-24}
export TOSA_SCHEDULE=${TOSA_SCHEDULE:-uniform}
export TOSA_ROUNDS=${TOSA_ROUNDS:-26}
export TOSA_DEBUG=${TOSA_DEBUG:-0}
# The Fig.4 selection export is debug-only. Default it off here so a stray
# `export TOKEN_SEL=1` in the shell cannot turn a full benchmark run into a
# per-scene record dump.
export TOKEN_SEL=${TOKEN_SEL:-0}
export TOKEN_SEL_DIR=${TOKEN_SEL_DIR:-pred/token_selection}
export EVAL_VISUALIZE=${EVAL_VISUALIZE:-0}

VISUALIZE_ARGS=()
if [ "${EVAL_VISUALIZE}" == "1" ]; then
    VISUALIZE_ARGS+=(--visualize)
fi

# Optional subset switches, for smoke-testing a method without walking the
# whole split. EVAL_SCENE takes one scene ID or a comma-separated list;
# EVAL_LIMIT caps the question count. Both unset = the full benchmark.
SUBSET_ARGS=()
if [ -n "${EVAL_SCENE}" ]; then
    SUBSET_ARGS+=(--scene "${EVAL_SCENE}")
fi
if [ -n "${EVAL_LIMIT}" ] && [ "${EVAL_LIMIT}" -gt 0 ]; then
    SUBSET_ARGS+=(--limit "${EVAL_LIMIT}")
fi

mkdir -p pred results

# A subset run must never overwrite a full-split prediction file.
SUBSET_SUFFIX=""
if [ ${#SUBSET_ARGS[@]} -gt 0 ]; then
    SUBSET_SUFFIX="_subset"
fi
# EVAL_QUESTION_FILE runs a fixed question subset (e.g. playground/data/annotations/
# subsets/<dataset>_sub250.json); its file stem goes into the prediction name.
if [ -n "${EVAL_QUESTION_FILE}" ]; then
    SUBSET_SUFFIX="${SUBSET_SUFFIX}_$(basename "${EVAL_QUESTION_FILE}" .json)"
fi
# PRED_SUFFIX tags one variant of a method so an ablation sweep can run several
# settings of the same POOLING at once without them overwriting each other.
SUBSET_SUFFIX="${SUBSET_SUFFIX}${PRED_SUFFIX:-}"


# A non-default ablation knob gets its own prediction file. Two runs of the same
# method and K that differ only in an env var used to write to the same path --
# and if they overlap, both files are silently interleaved and ruined.
VARIANT=""
if [ "${COMPRESS_UNITS}" != "voxel" ]; then VARIANT="${VARIANT}_${COMPRESS_UNITS}units"; fi
case "${LABEL}" in
    sparsevlm)
        [ "${SPARSEVLM_MODE}" != "fixed_budget" ] && VARIANT="${VARIANT}_${SPARSEVLM_MODE}"
        [ "${LLM_PRUNE_BUDGET}" != "average" ] && VARIANT="${VARIANT}_${LLM_PRUNE_BUDGET}"
        [ "${SPARSEVLM_RECONSTRUCT}" != "mean" ] && VARIANT="${VARIANT}_${SPARSEVLM_RECONSTRUCT}"
        [ "${SPARSEVLM_POSITION_IDS}" != "renumber" ] && VARIANT="${VARIANT}_pos${SPARSEVLM_POSITION_IDS}"
        ;;
    fastv)
        [ "${LLM_PRUNE_BUDGET}" != "retained" ] && VARIANT="${VARIANT}_${LLM_PRUNE_BUDGET}"
        ;;
    divprune)
        case "${DIVPRUNE_FEATURE_SPACE}" in
            post_projector|projected) ;;
            *) VARIANT="${VARIANT}_${DIVPRUNE_FEATURE_SPACE}" ;;
        esac
        [ "${DIVPRUNE_PRECISION}" != "fp32" ] && VARIANT="${VARIANT}_${DIVPRUNE_PRECISION}"
        [ "${DIVPRUNE_ORDER}" != "original" ] && VARIANT="${VARIANT}_order${DIVPRUNE_ORDER}"
        ;;
esac

if [ "${TARGET_TOKENS}" -gt 0 ]; then
    PRED_FILE=./pred/scanqa_${LABEL}_vs${VOXEL_SIZE}_t${TARGET_TOKENS}${VARIANT}${SUBSET_SUFFIX}.json
else
    PRED_FILE=./pred/scanqa_${LABEL}_vs${VOXEL_SIZE}${VARIANT}${SUBSET_SUFFIX}.json
fi

# Refuse to start on top of a file another run is still writing.
if [ -e "${PRED_FILE}" ] && command -v fuser >/dev/null 2>&1 && fuser "${PRED_FILE}" >/dev/null 2>&1; then
    echo "ERROR: ${PRED_FILE} is open by another process -- a second run would interleave" >&2
    echo "       both outputs. Wait for it to finish, or change the variant." >&2
    exit 1
fi
echo "=== Predictions: ${PRED_FILE} ==="

GT_FILE=playground/data/annotations/llava3d_scanqa_val_answer.json
QUESTION_FILE=${EVAL_QUESTION_FILE:-playground/data/annotations/llava-3d-scanqa_val_question.json}

echo "=== ScanQA: POOLING=${POOLING}, COMPRESS_UNITS=${COMPRESS_UNITS}, COMPRESS_ATTN_POOL=${COMPRESS_ATTN_POOL}, VOXEL_SIZE=${VOXEL_SIZE}, TARGET_TOKENS=${TARGET_TOKENS}, DTC_SEED=${DTC_SEED}, DTC_DEBUG=${DTC_DEBUG}, DTC_EXACT_BUDGET=${DTC_EXACT_BUDGET}, VISPRUNER_TOKENS=${VISPRUNER_TOKENS}, VISPRUNER_IMPORTANT_RATIO=${VISPRUNER_IMPORTANT_RATIO}, VISPRUNER_DEBUG=${VISPRUNER_DEBUG}, SEGPRUNER_IMPORTANT_RATIO=${SEGPRUNER_IMPORTANT_RATIO}, SEGPRUNER_LAMBDA=${SEGPRUNER_LAMBDA}, DIVPRUNE_SPACE=${DIVPRUNE_SPACE}, DIVPRUNE_PRECISION=${DIVPRUNE_PRECISION}, VISIONZIP_CONTEXTUAL_RATIO=${VISIONZIP_CONTEXTUAL_RATIO}, VISIONZIP_MERGE_SPACE=${VISIONZIP_MERGE_SPACE}, TOSA_ALPHA=${TOSA_ALPHA}, TOSA_ALPHA_STYLE=${TOSA_ALPHA_STYLE}, LLM_PRUNE=${LLM_PRUNE}, LLM_PRUNE_TOKENS=${LLM_PRUNE_TOKENS}, LLM_PRUNE_BUDGET=${LLM_PRUNE_BUDGET}, SPARSEVLM_MODE=${SPARSEVLM_MODE}, SPARSEVLM_RECONSTRUCT=${SPARSEVLM_RECONSTRUCT}, TOKEN_SEL=${TOKEN_SEL} ==="
echo "=== Visualization: EVAL_VISUALIZE=${EVAL_VISUALIZE}, TOKEN_SEL_DIR=${TOKEN_SEL_DIR} ==="
if [ -n "${EVAL_QUESTION_FILE}" ]; then
    echo "=== Question subset: ${QUESTION_FILE} ==="
fi
if [ ${#SUBSET_ARGS[@]} -gt 0 ]; then
    echo "=== Subset: EVAL_SCENE=${EVAL_SCENE:-<all>}, EVAL_LIMIT=${EVAL_LIMIT:-<all>} ==="
fi

python llava/eval/model_scanqa.py \
    --model-path ChaimZhu/LLaVA-3D-7B \
    --question-file ${QUESTION_FILE} \
    --answers-file ${PRED_FILE} \
    "${VISUALIZE_ARGS[@]}" \
    "${SUBSET_ARGS[@]}"

#EVAL_VISUALIZE=1

python llava/eval/scanqa_evaluator.py \
    --pred-file ${PRED_FILE} \
    --gt-file ${GT_FILE}
