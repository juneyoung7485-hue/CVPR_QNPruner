# Sourced by eval_{scanqa,sqa3d,openeqa}.sh with their arguments: <method> [voxel_size] [K].
# Sets POOLING / VOXEL_SIZE / TARGET_TOKENS for the video tower, LABEL / SUFFIX for the prediction file name and
# LIMIT_ARGS for an optional EVAL_LIMIT subset. Imports this tree's llava/ and qnpruner/.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
METHOD=${1:?usage: <qnpruner|native> [voxel_size] [K]}
export VOXEL_SIZE=${2:-0.2}
export TARGET_TOKENS=${3:-0}
case "${METHOD}" in
    native)   export POOLING=voxelize; TARGET_TOKENS=0; LABEL=native_vs${VOXEL_SIZE} ;;
    qnpruner) export POOLING=qnpruner
              if [ "${TARGET_TOKENS}" -le 0 ]; then echo "qnpruner needs K > 0 (third argument)" >&2; return 1; fi
              LABEL=qnpruner_vs${VOXEL_SIZE}_t${TARGET_TOKENS} ;;
    *) echo "Unknown method '${METHOD}'. Use: qnpruner | native" >&2; return 1 ;;
esac
LIMIT_ARGS=()
SUFFIX=${PRED_SUFFIX:-}
if [ -n "${EVAL_LIMIT}" ] && [ "${EVAL_LIMIT}" -gt 0 ]; then
    LIMIT_ARGS=(--limit "${EVAL_LIMIT}"); SUFFIX="${SUFFIX}_subset${EVAL_LIMIT}"
fi
mkdir -p pred
