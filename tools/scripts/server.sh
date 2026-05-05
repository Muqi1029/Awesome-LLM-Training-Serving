#!/usr/bin/env bash

set -exo pipefail

# args
MODEL_PATH=${MODEL_PATH:-${QWEN3_06B}}
PORT=${PORT:-8888}

DISABLE_LOG=${DISABLE_LOG:-0}
DISABLE_FP8_KVCACHE=${DISABLE_FP8_KVCACHE:-0}

TP=${TP:-1}

# defer mode type
if [[ -z "${MODEL_TYPE}" ]]; then
    MODEL_NAME=$(basename $(dirname ${MODEL_PATH}))
    IFS='-' read -r MODEL_TYPE NUM_WEIGHTS _ <<< "$MODEL_NAME"

    NUM_WEIGHTS=${NUM_WEIGHTS%B}

    if [[ $((NUM_WEIGHTS > 300)) == 1 ]]; then
        TP=8
    elif [[ $((NUM_WEIGHTS >= 30)) == 1 ]]; then
        TP=2
    fi
fi

SERVER_LAUNCH="sglang serve \
    --model-path ${MODEL_PATH} \
    --port ${PORT} \
    --enable-metrics \
    --tp-size ${TP}"

if [[ "${DISABLE_LOG}" == 0 ]]; then
    SERVER_LAUNCH="${SERVER_LAUNCH} --log-requests --log-requests-level 3"
fi

if [[ "$MODEL_PATH" == *FP8* ]]; then
    if [[ "${DISABLE_FP8_KVCACHE}" != '1'  ]]; then
        SERVER_LAUNCH="${SERVER_LAUNCH} --kv-cache-dtype fp8_e4m3"
    fi
fi

case "$MODEL_TYPE" in
    "Qwen3.5")
        SERVER_LAUNCH="${SERVER_LAUNCH} --reasoning-parser qwen3 --tool-call-parser qwen3_coder"
        ;;
    "Qwen3")
        SERVER_LAUNCH="${SERVER_LAUNCH} --reasoning-parser qwen3 --tool-call-parser qwen"
        ;;
    *)
        echo "Unknown MODEL_TYPE: ${MODEL_TYPE}" >&2
        exit 1
        ;;
esac

${SERVER_LAUNCH}
