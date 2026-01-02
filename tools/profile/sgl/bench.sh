#!/usr/bin/env bash

# export SGLANG_TORCH_PROFILER_DIR="/sgl-workspace"
    # --tokenizer deepseek-ai/DeepSeek-R1-0528 \
    # --model deepseek-ai/DeepSeek-R1-0528 \
    #
curl -X GET ${BASE_URL}/flush_cache \
    -H "Authorization: ${API_KEY}"


MODEL_PATH=deepseek-ai/DeepSeek-V3-0324

python -m sglang.bench_serving \
    --backend sglang-oai \
    --dataset-name random \
    --num-prompts 1 \
    --random-input 2000 \
    --random-output 1500 \
    --random-range-ratio 1 \
    --tokenizer ${MODEL_PATH} \
    --model ${MODEL_PATH} \
    --base-url ${BASE_URL} \
    --dataset-path ${SHAREGPT_DATAPATH} \
    --profile
