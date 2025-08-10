#!/usr/bin/env bash
set -euo pipefail

# 捕获 Ctrl+C / kill，杀掉整个进程组
trap "echo 'Caught signal, cleaning up...'; kill 0" SIGINT SIGTERM

# ---------------------
# 参数解析
# ---------------------
port=${1:-8888}
echo "Using port: ${port}"

host="127.0.0.1"
max_bs=32
input_len=1200
output_len=800
selected_request_rate=({4..64..4})
model_path="${QWEN332BFP8:?QWEN332BFP8 environment variable not set}"
output_dir="exp32b"
mkdir -p "${output_dir}"

tp_lists=(1 2 4 8)

# ---------------------
# 启动与测试循环
# ---------------------
for tp in "${tp_lists[@]}"; do
    echo "Launching server: ${host}:${port} with tp-size=${tp}"
    python -m sglang.launch_server \
        --host "${host}" \
        --model-path "${model_path}" \
        --port "${port}" \
        --tp-size "${tp}" \
        --enable-metrics \
        --log-requests \
        --log-requests-level 2 \
        --cuda-graph-max-bs "${max_bs}" \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen25 \
        --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
        --tokenizer-mode auto \
        --enable-torch-compile \
        --torch-compile-max-bs "${max_bs}" \
        --kv-cache-dtype fp8_e4m3 \
        --disable-radix-cache &

    server_pid=$!

    # 等待 server 就绪
    echo "Waiting for server (PID $server_pid) to be ready..."
    for _ in {1..300}; do
        if [[ "$(curl -s -o /dev/null -w "%{http_code}" \
            "http://${host}:${port}/v1/models" \
            -H "Authorization: Bear None")" == "200" ]]; then
            echo "Server ready (status 200)"
            break
        fi
        sleep 1
    done

    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "Server failed to start, skipping tp=${tp}"
        continue
    fi

    # warmup（可选）
    if [[ "${WARMUP:-1}" -eq 1 ]]; then
        echo "Warming up..."
        python -m sglang.bench_serving \
            --port "${port}" \
            --backend sglang-oai \
            --dataset-path ~/muqi/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
            --dataset-name random \
            --random-range-ratio 1 \
            --random-input-len "${input_len}" \
            --random-output-len "${output_len}" \
            --request-rate 32 \
            --max-concurrency 32 \
            --num-prompt $((32 * 5)) \
            --output-file "warm_up.jsonl"
        echo "Warmup done."
    fi

    # benchmark（可选）
    if [[ "${BENCHMARK:-1}" -eq 1 ]]; then
        for request_rate in "${selected_request_rate[@]}"; do
            echo "Benchmarking request_rate=${request_rate}"
            python -m sglang.bench_serving \
                --port "${port}" \
                --backend sglang-oai \
                --dataset-path ~/muqi/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
                --dataset-name random \
                --random-range-ratio 1 \
                --random-input-len "${input_len}" \
                --random-output-len "${output_len}" \
                --request-rate "${request_rate}" \
                --max-concurrency "${request_rate}" \
                --num-prompt $((request_rate * 10)) \
                --output-file "${output_dir}/${tp}_input${input_len}_output${output_len}_rate${request_rate}.jsonl"
        done
    fi

    echo "Killing server PID ${server_pid}..."
    kill -9 "$server_pid"
done
