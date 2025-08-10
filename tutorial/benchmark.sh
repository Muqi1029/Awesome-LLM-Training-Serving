#!/usr/bin/env bash

if [ "$#" -eq 1 ]; then
      port="$1"
      echo "using port ${port}"
elif [ "$#" -eq 0 ]; then
      echo "Usage: ./run_server.sh <port>, now using 8888 as port by default"
      port=8888
fi

host=127.0.0.1
max_bs=32
input_len=1200
output_len=800
selected_request_rate={4..64..4}
model_path=${QWEN332BFP8}
output_dir="exp32b"
mkdir -p ${output_dir}

echo "Luanch server: ${host}:${port}"

tp_lists=(1 2 4 8)

for tp in "${tp_lists[@]}"; do
      # run server in the background
      python -m sglang.launch_server \
            --host ${host} \
            --model-path ${model_path} \
            --port ${port} \
            --tp-size ${tp} \
            --enable-metrics \
            --log-requests \
            --log-requests-level 2 \
            --cuda-graph-max-bs ${max_bs} \
            --reasoning-parser qwen3 \
            --tool-call-parser qwen25 \
            --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
            --tokenizer-mode auto \
            --enable-torch-compile \
            --torch-compile-max-bs ${max_bs} \
            --kv-cache-dtype fp8_e4m3 \
            --disable-radix-cache &

      server_pid=$!

      echo "Waiting for server (PID $server_pid) to be ready..."
      for i in {1..300}; do
            status_code=$(curl -s -o /dev/null -w "%{http_code}" \
                  "http://${host}:${port}/v1/models" \
                  -H "Authorization: Bear None")

            if [[ "$status_code" == "200" ]]; then
                  echo "Server ready with status 200"
                  break
            fi
            sleep 1
      done

      if [[ "$status_code" != "200" ]]; then
            echo "Server failed to start in time, killing PID $server_pid"
            kill "$server_pid"
            wait "$server_pid" 2>/dev/null
            continue
      fi



      # ##############
      # run clients
      # #############
      
      # warmp
      # echo "Warm up the engine..."
      # python -m sglang.bench_serving \
      #       --port ${port} \
      #       --backend sglang-oai \
      #       --dataset-path ~/muqi/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
      #       --dataset-name random \
      #       --random-range-ratio 1 \
      #       --random-input-len ${input_len} \
      #       --random-output-len ${output_len} \
      #       --request-rate 32 \
      #       --max-concurrency 32 \
      #       --num-prompt $((32 * 5)) \
      #       --output-file "warm_up.jsonl"

      # echo "Finish warmup"

      # # formal test
      # echo "Starting formal client requests"
      # for request_rate in "${selected_request_rate[@]}"; do
      #       echo "Benchmarking request_rate: ${request_rate}"
      #       # client 
      #       python -m sglang.bench_serving \
      #             --port ${port} \
      #             --backend sglang-oai \
      #             --dataset-path ~/muqi/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
      #             --dataset-name random \
      #             --random-range-ratio 1 \
      #             --random-input-len ${input_len} \
      #             --random-output-len ${output_len} \
      #             --request-rate ${request_rate} \
      #             --max-concurrency ${request_rate} \
      #             --num-prompt $((request_rate * 10)) \
      #             --output-file "${output_dir}/${tp}_sglang-oai_input${input_len}_output${output_len}.jsonl"
      # done

      echo "Kill the server..."
      kill -9 $server_pid
done
