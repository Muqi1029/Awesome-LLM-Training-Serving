#!/usr/bin/env bash

set -euo pipefail

# args
port=8888
host=127.0.0.1

random_input_len=1200
random_output_len=800

export CUDA_VISIBLE_DEVICES=7

###################
## Starting Server
echo "Starting server on ${host}:${port}..."
python -m sglang.launch_server --model-path "$QWEN306B" --host "${host}" --port "${port}" \
	> server.log 2>&1 &

server_pid=$!

cleanup() {
	if kill -0 "$server_pid" 2>/dev/null; then
		echo "Cleaning up: Killing server (PID $server_pid)..."
		kill -9 "$server_pid" 2>/dev/null
		echo "Server killed."
	else
		echo "Server process $server_pid not found, nothing to kill."
	fi
}

# trap signals for killing all process group
trap cleanup EXIT SIGINT SIGTERM

echo "Waiting for server (PID $server_pid) to be ready..."

for i in {1..300}; do
	if [[ "$(curl -s -o /dev/null -w "%{http_code}" \
		"http://${host}:${port}/v1/models" \
		-H "Authorization: Bear None")" == "200" ]]; then
			echo "Server ready (status 200)"
			break
	fi
	sleep 1
done


# check server
if ! kill -0 "$server_pid" 2>/dev/null; then
	echo "Server failed to start, exiting."
	exit 1
fi

echo "Running Python tuning script..."
python tune.py --host "${host}" \
	--port "${port}" \
	--left 5 \
	--right 6 \
	--mode general \
	--random-input-len ${random_input_len} \
	--random-output-len ${random_output_len} \
	--output-dir test1

echo "Python tuning script finished."
