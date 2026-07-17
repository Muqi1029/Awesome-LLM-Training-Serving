#!/usr/bin/env bash
set -euo pipefail

: "${BASE_URL:?BASE_URL is required}"
: "${API_KEY:?API_KEY is required}"

curl -X POST "${BASE_URL}/start_profile" \
    -H "Content-Type: application/json" \
    -H "Authorization: ${API_KEY}" \
    -d '{
        "output_dir": "/tmp/sgl_profile"
    }'

curl -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: ${API_KEY}" \
    -d '{
        "messages": [{"role": "user", "content": "who are you"}],
        "max_tokens": 1000,
        "ignore_eos": true
    }'

curl -X POST "${BASE_URL}/stop_profile" \
    -H "Authorization: ${API_KEY}"
