#!/usr/bin/env bash

set -exo pipefail

BASE_URL=${BASE_URL-"localhost"}
PORT=${PORT:-8888}
API_KEY=${API_KEY-"Just Keep Me"}


curl -X POST http://${BASE_URL}:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d '{"messages": [{"role": "user", "content": "who are you"}]}' | jq .
