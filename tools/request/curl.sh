#!/usr/bin/env bash

set -ou pipefail

curl -X POST http://localhost:8889/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "who are you"}]}' | jq .
