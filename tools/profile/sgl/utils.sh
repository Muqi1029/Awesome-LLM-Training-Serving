#!/usr/bin/env bash

set -x

curl -X POST ${BASE_URL}/start_profile \
    -H "Content-Type: application/json" \
    -H "Authorization: ${API_KEY}" \
    -d '{
        "output_dir": "/tmp/sgl_055"
    }'

curl -X POST ${BASE_URL}/stop_profile \
    -H "Authorization: ${API_KEY}"
