#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASE_URL:-}" ]]; then
	echo "Usage: BASE_URL=http://localhost:8000 bash $0" >&2
	exit 1
fi

EVAL_NAMES=(gsm8k aime25 mmlu)

REPEAT=${REPEAT:-1}
MAX_TOKENS=${MAX_TOKENS:-10240}
NUM_EXAMPLES=${NUM_EXAMPLES:-100}

for eval_name in "${EVAL_NAMES[@]}"; do
	echo "====== RUN EVAL ${eval_name} ========"
	python -m sglang.test.run_eval \
		--eval-name "$eval_name" \
		--base-url "$BASE_URL" \
		--repeat ${REPEAT} \
		--max-tokens ${MAX_TOKENS} \
		--num-examples ${NUM_EXAMPLES}
done
