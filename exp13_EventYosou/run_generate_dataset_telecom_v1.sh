#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Accept OPEN_AI_KEY as an alias for OPENAI_API_KEY
if [[ -z "${OPENAI_API_KEY:-}" && -n "${OPEN_AI_KEY:-}" ]]; then
  export OPENAI_API_KEY="${OPEN_AI_KEY}"
fi

PROMPT_PATH="${REPO_ROOT}/seimei/exp13_EventYosou/generate_validate/data_generators_yosou/excel_events_small.md"
TOPICS_PATH="${REPO_ROOT}/seimei/exp13_EventYosou/generate_validate/data_generators_yosou/excel_topics_3.json"
# Using a small subset for quick debug runs.
EXP_DIR="${REPO_ROOT}/exp13_EventYosou"
# Iteration number
ITER=${ITER:-5}

python "${REPO_ROOT}/exp13_EventYosou/generate_validate/generate_dataset_excel_telecom_v1.py" \
  --prompt-path "${PROMPT_PATH}" \
  --topics-path "${TOPICS_PATH}" \
  --exp-dir "${EXP_DIR}" \
  --n-hyper-params 1 \
  --batch-size 3 \
  --max-attempts 7 \
  --llm-kv timeout=300 \
  --llm-kv usage_log_path=exp13_EventYosou/usage_log_iter${ITER}.csv \
  --model "gpt-5-mini" 