#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Accept OPEN_AI_KEY as an alias for OPENAI_API_KEY
if [[ -z "${OPENAI_API_KEY:-}" && -n "${OPEN_AI_KEY:-}" ]]; then
  export OPENAI_API_KEY="${OPEN_AI_KEY}"
fi

PROMPT_PATH="${REPO_ROOT}/seimei/eval/data_generators_yosou/excel_events_telecom_v1.md"
TOPICS_PATH="${REPO_ROOT}/seimei/eval/data_generators_yosou/excel_topics_3.json"
# Using a small subset for quick debug runs.
EXP_DIR="${REPO_ROOT}/exp13_EventYosou"

python "${REPO_ROOT}/seimei/eval/generate_dataset_excel_telecom_v1.py" \
  --prompt-path "${PROMPT_PATH}" \
  --topics-path "${TOPICS_PATH}" \
  --exp-dir "${EXP_DIR}" \
  --n-hyper-params 1 \
  --batch-size 1 \
  --max-attempts 10 \
  --llm-kv timeout=300 \
  --model "gpt-5-nano"