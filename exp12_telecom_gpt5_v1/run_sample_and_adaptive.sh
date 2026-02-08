#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this script so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure local seimei package is importable (tolerate unset PYTHONPATH)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "Running eval v4 sample 2 script..."
LOG_FILE="${SCRIPT_DIR}/server_trainv4.log"

nohup python3 "${SCRIPT_DIR}/train_v4_eval_sample2.py" \
  --dataset-path "${SCRIPT_DIR}/dataset_10.json" \
  --output-path "${SCRIPT_DIR}/train_v4_eval_sample2_results_telecom_GPT5.json" \
  > "${LOG_FILE}" 2>&1 &

echo "Started in background (PID $!). Logs: ${LOG_FILE}"


# Baseline sample eval
# python "${SCRIPT_DIR}/train_v3_eval_sample.py" \
#   --dataset-path "${SCRIPT_DIR}/dataset.json" \
#   --output-path "${SCRIPT_DIR}/train_v3_eval_sample_results_baseline_GPT4.json"


# echo "Running eval sample adaptive script..."
# Adaptive sample eval
# python "${SCRIPT_DIR}/train_v3_eval_sample_adaptive.py" \
#   --dataset-path "${SCRIPT_DIR}/dataset.json" \
#   --output-path "${SCRIPT_DIR}/train_v3_eval_sample_results41_adaptive.json"
