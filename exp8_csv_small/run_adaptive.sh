#!/usr/bin/env bash
# Run adaptive training and evaluation for exp8_csv_small.

set -euo pipefail

ENV_NAME="kyotoai"
# Resolve paths relative to this script so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNS_DIR="${REPO_ROOT}/seimei_runs"
EXP_DIR="${SCRIPT_DIR}"
DATASET_PATH="${EXP_DIR}/dataset.json"
TRAIN_OUTPUT="${EXP_DIR}/train_v3_dpo_3_TEST_adaptive.json"
EVAL_OUTPUT="${EXP_DIR}/train_v3_eval_adaptive.json"

echo "Activating conda environment: ${ENV_NAME}"
# Prefer explicit miniconda path; fall back to conda on PATH.
if [ -f "/Users/juan/miniconda3/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "/Users/juan/miniconda3/bin/activate"
  conda activate "${ENV_NAME}"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/bin/activate"
  conda activate "${ENV_NAME}"
else
  echo "conda not found; ensure ${ENV_NAME} is active before running." >&2
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY not set. Please paste it now (input hidden):"
  # shellcheck disable=SC2162
  read -s OPENAI_API_KEY
  export OPENAI_API_KEY
  echo
fi

echo "Running adaptive training..."
python "${EXP_DIR}/train_v3_TEST.py" \
  --runs-dir "${RUNS_DIR}" \
  --exp-dir "${EXP_DIR}" \
  --result-output-path "${TRAIN_OUTPUT}"

echo "Running evaluation..."
python "${EXP_DIR}/train_v3_eval.py" \
  --dataset-path "${DATASET_PATH}" \
  --output-path "${EVAL_OUTPUT}"

echo "Done. Outputs:"
echo "  Training tracker: ${TRAIN_OUTPUT}"
echo "  Evaluation log:   ${EVAL_OUTPUT}"

# chmod +x exp8_csv_small/run_adaptive.sh   # once
# exp8_csv_small/run_adaptive.sh
