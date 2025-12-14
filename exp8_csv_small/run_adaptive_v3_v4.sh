#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this script so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run v3 adaptive evaluation
python "${SCRIPT_DIR}/train_v3_eval_adaptive.py" \
  --dataset-path "${SCRIPT_DIR}/dataset.json" \
  --output-path "${SCRIPT_DIR}/train_v3_eval_adaptive.json"

# Run v4 adaptive evaluation
python "${SCRIPT_DIR}/train_v4_eval_adaptive.py" \
  --dataset-path "${SCRIPT_DIR}/dataset.json" \
  --output-path "${SCRIPT_DIR}/train_v4_eval_adaptive.json"
