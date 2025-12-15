#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this script so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run v4 adaptive evaluation version 1
python "${SCRIPT_DIR}/train_v4_eval_adaptive.py" \
  --dataset-path "${SCRIPT_DIR}/dataset.json" \
  --output-path "${SCRIPT_DIR}/train_v4_eval_adaptive_2run.json"

# Run v4 adaptive evaluation version 2
python "${SCRIPT_DIR}/train_v4_eval2_adaptive.py" \
  --dataset-path "${SCRIPT_DIR}/dataset.json" \
  --output-path "${SCRIPT_DIR}/train_v4_eval2_adaptive.json"

# Run v4 adaptive evaluation version 3 for the first run of eval_adaptive.py
python "${SCRIPT_DIR}/train_v4_eval3_adaptive.py" \
  --dataset-path "${SCRIPT_DIR}/dataset.json" \
  --input-path "${SCRIPT_DIR}/train_v4_eval_adaptive.json" \
  --output-path "${SCRIPT_DIR}/train_v4_eval_adaptive_analysis.json"
