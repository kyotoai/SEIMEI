"""Adaptive wrapper for train_v4_eval.py with added adaptive hint and output path override."""

import asyncio
from typing import Any, Dict, List, Optional

import train_v4_eval as base

# Redirect output to adaptive file
base.DEFAULT_RESULT_PATH = base.EXP_DIR / "train_v4_eval_adaptive.json"

# Adaptive knowledge hint inspired by train_v3_TEST.py’s adaptive notes
ADAPTIVE_HINT = {
    "id": "adaptive_pivot",
    "agent": "think",
    "step": None,
    "text": (
        "If scores stall/regress: pivot to a targeted verification, surface missing clues/feedback, "
        "and preserve correct computations/parameter mappings (e.g., enforce total_rows = sensor_count × 24 "
        "and use canonical params)."
    ),
    "tags": ["adaptive", "stability", "mapping"],
}

# Extend the existing pool with the adaptive hint
DEFAULT_KNOWLEDGE_POOL: List[Dict[str, Any]] = list(getattr(base, "DEFAULT_KNOWLEDGE_POOL", [])) + [
    ADAPTIVE_HINT
]


def build_knowledge_config(manual_entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "generate_knowledge": True,
        "knowledge_pool": DEFAULT_KNOWLEDGE_POOL,
    }
    if manual_entries:
        cfg["knowledge"] = manual_entries
    return cfg


# Monkey-patch the base module to use the adaptive config
base.DEFAULT_KNOWLEDGE_POOL = DEFAULT_KNOWLEDGE_POOL
base.build_knowledge_config = build_knowledge_config


if __name__ == "__main__":
    asyncio.run(base.run_evaluation(base.parse_args()))
