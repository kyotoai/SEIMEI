"""Adaptive wrapper for train_v3_eval.py with built-in knowledge pool and output path."""

import asyncio
from typing import Any, Dict, List, Optional

import train_v3_eval as base

# Redirect output to adaptive file
base.DEFAULT_RESULT_PATH = base.EXP_DIR / "train_v3_eval_adaptive.json"

# Adaptive knowledge hint inspired by train_v3_TEST.py’s _build_adaptive_notes
ADAPTIVE_HINT = {
    "id": "adaptive_pivot",
    "agent": "think",
    "step": None,
    "text": (
        "If scores stall/regress: pivot to a targeted check, surface missing clues/feedback, "
        "and preserve correct computations/parameter mappings (e.g., enforce total_rows = sensor_count × 24 "
        "and use canonical params)."
    ),
    "tags": ["adaptive", "stability", "mapping"],
}

# Optional: add a small reusable pool (you can extend this list as needed)
DEFAULT_KNOWLEDGE_POOL: List[Dict[str, Any]] = [ADAPTIVE_HINT]


def build_knowledge_config(manual_entries: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Enable knowledge generation and seed with adaptive hints."""
    cfg: Dict[str, Any] = {
        "generate_knowledge": True,
        "knowledge_pool": DEFAULT_KNOWLEDGE_POOL,
    }
    if manual_entries:
        cfg["knowledge"] = manual_entries
    return cfg


# Monkey-patch the base module to use the adaptive config
base.build_knowledge_config = build_knowledge_config


if __name__ == "__main__":
    asyncio.run(base.run_evaluation(base.parse_args()))
