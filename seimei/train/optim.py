"""KlgOptimizer: knowledge pool optimization via LLM-based assessment."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional


def KlgOptimizer(
    *,
    dataset: List[Dict[str, Any]],
    optimizer_type: str = "seimei_v1",
    n_sample: int = 1,
    n_epoch: int = 1,
    n_new_klg_per_epoch: int = 3,
    update_klg_threshold: float = 0.1,
    metric: str = "answer_exact_match",
    load_knowledge_path: str,
    save_knowledge_path: str,
    cache_path: str = "cache.json",
    workspace_root: Optional[str] = None,
    patch_dir: Optional[str] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Optimize a knowledge pool for seimei using an LLM-based optimizer.

    Args:
        dataset: List of dicts, each with 'Question' and 'CorrectAnswer'.
        optimizer_type: Name of the optimizer module under seimei/train/klg_optim/.
        n_sample: Number of seimei runs per dataset row per epoch.
        n_epoch: Number of optimization epochs.
        n_new_klg_per_epoch: Number of new knowledge entries to generate per epoch.
        update_klg_threshold: Minimum mean score improvement to accept a knowledge
            update (range 0.0–1.0).
        metric: Scoring metric. Currently supports 'answer_exact_match' (LLM judge).
        load_knowledge_path: Path to the initial knowledge CSV to load.
        save_knowledge_path: Path to save the optimized knowledge CSV.
        cache_path: Path for the run-cache JSON (enables resume on restart).
        workspace_root: Optional directory for workspace copies (workspace mode).
            If given, each seimei call receives a workspace= argument pointing to
            a subdirectory of workspace_root.
        patch_dir: Optional directory of patch files (requires workspace_root).
            When provided, PatchWorkspaceManager is used to apply patches per row.
        **kwargs: Passed through to the optimizer.  Typically includes seimei
            constructor args (e.g. llm_config, agent_config, log_dir, max_steps,
            allow_code_exec) and seimei.__call__ args (e.g. knowledge_search_config,
            knowledge_search_mode, system).  May also include repo_root (str/Path)
            for workspace-mode preparation.

    Returns:
        Updated knowledge pool as a list of dicts (same format as default.csv rows).
    """
    # --- Input validation ---
    if not isinstance(dataset, list):
        raise TypeError("dataset must be a list")
    if len(dataset) == 0:
        raise ValueError("dataset must be non-empty")
    for i, row in enumerate(dataset):
        if not isinstance(row, dict):
            raise TypeError(f"dataset[{i}] must be a dict, got {type(row).__name__}")
        if "Question" not in row:
            raise ValueError(f"dataset[{i}] missing required field 'Question'")
        if "CorrectAnswer" not in row:
            raise ValueError(f"dataset[{i}] missing required field 'CorrectAnswer'")

    if not isinstance(optimizer_type, str) or not optimizer_type.strip():
        raise ValueError("optimizer_type must be a non-empty string")
    if not isinstance(n_sample, int) or n_sample < 1:
        raise ValueError("n_sample must be a positive integer")
    if not isinstance(n_epoch, int) or n_epoch < 1:
        raise ValueError("n_epoch must be a positive integer")
    if not isinstance(n_new_klg_per_epoch, int) or n_new_klg_per_epoch < 0:
        raise ValueError("n_new_klg_per_epoch must be a non-negative integer")
    if not isinstance(update_klg_threshold, (int, float)):
        raise ValueError("update_klg_threshold must be a number")
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric must be a non-empty string")

    load_knowledge_path = str(load_knowledge_path)
    save_knowledge_path = str(save_knowledge_path)
    cache_path = str(cache_path)

    if not Path(load_knowledge_path).exists():
        raise FileNotFoundError(
            f"load_knowledge_path does not exist: {load_knowledge_path}"
        )

    # --- Resolve and load optimizer module ---
    klg_optim_dir = Path(__file__).resolve().parent / "klg_optim"
    optimizer_file = klg_optim_dir / f"{optimizer_type}.py"
    if not optimizer_file.exists():
        available = sorted(
            p.stem for p in klg_optim_dir.glob("*.py") if p.stem != "__init__"
        )
        raise ValueError(
            f"Unknown optimizer_type '{optimizer_type}'. "
            f"Available: {available}. Expected file: {optimizer_file}"
        )

    module = importlib.import_module(f"seimei.train.klg_optim.{optimizer_type}")

    return module.main(
        dataset=dataset,
        n_sample=n_sample,
        n_epoch=n_epoch,
        n_new_klg_per_epoch=n_new_klg_per_epoch,
        update_klg_threshold=update_klg_threshold,
        metric=metric,
        load_knowledge_path=load_knowledge_path,
        save_knowledge_path=save_knowledge_path,
        cache_path=cache_path,
        workspace_root=workspace_root,
        patch_dir=patch_dir,
        **kwargs,
    )
