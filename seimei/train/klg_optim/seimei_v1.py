"""seimei_v1: LLM-based knowledge pool optimizer.

Algorithm per epoch:
  1. Run seimei on every dataset row for n_sample times; score each output.
  2. Build a map of knowledge_id → list of inferences that used it.
  3. For each knowledge entry, ask the LLM to assess its impact and propose
     improved text.
  4. Ask the LLM to generate n_new_klg_per_epoch brand-new knowledge entries
     and add them to the pool.
  5. Re-run all inferences with the updated pool; compute per-knowledge mean
     score improvement (rerun − baseline).  Accept updates only for entries
     whose improvement exceeds update_klg_threshold.
  6. If n_epoch > 1, the re-run results from step 5 become step 1 of the next
     epoch (no redundant re-run).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from seimei import seimei as Seimei
from seimei.train.sampling_utils import (
    coerce_score,
    extract_knowledge_entries_from_messages,
    load_knowledge_pool_csv,
    normalize_pool_ids,
    parse_json_response,
    write_knowledge_pool_csv,
)

# ---------------------------------------------------------------------------
# seimei kwarg partitioning
# ---------------------------------------------------------------------------

_SEIMEI_INIT_KEYS: frozenset = frozenset({
    "agent_config", "llm_config", "rm_config", "log_dir", "max_steps",
    "allow_code_exec", "allowed_commands", "approval_callback",
    "agent_log_head_lines", "max_tokens_per_question", "llm_client_class",
})

# Keys we allow the user to pass through to seimei.__call__ (we manage
# knowledge_load_config and agent_search_mode ourselves).
_SEIMEI_CALL_KEYS: frozenset = frozenset({
    "system", "stop_when", "return_usage",
    "agent_search_config",
    "knowledge_search_mode", "knowledge_search_config",
    "knowledge_generate_config",
})

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_SCORING_SYSTEM_PROMPT = "Return only JSON."
_ASSESSMENT_SYSTEM_PROMPT = "Return only JSON."
_NEW_KLG_SYSTEM_PROMPT = "Return only JSON."


def _build_scoring_prompt(question: str, correct_answer: str, model_answer: str) -> str:
    return (
        "Evaluate whether the model answer correctly answers the question.\n\n"
        f"<QUESTION>\n{question}\n</QUESTION>\n\n"
        f"<CORRECT ANSWER>\n{correct_answer}\n</CORRECT ANSWER>\n\n"
        f"<MODEL ANSWER>\n{model_answer}\n</MODEL ANSWER>\n\n"
        "Score how accurately the model answer matches the correct answer "
        "(1.0 = fully correct, 0.0 = completely wrong).\n"
        "Return ONLY a JSON object with keys:\n"
        "  'score'    (float 0.0–1.0)\n"
        "  'feedback' (concise string explaining the score)"
    )


def _build_assessment_prompt(
    klg_id: Any,
    agent: str,
    original_text: str,
    inferences: List[Dict[str, Any]],
) -> str:
    lines = []
    for i, inf in enumerate(inferences[:10], 1):  # cap at 10 to keep prompt size sane
        lines.append(
            f"[{i}] Question : {inf['question']}\n"
            f"     Answer   : {inf['output']}\n"
            f"     Score    : {inf['score']}\n"
            f"     Feedback : {inf['feedback']}"
        )
    inferences_text = "\n\n".join(lines) if lines else "(no inferences recorded this epoch)"
    return (
        f"Knowledge snippet (id={klg_id}, agent={agent}):\n{original_text}\n\n"
        f"Inferences that used this snippet:\n\n{inferences_text}\n\n"
        "Based on the scores and feedback above, write an improved version of this "
        "knowledge snippet that would lead to better answers in future runs.\n"
        "Return ONLY a JSON object with keys:\n"
        "  'improved_text' (string — the revised snippet)\n"
        "  'rationale'     (string — brief explanation of the change)"
    )


def _build_new_knowledge_prompt(
    n: int,
    assessments: List[Dict[str, Any]],
    available_agents: List[str],
) -> str:
    lines = []
    for a in assessments[:20]:  # cap to keep prompt manageable
        rationale = (a.get("rationale") or "").strip()
        if rationale:
            lines.append(f"- id={a['klg_id']}, agent={a['agent']}: {rationale}")
    assessments_text = "\n".join(lines) if lines else "(none)"
    agents_str = ", ".join(sorted(available_agents))
    return (
        f"Generate {n} new knowledge snippets to help a reasoning agent answer "
        "questions more accurately.\n\n"
        "Patterns identified from existing knowledge assessments:\n"
        f"{assessments_text}\n\n"
        "Each new snippet should address a gap or failure mode not already covered "
        "by the existing knowledge.\n"
        f"Available agents: {agents_str}\n\n"
        f"Return ONLY a JSON object with key 'new_knowledge' (array of {n} items). "
        "Each item must have:\n"
        f"  'agent' (string, one of: {agents_str})\n"
        "  'text'  (string — the knowledge snippet)\n"
        "  'tags'  (array of strings — descriptive labels)"
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache(cache_path: str) -> Dict[str, Any]:
    p = Path(cache_path)
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                if not isinstance(raw.get("run_cache"), dict):
                    raw["run_cache"] = {}
                return raw
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[seimei_v1] Failed to load cache from {cache_path}: {exc}")
    return {"schema_version": 2, "run_cache": {}}


def _save_cache(cache: Dict[str, Any], cache_path: str) -> None:
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    except OSError as exc:
        print(f"[seimei_v1] Failed to save cache to {cache_path}: {exc}")


def _run_key(dataset_idx: int, epoch: int, sample: int, suffix: str = "") -> str:
    base = f"{dataset_idx}::{epoch}::{sample}"
    return f"{base}::{suffix}" if suffix else base


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

async def _call_seimei(
    orchestrator: Any,
    question: str,
    knowledge_path: str,
    run_name: str,
    call_kwargs: Dict[str, Any],
    workspace_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Invoke seimei.__call__ for a single question."""
    messages = [{"role": "user", "content": question}]
    extra: Dict[str, Any] = {}
    if workspace_path is not None:
        extra["workspace"] = workspace_path
    try:
        result = await orchestrator(
            messages=messages,
            run_name=run_name,
            knowledge_load_config=[{"load_knowledge_path": knowledge_path}],
            **extra,
            **call_kwargs,
        )
    except Exception as exc:
        print(f"[seimei_v1] seimei call failed ({run_name}): {exc}")
        result = {"run_id": "", "output": "", "msg_history": []}
    return result


async def _score_output(
    llm: Any,
    question: str,
    correct_answer: str,
    model_answer: str,
) -> Tuple[float, str]:
    """LLM judge: score model_answer against correct_answer."""
    prompt = _build_scoring_prompt(question, correct_answer, model_answer)
    try:
        response, _ = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system=_SCORING_SYSTEM_PROMPT,
        )
        parsed = parse_json_response(response)
        score = coerce_score(parsed.get("score"))
        feedback = str(parsed.get("feedback") or "").strip()
        return score, feedback
    except Exception as exc:
        return 0.0, f"Scoring failed: {exc}"


# ---------------------------------------------------------------------------
# Epoch runner  (sequential to avoid shared-state issues in seimei)
# ---------------------------------------------------------------------------

async def _run_epoch(
    orchestrator: Any,
    dataset: List[Dict[str, Any]],
    knowledge_path: str,
    n_sample: int,
    epoch: int,
    call_kwargs: Dict[str, Any],
    run_cache: Dict[str, Any],
    suffix: str,
    workspace_path: Optional[str],
    patch_managers: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    """Run seimei + score for every (dataset row, sample) pair.

    Returns a list of run records:
        {dataset_idx, output, score, feedback, knowledge_ids}
    """
    records: List[Dict[str, Any]] = []

    for idx, row in enumerate(dataset):
        question = row["Question"]
        correct_answer = row["CorrectAnswer"]

        for sample in range(n_sample):
            key = _run_key(idx, epoch, sample, suffix)
            cached = run_cache.get(key)
            if isinstance(cached, dict):
                records.append({
                    "dataset_idx": idx,
                    "output": cached.get("output", ""),
                    "score": cached.get("score", 0.0),
                    "feedback": cached.get("feedback", ""),
                    "knowledge_ids": cached.get("knowledge_ids", []),
                })
                continue

            run_name = (
                f"klg_optim_e{epoch}_i{idx}_s{sample}"
                + (f"_{suffix}" if suffix else "")
            )

            # Workspace mode: apply patch if patch_manager is available
            pm = None
            if patch_managers:
                pm = patch_managers[idx % len(patch_managers)]

            if pm is not None:
                ctx = pm.apply_for_problem(row, idx)
            else:
                from contextlib import nullcontext
                ctx = nullcontext()

            with ctx:
                result = await _call_seimei(
                    orchestrator, question, knowledge_path, run_name,
                    call_kwargs, workspace_path,
                )

            # Reset workspace after patch use
            if pm is not None:
                from seimei.train.sampling_utils import prepare_workspace
                prepare_workspace(pm.workspace, pm.workspace.parent.parent)

            output = str(result.get("output") or "")
            msg_history: List[Dict[str, Any]] = result.get("msg_history") or []

            score, feedback = await _score_output(
                orchestrator.llm, question, correct_answer, output
            )

            klg_entries = extract_knowledge_entries_from_messages(msg_history, orchestrator)
            # Normalise IDs to strings for consistent dict keys
            knowledge_ids = [
                str(e["id"]) for e in klg_entries if e.get("id") is not None
            ]

            record: Dict[str, Any] = {
                "dataset_idx": idx,
                "output": output,
                "score": score,
                "feedback": feedback,
                "knowledge_ids": knowledge_ids,
            }
            records.append(record)

            run_cache[key] = {
                "output": output,
                "score": score,
                "feedback": feedback,
                "knowledge_ids": knowledge_ids,
            }

    return records


# ---------------------------------------------------------------------------
# Knowledge assessment
# ---------------------------------------------------------------------------

async def _assess_one(
    llm: Any,
    entry: Dict[str, Any],
    inferences: List[Dict[str, Any]],
) -> Dict[str, Any]:
    klg_id = entry.get("id")
    agent = str(entry.get("agent") or "")
    original_text = str(entry.get("text") or "").strip()

    if not inferences:
        return {
            "klg_id": klg_id,
            "agent": agent,
            "original_text": original_text,
            "improved_text": original_text,
            "rationale": "Not used in any inference this epoch.",
        }

    prompt = _build_assessment_prompt(klg_id, agent, original_text, inferences)
    try:
        response, _ = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system=_ASSESSMENT_SYSTEM_PROMPT,
        )
        parsed = parse_json_response(response)
        improved = str(parsed.get("improved_text") or original_text).strip() or original_text
        rationale = str(parsed.get("rationale") or "").strip()
    except Exception as exc:
        print(f"[seimei_v1] Assessment failed for knowledge {klg_id}: {exc}")
        improved = original_text
        rationale = f"Assessment error: {exc}"

    return {
        "klg_id": klg_id,
        "agent": agent,
        "original_text": original_text,
        "improved_text": improved,
        "rationale": rationale,
    }


async def _assess_all(
    llm: Any,
    pool: List[Dict[str, Any]],
    klg_map: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Assess all knowledge entries sequentially (avoids LLM rate-limit storms)."""
    assessments = []
    for entry in pool:
        kid_str = str(entry.get("id")) if entry.get("id") is not None else ""
        inferences = klg_map.get(kid_str, [])
        assessment = await _assess_one(llm, entry, inferences)
        assessments.append(assessment)
    return assessments


async def _generate_new_knowledge(
    llm: Any,
    n: int,
    assessments: List[Dict[str, Any]],
    pool: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if n <= 0:
        return []

    available_agents = sorted({str(e.get("agent") or "") for e in pool if e.get("agent")})
    if not available_agents:
        available_agents = ["code_act", "think", "answer"]

    prompt = _build_new_knowledge_prompt(n, assessments, available_agents)
    try:
        response, _ = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system=_NEW_KLG_SYSTEM_PROMPT,
        )
        parsed = parse_json_response(response)
        raw_list = parsed.get("new_knowledge") or []
        entries: List[Dict[str, Any]] = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            agent = str(item.get("agent") or "").strip()
            if not text or not agent:
                continue
            tags = item.get("tags") or []
            if not isinstance(tags, list):
                tags = [str(tags)] if tags else []
            entries.append({"agent": agent, "text": text, "tags": tags})
        return entries[:n]
    except Exception as exc:
        print(f"[seimei_v1] New knowledge generation failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------

def _build_klg_inference_map(
    run_records: List[Dict[str, Any]],
    dataset: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Map knowledge ID (string) → list of inference dicts."""
    klg_map: Dict[str, List[Dict[str, Any]]] = {}
    for record in run_records:
        idx = record["dataset_idx"]
        row = dataset[idx]
        for kid in record.get("knowledge_ids", []):
            kid_str = str(kid)
            klg_map.setdefault(kid_str, []).append({
                "question": row["Question"],
                "output": record["output"],
                "score": record["score"],
                "feedback": record["feedback"],
            })
    return klg_map


def _next_int_id(pool: List[Dict[str, Any]]) -> int:
    max_id = 0
    for e in pool:
        try:
            v = int(e.get("id", 0))
            if v > max_id:
                max_id = v
        except (TypeError, ValueError):
            pass
    return max_id + 1


def _build_updated_pool(
    original_pool: List[Dict[str, Any]],
    assessments: List[Dict[str, Any]],
    new_entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Replace existing entries with improved texts; append new entries."""
    improved_map = {str(a["klg_id"]): a["improved_text"] for a in assessments}
    updated: List[Dict[str, Any]] = []
    for entry in original_pool:
        new = dict(entry)
        kid_str = str(entry.get("id")) if entry.get("id") is not None else ""
        if kid_str in improved_map:
            new["text"] = improved_map[kid_str]
        updated.append(new)

    next_id = _next_int_id(original_pool)
    for item in new_entries:
        updated.append({
            "id": next_id,
            "agent": item["agent"],
            "text": item["text"],
            "tags": item.get("tags") or [],
        })
        next_id += 1
    return updated


def _mean_scores_by_klg(run_records: List[Dict[str, Any]]) -> Dict[str, float]:
    scores: Dict[str, List[float]] = {}
    for record in run_records:
        for kid in record.get("knowledge_ids", []):
            scores.setdefault(str(kid), []).append(record["score"])
    return {kid: sum(v) / len(v) for kid, v in scores.items()}


def _apply_threshold(
    original_pool: List[Dict[str, Any]],
    updated_pool: List[Dict[str, Any]],
    baseline: Dict[str, float],
    rerun: Dict[str, float],
    threshold: float,
    original_ids: Set[str],
) -> List[Dict[str, Any]]:
    """Decide which updated pool entries survive the threshold filter.

    Rules:
    - improvement > threshold  → keep the updated entry (improved text / new entry)
    - improvement <= threshold and entry is original → revert to original text
    - improvement <= threshold and entry is new → discard
    - no score data available  → keep original entries; discard new entries
    """
    original_map: Dict[str, Dict[str, Any]] = {
        str(e.get("id")): e for e in original_pool
    }
    result: List[Dict[str, Any]] = []

    for entry in updated_pool:
        kid_str = str(entry.get("id")) if entry.get("id") is not None else ""
        is_original = kid_str in original_ids
        b = baseline.get(kid_str)
        r = rerun.get(kid_str)

        if b is None or r is None:
            # No usage data this epoch
            if is_original:
                result.append(dict(original_map.get(kid_str, entry)))
            # New entry never used → silently discard
            continue

        if r - b > threshold:
            result.append(dict(entry))
        elif is_original:
            result.append(dict(original_map.get(kid_str, entry)))
        # else: new entry below threshold → discard

    return result


# ---------------------------------------------------------------------------
# Main async logic
# ---------------------------------------------------------------------------

async def _main_async(
    dataset: List[Dict[str, Any]],
    n_sample: int,
    n_epoch: int,
    n_new_klg_per_epoch: int,
    update_klg_threshold: float,
    metric: str,
    load_knowledge_path: str,
    save_knowledge_path: str,
    cache_path: str,
    workspace_root: Optional[str],
    patch_dir: Optional[str],
    **kwargs: Any,
) -> List[Dict[str, Any]]:

    # --- Separate seimei init / call kwargs from everything else ---
    repo_root_arg = kwargs.pop("repo_root", None)
    seimei_init_kwargs = {k: v for k, v in kwargs.items() if k in _SEIMEI_INIT_KEYS}
    seimei_call_kwargs = {k: v for k, v in kwargs.items() if k in _SEIMEI_CALL_KEYS}

    if "llm_config" not in seimei_init_kwargs:
        raise ValueError("llm_config must be provided in kwargs for seimei_v1")

    if metric != "answer_exact_match":
        raise ValueError(
            f"Unsupported metric '{metric}'. Currently supported: 'answer_exact_match'."
        )

    # Default agent_search_mode to "klg" so the knowledge pool is actually used
    seimei_call_kwargs.setdefault("agent_search_mode", "klg")

    # --- Load knowledge pool ---
    pool = load_knowledge_pool_csv(Path(load_knowledge_path))
    pool = normalize_pool_ids(pool)
    if not pool:
        raise ValueError(f"Knowledge pool is empty: {load_knowledge_path}")
    original_ids: Set[str] = {str(e.get("id")) for e in pool}

    # --- Cache ---
    cache = _load_cache(cache_path)
    run_cache: Dict[str, Any] = cache.setdefault("run_cache", {})

    # --- Build seimei orchestrator ---
    orchestrator = Seimei(**seimei_init_kwargs)

    # --- Workspace setup ---
    workspace_path: Optional[str] = None
    patch_managers: Optional[List[Any]] = None

    if workspace_root is not None:
        from seimei.train.sampling_utils import PatchWorkspaceManager, prepare_workspace

        ws_root = Path(workspace_root)
        ws_root.mkdir(parents=True, exist_ok=True)
        ws = ws_root / "ws_00"
        ws.mkdir(parents=True, exist_ok=True)
        workspace_path = str(ws)

        if repo_root_arg is not None:
            prepare_workspace(ws, Path(repo_root_arg))

        if patch_dir is not None:
            patch_managers = [PatchWorkspaceManager(ws, Path(patch_dir))]

    # --- Write initial pool so the orchestrator can load it ---
    write_knowledge_pool_csv(Path(save_knowledge_path), pool)

    prev_run_records: Optional[List[Dict[str, Any]]] = None

    for epoch in range(n_epoch):
        print(f"[seimei_v1] === Epoch {epoch + 1}/{n_epoch} ===")

        # ------------------------------------------------------------------
        # Step 1: Inference
        # (re-use previous epoch's rerun results from step 5 when available)
        # ------------------------------------------------------------------
        if epoch == 0 or prev_run_records is None:
            total = len(dataset) * n_sample
            print(f"[seimei_v1] Step 1: Running {total} inference(s)...")
            run_records = await _run_epoch(
                orchestrator, dataset, save_knowledge_path,
                n_sample, epoch, seimei_call_kwargs, run_cache,
                suffix="", workspace_path=workspace_path,
                patch_managers=patch_managers,
            )
        else:
            print("[seimei_v1] Step 1: Reusing previous epoch's rerun results.")
            run_records = prev_run_records

        _save_cache(cache, cache_path)

        # ------------------------------------------------------------------
        # Step 2: Build knowledge → inference map
        # ------------------------------------------------------------------
        print("[seimei_v1] Step 2: Building knowledge-inference map...")
        klg_map = _build_klg_inference_map(run_records, dataset)
        used = sum(1 for v in klg_map.values() if v)
        print(f"[seimei_v1]   {used}/{len(pool)} knowledge entries were used.")

        # ------------------------------------------------------------------
        # Step 3: Assess each knowledge entry and propose improved text
        # ------------------------------------------------------------------
        print(f"[seimei_v1] Step 3: Assessing {len(pool)} knowledge entries...")
        assessments = await _assess_all(orchestrator.llm, pool, klg_map)

        # ------------------------------------------------------------------
        # Step 4: Generate new knowledge entries
        # ------------------------------------------------------------------
        print(f"[seimei_v1] Step 4: Generating {n_new_klg_per_epoch} new knowledge entry(s)...")
        new_entries = await _generate_new_knowledge(
            orchestrator.llm, n_new_klg_per_epoch, assessments, pool
        )
        print(f"[seimei_v1]   Generated {len(new_entries)} new entries.")

        # Build updated pool and persist it for the rerun
        updated_pool = _build_updated_pool(pool, assessments, new_entries)
        write_knowledge_pool_csv(Path(save_knowledge_path), updated_pool)

        # ------------------------------------------------------------------
        # Step 5: Rerun with updated pool; measure per-knowledge improvement
        # ------------------------------------------------------------------
        total = len(dataset) * n_sample
        print(f"[seimei_v1] Step 5: Rerunning {total} inference(s) with updated pool...")
        rerun_records = await _run_epoch(
            orchestrator, dataset, save_knowledge_path,
            n_sample, epoch, seimei_call_kwargs, run_cache,
            suffix="rerun", workspace_path=workspace_path,
            patch_managers=patch_managers,
        )
        _save_cache(cache, cache_path)

        baseline_means = _mean_scores_by_klg(run_records)
        rerun_means = _mean_scores_by_klg(rerun_records)

        all_kids = sorted(
            set(list(baseline_means.keys()) + list(rerun_means.keys())),
            key=lambda x: (int(x) if x.isdigit() else 0, x),
        )
        print(f"[seimei_v1]   Per-knowledge score improvements "
              f"(threshold={update_klg_threshold}):")
        for kid in all_kids:
            b = baseline_means.get(kid, 0.0)
            r = rerun_means.get(kid, 0.0)
            delta = r - b
            verdict = "keep" if delta > update_klg_threshold else "revert/discard"
            print(f"[seimei_v1]     id={kid}: {b:.3f} -> {r:.3f} "
                  f"(delta={delta:+.3f}) [{verdict}]")

        # Apply threshold: keep updates that improved, revert/discard others
        pool = _apply_threshold(
            pool, updated_pool,
            baseline_means, rerun_means,
            update_klg_threshold, original_ids,
        )
        # Update original_ids for the next epoch
        original_ids = {str(e.get("id")) for e in pool}

        write_knowledge_pool_csv(Path(save_knowledge_path), pool)
        _save_cache(cache, cache_path)

        prev_run_records = rerun_records
        print(f"[seimei_v1] Epoch {epoch + 1} done. "
              f"Final pool size: {len(pool)} entries.\n")

    return pool


# ---------------------------------------------------------------------------
# Public entry point (called by KlgOptimizer)
# ---------------------------------------------------------------------------

def main(**kwargs: Any) -> List[Dict[str, Any]]:
    """Entry point called by KlgOptimizer.  Runs the async loop synchronously."""
    required = [
        "dataset", "n_sample", "n_epoch", "n_new_klg_per_epoch",
        "update_klg_threshold", "metric", "load_knowledge_path",
        "save_knowledge_path", "cache_path",
    ]
    for key in required:
        if key not in kwargs:
            raise KeyError(f"seimei_v1.main() missing required argument: '{key}'")

    params = {k: kwargs.pop(k) for k in required}
    workspace_root = kwargs.pop("workspace_root", None)
    patch_dir = kwargs.pop("patch_dir", None)

    return asyncio.run(
        _main_async(
            **params,
            workspace_root=workspace_root,
            patch_dir=patch_dir,
            **kwargs,
        )
    )
