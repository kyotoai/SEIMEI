# SEIMEI exp5/train_v2 Knowledge Loop

This document describes the algorithm implemented in `exp5/train_v2.py`, the required inputs and produced outputs, and how to run the workflow end-to-end.

## Algorithm overview

`train_v2.py` automates a six-stage loop for each question in `exp5/dataset.json`:

1. **Baseline inference** – Run SEIMEI with no additional knowledge to capture the agent's natural reasoning.
2. **Step diagnosis** – Feed the full transcript to `seimei/knowledge/prompts/gen_step.md` so an auxiliary LLM can identify the first agent step whose reasoning drifts away from the correct solution path.
3. **Knowledge synthesis** – Using the diagnosed step, question context, and the currently best transcript, `train_v2.py` calls an internal prompt that explicitly states "this knowledge replaces agent step _k_". The prompt requests a JSON payload containing the knowledge text, agent target, tags, and rationale.
4. **Knowledge-injected replay** – The script truncates the recorded run before the diagnosed step (leveraging the `load_run_messages` resume flow from `seimei/README.md`) and replays the run with a manual `knowledge_config` entry for that step. Knowledge generation is disabled inside SEIMEI; only the manually supplied entry guides the agent.
5. **check_knowledge evaluation** – `check_knowledge(...)` re-runs SEIMEI, scores the new answer (0–10) via the `SCORING_SYSTEM_PROMPT`, and compares it with the best score so far. When a new score exceeds the previous best, the new run becomes the reference for the next iteration and the `[chosen, rejected]` indices are recorded for DPO comparisons.
6. **Iterative refinement** – Steps 3–5 repeat `MAX_KNOWLEDGE_ITERATIONS` times (default: 3) so that each iteration analyzes the latest best transcript and proposes a sharper piece of knowledge for the same step.

After the loop finishes, the best knowledge snippet (if any) is appended to `seimei_knowledge/exp5_train_v2_1.csv`, and the full trajectory (baseline plus all knowledge attempts) is serialized into a DPO-ready JSON entry.

## Inputs

- `exp5/dataset.json`: Each entry must provide `Question`, `CorrectAnswer`, and `CSVPath`.
- Existing knowledge store (optional): if `seimei_knowledge/exp5_train_v2_1.csv` already contains knowledge, SEIMEI loads it before running.
- Agent roster: `train_v2.py` uses the built-in `code_act` agent (`seimei/agents/code_act.py`). Adjust `agent_config` if you need additional agents.

## Outputs

1. **Knowledge CSV** – The best-performing knowledge snippet per problem (when non-empty) is appended to `seimei_knowledge/exp5_train_v2_1.csv`. Columns include `run_id`, `agent`, `knowledge`, `tags`, `step`, `score`, and `source=train_v2_best`.
2. **DPO dataset** – `exp5/train_v2_dpo.json` is written with a list of structures of the form:

```json
[
  {
    "run_ids": ["run-...base", "run-...k1", ...],
    "step": 3,
    "message": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ],
    "knowledge": [
      {"text": null, "step": 3, "iteration": 0, ...},
      {"text": "...", "step": 3, "iteration": 1, ...},
      ...
    ],
    "comparison": [[1, 0], [2, 1]],
    "scores": [4.0, 7.0, 8.0]
  }
]
```

This matches the requested format: index-aligned `run_ids`, `knowledge`, and `scores` arrays, plus `comparison` references for the DPO preference pairs.

## Example usage

Run the training loop from the repository root:

```bash
python exp5/train_v2.py
```

The script will iterate through every dataset entry, printing status messages such as the diagnosed step, per-iteration score, and the saved JSON path once the DPO file is written. Re-run the script to add new samples or regenerate the DPO dataset after updating `exp5/dataset.json`.

### Customization tips

- Update `MAX_KNOWLEDGE_ITERATIONS` in `train_v2.py` to explore deeper refinement loops.
- Replace `code_act` with other agents by editing `agent_config`.
- Swap in a different scoring model by changing `llm_kwargs["model"]` or the `SCORING_SYSTEM_PROMPT`.
- `train_v2_dpo.json` is overwritten on each run; version it separately if you need checkpoints.

With these steps, `train_v2.py` performs the requested "infer → diagnose → relaunch with knowledge → compare" cycle and produces both reusable knowledge and preference data for DPO-style training.
