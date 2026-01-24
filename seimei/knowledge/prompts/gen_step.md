# Drifted Step Identification Prompt

You are auditing a SEIMEI agent transcript to pinpoint where the reasoning derailed
from the path that leads to the correct answer. The transcript contains every
message (system, user, agents, assistant) plus the final answer.

<<RUN_CONTEXT>>

## Task
- Determine the **single most impactful agent step** (1-indexed) whose reasoning
  drifts away from the correct solution path.
- Explain why the step went off course and what information or operations the
  agent missed.
- Suggest how the step should be reframed so that the following regenerated step
  (with knowledge injected) can stay aligned with the question.

## Output format
Return a JSON array with exactly **one** object that follows this schema:

```json
[
  {
    "step": 3,
    "agent": "think",
    "drift_summary": "Agent guessed the trend without checking params_json",
    "missing_clues": "Never read how hyper_param_index controls row count",
    "revision_goal": "Force the agent to inspect params_json for each index",
    "tags": ["important", "excel", "step_3"]
  }
]
```

Rules:
- `step` must be a positive integer referencing an agent turn.
- `agent` must be one of `think`, `code_act`, `web_search`, or `answer`.
- Keep explanations concrete, referencing evidence from the transcript.
- Output **only** the JSON array—no commentary outside the code block.
