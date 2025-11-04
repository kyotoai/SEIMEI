# SEIMEI Run Retrospective Prompt

You are reviewing one or more completed SEIMEI runs. Each run includes the conversation,
agent step logs, and final outcome.

<<RUN_CONTEXT>>

## Task
- Identify concrete improvements for future runs by inspecting how each agent performed.
- Focus on mistakes, missed opportunities, or heuristics that would have improved the outcome.
- Translate these insights into reusable knowledge snippets grouped by agent name.
- Prefer actionable instructions over generic advice. Keep statements concise.

## Output format
Return a single JSON array like:

```json
[
  {
    "agent": "think",
    "knowledge": "Summarize user goals and constraints before invoking specialised agents.",
    "tags": ["planning", "handoff"]
  }
]
```

Rules:
- Use lowercase agent identifiers.
- Keep `knowledge` under 220 characters.
- Include 1–3 short `tags` per entry.
- Do not output anything outside the JSON array.

