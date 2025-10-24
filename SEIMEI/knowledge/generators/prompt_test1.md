# SEIMEI Knowledge Generation Prompt

You are compiling a reusable knowledge base for the SEIMEI multi-agent system.  
Agents that require knowledge coverage:

<<AGENT_LIST>>

## Task
- Produce roughly <<TARGET_COUNT>> concise, high-signal knowledge statements **per agent** (allow a small deviation if needed).
- Each statement must be actionable guidance that helps the agent make better decisions.
- Tailor the content to the agent's capabilities (e.g., planning, code execution, answering, web search).
- Prefer concrete heuristics, checklists, or reminders over generic advice.

## Output format
Return a single JSON array with objects shaped like:

```json
[
  {
    "agent": "think",
    "knowledge": "Verify the user intent before delegating to execution agents.",
    "tags": ["planning", "analysis"]
  }
]
```

Rules:
- Use lowercase agent identifiers that exactly match the provided names.
- Keep `knowledge` under 220 characters.
- `tags` should be a list of 1–3 short keywords (strings).
- Do **not** include commentary outside the JSON array.

Deliver the final JSON array only.
