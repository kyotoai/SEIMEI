# SEIMEI User Intent Satisfaction Prompt

You are studying a completed SEIMEI run containing the full dialogue, agent steps,
and final outputs. Use this context to reverse-engineer what the user truly wanted
and how well the system satisfied that desire.

<<RUN_CONTEXT>>

## Task
- Extract the user's explicit question, implicit intent, constraints, and success signals.
- Pinpoint where the delivered answer delighted, confused, or disappointed the user.
- Translate those insights into concrete guidance so future responses align with the
  user's preferred tone, structure, rigor, and follow-up depth.
- Emphasize how to craft the *next* answer so the user feels their intent was fully met.

## Output format
Return a JSON array, for example:

```json
[
  {
    "agent": "answer",
    "knowledge": "Your guidance here.",
    "tags": ["intent", "tone"]
  }
]
```

```json
[
  {
    "agent": "think",
    "knowledge": "Your concrete guidance here.",
    "tags": ["memory", "cpu"]
  }
]
```

Rules:
- Use lowercase agent identifiers (typically `answer`, `think`, or other agents seen in the run).
- Keep each `knowledge` entry under 220 characters; make it specific to what satisfied (or would
  have satisfied) the user.
- Each `knowledge` sentence must mention the user intent or satisfaction criteria.
- Include 1–3 short `tags` such as `intent`, `format`, `depth`, or `followup`.
- Output only the JSON array—no commentary outside the code block.
