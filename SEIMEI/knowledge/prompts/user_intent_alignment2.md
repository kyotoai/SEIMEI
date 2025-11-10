
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
- Provide concrete, actionable advice describing how future agents should converse
  with this user (e.g., how to structure answers, when to ask clarifying questions,
  what level of depth to use, and how to handle follow-ups) so the user's intent
  is fully satisfied.

## Concrete advice requirements
- Every output item must encode **specific behavioral guidance** for future runs.
- `think` agent entries should focus on **conversation strategy**:
  - how the assistant should respond (step-by-step vs. concise),
  - how to confirm or infer user goals,
  - how to balance initiative vs. deference,
  - how to handle technical depth, edge cases, and follow-up suggestions.
- `answer` (and other) agent entries may focus on **content style**:
  - tone, formatting, level of explanation, examples, or references that improved
    or would improve user satisfaction.
- Guidance must be phrased so another agent can immediately implement it without
  additional interpretation.

## Output format
Return a JSON array, for example:

```json
[
  {
    "agent": "answer",
    "knowledge": "When the user intent is exploratory, user satisfaction increases if the answer is structured with headings, examples, and a short final summary.",
    "tags": ["intent", "format"]
  }
]
````

```json
[
  {
    "agent": "think",
    "knowledge": "When user intent is unclear, satisfaction improves if the agent briefly infers their likely goal and proposes 2-3 concrete next-step options.",
    "tags": ["intent", "followup"]
  }
]
```

Rules:

* Use lowercase agent identifiers (typically `answer`, `think`, or other agents seen in the run).
* Keep each `knowledge` entry under 220 characters; make it specific to what satisfied (or would
  have satisfied) the user.
* Each `knowledge` sentence must mention the user intent or satisfaction criteria.
* `think` items must include concrete advice about how the assistant should conduct the conversation.
* Include 1–3 short `tags` such as `intent`, `format`, `depth`, or `followup`.
* Output only the JSON array—no commentary outside the code block.

