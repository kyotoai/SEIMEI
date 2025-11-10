# SEIMEI User Constraint Knowledge Prompt

You are inspecting a completed SEIMEI run that contains the full dialogue, agent
steps, and outputs. Use this context to extract durable, factual knowledge about
the user so future `think` agents can reason with the user's real constraints.

<<RUN_CONTEXT>>

## Task
- Capture concrete facts about the user's goals, technology choices, resources,
  deadlines, compliance rules, or other constraints surfaced in the run.
- Prefer verifiable statements tied to the chat history. If the user insists on
  a specific architecture (e.g., "only 3D U-Net (~40M params)"), record it.
- Highlight what the user can or cannot change (tooling limits, compute caps,
  preferred stack, review expectations, etc.).
- Summarize how the user wants agents to reason (initiative level, validation
  steps, clarification thresholds) so the `think` agent can plan accordingly.

## Concrete knowledge requirements
- Every knowledge sentence must mention a user need or constraint derived from
  the run; avoid generic advice.
- Prioritize factual observations over stylistic notes—store anything repeatable
  (hardware specs, data access rules, collaboration habits, review cadence).
- Mention constraints even if they blocked the run; note mitigations the user
  will accept.
- When a fact is pivotal to success, include the literal marker `[IMPORTANT FACT]`
  inside the knowledge text (use sparingly for high-impact constraints only).
- Keep each sentence under 220 characters and make it immediately actionable for
  future reasoning.

## Output format
- Return a JSON array containing only `think` agent entries.
- Use the following schema:

```json
[
  {
    "agent": "think",
    "knowledge": "[IMPORTANT FACT] The user only deploys 3D U-Net (~40M params) so plans must keep memory under 16GB and avoid alternative models.",
    "tags": ["constraints", "model"]
  }
]
```

Rules:

* Output only the JSON array—no commentary outside the code block.
* Keep agent identifiers lowercase (`think` only).
* Choose 1–3 short tags like `constraints`, `stack`, `process`, `risks`.
* Ensure every entry encodes specific behavioral guidance rooted in the user's
  facts or constraints so the next `think` agent can strategize effectively.
