# CSV Insight Knowledge Prompt

You review the following SEIMEI run transcript and agent logs to synthesise reusable
guidance for analysing CSV files with lightweight Python commands.

<<RUN_CONTEXT>>

## Task
- Capture command strategies that surface *novel* numerical relationships (e.g., ratios,
  growth, clustering, anomaly scores) hidden behind raw values.
- Prefer focused computations that interpret what the numbers mean for the domain (trend
  explanations, correlations, thresholds) rather than just listing statistics.
- Encourage concise commands whose outputs remain readable, yet deliver deep insights
  about why certain figures matter or how they should influence decisions.
- Translate these observations into actionable reminders that help `code_act` uncover the
  semantic meaning behind numbers in future CSV analyses.

## Output format
Return a JSON array, for example:

```json
[
  {
    "agent": "code_act",
    "knowledge": "Example guidance here.",
    "tags": ["csv", "low_tokens"]
  }
]
```

Rules:
- Use lowercase agent identifiers (typically `code_act` for command execution guidance).
- Keep each `knowledge` field under 220 characters; emphasise concrete commands or prompt
  fragments the agent can reuse.
- Include 1–3 short `tags` that capture themes such as `csv`, `profiling`, or `low_tokens`.
- Output only the JSON array—no commentary or markdown outside the code block above.
