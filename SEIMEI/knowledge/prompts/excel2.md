# Knowledge Generation Prompt

You are inspecting a completed agent run that contains the full dialogue, agent
steps, and outputs. Use this context to generate useful advice and knowledge about
the agent inference so future agent system can reason more deeply and accurately about the user question.

<<RUN_CONTEXT>>

## Task
- Capture command strategies that surface *novel* numerical relationships (e.g., distribution,
  growth, clustering, anomaly scores, hidden parameters, generator functions) hidden behind raw values.
- Prefer focused computations that interpret what the numbers mean for the domain (trend
  explanations, correlations, thresholds) rather than just listing statistics.
- Encourage concise commands whose outputs remain readable, yet deliver deep insights
  about why certain figures matter or how they should influence decisions.
- Translate these observations into actionable reminders that help `code_act` uncover the
  semantic meaning behind numbers in future CSV analyses.

## Tags
- Tags should include how important the knowledge is. Ex. "important", "very important in ...", "useful in ...", ...
- Tags should include the type of knowledge. Ex. "constraints of ...", "condition for ...", "advice about ..." ...

## Agents
- Pick agent from `think`, `code_act`, `web_search` or `answer` depending on which agent the generated knowledge is about.

## Output format
- Return a JSON array.
- Use the following schema:

Example:

```json
[
  {
    "agent": "code_act",
    "knowledge": "Try to make python function that generates the csv file and see the difference from original",
    "tags": ["important in finding a relation behind", "advice to get hidden parameters behind csv file"]
  }
]
```

Rules:
* Output only the JSON array—no commentary outside the code block.
