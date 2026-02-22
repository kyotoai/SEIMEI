### Knowledge + routing configuration reference

Pass routing/knowledge options to `seimei.__call__` to control each run:

- `agent_search_mode`: `"llm"`, `"rm"`, or `"klg"` (knowledge-first routing).
- `agent_search_config`: list of step overrides. Each entry supports:
  - `mode`: `"llm"`, `"rm"`, or `"klg"`.
  - `step`: int list or string expression (`"<3"`, `">=1,<=2"`, `">2"`).
  - `topk`: optional; must be `1` if provided.
  - `sampling_topk`: optional; number of candidates to retrieve before sampling.
  - `sampling_distribution`: optional; list of non-negative weights by rank.
  - `sampling_distribution_decay_rate`: optional; decay rate to generate weights `[1, r, r^2, ...]` when `sampling_distribution` is absent.
  - `random_sampling_rate`: optional; probability in `[0, 1]` to bypass search and pick uniformly at random.
- `knowledge_search_mode`: `"llm"` or `"rm"`.
- `knowledge_search_config`: list of step overrides. Each entry supports the same fields as `agent_search_config`, but `topk` may be `>1`.
- `knowledge_load_config`: list of inline knowledge/load directives. Each entry supports:
  - `step`: int list or string expression (`"<3"`, `">=1,<=2"`, `">2"`).
  - `load_knowledge_path`: merge a CSV/JSON/JSONL file for the matching steps.
  - `text`: free-form knowledge snippet.
  - `agent`: string or list of agents to receive the knowledge.
  - `tags`: optional list of labels.
  - `id`: optional numeric identifier.
- `knowledge_generate_config`: dict with:
  - `save_knowledge_path`: output CSV path for generated knowledge.
  - `knowledge_generation_prompt_path`: prompt template for retrospection.

Sampling behavior:
- If `random_sampling_rate` triggers, selection is uniform over all candidates for that step (agents constrained by `agent_config` / `knowledge_load_config` and any step routing).
- Otherwise, SEIMEI fetches the top `sampling_topk` results from the chosen mode and then selects `topk` items by rank-weighted sampling. When `topk > 1`, sampling is without replacement.
- If both `sampling_distribution` and `sampling_distribution_decay_rate` are provided, the explicit distribution wins.
- With no sampling fields, selection is deterministic top-k.
- When `topk` is set in a step config, it overrides the `k` requested by the search call for that step.
- When an `agent_search_config` entry sets `mode="klg"` and overlaps a `knowledge_search_config` entry, the agent config overrides the knowledge search mode and sampling params for that step.

Example:

```python
knowledge_load_config = [
    {
        "text": "Prefer concise shell commands when drafting automation plans.",
        "tags": ["code_act", "heuristic"],
        "agent": "think",
    },
    {
        "step": ">=1,<=2",
        "load_knowledge_path": "seimei_knowledge/design_briefs.csv",
    },
    {
        "step": ">2",
        "text": "Before final answers double-check every cited number against the workspace files.",
        "id": 9001,
        "agent": ["think", "answer"],
    },
]
knowledge_generate_config = {
    "save_knowledge_path": "seimei_knowledge/yc_demo_knowledge4_output.csv",
    "knowledge_generation_prompt_path": "seimei/knowledge/prompts/user_intent_alignment3.md",
}
result = await orchestrator(
    messages=dialogue,
    agent_search_mode="klg",
    knowledge_search_mode="rm",
    agent_search_config=[
        {
            "mode": "klg",
            "step": "<3",
            "sampling_topk": 5,
            "sampling_distribution": [1, 0.5, 0.25, 0.125, 0.0625],
            "random_sampling_rate": 0.2,
        }
    ],
    knowledge_search_config=[
        {
            "mode": "rm",
            "step": "<5",
            "topk": 2,
            "sampling_topk": 5,
            "sampling_distribution_decay_rate": 0.5,
            "random_sampling_rate": 0.2,
        }
    ],
    knowledge_load_config=knowledge_load_config,
    knowledge_generate_config=knowledge_generate_config,
)
```

When a knowledge entry defines both `step` and `agent`, SEIMEI constrains routing to those agents for the matching steps (or, in `"klg"` mode, routes via the selected knowledge item).
