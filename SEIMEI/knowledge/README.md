# SEIMEI Knowledge Generation

The knowledge module bootstraps reusable guidance snippets for each built-in agent.  
Use `generate_from_generators.py` to query an OpenAI-compatible model and persist the results as a CSV file, or `generate_from_runs.py` to curate knowledge from completed SEIMEI runs.

## Quick start
- Ensure you have network access and an API key exposed via `OPENAI_API_KEY` (or pass `--api-key` explicitly).
- Run the generator from the repository root:
  ```bash
  python -m seimei.knowledge.generate_from_generators --count 25 --output seimei_knowledge/knowledge.csv
  ```
- The script reads the default prompt at `seimei/knowledge/generators/prompt_test1.md`, requests roughly `count` entries per agent, and writes a CSV with columns `agent`, `knowledge`, and `tags`.

To derive knowledge from past runs:
```bash
python -m seimei.knowledge.generate_from_runs run-20251103-203312-0338e318 --save-file-path seimei_knowledge/knowledge.csv
```
The command above inspects the run artefacts stored under `seimei_runs/<run_id>` and appends new rows to the target CSV.

## Useful options
- `--prompt`: Point to a different prompt template.
- `--agents`: Supply a custom list of agent identifiers.
- `--model`, `--base-url`, `--api-key`: Override LLM connection settings (defaults match `seimei.llm.LLMClient`).
- `--system`: Provide a custom system prompt for the LLM call.

## Output structure
Each CSV row stores:
- `agent`: lowercase agent name.
- `knowledge`: actionable statement for that agent.
- `tags`: JSON-encoded list of short keywords.

The generated file can be loaded at runtime and injected into `shared_ctx["knowledge"]` for agents to consume.

## Using knowledge in SEIMEI
- Instantiate the orchestrator with the `save_knowledge_path` argument:
  ```python
  orchestrator = seimei(
      agent_config=[{"dir_path": "seimei/agents"}],
      llm_kwargs={"model": "gpt-4o-mini"},
      save_knowledge_path="seimei_knowledge/knowledge.csv",
  )
  ```
- The file can be CSV, JSON, or JSONL. Entries are grouped by the `agent` field, with `*` acting as a wildcard shared by all agents.
- At runtime the knowledge is available through `shared_ctx["knowledge"]`, and helper utilities (e.g., `get_agent_knowledge`) deliver normalized snippets to each agent.
- When calling the orchestrator you can set `generate_knowledge=True` to append fresh insights from each run into `seimei_knowledge/knowledge.csv` (or a custom path via `save_knowledge_path`). The knowledge store is reloaded automatically after each append.
