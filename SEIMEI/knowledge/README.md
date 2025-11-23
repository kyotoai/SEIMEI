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
By default it loads the retrospection prompt at `seimei/knowledge/prompts/generate_from_runs.md`, but you can pass
`--prompt seimei/knowledge/prompts/excel.md` (or another custom file) for task-specific guidance.

## Useful options
- `--prompt`: Point to a different prompt template (defaults to `seimei/knowledge/prompts/generate_from_runs.md`).
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
- Instantiate the orchestrator as usual and pass a `knowledge_config` dictionary whenever you invoke it:
  ```python
  result = await orchestrator(
      messages=[{"role": "user", "content": "Summarize the ETL runbook."}],
      knowledge_config={
          "load_knowledge_path": "seimei_knowledge/knowledge.csv",
          "generate_knowledge": True,
          "save_knowledge_path": "seimei_knowledge/knowledge_output.csv",
          "knowledge": [
              {"text": "Favor pandas for quick CSV exploration.", "tags": ["code_act"]},
              {"step": 2, "load_knowledge_path": "seimei_knowledge/audit_notes.csv"},
          ],
      },
  )
  ```
- The referenced files can be CSV, JSON, or JSONL. Entries are grouped by the `agent` field, with `*` acting as a wildcard shared by all agents.
- Inline knowledge entries accept optional `step`, `id`, `load_knowledge_path`, `text`, and `tags` fields. When `step` is omitted, the knowledge applies to every agent step; otherwise it is injected only when the specified step runs.
- At runtime the knowledge is available through `shared_ctx["knowledge"]`, and helper utilities (e.g., `get_agent_knowledge`) deliver normalized snippets to each agent.
- Automatic retrospectives run whenever `knowledge_config["generate_knowledge"]` is true. The helper appends fresh insights to `save_knowledge_path` (defaulting to the last load path) and the orchestrator transparently reloads the store.
- Successful retrospectives now surface their payloads directly in the call result: `result["knowledge_result"]` contains metadata (save path, usage, run IDs) and `result["generated_knowledge"]` lists the exact entries that were written, which is handy for quick manual review.
