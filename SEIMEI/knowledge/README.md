# SEIMEI Knowledge Generation

The knowledge module bootstraps reusable guidance snippets for each built-in agent.  
Use `generate.py` to query an OpenAI-compatible model and persist the results as a CSV file.

## Quick start
- Ensure you have network access and an API key exposed via `OPENAI_API_KEY` (or pass `--api-key` explicitly).
- Run the generator from the repository root:
  ```bash
  python -m seimei.knowledge.generate --count 25 --output data/knowledge.csv
  ```
- The script reads the default prompt at `seimei/knowledge/generators/prompt_test1.md`, requests roughly `count` entries per agent, and writes a CSV with columns `agent`, `knowledge`, and `tags`.

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
