## Make Your Own Agent

### Built-in Agent Demos

SEIMEI ships with two lightweight reference agents under `seimei/agents`. The snippets below show end-to-end runs for each agent with sample questions you can adapt.

#### `code_act`: controlled shell execution

Sample question — *"Run `ls` in the workspace and report the output."*

```python
import asyncio
from seimei import seimei

async def demo_code_act():
    orchestrator = seimei(
        agent_config=[{"name": "code_act"}],
        llm_config={"model": "gpt-4o-mini"},
        allow_code_exec=True,
        allowed_commands=["ls", "echo"],
        agent_log_head_lines=1,
        max_tokens_per_question=2000,
    )

    result = await orchestrator(
        messages=[
            {"role": "system", "content": "You are an execution assistant that never runs unasked commands."},
            {"role": "user", "content": "Run ```bash\nls\n``` and summarize the stdout."},
        ]
    )
    # The code_act reply is stored as the last agent message
    print(result["msg_history"][-2]["content"])

asyncio.run(demo_code_act())
```

#### `web_search`: fast fact gathering

Sample question — *"What are three recent applications of perovskite solar cells?"*

> Provide Google Custom Search credentials via `GOOGLE_CSE_API_KEY` and `GOOGLE_CSE_CX` to use Google's API.  
> Without those variables, install `pip install duckduckgo_search` for the DuckDuckGo fallback.

```python
import asyncio
from seimei import seimei

async def demo_web_search():
    orchestrator = seimei(
        agent_config=[{"name": "web_search"}],
        llm_config={"model": "gpt-4o-mini"},
        agent_log_head_lines=2,
        max_tokens_per_question=4000,
    )

    result = await orchestrator(
        messages=[
            {"role": "system", "content": "You gather concise search summaries."},
            {"role": "user", "content": "Search the web for recent applications of perovskite solar cells."},
        ]
    )
    print(result["msg_history"][-2]["content"])

asyncio.run(demo_web_search())
```

Google Custom Search setup (optional but recommended):

1. Enable the [Programmable Search Engine](https://programmablesearchengine.google.com/) and create a search engine that crawls the public web. Copy its Search Engine ID (`cx`).
2. Create an API key in Google Cloud (`APIs & Services → Credentials`) with access to the Custom Search API.
3. Export both values before running SEIMEI:

```bash
export GOOGLE_CSE_API_KEY="your-google-key"
export GOOGLE_CSE_CX="your-search-engine-id"
```

When these variables are present the agent queries Google first; if they are missing or the API call fails it gracefully falls back to DuckDuckGo.
