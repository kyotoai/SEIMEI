import asyncio
from seimei import seimei

async def demo_web_search():
    orchestrator = seimei(
        agent_config=[{"file_path": "seimei/agents/web_search.py"}],
        llm_kwargs={"model": "gpt-5-nano"},
        agent_log_head_lines=2,
        max_tokens_per_question=5000,
    )

    result = await orchestrator(
        messages=[
            {"role": "system", "content": "You gather concise search summaries."},
            {"role": "user", "content": "Search the web for recent applications of perovskite solar cells."},
        ]
    )
    #print(result["msg_history"][-2]["content"])

asyncio.run(demo_web_search())