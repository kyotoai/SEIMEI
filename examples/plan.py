import asyncio
from seimei import seimei

async def demo_code_act():
    orchestrator = seimei(
        llm_kwargs={"model": "gpt-5-nano"},
        allow_code_exec=True,
        agent_log_head_lines=1,
        max_tokens_per_question=10000,
    )

    result = await orchestrator(
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Think deeply about the user's instruction and provide answer."},
            {"role": "user", "content": "Design a single 7-day endgame plan for my turbulence surrogate project based on my past history."},
        ],
        load_knowledge_path="seimei_knowledge/yc_demo_knowledge2.csv",
    )

asyncio.run(demo_code_act())
