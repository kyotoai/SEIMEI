import asyncio
from seimei import seimei

async def demo_edit_file():
    orchestrator = seimei(
        agent_config=[{"name": "edit_file"}, {"name": "think"}, {"name": "code_act"}],
        llm_config={"model": "gpt-5-nano"},
        max_tokens_per_question=80000,
    )

    result = await orchestrator(
        messages=[
            {"role": "system", "content": "In the first several steps, you should see file's content by code_act agents and use edit_file agent after that."},
            {"role": "user", "content": "See README.md and modify grammer errors."},
        ],
    )

asyncio.run(demo_edit_file())
