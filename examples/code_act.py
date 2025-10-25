import asyncio
from seimei import seimei

async def demo_code_act():
    orchestrator = seimei(
        agent_config=[{"file_path": "seimei/agents/code_act.py"}],
        llm_kwargs={"model": "gpt-5-nano"},
        allow_code_exec=True,
        #allowed_commands=["ls", "echo"],
        agent_log_head_lines=1,
        max_tokens_per_question=10000,
    )

    result = await orchestrator(
        messages=[
            {"role": "system", "content": "You are an execution assistant that never runs unasked commands."},
            {"role": "user", "content": "Analyze the folder where you are now and see all files inside."},
        ]
    )
    # The code_act reply is stored as the last agent message
    # print(result["msg_history"][-2]["content"])

asyncio.run(demo_code_act())