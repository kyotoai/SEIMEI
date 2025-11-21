import asyncio
from seimei import seimei

async def demo_code_act():
    orchestrator = seimei(
        agent_config=[{"file_path": "seimei/agents/code_act.py"}],
        llm_kwargs={"model": "gpt-5-nano"},
        allow_code_exec=True,
        #allowed_commands=["ls", "echo"],
        agent_log_head_lines=1,
        max_tokens_per_question=20000,
    )

    result = await orchestrator(
        messages=[
            {"role": "system", "content": "Get at least around 5 steps of agent outputs and make the answer."},
            {"role": "user", "content": "Analyze the files inside the current folder using python code and tell me what's SEIMEI."},
        ],
        knowledge_config={
            "generate_knowledge": True,
            "save_knowledge_path": "seimei_knowledge/knowledge.csv",
        },
    )
    # The code_act reply is stored as the last agent message
    # print(result["msg_history"][-2]["content"])

asyncio.run(demo_code_act())
