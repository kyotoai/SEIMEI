import asyncio
from seimei import seimei

async def demo_code_act():
    orchestrator = seimei(
        #agent_config=[{"file_path": "seimei/agents/code_act.py"}],
        llm_kwargs={"model": "gpt-5-nano"}, #"base_url": "https://0lhuputae2fexu-8000.proxy.runpod.net/generate"},
        rm_kwargs={"url": "https://phpb5lzs0u9reb-8000.proxy.runpod.net/rmsearch", "agent_routing":False, "knowledge_search":True},
        allow_code_exec=True,
        agent_log_head_lines=1,
        max_tokens_per_question=30000,
        #load_knowledge_path="seimei_knowledge/yc_demo_knowledge2.csv",
    )

    result = await orchestrator(
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Think deeply about the user's instruction and provide answer."},
            {"role": "user", "content": "Design a single 7-day endgame plan for my turbulence surrogate project."},
        ],
        #generate_knowledge=True,
        #save_knowledge_path="seimei_knowledge/excel.csv",
        #knowledge_prompt_path="seimei/knowledge/prompts/excel.md",
    )
    # The code_act reply is stored as the last agent message
    # print(result["msg_history"][-2]["content"])

asyncio.run(demo_code_act())
