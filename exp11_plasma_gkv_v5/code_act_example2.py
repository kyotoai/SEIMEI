import asyncio
from seimei import seimei

async def demo_code_act():
    orchestrator = seimei(
        agent_config=[{"name": "code_act"}, {"name": "think"}, {"name": "answer"}],
        llm_config={"base_url":"https://rrg48ysue3kx3u-8000.proxy.runpod.net/v1", "model":"/workspace/gpt-oss-20b"},
        rm_config={"base_url":"https://yjdqlhgc1318ei-8000.proxy.runpod.net/rmsearch"},
        allow_code_exec=True,
        max_tokens_per_question=20000,
    )

    result = await orchestrator(
        messages=[
            {"role": "user", "content": "Analyze seimei.py and llm.py and tell me what this library does."},
        ],
        agent_search_mode = "klg",
        knowledge_search_mode = "rm",
        knowledge_load_config=[
            {"load_knowledge_path": "exp11_plasma_gkv_v5/knowledge_v6_6_modified.csv",}
        ]
    )
    # The code_act reply is stored as the last agent message
    # print(result["msg_history"][-2]["content"])
    if result.get("generated_knowledge"):
        print("New knowledge rows saved to CSV:")
        for entry in result["generated_knowledge"]:
            tags = entry.get("tags") or []
            tag_suffix = f" ({', '.join(tags)})" if tags else ""
            print(f"- [{entry.get('agent', '*')}] {entry.get('knowledge', '')}{tag_suffix}")

asyncio.run(demo_code_act())
