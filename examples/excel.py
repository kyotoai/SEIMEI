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
            {"role": "system", "content": "You are an execution assistant that never runs unasked commands."},
            {"role": "user", "content": "Analyze exp1/csv/ecommerce_orders_001.csv inside and see some features in the csv file."},
        ],
        knowledge_config={
            "generate_knowledge": True,
            "save_knowledge_path": "seimei_knowledge/excel.csv",
            "knowledge_prompt_path": "seimei/knowledge/prompts/excel.md",
            "load_knowledge_path": "seimei_knowledge/excel.csv",
        },
    )
    # The code_act reply is stored as the last agent message
    # print(result["msg_history"][-2]["content"])
    if result.get("generated_knowledge"):
        print("Generated knowledge (excel prompt):")
        for entry in result["generated_knowledge"]:
            tags = entry.get("tags") or []
            tag_suffix = f" [{', '.join(tags)}]" if tags else ""
            print(f"- {entry.get('agent', '*')}: {entry.get('knowledge', '')}{tag_suffix}")

asyncio.run(demo_code_act())
