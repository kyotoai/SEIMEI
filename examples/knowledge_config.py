import asyncio

from seimei import seimei


async def demo_knowledge_config() -> None:
    orchestrator = seimei(
        agent_config=[{"file_path": "seimei/agents/code_act.py"}],
        llm_config={"model": "gpt-5-nano"},
        allow_code_exec=True,
        agent_log_head_lines=2,
        max_tokens_per_question=30000,
    )

    knowledge_load_config = [
        {"load_knowledge_path": "seimei_knowledge/yc_demo_knowledge4.csv"},
        {
            "step": [1, 2],
            "text": "Break automation plans into numbered steps before executing shell commands.",
            "agent": "code_act",
            "tags": ["code_act", "planning"],
        },
        {
            "step": 3,
            "text": "Verify every command's output before crafting the final summary.",
            "agent": "code_act",
            "tags": ["code_act", "planning"],
        },
    ]
    # knowledge_generate_config = {
    #     "save_knowledge_path": "seimei_knowledge/yc_demo_knowledge4_output.csv",
    #     "knowledge_generation_prompt_path": "seimei/knowledge/prompts/user_intent_alignment3.md",
    # }

    result = await orchestrator(
        messages=[
            {
                "role": "system",
                "content": "You are a meticulous executor who double-checks instructions before replying. Get at least around 4 agent outputs before you make a final answer.",
            },
            {
                "role": "user",
                "content": "List the main files under the repository root and highlight any demo scripts.",
            },
        ],
        knowledge_load_config=knowledge_load_config,
        # knowledge_generate_config=knowledge_generate_config,
    )
    print(result["output"])
    if result.get("knowledge_result"):
        print("Generated knowledge entries:", result["knowledge_result"].get("count", 0))
    if result.get("generated_knowledge"):
        print("Latest generated knowledge:")
        for entry in result["generated_knowledge"]:
            print(f"- {entry.get('agent', '*')}: {entry.get('knowledge', '')}")


if __name__ == "__main__":
    asyncio.run(demo_knowledge_config())
