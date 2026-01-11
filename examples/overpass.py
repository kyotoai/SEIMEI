import asyncio

from seimei import seimei


async def demo_overpass():
    """Minimal example that routes a single request through the Overpass agent."""
    orchestrator = seimei(
        agent_config=[{"file_path": "seimei/agents/overpass.py"}, {"file_path": "seimei/agents/answer.py"}],
        llm_config={"model": "gpt-5-nano"},
        max_steps=3,
        max_tokens_per_question=30000,
    )

    messages = [
        {"role": "system", "content": "You analyze nearby buildings using the Overpass API."},
        {
            "role": "user",
            "content": (
                "Use Overpass to summarize buildings around lat=34.9855, lon=135.7588 within 800 meters. "
                "Report height coverage and notable examples."
            ),
        },
    ]

    result = await orchestrator(messages=messages)
    final_entry = result["msg_history"][-1]
    print(final_entry.get("content", "No response captured."))


if __name__ == "__main__":
    asyncio.run(demo_overpass())
