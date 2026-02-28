import asyncio
from seimei import seimei

async def demo_web_search():
    orchestrator = seimei(
        llm_config={"model": "gpt-5-nano"},
        agent_log_head_lines=2,
        max_tokens_per_question=50000,
    )

    result = await orchestrator(
        messages=[
            {"role": "system", "content": "You gather concise search summaries. If you encounter same search outputs, try to change the query so that you can get more various output."},
            {"role": "user", "content": "Paraphrase the first sentence in the introduction of https://arxiv.org/pdf/2511.17072 "},
        ],
        knowledge_load_config=[
            {"load_knowledge_path": "seimei_knowledge/knowledge.csv"},
            {
                "step": [1, 2],
                "agent": "web_search",
                "text": "Search the web to find relevant information.",
                "tags": ["safety", "planning"],
            },
            {
                "step": 3,
                "agent": "answer",
                "text": "Answer question based on web-search results.",
                "tags": ["quality"],
            },
        ],        
    )
    #print(result["msg_history"][-2]["content"])

asyncio.run(demo_web_search())

