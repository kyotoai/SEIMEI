import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="https://v5710arnysphb8-8000.proxy.runpod.net/v1", api_key="EMPTY") 

async def one(prompt: str):
    r = await client.chat.completions.create(
        model="/workspace/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return r.choices[0].message.content

async def main():
    prompts = [f"Give me one insight about number {i}." for i in range(10)]
    # fire 20 requests concurrently
    results = await asyncio.gather(*(one(p) for p in prompts))
    for i, out in enumerate(results):
        print(i, out[:120])

asyncio.run(main())