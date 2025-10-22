import asyncio
from seimei import seimei  # class name is `seimei` (lowercase) for convenience

system = "You're a genius mathematician."

problems = [
    "Three airline companies operate flights from Dodola island. Each company has a different schedule of departures. The first company departs every 100 days, the second every 120 days and the third every 150 days. What is the greatest positive integer d for which it is true that there will be d consecutive days without a flight from Dodola island, regardless of the departure times of the various airlines?",
    "Fred and George take part in a tennis tournament with 4046 other players. In each round, the players are paired into 2024 matches. How many ways are there to arrange the first round such that Fred and George do not have to play each other?",
]

agent_config = [
    {"dir_path": "./agents"},  # can be a folder or a single file path
]

llm_kwargs = {
    "model": "gpt-5-nano",
    # "max_inference_time": 1000,
    # OR talk to a local OpenAI-compatible server:
    # "base_url": "http://localhost:7000/v1"
}

rm_kwargs = {
    "base_url": "http://localhost:7000/v1",  # optional for `rmsearch` if you use it
}

sm = seimei(
    agent_config=agent_config,
    llm_kwargs=llm_kwargs,
    #rm_kwargs=rm_kwargs,
    log_dir="./seimei_runs",
)

async def main():
    results = await asyncio.gather(
        *[
            sm(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": problem},
                ]
            )
            for problem in problems
        ]
    )
    print(results)

asyncio.run(main())