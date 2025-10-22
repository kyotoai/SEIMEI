# seimei usage

## Overview

1. `seimei.py`: seimei class is defined here. 
2. `llm.py`: this handles llm client with openai api, vllm, ...
3. `agent.py`: agent class is defined here. 


## Install SEIMEI

```bash
git clone https://github.com/kyotoai/SEIMEI.git
pip install -e SEIMEI/.
```

## Quick start

```python
import seimei

system = "You're genius mathematician"

problems = [
    "Three airline companies operate flights from Dodola island. Each company has a different schedule of departures. The first company departs every 100 days, the second every 120 days and the third every 150 days. What is the greatest positive integer $d$ for which it is true that there will be $d$ consecutive days without a flight from Dodola island, regardless of the departure times of the various airlines?",
    "Fred and George take part in a tennis tournament with $4046$ other players. In each round, the players are paired into $2024$ matches. How many ways are there to arrange the first round such that Fred and George do not have to play each other? (Two arrangements for the first round are \textit{different} if there is a player with a different opponent in the two arrangements.)",
]

agent_config = [
    {
        "dir_path" : "../agents/test", # can be either folder or file
    }
]

llm_kwargs = {
    "model": "gpt-5-nano",
    "max_inference_time": 1000,
    # other llm kwargs
}

rm_kwargs = {
    "base_url": "http://localhost:7000/v1",
}

'''vllm version
llm_kwargs = {
    "base_url": "http://localhost:7000/v1",
    "max_inference_time": 1000,
    # other llm kwargs
}
'''

sm = seimei(
    agent_config = agent_config,
    llm_kwargs = llm_kwargs,
    rm_kwargs = rm_kwargs,
)

results = await asyncio.gather(
    *[
        sm(
            messages=[
                {"role":"system", "content":system},
                {"role":"user", "content":problem}
            ]
        ) for problem in problems
    ]
)

'''
results = [
    {"output":str, "msg_history": [{"role":"system", ...}, {"role":"user", ...}, {"role":"agent", "content":"...", "code":"...", "chosen_instructions":[{}], "name":"default.code_act"}, {"role":"agent","name":"default.think", "content":"..."}, ... {"role":"agent",}, {"role":"assistant", "content":"..."}]}
]
'''

print(results[0]["output"])
```


## Create agent

You can create agents to help llm inference

```python
from seimei import seimei, llm, agent
from rmsearch import rmsearch
import asyncio

# Your custom agent should inherit agent class
class think(agent):

    description = "This agent will think about the current message and decide what to do next."

    def inference(self, messages):

        think_instructions = [...]

        top_inst = await rmsearch(query = messages, keys = think_instructions, k_key = 1)

        system = "You're an excellent thinker who plans next action brilliantly. Following the instruction, think what to do next."

        prompt = f"""Instruction: {top_inst[0]["key"]}"""

        next_msg = {"role":"agent", "log":"think.test", "content":prompt}

        messages.append(next_msg)

        answer, updated_msgs = await llm(messages, system = system)

        # updated_msgs

        return {"content":answer}
```



## `seimei.__init__`

**Example**

```bash
(usage)
```

**Explanation**

**Arguments**
- `--agent-config`: ...


**Outputs**
- ...

**Notices**
- ...



## `seimei.__call__`

(same format as seimei.__init__)