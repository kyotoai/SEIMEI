# Document about Runpod

## Host model on runpod

1. Ask for runpod account and password from some KyotoAI member, and go to https://console.runpod.io/pods

2. Open `Pods` -> ` Deploy`

3. Select `volume2` from Network Volume and choose number of gpus

4. Select gpu to use
* For gpt-oss-20b and other 7b~20b models, RTX5000 is good enough
* For small training, A40 * 2 is good enough
* For more expensive gpus, ask for permission to kentaro

5. Edit pod
* When you create pod, `Edit Pod` -> Expose HTTP ports (max 10) `8888` -> `8888, 8000`

6. Deploy pod and connect through jupyter notebook or vscode ssh
* You can refer to https://docs.runpod.io/pods/configuration/use-ssh for how to connect through ssh

7. Set up

Make directory and go there

```
mkdir /workspace/(your_name) # Optional
cd /workspace/(your_name)
```

Install prerequisites
```
pip install -r /workspace/kentarrito/SEIMEI/requirements_developper.txt
```

8. Run vllm serve

For generative models:

```
nohup vllm serve (model_saved_path_on_volume2) --data-parallel-size (number_of_gpus_set_in_step_3) --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

Ex:
```
nohup vllm serve /workspace/gpt-oss-20b --data-parallel-size 1 --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

For reward models:

```
export RMSEARCH_MODEL_NAME=/workspace/qwen4b-reward
export VLLM_USE_V1=0
nohup vllm serve $RMSEARCH_MODEL_NAME \
  --runner pooling --host 0.0.0.0 --port 9000 \
  > server-vllm-reward.log 2>&1 &
nohup uvicorn seimei.rmsearch:app --host 0.0.0.0 --port 8000 > server-rmsearch.log 2>&1 &
```

* You can see log from /workspace/kentarrito/server.log

### Further Note

* If you get some errors in server.log, you should kill the process by the following steps
  1. Run `ps aux | grep nohup`
  2. kill -9 (process_number)
    * You shouldn't kill `... grep --color=auto` process

## Test LLM Request

### From runpod

```
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

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
```


### From your local computer

* You need to copy the url in `Select your pod` -> `Connect` -> `HTTP services` -> `Port 8000` -> `HTTP Service`

```
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="(paste_url_here)/v1", api_key="EMPTY")
# Ex. client = AsyncOpenAI(base_url="https://v5710arnysphb8-8000.proxy.runpod.net/v1", api_key="EMPTY") 

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
```

## LLM Request using SEIMEI

```
# Simple Example

import asyncio
from seimei import seimei

async def demo_code_act():
    orchestrator = seimei(
        agent_config=[{"file_path": "seimei/agents/code_act.py"}],
        llm_kwargs={"base_url": "https://v5710arnysphb8-8000.proxy.runpod.net/v1", "model":"/workspace/gpt-oss-20b"},
        max_tokens_per_question=20000,
    )

    result = await orchestrator(
        messages=[
            {"role": "system", "content": "Get at least around 5 steps of agent outputs and make the answer."},
            {"role": "user", "content": "Analyze the files inside the current folder using python code and tell me what's SEIMEI."},
        ]
    )

asyncio.run(demo_code_act())
```



