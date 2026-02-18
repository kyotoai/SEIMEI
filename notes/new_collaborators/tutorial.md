# Tutorial

## Run SEIMEI using OpenAI API key

### 1. Ask the api key and access to SEIMEI_ (private repo) from kentarrito

### 2. Set up

Set up the API

```bash
export OPENAI_API_KEY = "(your_openai_api_key)"
```

and install prerequisites

```
git clone https://github.com/kyotoai/SEIMEI_.git
mv SEIMEI_ SEIMEI
pip install -e SEIMEI/requirements_developper.txt
```

### 3. Run SEIMEI

```bash
seimei
```

or 

```python
import asyncio
from seimei import seimei

async def demo_code_act():
    orchestrator = seimei(
        llm_config={"model": "gpt-5-nano"},
        max_tokens_per_question=30000,
    )

    result = await orchestrator(
        messages=[
            {"role": "user", "content": "Design a single 7-day endgame plan for my turbulence surrogate project based on my past history."},
        ],
        knowledge_load_config=[
            {"load_knowledge_path": "seimei_knowledge/knowledge.csv"},
        ],
    )

asyncio.run(demo_code_act())
```

## Run SEIMEI using Runpod GPUs

### 1. Prepare models

See runpod.md (only `Host model on runpod` section) for how to prepare models

### 2. SEIMEI set ups

Once you set up the models using vllm serve and get url of the pod, use the models by

```
orchestrator = seimei(
    llm_config={"base_url": "(llm url)"},
    rm_config={"base_url": "(rmsearch url)"},
)
```


## Train





