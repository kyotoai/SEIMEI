from __future__ import annotations

# this is about batch processing
# when you wanna data parallel

"""
RMSearch FastAPI service exposing reward-model ranking over a REST endpoint.

This version is concurrency-safe by delegating embedding work to a vLLM server
(runner=pooling) via the /pooling endpoint. Your FastAPI service stays a thin
wrapper and preserves the SAME input/output format as before.

Typical setup:
  1) Start vLLM pooling server (example):
     CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/model \
       --runner pooling --host 0.0.0.0 --port 8000 \
       --tensor-parallel-size 1 --pipeline-parallel-size 2

  2) Start this API (override defaults via env vars on the command line):
     RMSEARCH_MODEL_NAME=/path/to/model \
     RMSEARCH_SCORE_PATH=/path/to/model/score.pt \
     RMSEARCH_VLLM_BASE_URL=http://0.0.0.0:8000 \
     uvicorn seimei.rmsearch:app --host 0.0.0.0 --port 9000
Ex.
```
export RMSEARCH_MODEL_NAME=/workspace/qwen4b-reward-exp11-model3-1480

nohup vllm serve $RMSEARCH_MODEL_NAME \
  --runner pooling --host 0.0.0.0 --port 9000 --data-parallel-size 1 \
  > server-vllm-reward.log 2>&1 &

nohup uvicorn seimei.rmsearch:app --host 0.0.0.0 --port 8000 > server-rmsearch.log 2>&1 &
```


Input Format:
- queries: list of query strings or chat-style objects
  - string: treated as a raw query body (can already include <query>...</query>)
  - object: {"message": [{"role": "...", "content": "..."}, ...]} (also accepts "messages")
- keys: list of candidate strings (raw or already formatted <key>...</key>)
- k: top-k results per query (default 5)
- batch_size: max in-flight embedding requests per chunk (optional)
- query_batch_size: max queries expanded per chunk (optional)
- Optional overrides per request: model, score_path, vllm_base_url

If input formatting is invalid, the server responds with HTTP 400 and a
human-readable "details" list explaining what went wrong.

Ex.
{
    "queries": ["How to tune a reward model?", "What is LLM?"],
    "keys": ["Reward models score sequences.", "LLM is large language model"],
    "k": 2
}


Output Format:
- output: list of per-query results (same order as input queries)
- queries/results: aliases of output for backward compatibility
- usage: simple counters (tokenCount if provided by vLLM, requestCount, pairCount, etc.)

Ex.
{
   "output":[
      {
         "query":"How to tune a reward model?",
         "query_id":0,
         "keys":[
            {
               "key_id":0,
               "key":"Reward models score sequences.",
               "relevance":-0.031213754788041115
            },
            {
               "key_id":1,
               "key":"LLM is large language model",
               "relevance":-0.07688609510660172
            }
         ]
      },
      {
         "query":"What is LLM?",
         "query_id":1,
         "keys":[
            {
               "key_id":1,
               "key":"LLM is large language model",
               "relevance":0.0337870717048645
            },
            {
               "key_id":0,
               "key":"Reward models score sequences.",
               "relevance":-0.11700551956892014
            }
         ]
      }
   ],
   "usage":{
      "tokenCount":6,
      "someOtherMetric":65
   }
}

"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal

import torch
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessageParam
from openai.types.create_embedding_response import CreateEmbeddingResponse
from pydantic import BaseModel, Field

DEFAULT_MODEL = os.getenv("RMSEARCH_MODEL_NAME", "/workspace/qwen4b-reward-converted-model").strip()
DEFAULT_SCORE_PATH = os.getenv("RMSEARCH_SCORE_PATH", os.path.join(DEFAULT_MODEL, "score.pt")).strip()
DEFAULT_VLLM_BASE_URL = os.getenv("RMSEARCH_VLLM_BASE_URL", "http://0.0.0.0:9000").strip()

_CLIENT_CACHE: Dict[str, AsyncOpenAI] = {}
_SCORE_CACHE: Dict[str, Any] = {}

app = FastAPI()


def _normalize_base_url(base_url: str) -> str:
    base = (base_url or "").strip()
    if not base:
        raise ValueError("vllm_base_url is empty")
    if base.endswith("/v1"):
        return base
    return base.rstrip("/") + "/v1"


def _get_client(base_url: str) -> AsyncOpenAI:
    normalized = _normalize_base_url(base_url)
    client = _CLIENT_CACHE.get(normalized)
    if client is None:
        client = AsyncOpenAI(base_url=normalized, api_key="EMPTY")
        _CLIENT_CACHE[normalized] = client
    return client


def _load_score(score_path: str) -> Any:
    path = (score_path or "").strip()
    if not path:
        raise ValueError("score_path is empty")
    cached = _SCORE_CACHE.get(path)
    if cached is None:
        score_obj = torch.load(path, map_location="cpu")
        if hasattr(score_obj, "eval"):
            score_obj.eval()
        _SCORE_CACHE[path] = score_obj
        cached = score_obj
    return cached


async def create_chat_embeddings(
    client: AsyncOpenAI,
    *,
    messages: List[ChatCompletionMessageParam],
    model: str,
    encoding_format: Union[Literal["base64", "float"], NotGiven] = NOT_GIVEN,
) -> CreateEmbeddingResponse:
    return await client.post(
        "/embeddings",
        cast_to=CreateEmbeddingResponse,
        body={"messages": messages, "model": model, "encoding_format": encoding_format},
    )



# ── Request / Response schemas ───────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatQuery(BaseModel):
    message: List[ChatMessage]


QueryInput = Any

class SearchRequest(BaseModel):
    queries: List[QueryInput]
    keys: Optional[List[Any]] = None
    k: int = Field(default=5, ge=1)
    batch_size: Optional[int] = Field(default=None, ge=1)         # encode batch size
    query_batch_size: Optional[int] = Field(default=None, ge=1)   # template+encode chunk size
    model: Optional[str] = None
    score_path: Optional[str] = None
    vllm_base_url: Optional[str] = None


class KeyOut(BaseModel):
    key_id: int
    key: str
    relevance: float


class QueryOut(BaseModel):
    query: str
    query_id: int
    keys: List[KeyOut]


class UsageOut(BaseModel):
    tokenCount: int
    requestCount: int
    pairCount: int
    queryCount: int
    keyCount: int


class SearchResponse(BaseModel):
    output: List[QueryOut]
    queries: Optional[List[QueryOut]] = None
    results: Optional[List[QueryOut]] = None
    usage: UsageOut


def _format_query_block(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("<query>") and "</query>" in cleaned:
        return cleaned
    if not cleaned:
        cleaned = "[missing query context]"
    return f"<query>\n{cleaned}\n</query>"


def _format_key_block(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("<key>") and "</key>" in cleaned:
        formatted = cleaned
    else:
        if not cleaned:
            cleaned = "[missing key text]"
        formatted = f"<key>\n{cleaned}\n</key>"
    if "Query-Key Relevance Score:" not in formatted:
        formatted = f"{formatted}\n\n\nQuery-Key Relevance Score:"
    return formatted


def _messages_to_text(messages: Sequence[Any], *, idx: int) -> str:
    lines: List[str] = []
    for msg_idx, msg in enumerate(messages):
        if isinstance(msg, ChatMessage):
            role = str(msg.role or "").strip()
            content = str(msg.content or "").strip()
        elif isinstance(msg, dict):
            role = str(msg.get("role") or "").strip()
            content = str(msg.get("content") or "").strip()
        else:
            raise ValueError(f"queries[{idx}] message[{msg_idx}] must be an object with role/content")
        if not content:
            raise ValueError(f"queries[{idx}] message[{msg_idx}] content is empty")
        if not role:
            role = "user"
        lines.append(f"{role}: {content}")
    joined = "\n".join(lines).strip()
    if not joined:
        raise ValueError(f"queries[{idx}] has no usable message content")
    return joined


def _coerce_query(item: Any, *, idx: int) -> str:
    if isinstance(item, str):
        text = item.strip()
        if not text:
            raise ValueError(f"queries[{idx}] is an empty string")
        return text
    if isinstance(item, ChatQuery):
        return _messages_to_text(item.message, idx=idx)
    if isinstance(item, dict):
        if "message" in item:
            messages = item.get("message")
        elif "messages" in item:
            messages = item.get("messages")
        else:
            raise ValueError(f"queries[{idx}] object must include 'message' or 'messages'")
        if not isinstance(messages, Sequence):
            raise ValueError(f"queries[{idx}] messages must be a list")
        return _messages_to_text(messages, idx=idx)
    if isinstance(item, Sequence):
        return _messages_to_text(item, idx=idx)
    raise ValueError(f"queries[{idx}] must be a string or chat message list")


def _coerce_key(item: Any, *, idx: int) -> str:
    if not isinstance(item, str):
        raise ValueError(f"keys[{idx}] must be a string")
    text = item.strip()
    if not text:
        raise ValueError(f"keys[{idx}] is an empty string")
    return text


def _score_embedding(embedding: Sequence[float], score_obj: Any) -> float:
    emb = torch.tensor(embedding, dtype=torch.float32)
    with torch.no_grad():
        if isinstance(score_obj, torch.nn.Module):
            out = score_obj(emb)
        elif isinstance(score_obj, dict) and "weight" in score_obj:
            weight = torch.as_tensor(score_obj.get("weight"), dtype=torch.float32)
            bias = score_obj.get("bias")
            if weight.ndim == 2 and weight.shape[0] == 1:
                out = torch.matmul(weight, emb)
            elif weight.ndim == 2 and weight.shape[1] == 1:
                out = torch.matmul(emb, weight)
            else:
                out = torch.matmul(emb, weight.T if weight.ndim == 2 else weight)
            if bias is not None:
                out = out + torch.as_tensor(bias, dtype=torch.float32)
        else:
            weight = torch.as_tensor(score_obj, dtype=torch.float32)
            if weight.ndim == 1:
                out = torch.dot(emb, weight)
            elif weight.ndim == 2:
                if weight.shape[0] == 1:
                    out = torch.matmul(weight, emb)
                elif weight.shape[1] == 1:
                    out = torch.matmul(emb, weight)
                elif weight.shape[1] == emb.shape[0]:
                    out = torch.matmul(emb, weight.T)
                elif weight.shape[0] == emb.shape[0]:
                    out = torch.matmul(weight, emb)
                else:
                    raise ValueError("score tensor shape does not match embedding")
            else:
                raise ValueError("score tensor has unsupported dimensions")
    return float(out.squeeze().item())


def _extract_usage_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    for attr in ("total_tokens", "prompt_tokens"):
        value = getattr(usage, attr, None)
        if isinstance(value, int):
            return value
    return 0


def _chunk_indices(count: int, size: Optional[int]) -> List[Tuple[int, int]]:
    if not size or size <= 0:
        return [(0, count)]
    return [(start, min(start + size, count)) for start in range(0, count, size)]


# ── Endpoint ─────────────────────────────────────────────────────────────────
@app.post("/rmsearch", response_model=SearchResponse)
async def rmsearch(req: SearchRequest) -> SearchResponse:
    errors: List[str] = []
    if not req.queries:
        errors.append("queries must be a non-empty list")
    if not req.keys:
        errors.append("keys must be a non-empty list of strings")
    if errors:
        raise HTTPException(status_code=400, detail={"error": "Invalid rmsearch input", "details": errors})

    query_texts: List[str] = []
    for idx, item in enumerate(req.queries):
        try:
            query_texts.append(_coerce_query(item, idx=idx))
        except ValueError as exc:
            errors.append(str(exc))

    key_texts: List[str] = []
    for idx, item in enumerate(req.keys or []):
        try:
            key_texts.append(_coerce_key(item, idx=idx))
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        raise HTTPException(status_code=400, detail={"error": "Invalid rmsearch input", "details": errors})

    if not key_texts or not query_texts:
        raise HTTPException(status_code=400, detail={"error": "Invalid rmsearch input", "details": ["queries and keys cannot be empty"]})

    k = min(req.k, len(key_texts))
    model_name = (req.model or DEFAULT_MODEL).strip()
    score_path = (req.score_path or DEFAULT_SCORE_PATH).strip()
    base_url = (req.vllm_base_url or DEFAULT_VLLM_BASE_URL).strip()
    if not model_name:
        raise HTTPException(status_code=500, detail={"error": "RMSearch configuration error", "details": ["model name is empty"]})

    try:
        score_obj = _load_score(score_path)
        client = _get_client(base_url)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "RMSearch configuration error", "details": [str(exc)]})

    query_blocks = [_format_query_block(text) for text in query_texts]
    key_blocks = [_format_key_block(text) for text in key_texts]
    query_chunks = _chunk_indices(len(query_texts), req.query_batch_size)
    batch_size = req.batch_size

    scored: Dict[int, List[Tuple[int, float]]] = {idx: [] for idx in range(len(query_texts))}
    token_count = 0
    request_count = 0

    for q_start, q_end in query_chunks:
        prompts: List[str] = []
        metadata: List[Tuple[int, int]] = []
        for q_idx in range(q_start, q_end):
            query_block = query_blocks[q_idx]
            for k_idx, key_block in enumerate(key_blocks):
                prompts.append(f"{query_block}\n\n\n{key_block}")
                metadata.append((q_idx, k_idx))

        for p_start, p_end in _chunk_indices(len(prompts), batch_size):
            batch_prompts = prompts[p_start:p_end]
            batch_meta = metadata[p_start:p_end]
            requests = [
                create_chat_embeddings(
                    client,
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    encoding_format="float",
                )
                for prompt in batch_prompts
            ]
            results = await asyncio.gather(*requests, return_exceptions=True)
            request_count += len(results)
            for result, (q_idx, k_idx) in zip(results, batch_meta):
                if isinstance(result, Exception):
                    raise HTTPException(
                        status_code=502,
                        detail={"error": "Embedding request failed", "details": [str(result)]},
                    )
                token_count += _extract_usage_tokens(result)
                try:
                    embedding = result.data[0].embedding
                except Exception as exc:
                    raise HTTPException(
                        status_code=502,
                        detail={"error": "Embedding response malformed", "details": [str(exc)]},
                    )
                try:
                    relevance = _score_embedding(embedding, score_obj)
                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail={"error": "Reward score computation failed", "details": [str(exc)]},
                    )
                scored[q_idx].append((k_idx, relevance))

    output: List[QueryOut] = []
    for q_idx, query_text in enumerate(query_texts):
        ranked = sorted(scored[q_idx], key=lambda item: item[1], reverse=True)[:k]
        keys_out = [
            KeyOut(key_id=key_idx, key=key_texts[key_idx], relevance=score)
            for key_idx, score in ranked
        ]
        output.append(QueryOut(query=query_text, query_id=q_idx, keys=keys_out))

    usage = UsageOut(
        tokenCount=token_count,
        requestCount=request_count,
        pairCount=len(query_texts) * len(key_texts),
        queryCount=len(query_texts),
        keyCount=len(key_texts),
    )

    return SearchResponse(output=output, queries=output, results=output, usage=usage)


__all__ = ["app"]
