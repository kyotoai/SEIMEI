import asyncio
from seimei import LLMClient, seimei, format_agent_history, TokenLimiter
from typing import Any, Dict, List, Optional, Sequence, Tuple

def _stringify_content(value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value)

class QwenVL_LLMClient(LLMClient):
    """Qwen3-VL local client that preserves multimodal message blocks."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_concurrent_requests: Optional[int] = None,
        processor_path: Optional[str] = None,
        torch_dtype: Any = "auto",
        device_map: Any = "auto",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = True,
        max_new_tokens: int = 128,
        **kwargs: Any,
    ) -> None:
        generation_kwargs = kwargs.pop("generation_kwargs", None) or {}
        model_kwargs = kwargs.pop("model_kwargs", None) or {}
        processor_kwargs = kwargs.pop("processor_kwargs", None) or {}
        resolved_base_url = base_url if base_url not in (None, "") else "http://localhost/qwen3-vl"
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=resolved_base_url,
            timeout=timeout,
            max_concurrent_requests=max_concurrent_requests,
            **kwargs,
        )

        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "QwenVL_LLMClient requires transformers. Install with: pip install transformers"
            ) from exc
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("QwenVL_LLMClient requires torch. Install with: pip install torch") from exc

        self._torch = torch
        self.processor_path = processor_path or model
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self.generation_kwargs = dict(generation_kwargs)

        resolved_model_kwargs = dict(model_kwargs)
        if torch_dtype is not None:
            resolved_model_kwargs.setdefault("dtype", torch_dtype)
        if device_map is not None:
            resolved_model_kwargs.setdefault("device_map", device_map)
        if attn_implementation:
            resolved_model_kwargs.setdefault("attn_implementation", attn_implementation)
        resolved_model_kwargs.setdefault("trust_remote_code", trust_remote_code)

        self.model_engine = Qwen3VLForConditionalGeneration.from_pretrained(
            model,
            **resolved_model_kwargs,
        )
        resolved_processor_kwargs = dict(processor_kwargs)
        resolved_processor_kwargs.setdefault("trust_remote_code", trust_remote_code)
        self.processor = AutoProcessor.from_pretrained(self.processor_path, **resolved_processor_kwargs)

    @staticmethod
    def _is_agent_message(msg: Dict[str, Any]) -> bool:
        role = str(msg.get("role") or "").lower()
        if role == "agent":
            return True
        return role == "system" and bool(msg.get("agent"))

    @staticmethod
    def _clone_content(content: Any) -> Any:
        if isinstance(content, list):
            cloned: List[Any] = []
            for item in content:
                if isinstance(item, dict):
                    cloned.append(dict(item))
                elif item is not None:
                    text = str(item).strip()
                    if text:
                        cloned.append({"type": "text", "text": text})
            return cloned
        if isinstance(content, dict):
            return _stringify_content(content)
        if content is None:
            return ""
        return str(content)

    def _prepare_qwen_messages(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        system: Optional[str],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for raw in messages or []:
            if not isinstance(raw, dict):
                raise Exception("messages must be list of dict but it has non dict in it.")
            if not raw.get("role"):
                raise Exception("dict in messages must include role.")
            normalized.append(raw)

        last_user_idx: Optional[int] = None
        for idx, msg in enumerate(normalized):
            if str(msg.get("role") or "").lower() == "user":
                last_user_idx = idx

        trailing_agent_messages: List[Dict[str, Any]] = []
        if last_user_idx is not None:
            for msg in normalized[last_user_idx + 1 :]:
                if self._is_agent_message(msg):
                    trailing_agent_messages.append({"content": msg.get("content")})

        prepared: List[Dict[str, Any]] = []
        for msg in normalized:
            if self._is_agent_message(msg):
                continue
            role = str(msg.get("role") or "user").lower()
            if role == "developer":
                role = "system"
            if role not in ("system", "user", "assistant"):
                role = "user"
            prepared.append(
                {
                    "role": role,
                    "content": self._clone_content(msg.get("content", "")),
                }
            )

        if system not in (None, ""):
            prepared.insert(0, {"role": "system", "content": str(system)})

        agent_history = format_agent_history(trailing_agent_messages)
        if agent_history:
            last_user_prepared_idx: Optional[int] = None
            for idx in range(len(prepared) - 1, -1, -1):
                if prepared[idx].get("role") == "user":
                    last_user_prepared_idx = idx
                    break
            if last_user_prepared_idx is None:
                prepared.append({"role": "user", "content": agent_history})
            else:
                target = prepared[last_user_prepared_idx]
                content = target.get("content", "")
                if isinstance(content, list):
                    updated = list(content)
                    updated.append({"type": "text", "text": agent_history})
                    target["content"] = updated
                else:
                    existing = _stringify_content(content)
                    target["content"] = f"{existing}\n\n{agent_history}" if existing else agent_history

        return prepared

    @staticmethod
    def _estimate_qwen_prompt_tokens(messages: Sequence[Dict[str, Any]]) -> int:
        flattened: List[Dict[str, Any]] = []
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(_stringify_content(item))
                normalized_content = "\n".join(part for part in parts if part)
            else:
                normalized_content = _stringify_content(content)
            flattened.append({"role": msg.get("role", "user"), "content": normalized_content})
        return LLMClient._estimate_prompt_tokens(flattened)

    def _run_qwen_generation(
        self,
        prepared_msgs: Sequence[Dict[str, Any]],
        generation_options: Dict[str, Any],
    ) -> Tuple[str, Dict[str, int]]:
        inputs = self.processor.apply_chat_template(
            prepared_msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model_engine.device)

        with self._torch.inference_mode():
            generated_ids = self.model_engine.generate(**inputs, **generation_options)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answer = output_text[0] if output_text else ""

        prompt_tokens = 0
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            prompt_tokens = int(attention_mask.sum().item())
        else:
            for in_ids in inputs.input_ids:
                prompt_tokens += int(len(in_ids))

        completion_tokens = 0
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids):
            completion_tokens += max(int(len(out_ids) - len(in_ids)), 0)

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return answer, usage

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        token_limiter: Optional[TokenLimiter] = None,
        **call_kwargs: Any,
    ) -> Tuple[str, Dict[str, int]]:
        prepared_msgs = self._prepare_qwen_messages(messages, system=system)
        estimated_prompt_tokens = self._estimate_qwen_prompt_tokens(prepared_msgs)
        if token_limiter:
            token_limiter.ensure_available()
            token_limiter.preview(
                estimated_prompt_tokens,
                {
                    "total_tokens": token_limiter.consumed + estimated_prompt_tokens,
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                },
            )

        generation_options = dict(self.kwargs)
        generation_options.update(self.generation_kwargs)
        generation_options.update(call_kwargs)
        generation_options = {k: v for k, v in generation_options.items() if v is not None}
        generation_options.setdefault("max_new_tokens", self.max_new_tokens)

        if token_limiter:
            token_limiter.ensure_available()

        async def _execute() -> Tuple[str, Dict[str, int]]:
            loop = asyncio.get_running_loop()
            content, usage = await loop.run_in_executor(
                None,
                self._run_qwen_generation,
                prepared_msgs,
                generation_options,
            )
            self.last_response = {
                "choices": [{"message": {"content": content}}],
                "usage": usage,
            }
            if token_limiter:
                token_limiter.record(usage)
            self._log_usage(usage)
            return content, usage

        if self._semaphore is None:
            return await _execute()
        async with self._semaphore:
            return await _execute()




async def main() -> None:
    orchestrator = seimei(
        agent_config=[{"name": "answer"}],
        llm_client_class=QwenVL_LLMClient,
        llm_config={
            "model": "/workspace/qwen8b-vl/",
            "processor_path": "/workspace/qwen8b-vl/",
            "device_map": "auto",
            "torch_dtype": "auto",
            "max_new_tokens": 128,
            # "attn_implementation": "flash_attention_2",
        },
        max_steps=1,
    )

    result = await orchestrator(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
    )
    print(result["output"])


if __name__ == "__main__":
    asyncio.run(main())
