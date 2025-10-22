from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

_AGENT_REGISTRY: Dict[str, Type['Agent']] = {}

def get_agent_subclasses() -> Dict[str, Type['Agent']]:
    return dict(_AGENT_REGISTRY)

def _register_agent(subcls: Type['Agent']) -> None:
    # Use explicit `name` if provided, otherwise class name in snake-ish
    name = getattr(subcls, "name", None) or subcls.__name__
    setattr(subcls, "name", name)
    _AGENT_REGISTRY[name] = subcls

@dataclass
class Agent:
    \"\"\"Base Agent with friendly async call and step logging.

    Subclasses should override `inference` and optionally set:
      - `name`: short identifier (default = class name)
      - `description`: routing hint for rmsearch/heuristics
    \"\"\"

    name: str = field(default_factory=lambda: type("X",(object,),{}).__name__)
    description: str = ""

    async def __call__(
        self,
        messages: List[Dict[str, Any]],
        shared_ctx: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Wraps concrete inference with basic logging/error capture.
        try:
            res = await self.inference(messages=messages, shared_ctx=shared_ctx or {}, **kwargs)
            if not isinstance(res, dict):
                res = {"content": str(res)}
            # Normalize shape
            res.setdefault("content", "")
            return res
        except Exception as e:
            return {"content": f"[error] {type(e).__name__}: {e}"}

    # ------------------ to override ------------------
    async def inference(
        self,
        messages: List[Dict[str, Any]],
        shared_ctx: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError


# Class decorator to auto-register agents
def register(cls: Type[Agent]) -> Type[Agent]:
    _register_agent(cls)
    return cls
