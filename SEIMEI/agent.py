from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from seimei.llm import TokenLimitExceeded

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
    """Base Agent with friendly async call and step logging.

    Subclasses should override `inference` and optionally set:
      - `name`: short identifier (default = class name)
      - `description`: routing hint for rmsearch/heuristics
    """

    name: str = field(init=False)
    description: str = field(init=False, default="")

    def __post_init__(self) -> None:
        cls = type(self)
        cls_name = getattr(cls, "name", None) or cls.__name__
        setattr(cls, "name", cls_name)
        self.name = cls_name

        cls_desc = getattr(cls, "description", "")
        setattr(cls, "description", cls_desc)
        self.description = cls_desc

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
        except TokenLimitExceeded:
            # Surface token limit errors so the orchestrator can halt the run.
            raise
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
