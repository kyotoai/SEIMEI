from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests

from .utils import (
    format_query_for_rmsearch,
    format_key_for_rmsearch,
    _coerce_tags,
    _coerce_messages,
    _prepare_query_input,
    _normalize_purpose,
)

logger = logging.getLogger("seimei")


class RM:
    """Reward-model search client for SEIMEI.

    Wraps the rmsearch HTTP endpoint and handles query/key formatting,
    response parsing, and token-budget enforcement.

    Args:
        rm_config: Configuration dict. Recognised keys:
            - ``base_url`` / ``url``: RMSearch endpoint URL.
            - ``k``: Default top-k to return.
            - ``timeout``: Request timeout in seconds.
            - ``purpose``: Default purpose string.
    """

    def __init__(self, rm_config: Dict[str, Any]) -> None:
        self.rm_config = rm_config
        self._rm_warned_missing_url = False

    # ------------------------------------------------------------------
    # Public / semi-public interface
    # ------------------------------------------------------------------

    def _should_use_rmsearch(self, cfg: Optional[Dict[str, Any]] = None) -> bool:
        config = cfg or self.rm_config
        url = str(config.get("base_url") or config.get("url") or "").strip()
        return bool(url)

    def _rmsearch(
        self,
        *,
        query: Union[str, Sequence[Dict[str, Any]]],
        keys: Sequence[Dict[str, Any]],
        k_key: Optional[int] = None,
        k: Optional[int] = None,
        purpose: Optional[str] = None,
        timeout: Optional[float] = None,
        **overrides: Any,
    ) -> List[Dict[str, Any]]:
        config = dict(self.rm_config)
        config.update(overrides)
        effective_purpose = _normalize_purpose(purpose or config.get("purpose"))

        if not self._should_use_rmsearch(config):
            return []

        url = str(config.get("base_url") or config.get("url") or "").strip()
        if not url:
            if not self._rm_warned_missing_url:
                logger.warning("[seimei] rmsearch skipped: rm_config['base_url'] not set.")
                self._rm_warned_missing_url = True
            return []

        limit_raw = k_key if k_key is not None else k if k is not None else config.get("k")
        try:
            limit = max(int(limit_raw), 1)
        except (TypeError, ValueError):
            limit = 1

        final_timeout = timeout if timeout is not None else config.get("timeout")

        #print()
        #print("--- rmsearch 1 ---")
        #print("query: ", query)

        try:
            return self._rmsearch_http(
                url=url,
                query=query,
                keys=keys,
                limit=limit,
                purpose=effective_purpose,
                timeout=final_timeout,
            )
        except Exception as exc:
            logger.error("[seimei] rmsearch request failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rmsearch_http(
        self,
        *,
        url: str,
        query: Union[str, Sequence[Dict[str, Any]]],
        keys: Sequence[Dict[str, Any]],
        limit: int,
        purpose: str,
        timeout: Optional[float],
    ) -> List[Dict[str, Any]]:
        serialized_query = self._format_rmsearch_query(query)
        key_payload, index_map, text_map = self._format_rmsearch_keys(keys)
        if not key_payload:
            return []

        #print()
        #print("--- rmsearch 2 ---")
        #print("serialized_query: ", serialized_query)

        payload: Dict[str, Any] = {
            "queries": [serialized_query],
            "keys": key_payload,
            "k": limit,
        }
        if purpose:
            payload["purpose"] = purpose

        api_key = os.getenv("KYOTOAI_API_KEY")
        if not api_key:
            logger.warning("[seimei] KYOTOAI_API_KEY variable is not set")
            #raise RuntimeError("KYOTOAI_API_KEY environment variable is not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(url, json=payload, headers=headers, timeout=timeout or 10)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON from RMSearch: {exc}") from exc

        return self._parse_rmsearch_response(
            data=data,
            limit=limit,
            index_map=index_map,
            text_map=text_map,
        )

    @staticmethod
    def _format_rmsearch_query(query: Union[str, Sequence[Dict[str, Any]]]) -> str:
        if isinstance(query, str):
            stripped = query.strip()
            return format_query_for_rmsearch(stripped or query)
        if isinstance(query, Sequence):
            messages = _coerce_messages(query)
            _, conversation_text, focus_text = _prepare_query_input(messages)
            body = focus_text or conversation_text or ""
            return format_query_for_rmsearch(body)
        return format_query_for_rmsearch(str(query))

    @staticmethod
    def _format_rmsearch_keys(
        keys: Sequence[Dict[str, Any]]
    ) -> Tuple[List[str], Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        payload: List[str] = []
        index_map: Dict[int, Dict[str, Any]] = {}
        text_map: Dict[str, Dict[str, Any]] = {}
        for item in keys:
            if not isinstance(item, dict):
                continue
            key_text = str(item.get("key") or "").strip()
            if not key_text:
                continue
            tags = _coerce_tags(item.get("tags"))
            formatted = format_key_for_rmsearch(key_text, tags=tags)
            index_map[len(payload)] = item
            payload.append(formatted)
            text_map.setdefault(formatted, item)
        return payload, index_map, text_map

    @staticmethod
    def _parse_rmsearch_response(
        *,
        data: Any,
        limit: int,
        index_map: Dict[int, Dict[str, Any]],
        text_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        entries: Sequence[Any]
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = data.get("output")
            if isinstance(entries, dict):
                entries = [entries]
            if not isinstance(entries, list):
                entries = []
        else:
            return []

        for entry in entries:
            keys = []
            if isinstance(entry, dict):
                keys = entry.get("keys") or []
            if not isinstance(keys, Sequence):
                continue
            for item in keys:
                if not isinstance(item, dict):
                    continue
                key_idx = item.get("key_id")
                key_text = item.get("key")
                payload = None
                if isinstance(key_idx, int) and key_idx in index_map:
                    payload = index_map[key_idx]
                elif isinstance(key_text, str) and key_text in text_map:
                    payload = text_map[key_text]
                if not payload:
                    continue
                result = {
                    "key": payload.get("key"),
                    "payload": payload,
                    "score": item.get("relevance"),
                    "source": "rmsearch",
                }
                if "reason" in item:
                    result["reason"] = item["reason"]
                records.append(result)
                if len(records) >= limit:
                    return records
        return records[:limit]


__all__ = ["RM"]
