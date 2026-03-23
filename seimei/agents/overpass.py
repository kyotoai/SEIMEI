from __future__ import annotations

import asyncio
import json
import re
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple

import requests

from seimei.agent import Agent, register
from seimei.prompts.default import OVERPASS_SYSTEM_PROMPT

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_DEFAULT_RADIUS_M = 800
_DEFAULT_FLOOR_HEIGHT_M = 3.0
_MAX_SAMPLE_BUILDINGS = 5
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_LAT_RE = re.compile(r"lat(?:itude)?\s*[:=]?\s*(-?\d{1,2}\.\d+)")
_LON_RE = re.compile(r"lon(?:gitude)?\s*[:=]?\s*(-?\d{1,3}\.\d+)")
_PAIR_RE = re.compile(r"(-?\d{1,2}\.\d+)[,\s]+(-?\d{1,3}\.\d+)")
_RADIUS_RE = re.compile(r"(?:radius|within)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(km|m)?", re.IGNORECASE)


@register
class overpass(Agent):
    """Query the Overpass API for building information around geographic points."""

    description = "Fetch OpenStreetMap building data around coordinates using Overpass and summarize heights."

    async def inference(
        self,
        messages: List[Dict[str, Any]],
        shared_ctx: Dict[str, Any],
        **_: Any,
    ) -> Dict[str, Any]:
        user_request = _latest_user_query(messages)
        if not user_request:
            return {"content": "No user request detected for the Overpass agent."}

        fallback_lat, fallback_lon = _extract_coordinates(user_request)
        fallback_radius = _extract_radius(user_request) or _DEFAULT_RADIUS_M
        llm = shared_ctx.get("llm")

        query_meta: Dict[str, Any] = {}
        llm_note: Optional[str] = None
        if llm:
            llm_meta, llm_note = await _llm_generate_query(llm, user_request, fallback_lat, fallback_lon, fallback_radius)
            query_meta.update(llm_meta)
        else:
            llm_note = "LLM unavailable; relying on literal coordinates."

        latitude = _coalesce_float(query_meta.get("latitude"), fallback_lat)
        longitude = _coalesce_float(query_meta.get("longitude"), fallback_lon)
        radius_m = _coalesce_number(query_meta.get("radius_m"), fallback_radius) or _DEFAULT_RADIUS_M
        overpass_query = query_meta.get("query")

        if not overpass_query:
            if latitude is None or longitude is None:
                return {
                    "content": (
                        "Unable to form an Overpass query because latitude/longitude were not provided. "
                        "Please supply coordinates (lat, lon)."
                    ),
                    "log": {"note": llm_note},
                }
            overpass_query = _build_default_query(latitude, longitude, radius_m, query_meta.get("filters") or [])
            query_meta.setdefault("reason", "Generated default building query.")

        try:
            response_data = await _execute_overpass_query(overpass_query)
        except requests.RequestException as exc:
            return {
                "content": f"Overpass request failed: {exc}",
                "log": {"query": overpass_query, "note": llm_note, "parameters": query_meta},
            }

        buildings = _extract_building_rows(response_data)
        #print("buildings: ", buildings)
        stats = _summarize_building_stats(buildings, radius_m)
        #print("stats: ", stats)
        summary_text = _format_summary(
            latitude=latitude,
            longitude=longitude,
            radius_m=radius_m,
            stats=stats,
            llm_reason=query_meta.get("reason"),
            extra_note=llm_note,
        )
        #print("summary_text: ", summary_text)
        log_payload: Dict[str, Any] = {
            "query": overpass_query,
            "parameters": query_meta,
            "stats": stats,
            "record_count": len(buildings),
        }
        if buildings:
            log_payload["sample_buildings"] = buildings[: _MAX_SAMPLE_BUILDINGS]

        print()
        print()
        print({
            "content": summary_text,
            "data": {"buildings": buildings, "stats": stats},
            "log": log_payload,
        })

        return {
            "content": log_payload, #summary_text,
            "data": {"buildings": buildings, "stats": stats},
            "log": log_payload,
        }


def _latest_user_query(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


async def _llm_generate_query(
    llm: Any,
    user_request: str,
    fallback_lat: Optional[float],
    fallback_lon: Optional[float],
    fallback_radius: int,
) -> Tuple[Dict[str, Any], Optional[str]]:
    hints: List[str] = [f"User request:\n{user_request.strip()}"]
    if fallback_lat is not None and fallback_lon is not None:
        hints.append(f"Coordinate hint: lat={fallback_lat}, lon={fallback_lon}")
    if fallback_radius:
        hints.append(f"Approximate radius hint: {fallback_radius} meters")
    content = "\n".join(hints)
    try:
        llm_output, _ = await llm.chat(
            messages=[{"role": "user", "content": content}],
            system=OVERPASS_SYSTEM_PROMPT,
        )
    except Exception as exc:
        return {}, f"LLM query generation failed: {exc}"

    parsed = _parse_llm_json(llm_output or "")
    if fallback_lat is not None and parsed.get("latitude") is None:
        parsed["latitude"] = fallback_lat
    if fallback_lon is not None and parsed.get("longitude") is None:
        parsed["longitude"] = fallback_lon
    if parsed.get("radius_m") is None:
        parsed["radius_m"] = fallback_radius
    return parsed, None


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    candidate = raw.strip()
    match = _JSON_BLOCK_RE.search(candidate)
    if match:
        candidate = match.group(1).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return {}

    filters: List[str] = []
    raw_filters = payload.get("filters")
    if isinstance(raw_filters, list):
        filters = [str(item).strip() for item in raw_filters if str(item).strip()]
    elif isinstance(raw_filters, str):
        value = raw_filters.strip()
        if value:
            filters = [value]

    return {
        "query": str(payload.get("query") or payload.get("overpass") or "").strip(),
        "latitude": _coalesce_float(payload.get("latitude"), payload.get("lat")),
        "longitude": _coalesce_float(payload.get("longitude"), payload.get("lon")),
        "radius_m": _coalesce_number(payload.get("radius_m"), payload.get("radius"), payload.get("radiusMeters")),
        "filters": filters,
        "reason": str(payload.get("reason") or payload.get("explanation") or "").strip(),
    }


def _build_default_query(lat: float, lon: float, radius_m: int, filters: List[str]) -> str:
    extra_clause = "".join(_format_filter_clause(flt) for flt in filters if flt)
    filter_block = f'["building"]{extra_clause}'
    return (
        "[out:json][timeout:60];\n"
        "(\n"
        f"  way{filter_block}(around:{int(radius_m)},{lat},{lon});\n"
        f"  relation{filter_block}(around:{int(radius_m)},{lat},{lon});\n"
        ");\n"
        "out tags center;"
    )


def _format_filter_clause(flt: str) -> str:
    clean = flt.strip().strip('"')
    if not clean:
        return ""
    if "=" not in clean:
        return f'["{clean}"]'
    key, value = clean.split("=", 1)
    return f'["{key.strip()}"="{value.strip()}"]'


async def _execute_overpass_query(query: str) -> Dict[str, Any]:
    def _run() -> Dict[str, Any]:
        resp = requests.post(
            OVERPASS_URL,
            data=query.encode("utf-8"),
            headers={"User-Agent": "seimei-overpass-agent/1.0", "Content-Type": "application/x-www-form-urlencoded"},
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


def _extract_building_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for element in data.get("elements", []):
        tags = element.get("tags") or {}
        center = element.get("center") or {}
        lat = _coalesce_float(center.get("lat"), element.get("lat"))
        lon = _coalesce_float(center.get("lon"), element.get("lon"))
        height_m = _estimate_height(tags)
        rows.append(
            {
                "osm_type": element.get("type"),
                "osm_id": element.get("id"),
                "lat": lat,
                "lon": lon,
                "height_m": height_m,
                "has_height_tag": tags.get("height") is not None,
                "building_levels": tags.get("building:levels"),
                "building": tags.get("building"),
                "name": tags.get("name"),
            }
        )
    return rows


def _estimate_height(tags: Dict[str, Any]) -> Optional[float]:
    height = _parse_height_m(tags.get("height"))
    if height is not None:
        return height
    levels = tags.get("building:levels")
    if levels is None:
        return None
    try:
        levels_val = float(levels)
    except (TypeError, ValueError):
        return None
    return levels_val * _DEFAULT_FLOOR_HEIGHT_M


def _parse_height_m(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    num = float(match.group(0))
    if "ft" in text or "feet" in text:
        return num * 0.3048
    return num


def _summarize_building_stats(rows: List[Dict[str, Any]], radius_m: int) -> Dict[str, Any]:
    heights = [row["height_m"] for row in rows if row.get("height_m") is not None]
    stats: Dict[str, Any] = {
        "radius_m": radius_m,
        "total_buildings": len(rows),
        "height_available": len(heights),
        "coverage_ratio": (len(heights) / len(rows)) if rows else 0.0,
    }
    if heights:
        stats["mean_m"] = mean(heights)
        stats["median_m"] = median(heights)
        stats["min_m"] = min(heights)
        stats["max_m"] = max(heights)
        if len(heights) > 1:
            stats["std_m"] = stdev(heights)
    samples: List[Dict[str, Any]] = []
    for row in rows[: _MAX_SAMPLE_BUILDINGS]:
        samples.append(
            {
                "osm_type": row.get("osm_type"),
                "osm_id": row.get("osm_id"),
                "name": row.get("name"),
                "height_m": row.get("height_m"),
                "building": row.get("building"),
                "lat": row.get("lat"),
                "lon": row.get("lon"),
            }
        )
    stats["samples"] = samples
    return stats


def _format_summary(
    *,
    latitude: Optional[float],
    longitude: Optional[float],
    radius_m: int,
    stats: Dict[str, Any],
    llm_reason: Optional[str],
    extra_note: Optional[str],
) -> str:
    coord_text = "unknown coordinates"
    if latitude is not None and longitude is not None:
        coord_text = f"{latitude:.5f}, {longitude:.5f}"
    lines = [
        f"Overpass building scan around {coord_text} (radius ≈ {radius_m} m)",
        f"Total buildings: {stats.get('total_buildings', 0)} | Height coverage: {stats.get('height_available', 0)} ({stats.get('coverage_ratio', 0.0):.1%})",
    ]
    if stats.get("height_available"):
        lines.append(
            "Height stats (m): "
            f"mean {stats.get('mean_m', 0):.1f}, "
            f"median {stats.get('median_m', 0):.1f}, "
            f"min {stats.get('min_m', 0):.1f}, "
            f"max {stats.get('max_m', 0):.1f}"
        )
    samples = stats.get("samples") or []
    if samples:
        lines.append("Sample buildings:")
        for sample in samples:
            name = sample.get("name") or sample.get("building") or sample.get("osm_type")
            height = sample.get("height_m")
            height_str = f"{height:.1f} m" if isinstance(height, (int, float)) else "unknown height"
            lines.append(f"- {name} ({sample.get('osm_type')} #{sample.get('osm_id')}): {height_str}")
    if llm_reason:
        lines.append(f"LLM routing hint: {llm_reason}")
    if extra_note:
        lines.append(f"Note: {extra_note}")
    if not samples:
        lines.append("No building samples returned; consider increasing the radius or providing more precise coordinates.")
    return "\n".join(lines)


def _extract_coordinates(text: str) -> Tuple[Optional[float], Optional[float]]:
    lat = _coalesce_float(_first_match_float(_LAT_RE, text))
    lon = _coalesce_float(_first_match_float(_LON_RE, text))
    if lat is not None and lon is not None:
        return lat, lon
    for match in _PAIR_RE.finditer(text):
        cand_lat = float(match.group(1))
        cand_lon = float(match.group(2))
        if abs(cand_lat) <= 90 and abs(cand_lon) <= 180:
            return cand_lat, cand_lon
    return lat, lon


def _extract_radius(text: str) -> Optional[int]:
    match = _RADIUS_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    unit = (match.group(2) or "m").lower()
    if unit == "km":
        value *= 1000.0
    return int(value)


def _first_match_float(pattern: re.Pattern[str], text: str) -> Optional[float]:
    match = pattern.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _coalesce_float(*values: Any) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _coalesce_number(*values: Any) -> Optional[int]:
    for value in values:
        if value is None:
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return None
