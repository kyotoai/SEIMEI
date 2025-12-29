import re
import math
import requests
import pandas as pd

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def _parse_height_m(val: str):
    """
    Parse OSM height strings like:
      '12', '12m', '12 m', '39.5', '120 ft'
    Returns meters (float) or None.
    """
    if not val:
        return None
    s = val.strip().lower()

    # grab the first number
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        return None
    num = float(m.group(0))

    # unit handling (very lightweight)
    if "ft" in s or "feet" in s:
        return num * 0.3048
    # default assume meters
    return num

def fetch_building_heights(lat: float, lon: float, radius_m: int = 500, level_height_m: float = 3.0):
    # Query buildings (ways + relations) around a point
    # We ask for tags because height/levels live in tags.
    query = f"""
    [out:json][timeout:60];
    (
      way["building"](around:{radius_m},{lat},{lon});
      relation["building"](around:{radius_m},{lat},{lon});
    );
    out tags center;
    """

    r = requests.post(
        OVERPASS_URL,
        data=query.encode("utf-8"),
        headers={
            "User-Agent": "building-height-stats/1.0 (your_email_or_site_here)"
        },
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()

    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}

        # OSM total height is usually in tags["height"] (meters typically). :contentReference[oaicite:2]{index=2}
        h = _parse_height_m(tags.get("height"))

        # If missing, estimate from building:levels (floors). :contentReference[oaicite:3]{index=3}
        if h is None:
            levels = tags.get("building:levels")
            if levels is not None:
                try:
                    h = float(levels) * level_height_m
                except ValueError:
                    h = None

        # center can be "center" (for ways/relations) or "lat/lon" (for nodes; rare here)
        center = el.get("center") or {}
        rows.append({
            "osm_type": el.get("type"),
            "osm_id": el.get("id"),
            "lat": center.get("lat"),
            "lon": center.get("lon"),
            "height_m": h,
            "has_height_tag": tags.get("height") is not None,
            "building_levels": tags.get("building:levels"),
            "building": tags.get("building"),
        })

    df = pd.DataFrame(rows)

    # Stats on known/estimated heights
    if "height_m" in df:
        h = df["height_m"].dropna()
        stats = {
            "radius_m": radius_m,
            "total_buildings": int(len(df)),
            "height_available": int(h.shape[0]),
            "coverage_ratio": float(h.shape[0] / len(df)) if len(df) else 0.0,
            "mean_m": float(h.mean()) if len(h) else None,
            "median_m": float(h.median()) if len(h) else None,
            "std_m": float(h.std(ddof=1)) if len(h) > 1 else None,
            "min_m": float(h.min()) if len(h) else None,
            "max_m": float(h.max()) if len(h) else None,
        }
        return df, stats
    else:
        return df, None

if __name__ == "__main__":
    # Example: Kyoto Station area (edit)
    df, stats = fetch_building_heights(lat=33.80232, lon=135.141182, radius_m=5000)
    print(stats)
    print(df.head())

