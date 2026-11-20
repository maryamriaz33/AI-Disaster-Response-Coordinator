"""
RESQ-AI :: Zone Division & Route Recommendation
--------------------------------------------------
- divide_into_zones(): splits an affected bounding box into a grid of zones
- recommend_route(): picks the best of several candidate routes given
  blockage risk, avoiding roads flagged by flood overlap or citizen reports

NOTE ON ROUTING HONESTY (per plan Section 27):
We never claim "guaranteed safe route." Output is always phrased as
"AI-recommended accessible route based on available information."
"""

from __future__ import annotations
import itertools


def divide_into_zones(lat_min, lat_max, lon_min, lon_max, rows=3, cols=3):
    """Split a bounding box into a rows x cols grid of zones with ids A1, A2, ..."""
    zones = []
    lat_step = (lat_max - lat_min) / rows
    lon_step = (lon_max - lon_min) / cols
    row_labels = list("ABCDEFGHIJ"[:rows])

    for r, row_label in zip(range(rows), row_labels):
        for c in range(cols):
            zone_id = f"{row_label}{c + 1}"
            z_lat_min = lat_min + r * lat_step
            z_lat_max = lat_min + (r + 1) * lat_step
            z_lon_min = lon_min + c * lon_step
            z_lon_max = lon_min + (c + 1) * lon_step
            zones.append({
                "zone_id": zone_id,
                "lat": round((z_lat_min + z_lat_max) / 2, 6),
                "lon": round((z_lon_min + z_lon_max) / 2, 6),
                "bounds": {
                    "lat_min": round(z_lat_min, 6), "lat_max": round(z_lat_max, 6),
                    "lon_min": round(z_lon_min, 6), "lon_max": round(z_lon_max, 6),
                },
            })
    return zones


def recommend_route(candidate_routes: list[dict], blocked_road_ids: set[str]) -> dict:
    """
    Args:
        candidate_routes: [
          {"route_id": "Route A", "roads": ["Road1","Road2"],
           "distance_km": 4.2, "est_minutes": 11}, ...
        ]
        blocked_road_ids: set of road ids considered blocked/high-risk

    Returns best viable route, or the least-bad one with a clear warning
    if every candidate touches a blocked road.
    """
    scored = []
    for route in candidate_routes:
        blocked_hits = [r for r in route["roads"] if r in blocked_road_ids]
        scored.append({
            **route,
            "blocked_segments": blocked_hits,
            "is_clear": len(blocked_hits) == 0,
        })

    clear_routes = [r for r in scored if r["is_clear"]]
    pool = clear_routes if clear_routes else scored
    best = min(pool, key=lambda r: r["distance_km"])

    warning = None
    if not clear_routes:
        warning = (
            "No fully clear route found. This is the least-affected candidate — "
            "confirm with ground teams before dispatch."
        )

    return {
        "recommended_route_id": best["route_id"],
        "distance_km": best["distance_km"],
        "est_minutes": best["est_minutes"],
        "blocked_segments": best["blocked_segments"],
        "warning": warning,
        "disclaimer": "AI-recommended accessible route based on available geographic "
                       "and disaster information. Field conditions can change.",
        "alternatives": [r["route_id"] for r in scored if r["route_id"] != best["route_id"]],
    }


if __name__ == "__main__":
    zones = divide_into_zones(24.85, 24.95, 67.00, 67.10, rows=3, cols=3)
    print(f"Generated {len(zones)} zones, e.g. {zones[0]}")

    routes = [
        {"route_id": "Route A", "roads": ["Road1", "Road2"], "distance_km": 3.1, "est_minutes": 9},
        {"route_id": "Route B", "roads": ["Road3"], "distance_km": 4.8, "est_minutes": 14},
        {"route_id": "Route C", "roads": ["Road2", "Road5"], "distance_km": 5.5, "est_minutes": 16},
    ]
    print(recommend_route(routes, blocked_road_ids={"Road2", "Road5"}))
