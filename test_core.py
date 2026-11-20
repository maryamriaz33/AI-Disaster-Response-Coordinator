"""
Run with:  python3 -m pytest tests/ -v
(or just:  python3 tests/test_core.py   -- runs without pytest too)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.flood.detector import predict_flood_mask
from src.nlp.classifier import classify_report
from src.risk.priority_engine import compute_priority, estimate_people_at_risk
from src.routing.zones_and_routes import divide_into_zones, recommend_route


def test_flood_detector_finds_water_patch():
    rng = np.random.default_rng(1)
    img = rng.integers(140, 220, size=(200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150] = [60, 80, 100]  # muddy water patch, 25% of image
    result = predict_flood_mask(img)
    assert 15 <= result["flood_coverage_pct"] <= 40, result
    assert 0 <= result["confidence"] <= 1


def test_classifier_english_rescue():
    r = classify_report("People are trapped near the bridge and need rescue")
    assert r["category"] == "PEOPLE_NEED_RESCUE"
    assert r["severity"] == "CRITICAL"


def test_classifier_urdu_rescue():
    r = classify_report("لوگ پل کے قریب پھنسے ہوئے ہیں")
    assert r["language"] == "ur"
    assert r["category"] == "PEOPLE_NEED_RESCUE"


def test_classifier_roman_urdu_flood():
    r = classify_report("hamaray ilaqay mein pani bohat zyada hai")
    assert r["language"] == "roman_ur"
    assert r["category"] == "FLOOD"


def test_priority_engine_matches_formula():
    factors = {
        "people_risk": 91, "flood_severity": 87, "infrastructure_damage": 79,
        "road_accessibility_risk": 95, "weather_risk": 83,
    }
    expected = 0.30*91 + 0.25*87 + 0.20*79 + 0.15*95 + 0.10*83
    result = compute_priority(factors)
    assert abs(result["score"] - round(expected, 1)) < 0.2
    assert result["band"] == "CRITICAL"


def test_people_at_risk_estimate():
    r = estimate_people_at_risk(3500, 0.4)
    assert r["estimated_people_at_risk"] == 1400
    assert r["is_estimate"] is True


def test_zones_grid_generation():
    zones = divide_into_zones(24.85, 24.95, 67.00, 67.10, rows=3, cols=3)
    assert len(zones) == 9
    assert zones[0]["zone_id"] == "A1"


def test_route_avoids_blocked_roads():
    routes = [
        {"route_id": "Route A", "roads": ["Road1"], "distance_km": 3.0, "est_minutes": 8},
        {"route_id": "Route B", "roads": ["Road2"], "distance_km": 5.0, "est_minutes": 15},
    ]
    result = recommend_route(routes, blocked_road_ids={"Road1"})
    assert result["recommended_route_id"] == "Route B"


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
