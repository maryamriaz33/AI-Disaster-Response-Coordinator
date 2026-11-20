"""
RESQ-AI :: FastAPI Backend
-----------------------------
Run locally (on a machine WITH internet, to install deps):

    pip install fastapi uvicorn python-multipart pillow numpy opencv-python-headless
    uvicorn api.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger docs.

Endpoints implemented (matching the project plan, Section 30):
    POST /analyze-image        -> flood coverage from an uploaded image
    POST /analyze-report       -> bilingual citizen report classification
    POST /priority-score       -> compute explainable zone priority
    GET  /zones                -> generate a demo zone grid
    POST /recommended-route    -> best route avoiding blocked roads

This file imports directly from src/ so the "AI methodology, datasets,
evaluation, architecture and decisions" all live in your own code, not
inside a black-box call — this matters when judges ask what your team
actually built.
"""

from __future__ import annotations
import sys
import os
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np

from src.flood.detector import predict_flood_mask
from src.nlp.classifier import classify_report
from src.risk.priority_engine import compute_priority, estimate_people_at_risk
from src.routing.zones_and_routes import divide_into_zones, recommend_route

app = FastAPI(
    title="RESQ-AI API",
    description="AI decision-support & prioritization system for disaster response "
                "coordination in Pakistan. Provides estimates and recommendations to "
                "authorized responders — it does not autonomously dispatch resources.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "name": "RESQ-AI",
        "status": "ok",
        "note": "AI decision-support system for disaster response. "
                "All outputs are estimates/recommendations for human responders.",
    }


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    result = predict_flood_mask(image)
    # mask is a big array — don't ship raw pixels over JSON, just the summary
    return {
        "flood_coverage_pct": result["flood_coverage_pct"],
        "confidence": result["confidence"],
        "method": result["method"],
    }


class ReportIn(BaseModel):
    text: str
    latitude: float | None = None
    longitude: float | None = None


@app.post("/analyze-report")
def analyze_report(payload: ReportIn):
    result = classify_report(payload.text)
    return {**result, "latitude": payload.latitude, "longitude": payload.longitude}


class PriorityIn(BaseModel):
    people_risk: float = 0
    flood_severity: float = 0
    infrastructure_damage: float = 0
    road_accessibility_risk: float = 0
    weather_risk: float = 0


@app.post("/priority-score")
def priority_score(payload: PriorityIn):
    return compute_priority(payload.model_dump())


class PopulationIn(BaseModel):
    population_density_per_km2: float
    flooded_area_km2: float


@app.post("/estimate-people-at-risk")
def people_at_risk(payload: PopulationIn):
    return estimate_people_at_risk(
        payload.population_density_per_km2, payload.flooded_area_km2
    )


@app.get("/zones")
def zones(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
          rows: int = 3, cols: int = 3):
    return divide_into_zones(lat_min, lat_max, lon_min, lon_max, rows, cols)


class RouteIn(BaseModel):
    candidate_routes: list[dict]
    blocked_road_ids: list[str]


@app.post("/recommended-route")
def route(payload: RouteIn):
    return recommend_route(payload.candidate_routes, set(payload.blocked_road_ids))
