"""
main.py — FastAPI application entry point
==========================================
Run with:
    uvicorn api.main:app --reload

The --reload flag restarts the server automatically whenever
you save a file — useful during development.

Once running, visit:
    http://localhost:8000        → serves the frontend HTML
    http://localhost:8000/docs   → auto-generated API docs (Swagger UI)
    http://localhost:8000/redoc  → alternative API docs (ReDoc)
"""

import sys
from pathlib import Path

# ── Ensure the project root is on the Python path ─────────────────────────────
# This lets the routers import your existing scripts directly,
# e.g. from tire_degradation_analyzer import load_season
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routers import tire, predictor, conversion
from api.schemas import HealthResponse, CalendarResponse

import fastf1

# ── FastF1 cache ───────────────────────────────────────────────────────────────
CACHE_DIR = ROOT / "f1_cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="F1 Analytics API",
    description="Race prediction, tire degradation, and pace conversion scores powered by FastF1.",
    version="1.0.0",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Allows the frontend (running as a plain HTML file or on a different port)
# to make fetch() calls to this server without browser security errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten this in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(tire.router,       prefix="/tire",       tags=["Tire Degradation"])
app.include_router(predictor.router,  prefix="/predict",    tags=["Race Predictor"])
app.include_router(conversion.router, prefix="/conversion", tags=["Pace Conversion"])

# ── Static files (serves the frontend HTML app) ────────────────────────────────
# Place your f1_app.html in the project root and it will be served at /
STATIC_DIR = ROOT
if (STATIC_DIR / "f1_app.html").exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        return FileResponse(str(STATIC_DIR / "f1_app.html"))

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health_check():
    """Quick check that the server is up and FastF1 is importable."""
    return HealthResponse(status="ok", message="F1 Analytics API is running.")

# ── Calendar helper ────────────────────────────────────────────────────────────
@app.get("/calendar/{year}", response_model=CalendarResponse, tags=["Status"])
def get_calendar(year: int):
    """
    Returns the list of race names for a given season.
    The frontend uses this to populate the race dropdown dynamically.
    """
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        races = (
            schedule[schedule["EventFormat"] != "testing"]["EventName"]
            .tolist()
        )
        return CalendarResponse(year=year, races=races)
    except Exception as e:
        return CalendarResponse(year=year, races=[])