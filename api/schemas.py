"""
schemas.py — Pydantic response models for the F1 Analytics API
==============================================================
Every endpoint returns one of these models as JSON.
The frontend fetch() calls expect exactly these shapes.

Pydantic validates the data before it leaves the server —
if a field is missing or the wrong type, FastAPI returns a
clear 422 error instead of silently sending broken JSON.
"""

from typing import Optional
from pydantic import BaseModel



# SHARED / COMMON


class HealthResponse(BaseModel):
    status: str
    message: str


class CalendarResponse(BaseModel):
    year: int
    races: list[str]



# TIRE DEGRADATION


class DegradationCurvePoint(BaseModel):
    """One data point on a degradation curve — stint lap + median delta."""
    stint_lap: int
    median_delta: float


class DriverDegradationRate(BaseModel):
    """Degradation rate for one driver at one race on one compound."""
    driver: str
    constructor: str
    deg_rate: float         # s/lap — slope of linear regression
    base_time: float        # predicted lap 1 time (intercept)
    r2: float               # goodness of fit
    p_value: float
    stint_laps: int         # number of laps used in regression
    median_lap_time: float


class DvDDataPoint(BaseModel):
    """Single lap for driver vs driver comparison."""
    stint_lap: int
    lap_time: float
    delta_from_lap1: float


class DriverVsDriverResponse(BaseModel):
    """Head-to-head comparison of two drivers on the same compound."""
    driver1: str
    driver2: str
    compound: str
    race_name: str
    year: int
    driver1_laps: list[DvDDataPoint]
    driver2_laps: list[DvDDataPoint]
    driver1_deg_rate: float
    driver2_deg_rate: float
    better_manager: str


class TireAnalysisResponse(BaseModel):
    """Full tire degradation analysis for one race and compound."""
    race_name: str
    year: int
    compound: str
    # Degradation curve (median across all drivers)
    degradation_curve: list[DegradationCurvePoint]
    # Per-driver degradation rates (for ranking table)
    driver_rates: list[DriverDegradationRate]
    # Summary metrics
    avg_deg_rate: float
    best_manager: str
    worst_manager: str
    max_stint_laps: int


class SeasonHeatmapCell(BaseModel):
    driver: str
    race_name: str
    deg_rate: Optional[float] = None


class SeasonHeatmapResponse(BaseModel):
    year: int
    compound: str
    drivers: list[str]
    races: list[str]
    cells: list[SeasonHeatmapCell]



# RACE PREDICTOR


class DriverPrediction(BaseModel):
    """Prediction for a single driver."""
    driver: str
    constructor: str
    predicted_rank: int
    win_probability: float      # 0-100
    stat_score: float
    ml_score: Optional[float] = None
    ensemble_score: float
    quali_position: Optional[int] = None
    weighted_pace_delta: float
    sessions_available: int


class ConstructorPrediction(BaseModel):
    constructor: str
    constructor_rank: int
    win_probability: float      # aggregated from drivers


class SessionPaceDelta(BaseModel):
    """Per-session pace delta for one driver."""
    driver: str
    q_delta: Optional[float] = None
    fp3_delta: Optional[float] = None
    fp2_delta: Optional[float] = None
    fp1_delta: Optional[float] = None


class PredictionResponse(BaseModel):
    """Full race prediction response."""
    race_name: str
    year: int
    model_type: str             # 'ensemble', 'statistical', 'ml'
    sessions_used: int
    # Top 3 podium
    podium: list[DriverPrediction]
    # Full field ranked
    all_drivers: list[DriverPrediction]
    # Constructor standings
    constructors: list[ConstructorPrediction]
    # Session pace deltas for chart
    session_deltas: list[SessionPaceDelta]
    # ML model available
    model_loaded: bool



# QUALI → RACE PACE CONVERSION


class RaceConversionScore(BaseModel):
    """Conversion score for one driver at one race."""
    driver: str
    constructor: str
    race_name: str
    year: int
    race_index: int
    quali_position: Optional[int] = None
    finish_position: Optional[int] = None
    position_delta: Optional[float] = None
    quali_time: Optional[float] = None
    median_race_pace: Optional[float] = None
    pace_delta: Optional[float] = None
    position_score: Optional[float] = None
    pace_score: Optional[float] = None
    conversion_score: Optional[float] = None
    status: str
    flagged: bool
    dnf_dsq: bool
    sc_affected: bool


class DriverSeasonSummary(BaseModel):
    """Season-level summary for one driver."""
    driver: str
    constructor: str
    mean_score: float
    std_score: float
    best_score: float
    worst_score: float
    races_scored: int
    flagged_races: int
    trend_slope: float
    trend_label: str            # 'Improving', 'Stable', 'Declining'
    percentile: float
    pace_score_avg: float
    position_score_avg: float
    # Per-race scores for trend chart
    race_scores: list[float]
    race_names: list[str]


class ConversionSeasonResponse(BaseModel):
    """Full season conversion score response."""
    year: int
    drivers: list[DriverSeasonSummary]
    # Raw per-race scores for heatmap
    race_scores: list[RaceConversionScore]
    # Season metrics
    field_average: float
    top_converter: str
    improving_count: int
    total_races: int