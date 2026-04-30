"""
routers/tire.py — Tire Degradation endpoints
=============================================
Imports directly from tire_degradation_analyzer.py in the project root.

Endpoints:
    GET /tire/analyze?year=2025&race=Bahrain+Grand+Prix&compound=MEDIUM
    GET /tire/heatmap?year=2025&compound=MEDIUM
    GET /tire/dvd?year=2025&race=Bahrain+Grand+Prix&compound=SOFT&d1=VER&d2=NOR
"""

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from tiredegradation import (
    load_race_laps,
    compute_degradation_rate,
    compute_stint_delta,
    get_calendar,
)
from api.schemas import (
    TireAnalysisResponse,
    DegradationCurvePoint,
    DriverDegradationRate,
    DriverVsDriverResponse,
    DvDDataPoint,
    SeasonHeatmapResponse,
    SeasonHeatmapCell,
)

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_float(val, decimals: int = 4) -> float:
    """Convert numpy/pandas floats to plain Python float, handle NaN."""
    try:
        f = float(val)
        return round(f, decimals) if not (f != f) else 0.0  # NaN check
    except (TypeError, ValueError):
        return 0.0


def load_and_validate(year: int, race: str):
    """Load race laps and raise HTTP 404 if no data available."""
    laps = load_race_laps(year, race)
    if laps is None or laps.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data available for {race} {year}. "
                   "Check that this race has occurred and FastF1 cache is populated."
        )
    return laps


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/analyze", response_model=TireAnalysisResponse)
def analyze_tire_degradation(
    year: int = Query(2025, description="Season year"),
    race: str = Query(..., description="Exact race name e.g. 'Bahrain Grand Prix'"),
    compound: str = Query("MEDIUM", description="Tyre compound: SOFT, MEDIUM, HARD"),
):
    """
    Returns degradation curves, per-driver rates, and summary metrics
    for a single race and compound.
    """
    laps = load_and_validate(year, race)

    # Filter to requested compound
    compound_laps = laps[laps["Compound"] == compound.upper()]
    if compound_laps.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No {compound} laps found for {race} {year}."
        )

    # Load constructor map from session results
    try:
        import fastf1
        session = fastf1.get_session(year, race, "R")
        session.load(telemetry=False, weather=False, messages=False)
        constructor_map = {
            str(row.get("Abbreviation", "")): str(row.get("TeamName", "Unknown"))
            for _, row in session.results.iterrows()
            if row.get("Abbreviation")
        }
    except Exception:
        constructor_map = {}

    # Compute degradation rates
    deg_rates = compute_degradation_rate(laps)
    compound_rates = deg_rates[
        (deg_rates["RaceName"] == race) &
        (deg_rates["Compound"] == compound.upper()) &
        (deg_rates["StintLaps"] >= 4)
    ]

    # Build degradation curve (median delta across all drivers)
    stint_laps = compute_stint_delta(compound_laps)
    median_curve = (
        stint_laps.groupby("StintLap")["LapDelta"]
        .median()
        .reset_index()
        .sort_values("StintLap")
    )

    curve = [
        DegradationCurvePoint(
            stint_lap=int(row["StintLap"]),
            median_delta=safe_float(row["LapDelta"], 3),
        )
        for _, row in median_curve.iterrows()
    ]

    # Build per-driver rates list
    driver_rates = [
        DriverDegradationRate(
            driver=str(row["Driver"]),
            constructor=constructor_map.get(str(row["Driver"]), "Unknown"),
            deg_rate=safe_float(row["DegRate"]),
            base_time=safe_float(row["BaseTime"], 3),
            r2=safe_float(row["R2"], 4),
            p_value=safe_float(row["PValue"], 4),
            stint_laps=int(row["StintLaps"]),
            median_lap_time=safe_float(row["MedianLapTime"], 3),
        )
        for _, row in compound_rates.sort_values("DegRate").iterrows()
    ]

    if not driver_rates:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough stint data to compute degradation rates for {compound} at {race} {year}."
        )

    best = min(driver_rates, key=lambda x: x.deg_rate)
    worst = max(driver_rates, key=lambda x: x.deg_rate)
    avg_deg = round(sum(d.deg_rate for d in driver_rates) / len(driver_rates), 4)
    max_stint = int(compound_laps.groupby(["Driver", "Stint"])["StintLap"].max().max())

    return TireAnalysisResponse(
        race_name=race,
        year=year,
        compound=compound.upper(),
        degradation_curve=curve,
        driver_rates=driver_rates,
        avg_deg_rate=avg_deg,
        best_manager=best.driver,
        worst_manager=worst.driver,
        max_stint_laps=max_stint,
    )


@router.get("/dvd", response_model=DriverVsDriverResponse)
def driver_vs_driver(
    year: int = Query(2025),
    race: str = Query(..., description="Exact race name"),
    compound: str = Query("MEDIUM"),
    d1: str = Query(..., description="Driver 1 abbreviation e.g. VER"),
    d2: str = Query(..., description="Driver 2 abbreviation e.g. NOR"),
):
    """
    Head-to-head comparison of two drivers on the same compound.
    Returns lap-by-lap times and deltas for both drivers.
    """
    laps = load_and_validate(year, race)

    data = laps[
        (laps["Compound"] == compound.upper()) &
        (laps["Driver"].isin([d1.upper(), d2.upper()]))
    ].copy()

    if data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No {compound} laps found for {d1} or {d2} at {race} {year}."
        )

    data = compute_stint_delta(data)

    def build_laps(driver_code: str) -> list[DvDDataPoint]:
        sub = data[data["Driver"] == driver_code].sort_values("StintLap")
        return [
            DvDDataPoint(
                stint_lap=int(row["StintLap"]),
                lap_time=safe_float(row["LapTimeSeconds"], 3),
                delta_from_lap1=safe_float(row["LapDelta"], 3),
            )
            for _, row in sub.iterrows()
        ]

    # Degradation rates for both drivers
    deg_rates = compute_degradation_rate(laps)

    def get_deg_rate(driver_code: str) -> float:
        sub = deg_rates[
            (deg_rates["Driver"] == driver_code) &
            (deg_rates["RaceName"] == race) &
            (deg_rates["Compound"] == compound.upper())
        ]
        return safe_float(sub["DegRate"].iloc[0]) if not sub.empty else 0.0

    d1_rate = get_deg_rate(d1.upper())
    d2_rate = get_deg_rate(d2.upper())
    better = d1.upper() if d1_rate <= d2_rate else d2.upper()

    return DriverVsDriverResponse(
        driver1=d1.upper(),
        driver2=d2.upper(),
        compound=compound.upper(),
        race_name=race,
        year=year,
        driver1_laps=build_laps(d1.upper()),
        driver2_laps=build_laps(d2.upper()),
        driver1_deg_rate=d1_rate,
        driver2_deg_rate=d2_rate,
        better_manager=better,
    )


@router.get("/heatmap", response_model=SeasonHeatmapResponse)
def season_heatmap(
    year: int = Query(2025),
    compound: str = Query("MEDIUM"),
    max_races: int = Query(12, description="Limit to first N races for performance"),
):
    """
    Returns degradation rate data for all drivers across the season.
    Used to render the season heatmap in the frontend.
    """
    races = get_calendar(year)
    if not races:
        raise HTTPException(status_code=404, detail=f"No calendar found for {year}.")

    races = races[:max_races]
    all_cells = []
    drivers_seen = set()

    for race in races:
        laps = load_race_laps(year, race)
        if laps is None or laps.empty:
            continue

        deg_rates = compute_degradation_rate(laps)
        compound_rates = deg_rates[
            (deg_rates["Compound"] == compound.upper()) &
            (deg_rates["StintLaps"] >= 4)
        ]

        for _, row in compound_rates.iterrows():
            driver = str(row["Driver"])
            drivers_seen.add(driver)
            all_cells.append(SeasonHeatmapCell(
                driver=driver,
                race_name=race,
                deg_rate=safe_float(row["DegRate"]),
            ))

    return SeasonHeatmapResponse(
        year=year,
        compound=compound.upper(),
        drivers=sorted(list(drivers_seen)),
        races=races,
        cells=all_cells,
    )