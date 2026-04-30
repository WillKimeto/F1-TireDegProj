"""
routers/conversion.py — Pace Conversion Score endpoints
========================================================
Imports directly from quali_race_conversion.py in the project root.

Endpoints:
    GET /conversion/season?year=2025
    GET /conversion/race?year=2025&race=Monaco+Grand+Prix
    GET /conversion/driver?year=2025&driver=VER
"""

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from qualiraceconversion import (
    build_season_scores,
    compute_season_summary,
    get_calendar,
)
from api.schemas import (
    ConversionSeasonResponse,
    DriverSeasonSummary,
    RaceConversionScore,
)

router = APIRouter()


def safe_float(val, decimals: int = 2) -> float:
    try:
        f = float(val)
        return round(f, decimals) if not (f != f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def safe_int_or_none(val):
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/season", response_model=ConversionSeasonResponse)
def get_season_conversion(
    year: int = Query(2025, description="Season year"),
):
    """
    Returns full season conversion scores for all drivers.
    Includes per-race raw scores (for heatmap) and season summaries
    (for rankings table, trend charts, scatter plot).
    """
    try:
        season_df = build_season_scores(year)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))

    summary_df = compute_season_summary(season_df)

    if summary_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No conversion data available for {year}."
        )

    # Build per-driver season summaries
    drivers = []
    for _, row in summary_df.iterrows():
        driver = str(row["Driver"])

        # Per-race scores and race names for trend chart
        driver_races = season_df[season_df["Driver"] == driver].sort_values("RaceIndex")
        race_scores = [safe_float(v, 2) for v in driver_races["ConversionScore"].fillna(0).tolist()]
        race_names  = [str(r) for r in driver_races["RaceName"].tolist()]

        # Average pace and position scores
        pace_avg = safe_float(driver_races["PaceScore"].mean(), 2) if "PaceScore" in driver_races else 0.0
        pos_avg  = safe_float(driver_races["PositionScore"].mean(), 2) if "PositionScore" in driver_races else 0.0

        drivers.append(DriverSeasonSummary(
            driver=driver,
            constructor=str(row.get("Constructor", "Unknown")),
            mean_score=safe_float(row["MeanScore"], 2),
            std_score=safe_float(row.get("StdScore", 0), 2),
            best_score=safe_float(row["BestScore"], 2),
            worst_score=safe_float(row["WorstScore"], 2),
            races_scored=int(row["RacesScored"]),
            flagged_races=int(row.get("FlaggedRaces", 0)),
            trend_slope=safe_float(row["TrendSlope"], 4),
            trend_label=str(row["TrendLabel"]),
            percentile=safe_float(row["Percentile"], 1),
            pace_score_avg=pace_avg,
            position_score_avg=pos_avg,
            race_scores=race_scores,
            race_names=race_names,
        ))

    # Build raw per-race scores for heatmap
    race_scores_list = [
        RaceConversionScore(
            driver=str(row["Driver"]),
            constructor=str(row.get("Constructor", "Unknown")),
            race_name=str(row["RaceName"]),
            year=int(row["Year"]),
            race_index=int(row["RaceIndex"]),
            quali_position=safe_int_or_none(row.get("QualiPosition")),
            finish_position=safe_int_or_none(row.get("FinishPosition")),
            position_delta=safe_float(row["PositionDelta"], 2) if row.get("PositionDelta") == row.get("PositionDelta") else None,
            quali_time=safe_float(row.get("QualiTime", 0), 3),
            median_race_pace=safe_float(row.get("MedianRacePace", 0), 3),
            pace_delta=safe_float(row["PaceDelta"], 3) if row.get("PaceDelta") == row.get("PaceDelta") else None,
            position_score=safe_float(row["PositionScore"], 2) if row.get("PositionScore") == row.get("PositionScore") else None,
            pace_score=safe_float(row["PaceScore"], 2) if row.get("PaceScore") == row.get("PaceScore") else None,
            conversion_score=safe_float(row["ConversionScore"], 2) if row.get("ConversionScore") == row.get("ConversionScore") else None,
            status=str(row.get("Status", "Unknown")),
            flagged=bool(row.get("Flagged", False)),
            dnf_dsq=bool(row.get("DNF_DSQ", False)),
            sc_affected=bool(row.get("SC_Affected", False)),
        )
        for _, row in season_df.iterrows()
    ]

    # Season-level metrics
    field_average = safe_float(summary_df["MeanScore"].mean(), 2)
    top_converter = str(summary_df.iloc[0]["Driver"]) if not summary_df.empty else "N/A"
    improving_count = int((summary_df["TrendLabel"] == "Improving").sum())
    total_races = len(get_calendar(year))

    return ConversionSeasonResponse(
        year=year,
        drivers=drivers,
        race_scores=race_scores_list,
        field_average=field_average,
        top_converter=top_converter,
        improving_count=improving_count,
        total_races=total_races,
    )


@router.get("/race", response_model=ConversionSeasonResponse)
def get_race_conversion(
    year: int = Query(2025),
    race: str = Query(..., description="Exact race name e.g. 'Monaco Grand Prix'"),
):
    """
    Returns conversion scores for a single race only.
    Useful for drilling down into one round.
    """
    try:
        season_df = build_season_scores(year, races=[race])
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))

    summary_df = compute_season_summary(season_df)

    if summary_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No conversion data for {race} {year}."
        )

    drivers = []
    for _, row in summary_df.iterrows():
        driver = str(row["Driver"])
        driver_races = season_df[season_df["Driver"] == driver].sort_values("RaceIndex")
        race_scores = [safe_float(v, 2) for v in driver_races["ConversionScore"].fillna(0).tolist()]
        race_names  = [str(r) for r in driver_races["RaceName"].tolist()]
        pace_avg = safe_float(driver_races["PaceScore"].mean(), 2) if "PaceScore" in driver_races else 0.0
        pos_avg  = safe_float(driver_races["PositionScore"].mean(), 2) if "PositionScore" in driver_races else 0.0

        drivers.append(DriverSeasonSummary(
            driver=driver,
            constructor=str(row.get("Constructor", "Unknown")),
            mean_score=safe_float(row["MeanScore"], 2),
            std_score=safe_float(row.get("StdScore", 0), 2),
            best_score=safe_float(row["BestScore"], 2),
            worst_score=safe_float(row["WorstScore"], 2),
            races_scored=int(row["RacesScored"]),
            flagged_races=int(row.get("FlaggedRaces", 0)),
            trend_slope=safe_float(row["TrendSlope"], 4),
            trend_label=str(row["TrendLabel"]),
            percentile=safe_float(row["Percentile"], 1),
            pace_score_avg=pace_avg,
            position_score_avg=pos_avg,
            race_scores=race_scores,
            race_names=race_names,
        ))

    race_scores_list = [
        RaceConversionScore(
            driver=str(row["Driver"]),
            constructor=str(row.get("Constructor", "Unknown")),
            race_name=str(row["RaceName"]),
            year=int(row["Year"]),
            race_index=int(row["RaceIndex"]),
            quali_position=safe_int_or_none(row.get("QualiPosition")),
            finish_position=safe_int_or_none(row.get("FinishPosition")),
            position_delta=safe_float(row["PositionDelta"], 2) if row.get("PositionDelta") == row.get("PositionDelta") else None,
            quali_time=safe_float(row.get("QualiTime", 0), 3),
            median_race_pace=safe_float(row.get("MedianRacePace", 0), 3),
            pace_delta=safe_float(row["PaceDelta"], 3) if row.get("PaceDelta") == row.get("PaceDelta") else None,
            position_score=safe_float(row["PositionScore"], 2) if row.get("PositionScore") == row.get("PositionScore") else None,
            pace_score=safe_float(row["PaceScore"], 2) if row.get("PaceScore") == row.get("PaceScore") else None,
            conversion_score=safe_float(row["ConversionScore"], 2) if row.get("ConversionScore") == row.get("ConversionScore") else None,
            status=str(row.get("Status", "Unknown")),
            flagged=bool(row.get("Flagged", False)),
            dnf_dsq=bool(row.get("DNF_DSQ", False)),
            sc_affected=bool(row.get("SC_Affected", False)),
        )
        for _, row in season_df.iterrows()
    ]

    return ConversionSeasonResponse(
        year=year,
        drivers=drivers,
        race_scores=race_scores_list,
        field_average=safe_float(summary_df["MeanScore"].mean(), 2),
        top_converter=str(summary_df.iloc[0]["Driver"]),
        improving_count=int((summary_df["TrendLabel"] == "Improving").sum()),
        total_races=1,
    )