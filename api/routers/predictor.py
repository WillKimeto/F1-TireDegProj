"""
routers/predictor.py — Race Predictor endpoints
================================================
Imports directly from race_winner_predictor.py in the project root.

Endpoints:
    GET /predict/race?year=2026&race=Bahrain+Grand+Prix&model=ensemble
    GET /predict/train    — triggers model training (slow, run once)
"""

import pickle
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks

from racewinnerpredictor import (
    build_features,
    statistical_prediction,
    ml_prediction,
    predict_race,
    build_training_data,
    train_model,
    MODEL_PATH,
    SCALER_PATH,
)
from api.schemas import (
    PredictionResponse,
    DriverPrediction,
    ConstructorPrediction,
    SessionPaceDelta,
)

router = APIRouter()

# Module-level model cache — loaded once on first prediction request
_model = None
_scaler = None


def load_model_if_available():
    """Load saved model and scaler into memory if they exist."""
    global _model, _scaler
    if _model is not None:
        return True
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
        return True
    return False


def safe_float(val, decimals: int = 4) -> float:
    try:
        f = float(val)
        return round(f, decimals) if not (f != f) else 0.0
    except (TypeError, ValueError):
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/race", response_model=PredictionResponse)
def predict_race_endpoint(
    year: int = Query(2026, description="Season year for prediction"),
    race: str = Query(..., description="Exact race name e.g. 'Bahrain Grand Prix'"),
    model: str = Query("ensemble", description="Model type: ensemble, statistical, ml"),
    stat_weight: float = Query(0.35, description="Weight for statistical model (0-1)"),
    ml_weight: float = Query(0.65, description="Weight for ML model (0-1)"),
):
    """
    Predict the race winner for a given race weekend.
    Requires that FP1, FP2, FP3, and/or Qualifying have already taken place
    so FastF1 has session data to pull pace from.
    """
    model_loaded = load_model_if_available()

    # If ml-only requested but no model, fall back gracefully
    active_model = None
    active_scaler = None

    if model in ("ensemble", "ml") and model_loaded:
        active_model = _model
        active_scaler = _scaler
    elif model == "ml" and not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="No trained ML model found. Run GET /predict/train first."
        )

    try:
        driver_pred, constructor_pred = predict_race(
            year=year,
            race=race,
            model=active_model,
            scaler=active_scaler,
            stat_weight=stat_weight,
            ml_weight=ml_weight,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if driver_pred.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No session data found for {race} {year}. "
                   "Ensure FP or Qualifying sessions have occurred."
        )

    # Build driver prediction list
    all_drivers = [
        DriverPrediction(
            driver=str(row["Driver"]),
            constructor=str(row.get("Constructor", "Unknown")),
            predicted_rank=int(row["PredictedRank"]),
            win_probability=safe_float(row["WinProbability"], 2),
            stat_score=safe_float(row.get("StatScore", 0), 2),
            ml_score=safe_float(row["MLScore"], 2) if "MLScore" in row and row["MLScore"] == row["MLScore"] else None,
            ensemble_score=safe_float(row.get("EnsembleScore", row.get("StatScore", 0)), 2),
            quali_position=int(row["quali_position"]) if "quali_position" in row else None,
            weighted_pace_delta=safe_float(row.get("weighted_pace_delta", 0), 4),
            sessions_available=int(row.get("sessions_available", 0)),
        )
        for _, row in driver_pred.iterrows()
    ]

    # Build constructor prediction list
    constructors = [
        ConstructorPrediction(
            constructor=str(row["Constructor"]),
            constructor_rank=int(row["ConstructorRank"]),
            win_probability=safe_float(row["ConstructorWinProb"], 2),
        )
        for _, row in constructor_pred.iterrows()
    ]

    # Session pace deltas for chart
    session_cols = ["q_delta", "fp3_delta", "fp2_delta", "fp1_delta"]
    session_deltas = [
        SessionPaceDelta(
            driver=str(row["Driver"]),
            q_delta=safe_float(row["q_delta"]) if "q_delta" in row and row["q_delta"] == row["q_delta"] else None,
            fp3_delta=safe_float(row["fp3_delta"]) if "fp3_delta" in row and row["fp3_delta"] == row["fp3_delta"] else None,
            fp2_delta=safe_float(row["fp2_delta"]) if "fp2_delta" in row and row["fp2_delta"] == row["fp2_delta"] else None,
            fp1_delta=safe_float(row["fp1_delta"]) if "fp1_delta" in row and row["fp1_delta"] == row["fp1_delta"] else None,
        )
        for _, row in driver_pred.iterrows()
    ]

    sessions_used = int(driver_pred["sessions_available"].iloc[0]) if not driver_pred.empty else 0

    return PredictionResponse(
        race_name=race,
        year=year,
        model_type=model if model_loaded else "statistical",
        sessions_used=sessions_used,
        podium=all_drivers[:3],
        all_drivers=all_drivers,
        constructors=constructors,
        session_deltas=session_deltas,
        model_loaded=model_loaded,
    )


@router.get("/train")
def trigger_training(background_tasks: BackgroundTasks):
    """
    Triggers model training on 2024 + 2025 data.
    Training runs in the background — this endpoint returns immediately.
    Check /health to confirm the server is still running after training completes.

    Note: training downloads FastF1 data for every race in 2024+2025.
    First run can take 20-40 minutes depending on cache state.
    """
    def run_training():
        global _model, _scaler
        df = build_training_data(years=[2024, 2025])
        model, scaler, ndcg = train_model(df)
        _model = model
        _scaler = scaler
        print(f"Training complete. NDCG@3: {ndcg:.4f}")

    background_tasks.add_task(run_training)
    return {
        "status": "training_started",
        "message": "Model training has started in the background. "
                   "This will take several minutes. Check server logs for progress."
    }