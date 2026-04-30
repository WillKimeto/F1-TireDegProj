"""
F1 Race Winner Predictor
Predicts Top 3 race finishers( driver + constructor) using:
- free practice 1, 2, 3 session pace
- qualifying session lap times
- historical race results for model training

Architecture:
- statistical baseline: wieghed pace ranking across sessions
- ML layer: XGBoost ranker trained on 2024+2025 historical data
- Missing sessions are handled by reweighing remaining available sessions
- Output: Top 3 ranked drivers + full probability distribution

Session weights:
- Qualifying: 0.50
-FP3: 0.25
- FP2: 0.15
- FP1: 0.10

Usage:
#train on historical data
python racewinnerpredictor.py --mode train

#predict current race weekend(only after FP/Quali sessions)
python racewinnerpredictor.py --mode predict --year 2026 --race"Japan"

Requirements:
 pip install fastf1 pandas numpy matplotlib scikit-learn xgboost scipy
"""


import warnings
warnings.filterwarnings("ignore")

import argparse
import pickle
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score
import xgboost as xgb

#cache
CACHE_DIR = Path("./f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

OUTPUT_DIR = Path("./predictor_output")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "xgb_ranker.pk1"
SCALER_PATH = OUTPUT_DIR / "scaler.pk1"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"

#session weights
SESSION_WEIGHTS = {
    "Q":   0.65,
    "FP3": 0.20,
    "FP2": 0.10,
    "FP1": 0.05,
}

#plot settings
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#cccccc",
    "xtick.color": "#999999",
    "ytick.color": "#999999",
    "text.color": "#eeeeee",
    "grid.color": "#2a2a2a",
    "grid.linewidth": 0.8,
    "font.family": "monospace",
})

PODIUM_COLORS = ["#FFD700", "#C0C0C0", "#CD7F32"]

#calendar
def get_calendar(year: int):
    """Return list of race event names for a given season."""
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        return schedule[schedule["EventFormat"] != "testing"]["EventName"].tolist()
    except Exception as e:
        print(f"Could not fetch {year} calendar: {e}")
        return []

# session loading
def load_session_pace(year: int, race: str, session_type: str):
    """
    Load best lap time per driver for a given session.
    Returns a dict {driver_code: best_lap_seconds} or None if session cannot be loaded (e.g. session cancelled or not yet run).
    """
    try:
        session = fastf1.get_session(year, race, session_type)
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps.copy()

        if laps.empty:
            return None
        
        laps = laps[laps["LapTime"].notna()]
        laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

        # Remove outliers per driver (>107% of their own best)
        def clean(grp):
            best = grp["LapTimeSeconds"].min()
            return grp[grp["LapTimeSeconds"] <= best * 1.07]
        
        laps = laps.groupby("Driver", group_keys=False).apply(clean)

        # Build number → abbreviation map from session results
        # Needed because some sessions return driver numbers instead of abbreviations
        abbrev_map = {}
        try:
            for _, row in session.results.iterrows():
                num    = str(row.get("DriverNumber", ""))
                abbrev = str(row.get("Abbreviation", ""))
                if num and abbrev:
                    abbrev_map[num] = abbrev
        except Exception:
            pass

        # Get best lap per driver key (may be number or abbreviation)
        best_laps_raw = laps.groupby("Driver")["LapTimeSeconds"].min().to_dict()

        # Normalise all keys to abbreviations
        best_laps = {}
        for driver_key, time in best_laps_raw.items():
            abbrev = abbrev_map.get(str(driver_key), driver_key)
            best_laps[abbrev] = time

        return best_laps if best_laps else None
    
    except Exception:
        return None


def load_race_results(year: int, race: str):
    """Load race finishing positions for a completed race.
    Returns a dict {driver_code: finish_position} or None.
    """
    try:
        session = fastf1.get_session(year, race, "R")
        session.load(telemetry=False, weather=False, messages=False)
        results = session.results

        if results is None or results.empty:
            return None
        
        pos_map = {}
        for _, row in results.iterrows():
            driver = row.get("Abbreviation") or row.get("DriverId")
            position = row.get("Position") or row.get("ClassifiedPosition")
            if driver and position:
                try:
                    pos_map[str(driver)] = int(float(position))
                except (ValueError, TypeError):
                    continue

        return pos_map if pos_map else None

    except Exception:
        return None


def load_constructor_map(year: int, race: str) -> dict:
    """Return {driver_code: team_name} for a given race weekend"""
    try:
        session = fastf1.get_session(year, race, "Q")
        session.load(telemetry=False, weather=False, messages=False)
        results = session.results
        if results is None or results.empty:
            return{}
        return {
            str(row.get("Abbreviation", "")): str(row.get("TeamName", "Unknown"))
            for _, row in results.iterrows()
            if row.get("Abbreviation")
        } 
    except Exception:
        return{}
    
#feature engineering
def reweight_session(available: dict) -> dict:
    """
    Reweight session weights when one or more sessions are missing.
    Distributes missing weight proportionally across available sessions.
    """
    total = sum(SESSION_WEIGHTS[s] for s in available if s in SESSION_WEIGHTS)
    if total == 0:
        return{}
    return{s: SESSION_WEIGHTS[s] / total for s in available if s in SESSION_WEIGHTS}


def build_features(year: int, race: str) -> pd.DataFrame:
    """
    Build a feature DataFrame for all drivers at a given race weekend.
    
    Features per driver:
    - weighted_pace_delta : weighted average delta for fastest driver (s)
    - quali_dleta : delta from pole(s), NaN if no quali
    - fp3_delta: delta from FP3(s)
    - fp2_delta: delta from FP2 fastest(s)
    - fp1_delta: delta from FP1 fastest(s)
    - sessions_available: number of sessions with valid data (0-4)
    - quali_position: qualifying position (1-based)
    - pace consistency: std dev of deltas across available sessions
    - comnstructor_avg_delta: mean delta of both drivers(team strength proxy)
    
    Returns DataFrame with one row per driver, or empty DataFrame on failure
    """
    session_map = {"Q": "Qualifying", "FP3": "FP3", "FP2": "FP2", "FP1": "FP1"}
    raw_paces = {}

    for key in ["Q", "FP3", "FP2", "FP1"]:
        pace = load_session_pace(year,race,key)
        if pace:
            raw_paces[key] = pace

    if not raw_paces:
        print(f"No session data available for {race} {year}")
        return pd.DataFrame()

    weights = reweight_session(raw_paces)

    #collect all drivers seen across any session
    all_drivers = set()
    for pace in raw_paces.values():
        all_drivers.update(pace.keys())

    if not all_drivers:
        return pd.DataFrame()

    records = []
    for driver in all_drivers:
        row = {"Driver": driver, "Year": year, "RaceName": race}

        session_deltas = []
        for key, pace in raw_paces.items():
            if not pace:
                continue
            fastest = min(pace.values())
            
            if driver in pace:
               delta = pace[driver] - fastest
            else:
                delta = float(np.median(list(pace.values()))) - fastest
            col = f"{key.lower()}_delta"
            row[col] = round(delta, 4)
            session_deltas.append(delta * weights.get(key, 0))

            #fill missing session columns with NaN
        for key in ["Q", "FP3", "FP2", "FP1"]:
                col = f"{key.lower()}_delta"
                if col not in row:
                    row[col] = np.nan

        row["weighted_pace_delta"] = round(sum(session_deltas), 4)
        row["sessions_available"] = len(raw_paces)

            #qualifying position
        if "Q" in raw_paces and raw_paces["Q"]:
                sorted_q = sorted(raw_paces["Q"].items(), key=lambda x: x[1])
                quali_positions = {d: i + 1 for i, (d, _) in enumerate(sorted_q)}
                row["quali_position"] = quali_positions.get(driver, len(all_drivers))
        else:
                row["quali_position"] = len(all_drivers)

                #pace consistency across sessions
        available_deltas = [
                row[f"{k.lower()}_delta"]
                for k in raw_paces
                if not np.isnan(row.get(f"{k.lower()}_delta", np.nan))
            ]
        row["pace_consistency"] = round(float(np.std(available_deltas)), 4) if len(available_deltas) > 1 else 0.0

        records.append(row)

    df = pd.DataFrame(records)

                #constructor map
    constructor_map = load_constructor_map(year, race)
    df["Constructor"] = df["Driver"].map(constructor_map).fillna("Unknown")

                #constructor average pace delta ( team strength proxy)
    team_avg = df.groupby("Constructor")["weighted_pace_delta"].transform("mean")
    df["constructor_avg_delta"] = team_avg.round(4)

    return df

#training data builder
def build_training_data(years: list = None) -> pd.DataFrame:
    """
    Build full training dataset from completed race weekends.
    For each race: builds features from FP/Quali, loads actual finish positions.
    Saves to CSV for reuse.
    """
    if years is None:
        years = [2024,2025]

    all_records = []

    for year in years:
        races = get_calendar(year)
        print(f"\nBuilding training data for {year} ({len(races)} races)...")

        for race in races:
            print(f" Processing {race}...")

            features = build_features(year, race)
            if features.empty:
                continue

            results = load_race_results(year, race)
            if not results:
                print(f" No race results for {race} {year} - skipping")
                continue

            features["FinishPosition"] = features["Driver"].map(results)
            features = features[features["FinishPosition"].notna()]

            if features.empty:
                continue

            features["FinishPosition"] = features["FinishPosition"].astype(int)

            #target: points-style relevance label(high = better finish)
            max_pos = features["FinishPosition"].max()
            features["Relevance"] = (max_pos - features["FinishPosition"] + 1).clip(lower=0)

            all_records.append(features)
            print(f" {race}: {len(features)} drivers")

    if not all_records:
        raise RuntimeError("No training data collected.")

    df = pd.concat(all_records, ignore_index=True)
    df.to_csv(HISTORY_PATH, index=False)
    print(f"\n Training data saved: {len(df)} rows across {df['RaceName'].nunique()} races")
    return df

#model training
FEATURE_COLS = [
    "weighted_pace_delta",
    "quali_delta",
    "fp3_delta",
    "fp2_delta",
    "fp1_delta",
    "quali_position",
    "pace_consistency",
    "constructor_avg_delta",
    "sessions_available",
]

def train_model(df: pd.DataFrame):
    """
    Train XGBoost learning-to-rank model on historical race data.
    Uses Leave-One-Group-Out CV(per race) for validation.
    Saves trained model and scaler to disk
    """
    df = df.copy()

    #encode race group for LOGO CV
    race_groups = df["RaceName"].astype("category").cat.codes.values

    #fill NaN features with column median
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    x = df[FEATURE_COLS].values
    y = df["Relevance"].values
    groups = df.groupby(["Year", "RaceName"]).ngroup().values

    #scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    #group sizes for xgboost ranker
    group_sizes = df.groupby(["Year", "RaceName"]).size().values

    model = xgb.XGBRanker(
        objective="rank:ndcg",
        learning_rate=0.05,
        n_estimators=300,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )         
    print("\nTraining XGBoost ranker...")
    model.fit(X_scaled, y, group=group_sizes)

    #leave-one-race-out validation
    logo = LeaveOneGroupOut()
    ndcg_scores = []

    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        grp_tr = df.iloc[train_idx].groupby(["Year", "RaceName"]).size().values
        grp_te = df.iloc[test_idx].groupby(["Year", "RaceName"]).size().values

        cv_model = xgb.XGBRanker(
            objective="rank:ndcg",
            learning_rate=0.05,
            n_estimators=300,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        cv_model.fit(X_tr, y_tr, group=grp_tr)
        preds = cv_model.predict(X_te)

        if len(y_te) > 1:
            score = ndcg_score([y_te], [preds], k=3)
            ndcg_scores.append(score)

    mean_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    print(f" Leave-one-race-out NDCG@3: {mean_ndcg: .4f}")

    #save model and scalar
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler,f)

    print(f" Model saved: {MODEL_PATH}")
    print(f" Scaler saved: {SCALER_PATH}")

    #feature importance plot
    plot_feature_importance(model)

    return model, scaler, mean_ndcg


def plot_feature_importance(model, save:bool = True):
    """Bar chart of XGBoost feature importances"""
    importance = model.feature_importances_
    indices = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(8,5))
    fig.suptitle("Feature Importance - XGBoost Ranker", fontsize=13, color="#ffffff")

    colors = ["#E8002D" if importance[i] == max(importance) else "#555555" for i in indices]
    ax.barh(
        [FEATURE_COLS[i] for i in indices],
        importance[indices],
        color=colors, height=0.6,
    )
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / "feature_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    plt.show()
    plt.close()

#statistical baseline
def statistical_prediction(features: pd.DataFrame) -> pd.DataFrame:
    """
    Rank drivers purely by weighted pace delta across sessions.
    lower weighted_pace_delta = faster = ranked higher.
    Returns dataframe sorted by predicted rank with a normalized score.
    """
    df = features.copy()
    df = df.sort_values("weighted_pace_delta").reset_index(drop=True)
    df["StatRank"] = df.index + 1

    #normalise to a 0-100score(100=fastest)
    min_d = df["weighted_pace_delta"].min()
    max_d = df["weighted_pace_delta"].max()
    spread = max_d - min_d if max_d != min_d else 1.0
    df["StatScore"] = ((max_d - df["weighted_pace_delta"]) / spread * 100).round(2)

    return df

#ML Prediction
def ml_prediction(features: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """
    Run XGBoost ranker on a race weekend feature set.
    Returns DataFrame with ML rank and porbability-style scores.
    """
    df = features.copy()

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    X = scaler.transform(df[FEATURE_COLS].values)
    raw_scores = model.predict(X)

    df["MLScore"] = raw_scores
    df = df.sort_values("MLScore", ascending=False).reset_index(drop=True)
    df["MLRank"] = df.index + 1

    #softmax-style probability distribution
    exp_scores = np.exp(raw_scores - raw_scores.max())
    df["MLProbability"] = (exp_scores / exp_scores.sum() * 100).round(2)

    return df


#ensemble prediction
def predict_race(
    year: int,
    race:str,
    model=None,
    scaler=None,
    stat_weight: float = 0.35,
    ml_weight: float = 0.65,
) -> pd.DataFrame:
    """
    Generate Top 3 + full probability for a race weekend
    
    Combines:
    -Statistical baseline(weighted pace ranking)
    - XGBoost ML ranker (if model available)
    
    if no model is loaded, falls back to statistical only.
    """
    print(f"\nBuilding features for {race}, {year}... ")
    features = build_features(year, race)

    if features.empty:
        raise RuntimeError(f"Could not build features for {race} {year}.")
    
    #statistical baseline
    stat_df = statistical_prediction(features)

    #ml layer
    if model is not None and scaler is not None:
        ml_df = ml_prediction(features, model, scaler)

        merged = stat_df.merge(
            ml_df[["Driver", "MLRank", "MLScore", "MLProbability"]],
            on="Driver", how="left",
        )
        # Safety: ensure StatScore survived the merge
        if "StatScore" not in merged.columns:
            merged["StatScore"] = stat_df.set_index("Driver").loc[merged["Driver"], "StatScore"].values

        #normalise stat score to 0-1
        stat_norm = merged["StatScore"] / 100.0
        ml_norm = merged["MLProbability"] / 100.0

        merged["EnsembleScore"] = (
            stat_weight * stat_norm + ml_weight * ml_norm
        ) * 100

        merged = merged.sort_values("EnsembleScore", ascending=False).reset_index(drop=True)
        merged["PredictedRank"] = merged.index + 1

        exp_ens = np.exp(merged["EnsembleScore"] - merged["EnsembleScore"].max())
        merged["WinProbability"] = (exp_ens / exp_ens.sum() * 100).round(2)

    else:
        print(" No ML model found - using statistical baseline only")
        merged = stat_df.copy()
        merged["PredictedRank"] = merged["StatRank"]
        merged["WinProbability"] = merged["StatScore"]
        merged["MLRank"] = np.nan
        merged["MLProbability"] = np.nan
        merged["EnsembleScore"] = merged["StatScore"]
 
    # Constructor prediction — aggregate driver probabilities per team
    constructor_pred = (
        merged.groupby("Constructor")["WinProbability"]
        .sum()
        .reset_index()
        .rename(columns={"WinProbability": "ConstructorWinProb"})
        .sort_values("ConstructorWinProb", ascending=False)
        .reset_index(drop=True)
    )
    constructor_pred["ConstructorRank"] = constructor_pred.index + 1
 
    return merged, constructor_pred


#visualization 
def plot_prediction(
        driver_pred: pd.DataFrame,
        constructor_pred: pd.DataFrame,
        race: str,
        year: int,
        save: bool = True,
):
    """
    4-panel prediction dashboard:
       [0] Top 3 podium cards
       [1] Full driver win probability distribution
       [2] Constructor probability chart
       [3] Session pace delta heatmap(quali vs FP3 vs FP2 vs FP1)
       """
    fig = plt.figure(figsize=(16,12))
    fig.suptitle(
        f"Race Prediction - {race} {year}",
        fontsize=15, color="#ffffff", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    #panel 0: Top 3 podium
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.set_title("Predicted Podium", color="#aaaaaa", fontsize=11, pad=10)

    top3 = driver_pred.head(3)
    for i, (_,row) in enumerate(top3.iterrows()):
        color = PODIUM_COLORS[i]
        y_pos = 0.75 - i * 0.32
        ax0.add_patch(plt.Rectangle((0.05, y_pos - 0.1), 0.9, 0.28,
                                    color=color, alpha=0.15,
                                    transform=ax0.transAxes))
        ax0.text(0.12, y_pos + 0.08, f"P{i+1}", fontsize=18,
                 color=color, transform=ax0.transAxes)
        ax0.text(0.28, y_pos + 0.08, row["Driver"], fontsize=14,
                 color="#ffffff", transform=ax0.transAxes)
        ax0.text(0.28, y_pos - 0.01, row.get("Constructor", ""),
                 fontsize=9, color="#888888", transform=ax0.transAxes)
        ax0.text(0.75, y_pos + 0.08, f"{row['WinProbability']:.if}%",
                 fontsize=13, color=color, transform=ax0.transAxes)
        
        #panel 1: driver probability distribution
    ax1 = fig.add_subplot(gs[0, 1])
    top_n = driver_pred.head(12)
    colors_bar = [PODIUM_COLORS[i] if i < 3 else " #444444"
                  for i in range(len(top_n))]
    ax1.barh(top_n["Driver"][::-1], top_n["WinProbability"][::-1],
            color=colors_bar[::-1], height=0.65)
    ax1.set_xlabel("Win Probability (%)")
    ax1.set_title("Driver Win Probability", color="#aaaaaa", fontsize=11)
    ax1.grid(True, axis="x", alpha=0.3)

        #panel 2 Constructor probability
    ax2 = fig.add_subplot(gs[1, 0])
    top_teams = constructor_pred.head(8)
    team_colors = ["#E8002D" if i == 0 else "#555555" for i in range(len(top_teams))]
    ax2.barh(top_teams["Constructor"][::-1],
            top_teams["ConstructorWinProb"][::-1],
            color=team_colors[::-1], height=0.65)
    ax2.set_xlabel("Aggregated Win Probability (%)")
    ax2.set_title("Constructor Probability", color="#aaaaaa", fontsize=11)
    ax2.grid(True, axis="x", alpha=0.3)

        #panel 3 Session pace delta comparison
    ax3 = fig.add_subplot(gs[1, 1])
    top15 = driver_pred.head(15)
    delta_cols = [c for c in ["q_delta", "fp3_delta", "fp2_delta", "fp1_delta"]
                 if c in top15.columns and top15[c].notna().any()]
        
    if delta_cols:
        x = np.arange(len(top15))
        width = 0.8 / len(delta_cols)
        session_colors = {"q_delta": "#E8002D", "fp3_delta": "#FFF200",
                              "fp2_delta": "#39B54A", "fp1_delta": "#0067FF"}
        session_labels = {"q_delta": "Quali", "fp3_delta": "FP3",
                              "fp2_delta": "FP2", "fp1_delta": "FP1"}

        for idx, col in enumerate(delta_cols):
            offset = (idx - len(delta_cols) / 2) * width + width / 2
            vals = top15[col].fillna(top15[col].median())
            ax3.bar(x + offset, vals,
                    width=width * 0.9,
                    color=session_colors.get(col, "#888888"),
                    alpha=0.8,
                    label=session_labels.get(col, col))

        ax3.set_xticks(x)
        ax3.set_xticklabels(top15["Driver"], rotation=45, ha="right", fontsize=8)    
        ax3.set_ylabel("Delta from Fastest (s)")
        ax3.set_title("Session Pace Deltas", color="#aaaaaa", fontsize=11)
        ax3.grid(True, axis="y", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No session delta data", ha="center", va="center", color="#888888, transform=ax3.transAxes")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        path = OUTPUT_DIR / f"prediction_{race.replace(' ', '_')}_{year}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f" Saved: {path}")
    plt.show()
    plt.close()


def print_prediction_summary(
        driver_pred: pd.DataFrame,
        constructor_pred: pd.DataFrame,
        race: str,
        year: int,
):
    """Print a clean text summary of the prediction to console."""
    print(f"\n{'=' * 55}")
    print(f" Race Prediction - {race.upper()} {year}")
    print(f"{'=' * 55}")

    print("\n Predicted Podium {Drivers}")
    print(f" {'Pos' :<5} {'Driver':<8} {'Constructor' :<22} {'Probability'}")
    print(f" { '-' * 50}")
    for _, row in driver_pred.head(3).iterrows():
        medal = ["P1", "P2", "P3"][int(row["Predicted Rank"]) - 1]
        print(
            f" {medal} P{int(row['Predicted Rank'])}"
            f"{row['Driver']:<8} {row.get('Constructor', 'Unknown'):<22}"
            f"{row['WinProbability']:.if}%"
        )

    print("\n Predicted Top 3 Constructors")
    print(f" {'Pos':<5} {'Constructor':<28} {'Probability'}")
    print(f" {'-' * 45}")
    for _, row in constructor_pred.head(3).iterrows():
        print(
            f"P{int(row['ConstructorRank'])}"
            f"{row['Constructor']:<28}"
            f"{row['ConstructorWinProb']:.if}%"
        )

    sessions = driver_pred["sessions_available"].iloc[0] if not driver_pred.empty else 0 
    model_note = "Ensemble (Statistical + XGBoost)" if "MLRank" in driver_pred.columns and driver_pred["MLRank"].notna().any() else "Statistical baseline only"
    print(f"\n Sessions used: {int(sessions)}/4")
    print(f" Model  : {model_note}")
    print(f"{'=' * 55}\n")


#main
def main():
    parser = argparse.ArgumentParser(description="F1 Race Winner Predictior")
    parser.add_argument(
       "--mode", choices=["train", "predict", "both"],
       default="both",
       help="train: build model from historical data | predict: predict a race | both: train then predict"
    )
    parser.add_argument("--year", type=int, default=2026, help="Year for prediction (default: 2026)")
    parser.add_argument("--race", type=str, default="Bahrain", help="Race name for prediction")
    parser.add_argument("--stat-weight", type=float, default=0.35, help="Weight for statistical model (0-1)")
    parser.add_argument("--ml-weight", type=float, default=0.65, help="Weight for ML model (0-1)")
    args = parser.parse_args()

    model, scaler = None, None 
                      
#train
    if args.mode in ("train", "both"):
       print("\nBuilding training dataset (2024 + 2025)...")
       df = build_training_data(years=[2024, 2025])
       model, scaler, ndcg = train_model(df)
       print(f"\n Training complete. NDCG@3: {ndcg:.4f}")

    #load saved model if predict only
    if args.mode == "predict":
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            print(f"Model loaded from {MODEL_PATH}")
        else:
             print("No saved model found - run with --mode train first. Falling back to statistical only.")

 #predict
    if args.mode in ("predict", "both"):
        driver_pred, constructor_pred = predict_race(
            year=args.year,
            race=args.race,
            model=model,
            scaler=scaler,
            stat_weight=args.stat_weight,
            ml_weight=args.ml_weight,
        )

        print_prediction_summary(driver_pred, constructor_pred, args.race, args.year)
        plot_prediction(driver_pred, constructor_pred, args.race, args.year)

        #saving prediction data to csv
        driver_pred.to_csv(OUTPUT_DIR / f"pred_drivers_{args.race.replace('', '_')}_{args.year}.csv", index=False)
        constructor_pred.to_csv(OUTPUT_DIR / f"pred_construction_{args.race.replace(' ', '_')}_{args.year}.csv", index=False)

if __name__ == "__main__":
    main()
                     