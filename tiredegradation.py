"""
F1 Tire Degradation Analyzer  - 2024 Season

Features:
- Lap time delta per stint(degradation curves)
- Compound comparison (Soft / Medium / Hard)
- Driver vs driver on same compound
- Degradation ranking table per track

Usage:
python tiredegradation.py

Requirements:
 pip install fastf1 pandas numpy matplotlib seaborn scipy
 """


import warnings
warnings.filterwarnings("ignore")

import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# cache
CACHE_DIR = Path("./f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

OUTPUT_DIR = Path("./tire_analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

#calendar
def get_calendar(year: int) -> list[str]:
    """
    Fetch the official race calendar for any given year from FastF1
    Automatically excludes testing sessions.
    Works for the 2025,2026 and any future season as data becomes available.
    """
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        races = (schedule[schedule["EventFormat"] != "testing"]["EventName"]
                 .tolist())
        print(f" {year} calendar: {len(races)} races found")
        return races
    except Exception as e:
        print(f" Error could not fetch {year} calendar: {e}")
        return []

COMPOUND_COLORS = {
    "SOFT": "#E8002D",
    "MEDIUM": "#FFF200",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067FF",
}

COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

#data loading

def load_race_laps(year: int, race_name: str) -> pd.DataFrame | None:
    """Load and clean laps for a single race."""
    try: 
        session = fastf1.get_session(year, race_name, "R")
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps.copy()

        #cleaning
         #drops nulls early
        laps = laps[~laps["PitOutTime"].notna()] #removes out- laps
        laps = laps[~laps["PitInTime"].notna()] #removed in-laps
        laps = laps[laps["LapTime"].notna()]
        laps = laps[laps["Compound"].isin(COMPOUND_ORDER)]
        laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

        # remove outliers per driver per compound(>107% median)
        def filter_outliers(grp):
            median = grp["LapTimeSeconds"].median()
            return grp[grp["LapTimeSeconds"] <= median * 1.07]
        
        laps = laps.groupby(["Driver", "Compound"], group_keys=False).apply(filter_outliers)
        
        laps["RaceName"] = race_name
        laps["Year"] = year

        #stint lap number
        laps = laps.sort_values(["Driver", "LapNumber"])
        laps["StintLap"] = laps.groupby(["Driver", "Stint"]).cumcount() + 1
        
        print(f" {race_name}: {len(laps)} clean laps loaded")
        return laps
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def load_season(year: int = 2025,races: list[str] | None = None) -> pd.DataFrame:
    """
    Load laps for an entire season (or subset of races).
    Defaults to 2025 season. Pass year=2026 etc as future seasons become available.
    If no races list is provided, the full calendar is fetched dynamically.
    """
    target = races or get_calendar(year)
    if not target:
        raise RuntimeError(f"No races found for {year} - calendar may not yet be available")
    all_laps = []

    print(f"\nLoading {year} season data ({len(target)} races)...")
    for race in target:
        df = load_race_laps(year, race)
        if df is not None:
            all_laps.append(df)

    if not all_laps:
        raise RuntimeError("No data loaded - check FastF1 cache/connection.")
    
    combined = pd.concat(all_laps, ignore_index=True)
    print(f"\n Season loaded: {len(combined):,} total laps across "
              f"{combined['RaceName'].nunique()} races\n")
    return combined 
    

#degredation metrics
def compute_degradation_rate(laps: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Driver, RaceName, Compound) group, fit a linear regression
    of LaptTimeSeconds ~ StintLap to get degradation rate (seconds/lap)
    """
    records = []

    for (driver, race, compound), grp in laps.groupby(
            ["Driver", "RaceName", "Compound"]):
        grp = grp.sort_values("StintLap")
        if len(grp) < 4:
            continue

        x = grp["StintLap"].values
        y = grp["LapTimeSeconds"].values

        slope, intercept, r, p, _ = stats.linregress(x,y)

        records.append({
            "Driver": driver,
            "RaceName": race,
            "Compound": compound,
            "StintLaps": len(grp),
            "DegRate": round(slope, 4), #s/lap
            "BaseTime": round(intercept, 3), #predicted lap 1 time
            "R2": round(r**2, 4),
            "PValue": round(p,4),
            "MedianLapTime": round(y.median() if hasattr(y,'median')
                                   else np.median(y), 3),
        })
    
    df = pd.DataFrame(records)
    df["DegRatePct"] = (df["DegRate"] /df["BaseTime"] * 100).round(4)
    return df


def compute_stint_delta(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lap-by-lap time delta from the first lap of each stint.
    useful for plotting degradation curves.
    """
    def delta(grp):
        grp = grp.sort_values("StintLap").copy()
        grp["LapDelta"] = grp["LapTimeSeconds"] - grp["LapTimeSeconds"].iloc[0]
        return grp

    return(laps.groupby(["Driver", "RaceName", "Stint"], group_keys=False)
           .apply(delta))


#visualizations
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc",
    "xtick.color": "#ccc",
    "ytick.color": "#999",
    "text.color": "#eee",
    "grid.color": "#2a2a2a",
    "grid.linewidth": 0.8,
    "font.family": "monospace",
})


def plot_degradation_curves(laps: pd.DataFrame, race_name: str, year: int = 2025, save: bool = True):
    """
    Degradation curves per compound for a single race.
    Shows individual driver lines (faint) and median trend (bold).
    """
    race_laps = laps[laps["RaceName"] == race_name].copy()
    race_laps = compute_stint_delta(race_laps)
 
    compounds = [c for c in COMPOUND_ORDER if c in race_laps["Compound"].unique()]
    if not compounds:
        print(f"No compound data for {race_name}")
        return
 
    fig, axes = plt.subplots(1, len(compounds), figsize=(6 * len(compounds), 6), sharey=True)
    if len(compounds) == 1:
        axes = [axes]
 
    fig.suptitle(f"Tire Degradation Curves — {race_name} {year}", fontsize=14, y=1.02, color="#ffffff")
 
    for ax, compound in zip(axes, compounds):
        data = race_laps[race_laps["Compound"] == compound]
        color = COMPOUND_COLORS[compound]
 
        for driver, driver_grp in data.groupby("Driver"):
            driver_grp = driver_grp.sort_values("StintLap")
            ax.plot(driver_grp["StintLap"], driver_grp["LapDelta"],
                    color=color, alpha=0.15, linewidth=0.8)
 
        median = data.groupby("StintLap")["LapDelta"].median().reset_index()
        ax.plot(median["StintLap"], median["LapDelta"],
                color=color, linewidth=2.5, label="Median")
 
        ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_title(compound, color=color, fontsize=11)
        ax.set_xlabel("Stint Lap")
        ax.grid(True, alpha=0.4)
 
    axes[0].set_ylabel("Lap Time Delta (s)")
    plt.tight_layout()
 
    if save:
        path = OUTPUT_DIR / f"deg_curves_{race_name.replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {path}")
    plt.show()
    plt.close()
 
 
def plot_compound_comparison(deg_rates: pd.DataFrame, race_name: str, year: int = 2025, save: bool = True):
    """Bar chart comparing degradation rate by compound for a single race."""
    data = deg_rates[(deg_rates["RaceName"] == race_name) & (deg_rates["StintLaps"] >= 4)].copy()
 
    if data.empty:
        print(f"No degradation data for {race_name}")
        return
 
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Degradation Rate by Compound — {race_name} {year}", fontsize=13, color="#ffffff")
 
    for compound in [c for c in COMPOUND_ORDER if c in data["Compound"].unique()]:
        sub = data[data["Compound"] == compound].sort_values("DegRate")
        ax.barh(sub["Driver"], sub["DegRate"],
                color=COMPOUND_COLORS[compound], alpha=0.85, label=compound, height=0.6)
 
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Degradation Rate (s/lap)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.4)
    plt.tight_layout()
 
    if save:
        path = OUTPUT_DIR / f"compound_cmp_{race_name.replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {path}")
    plt.show()
    plt.close()
 
 
def plot_driver_vs_driver(
    laps: pd.DataFrame,
    race_name: str,
    driver1: str,
    driver2: str,
    compound: str,
    year: int = 2025,
    save: bool = True,
):
    """Compare two drivers' tire degradation on the same compound at a given race."""
    data = laps[
        (laps["RaceName"] == race_name) &
        (laps["Compound"] == compound) &
        (laps["Driver"].isin([driver1, driver2]))
    ].copy()
 
    if data.empty:
        print(f"No data for {driver1} vs {driver2} on {compound} at {race_name}")
        return
 
    data = compute_stint_delta(data)
    color = COMPOUND_COLORS[compound]
 
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"{driver1} vs {driver2} — {compound} | {race_name} {year}",
        fontsize=13, color="#ffffff"
    )
 
    styles = ["-", "--"]
    for (driver, driver_grp), style in zip(data.groupby("Driver"), styles):
        med = driver_grp.groupby("StintLap")["LapDelta"].median().reset_index()
        ax.plot(med["StintLap"], med["LapDelta"],
                color=color, linestyle=style, linewidth=2.2, label=driver)
 
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Stint Lap")
    ax.set_ylabel("Lap Time Delta (s)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
 
    if save:
        path = OUTPUT_DIR / f"d_vs_d_{driver1}_{driver2}_{race_name.replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {path}")
    plt.show()
    plt.close()
 
 
def plot_season_degradation_heatmap(
    deg_rates: pd.DataFrame,
    year: int = 2025,
    compound: str = "MEDIUM",
    top_n_drivers: int = 15,
    save: bool = True,
):
    """
    Heatmap of drivers (rows) vs races (cols) showing mean degradation rate
    for one compound across the season. Uses seaborn for the heatmap rendering.
    """
    data = deg_rates[
        (deg_rates["Compound"] == compound) &
        (deg_rates["StintLaps"] >= 4)
    ].copy()
 
    if data.empty:
        print(f"No season heatmap data for {compound}")
        return
 
    pivot = data.pivot_table(index="Driver", columns="RaceName", values="DegRate", aggfunc="mean")
 
    top_drivers = pivot.count(axis=1).nlargest(top_n_drivers).index
    pivot = pivot.loc[top_drivers]
 
    race_order = [r for r in get_calendar(year) if r in pivot.columns]
    pivot = pivot[race_order]
 
    fig, ax = plt.subplots(figsize=(max(14, len(race_order) * 0.7), max(6, top_n_drivers * 0.45)))
    fig.suptitle(f"Season Degradation Heatmap — {compound} Compound {year}", fontsize=13, color="#ffffff")
 
    sns.heatmap(
        pivot, ax=ax,
        cmap="RdYlGn_r",
        center=0,
        annot=True, fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.3,
        linecolor="#0f0f0f",
        cbar_kws={"label": "Deg Rate (s/lap)"},
    )
 
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()
 
    if save:
        path = OUTPUT_DIR / f"season_heatmap_{compound}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {path}")
    plt.show()
    plt.close()
 
 
def plot_degradation_ranking(deg_rates: pd.DataFrame, race_name: str, year: int = 2025, save: bool = True):
    """Horizontal ranked bar chart of drivers by mean degradation rate at a given race."""
    data = (
        deg_rates[
            (deg_rates["RaceName"] == race_name) &
            (deg_rates["StintLaps"] >= 4)
        ]
        .groupby("Driver")["DegRate"]
        .mean()
        .sort_values()
        .reset_index()
    )
 
    if data.empty:
        print(f"No ranking data for {race_name}")
        return
 
    fig, ax = plt.subplots(figsize=(8, max(5, len(data) * 0.35)))
    fig.suptitle(f"Driver Degradation Ranking — {race_name} {year}", fontsize=13, color="#ffffff")
 
    colors = ["#E8002D" if v > 0 else "#39B54A" for v in data["DegRate"]]
    ax.barh(data["Driver"], data["DegRate"], color=colors, height=0.65)
    ax.axvline(0, color="#aaaaaa", linewidth=0.8)
    ax.set_xlabel("Mean Degradation Rate (s/lap)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
 
    if save:
        path = OUTPUT_DIR / f"ranking_{race_name.replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {path}")
    plt.show()
    plt.close()

#summmarytable
def build_summary_table(deg_rates: pd.DataFrame) -> pd.DataFrame:
    """Season-level driver degradation summary across all compounds"""
    summary = (deg_rates[deg_rates["StintLaps"] >= 4]
               .groupby(["Driver", "Compound"])
               .agg(
                   MeanDegRate=("DegRate", "mean"),
                   StdDegRate=("DegRate", "std"),
                   Races=("RaceName", "nunique"),
                   TotalStints=("StintLaps", "count"),
                )
                .round(4)
                .reset_index()
    )            
    return summary.sort_values(["Compound", "MeanDegRate"])

#main

def main():
    # 2025 = last full season (all races available in FastF1)
    # 2026 = current season (only completed races load, future rounds skipped)
    #
    # Quick test with 5 races from 2025:
    test_races = ["Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix", "Japanense Grand Prix", "Chinese Grand Prix"]
    # Full 2025 season:  load_season(2025)
    # 2026 so far:       load_season(2026)
 
    year = 2025
    laps = load_season(year, races=test_races)
    deg_rates = compute_degradation_rate(laps)
 
    # Save CSVs
    laps.to_csv(OUTPUT_DIR / "laps_clean.csv", index=False)
    deg_rates.to_csv(OUTPUT_DIR / "degradation_rates.csv", index=False)
    build_summary_table(deg_rates).to_csv(OUTPUT_DIR / "season_summary.csv", index=False)
    print(f"\nCSVs saved to {OUTPUT_DIR}/\n")
 
    # Per-race plots
    for race in laps["RaceName"].unique():
        print(f"\n── {race} ──")
        plot_degradation_curves(laps, race, year=year)
        plot_compound_comparison(deg_rates, race, year=year)
        plot_degradation_ranking(deg_rates, race, year=year)
 
    # Season heatmap per compound
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        plot_season_degradation_heatmap(deg_rates, year=year, compound=compound)
 
    print("\n All analysis complete!")
 
 
if __name__ == "__main__":  
    main()                                           