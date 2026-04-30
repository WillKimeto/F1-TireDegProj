"""
F1 Qualifying → Race Pace Conversion Score — 2024 + 2025
=========================================================
Measures how well each driver converts qualifying pace into race pace.
 
Composite score (0-100, higher = better conversion):
  - Position component (50%): quali position vs finish position delta,
    normalised across the field
  - Pace component (50%): quali lap time vs median race pace delta,
    normalised across the field
 
Additional flags:
  - DNF / DSQ races are flagged and scored where lap data allows,
    otherwise marked as incomplete
  - Safety car / VSC periods are detected and flagged on affected races
 
Output per driver:
  - Conversion score per race
  - Season average score
  - Percentile rank among all drivers
  - Trend across season (improving / declining / stable)
 
Usage:
    python quali_race_conversion.py --year 2025
    python quali_race_conversion.py --year 2024 --year 2025
    python quali_race_conversion.py --year 2025 --race Monaco
 
Requirements:
    pip install fastf1 pandas numpy matplotlib seaborn scipy
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

#cache
CACHE_DIR = Path("./f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

OUTPUT_DIR = Path("./conversion_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

#weights
POSITION_WEIGHT = 0.50
PACE_WEIGHT = 0.50

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

#calendar
def get_calendar(year: int) -> list:
    """Return list of race event names for a given season ."""
    try:
       schedule = fastf1.get_event_schedule(year, include_testing=False)
       return schedule[schedule["EventFormat"] != "testing"]["EventName"].tolist()
    except Exception as e:
       print(f"Could not fetch {year} calendar: {e}")
       return[]
    
#session loading
def load_qualifying(year: int, race: str) -> dict:
   """
   Load qualifying results for a race weekend.
   Returns {driver: (best_lap_seconds, quali_position)} or empty dict.
   """
   try:
      session = fastf1.get_session(year, race, "Q")
      session.load(telemetry=False, weather=False, messages=False)
      laps = session.laps.copy()

      if laps.empty:
         return {}

      laps = laps[laps["LapTime"].notna()]
      laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

      #best lap per driver
      best = laps.groupby("Driver")["LapTimeSeconds"].min().reset_index()
      best = best.sort_values("LapTimeSeconds").reset_index(drop=True)
      best["QualiPosition"] = best.index + 1

      return {
         row["Driver"]: {
            "QualiTime":  round(row["LapTimeSeconds"], 4),
            "QualiPosition": int(row["QualiPosition"]), 
         }
         for _, row in best.iterrows()
      }

   except Exception as e:
      print(f" Quali load failed for {race} {year}: {e}")
      return {}


def load_race_pace(year: int, race: str) -> dict:
   """
   Load race pace data per driver.
   Returns {driver: {
      'median_pace': float,      - median clean lap time(s)
      'finish_position': int,
      'status': str,             - 'Finished', 'DNF', 'DSQ', etc
      'sc_laps': int,            - 'laps under safety car / VSC
      'total_laps': int,
      'flagged': bool            - True if SC/VSC affected or DNF/DSQ
    }}
    """   
   try: 
      session = fastf1.get_session(year, race, "R")
      session.load(telemetry=False, weather=False, messages=False)
      laps = session.laps.copy()
      results = session.results

      if laps.empty or results is None or results.empty:
         return {}

      laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

      #Detect SC / VSC laps via trackstatus
      # Track status codes: 1=clear, 2=yellow, 4=SC, 6=VSC, 7=red
      sc_laps = set()
      if "TrackStatus" in laps.columns:
         sc_mask = laps["TrackStatus"].astype(str).str.contains("4|6", na=False)
         sc_laps = set(laps[sc_mask]["LapNumber"].unique())

         driver_data = {}

         for _, driver_row in results.iterrows():
            driver = str(driver_row.get("Abbreviation", ""))
            if not driver:
               continue

            status = str(driver_row.get("Status", "Finished"))
            finish_pos = driver_row.get("Position") or driver_row.get("ClassifiedPosition")

            try:
               finish_pos = int(float(finish_pos))
            except (TypeError, ValueError):
               finish_pos = None


            d_laps = laps[laps["Driver"] == driver].copy()
            d_laps = d_laps[d_laps["LapTimeSeconds"].notna()]

            #remove SC/VSC affected laps and pit laps for clean pace
            clean_laps = d_laps[
               (~d_laps["LapNumber"].isin(sc_laps)) &
               (d_laps["PitOutTime"].isna()) &
               (d_laps["PitInTime"].isna())
            ]

            #remove outliers (>107% of driver's own median)       
            if not clean_laps.empty:
               med = clean_laps["LapTimeSeconds"].median()
               clean_laps = clean_laps[clean_laps["LapTimeSeconds"] <= med * 1.07]

            median_pace = float(clean_laps["LapTimeSeconds"].median()) if not clean_laps.empty else None

            is_dnf_dsq = status.upper() not in ("FINISHED", "+1 LAP", "+2 LAPS", "+3 LAPS", "+4 LAPS", "+5 LAPS", "1 LAP", "2 LAPS")
            sc_affected = len(sc_laps) > 3 #flag if more than 3 SC laps in a race

            driver_data[driver] = {
               "median_pace":  median_pace,
               "finish_position": finish_pos,
               "status": status,
               "sc_laps": len(sc_laps),
               "total_laps": len(d_laps),
               "flagged": is_dnf_dsq or sc_affected,
               "dnf_dsq": is_dnf_dsq,
               "sc_affected": sc_affected,
            }   

         return driver_data

   except Exception as e:
      print(f" Race load failed for {race} {year}: {e}")
      return {}


def load_constructor_map(year: int, race: str) -> dict:
   """Return {driver: team_name} for a given race weekend."""
   try:
      session = fastf1.get_session(year, race, "R")
      session.load(telemetry=False, weather=False, messages=False)
      results = session.results
      if results is None or results.empty:
         return {}
      return {
         str(row.get("Abbreviation", "")): str(row.get("TeamName", "Unknown"))
         for _, row in results.iterrows()
         if row.get("Abbreviation")
      }
   except Exception:
      return{}


#scoring engine
def compute_conversion_score(quali: dict, race: dict) -> pd.DataFrame:
    """
    Compute conversion scores for all drivers at a single race.
 
    Position component:
      delta = finish_position - quali_position
      negative delta = gained positions (good conversion)
      normalised 0-100 across the field for this race
 
    Pace component:
      quali_time vs driver's median race pace
      a smaller gap (or negative = faster in race) = good conversion
      normalised 0-100 across the field
 
    Composite = 50% position score + 50% pace score
    """
    records = []
    drivers = set(quali.keys()) & set(race.keys())
 
    if not drivers:
        return pd.DataFrame()
 
    for driver in drivers:
        q = quali[driver]
        r = race[driver]
 
        record = {
            "Driver":          driver,
            "QualiTime":       q["QualiTime"],
            "QualiPosition":   q["QualiPosition"],
            "FinishPosition":  r["finish_position"],
            "MedianRacePace":  r["median_pace"],
            "Status":          r["status"],
            "SCLaps":          r["sc_laps"],
            "Flagged":         r["flagged"],
            "DNF_DSQ":         r["dnf_dsq"],
            "SC_Affected":     r["sc_affected"],
        }
 
        # Position delta (negative = gained positions)
        # Always set explicitly so the column always exists in the DataFrame
        record["PositionDelta"] = (
            q["QualiPosition"] - r["finish_position"]
            if r["finish_position"] is not None
            else np.nan
        )
 
        # Pace delta: race pace vs quali time
        # Always set explicitly so the column always exists in the DataFrame
        record["PaceDelta"] = (
            round(r["median_pace"] - q["QualiTime"], 4)
            if r["median_pace"] is not None
            else np.nan
        )
 
        records.append(record)
 
    df = pd.DataFrame(records)
 
    # Ensure score columns always exist to avoid KeyError on .loc assignment
    df["PositionScore"]   = np.nan
    df["PaceScore"]       = np.nan
    df["ConversionScore"] = np.nan
 
    # ── Position score (0-100) ─────────────────────────────────────────────────
    valid_pos = df["PositionDelta"].notna()
    if valid_pos.sum() > 1:
        pos_min = df.loc[valid_pos, "PositionDelta"].min()
        pos_max = df.loc[valid_pos, "PositionDelta"].max()
        spread = pos_max - pos_min if pos_max != pos_min else 1.0
        df.loc[valid_pos, "PositionScore"] = (
            (df.loc[valid_pos, "PositionDelta"] - pos_min) / spread * 100
        ).round(2)
    else:
        df["PositionScore"] = np.nan
 
    # ── Pace score (0-100) ─────────────────────────────────────────────────────
    # Lower PaceDelta = better (closer to or faster than quali pace)
    valid_pace = df["PaceDelta"].notna()
    if valid_pace.sum() > 1:
        pace_min = df.loc[valid_pace, "PaceDelta"].min()
        pace_max = df.loc[valid_pace, "PaceDelta"].max()
        spread = pace_max - pace_min if pace_max != pace_min else 1.0
        df.loc[valid_pace, "PaceScore"] = (
            (pace_max - df.loc[valid_pace, "PaceDelta"]) / spread * 100
        ).round(2)
    else:
        df["PaceScore"] = np.nan
 
    # ── Composite score ────────────────────────────────────────────────────────
    both_valid = df["PositionScore"].notna() & df["PaceScore"].notna()
    pos_only   = df["PositionScore"].notna() & df["PaceScore"].isna()
    pace_only  = df["PaceScore"].notna() & df["PositionScore"].isna()
 
    df.loc[both_valid, "ConversionScore"] = (
        POSITION_WEIGHT * df.loc[both_valid, "PositionScore"] +
        PACE_WEIGHT     * df.loc[both_valid, "PaceScore"]
    ).round(2)
 
    # Partial score if only one component available (flagged race)
    df.loc[pos_only,  "ConversionScore"] = df.loc[pos_only,  "PositionScore"].round(2)
    df.loc[pace_only, "ConversionScore"] = df.loc[pace_only, "PaceScore"].round(2)
    df.loc[~(both_valid | pos_only | pace_only), "ConversionScore"] = np.nan
 
    return df


#season builder
def build_season_scores(year: int, races: list = None) -> pd.DataFrame:
   """
   Compute conversion scores for every race in a season.
   Returns a long-format DataFrame with one row per (driver, race).
   """
   target = races if races is not None else get_calendar(year)
   if not target:
      raise RuntimeError(f"No races found for {year}.")

   all_records = []
   print(f"\nBuilding conversion scores for {year} ({len(target)} races)...")

   for i, race in enumerate(target, 1):
      print(f" [{i:02d}/{len(target)}] {race}...")

      quali = load_qualifying(year, race)
      race_data = load_race_pace(year, race)
      constructor_map = load_constructor_map(year, race)

      if not quali or not race_data:
         print(f" Skipped - insufficient data")
         continue

      df = compute_conversion_score(quali, race_data)
      if df.empty:
         continue

      df["RaceName"] = race
      df["Year"] = year
      df["RaceIndex"] = i
      df["Constructor"] = df["Driver"].map(constructor_map).fillna("Unkown")


      all_records.append(df)
      print(f"{len(df)} drivers scored")

   if not all_records:
      raise RuntimeError("No conversion data collected.")
   
   season = pd.concat(all_records, ignore_index=True)
   print(f"\n Season complete: {len(season)} driver-race records")
   return season


def compute_season_summary(season: pd.DataFrame) -> pd.DataFrame:
   """
   Aggregate per-race scores into a season summary per driver.
   Includes: mean score, percentile rank, trend (slope), consistency.
   """

   records = []

   for driver, grp in season.groupby("Driver"):
      valid = grp[grp["ConversionScore"].notna()].copy()
      if valid.empty:
         continue

      scores = valid["ConversionScore"].values
      indices = valid["RaceIndex"].values

      mean_score = float(np.mean(scores))
      std_score = float(np.std(scores))
      best_score = float(np.max(scores))
      worst_score = float(np.min(scores))
      races_scored = len(scores)
      flagged_races = int(valid["Flagged"].sum())

      #linear regression of score over race index
      if len(scores)>= 3:
         slope, _, _, _, _=stats.linregress(indices, scores)
         trend = round(float(slope),4)
      else:
         trend = 0.0

      trend_label = (
         "Improving" if trend > 0.3 else
         "Declining" if trend < -0.3 else
         "Stable"
      )

      constructor = valid["Constructor"].mode().iloc[0] if not valid["Constructor"].empty else "Unknown"

      records.append({
         "Driver": driver,
         "Constructor": constructor,
         "MeanScore": round(mean_score,2),
         "StdScore": round(std_score, 2),
         "BestScore": round(best_score, 2),
         "WorstScore": round(worst_score,2),
         "RacesScored": races_scored,
         "FlaggedRaces": flagged_races,
         "TrendSlope": trend,
         "TrendLabel": trend_label,
      })

   summary = pd.DataFrame(records).sort_values("MeanScore", ascending=False).reset_index(drop=True)

   #percentile rank (higher score = higher percentile)
   summary["Percentile"] = (
      summary["MeanScore"].rank(pct=True) * 100
   ).round(1)

   return summary


#visualization
def plot_season_overview(summary: pd.DataFrame, year: int, save: bool = True):
   """
   Horizontal bar chart of season mean conversion scores.
   Color-coded by trend direction.
   """
   df = summary.sort_values("MeanScore")

   trend_colors = {
      "Improving": "#39B54A",
      "Stable": "#FFF200",
      "Declining": "#E8002D",
   }
   colors = [trend_colors.get(t, "#888888") for t in df["TrendLabel"]]

   fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.42)))
   fig.suptitle(
      f"Quali -> Race Concersion score - {year} Season",
      fontsize=13, color="#ffffff",
   )

   bars = ax.barh(df["Driver"], df["MeanScore"], color=colors, height=0.65)

   #percentile labels on bars
   for bar, (_, row) in zip(bars, df.iterrows()):
      ax.text(
         bar.get_width() + 0.5,
         bar.get_y() + bar.get_height() / 2,
         f"P{row['Percentile']:.0f} {row['TrendLabel']}",
         va="center", ha="left", fontsize=8, color="#aaaaaa",
      )

   ax.set_xlabel("Mean Conversion Score (0-100)")
   ax.set_xlim(0,115)
   ax.grid(True, axis="x", alpha=0.3)

   #legend
   for label, color in trend_colors.items():
      ax.barh([], [], color=color, label=label)
   ax.legend(loc="lower right", fontsize=9)

   plt.tight_layout()

   if save:
      path = OUTPUT_DIR / f"season_overview_{year}.png"
      plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
      print(f"Saved: {path}")
   plt.show()
   plt.close()


def plot_driver_trend(season: pd.DataFrame, driver: str, year: int, save: bool = True):
   """
   Line chart of a single driver's conversion score across the season,
   with flagged races highlighted.
   """
   df = season[(season["Driver"] == driver) & (season["Year"] == year)].copy()
   df = df.sort_values("RaceIndex")

   if df.empty:
      print(f"No data for {driver} in {year}")
      return

   fig,axes = plt.subplots(2, 1, figsize=(12,7), gridspec_kw={"height_ratios": [3, 1]})
   fig.suptitle(f"{driver} - Conversion Score Trend {year}", fontsize=13, color="#ffffff")

   ax = axes[0]

   #shaded flagged races
   for _, row in df[df["Flagged"]].iterrows():
      ax.axvspan(row["RaceIndex"] - 0.4, row["RaceIndex"] + 0.4,
                 alpha=0.15, color="#E8002D", zorder=0)


   #score line
   valid = df[df["ConversionScore"].notna()]
   ax.plot(valid["RaceIndex"] - 0.4, row["RaceIndex"] + 0.4,
           alpha=0.15, color="#E8002D", linewidth=2, marker="o", markersize=5, zorder=2)


   #trend line 
   if len(valid) >= 3:
      slope, intercept, _, _, _ = stats.linregress(
         valid["RaceIndex"], valid["ConversionScore"]
      )
      x_line = np.array([valid["RaceIndex"].min(), valid["RaceIndex"].max()])
      ax.plot(x_line, slope * x_line + intercept,
              color="#FFF200", linewidth=1.2, linestyle="--", alpha=0.7, label="Trend")

      ax.set_ylabel("Conversion Score")
      ax.set_xticks(df["RaceIndex"])
      ax.set_xticklabels(df["RaceName"], rotation=45, ha="right", fontsize=7)
      ax.set_ylim(0,105)
      ax.axhline(50, color="#555555", linewidth=0.8, linestyle=":")
      ax.grid(True, alpha=0.3)
      ax.legend(fontsize=8)

      #position delta subplot
      ax2 = axes[1]
      valid2 = df[df["PositionDelta"].notna()]
      colors2 = ["#39B54A" if v >= 0 else "#E8002D" for v in valid2["PositionDelta"]]
      ax2.bar(valid2["RaceIndex"], valid2["PositionDelta"], color=colors2, width=0.6)
      ax2.axhline(0, color="#aaaaaa", linewidth=0.8)
      ax2.set_ylabel("Pos. Delta")
      ax2.set_xticks(df["RaceIndex"])
      ax2.set_xticklabels([])
      ax2.grid(True, axis="y", alpha=0.3)

      plt.tight_layout()

      if save:
         path = OUTPUT_DIR / f"trend_{driver}_{year}.png"
         plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
         print(f" Saved: {path}")
      plt.show()
      plt.close()


def plot_score_heatmap(season: pd.DataFrame, year: int, save: bool = True):
   """
   Heatmap of conversion scores: drivers (rows) x races (cols).
   Flagged races shown with a different annotation.
   """
   df = season[season["Year"] == year].copy()

   pivot =df.pivot_table(
      index="Driver", columns="RaceName",
      values="ConversionScore", aggfunc="mean",
   )

   #order races by calendar
   calendar = get_calendar(year)
   race_order = [r for r in calendar if r in pivot.columns]
   pivot = pivot[race_order]

   #order races by calendar
   pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

   fig, ax = plt.subplots(figsize=(max(14, len(race_order) * 0.7), max(7, len(pivot) * 0.45)))
   fig.suptitle(f"Conversion Score Heatmap - {year}", fontsize=13, color="#ffffff")

   sns.heatmap(
      pivot, ax=ax,
      cmap="RdY1Gn",
      vmin=0, vmax=100,
      annot=True, fmt=".0f",
      annot_kws={"size": 7},
      linewidths=0.3,
      linecolor="#0f0f0f",
      cbar_kws={"label": "Conversion Score"},
   )

   ax.set_xlabel("")
   ax.set_ylabel("")
   plt.xticks(rotation=45, ha="right", fontsize=8)
   plt.yticks(fontsize=9)
   plt.tight_layout()

   if save:
      path = OUTPUT_DIR / f"heatmap_{year}.png"
      plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
      print(f"Saved: {path}")
   plt.show()
   plt.close()


def plot_pace_vs_position(season: pd.DataFrame, year: int, save: bool = True):
   """
   Scatter plot: PaceScore vs PositionScore per driver (season averages).
   Reveals drivers who convert via pace vs track position management.
   """
   df = season[season["Year"] == year].copy()
   agg = df.groupby("Driver").agg(
      PaceScore=("PaceScore", "mean"),
      PositionScore=("PositionScore", "mean"),
      ConversionScore=("ConversionScore", "mean"),
   ).reset_index()

   fig, ax = plt.subplots(figsize=(9, 7))
   fig.suptitle(
      f"Pace Score vs Position Score - {year} Season",
      fontsize=13, color="#ffffff",
   )

   scatter = ax.scatter(
      agg["PaceScore"], agg["PositionScore"],
      c=agg["ConversionScore"],
      cmap="RdY1Gn", vmin=0, vmax=100,
      s=80, zorder=3,
   )

   for _, row in agg.iterrows():
      ax.annotate(
         row["Driver"],
         (row["PaceScore"], row["PositionScore"]),
         textcoords="offset points", xytext=(5, 4),
         fontsize=8, color="#cccccc",
      )

   ax.axhline(50, color="#555555", linewidth=0.8, linestyle="--")
   ax.axvline(50, color="#555555", linewidth=0.8, linestyle="--")
   ax.set_xlabel("Pace Score (race pace vs quali pace)")
   ax.set_ylabel("Position Score (positions gained/lost)")
   ax.grid(True, alpha=0.3)
   plt.colorbar(scatter, ax=ax, label="Composite Conversion Score")
   plt.tight_layout()

   if save:
      path = OUTPUT_DIR / f"pace_vs_position_{year}.png"
      plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
      print(f"Saved: {path}")
   plt.show()
   plt.close()


def plot_constructor_comparison(summary: pd.DataFrame, year: int, save: bool = True):
   """
   Box plot comparing conversion score distributions per constructor.
   """
   constructor_avg = (
      summary.groupby("Constructor")["MeanScore"]
      .agg(["mean", "std", "count"])
      .reset_index()
      .rename(columns={"mean": "TeamScore", "std": "TeamStd", "count": "Drivers"})
      .sort_values("TeamScore", ascending=False)                 
   )  

   fig, ax = plt.subplots(figsize=(10,5))
   fig.suptitle(f"Constructor Conversion Score - {year}", fontsize=13, color="#ffffff")

   colors = ["#E8002D" if i == 0 else "#555555" for i in range(len(constructor_avg))]
   bars = ax.bar(
      constructor_avg["Constructor"],
      constructor_avg["TeamScore"],
      color=colors, width=0.6,
      yerr=constructor_avg["TeamStd"],
      capsize=4, error_kw={"ecolor": "#888888", "linewidth": 1},
   )

   ax.set_ylabel("Mean Conversion Score")
   ax.set_ylim(0,110)
   plt.xticks(rotation=30, ha="right", fontsize=9)
   ax.grid(True, axis="y", alpha=0.3)
   plt.tight_layout()

   if save:
      path = OUTPUT_DIR / f"constructor_comparison_{year}.png"
      plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
      print(f" Saved: {path}")
   plt.show()
   plt.close()


#summary print
def print_summary(summary: pd.DataFrame, year: int):
   """Print a clean ranked table of conversion scores to console."""
   print(f"\n{'=' * 65}")
   print(f" QUALI -> RACE CONVERSION SCORES - {year}")
   print(f"{'=' * 65}")
   print(f" {'Rank':<5} {'Driver':<7} {'Constructor':<22} {'Score':<8} {'Pctile':<8} {'Trend':<12} {'Races'}")
   print(f" {'-' * 62}")

   for i, row in summary.iterrows():
      trend_symbol = (
         "Improving" if row["TrendLabel"] == "Improving" else
         "Declining" if row["TrendLabel"] == "Declining" else
         "Stable"
      )
      print(
         f"{i+1:<5} {str(row['Driver']):<7} {str(row['Constructor']):<22}"
         f"{row['MeanScore']:<8.1f} {row['Percentile']:<8.0f}"
         f"{trend_symbol:<12} {row['RacesScored']}"
      )

   print(f"\n Note: Flagged races (DNF/DSQ/SC) scored where data allows.")
   print(f"{'=' * 65}\n")


#main
def main():
   parser = argparse.ArgumentParser(description="F1 Qualifying to Race Conversion Score")
   parser.add_argument(
      "--year", type=int, action="append", dest="years",
      help="Season year (can specify multiple: --year 2024 --year 2025)"
   )
   parser.add_argument("--race", type=str, default=None,
                       help="Limit to a singe race (e.g. Monaco)")
   parser.add_argument("--driver", type=str, default=None,
                       help="Plot trend for a specific driver (e.g. VER)")
   args = parser.parse_args()

   years = args.years if args.years else [2024, 2025]
   races = [args.race] if args.race else None

   all_seasons = []

   for year in years:
      season = build_season_scores(year, races=races)
      season.to_csv(OUTPUT_DIR / f"conversion_scores_{year}.csv", index=False)


      summary = compute_season_summary(season)
      summary.to_csv(OUTPUT_DIR / f"season_summary_{year}.csv", index=False)

      print_summary(summary, year)

      #plots
      plot_season_overview(summary, year)
      plot_score_heatmap(season, year)
      plot_pace_vs_position(season, year)
      plot_constructor_comparison(summary, year)

      #driver trend plot
      if args.driver:
         plot_driver_trend(season, args.driver.upper(), year)
      else:
         #plot top 3 and bottom 1 by default
         for driver in list(summary["Driver"].head(3)) + [summary["Driver"].iloc[-1]]:
            plot_driver_trend(season, driver, year)

      all_seasons.append(season)

   #multi-season combines CSV if both years loaded
   if len(all_seasons) >1:
      combined = pd.concat(all_seasons, ignore_index=True)
      combined.to_csv(OUTPUT_DIR / "conversion_scores_all.csv", index=False)
      print(f"Combined CSV saved: conversion_scores_all.csv")

   print("\n All conversion analysis complete")


if __name__ == "__main__":
   main()                                         
