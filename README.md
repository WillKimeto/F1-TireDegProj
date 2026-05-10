# Beyond the Pitlane — F1 Analytics Platform

A full-stack F1 analytics platform built on real race data from the FastF1 library. Three analytical modules, one unified web app, a FastAPI backend, and a machine learning model trained on 2024 and 2025 race weekends.

Built to learn how to design and ship a real ML pipeline end to end — feature engineering, model training, API design, and deployment — using Formula 1 as the domain.

**Live demo:** https://beyond-the-pitlane.onrender.com

---

## What it does

### Tire Degradation Analyzer
Loads every lap from a race weekend and fits a linear regression to each driver's stint data. Outputs degradation rates in seconds per lap, a median degradation curve per compound, driver vs driver head-to-head comparisons, and a season heatmap showing which drivers consistently manage their tyres and which ones don't.

### Race Winner Predictor
A two-layer prediction model. The statistical baseline ranks drivers by their weighted pace delta across all available sessions — qualifying weighted at 65%, FP3 at 20%, FP2 at 10%, FP1 at 5%. When sessions are missing (rain, cancellations), the weights redistribute automatically across whatever data exists. On top of that sits an XGBoost ranker trained on completed 2024 and 2025 race weekends, validated with leave-one-race-out cross-validation. The final output is an ensemble of both layers with a softmax probability distribution across the full field.

### Quali → Race Pace Conversion Score
Measures how well each driver converts qualifying pace into race performance. The score is a 50/50 composite of a position component (qualifying position vs finish position) and a pace component (qualifying lap time vs median race pace). DNFs and safety car periods are flagged and scored where lap data allows rather than being thrown out entirely. Each driver gets a per-race score, a season average, a percentile rank, and a trend label across the season.

---

## Project structure

```
F1-TireDegProj/
├── api/
│   ├── __init__.py
│   ├── main.py                   — FastAPI app entry point
│   ├── schemas.py                — Pydantic response models
│   └── routers/
│       ├── __init__.py
│       ├── tire.py               — Tire degradation endpoints
│       ├── predictor.py          — Race predictor endpoints
│       └── conversion.py         — Pace conversion endpoints
├── tiredegradationanalyzer.py    — Tire analysis pipeline
├── racewinnerpredictor.py        — Prediction model and feature engineering
├── qualiraceconversion.py        — Conversion score pipeline
├── f1_app.html                   — Frontend (single HTML file)
├── render.yaml                   — Render deployment config
├── requirements.txt
└── f1_cache/                     — FastF1 cache (auto-created, not committed)
```

---

## Requirements

- Python 3.10 or higher
- Git

All Python dependencies are in `requirements.txt`. The main ones are:

| Library | Purpose |
|---|---|
| `fastf1` | F1 session data (laps, timing, results) |
| `fastapi` | API framework |
| `uvicorn` | ASGI server |
| `pandas` / `numpy` | Data processing |
| `scipy` | Linear regression for degradation rates |
| `xgboost` | ML ranker for race prediction |
| `scikit-learn` | Cross-validation and feature scaling |
| `matplotlib` / `seaborn` | Plotting (scripts only, not the web app) |

---

## Running locally — step by step

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Step 2 — Create and activate a virtual environment

**macOS / Linux:**
```bash
python -m venv f1_env
source f1_env/bin/activate
```

**Windows:**
```bash
python -m venv f1_env
f1_env\Scripts\activate
```

You should see `(f1_env)` at the start of your terminal prompt once it's active.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This will install all required libraries including FastF1, FastAPI, XGBoost, and everything else. It may take a few minutes on first run.

### Step 4 — Start the API server

```bash
uvicorn api.main:app --reload
```

You should see this output:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```

The `--reload` flag automatically restarts the server whenever you save a file — useful during development.

### Step 5 — Open the app

Open your browser and go to:

```
http://localhost:8000
```

The full web app will load. You can also visit `http://localhost:8000/docs` for the auto-generated API documentation where you can test every endpoint directly in the browser.

### Step 6 — Select a season and race

Use the **Season** and **Grand Prix** dropdowns in the sidebar. The calendar loads dynamically from FastF1. On first run, FastF1 will download session data from the internet and cache it locally in `f1_cache/`. Subsequent runs for the same race will load from cache and be much faster.

### Step 7 — Train the ML model (optional)

The Race Predictor works without a trained model using the statistical baseline. To enable the XGBoost layer, click the **model training icon** (top right of the header) or visit:

```
http://localhost:8000/predict/train
```

This triggers training on 2024 and 2025 race data in the background. It will take 20–40 minutes on first run depending on how much data is already cached. Progress is logged to your terminal. Once complete, the model is saved to `predictor_outputs/xgb_ranker.pkl` and loaded automatically on future predictions.

---

## Using the three modules

### Tire Degradation Analyzer
1. Select a season and Grand Prix from the sidebar
2. Choose a compound (Soft / Medium / Hard)
3. Select Driver 1 and Driver 2 for head-to-head comparison
4. Click **Analyze**
5. Switch between the four sub-tabs: Curves, Ranking, Driver vs Driver, Heatmap

### Race Winner Predictor
1. Select a season and Grand Prix
2. Choose a model type — Ensemble is recommended (falls back to statistical if no trained model exists)
3. Click **Run Prediction**
4. The podium, driver probability chart, constructor chart, and session pace deltas all populate from the same API call

### Pace Conversion
1. Select a season from the sidebar
2. Click **Analyze** — this loads the full season in one call
3. Use the Sort By dropdown to reorder by score, pace, position, or trend
4. Switch between Overview, Pace vs Position scatter, Driver Trend, and Rankings Table

---

## API endpoints

The full interactive documentation is at `http://localhost:8000/docs`. Quick reference:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server status check |
| GET | `/calendar/{year}` | Race calendar for a given season |
| GET | `/tire/analyze` | Tire degradation analysis for one race and compound |
| GET | `/tire/dvd` | Driver vs driver comparison |
| GET | `/tire/heatmap` | Season degradation heatmap |
| GET | `/predict/race` | Race winner prediction |
| GET | `/predict/train` | Trigger ML model training (background) |
| GET | `/conversion/season` | Full season conversion scores |
| GET | `/conversion/race` | Conversion scores for a single race |

---

## Deployment on Render

The repo includes a `render.yaml` that configures automatic deployment. To deploy your own instance:

1. Fork or clone this repository to your GitHub account
2. Create a new **Web Service** on [render.com](https://render.com) and connect your GitHub repo
3. Render will detect `render.yaml` automatically and configure the build and start commands
4. Once deployed, open `f1_app.html` and update the API URL:

```javascript
// In f1_app.html, find this line and replace the URL with your Render service URL
const API = window.location.origin.includes('localhost')
    ? 'http://localhost:8000'
    : 'https://your-service-name.onrender.com';
```

5. Commit and push — Render will redeploy automatically

**Note on the free tier:** Render's free tier spins down after 15 minutes of inactivity. The first request after a period of inactivity will take 30–60 seconds to respond while the service wakes up. FastF1 data is also re-downloaded on each deploy since the filesystem is ephemeral on the free tier.

---

## Known limitations

**Practice session availability** — FP2 and FP3 data is not always available in FastF1 for every race, particularly earlier in a season. The predictor handles this by reweighting available sessions automatically, but predictions based on only qualifying and FP1 are less accurate than full-weekend predictions.

**ML model and regulation changes** — the XGBoost model is trained on 2024 and 2025 data. If regulations change significantly (as they did for 2026), the model's learned associations from previous seasons may not reflect the new competitive order. Retrain on more recent data as the season progresses by updating the training years in `racewinnerpredictor.py`:

```python
df = build_training_data(years=[2025, 2026])
```

**Cold starts on Render** — the free tier service sleeps after inactivity. Expect a slow first load.

---

## Tech stack

**Backend:** Python, FastAPI, uvicorn, Pydantic  
**Data:** FastF1, pandas, numpy, scipy  
**ML:** XGBoost, scikit-learn  
**Visualisation:** Chart.js (web app), matplotlib, seaborn (standalone scripts)  
**Frontend:** HTML, CSS, Tailwind CSS, vanilla JavaScript  
**Deployment:** Render

---

## License

MIT
