# EnergyWise

Hourly electricity-demand forecasting, plus a tariff-aware cost simulator, for small commercial buildings, built end to end: data pipeline, leakage-safe features, model comparison, API and dashboard.

> **Honest scope.** This is a portfolio project built on a public dataset of four academic-campus buildings in Delhi (2013-2017), not on retail or office customers. The tariff used for cost estimates is an unverified approximation. See [Limitations](#limitations) before drawing conclusions from any number here.

## What it does

- Turns 1-minute smart-meter readings into a clean hourly series, with explicit handling of missing data (a missing hour is never treated as zero consumption).
- Builds leakage-safe features (calendar, lags, rolling statistics, weather) and compares a seasonal-naive baseline, SARIMAX, Random Forest and LightGBM using chronological, rolling-origin backtests.
- Serves forecasts, history, data-quality information and cost simulations through a FastAPI service.
- Shows everything in a React dashboard: KPI cards, actual-vs-forecast chart, usage heatmap, model-quality table and data-status warnings.

## Results

Rolling-origin backtest, 24-hour forecast horizon, origin advanced weekly (168 h). **WAPE (%), lower is better.**

| Building   | Seasonal-naive | Random Forest | LightGBM (production) |
|------------|---------------:|--------------:|----------------------:|
| Academic   | 18.8 | 8.6  | **7.8**  |
| Facilities | 13.5 | 7.2  | **6.7**  |
| Lecture    | 141.6 | 57.6 | **47.9** |
| Library    | 32.0 | 14.9 | **13.0** |
| **All**    | 20.5 | 9.6  | **8.7**  |

Across all buildings: MAE 2.15 -> 1.01 -> **0.91 kWh**, and peak-period MAE 1.85 -> 0.81 -> **0.72 kWh** (naive -> Random Forest -> LightGBM).

Notes on reading these numbers:

- LightGBM beat both alternatives on every building and every metric, but its margin over Random Forest is modest (roughly 0.5-1.8 WAPE points). Its hyper-parameters are reasonable defaults, not tuned.
- **Lecture is the hardest building**: its load is small (mean about 0.64 kWh/h) with long zero-usage stretches, so percentage errors are large even when absolute errors are small.
- An earlier full daily-step run (Random Forest vs. naive only) gave overall WAPE 18.5% -> 9.0%. Absolute WAPE values shift between daily-step and weekly-step evaluation (for example Lecture's naive WAPE is 69% daily-step and 142% weekly-step), but the model ranking was the same. The three-way comparison above uses weekly steps to keep run time practical.
- **SARIMAX lost to the naive baseline** (Academic only, weekly-step: WAPE 25.4% vs 18.8%) with untuned default orders, and took about two hours for one building, so it was not extended to the other buildings.

### What the error analysis found

- The largest Academic errors cluster on specific Saturdays in early October of 2013, 2015 and 2017, most plausibly festival-season closures or special campus events that the simple fixed holiday list does not capture.
- The model over-predicts more often than it under-predicts (70-84% of errors for Academic, Lecture and Library; closer to balanced for Facilities).
- Lecture's three largest spikes (14 Feb 2015, 25 Mar 2017, 8 Apr 2017) are coherent multi-hour evening events, not sensor glitches. They are one-off, so no model learns them.

## How it works

```
 raw 1-min CSVs --> combined raw parquet --> validation report
        |
        v
 hourly kWh (>=45 of 60 minutes present, else missing) --+
                                                         +--> hourly_energy_weather.parquet
 Open-Meteo ERA5 weather (IST, hourly) ------------------+            |
                                                                      v
                          time + lag + rolling + weather features (backward-looking only)
                                                                      |
                          seasonal-naive / SARIMAX / Random Forest / LightGBM
                                                                      |
                          rolling-origin backtest -> MAE, WAPE, RMSE, peak MAE
                                                                      |
              FastAPI  <----------------------------------------------+
                 |
            React dashboard
```

Key design decisions:

- **Units and time.** Source `power` is in watts; hourly energy is `mean(W) / 1000` kWh. Timestamps are converted from Unix/UTC to Asia/Kolkata before any hour-of-day logic.
- **Missing is not zero.** An hour is valid only if at least 45 of its 60 minutes are present; otherwise it is `NaN` and flagged. Nothing is interpolated.
- **No leakage.** Lags and rolling statistics are computed per building from past values only (shift first, then roll). A test corrupts the future and asserts that earlier features do not change. Preprocessing is fit on the training window only.
- **Cost simulation is separate from forecasting.** Tariff and optimiser code never imports model code.

## Repository layout

```
backend/                FastAPI app (main.py)
frontend/               React + TypeScript + Vite + Tailwind dashboard
src/energywise/
  ingestion/            load_raw, clean_hourly, fetch_weather
  validation/           data-quality report
  features/             time and lag/rolling features
  models/               baseline, SARIMAX, Random Forest, LightGBM
  evaluation/           backtest harness, metrics, error analysis
  tariffs/              flat-tariff bill calculator
  optimization/         load-shifting scenario logic
scripts/                runnable backtest scripts
notebooks/              01_eda.ipynb (exploratory analysis)
tests/                  pytest suite
data/                   not committed (see data/README.md)
```

## Quick start

Examples use PowerShell. On macOS/Linux, replace `$env:PYTHONPATH="src"` with `export PYTHONPATH=src`.

### 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the data

The dataset is not included in this repository. Download I-BLEND from Figshare and unzip it so the CSVs sit in `data/raw/energy_dataset/`:

<https://springernature.figshare.com/articles/dataset/Energy_dataset_of_IIITD/6007637>

See `data/README.md` for the expected layout and the source registry. Check the dataset's own license terms before redistributing any of it.

### 3. Run the pipeline

```powershell
$env:PYTHONPATH="src"
python -m energywise.ingestion.load_raw        # -> data/raw/combined_raw.parquet
python -m energywise.validation.report         # -> data/quality_report.md
python -m energywise.ingestion.clean_hourly    # -> data/processed/hourly_energy.parquet
python -m energywise.ingestion.fetch_weather   # downloads weather, joins it
                                               # -> data/processed/hourly_energy_weather.parquet
```

Weather comes from Open-Meteo's historical API (no key required) and needs an internet connection.

### 4. Reproduce the model comparison

```powershell
python scripts/run_gbm_backtest.py             # naive vs Random Forest vs LightGBM (weekly step, roughly 10 minutes)
python scripts/run_random_forest_backtest.py   # daily-step Random Forest vs naive (long)
```

Results are written to `data/`. Open `notebooks/01_eda.ipynb` for the exploratory analysis.

### 5. Run the API and dashboard

```powershell
$env:PYTHONPATH="src"
uvicorn backend.main:app --reload              # http://localhost:8000/docs
```

```powershell
cd frontend
npm install
npm run dev                                    # http://localhost:5173
```

### 6. Docker

```powershell
docker compose up --build
```

Dashboard at <http://localhost:5173>, API docs at <http://localhost:8000/docs>. The `data/` and `models/` folders are mounted into the backend container rather than baked into the image, so run the pipeline first. The frontend container runs Vite's development server, which is fine for a demo; a production deployment would serve a static build instead. The first build downloads several hundred MB of Python packages.

### 7. Tests

```powershell
$env:PYTHONPATH="src"
pytest
```

96 tests passed at the time of writing. Some tests read the processed data files, so run the pipeline first.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status |
| `GET /metadata` | Model version, training range, evaluation note |
| `GET /history?building=&days=` | Recent actual hourly kWh |
| `GET /data-status?building=` | Coverage, missing hours, longest gap, warning |
| `POST /forecast` | 24-hour (up to 168 h) forecast for a building |
| `POST /cost-simulation` | Load-shifting scenario under the configured tariff |
| `GET /metrics` | Backtest results (stored values from the offline runs, not computed live) |

## Limitations

- **Dataset.** Four buildings on one academic campus (Academic, Facilities, Lecture, Library), 10 Aug 2013 to 31 Dec 2017. Not small retail or office customers; nothing here shows that the results generalise to them.
- **Missing data.** The share of missing hours is about 1.9% for Academic and Lecture, 5.9% for Facilities and 25.9% for Library, whose longest gap is about 204 days. Library results rest on far less data.
- **Weather is realised, not forecast.** Historical weather comes from ERA5 reanalysis. A real deployment would feed in forecast weather with its own errors, so these backtests are somewhat optimistic.
- **`/forecast` is a held-out demonstration.** The data ends on 31 Dec 2017, so the endpoint trains on everything except the last `horizon_hours` and predicts that held-out window. It does not predict real future time.
- **Weekly-step evaluation.** The headline comparison advances the forecast origin weekly, not daily.
- **Holiday feature is basic.** A fixed list of Indian public holidays, flagged as incomplete in the code; it misses multi-day festival closures and campus events.
- **Tariff is approximate and unverified.** Defaults (about Rs 12.5/kWh effective energy rate including an estimated PPAC surcharge, Rs 200/kW/month fixed, 5% duty) come from secondary sources, not a DERC tariff order. Verify before making any cost claim.
- **Cost simulator.** Because the tariff is flat (no time-of-day component), shifting when energy is used cannot change the bill, and the simulator reports "no savings opportunity" rather than inventing one. All simulator output is labelled as simulated, never measured. The sanctioned load, when not provided, is estimated from the building's historical peak and is not a contracted figure.
- **Model tuning.** LightGBM and Random Forest hyper-parameters are untuned starting values. SARIMAX was only evaluated on one building.

## Data sources and attribution

- **Energy data:** Rashid, H., Singh, P., & Singh, A. (2019). *I-BLEND, a campus-scale commercial and residential buildings electrical energy dataset.* Scientific Data, 6, 190015. <https://doi.org/10.1038/sdata.2019.15>
- **Weather:** [Open-Meteo](https://open-meteo.com/) historical weather API (ERA5 reanalysis). See Open-Meteo's terms for attribution requirements.

## Project status

All build phases through the dashboard and Docker setup are complete. See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the phase checklist and the full decisions log, including the bugs found and fixed along the way.