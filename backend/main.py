"""EnergyWise FastAPI application.

Run locally with: uvicorn backend.main:app --reload
"""

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from energywise.models.gbm_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_matrix,
    prepare_features,
    train_and_forecast_gbm,
)
from energywise.tariffs.bill_calculator import Tariff, calculate_bill
from energywise.optimization.load_shifting import FlexibleLoad, find_cheapest_window

app = FastAPI(
    title="EnergyWise API",
    description="Forecasting and cost-simulation API for small commercial buildings.",
    version="0.1.0",
)

# CORS: restrict to the deployed frontend origin once known.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server default
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

VALID_BUILDINGS = {"academic", "facilities", "lecture", "library"}


@app.get("/health")
def health() -> dict:
    """Service health and readiness status."""
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict:
    """Model version, training range, target unit, horizon, data-source notes."""
    hourly = pd.read_parquet("data/processed/hourly_energy_weather.parquet")
    training_start = hourly["timestamp_ist"].min().isoformat()
    training_end = hourly["timestamp_ist"].max().isoformat()

    return {
        "model_version": "GBM (LightGBM) v1 - best of {naive, SARIMAX, RandomForest, GBM}",
        "training_range": {"start": training_start, "end": training_end},
        "target_unit": "kWh",
        "horizon_hours": 24,
        "evaluation_note": (
            "Evaluated via weekly-step (168h) rolling-origin backtesting "
            "across 4 buildings; overall WAPE improved from 20.5% (naive) "
            "to 8.7% (GBM)."
        ),
    }


@app.get("/history")
def history(building: str, days: int = 7) -> dict:
    """Recent actual hourly energy use for a building, for charting against
    the forecast.
    """
    if building not in VALID_BUILDINGS:
        return {"error": f"Unknown building. Must be one of {sorted(VALID_BUILDINGS)}"}

    hourly = pd.read_parquet("data/processed/hourly_energy_weather.parquet")
    building_df = hourly[hourly["building"] == building].sort_values("timestamp_ist")

    cutoff = building_df["timestamp_ist"].max() - pd.Timedelta(days=days)
    recent = building_df[building_df["timestamp_ist"] > cutoff]

    return {
        "building": building,
        "unit": "kWh",
        "history": [
            {
                "timestamp": row["timestamp_ist"].isoformat(),
                "actual_kwh": None if pd.isna(row["energy_kwh"]) else round(float(row["energy_kwh"]), 3),
            }
            for _, row in recent.iterrows()
        ],
    }


@app.get("/data-status")
def data_status(building: str) -> dict:
    """Data coverage and quality info for a building."""
    if building not in VALID_BUILDINGS:
        return {"error": f"Unknown building. Must be one of {sorted(VALID_BUILDINGS)}"}

    hourly = pd.read_parquet("data/processed/hourly_energy_weather.parquet")
    building_df = hourly[hourly["building"] == building].sort_values("timestamp_ist")

    total_hours = len(building_df)
    missing_hours = int(building_df["energy_kwh"].isna().sum())
    missing_pct = round(100 * missing_hours / total_hours, 2) if total_hours else 0.0

    # Find the largest gap (consecutive missing hours) for a quick warning.
    is_missing = building_df["energy_kwh"].isna().to_numpy()
    longest_gap = 0
    current_gap = 0
    for m in is_missing:
        current_gap = current_gap + 1 if m else 0
        longest_gap = max(longest_gap, current_gap)

    return {
        "building": building,
        "data_range": {
            "start": building_df["timestamp_ist"].min().isoformat(),
            "end": building_df["timestamp_ist"].max().isoformat(),
        },
        "total_hours": total_hours,
        "missing_hours": missing_hours,
        "missing_pct": missing_pct,
        "longest_gap_hours": int(longest_gap),
        "warning": (
            f"{missing_pct}% of hours are missing, with the longest single "
            f"gap spanning {longest_gap} hours. Forecasts near large gaps "
            "may be less reliable."
            if missing_pct > 5
            else None
        ),
    }


class ForecastRequest(BaseModel):
    building: str
    horizon_hours: int = 24


@app.post("/forecast")
def forecast(request: ForecastRequest) -> dict:
    """Forecast energy demand for a building over the next horizon_hours."""
    if request.building not in VALID_BUILDINGS:
        return {"error": f"Unknown building. Must be one of {sorted(VALID_BUILDINGS)}"}
    if not (1 <= request.horizon_hours <= 168):
        return {"error": "horizon_hours must be between 1 and 168"}

    hourly = pd.read_parquet("data/processed/hourly_energy_weather.parquet")
    hourly = prepare_features(hourly)

    # Filter to rows with no NaN in any required column, WITHOUT losing
    # timestamp_ist or scrambling row alignment (build_feature_matrix resets
    # its index, which breaks a timestamp join done after filtering).
    required_cols = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    clean = hourly.dropna(subset=required_cols)

    building_df = clean[clean["building"] == request.building].sort_values("timestamp_ist")

    if len(building_df) < request.horizon_hours + 1:
        return {"error": "Insufficient clean history for this building"}

    train_df = building_df.iloc[:-request.horizon_hours]
    test_df = building_df.iloc[-request.horizon_hours:]

    predictions = train_and_forecast_gbm(train_df, test_df)

    timestamps = test_df["timestamp_ist"].dt.tz_convert("Asia/Kolkata").astype(str).tolist()

    return {
        "building": request.building,
        "horizon_hours": request.horizon_hours,
        "unit": "kWh",
        "forecast": [
            {"timestamp": ts, "predicted_kwh": round(float(p), 3)}
            for ts, p in zip(timestamps, predictions)
        ],
    }


class CostSimulationRequest(BaseModel):
    building: str
    load_name: str = "Flexible Appliance"
    power_kw: float
    duration_hours: int
    allowed_window_start_hour: int = 0
    allowed_window_end_hour: int = 23
    baseline_start_hour: int
    sanctioned_load_kw: float | None = None  # auto-estimated from peak usage if not given
    recent_days: int = 30  # how many days of recent history to use as the baseline profile


@app.post("/cost-simulation")
def cost_simulation(request: CostSimulationRequest) -> dict:
    """Simulate whether shifting a flexible load's schedule changes the
    estimated bill, using recent real usage as the baseline profile.

    This is always a SIMULATION based on stated assumptions, never a claim
    of verified or measured savings.
    """
    if request.building not in VALID_BUILDINGS:
        return {"error": f"Unknown building. Must be one of {sorted(VALID_BUILDINGS)}"}

    hourly = pd.read_parquet("data/processed/hourly_energy_weather.parquet")
    building_df = hourly[hourly["building"] == request.building].sort_values("timestamp_ist")
    building_df = building_df.dropna(subset=["energy_kwh"])

    if building_df.empty:
        return {"error": "No usable data for this building"}

    sanctioned_kw = request.sanctioned_load_kw
    auto_estimated = False
    if sanctioned_kw is None:
        peak_kwh = float(building_df["energy_kwh"].max())
        # Simple heuristic: round the building's historical peak hourly kWh
        # up to the nearest 10, with a 10% margin -- a rough stand-in for a
        # real sanctioned-load figure, which we don't have. Document this
        # as an assumption, not a verified contracted capacity.
        sanctioned_kw = ((peak_kwh * 1.1) // 10 + 1) * 10
        auto_estimated = True

    cutoff = building_df["timestamp_ist"].max() - pd.Timedelta(days=request.recent_days)
    recent = building_df[building_df["timestamp_ist"] > cutoff].copy()
    recent = recent.set_index("timestamp_ist")
    baseline_hourly_kwh = recent["energy_kwh"]

    tariff = Tariff()  # default approximate flat tariff; see tariff source_note
    load = FlexibleLoad(
        name=request.load_name,
        power_kw=request.power_kw,
        duration_hours=request.duration_hours,
        allowed_window_start_hour=request.allowed_window_start_hour,
        allowed_window_end_hour=request.allowed_window_end_hour,
        baseline_start_hour=request.baseline_start_hour,
    )

    result = find_cheapest_window(load, tariff, baseline_hourly_kwh, sanctioned_kw)
    result["sanctioned_load_kw_used"] = sanctioned_kw
    result["sanctioned_load_auto_estimated"] = auto_estimated
    result["tariff_source_note"] = tariff.source_note
    result["baseline_period"] = {
        "start": recent.index.min().isoformat() if not recent.empty else None,
        "end": recent.index.max().isoformat() if not recent.empty else None,
        "days_used": request.recent_days,
    }
    return result


@app.get("/metrics")
def metrics() -> dict:
    """Approved backtest metrics for display. From the weekly-step
    rolling-origin backtest comparing naive, Random Forest, and GBM.
    """
    return {
        "evaluation_method": "weekly-step (168h) rolling-origin backtest",
        "production_model": "GBM (LightGBM)",
        "results_by_building": {
            "academic":   {"naive_wape": 18.77, "rf_wape": 8.61, "gbm_wape": 7.84},
            "facilities": {"naive_wape": 13.46, "rf_wape": 7.20, "gbm_wape": 6.70},
            "lecture":    {"naive_wape": 141.55, "rf_wape": 57.58, "gbm_wape": 47.90},
            "library":    {"naive_wape": 31.97, "rf_wape": 14.94, "gbm_wape": 13.02},
        },
        "overall": {"naive_wape": 20.46, "rf_wape": 9.61, "gbm_wape": 8.67},
        "note": (
            "Metrics from an offline weekly-step backtest, not computed "
            "live. See PROJECT_STATUS.md decisions log for the full "
            "daily-step Random Forest comparison and the error-analysis "
            "findings."
        ),
    }