"""Expanding-window (rolling-origin) backtesting for the EnergyWise series.

The harness replays history the way it would have happened in production: it
chooses an initial training cutoff, forecasts the *next* 24 hours from that
cutoff, then slides the cutoff forward 24 hours and forecasts again, repeating
until the end of the series.

Guarantees
----------
* **Per building.**  Each building is backtested on its own, on its own hourly
  grid; one building's history is never mixed into another's.
* **Never shuffled.**  Rows are used strictly in chronological order.  The
  training window is ``[start, cutoff)`` and the test window is
  ``[cutoff, cutoff + horizon)`` -- so a fold's test hours can never appear in
  that same fold's training data.  (This is asserted internally and is recorded
  in the fold table as ``train_end < test_start``.)
* **Expanding window.**  Each successive fold's training set grows; earlier
  hours are never dropped.
* **All-NaN blocks are skipped.**  A 24-hour test block whose *true* values are
  entirely NaN is not scored (there is nothing to compare against) and does not
  advance the reported metrics.
* **NaNs are excluded, never fabricated.**  Inside a scored block, any hour
  whose actual *or* forecast is NaN is dropped from the metrics pairwise; it is
  never replaced with 0.
* **Stable peak definition.**  Each building's "peak" hours (the top 25% of
  hours by ``energy_kwh``) are derived **once from the full series**, not per
  fold, and every prediction row is tagged with ``is_peak`` so
  :func:`~energywise.evaluation.metrics.peak_period_mae` can be reported
  alongside the overall metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from energywise.evaluation import metrics
from energywise.evaluation.metrics import peak_hour_mask, peak_period_mae
from energywise.models.baseline import (
    DEFAULT_HORIZON,
    DEFAULT_SEASON,
    seasonal_naive_forecast,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Fold",
    "BacktestResult",
    "rolling_origin_backtest",
    "summarise",
    "run_baseline_backtest",
    "run_sarimax_backtest",
    "run_academic_comparison",
    "format_comparison",
    "INPUT_PATH",
]

#: Repository root: .../energywise (this file is src/energywise/evaluation/backtest.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Default input: the hourly energy + weather table from Phase 2.
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_energy_weather.parquet"

#: Columns the input frame must provide.
REQUIRED_COLUMNS = ("building", "timestamp_ist", "energy_kwh")

#: Default number of history hours used for the very first fold's training set.
DEFAULT_INITIAL_TRAIN_SIZE = 168 * 4  # four weeks of warm-up

#: Default origin advance between folds (equals the forecast horizon).
DEFAULT_STEP = DEFAULT_HORIZON


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """Metadata for a single rolling-origin fold of one building."""

    building: str
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # last hour *included* in training
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # last hour *included* in the test window
    n_train: int
    n_test: int
    scored: bool
    skipped_reason: str = ""


@dataclass
class BacktestResult:
    """The output of a rolling-origin backtest run.

    Attributes
    ----------
    predictions:
        Long frame with one row per scored forecast hour and the columns
        ``building``, ``timestamp_ist``, ``actual``, ``forecast`` and ``fold``.
    folds:
        One row per candidate fold (scored or skipped) with the date ranges
        actually used, so leakage can be asserted against.
    """

    predictions: pd.DataFrame
    folds: pd.DataFrame
    horizon: int = DEFAULT_HORIZON
    season: int = DEFAULT_SEASON
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE
    step: int = DEFAULT_STEP
    candidate_hours: int = 0
    skipped_hours: int = 0
    excluded_nan_hours: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hourly_grid(
    group: pd.DataFrame,
    time_col: str = "timestamp_ist",
    value_col: str = "energy_kwh",
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Return a complete, gap-free hourly index and its aligned value array.

    The building's series is reindexed onto a continuous ``1h`` grid spanning
    its full range.  Hours that are absent from the input become NaN, so
    positional offsets (e.g. "168 hours earlier") always mean 168 real hours.
    """
    ordered = group.sort_values(time_col, kind="mergesort")
    timestamps = pd.DatetimeIndex(ordered[time_col])
    values = ordered[value_col].to_numpy(dtype=float)

    if len(timestamps) == 0:
        return timestamps, values

    full_index = pd.date_range(timestamps.min(), timestamps.max(), freq="1h")
    aligned = pd.Series(values, index=timestamps).reindex(full_index)
    return full_index, aligned.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------


def rolling_origin_backtest(
    hourly: pd.DataFrame,
    forecast_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
    horizon: int = DEFAULT_HORIZON,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    step: int = DEFAULT_STEP,
    season: int = DEFAULT_SEASON,
    initial_cutoff: pd.Timestamp | None = None,
    buildings: str | list[str] | tuple[str, ...] | None = None,
) -> BacktestResult:
    """Run an expanding-window rolling-origin backtest, per building.

    Parameters
    ----------
    hourly:
        Hourly frame containing ``building``, ``timestamp_ist`` and
        ``energy_kwh``.  May contain NaNs for missing hours.
    forecast_fn:
        ``forecast_fn(history, horizon) -> array``.  ``history`` is the
        building's value array strictly before the fold's cutoff.  Defaults to
        the seasonal-naive baseline.
    horizon:
        Forecast length per fold, in hours (default 24).
    initial_train_size:
        History length, in hours, used for the first fold's training set.
        Ignored when ``initial_cutoff`` is given.
    step:
        How far the origin advances between folds, in hours (default 24).
    season:
        Seasonal lag forwarded to the default seasonal-naive baseline.
    initial_cutoff:
        Optional explicit timestamp for the first fold's cutoff.  History ends
        just before this timestamp.
    buildings:
        Optional building filter.  Accepts a single building name or an
        iterable of names; only those buildings are backtested.  ``None``
        (default) backtests every building present in ``hourly``.  Seldom-used
        buildings can be pinned with this so a run stays fast.

    Returns
    -------
    BacktestResult
        Predictions, fold metadata and NaN-exclusion counts.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon}")
    if step <= 0:
        raise ValueError(f"step must be a positive integer, got {step}")

    missing = [c for c in REQUIRED_COLUMNS if c not in hourly.columns]
    if missing:
        raise ValueError(f"Input frame is missing required column(s): {missing}")

    if forecast_fn is None:

        def forecast_fn(history: np.ndarray, h: int) -> np.ndarray:
            return seasonal_naive_forecast(history, horizon=h, season=season)

    if buildings is None:
        selected_buildings = sorted(hourly["building"].unique())
    elif isinstance(buildings, str):
        selected_buildings = [buildings]
    else:
        selected_buildings = list(buildings)

    prediction_rows: list[dict] = []
    fold_rows: list[dict] = []
    candidate_hours = 0
    skipped_hours = 0

    for building in selected_buildings:
        group = hourly[hourly["building"] == building]
        timestamps, values = _hourly_grid(group)
        n = len(timestamps)
        if n == 0:
            continue

        # Peak hours are derived once from the building's *full* series (not per
        # fold) so "peak" is a stable, building-level concept across folds.
        is_peak = peak_hour_mask(values)

        # Where does the first fold's training set end?
        if initial_cutoff is not None:
            start_pos = int(timestamps.searchsorted(pd.Timestamp(initial_cutoff)))
        else:
            start_pos = int(initial_train_size)

        fold_index = 0
        cutoff = start_pos
        while cutoff + horizon <= n:
            test_start_pos = cutoff
            test_end_pos = cutoff + horizon  # exclusive

            train_slice = slice(0, cutoff)  # strictly before the test window
            test_slice = slice(test_start_pos, test_end_pos)

            train_start_ts = timestamps[0]
            train_end_ts = timestamps[cutoff - 1] if cutoff > 0 else timestamps[0]
            test_start_ts = timestamps[test_start_pos]
            test_end_ts = timestamps[test_end_pos - 1]

            # Invariant: no test hour may be inside this fold's training set.
            assert train_end_ts < test_start_ts, (
                "fold training window overlaps its test window: "
                f"{train_end_ts} !< {test_start_ts}"
            )

            actual = values[test_slice]
            candidate_hours += horizon

            # Skip a block whose true values are *entirely* missing.
            if np.isnan(actual).all():
                skipped_hours += horizon
                fold_rows.append(
                    _fold_row(
                        building, fold_index, train_start_ts, train_end_ts,
                        test_start_ts, test_end_ts, cutoff, horizon, False,
                        "all_actual_nan",
                    )
                )
                fold_index += 1
                cutoff += step
                continue

            history = values[train_slice]
            forecast = np.asarray(forecast_fn(history, horizon), dtype=float)
            if forecast.shape[0] != horizon:
                raise ValueError(
                    "forecast_fn returned "
                    f"{forecast.shape[0]} values, expected {horizon}"
                )

            for offset in range(horizon):
                position = test_start_pos + offset
                prediction_rows.append(
                    {
                        "building": building,
                        "timestamp_ist": timestamps[position],
                        "actual": float(actual[offset]),
                        "forecast": float(forecast[offset]),
                        "fold": fold_index,
                        "is_peak": bool(is_peak[position]),
                    }
                )

            fold_rows.append(
                _fold_row(
                    building, fold_index, train_start_ts, train_end_ts,
                    test_start_ts, test_end_ts, cutoff, horizon, True, "",
                )
            )

            fold_index += 1
            cutoff += step

    predictions = pd.DataFrame(
        prediction_rows,
        columns=[
            "building", "timestamp_ist", "actual", "forecast", "fold", "is_peak",
        ],
    )
    folds = pd.DataFrame(
        fold_rows,
        columns=[
            "building", "fold", "train_start", "train_end", "test_start",
            "test_end", "n_train", "n_test", "scored", "skipped_reason",
        ],
    )

    # Hours dropped from scoring because actual or forecast is NaN (pairwise).
    if len(predictions) == 0:
        excluded_nan_hours = skipped_hours
    else:
        both_present = predictions["actual"].notna() & predictions["forecast"].notna()
        excluded_nan_hours = skipped_hours + int((~both_present).sum())

    result = BacktestResult(
        predictions=predictions,
        folds=folds,
        horizon=horizon,
        season=season,
        initial_train_size=initial_train_size,
        step=step,
        candidate_hours=candidate_hours,
        skipped_hours=skipped_hours,
        excluded_nan_hours=excluded_nan_hours,
    )
    logger.info(
        "Backtest complete: %d prediction row(s), %d fold(s), "
        "%d excluded hour(s) due to NaNs",
        len(predictions),
        len(folds),
        excluded_nan_hours,
    )
    return result


def _fold_row(
    building: str,
    fold_index: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    n_train: int,
    n_test: int,
    scored: bool,
    skipped_reason: str,
) -> dict:
    """Build a single row of the fold-metadata table."""
    return {
        "building": building,
        "fold": fold_index,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "n_train": n_train,
        "n_test": n_test,
        "scored": scored,
        "skipped_reason": skipped_reason,
    }


# ---------------------------------------------------------------------------
# Scoring / reporting
# ---------------------------------------------------------------------------


def summarise(result: BacktestResult) -> pd.DataFrame:
    """Compute per-building MAE, WAPE and RMSE plus NaN-exclusion counts.

    Returns a frame with one row per building (plus a final ``ALL`` row) and
    the columns ``mae``, ``wape``, ``rmse``, ``n_scored``, ``n_excluded`` and
    ``n_candidate``.  Metrics use pairwise NaN exclusion (see
    :mod:`energywise.evaluation.metrics`).
    """
    rows: list[dict] = []
    preds = result.predictions
    folds = result.folds

    def _metrics_for(subset: pd.DataFrame) -> tuple[float, float, float, float, int]:
        if len(subset) == 0:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        a = subset["actual"]
        f = subset["forecast"]
        peak = subset["is_peak"] if "is_peak" in subset.columns else False
        both = a.notna() & f.notna()
        return (
            metrics.mae(a, f),
            metrics.wape(a, f),
            metrics.rmse(a, f),
            peak_period_mae(a, f, peak),
            int(both.sum()),
        )

    # Candidate hours per building == every folded hour (scored or skipped).
    per_building_candidate = {
        building: int(len(group) * result.horizon)
        for building, group in folds.groupby("building")
    } if len(folds) else {}

    buildings = sorted(preds["building"].unique()) if len(preds) else []
    for building in buildings:
        subset = preds[preds["building"] == building]
        mae_v, wape_v, rmse_v, peak_v, n_scored = _metrics_for(subset)
        n_candidate = per_building_candidate.get(building, n_scored)
        rows.append(
            {
                "building": building,
                "mae": mae_v,
                "wape": wape_v,
                "rmse": rmse_v,
                "peak_mae": peak_v,
                "n_scored": n_scored,
                "n_candidate": n_candidate,
                "n_excluded": max(n_candidate - n_scored, 0),
            }
        )

    if len(preds):
        mae_v, wape_v, rmse_v, peak_v, n_scored = _metrics_for(preds)
        rows.append(
            {
                "building": "ALL",
                "mae": mae_v,
                "wape": wape_v,
                "rmse": rmse_v,
                "peak_mae": peak_v,
                "n_scored": n_scored,
                "n_candidate": int(result.candidate_hours),
                "n_excluded": int(result.excluded_nan_hours),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "building", "mae", "wape", "rmse", "peak_mae",
            "n_scored", "n_candidate", "n_excluded",
        ],
    )


#: The building the SARIMAX comparison is pinned to by default.
ACADEMIC_BUILDING = "academic"


def run_baseline_backtest(
    input_path: Path | str = INPUT_PATH,
    **kwargs,
) -> BacktestResult:
    """Load the hourly frame and run the seasonal-naive baseline through it."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Hourly frame not found: {input_path}")
    hourly = pd.read_parquet(input_path)
    return rolling_origin_backtest(hourly, **kwargs)


def run_sarimax_backtest(
    input_path: Path | str = INPUT_PATH,
    buildings: str | list[str] | tuple[str, ...] = ACADEMIC_BUILDING,
    **kwargs,
) -> BacktestResult:
    """Load the hourly frame and run the SARIMAX model through the harness.

    SARIMAX is *much* slower to fit than the seasonal-naive baseline, so this
    defaults to the academic building only.  Everything else -- the expanding
    window, the per-building isolation, the NaN handling -- is identical to
    :func:`run_baseline_backtest`; only the ``forecast_fn`` differs.
    """
    from energywise.models.sarimax_model import sarimax_forecast_fn

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Hourly frame not found: {input_path}")
    hourly = pd.read_parquet(input_path)
    kwargs.setdefault("buildings", buildings)
    return rolling_origin_backtest(hourly, forecast_fn=sarimax_forecast_fn(), **kwargs)


def run_academic_comparison(
    input_path: Path | str = INPUT_PATH,
    building: str = ACADEMIC_BUILDING,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Backtest SARIMAX and the seasonal-naive baseline on one building.

    Returns a mapping ``{"sarimax": summary, "seasonal_naive": summary}`` where
    each summary is the per-building table produced by :func:`summarise`,
    restricted to *building* (plus its ``ALL`` row, which for a single-building
    filter is the same set of hours).
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Hourly frame not found: {input_path}")
    hourly = pd.read_parquet(input_path)

    seasonal = rolling_origin_backtest(hourly, buildings=building, **kwargs)
    sarimax = run_sarimax_backtest(input_path, buildings=building, **kwargs)

    return {
        "sarimax": summarise(sarimax),
        "seasonal_naive": summarise(seasonal),
    }


def format_comparison(
    comparison: dict[str, pd.DataFrame],
    building: str = ACADEMIC_BUILDING,
) -> str:
    """Render SARIMAX vs the seasonal-naive baseline side by side.

    Produces one row per model with the same ``mae/wape/rmse/peak_mae`` columns
    the single-model summary table uses, so the two are directly comparable.
    """
    model_labels = {"sarimax": "SARIMAX(1,0,1)(1,0,1,24)", "seasonal_naive": "Seasonal-naive(168h)"}
    rows: list[dict] = []
    for key in ("seasonal_naive", "sarimax"):
        summary = comparison.get(key)
        if summary is None:
            continue
        match = summary[summary["building"] == building]
        if len(match) == 0:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "model": model_labels.get(key, key),
                "mae": row["mae"],
                "wape": row["wape"],
                "rmse": row["rmse"],
                "peak_mae": row["peak_mae"],
            }
        )
    display = pd.DataFrame(rows, columns=["model", "mae", "wape", "rmse", "peak_mae"])
    formatted = display.copy()
    for col in ("mae", "rmse", "peak_mae"):
        formatted[col] = formatted[col].map(
            lambda v: f"{v:,.3f}" if pd.notna(v) else "nan"
        )
    formatted["wape"] = formatted["wape"].map(
        lambda v: f"{v * 100:,.2f}%" if pd.notna(v) else "nan"
    )
    return formatted.to_string(index=False)


def _format_summary(summary: pd.DataFrame) -> str:
    """Render the summary table as aligned plain text."""
    display = summary.copy()
    for col in ("mae", "rmse", "peak_mae"):
        if col in display.columns:
            display[col] = display[col].map(
                lambda v: f"{v:,.3f}" if pd.notna(v) else "nan"
            )
    display["wape"] = display["wape"].map(
        lambda v: f"{v * 100:,.2f}%" if pd.notna(v) else "nan"
    )
    return display.to_string(index=False)


def main() -> None:
    """CLI: run the academic SARIMAX-vs-baseline comparison and print the table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    comparison = run_academic_comparison()
    print("Academic building: SARIMAX vs seasonal-naive baseline")
    print("=" * 58)
    print(format_comparison(comparison, building=ACADEMIC_BUILDING))


if __name__ == "__main__":
    main()
