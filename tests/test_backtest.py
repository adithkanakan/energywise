"""Unit tests for the rolling-origin backtest harness.

The non-negotiable property is **no leakage**: for every fold, the training
window must end strictly before the test window begins.  These tests assert on
the actual date ranges the harness used (recorded in ``result.folds``), not on
an assumption about them.  Supporting tests cover per-building isolation,
all-NaN block skipping, expanding windows and the NaN-exclusion count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energywise.evaluation.backtest import rolling_origin_backtest, summarise

IST_TZ = "Asia/Kolkata"


def _frame(building: str, values: list[float], start: str = "2024-01-01 00:00:00") -> pd.DataFrame:
    ts = pd.date_range(pd.Timestamp(start, tz=IST_TZ), periods=len(values), freq="1h")
    return pd.DataFrame(
        {"building": building, "timestamp_ist": ts, "energy_kwh": values}
    )


# ---------------------------------------------------------------------------
# The key property: no training timestamp inside the fold's test window
# ---------------------------------------------------------------------------


def test_no_training_hour_falls_inside_its_own_test_window():
    """Assert on the actual ranges: train_end < test_start for every scored fold."""
    values = [float(i % 50) for i in range(168 * 8)]
    frame = _frame("A", values)

    result = rolling_origin_backtest(
        frame, horizon=24, initial_train_size=168 * 4, step=24
    )

    scored = result.folds[result.folds["scored"]]
    assert len(scored) > 0

    # Direct range assertion on the ranges the harness reports.
    assert (scored["train_end"] < scored["test_start"]).all()
    # Training starts at the series start and expands.
    assert (scored["train_start"] == frame["timestamp_ist"].min()).all()

    # Cross-check against the raw training windows: no test timestamp may sit
    # inside its fold's [train_start, train_end] range.
    for _, row in scored.iterrows():
        test_ts = result.predictions[
            (result.predictions["building"] == row["building"])
            & (result.predictions["fold"] == row["fold"])
        ]["timestamp_ist"]
        assert (test_ts >= row["test_start"]).all()
        assert (test_ts <= row["test_end"]).all()
        assert (test_ts > row["train_end"]).all()  # strictly after training
        assert not ((test_ts >= row["train_start"]) & (test_ts <= row["train_end"])).any()


def test_training_data_given_to_model_excludes_test_hours():
    """The history handed to the forecaster never contains a test-window hour."""
    values = [float(i) for i in range(168 * 6)]
    frame = _frame("A", values)

    seen: list[tuple[int, int]] = []

    def spy(history: np.ndarray, horizon: int) -> np.ndarray:
        # Record the length of the history used for each fold.
        seen.append((len(history), horizon))
        # Forecast a constant so we don't need a real model.
        return np.full(horizon, np.nan)

    rolling_origin_backtest(
        frame, forecast_fn=spy, horizon=24, initial_train_size=168 * 4, step=24
    )

    # Every history length must be a multiple-of-24-expanded window starting at
    # the initial training size -- i.e. it never includes the test hours.
    initial = 168 * 4
    expected_lengths = list(range(initial, len(values) - 24 + 1, 24))
    assert [length for length, _ in seen] == expected_lengths


# ---------------------------------------------------------------------------
# Expanding window
# ---------------------------------------------------------------------------


def test_training_window_expands_by_step():
    """Each fold's training set grows by ``step`` and never shrinks."""
    values = [float(i) for i in range(168 * 6)]
    result = rolling_origin_backtest(
        _frame("A", values), horizon=24, initial_train_size=168 * 4, step=24
    )
    scored = result.folds[result.folds["scored"]].reset_index(drop=True)

    lengths = scored["n_train"].to_numpy()
    assert (np.diff(lengths) == 24).all()
    assert lengths[0] == 168 * 4


# ---------------------------------------------------------------------------
# All-NaN blocks are skipped
# ---------------------------------------------------------------------------


def test_all_nan_block_is_skipped_not_forecast():
    """A 24h block whose actuals are entirely NaN produces no predictions."""
    n = 168 * 6
    values = [float(i % 30) for i in range(n)]
    block_start = 168 * 4  # the first test block
    for i in range(block_start, block_start + 24):
        values[i] = np.nan

    result = rolling_origin_backtest(
        _frame("A", values), horizon=24, initial_train_size=168 * 4, step=24
    )

    # The first fold must be flagged skipped, the rest scored.
    first = result.folds.iloc[0]
    assert not bool(first["scored"])
    assert first["skipped_reason"] == "all_actual_nan"

    # No prediction row falls in the skipped window.
    skipped = result.predictions[
        (result.predictions["timestamp_ist"] >= first["test_start"])
        & (result.predictions["timestamp_ist"] <= first["test_end"])
    ]
    assert len(skipped) == 0


def test_partially_nan_block_is_scored_with_pairwise_exclusion():
    """A block with some NaNs is scored; only the NaN hours are excluded."""
    n = 168 * 6
    values = [float(i % 30) for i in range(n)]
    block_start = 168 * 4
    values[block_start + 5] = np.nan  # one missing hour in the first block

    result = rolling_origin_backtest(
        _frame("A", values), horizon=24, initial_train_size=168 * 4, step=24
    )

    first = result.folds.iloc[0]
    assert bool(first["scored"])

    first_preds = result.predictions[result.predictions["fold"] == 0]
    assert len(first_preds) == 24
    # Exactly the one hour is NaN in actual.
    assert int(first_preds["actual"].isna().sum()) == 1
    assert result.excluded_nan_hours >= 1


# ---------------------------------------------------------------------------
# Per-building isolation
# ---------------------------------------------------------------------------


def test_buildings_are_backtested_independently():
    """Two buildings produce separate predictions on their own grids."""
    a = _frame("A", [float(i % 40) for i in range(168 * 5)])
    b = _frame(
        "B",
        [float(i % 25) for i in range(168 * 5)],
        start="2024-03-01 00:00:00",
    )
    frame = pd.concat([a, b], ignore_index=True)

    result = rolling_origin_backtest(
        frame, horizon=24, initial_train_size=168 * 4, step=24
    )

    assert set(result.predictions["building"].unique()) == {"A", "B"}
    assert set(result.folds["building"].unique()) == {"A", "B"}
    # Each building's folds start at its own series start.
    for building, group in result.folds.groupby("building"):
        own = frame[frame["building"] == building]["timestamp_ist"]
        assert (group["train_start"] == own.min()).all()


# ---------------------------------------------------------------------------
# Summary / metrics plumbing
# ---------------------------------------------------------------------------


def test_summarise_reports_one_row_per_building_plus_all():
    values = [float(i % 40) for i in range(168 * 5)]
    result = rolling_origin_backtest(
        _frame("A", values), horizon=24, initial_train_size=168 * 4, step=24
    )
    summary = summarise(result)

    assert list(summary["building"]) == ["A", "ALL"]
    for col in ("mae", "wape", "rmse", "n_scored", "n_excluded"):
        assert col in summary.columns
    # The baseline on a repeating pattern is imperfect but finite.
    assert np.isfinite(summary.loc[summary["building"] == "A", "mae"].iloc[0])


def test_requires_expected_columns():
    with pytest.raises(ValueError, match="energy_kwh"):
        rolling_origin_backtest(
            pd.DataFrame({"building": ["A"], "timestamp_ist": [pd.Timestamp("2024-01-01")]})
        )