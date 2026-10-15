"""Full daily-step rolling-origin backtest: naive vs Random Forest vs GBM.

Runs the shared backtest harness for **all four buildings** with the
seasonal-naive baseline, the Random Forest and the LightGBM gradient-boosted
tree (GBM) forecaster, then prints (and saves) the per-building ``mae`` /
``wape`` / ``rmse`` / ``peak_mae`` table with all three models side by side.

Usage::

    python scripts/run_gbm_backtest.py

The tree models are much slower than the baseline (each refits once per fold,
~1,577 folds per building), so progress -- with elapsed and remaining-time
estimates -- is flushed to stdout as each building finishes, and the final table
is also written to ``data/gbm_backtest_results.txt``.  The script is safe to
poll: the output file only appears once the run has completed, and the flushed
stdout tells you how far along it is.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Make ``src`` importable when run as a plain script (src-layout).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from energywise.evaluation.backtest import (  # noqa: E402
    BacktestResult,
    rolling_origin_backtest,
    summarise,
)
from energywise.models.gbm_model import run_gbm_backtest  # noqa: E402
from energywise.models.random_forest_model import (  # noqa: E402
    run_random_forest_backtest,
)

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_energy_weather.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "gbm_backtest_results.txt"

HORIZON = 24      # daily forecast
# Coarser weekly-step origin advance (7 days) for faster iteration, matching the
# approach used for the SARIMAX comparison.
STEP = 168        # weekly-step origin advance
SEASON = 168      # one-week seasonal lag for the baseline

METRIC_NAMES = ["mae", "wape", "rmse", "peak_mae"]


def _format_seconds(seconds: float) -> str:
    """Render a duration as ``1h 05m 03s`` / ``5m 03s`` / ``3s``."""
    seconds = max(int(round(seconds)), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _combine_results(results: list[BacktestResult]) -> BacktestResult:
    """Merge per-building results into one result with a single combined window.

    ``summarise`` emits a ``building``-row plus an ``ALL`` row for whatever it
    is given, so calling it per building would yield several duplicate ``ALL``
    rows (which then collide when stacked into one table).  Concatenating the
    predictions/folds first and summarising once keeps exactly one ``ALL`` row.
    """
    first = results[0]
    return BacktestResult(
        predictions=pd.concat([r.predictions for r in results], ignore_index=True),
        folds=pd.concat([r.folds for r in results], ignore_index=True),
        horizon=first.horizon,
        season=first.season,
        initial_train_size=first.initial_train_size,
        step=first.step,
        candidate_hours=sum(r.candidate_hours for r in results),
        skipped_hours=sum(r.skipped_hours for r in results),
        excluded_nan_hours=sum(r.excluded_nan_hours for r in results),
    )


def _build_table(
    baseline_summary: pd.DataFrame,
    rf_summary: pd.DataFrame,
    gbm_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the naive / RF / GBM comparison table from per-model summaries."""
    base = baseline_summary.set_index("building")
    rf = rf_summary.set_index("building") if len(rf_summary) else pd.DataFrame()
    gbm = gbm_summary.set_index("building") if len(gbm_summary) else pd.DataFrame()

    rows: list[dict] = []
    for building in base.index:
        row: dict = {"building": building}
        for metric in METRIC_NAMES:
            row[f"naive_{metric}"] = base.loc[building, metric]
            row[f"rf_{metric}"] = (
                rf.loc[building, metric] if building in rf.index else float("nan")
            )
            row[f"gbm_{metric}"] = (
                gbm.loc[building, metric] if building in gbm.index else float("nan")
            )
        rows.append(row)

    columns = ["building"]
    for metric in METRIC_NAMES:
        columns += [f"naive_{metric}", f"rf_{metric}", f"gbm_{metric}"]
    return pd.DataFrame(rows, columns=columns)


def _format_table(table: pd.DataFrame) -> str:
    """Render the three-way comparison as aligned plain text."""
    display = table.copy()
    for col in (c for c in display.columns if c.endswith(("_mae", "_rmse"))):
        display[col] = display[col].map(lambda v: f"{v:,.3f}" if pd.notna(v) else "nan")
    for col in (c for c in display.columns if c.endswith("_wape")):
        display[col] = display[col].map(
            lambda v: f"{v * 100:,.2f}%" if pd.notna(v) else "nan"
        )
    return display.to_string(index=False)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    hourly = pd.read_parquet(INPUT_PATH)
    buildings = sorted(hourly["building"].unique())
    print(f"Buildings: {buildings}", flush=True)
    print(
        f"horizon={HORIZON}h, step={STEP}h, season={SEASON}h "
        f"(daily-step rolling-origin backtest)",
        flush=True,
    )

    # One unit of work == one tree-model backtest for one building.  The naive
    # baseline is fast, so it is run first and not counted towards the ETA.
    total_units = len(buildings) * 2
    units_done = 0

    t0 = time.time()
    print(f"[{_format_seconds(0)}] running seasonal-naive baseline...", flush=True)
    baseline_result = rolling_origin_backtest(
        hourly,
        horizon=HORIZON,
        step=STEP,
        season=SEASON,
        buildings=buildings,
    )
    baseline_summary = summarise(baseline_result)
    print(
        f"[{_format_seconds(time.time() - t0)}] baseline done "
        f"({len(baseline_result.predictions)} rows)",
        flush=True,
    )

    rf_results: list[BacktestResult] = []
    gbm_results: list[BacktestResult] = []
    for building in buildings:
        for label, runner, store in (
            ("RandomForest", run_random_forest_backtest, rf_results),
            ("GBM", run_gbm_backtest, gbm_results),
        ):
            result = runner(
                hourly,
                buildings=building,
                horizon=HORIZON,
                step=STEP,
                season=SEASON,
            )
            store.append(result)
            units_done += 1

            elapsed = time.time() - t0
            per_unit = elapsed / units_done
            remaining = per_unit * (total_units - units_done)
            print(
                f"[{_format_seconds(elapsed)}] {label} {building}: "
                f"{len(result.predictions)} rows, {len(result.folds)} folds "
                f"({units_done}/{total_units} units, "
                f"~{_format_seconds(remaining)} remaining)",
                flush=True,
            )

    rf_summary = summarise(_combine_results(rf_results)) if rf_results else pd.DataFrame()
    gbm_summary = (
        summarise(_combine_results(gbm_results)) if gbm_results else pd.DataFrame()
    )
    table = _build_table(baseline_summary, rf_summary, gbm_summary)
    elapsed = time.time() - t0

    header = "Naive vs Random Forest vs GBM (daily-step rolling-origin backtest)"
    body = _format_table(table)
    report = "\n".join(
        [header, "=" * 92, body, "", f"Elapsed: {elapsed / 60:.1f} min"]
    )

    print("\n" + report, flush=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved -> {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()