"""Full daily-step rolling-origin backtest: Random Forest vs seasonal-naive.

Runs the shared backtest harness for **all four buildings** with the Random
Forest forecaster and the seasonal-naive baseline, then prints (and saves) the
per-building ``mae`` / ``wape`` / ``peak_mae`` / ``rmse`` table side by side.

Usage::

    python scripts/run_random_forest_backtest.py

The Random Forest is slower than the baseline (it refits ~1,577 trees per
building), so progress is flushed to stdout and the final table is also written
to ``data/rf_backtest_results.txt``.
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

from energywise.models.random_forest_model import (  # noqa: E402
    compare_random_forest_vs_baseline,
    format_rf_comparison,
)

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_energy_weather.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "rf_backtest_results.txt"

HORIZON = 24      # daily forecast
STEP = 24         # daily-step origin advance
SEASON = 168      # one-week seasonal lag for the baseline


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

    t0 = time.time()
    table = compare_random_forest_vs_baseline(
        INPUT_PATH,
        buildings=buildings,
        horizon=HORIZON,
        step=STEP,
        season=SEASON,
    )
    elapsed = time.time() - t0

    header = "Random Forest vs seasonal-naive baseline (daily-step rolling-origin backtest)"
    body = format_rf_comparison(table)
    report = "\n".join([header, "=" * 92, body, "", f"Elapsed: {elapsed / 60:.1f} min"])

    print("\n" + report, flush=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved -> {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()