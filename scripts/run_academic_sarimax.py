import time
import numpy as np
import pandas as pd

from energywise.models.baseline import seasonal_naive_forecast
from energywise.models.sarimax_model import fit_and_forecast_sarimax

HORIZON = 24
INITIAL_TRAIN_SIZE = 672
STEP = 168  # weekly folds instead of daily -> ~7x fewer fits

d = pd.read_parquet("data/processed/hourly_energy_weather.parquet")
d = d[d["building"] == "academic"].sort_values("timestamp_ist").reset_index(drop=True)
values = d["energy_kwh"].to_numpy()
n = len(values)

starts = list(range(INITIAL_TRAIN_SIZE, n - HORIZON + 1, STEP))
print(f"Total folds to run: {len(starts)}", flush=True)

naive_actual, naive_fc_all = [], []
sar_actual, sar_fc_all = [], []
t0 = time.time()

for i, s in enumerate(starts):
    history = values[:s]
    actual = values[s:s + HORIZON]
    if np.all(np.isnan(actual)):
        continue

    naive_fc = seasonal_naive_forecast(history, HORIZON)
    sar_fc = fit_and_forecast_sarimax(history, HORIZON)

    naive_actual.append(actual); naive_fc_all.append(naive_fc)
    sar_actual.append(actual); sar_fc_all.append(sar_fc)

    elapsed = time.time() - t0
    avg = elapsed / (i + 1)
    remaining = avg * (len(starts) - (i + 1))
    print(f"[{i+1}/{len(starts)}] done | elapsed {elapsed/60:.1f} min | "
          f"est. remaining {remaining/60:.1f} min", flush=True)

def summarise(actual_list, fc_list, label):
    a = np.concatenate(actual_list)
    f = np.concatenate(fc_list)
    mask = ~np.isnan(a) & ~np.isnan(f)
    diff = a[mask] - f[mask]
    mae = np.mean(np.abs(diff))
    wape = np.sum(np.abs(diff)) / np.sum(np.abs(a[mask])) * 100
    rmse = np.sqrt(np.mean(diff ** 2))
    print(f"\n{label}: MAE={mae:.3f}  WAPE={wape:.2f}%  RMSE={rmse:.3f}  n={mask.sum()}")

print("\n--- RESULTS (weekly folds, Academic) ---")
summarise(naive_actual, naive_fc_all, "Seasonal-naive")
summarise(sar_actual, sar_fc_all, "SARIMAX")