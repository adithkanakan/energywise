# Model card: EnergyWise demand forecaster

## Model details

- **Model:** LightGBM gradient-boosted trees (`GBM (LightGBM) v1`), used as the production model in this project.
- **Task:** Hourly electricity-demand (kWh) forecasting, 24-hour horizon (up to 168 h via the API).
- **Author:** Adith A. Portfolio project, October 2026.
- **Compared against:** seasonal-naive baseline, SARIMAX (Academic only) and Random Forest.
- **Hyper-parameters:** reasonable defaults, not tuned.

## Intended use

- Demonstrating an end-to-end forecasting workflow: ingestion, leakage-safe features, backtesting, API and dashboard.
- Learning and portfolio review.

## Out of scope

- Real operational decisions, such as procurement, billing or demand-response commitments.
- Forecasting for buildings other than the four campus buildings in the training data (Academic, Facilities, Lecture, Library). Nothing here shows the results generalise to small retail or office customers.
- Predicting real future time. The data ends on 31 Dec 2017, and `/forecast` predicts a held-out window.

## Data

- **Source:** I-BLEND campus dataset (IIIT-Delhi), 1-minute smart-meter readings, 10 Aug 2013 to 31 Dec 2017. Not redistributed in this repo; see its own license.
- **Processing:** watts converted to hourly kWh. An hour is valid only if at least 45 of 60 minutes are present; otherwise it is missing (never zero, never interpolated).
- **Weather:** Open-Meteo ERA5 reanalysis (realised weather, not forecasts).
- **Missing hours:** about 1.9% (Academic, Lecture), 5.9% (Facilities), 25.9% (Library, longest gap about 204 days).

## Features

Calendar features, a fixed list of Indian public holidays, lagged values, rolling statistics and weather. All lag and rolling features use past values only, and a test checks that corrupting the future does not change earlier features.

## Evaluation

Rolling-origin backtest, 24 h horizon, origin advanced weekly (168 h), chronological splits. WAPE (%), lower is better:

| Building   | Seasonal-naive | Random Forest | LightGBM |
|------------|---------------:|--------------:|---------:|
| Academic   | 18.8  | 8.6  | 7.8  |
| Facilities | 13.5  | 7.2  | 6.7  |
| Lecture    | 141.6 | 57.6 | 47.9 |
| Library    | 32.0  | 14.9 | 13.0 |
| **All**    | 20.5  | 9.6  | 8.7  |

Across all buildings, MAE is 2.15 (naive), 1.01 (Random Forest) and 0.91 kWh (LightGBM).

## Known weaknesses

- **Lecture building:** low load with long zero-usage stretches, so percentage errors are large (WAPE 47.9%).
- **Unscheduled events:** the model misses one-off events and festival-season closures (for example early-October Saturdays and three Lecture spikes), because the holiday list is basic.
- **Bias:** over-predicts more often than it under-predicts (70-84% of errors for Academic, Lecture and Library).
- **Optimistic backtest:** uses realised weather. Real deployment would use forecast weather, which adds error.
- **Margin over Random Forest is modest** (about 0.5-1.8 WAPE points) and was not tested for statistical significance.
- **Cost estimates:** the tariff is an unverified approximation, and the simulator reports "no savings opportunity" on the flat tariff.

## Ethical and safety notes

The data is building-level energy use with no personal information. The main risk is over-trusting the numbers: they come from one campus and should not be used for cost claims or operational decisions without re-validation on the target buildings.

## Citation

Rashid, H., Singh, P., & Singh, A. (2019). I-BLEND, a campus-scale commercial and residential buildings electrical energy dataset. Scientific Data, 6, 190015. https://doi.org/10.1038/sdata.2019.15
