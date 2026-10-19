# Project status

Tracks progress against the build plan. Update the checkbox as each numbered task
from the plan is completed.

- [x] Phase 0 — Repo skeleton
- [x] Phase 1 — Data feasibility (dataset: I-BLEND)
- [x] Phase 2 — Ingestion & validation
- [x] Phase 3 — Exploratory analysis
- [x] Phase 4 — Feature engineering (leakage-safe)
- [x] Phase 5 — Baseline + backtesting harness
- [x] Phase 6 — Candidate models
- [x] Phase 7 — Error analysis & explainability
- [x] Phase 8 — Tariff & cost-simulation module
- [x] Phase 9 — API
- [x] Phase 10 — Frontend (all 6 dashboard pages: Overview, Demand chart, Usage Analysis, Cost Simulator, Model Quality, Data Status — all built and verified against real data)
- [x] Phase 11 — Packaging, tests, deployment (Docker setup and GitHub Actions CI done; "deployment" means local `docker compose` only, there is no hosted deployment)
- [ ] Phase 12 — Final report & resume line

## Notes / decisions log

- Dataset: I-BLEND (IIIT-Delhi campus), CC-licensed via Figshare — confirm exact
  license text once the zip is unpacked.
- Buildings in scope: Academic, Lecture Hall, Library, Facilities.
- Weather: I-BLEND's own 4-month weather window + a supplementary source (TBD, Task 2)
  for the rest of the 52-month range.
- power is in watts, not kW (confirmed via dataset README); hourly kWh = mean(power)/1000
- timestamps converted from Unix epoch (UTC) to Asia/Kolkata (+5:30) before any hour-of-day/resampling logic
- hour counted valid only if >=45 of 60 minutes present; else energy_kwh = NaN, is_missing = True
- Library usable only from ~Sep 2015 onward (three long blackouts before that, ~360 days total)
- Lecture has real multi-week zero-usage stretches (not missing data) - likely academic-calendar shutdowns; spikes on Feb 14 2015, Mar 25 2017, Apr 8 2017 confirmed as real unscheduled evening events (not sensor errors) via error-analysis - coherent multi-hour rise/fall pattern, GBM underpredicts all of them since they're one-off with no recurring signal
- weather = Open-Meteo ERA5 (realized, not forecast) for ~28.545N 77.273E; backtest will be slightly optimistic vs a real forecast-weather deployment
- Random Forest beat seasonal-naive baseline on all 4 buildings, all metrics (overall WAPE 18.5% -> 9.0%); biggest relative gain on Lecture (69.1% -> 32.5% WAPE), confirming lag/holiday features fix the baseline's specific weakness there
- GBM beats Random Forest and naive baseline on all 4 buildings (weekly-step backtest: overall WAPE naive 20.5% -> RF 9.6% -> GBM 8.7%); GBM chosen as final model
- Error analysis: largest Academic errors cluster on specific Saturdays around Oct 2013/2015/2017 - likely festival-season (Dussehra/Navratri) and unscheduled campus events not captured by the fixed is_holiday list; model is systematically biased toward overprediction (70-84% of errors are overpredictions for Academic/Lecture/Library, closer to balanced for Facilities)
- API complete: /health, /metadata, /history, /forecast, /cost-simulation, /metrics, /data-status all manually tested against real data via Invoke-RestMethod; /cost-simulation correctly reports "no savings opportunity" on the flat tariff rather than a false positive
- Fixed /forecast endpoint bug: build_feature_matrix() resets its row index after dropping NaN rows, which broke a timestamp join (was showing forecasts from Sep 2017 instead of Jan 2018); fixed by filtering NaN rows directly on the dataframe that retains timestamp_ist, using the same FEATURE_COLUMNS/TARGET_COLUMN list
- Fixed tariff pricing to reflect ~2026 Delhi commercial rates (energy_charge_per_kwh 8.0 -> 12.5, based on secondary-sourced effective commercial rate range Rs 12.5-15.6/unit)
- Fixed a real bug in find_cheapest_window(): clipping the adjusted usage profile at 0 silently deleted energy on days where real usage dipped below the appliance's assumed draw, fabricating fake savings on a flat tariff (found via manual testing in the dashboard: 8kW/4hr/hour-11 showed a false Rs42 saving). Removed the clip; added a regression test using randomized realistic data that specifically covers this case.
- Rewrote find_cheapest_window() to return a deterministic "no savings opportunity" result instead of running a per-hour search, since our Tariff model is always flat (no time-of-day structure) -- a search can never find real savings here, and two separate bugs (negative-value clipping, then a data-gap mismatch between "from" and "to" hours) both produced tiny fake non-zero savings. The deterministic version is simpler and structurally immune to both.
- Added auto-estimated sanctioned_load_kw (from building's historical peak usage) since the previous hardcoded 10kW default was unrealistic (Academic's real peak is 60-80kW)
- Frontend dashboard complete: Overview (KPI cards), Demand chart (actual vs forecast with "Now" boundary), Usage Analysis (hour x weekday heatmap, 90-day window), Cost Simulator, Model Quality (metrics table), Data Status (coverage + missing-gap warnings) -- all 6 pages built and cross-checked against earlier validation/EDA numbers
- Added GitHub Actions CI (.github/workflows/ci.yml) with three jobs: backend pytest, frontend `npm ci` + `npm run build`, and Docker image builds for backend and frontend. First run (CI #1, commit 99b347a, 19 Oct 2026) passed on main in 1m10s.
- Tests that need the processed data files are skipped in CI because data/ is not committed, so the CI test count is lower than the local count. Run the full suite locally after running the pipeline.
- The CI Docker jobs only confirm that the images build; they do not start the containers. A full `docker compose up --build` smoke test is still a manual check.