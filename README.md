# EnergyWise

Intelligent energy forecasting & cost optimization for small commercial buildings.

A full-stack applied-ML project: a Python/FastAPI forecasting service backed by a
reproducible data pipeline, plus a React dashboard for visualizing actual-vs-predicted
demand and simulating tariff-aware cost scenarios.

> **Status:** early scaffold. See `PROJECT_STATUS.md` for what's implemented vs planned.

## Dataset

Using [I-BLEND](https://doi.org/10.1038/sdata.2019.15) — a 52-month, 1-minute-sampled
electrical energy dataset from 7 buildings (commercial and residential) on the IIIT-Delhi
campus, released by Rashid, Singh & Singh (2019), *Scientific Data*.

**Limitation (disclosed, not hidden):** these are academic-campus buildings, not
retail/office small businesses. Used here as a real, India-based stand-in dataset for
pipeline and model development — not a claim of direct generalization to small
commercial businesses. See `data/README.md`.

## Repository layout

```
energywise/
├── frontend/              # React + TypeScript + Vite dashboard
├── backend/                # FastAPI app (routes, schemas)
├── src/energywise/
│   ├── ingestion/           # source adapters and download jobs
│   ├── validation/          # schema and data-quality checks
│   ├── features/            # leakage-safe feature engineering
│   ├── models/               # baselines and candidate forecasting models
│   ├── evaluation/           # backtesting, metrics, error analysis
│   ├── tariffs/               # tariff definitions and bill calculator
│   └── optimization/          # constrained load-shifting scenario logic
├── data/                    # raw/processed data (not committed — see data/README.md)
├── models/                  # versioned trained-model artifacts (not committed)
├── tests/                   # pytest suite
├── notebooks/               # exploratory analysis (not production logic)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup

### Backend / ML pipeline (Python)

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the downloaded I-BLEND data into `data/raw/` (see `data/README.md` for expected
layout), then run the ingestion/cleaning pipeline (once implemented — Phase 2 of the
build plan).

Run the API locally:

```bash
uvicorn backend.main:app --reload
```

Then visit `http://localhost:8000/docs` for the interactive API docs.

### Frontend (React dashboard)

```bash
cd frontend
npm install
npm run dev
```

### Tests

```bash
pytest
```

### Docker (full stack)

```bash
docker compose up --build
```

## Build plan

This repo is being built phase-by-phase against a task list (data feasibility →
ingestion → baselines → models → error analysis → tariff logic → API → frontend →
deployment). See the project blueprint / build plan for the full breakdown.

## Limitations & honesty notes

- Cost-simulation output is always a **simulation**, never a claim of verified/measured
  savings, unless backed by real observed before/after data.
- Forecasts are only as good as the disclosed dataset and evaluation window — see the
  Model Quality page (once built) for backtest metrics and known failure modes.
