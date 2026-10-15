"""Historical hourly weather ingestion for the EnergyWise dataset.

Downloads **hourly historical weather** for IIIT-Delhi (approx. latitude
28.545, longitude 77.273) from Open-Meteo's *archive* API
(``https://archive-api.open-meteo.com/v1/archive``, no API key required) for
the period 2013-08-10 to 2017-12-31, saves the raw JSON responses unchanged,
and builds the processed hourly weather table used to enrich the energy data.

.. note::
   The archive API serves **ERA5 reanalysis** values.  These are *realized
   weather* (a physically consistent reconstruction of what the weather
   actually was), **not** archived weather forecasts.  They must therefore not
   be described or used as forecast features.

Pipeline
--------

1. :func:`fetch_weather` requests ``temperature_2m``,
   ``relative_humidity_2m`` and ``precipitation`` with
   ``timezone=Asia/Kolkata``.  The full date range is split into chunks of at
   most one year; each request is retried on failure with a short pause
   between calls.  Every raw JSON response is written **unchanged** under
   ``data/raw/weather/`` and the request URL plus retrieval time are logged.
2. :func:`build_weather_hourly` concatenates the saved responses into
   ``data/processed/weather_hourly.parquet`` with columns ``timestamp_ist``
   (timezone-aware, hour start, ``Asia/Kolkata``), ``temperature_c``,
   ``humidity_pct`` and ``precipitation_mm``.
3. :func:`join_energy_weather` left-joins that table onto
   ``data/processed/hourly_energy.parquet`` by ``timestamp_ist`` and writes
   ``data/processed/hourly_energy_weather.parquet``.  The join must preserve
   the energy row count exactly (no duplicated or dropped rows); it is
   asserted and the number of hours with missing weather is logged.  Weather
   values are **never shifted or interpolated**.

Typical usage::

    python -m energywise.ingestion.fetch_weather
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------

#: Repository root: .../energywise (this file is src/energywise/ingestion/fetch_weather.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
WEATHER_HOURLY_PATH = PROJECT_ROOT / "data" / "processed" / "weather_hourly.parquet"
ENERGY_HOURLY_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_energy.parquet"
ENERGY_WEATHER_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_energy_weather.parquet"

#: Open-Meteo historical archive endpoint (ERA5 reanalysis, no API key).
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"

#: IIIT-Delhi (approximate coordinates).
LATITUDE = 28.545
LONGITUDE = 77.273

#: Inclusive date range to download.
START_DATE = date(2013, 8, 10)
END_DATE = date(2017, 12, 31)

#: Hourly variables requested from the API.
HOURLY_VARIABLES = ["temperature_2m", "relative_humidity_2m", "precipitation"]

#: Reporting / binning timezone (the dataset convention).
IST_TZ = "Asia/Kolkata"

#: Maximum length of a single request window.
MAX_CHUNK_DAYS = 365

#: Download robustness knobs.
REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 5
RETRY_BACKOFF_S = 5.0
PAUSE_BETWEEN_CALLS_S = 1.0

#: Weather variable -> processed column name.
VARIABLE_TO_COLUMN = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "precipitation": "precipitation_mm",
}

#: Processed weather output columns, in order.
WEATHER_OUTPUT_COLUMNS = ["timestamp_ist", "temperature_c", "humidity_pct", "precipitation_mm"]

#: Columns added to the energy table by the join.
WEATHER_JOIN_COLUMNS = ["temperature_c", "humidity_pct", "precipitation_mm"]


def _configure_logging(level: int = logging.INFO) -> None:
    """Ensure the root logger has a console handler at an INFO level."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    if logger.getEffectiveLevel() > level:
        logger.setLevel(level)


# ---------------------------------------------------------------------------
# Date chunking
# ---------------------------------------------------------------------------


def _date_chunks(
    start: date = START_DATE,
    end: date = END_DATE,
    max_days: int = MAX_CHUNK_DAYS,
) -> list[tuple[date, date]]:
    """Split ``[start, end]`` into inclusive chunks of at most ``max_days``.

    Chunk boundaries are contiguous: each chunk starts the day after the
    previous one ended, so no hour is requested twice or skipped.
    """
    if end < start:
        raise ValueError(f"End date {end} is before start date {start}")

    chunks: list[tuple[date, date]] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=max_days - 1), end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
    return chunks


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


def _build_params(chunk_start: date, chunk_end: date) -> dict[str, str]:
    """Assemble the query parameters for one archive request."""
    return {
        "latitude": str(LATITUDE),
        "longitude": str(LONGITUDE),
        "start_date": chunk_start.isoformat(),
        "end_date": chunk_end.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": IST_TZ,
    }


def _download_chunk(
    chunk_start: date,
    chunk_end: date,
    raw_dir: Path,
    session: requests.Session,
) -> Path:
    """Download one chunk with retries and save the raw JSON unchanged.

    The request URL and retrieval time are logged on success.  On repeated
    failure a :class:`RuntimeError` is raised so the caller can decide whether
    to continue.
    """
    params = _build_params(chunk_start, chunk_end)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(ARCHIVE_API_URL, params=params, timeout=REQUEST_TIMEOUT_S)
            response.raise_for_status()
            retrieved_at = datetime.now(timezone.utc)

            # Save the raw body exactly as returned (no re-serialisation).
            raw_text = response.text
            # ``requests`` already decoded the URL; reconstruct a readable form.
            request_url = response.url

            raw_dir.mkdir(parents=True, exist_ok=True)
            out_path = raw_dir / f"weather_{chunk_start.isoformat()}_{chunk_end.isoformat()}.json"
            out_path.write_text(raw_text, encoding="utf-8")

            logger.info(
                "Retrieved %s..%s | url=%s | retrieved_at=%s | saved=%s",
                chunk_start.isoformat(),
                chunk_end.isoformat(),
                request_url,
                retrieved_at.isoformat(),
                out_path.name,
            )
            return out_path
        except Exception as exc:  # noqa: BLE001 - retry on any transport/HTTP error
            last_error = exc
            logger.warning(
                "Attempt %d/%d failed for %s..%s: %s",
                attempt,
                MAX_RETRIES,
                chunk_start.isoformat(),
                chunk_end.isoformat(),
                exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)

    raise RuntimeError(
        f"Failed to download weather for {chunk_start}..{chunk_end} after "
        f"{MAX_RETRIES} attempts"
    ) from last_error


def fetch_weather(
    raw_dir: Path | str = RAW_WEATHER_DIR,
    start: date = START_DATE,
    end: date = END_DATE,
    pause_s: float = PAUSE_BETWEEN_CALLS_S,
) -> list[Path]:
    """Download the full range in <=1-year chunks, saving raw JSON unchanged.

    Parameters
    ----------
    raw_dir:
        Directory for the raw JSON responses (created if needed).
    start, end:
        Inclusive date range to request.
    pause_s:
        Short pause between successive chunk requests.

    Returns
    -------
    list[pathlib.Path]
        The paths of the saved raw JSON files, in request order.
    """
    _configure_logging()

    raw_dir = Path(raw_dir)
    chunks = _date_chunks(start, end)
    logger.info(
        "Fetching weather for %.3f,%.3f | %s..%s | %d chunk(s) | timezone=%s",
        LATITUDE,
        LONGITUDE,
        start.isoformat(),
        end.isoformat(),
        len(chunks),
        IST_TZ,
    )

    saved: list[Path] = []
    with requests.Session() as session:
        for index, (chunk_start, chunk_end) in enumerate(chunks):
            saved.append(_download_chunk(chunk_start, chunk_end, raw_dir, session))
            if index < len(chunks) - 1 and pause_s:
                time.sleep(pause_s)

    logger.info("Saved %d raw weather file(s) under %s", len(saved), raw_dir)
    return saved


# ---------------------------------------------------------------------------
# Building the processed hourly table
# ---------------------------------------------------------------------------


def _raw_response_to_frame(payload: dict) -> pd.DataFrame:
    """Convert one Open-Meteo archive payload into a tidy hourly frame."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError("Open-Meteo response is missing the 'hourly' block")

    frame = pd.DataFrame(
        {
            "timestamp_ist": pd.to_datetime(hourly["time"]),
            **{VARIABLE_TO_COLUMN[var]: hourly.get(var) for var in HOURLY_VARIABLES if var in hourly},
        }
    )

    # The API returns naive local wall-clock strings for the requested tz.
    frame["timestamp_ist"] = frame["timestamp_ist"].dt.tz_localize(
        IST_TZ, ambiguous="NaT", nonexistent="NaT"
    )
    return frame[WEATHER_OUTPUT_COLUMNS]


def build_weather_hourly(
    raw_dir: Path | str = RAW_WEATHER_DIR,
    output_path: Path | str = WEATHER_HOURLY_PATH,
    write_parquet: bool = True,
) -> pd.DataFrame:
    """Concatenate the saved raw responses into the hourly weather Parquet.

    Returns a frame with ``timestamp_ist`` (tz-aware, hour start, Asia/Kolkata),
    ``temperature_c``, ``humidity_pct`` and ``precipitation_mm``.  Values are
    taken verbatim from the API (no shifting, interpolation or filling).
    """
    _configure_logging()

    raw_dir = Path(raw_dir)
    output_path = Path(output_path)

    files = sorted(raw_dir.glob("weather_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No raw weather JSON found in {raw_dir}. Run fetch_weather() first."
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames.append(_raw_response_to_frame(payload))

    weather = pd.concat(frames, ignore_index=True)
    weather = weather.sort_values("timestamp_ist").drop_duplicates(
        subset=["timestamp_ist"], keep="first"
    )
    weather = weather.reset_index(drop=True)

    logger.info(
        "Built hourly weather: %d rows | %s .. %s",
        len(weather),
        weather["timestamp_ist"].min(),
        weather["timestamp_ist"].max(),
    )

    if write_parquet:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weather.to_parquet(output_path, index=False)
        logger.info("Wrote hourly weather -> %s", output_path)

    return weather


# ---------------------------------------------------------------------------
# Joining onto the energy table
# ---------------------------------------------------------------------------


def _as_ist(series: pd.Series) -> pd.Series:
    """Coerce a datetime series to timezone-aware Asia/Kolkata hour starts."""
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST_TZ)
    else:
        ts = ts.dt.tz_convert(IST_TZ)
    return ts.dt.floor("h")


def join_energy_weather(
    energy_path: Path | str = ENERGY_HOURLY_PATH,
    weather_path: Path | str = WEATHER_HOURLY_PATH,
    output_path: Path | str = ENERGY_WEATHER_PATH,
    write_parquet: bool = True,
) -> pd.DataFrame:
    """Left-join hourly weather onto the hourly energy table by ``timestamp_ist``.

    The energy row count is preserved exactly: a join that duplicated or
    dropped rows would raise.  Weather values are attached as-is (never shifted
    or interpolated).  Hours whose weather is unavailable simply get NaN in the
    weather columns; the number of such hours is logged.

    Parameters
    ----------
    energy_path:
        ``data/processed/hourly_energy.parquet`` (read-only).
    weather_path:
        ``data/processed/weather_hourly.parquet`` (read-only).
    output_path:
        Destination for the joined Parquet file.
    write_parquet:
        When ``True`` (default) the joined frame is written to ``output_path``.

    Returns
    -------
    pandas.DataFrame
        The energy frame with the weather columns appended.
    """
    _configure_logging()

    energy_path = Path(energy_path)
    weather_path = Path(weather_path)
    output_path = Path(output_path)

    if not energy_path.exists():
        raise FileNotFoundError(
            f"Energy parquet not found: {energy_path}. Run clean_hourly() first."
        )
    if not weather_path.exists():
        raise FileNotFoundError(
            f"Weather parquet not found: {weather_path}. Run build_weather_hourly() first."
        )

    energy = pd.read_parquet(energy_path)
    weather = pd.read_parquet(weather_path)

    if "timestamp_ist" not in energy.columns:
        raise ValueError("Energy frame is missing the required 'timestamp_ist' column")

    energy = energy.copy()
    weather = weather.copy()
    energy["timestamp_ist"] = _as_ist(energy["timestamp_ist"])
    weather["timestamp_ist"] = _as_ist(weather["timestamp_ist"])

    # One weather row per hour is required, otherwise the left join would
    # multiply energy rows.  Fail loudly rather than silently duplicating.
    if weather["timestamp_ist"].duplicated().any():
        raise ValueError(
            "Weather table has duplicate timestamp_ist values; cannot perform a "
            "row-preserving left join."
        )

    energy_rows = len(energy)
    joined = energy.merge(
        weather[["timestamp_ist", *WEATHER_JOIN_COLUMNS]],
        on="timestamp_ist",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    joined_rows = len(joined)
    assert joined_rows == energy_rows, (
        f"Join changed the row count: energy={energy_rows}, joined={joined_rows}. "
        "The join must not duplicate or drop rows."
    )

    # No shifting / interpolation: report how many hours carry no weather.
    missing_mask = joined[WEATHER_JOIN_COLUMNS].isna().any(axis=1)
    n_missing = int(missing_mask.sum())
    if energy_rows:
        logger.info(
            "Joined weather onto energy: %d rows (%d hour(s), %.2f%%) have missing weather",
            joined_rows,
            n_missing,
            n_missing / energy_rows * 100,
        )
    else:
        logger.info("Joined weather onto energy: 0 rows")

    if write_parquet:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joined.to_parquet(output_path, index=False)
        logger.info("Wrote energy+weather -> %s", output_path)

    return joined


# ---------------------------------------------------------------------------
# Orchestration / CLI
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Run the full fetch -> build -> join pipeline."""
    _configure_logging()
    fetch_weather()
    build_weather_hourly()
    join_energy_weather()


def main() -> None:
    """CLI entry point."""
    run_pipeline()


if __name__ == "__main__":
    main()