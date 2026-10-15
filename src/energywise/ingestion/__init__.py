"""EnergyWise ingestion layer.

Exposes :func:`load_raw`, which reads the four building mains CSVs and writes
a combined raw-layer Parquet file; :func:`clean_hourly`, which reads that raw
layer and writes a cleaned hourly energy table; and the weather helpers from
:mod:`energywise.ingestion.fetch_weather`, which download historical hourly
weather and join it onto the hourly energy table.
"""

from energywise.ingestion.clean_hourly import (
    MIN_MINUTES_PRESENT,
    PROCESSED_OUTPUT_PATH,
    clean_hourly,
)
from energywise.ingestion.fetch_weather import (
    ENERGY_HOURLY_PATH,
    ENERGY_WEATHER_PATH,
    RAW_WEATHER_DIR,
    WEATHER_HOURLY_PATH,
    build_weather_hourly,
    fetch_weather,
    join_energy_weather,
)
from energywise.ingestion.load_raw import (
    BUILDING_FILES,
    RAW_DATA_DIR,
    RAW_OUTPUT_PATH,
    load_raw,
)

__all__ = [
    "BUILDING_FILES",
    "RAW_DATA_DIR",
    "RAW_OUTPUT_PATH",
    "load_raw",
    "MIN_MINUTES_PRESENT",
    "PROCESSED_OUTPUT_PATH",
    "clean_hourly",
    "ENERGY_HOURLY_PATH",
    "ENERGY_WEATHER_PATH",
    "RAW_WEATHER_DIR",
    "WEATHER_HOURLY_PATH",
    "build_weather_hourly",
    "fetch_weather",
    "join_energy_weather",
]