"""EnergyWise data validation module.

Exposes :func:`validate`, which reads ``data/raw/combined_raw.parquet`` and
writes a data-quality report to ``data/quality_report.md``.
"""

from __future__ import annotations

from energywise.validation.report import (
    REPORT_PATH,
    build_report,
    load_raw_parquet,
    validate,
)

__all__ = ["validate", "build_report", "load_raw_parquet", "REPORT_PATH"]