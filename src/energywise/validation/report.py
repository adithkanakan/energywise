"""Data-quality validation for the EnergyWise raw building meter layer.

This module reads the combined raw-layer Parquet file produced by
:mod:`energywise.ingestion.load_raw` (``data/raw/combined_raw.parquet``) and
writes a human-readable data-quality report to ``data/quality_report.md``.

The raw file is treated as **read-only** -- it is never modified.

Timestamps in the Parquet file are UTC, but every timestamp shown in the
report is converted to ``Asia/Kolkata`` (IST, UTC+05:30), matching the
dataset README.  ``power`` is in watts.

For each building the report covers:

1. total rows and date range,
2. duplicate timestamps (count, and whether duplicate rows are identical),
3. missing intervals (gaps > 1 minute between consecutive readings) with the
   count of gaps, total missing minutes, and the 10 largest gaps,
4. impossible values: negative power, and voltage / frequency far outside a
   plausible mains range (nominal 230 V / 50 Hz),
5. NA fraction for every numeric column.

``power_factor`` legitimately flips sign (leading vs. lagging) so its negative
values are reported as an informational statistic, **not** as an error.

Finally the per-building present-minute count is cross-checked against
``data/raw/energy_dataset/data_present_status_buildings.csv``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------

#: Repository root: .../energywise  (this file is src/energywise/validation/report.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_PARQUET_PATH = PROJECT_ROOT / "data" / "raw" / "combined_raw.parquet"
PRESENT_STATUS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "energy_dataset" / "data_present_status_buildings.csv"
)
REPORT_PATH = PROJECT_ROOT / "data" / "quality_report.md"

#: Reporting timezone (the dataset convention).
IST_TZ = "Asia/Kolkata"

#: Measurement columns present in the raw layer.
MEASUREMENT_COLUMNS = ["power", "current", "voltage", "frequency", "power_factor"]

#: Plausible mains envelope.  "Far outside" is interpreted as a generous
#: +/-20 % band around the 230 V nominal and +/-10 % around the 50 Hz nominal
#: so that ordinary sags/swells are NOT flagged, only clearly impossible values.
VOLTAGE_NOMINAL = 230.0
VOLTAGE_TOLERANCE = 0.20  # -> 184 V .. 276 V
VOLTAGE_MIN = VOLTAGE_NOMINAL * (1 - VOLTAGE_TOLERANCE)
VOLTAGE_MAX = VOLTAGE_NOMINAL * (1 + VOLTAGE_TOLERANCE)

FREQUENCY_NOMINAL = 50.0
FREQUENCY_TOLERANCE = 0.10  # -> 45 Hz .. 55 Hz
FREQUENCY_MIN = FREQUENCY_NOMINAL * (1 - FREQUENCY_TOLERANCE)
FREQUENCY_MAX = FREQUENCY_NOMINAL * (1 + FREQUENCY_TOLERANCE)

#: A gap is any interval between consecutive readings longer than this many
#: seconds.  The expected cadence is one reading per minute.
GAP_THRESHOLD_SECONDS = 60
EXPECTED_INTERVAL_SECONDS = 60

#: Building tag in the raw layer -> column in the present-status CSV.
BUILDING_TO_STATUS_COLUMN = {
    "academic": "Academic",
    "facilities": "Facilities",
    "lecture": "Lecture",
    "library": "Library",
}

#: Number of largest gaps to list per building.
TOP_N_GAPS = 10


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw_parquet(path: Path | str = RAW_PARQUET_PATH) -> pd.DataFrame:
    """Read the combined raw Parquet file (read-only) with IST timestamps."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw parquet not found: {path}. Run the ingestion step first "
            "(energywise.ingestion.load_raw)."
        )
    df = pd.read_parquet(path)
    return df


def _to_ist(ts) -> pd.Timestamp:
    """Return a timestamp converted to the reporting timezone (IST)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(IST_TZ)


def _fmt_ist(ts) -> str:
    """Format a timestamp as ``YYYY-MM-DD HH:MM:SS +0530`` in IST."""
    return _to_ist(ts).strftime("%Y-%m-%d %H:%M:%S %z")


# ---------------------------------------------------------------------------
# Per-building analyses
# ---------------------------------------------------------------------------


def _analyse_duplicates(df: pd.DataFrame) -> dict:
    """Count duplicate timestamps and check whether duplicate rows are identical."""
    dup_mask = df.duplicated(subset=["timestamp_utc"], keep=False)
    dup_df = df[dup_mask]

    duplicate_rows = int(df.duplicated(subset=["timestamp_utc"], keep="first").sum())
    duplicate_timestamps = int(dup_df["timestamp_utc"].nunique())

    identical = True
    if len(dup_df):
        value_cols = [c for c in MEASUREMENT_COLUMNS if c in dup_df.columns]
        # A group is identical when every value column has <= 1 unique value.
        grouped = dup_df.groupby("timestamp_utc")[value_cols]
        identical = bool(grouped.nunique(dropna=False).le(1).all().all())

    return {
        "duplicate_rows": duplicate_rows,
        "duplicate_timestamps": duplicate_timestamps,
        "identical": identical,
    }


def _analyse_gaps(df: pd.DataFrame) -> dict:
    """Find inter-reading gaps longer than the threshold (default 1 minute)."""
    ordered = df.sort_values("timestamp_utc")
    ts = ordered["timestamp_utc"].reset_index(drop=True)
    deltas = ts.diff().dt.total_seconds()

    gap_mask = deltas > GAP_THRESHOLD_SECONDS
    gap_seconds = deltas[gap_mask]

    # "Missing minutes" for a gap of ``g`` seconds is ``g/60 - 1`` (the minute
    # immediately after the last reading is the first missing one).
    missing_minutes = (gap_seconds / EXPECTED_INTERVAL_SECONDS - 1).clip(lower=0)

    # The gap begins right after the last reading before it.
    prev_ts = ts.shift(1)[gap_mask]

    gaps = pd.DataFrame(
        {
            "start": prev_ts.values,
            "missing_minutes": missing_minutes.values,
            "gap_seconds": gap_seconds.values,
        }
    ).sort_values("missing_minutes", ascending=False)

    return {
        "count": int(gap_mask.sum()),
        "total_missing_minutes": float(missing_minutes.sum()),
        "largest": gaps.head(TOP_N_GAPS),
    }


def _analyse_impossible(df: pd.DataFrame) -> dict:
    """Flag negative power and out-of-range voltage / frequency."""
    power = df["power"]
    voltage = df["voltage"]
    frequency = df["frequency"]

    neg_power = int((power < 0).sum())
    volt_low = int((voltage < VOLTAGE_MIN).sum())
    volt_high = int((voltage > VOLTAGE_MAX).sum())
    freq_low = int((frequency < FREQUENCY_MIN).sum())
    freq_high = int((frequency > FREQUENCY_MAX).sum())

    return {
        "negative_power": neg_power,
        "voltage_low": volt_low,
        "voltage_high": volt_high,
        "frequency_low": freq_low,
        "frequency_high": freq_high,
    }


def _na_fractions(df: pd.DataFrame) -> pd.Series:
    """NA fraction for every numeric column (including the epoch timestamp)."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df[numeric_cols].isna().mean()


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def build_report(
    raw_path: Path | str = RAW_PARQUET_PATH,
    present_status_path: Path | str = PRESENT_STATUS_PATH,
    top_n_gaps: int = TOP_N_GAPS,
) -> str:
    """Build the full Markdown data-quality report and return it as a string."""
    df = load_raw_parquet(raw_path)
    if "timestamp_ist" in df.columns:
        # Trust the stored IST column for display; fall back to conversion.
        pass

    overall_rows = len(df)
    buildings = sorted(df["building"].unique())

    now_utc = datetime.now(timezone.utc)
    lines: list[str] = []

    lines.append("# EnergyWise Data-Quality Report")
    lines.append("")
    lines.append(f"- **Source:** `{Path(raw_path)}` (read-only)")
    lines.append(f"- **Rows:** {overall_rows:,} across {len(buildings)} building(s)")
    lines.append(f"- **Buildings:** {', '.join(buildings)}")
    lines.append(
        f"- **Generated (UTC):** {now_utc.strftime('%Y-%m-%d %H:%M:%S')} "
        f"| **Asia/Kolkata (IST):** {_fmt_ist(now_utc)}"
    )
    lines.append(
        "- **Units / conventions:** `power` in watts; timestamps stored in UTC, "
        "**all times below are converted to Asia/Kolkata (IST, UTC+05:30)**."
    )
    lines.append(
        f"- **Plausible mains range:** voltage {VOLTAGE_MIN:.0f}-{VOLTAGE_MAX:.0f} V, "
        f"frequency {FREQUENCY_MIN:.1f}-{FREQUENCY_MAX:.1f} Hz "
        f"(nominal {VOLTAGE_NOMINAL:.0f} V / {FREQUENCY_NOMINAL:.0f} Hz)."
    )
    lines.append(
        f"- **Missing-interval rule:** any gap > {GAP_THRESHOLD_SECONDS} s "
        f"({EXPECTED_INTERVAL_SECONDS} s expected cadence)."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Cross-check (computed up front so it can be summarised).
    # ------------------------------------------------------------------
    cross_check = _cross_check_present_status(df, present_status_path)

    # ------------------------------------------------------------------
    # Per-building sections
    # ------------------------------------------------------------------
    pf_negative_total = 0
    for building in buildings:
        bdf = df[df["building"] == building]
        lines.append(f"## Building: `{building}`")
        lines.append("")

        # (1) total rows and date range
        ts_ist = bdf["timestamp_utc"].dt.tz_convert(IST_TZ)
        start = _fmt_ist(ts_ist.min()) if len(bdf) else "n/a"
        end = _fmt_ist(ts_ist.max()) if len(bdf) else "n/a"
        lines.append("### 1. Rows and date range")
        lines.append("")
        lines.append(f"- **Total rows:** {len(bdf):,}")
        lines.append(f"- **Date range (IST):** {start}  ->  {end}")
        lines.append("")

        # (2) duplicate timestamps
        dups = _analyse_duplicates(bdf)
        lines.append("### 2. Duplicate timestamps")
        lines.append("")
        lines.append(f"- **Duplicate rows (extra beyond first):** {dups['duplicate_rows']:,}")
        lines.append(
            f"- **Distinct timestamps that are duplicated:** "
            f"{dups['duplicate_timestamps']:,}"
        )
        if dups["duplicate_timestamps"]:
            lines.append(
                f"- **Duplicate rows identical across all values?** "
                f"**{'Yes' if dups['identical'] else 'No'}**"
            )
        else:
            lines.append("- **Duplicate rows identical across all values?** n/a (no duplicates)")
        lines.append("")

        # (3) missing intervals
        gaps = _analyse_gaps(bdf)
        lines.append("### 3. Missing intervals (gaps > 1 minute)")
        lines.append("")
        lines.append(f"- **Number of gaps:** {gaps['count']:,}")
        lines.append(f"- **Total missing minutes:** {gaps['total_missing_minutes']:,.0f}")
        if gaps["count"]:
            lines.append("")
            lines.append(f"**{min(top_n_gaps, gaps['count'])} largest gaps** "
                         "(start = last reading before the gap, IST):")
            lines.append("")
            rows = []
            for _, g in gaps["largest"].iterrows():
                rows.append(
                    [
                        _fmt_ist(g["start"]),
                        f"{g['missing_minutes']:,.0f}",
                        f"{g['gap_seconds'] / 60:,.1f}",
                    ]
                )
            lines.append(_markdown_table(["Gap start (IST)", "Missing minutes", "Span (min)"], rows))
        lines.append("")

        # (4) impossible values
        imp = _analyse_impossible(bdf)
        lines.append("### 4. Impossible values")
        lines.append("")
        lines.append(
            _markdown_table(
                ["Check", "Count"],
                [
                    ["Negative power (< 0 W)", f"{imp['negative_power']:,}"],
                    [f"Voltage below {VOLTAGE_MIN:.0f} V", f"{imp['voltage_low']:,}"],
                    [f"Voltage above {VOLTAGE_MAX:.0f} V", f"{imp['voltage_high']:,}"],
                    [f"Frequency below {FREQUENCY_MIN:.1f} Hz", f"{imp['frequency_low']:,}"],
                    [f"Frequency above {FREQUENCY_MAX:.1f} Hz", f"{imp['frequency_high']:,}"],
                ],
            )
        )
        lines.append("")

        # power_factor sign convention (informational)
        pf = bdf["power_factor"]
        pf_neg = int((pf < 0).sum())
        pf_nonnull = int(pf.notna().sum())
        pf_negative_total += pf_neg
        pf_pct = (pf_neg / pf_nonnull * 100) if pf_nonnull else 0.0
        lines.append(
            f"- **`power_factor` sign note:** {pf_neg:,} of {pf_nonnull:,} non-null values "
            f"({pf_pct:.2f}%) are **negative**. This is interpreted as a leading/lagging "
            "sign convention and is **not** counted as an error."
        )
        lines.append("")

        # (5) NA fraction for every numeric column
        na = _na_fractions(bdf)
        lines.append("### 5. NA fraction per numeric column")
        lines.append("")
        rows = []
        for col, frac in na.items():
            rows.append([f"`{col}`", f"{frac * 100:.4f}%", f"{int(bdf[col].isna().sum()):,}"])
        lines.append(_markdown_table(["Column", "NA fraction", "NA count"], rows))
        lines.append("")

    # ------------------------------------------------------------------
    # power_factor sign summary (overall)
    # ------------------------------------------------------------------
    lines.append("## `power_factor` sign convention (summary)")
    lines.append("")
    lines.append(
        f"A total of **{pf_negative_total:,}** rows have a negative `power_factor` "
        "across all buildings. Per the dataset this reflects the leading/lagging sign "
        "convention and is reported for information only -- it is **not** a data error."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Cross-check section
    # ------------------------------------------------------------------
    lines.append("## Cross-check: present-minute count vs. `data_present_status_buildings.csv`")
    lines.append("")
    lines.append(f"- **Reference file:** `{Path(present_status_path)}`")
    lines.append(
        "- **Method:** rows in the raw layer for each building vs. the count of "
        "minutes with status = 1 in the reference CSV."
    )
    lines.append("")
    rows = []
    any_mismatch = False
    for building in buildings:
        info = cross_check.get(building)
        if info is None:
            rows.append([building, "-", "-", "-", "no matching column"])
            continue
        delta = info["raw_rows"] - info["present_minutes"]
        status = "OK" if delta == 0 else "**MISMATCH**"
        if delta != 0:
            any_mismatch = True
        rows.append(
            [
                building,
                f"{info['raw_rows']:,}",
                f"{info['present_minutes']:,}",
                f"{delta:+,}",
                status,
            ]
        )
    lines.append(_markdown_table(["Building", "Raw rows", "Present minutes", "Delta", "Status"], rows))
    lines.append("")
    if any_mismatch:
        lines.append(
            "> **Note:** one or more buildings show a mismatch between the raw row count "
            "and the present-status reference. Differences indicate readings that exist in "
            "the raw data but are marked absent (or vice-versa) in the status file."
        )
    else:
        lines.append("All building present-minute counts match the reference file.")
    lines.append("")

    return "\n".join(lines)


def _cross_check_present_status(
    df: pd.DataFrame, present_status_path: Path | str
) -> dict:
    """Compare per-building raw row counts to the present-status reference CSV.

    The present-minute count is the number of rows in the reference CSV whose
    status column equals 1.  Results are keyed by building tag.
    """
    present_status_path = Path(present_status_path)
    results: dict[str, dict] = {}
    if not present_status_path.exists():
        logger.warning("Present-status file not found: %s", present_status_path)
        return results

    status = pd.read_csv(present_status_path)

    for building, column in BUILDING_TO_STATUS_COLUMN.items():
        raw_rows = int((df["building"] == building).sum())
        if column in status.columns:
            present_minutes = int((status[column] == 1).sum())
        else:
            present_minutes = None
        results[building] = {"raw_rows": raw_rows, "present_minutes": present_minutes}
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate(
    raw_path: Path | str = RAW_PARQUET_PATH,
    output_path: Path | str = REPORT_PATH,
    present_status_path: Path | str = PRESENT_STATUS_PATH,
) -> Path:
    """Run validation and write the Markdown report to ``output_path``.

    The raw Parquet file is read only; only ``output_path`` is written.

    Returns the path of the written report.
    """
    report = build_report(
        raw_path=raw_path,
        present_status_path=present_status_path,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info("Wrote data-quality report -> %s", output_path)
    return output_path


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    validate()


if __name__ == "__main__":
    main()
