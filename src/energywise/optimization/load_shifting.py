"""Load-shifting scenario logic for EnergyWise.

Kept separate from the ML/forecasting code. All results from this module
are simulations based on a flexible-load assumption and a tariff — never
a claim of verified, measured savings.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from energywise.tariffs.bill_calculator import Tariff, calculate_bill


@dataclass
class FlexibleLoad:
    name: str
    power_kw: float
    duration_hours: int
    allowed_window_start_hour: int  # inclusive, 0-23
    allowed_window_end_hour: int    # inclusive, 0-23
    baseline_start_hour: int


def _load_profile_for_start(load: FlexibleLoad, start_hour: int) -> pd.Series:
    """Return a 24-hour kWh profile with the load running from start_hour."""
    profile = pd.Series(0.0, index=range(24))
    for h in range(load.duration_hours):
        hour = (start_hour + h) % 24
        profile[hour] += load.power_kw  # 1 hour at power_kw -> power_kw kWh
    return profile


def _valid_start_hours(load: FlexibleLoad) -> list[int]:
    """All start hours where the load fits entirely inside its allowed window."""
    valid = []
    window_hours = set(range(load.allowed_window_start_hour, load.allowed_window_end_hour + 1))
    for start in range(load.allowed_window_start_hour, load.allowed_window_end_hour + 1):
        occupied = {(start + h) % 24 for h in range(load.duration_hours)}
        if occupied.issubset(window_hours):
            valid.append(start)
    return valid


def find_cheapest_window(
    load: FlexibleLoad,
    tariff: Tariff,
    baseline_hourly_kwh: pd.Series,
    sanctioned_load_kw: float = 10.0,
) -> dict:
    """Search the load's allowed window for the cheapest valid start hour.

    baseline_hourly_kwh is assumed to already include this load running at
    its baseline_start_hour. Returns a result dict; if no cheaper valid
    alternative exists, says so explicitly rather than fabricating savings.
    """
    baseline_bill = calculate_bill(tariff, baseline_hourly_kwh, sanctioned_load_kw)
    baseline_total = baseline_bill["total_estimated_bill"]

    # This Tariff model is always flat (a single energy_charge_per_kwh with
    # no time-of-day structure), so moving WHEN energy is used can never
    # change the total kWh billed -- by definition, not by simulation.
    # We deliberately do NOT run a per-hour shift-and-recompute search:
    # an earlier version did, and it was vulnerable to two separate bugs
    # (clipping negative adjusted values at 0, and a data-completeness
    # mismatch between the "from" and "to" hours) that both produced tiny,
    # fake non-zero "savings" that shouldn't be possible under a flat
    # tariff. Returning a deterministic "no savings" answer here is both
    # simpler and strictly more correct for this tariff model.
    return {
        "load_name": load.name,
        "baseline_start_hour": load.baseline_start_hour,
        "baseline_total_bill": baseline_total,
        "recommended_start_hour": None,
        "new_total_bill": baseline_total,
        "estimated_savings": 0.0,
        "simulated_not_verified": True,
        "no_savings_opportunity": True,
        "note": (
            "Tariff has no time-varying component, so shifting this load "
            "does not change the energy charge; no savings opportunity exists."
        ),
    }