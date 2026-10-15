"""Tariff and bill-calculation logic for EnergyWise.

Kept fully separate from the ML/forecasting code — no imports from
energywise.models or energywise.evaluation. A tariff bug should never be
able to affect a forecast, and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class Tariff:
    """A simple flat-rate commercial electricity tariff.

    NOTE: rates approximate 2026 Delhi LT commercial pricing based on
    secondary sources (not a primary DERC tariff order). The energy charge
    below (Rs 12.5/kWh) is the conservative end of a Rs 12.5-15.6/unit
    *effective* commercial rate range reported for 2026, which already
    folds in the Power Purchase Adjustment Cost (PPAC) surcharge rather
    than itemizing it separately (PPAC varies monthly and by DISCOM).
    TODO: replace with a verified, dated tariff before using this for any
    real cost claims.
    """

    name: str = "Approximate Delhi LT Commercial, 2026 (unverified)"
    fixed_charge_per_kw_month: float = 200.0
    energy_charge_per_kwh: float = 12.5
    electricity_duty_pct: float = 5.0
    effective_date: date = date(2026, 1, 1)
    source_note: str = (
        "Approximated from secondary sources (not a primary DERC/BSES "
        "tariff order), reflecting ~2026 Delhi commercial pricing. The "
        "energy charge already includes an estimated PPAC surcharge "
        "folded in, rather than itemized separately, since PPAC varies "
        "monthly and by DISCOM. Flat rate used since LT commercial "
        "connections typically do not carry time-of-day pricing in "
        "Delhi; that is usually reserved for larger HT/industrial "
        "consumers. Verify before making real cost claims."
    )

def calculate_bill(
    tariff: Tariff,
    hourly_energy_kwh: pd.Series,
    sanctioned_load_kw: float,
) -> dict:
    """Return an ESTIMATED bill breakdown for a period of hourly energy use.

    This is a simulation/estimate, not a verified bill.
    """
    total_kwh = float(hourly_energy_kwh.sum())

    energy_charge = total_kwh * tariff.energy_charge_per_kwh
    fixed_charge = sanctioned_load_kw * tariff.fixed_charge_per_kw_month
    duty = (energy_charge + fixed_charge) * (tariff.electricity_duty_pct / 100.0)
    total = energy_charge + fixed_charge + duty

    return {
        "tariff_name": tariff.name,
        "total_kwh": total_kwh,
        "energy_charge": round(energy_charge, 2),
        "fixed_charge": round(fixed_charge, 2),
        "electricity_duty": round(duty, 2),
        "total_estimated_bill": round(total, 2),
        "is_estimate": True,
        "source_note": tariff.source_note,
    }