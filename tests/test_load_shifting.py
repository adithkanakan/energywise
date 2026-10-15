"""Confirms the optimizer reports 'no savings' honestly on a flat tariff,
rather than fabricating a shift that doesn't actually save money.
"""

import pandas as pd

from energywise.optimization.load_shifting import FlexibleLoad, find_cheapest_window
from energywise.tariffs.bill_calculator import Tariff


def test_flat_tariff_reports_no_savings_opportunity():
    tariff = Tariff(
        fixed_charge_per_kw_month=200.0,
        energy_charge_per_kwh=8.0,  # flat rate, no time-of-day variation
        electricity_duty_pct=5.0,
    )

    load = FlexibleLoad(
        name="Test Appliance",
        power_kw=5.0,
        duration_hours=2,
        allowed_window_start_hour=0,
        allowed_window_end_hour=23,
        baseline_start_hour=10,
    )

    # One full day's baseline profile, hour-indexed 0-23, including the load.
    baseline = pd.Series(10.0, index=range(24))

    result = find_cheapest_window(load, tariff, baseline, sanctioned_load_kw=10.0)

    assert result["no_savings_opportunity"] is True
    assert result["estimated_savings"] == 0.0
    assert result["recommended_start_hour"] is None
    assert result["simulated_not_verified"] is True
def test_flat_tariff_no_fake_savings_with_realistic_dips():
    """Regression test: a day where real usage dips below the appliance's
    assumed draw must not produce fake savings on a flat tariff (this was
    caused by clipping the adjusted profile at 0, which silently deleted
    energy and made some candidate hours look cheaper than others).
    """
    tariff = Tariff(
        fixed_charge_per_kw_month=200.0,
        energy_charge_per_kwh=12.5,
        electricity_duty_pct=5.0,
    )

    load = FlexibleLoad(
        name="Test Appliance",
        power_kw=8.0,  # deliberately larger than some baseline hours below
        duration_hours=4,
        allowed_window_start_hour=0,
        allowed_window_end_hour=23,
        baseline_start_hour=11,
    )

    # Realistic dataset: 30 days, with usage sometimes dipping below 8 kWh
    # (the appliance's assumed draw) at various hours -- this is exactly
    # the condition that triggered the clipping bug.
    import numpy as np
    rng = np.random.default_rng(42)
    hours_index = pd.Index([h % 24 for h in range(30 * 24)])
    values = rng.uniform(low=2.0, high=15.0, size=30 * 24)  # some days < 8 kWh
    baseline = pd.Series(values, index=hours_index)

    result = find_cheapest_window(load, tariff, baseline, sanctioned_load_kw=10.0)

    assert result["no_savings_opportunity"] is True
    assert result["estimated_savings"] == 0.0
    assert result["recommended_start_hour"] is None