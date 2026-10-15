"""Hand-verified test for the bill calculator."""

import pandas as pd

from energywise.tariffs.bill_calculator import Tariff, calculate_bill


def test_calculate_bill_known_case():
    tariff = Tariff(
        fixed_charge_per_kw_month=200.0,
        energy_charge_per_kwh=12.5,
        electricity_duty_pct=5.0,
    )
    hourly = pd.Series([10.0] * 10)  # 100 kWh total
    result = calculate_bill(tariff, hourly, sanctioned_load_kw=5.0)

    # Hand-calculated expectation:
    # energy = 100 * 12.5 = 1250
    # fixed = 5 * 200.0 = 1000
    # duty = (1250 + 1000) * 0.05 = 112.5
    # total = 1250 + 1000 + 112.5 = 2362.5
    assert result["total_kwh"] == 100.0
    assert result["energy_charge"] == 1250.0
    assert result["fixed_charge"] == 1000.0
    assert result["electricity_duty"] == 112.5
    assert result["total_estimated_bill"] == 2362.5
    assert result["is_estimate"] is True