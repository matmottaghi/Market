"""All fixtures here are synthetic, never loaded by the live application."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from market_dss.calendar import TradingCalendar
from market_dss.options import OptionContract


@pytest.fixture
def synthetic_history():
    rng = np.random.default_rng(70)
    dates = TradingCalendar().next_sessions("2023-01-01", 420)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(dates))))
    return pd.DataFrame(
        {
            "instrument_id": "synthetic-underlying",
            "symbol": "TEST_ONLY",
            "source": "test_fixture",
            "trading_date": [str(d) for d in dates],
            "available_at": [f"{d}T13:00:00+00:00" for d in dates],
            "retrieved_at": [f"{d}T13:00:00+00:00" for d in dates],
            "close": close,
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "volume": rng.integers(100, 10000, len(close)),
        }
    )


@pytest.fixture
def contract_factory():
    def make(kind="call", strike=100, multiplier=10, **kwargs):
        values = dict(
            instrument_id=f"synthetic-{kind}-{strike}",
            symbol=f"TEST_{kind}_{strike}",
            underlying_id="synthetic-underlying",
            option_type=kind,
            strike=strike,
            expiration=date(2026, 12, 1),
            multiplier=multiplier,
            exercise_style="european",
            settlement="cash",
            terms_verified=True,
            terms_source="synthetic test terms",
            bid=4.0,
            ask=5.0,
            last=4.5,
            bid_quantity=1000,
            ask_quantity=1000,
            trading_date=date(2026, 10, 21),
        )
        values.update(kwargs)
        return OptionContract(**values)

    return make
