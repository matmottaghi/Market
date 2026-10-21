from types import SimpleNamespace

import numpy as np
import pandas as pd

from market_dss.backtesting import available_at, backtest_options
from market_dss.options import OptionLeg, Strategy, contracts_from_frame


def test_availability_is_not_trading_date(synthetic_history):
    history = synthetic_history.head(5).copy()
    history.loc[0, "available_at"] = "2099-01-01T00:00:00+00:00"
    known = available_at(history, history.trading_date.iloc[2])
    assert len(known) == 2
    assert history.trading_date.iloc[0] not in known.trading_date.tolist()


def test_missing_historical_chains_disable_backtest(synthetic_history):
    def forbidden(*args):
        raise AssertionError("Must not forecast or select without historical chains")

    result = backtest_options(synthetic_history.head(10), pd.DataFrame(), forbidden, forbidden)
    assert result.trades.empty
    assert result.metrics["status"] == "no_eligible_trades"
    assert not result.skipped.empty


def test_callbacks_receive_no_future_and_actual_exit_quotes(contract_factory, synthetic_history):
    prices = synthetic_history.head(6).copy()
    chain = []
    for index, day in enumerate(prices.trading_date):
        c = contract_factory(
            expiration=pd.Timestamp("2028-01-01").date(),
            trading_date=pd.Timestamp(day).date(),
            ask=5 + index,
            bid=4 + index,
        )
        chain.append(
            {
                **c.model_dump(mode="json"),
                "source": "test_fixture",
                "available_at": f"{day}T13:01:00Z",
                "retrieved_at": f"{day}T13:01:00Z",
                "execution_at": f"{day}T13:01:00Z",
            }
        )
    options = pd.DataFrame(chain)

    def forecast(known):
        cutoff = known.trading_date.iloc[-1]
        assert (known.trading_date <= cutoff).all()
        return SimpleNamespace(metadata={"input_data_cutoff": cutoff, "version": "synthetic_test_only"})

    def select(known, contracts, prediction, capital):
        assert contracts.trading_date.max() == known.trading_date.max()
        parsed, _ = contracts_from_frame(contracts)
        return Strategy("call", [OptionLeg(parsed[0], 1, parsed[0].ask)]), {"expected_pnl": 1}

    result = backtest_options(prices, options, forecast, select, holding_sessions=1, step=1)
    assert len(result.trades) == 3
    np.testing.assert_allclose(result.trades.realized_pnl, 0)
    # Earlier quotes cannot be filled using later same-day close information.
    options["execution_at"] = options.trading_date + "T12:00:00Z"
    rejected = backtest_options(prices, options, forecast, select, holding_sessions=1, step=1)
    assert rejected.trades.empty
    assert rejected.skipped.reason.str.contains("predates").all()
