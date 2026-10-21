from datetime import timedelta

import numpy as np
import pytest

from market_dss.options import (
    OptionLeg,
    Strategy,
    UnsupportedContract,
    bsm,
    build_strategy,
    implied_volatility,
)
from market_dss.risk import rank_strategies, scenario_metrics


@pytest.mark.parametrize("kind,expected", [("call", [-50, -50, 150]), ("put", [150, -50, -50])])
def test_long_option_payoff(contract_factory, kind, expected):
    leg = OptionLeg(contract_factory(kind), 1, 5)
    np.testing.assert_allclose(leg.payoff([80, 100, 120]), expected)


@pytest.mark.parametrize(
    "name,kinds,expected",
    [
        ("Bull Call Spread", ["call", "call"], [-10, 40, 90]),
        ("Bear Call Spread", ["call", "call"], [-10, -60, -110]),
        ("Bull Put Spread", ["put", "put"], [-110, -60, -10]),
        ("Bear Put Spread", ["put", "put"], [90, 40, -10]),
    ],
)
def test_spreads_execution_prices(contract_factory, name, kinds, expected):
    contracts = [contract_factory(kind, strike) for kind, strike in zip(kinds, [100, 110])]
    strategy = build_strategy(name, contracts, 100, "2026-10-21")
    np.testing.assert_allclose(strategy.payoff([90, 105, 120]), expected)
    assert strategy.capital is None  # No invented broker margin for credit or debit spreads.


def test_multipliers_fees_and_finite_expected_return(contract_factory):
    strategy = Strategy(
        "custom",
        [
            OptionLeg(contract_factory(multiplier=10), 1, 5, contracts=2, entry_fee=3, exit_fee=2),
            OptionLeg(contract_factory(strike=110, multiplier=20), -1, 2, contracts=1, entry_fee=1),
        ],
    )
    np.testing.assert_allclose(strategy.payoff([90, 120]), [-71, 129])
    result = scenario_metrics(strategy.payoff([90, 120]), capital=200, probabilities=[0.75, 0.25])
    assert result["expected_pnl"] == pytest.approx(-21)
    assert result["expected_return_pct"] == pytest.approx(-10.5)
    assert result["probability_profit"] == 0.25


def test_var_cvar_convention():
    result = scenario_metrics([-100, -20, 0, 50], confidence=0.75)
    assert result["var_loss"] == 20
    assert result["cvar_loss"] == 100
    assert result["expected_return_pct"] is None
    fractional = scenario_metrics([-100, -20, 0, 50], confidence=0.6)
    assert fractional["cvar_loss"] == pytest.approx((0.25 * 100 + 0.15 * 20) / 0.4)
    assert scenario_metrics([5, 10])["cvar_loss"] == 0


def test_missing_quotes_and_terms(contract_factory):
    with pytest.raises(ValueError, match="ask"):
        contract_factory(ask=None).execution_premium(1)
    with pytest.raises(UnsupportedContract):
        contract_factory(exercise_style="unknown").execution_premium(1)
    with pytest.raises(ValueError, match="unknown"):
        contract_factory(trading_date=None).execution_premium(1)
    with pytest.raises(ValueError, match="depth"):
        contract_factory(ask_quantity=None).execution_premium(1)


def test_expiration_consistency_and_bounds(contract_factory):
    call = contract_factory()
    with pytest.raises(UnsupportedContract):
        Strategy(
            "bad",
            [
                OptionLeg(call, 1, 5),
                OptionLeg(contract_factory(expiration=call.expiration + timedelta(days=1)), 1, 5),
            ],
        )
    long = Strategy("call", [OptionLeg(call, 1, 5)])
    assert long.theoretical()["max_loss"] == 50
    assert np.isinf(long.theoretical()["max_profit"])
    assert long.theoretical()["break_even"] == [105]
    short = Strategy("short", [OptionLeg(call, -1, 5)])
    assert np.isinf(short.theoretical()["max_loss"])
    assert short.capital is None


def test_bsm_parity_iv_and_before_expiration(contract_factory):
    call, put = bsm(100, 100, 1, 0.2, 0.05, "call"), bsm(100, 100, 1, 0.2, 0.05, "put")
    assert call - put == pytest.approx(100 - 100 * np.exp(-0.05))
    c = contract_factory()
    years = (c.expiration - c.trading_date).days / 365
    premium = float(bsm(100, 100, years, 0.3, 0.1, "call"))
    assert implied_volatility(c, premium, 100, c.trading_date, 0.1) == pytest.approx(0.3)
    strategy = Strategy("call", [OptionLeg(c, 1, 5)])
    assert strategy.mark_to_market(100, c.trading_date, 0.3) > strategy.payoff(100)
    with pytest.raises(ValueError):
        strategy.mark_to_market(100, c.expiration + timedelta(days=1), 0.3)


def test_ranking_budget_and_pareto():
    rows = [
        {
            "strategy": "a",
            "expected_return_pct": 10,
            "probability_profit": 0.6,
            "cvar_loss": 20,
            "capital": 100,
        },
        {
            "strategy": "b",
            "expected_return_pct": 5,
            "probability_profit": 0.5,
            "cvar_loss": 30,
            "capital": 100,
        },
        {
            "strategy": "c",
            "expected_return_pct": 99,
            "probability_profit": 0.9,
            "cvar_loss": 5,
            "capital": None,
        },
    ]
    result = rank_strategies(rows, capital_budget=100)
    assert result.strategy.tolist() == ["a", "b"]
    assert result.pareto.tolist() == [True, False]


def test_stock_exit_cost_not_in_initial_cashflow(contract_factory):
    strategy = Strategy(
        "covered",
        [OptionLeg(contract_factory(), -1, 5, entry_fee=2, exit_fee=3)],
        underlying_shares=10,
        underlying_entry_price=100,
        stock_fees=7,
        stock_exit_fees=11,
    )
    assert strategy.initial_cash_flow == -959
    assert strategy.capital == 1009
    assert strategy.transaction_costs == 23
    assert strategy.payoff(100) == 27
