"""Physical-scenario investment P&L. VaR/CVaR are positive tail losses in rial."""

import numpy as np
import pandas as pd

from .calendar import parse_date


def weighted_quantile(values, weights, probability):
    order = np.argsort(values)
    return float(
        np.asarray(values)[order][
            min(np.searchsorted(np.cumsum(np.asarray(weights)[order]), probability), len(values) - 1)
        ]
    )


def scenario_metrics(
    pnl, capital=None, probabilities=None, confidence=0.95, target_return=10.0, loss_threshold=10.0
):
    pnl = np.asarray(pnl, dtype=float)
    if pnl.ndim != 1 or len(pnl) == 0 or not np.isfinite(pnl).all() or not 0 < confidence < 1:
        raise ValueError("Invalid P&L scenarios or confidence")
    weights = (
        np.full(len(pnl), 1 / len(pnl)) if probabilities is None else np.asarray(probabilities, dtype=float)
    )
    if (
        weights.shape != pnl.shape
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Scenario probabilities must be nonnegative and sum to one")
    losses = -pnl
    var = weighted_quantile(losses, weights, confidence)
    # Integrate exactly the worst (1-confidence) probability mass, including fractional boundary mass.
    order = np.argsort(losses)[::-1]
    remaining, tail_loss = 1 - confidence, 0.0
    for i in order:
        take = min(float(weights[i]), remaining)
        tail_loss += take * losses[i]
        remaining -= take
        if remaining <= 1e-12:
            break
    mean = float(weights @ pnl)
    result = {
        "expected_pnl": mean,
        "median_pnl": weighted_quantile(pnl, weights, 0.5),
        "probability_profit": float(weights @ (pnl > 0)),
        "probability_loss": float(weights @ (pnl < 0)),
        "expected_downside_loss": float(weights @ np.maximum(-pnl, 0)),
        "pnl_std": float(np.sqrt(weights @ (pnl - mean) ** 2)),
        "var_loss": max(0.0, var),
        "cvar_loss": max(0.0, tail_loss / (1 - confidence)),
        "capital": capital,
    }
    fields = [
        "expected_return_pct",
        "median_return_pct",
        "return_std_pct",
        "return_q05_pct",
        "return_q95_pct",
        "probability_target_return",
        "probability_loss_threshold",
    ]
    result.update({key: None for key in fields})
    if capital is not None and np.isfinite(capital) and capital > 0:
        returns = 100 * pnl / capital
        result.update(
            expected_return_pct=100 * mean / capital,
            median_return_pct=weighted_quantile(returns, weights, 0.5),
            return_std_pct=result["pnl_std"] * 100 / capital,
            return_q05_pct=weighted_quantile(returns, weights, 0.05),
            return_q95_pct=weighted_quantile(returns, weights, 0.95),
            probability_target_return=float(weights @ (returns > target_return)),
            probability_loss_threshold=float(weights @ (returns < -loss_threshold)),
        )
    return result


def evaluate_strategy(
    strategy,
    forecast,
    horizon,
    volatility=0.3,
    rate=0.0,
    dividend_yield=0.0,
    liquidation_haircut=0.0,
    target_return=10.0,
    loss_threshold=10.0,
):
    if not 1 <= horizon <= forecast.paths.shape[1]:
        raise ValueError("Requested horizon is not in the predictive distribution")
    if str(strategy.legs[0].contract.underlying_id) != str(forecast.metadata["instrument_id"]):
        raise ValueError("Forecast and strategy underlyings differ")
    cutoff = parse_date(forecast.metadata["input_data_cutoff"])
    for leg in strategy.legs:
        leg.contract.require_terms()
        if not strategy.scenario_only and leg.contract.trading_date != cutoff:
            raise ValueError("Option quote and forecast cutoff dates differ")
        if not strategy.scenario_only:
            executable = leg.contract.execution_premium(leg.side, leg.contracts, on_date=cutoff)
            if (leg.side == 1 and leg.premium < executable) or (leg.side == -1 and leg.premium > executable):
                raise ValueError("Entry premium is more favorable than the executable quote")
        if leg.side == -1 and leg.contract.exercise_style == "american":
            raise ValueError("Short American assignment risk unsupported")
    on_date = parse_date(forecast.dates[horizon - 1])
    pnl = strategy.mark_to_market(
        forecast.paths[:, horizon - 1], on_date, volatility, rate, dividend_yield, liquidation_haircut
    )
    result = scenario_metrics(
        pnl, strategy.capital, target_return=target_return, loss_threshold=loss_threshold
    )
    result.update(
        strategy=strategy.name,
        horizon=horizon,
        target_date=str(on_date),
        valuation="expiration" if on_date == strategy.expiration else "BSM mark-to-market scenario",
        scenario_only=strategy.scenario_only,
        max_theoretical_loss=strategy.theoretical()["max_loss"],
        model_version=forecast.metadata["version"],
        input_data_cutoff=str(cutoff),
    )
    return result, pnl


OBJECTIVES = {
    "expected_return_pct": False,
    "probability_profit": False,
    "expected_downside_loss": True,
    "cvar_loss": True,
    "expected_pnl": False,
    "weighted_score": False,
}


def rank_strategies(results, objective="expected_return_pct", capital_budget=None, weights=(0.4, 0.3, 0.3)):
    if objective not in OBJECTIVES:
        raise ValueError("Unknown ranking objective")
    frame = pd.DataFrame(results).copy()
    if frame.empty:
        return frame
    if "feasible" in frame:
        frame = frame[frame.feasible.eq(True)]
    if "scenario_only" in frame:
        frame = frame[~frame.scenario_only.eq(True)]
    if capital_budget is not None:
        frame = frame[frame.capital.notna() & (frame.capital > 0) & (frame.capital <= capital_budget)]
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return frame
    if min(weights) < 0 or not np.isclose(sum(weights), 1):
        raise ValueError("Ranking weights must be nonnegative and sum to one")

    def normalized(column):
        values = pd.to_numeric(frame[column], errors="coerce")
        spread = values.max() - values.min()
        return (values - values.min()) / spread if spread > 0 else values * 0

    frame["weighted_score"] = (
        weights[0] * normalized("expected_return_pct")
        + weights[1] * normalized("probability_profit")
        - weights[2] * normalized("cvar_loss")
    )
    # A point is dominated if another has no lower expected return and no higher tail loss.
    frame["pareto"] = False
    valid = frame.expected_return_pct.notna() & frame.cvar_loss.notna()
    for i in frame.index[valid]:
        other = frame.loc[valid]
        dominated = (
            (other.expected_return_pct >= frame.loc[i, "expected_return_pct"])
            & (other.cvar_loss <= frame.loc[i, "cvar_loss"])
            & (
                (other.expected_return_pct > frame.loc[i, "expected_return_pct"])
                | (other.cvar_loss < frame.loc[i, "cvar_loss"])
            )
        ).any()
        frame.loc[i, "pareto"] = not dominated
    return frame.sort_values(objective, ascending=OBJECTIVES[objective], na_position="last").reset_index(
        drop=True
    )
