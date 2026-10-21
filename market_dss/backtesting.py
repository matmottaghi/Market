"""Non-overlapping, point-in-time option backtests using archived executable quotes."""

from dataclasses import dataclass
from datetime import datetime, time

import numpy as np
import pandas as pd

from .calendar import TEHRAN, parse_date
from .options import contracts_from_frame
from .storage import atomic_bytes, excel_bytes


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    skipped: pd.DataFrame
    metrics: dict

    def save(self, path):
        atomic_bytes(
            path,
            excel_bytes(
                {
                    "trades": self.trades,
                    "equity": self.equity,
                    "skipped": self.skipped,
                    "metrics": pd.DataFrame([self.metrics]),
                }
            ),
        )


def available_at(frame, day):
    if frame.empty or not {"available_at", "trading_date"} <= set(frame):
        return frame.iloc[:0].copy()
    cutoff = pd.Timestamp(datetime.combine(parse_date(day), time.max, TEHRAN)).tz_convert("UTC")
    dates = pd.to_datetime(frame.trading_date, errors="coerce")
    available = pd.to_datetime(frame.available_at, errors="coerce", utc=True)
    return frame[(dates <= pd.Timestamp(parse_date(day))) & (available <= cutoff)].copy()


def execution_time(row, day):
    """A source-verified executable quote time, not the later download time."""
    if pd.isna(row.get("execution_at")):
        raise ValueError(
            "Verified execution_at timestamp missing; daily snapshots alone cannot establish a fill"
        )
    timestamp = pd.Timestamp(row["execution_at"])
    if timestamp.tzinfo is None or timestamp.tz_convert("Asia/Tehran").date() != parse_date(day):
        raise ValueError("Execution timestamp needs an explicit timezone and matching trading date")
    return timestamp.tz_convert("UTC")


def performance_metrics(trades, equity, initial_capital):
    if trades.empty:
        return {"status": "no_eligible_trades", "trades": 0}
    pnl = trades.realized_pnl.to_numpy()
    wealth = np.r_[initial_capital, equity.equity.to_numpy()]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    return {
        "status": "out_of_sample",
        "trades": len(pnl),
        "cumulative_return_pct": (wealth[-1] / initial_capital - 1) * 100,
        "max_drawdown_pct": float(drawdown.min() * 100),
        "win_rate": float((pnl > 0).mean()),
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else None,
        "equity_basis": "realized P&L at exits; intratrade drawdowns are not measured",
    }


def backtest_options(
    prices,
    option_history,
    forecast_fn,
    select_strategy,
    initial_capital=100_000_000.0,
    holding_sessions=5,
    step=5,
    slippage_pct=0.0,
    start=None,
    price_versions=None,
):
    if initial_capital <= 0 or holding_sessions < 1 or step < 1 or not 0 <= slippage_pct < 1:
        raise ValueError("Invalid backtest settings")
    prices = prices.sort_values("trading_date").reset_index(drop=True)
    if prices.empty or prices.instrument_id.nunique() != 1 or prices.source.nunique() != 1:
        raise ValueError("Backtest needs one underlying and one price source")
    vintages = prices if price_versions is None else price_versions
    vintages = vintages[
        vintages.instrument_id.eq(prices.instrument_id.iloc[0]) & vintages.source.eq(prices.source.iloc[0])
    ]
    if not option_history.empty:
        if not {"underlying_id", "source"} <= set(option_history):
            raise ValueError("Historical option rows need underlying and source identifiers")
        option_history = option_history[
            option_history.underlying_id.eq(prices.instrument_id.iloc[0])
            & option_history.source.eq(prices.source.iloc[0])
        ]
    balance, last_exit, trades, equity, skipped = initial_capital, None, [], [], []
    for index in range(0, len(prices) - holding_sessions, step):
        day = parse_date(prices.trading_date.iloc[index])
        if (start and day < parse_date(start)) or (last_exit and day <= last_exit) or balance <= 0:
            continue
        known_prices = (
            available_at(vintages, day)
            .sort_values("retrieved_at")
            .drop_duplicates("trading_date", keep="last")
            .sort_values("trading_date")
        )
        chain = available_at(option_history, day)
        if not chain.empty:
            chain = (
                chain[chain.trading_date.eq(str(day))]
                .sort_values("retrieved_at")
                .drop_duplicates("instrument_id", keep="last")
            )
        if chain.empty or known_prices.empty or parse_date(known_prices.trading_date.iloc[-1]) != day:
            skipped.append(
                {"date": str(day), "reason": "No point-in-time price/option chain with same-day quotes"}
            )
            continue
        try:
            if "execution_at" not in chain or chain.execution_at.isna().all():
                raise ValueError(
                    "Verified execution_at missing; after-close signals cannot assume earlier same-day fills"
                )
            # Callbacks receive ONLY information known at this date; no exit prices/chains.
            forecast = forecast_fn(known_prices.copy())
            if parse_date(forecast.metadata["input_data_cutoff"]) != day:
                raise ValueError("Forecast cutoff differs from backtest date")
            strategy, expected = select_strategy(known_prices.copy(), chain.copy(), forecast, balance)
            if strategy.scenario_only:
                raise ValueError("Hypothetical strategies cannot enter historical results")
            if strategy.underlying_shares:
                raise ValueError(
                    "Historical stock hedge fills are unverified; stock-covered backtests are disabled"
                )
            if strategy.capital is None or strategy.capital <= 0 or strategy.capital > balance:
                raise ValueError("Verified capital unavailable or outside budget")
            entry_contracts, _ = contracts_from_frame(chain)
            entry_map = {c.instrument_id: c for c in entry_contracts}
            information_time = pd.to_datetime(known_prices.available_at, utc=True).max()
            chain_information_time = pd.to_datetime(chain.available_at, utc=True).max()
            entry_times = []
            for leg in strategy.legs:
                observed = entry_map.get(leg.contract.instrument_id)
                if observed is None:
                    raise ValueError("Strategy contract absent from historical chain")
                observed_row = chain[chain.instrument_id.eq(leg.contract.instrument_id)].iloc[-1]
                entry_time = execution_time(observed_row, day)
                if entry_time <= information_time or entry_time < chain_information_time:
                    raise ValueError("Execution quote predates information used by the signal/selection")
                entry_times.append(entry_time)
                premium = observed.execution_premium(leg.side, leg.contracts, slippage_pct, day)
                if not np.isclose(premium, leg.premium):
                    raise ValueError("Strategy premium differs from historical executable quote")
                if (observed.strike, observed.expiration, observed.multiplier, observed.option_type) != (
                    leg.contract.strike,
                    leg.contract.expiration,
                    leg.contract.multiplier,
                    leg.contract.option_type,
                ):
                    raise ValueError("Strategy terms differ from archived contract")
                if observed.exercise_style != "european" or observed.settlement != "cash":
                    raise ValueError("Historical backtest supports verified European cash settlement only")
            if len(set(entry_times)) != 1:
                raise ValueError("Multi-leg historical fills require a synchronized executable snapshot")
            exit_index = index + holding_sessions
            exit_day = parse_date(prices.trading_date.iloc[exit_index])
            # Do not fabricate settlement from a later spot or today's chain.
            if strategy.expiration < exit_day:
                matches = prices.index[prices.trading_date.eq(str(strategy.expiration))]
                if not len(matches) or matches[0] <= index:
                    raise ValueError("Expiration settlement observation absent")
                exit_index, exit_day = int(matches[0]), strategy.expiration
            if expected.get("target_date") and parse_date(expected["target_date"]) != exit_day:
                raise ValueError(
                    "Forecast target and realized exit dates differ; missing session or early expiry"
                )
            exit_observations = available_at(vintages, exit_day)
            exit_observations = exit_observations[
                exit_observations.trading_date.eq(str(exit_day))
            ].sort_values("retrieved_at")
            if exit_observations.empty:
                raise ValueError("No point-in-time exit underlying observation")
            exit_record = exit_observations.iloc[-1]
            exit_spot = float(exit_record.close)
            if exit_day == strategy.expiration:
                # Need an explicit exchange settlement price; closing price may differ.
                if "settlement_price" not in exit_record or pd.isna(exit_record.settlement_price):
                    raise ValueError("Verified exchange settlement price absent")
                pnl = float(strategy.payoff(float(exit_record.settlement_price)))
            else:
                exits = available_at(option_history, exit_day)
                exits = (
                    exits[exits.trading_date.eq(str(exit_day))]
                    .sort_values("retrieved_at")
                    .drop_duplicates("instrument_id", keep="last")
                )
                parsed, _ = contracts_from_frame(exits)
                lookup = {contract.instrument_id: contract for contract in parsed}
                pnl = (
                    strategy.underlying_shares * (exit_spot - strategy.underlying_entry_price)
                    - strategy.stock_fees
                    - strategy.stock_exit_fees
                )
                for leg in strategy.legs:
                    if leg.contract.instrument_id not in lookup:
                        raise ValueError("Missing historical liquidation quote")
                    execution_time(
                        exits[exits.instrument_id.eq(leg.contract.instrument_id)].iloc[-1], exit_day
                    )
                    exit_premium = lookup[leg.contract.instrument_id].execution_premium(
                        -leg.side, leg.contracts, slippage_pct, exit_day
                    )
                    pnl += leg.side * leg.units * (exit_premium - leg.premium) - leg.contracts * (
                        leg.entry_fee + leg.exit_fee
                    )
            balance += pnl
            trades.append(
                {
                    "entry_date": str(day),
                    "entry_timestamp": entry_times[0].isoformat(),
                    "exit_date": str(exit_day),
                    "strategy": strategy.name,
                    "capital": strategy.capital,
                    "expected_pnl": expected.get("expected_pnl"),
                    "expected_return_pct": expected.get("expected_return_pct"),
                    "realized_pnl": pnl,
                    "realized_return_pct": 100 * pnl / strategy.capital,
                    "capital_breach": -pnl > strategy.capital,
                    "model_version": forecast.metadata["version"],
                    "split": "out_of_sample",
                }
            )
            equity.append({"date": str(exit_day), "equity": balance})
            last_exit = exit_day
        except ValueError as exc:
            skipped.append({"date": str(day), "reason": str(exc)[:300]})
    trades, equity, skipped = pd.DataFrame(trades), pd.DataFrame(equity), pd.DataFrame(skipped)
    return BacktestResult(trades, equity, skipped, performance_metrics(trades, equity, initial_capital))
