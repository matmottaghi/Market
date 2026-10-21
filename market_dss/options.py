"""Contract-aware option valuation and common-expiration multi-leg strategies."""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import brentq
from scipy.stats import norm

from .calendar import parse_date


class UnsupportedContract(ValueError):
    pass


class OptionContract(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="ignore")
    instrument_id: str
    symbol: str
    underlying_id: str
    option_type: Literal["call", "put"]
    strike: float = Field(gt=0)
    expiration: date
    multiplier: float = Field(gt=0)
    exercise_style: Literal["european", "american", "unknown"] = "unknown"
    settlement: Literal["cash", "physical", "unknown"] = "unknown"
    terms_verified: bool = False
    terms_source: str = ""
    bid: float | None = Field(default=None, ge=0)
    ask: float | None = Field(default=None, ge=0)
    bid_quantity: float | None = Field(default=None, ge=0)
    ask_quantity: float | None = Field(default=None, ge=0)
    last: float | None = Field(default=None, ge=0)
    volume: float | None = Field(default=None, ge=0)
    open_interest: float | None = Field(default=None, ge=0)
    trading_date: date | None = None

    @model_validator(mode="after")
    def uncrossed(self):
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValueError("Crossed bid/ask quotes")
        return self

    def require_terms(self):
        if (
            not self.terms_verified
            or not self.terms_source
            or "unknown" in (self.exercise_style, self.settlement)
        ):
            raise UnsupportedContract(f"{self.symbol}: contract terms need a verified source")

    def execution_premium(self, side, contracts=1, slippage_pct=0, on_date=None):
        self.require_terms()
        if side not in (-1, 1) or contracts < 1 or slippage_pct < 0 or slippage_pct >= 1:
            raise ValueError("Invalid side, quantity or slippage")
        if self.trading_date is None:
            raise ValueError(f"{self.symbol}: quote trading date is unknown")
        if on_date is not None and self.trading_date != parse_date(on_date):
            raise ValueError(f"{self.symbol}: quote date does not match valuation date")
        premium = self.ask if side == 1 else self.bid
        depth = self.ask_quantity if side == 1 else self.bid_quantity
        if premium is None or premium <= 0:
            raise ValueError(f"{self.symbol}: executable {'ask' if side == 1 else 'bid'} unavailable")
        if depth is None or depth < contracts:
            raise ValueError(f"{self.symbol}: verified quote depth insufficient/unknown")
        return premium * (1 + side * slippage_pct)


def contracts_from_frame(frame):
    valid, issues = [], []
    for _, row in frame.iterrows():
        try:
            values = {key: value for key, value in row.items() if pd.notna(value)}
            valid.append(OptionContract.model_validate(values))
        except ValueError as exc:
            issues.append({"symbol": row.get("symbol"), "issue": str(exc)[:300]})
    return valid, issues


def intrinsic(spot, strike, kind):
    if kind not in ("call", "put"):
        raise ValueError("Unknown option type")
    spot = np.asarray(spot, dtype=float)
    if not np.isfinite(spot).all() or (spot < 0).any() or strike <= 0:
        raise ValueError("Spot must be nonnegative and strike positive")
    return np.maximum(spot - strike, 0) if kind == "call" else np.maximum(strike - spot, 0)


def bsm(spot, strike, years, volatility, rate, kind, dividend_yield=0.0):
    """European theoretical value; continuous rates, ACT/365, constant vol/yield."""
    spot = np.asarray(spot, dtype=float)
    if years < 0 or volatility < 0 or not np.isfinite([years, volatility, rate, dividend_yield]).all():
        raise ValueError("Invalid pricing assumptions")
    intrinsic(spot, strike, kind)
    if years == 0:
        return intrinsic(spot, strike, kind)
    discounted_spot, discounted_strike = (
        spot * np.exp(-dividend_yield * years),
        strike * np.exp(-rate * years),
    )
    if volatility == 0:
        return (
            np.maximum(discounted_spot - discounted_strike, 0)
            if kind == "call"
            else np.maximum(discounted_strike - discounted_spot, 0)
        )
    safe_spot = np.maximum(spot, np.finfo(float).tiny)
    d1 = (np.log(safe_spot / strike) + (rate - dividend_yield + volatility**2 / 2) * years) / (
        volatility * np.sqrt(years)
    )
    d2 = d1 - volatility * np.sqrt(years)
    return (
        discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
        if kind == "call"
        else discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)
    )


def implied_volatility(contract, premium, spot, valuation_date, rate=0.0, dividend_yield=0.0):
    contract.require_terms()
    if contract.exercise_style != "european":
        raise UnsupportedContract("BSM implied volatility requires European exercise")
    years = (contract.expiration - parse_date(valuation_date)).days / 365
    if years <= 0 or premium <= 0:
        raise ValueError("IV requires positive time and premium")
    lower = float(bsm(spot, contract.strike, years, 0, rate, contract.option_type, dividend_yield))
    upper = (
        spot * np.exp(-dividend_yield * years)
        if contract.option_type == "call"
        else contract.strike * np.exp(-rate * years)
    )
    if not lower < premium < upper:
        raise ValueError("Premium outside strict no-arbitrage bounds; IV unavailable")
    try:
        return brentq(
            lambda vol: (
                float(bsm(spot, contract.strike, years, vol, rate, contract.option_type, dividend_yield))
                - premium
            ),
            1e-6,
            10,
        )
    except ValueError:
        raise ValueError("IV not bracketed within supported volatility range") from None


@dataclass
class OptionLeg:
    contract: OptionContract
    side: int
    premium: float
    contracts: int = 1
    entry_fee: float = 0.0  # Rial per contract, not per unit.
    exit_fee: float = 0.0

    def __post_init__(self):
        if self.side not in (-1, 1) or not isinstance(self.contracts, int) or self.contracts <= 0:
            raise ValueError("Side must be +1/-1 and contracts a positive integer")
        if (
            min(self.premium, self.entry_fee, self.exit_fee) < 0
            or not np.isfinite([self.premium, self.entry_fee, self.exit_fee]).all()
        ):
            raise ValueError("Premium and fees must be finite and nonnegative")

    @property
    def units(self):
        return self.contracts * self.contract.multiplier

    def payoff(self, spots):
        return self.side * self.units * (
            intrinsic(spots, self.contract.strike, self.contract.option_type) - self.premium
        ) - self.contracts * (self.entry_fee + self.exit_fee)


@dataclass
class Strategy:
    name: str
    legs: list[OptionLeg]
    underlying_shares: float = 0.0
    underlying_entry_price: float = 0.0
    stock_fees: float = 0.0  # Stock entry fees in rial.
    committed_capital: float | None = None
    capital_source: str = ""
    scenario_only: bool = False
    notes: list[str] = field(default_factory=list)
    stock_exit_fees: float = 0.0

    def __post_init__(self):
        if not self.legs:
            raise ValueError("Strategy requires at least one option leg")
        if len({leg.contract.expiration for leg in self.legs}) != 1:
            raise UnsupportedContract(
                "Mixed expirations require path-specific valuation and are not supported"
            )
        if len({leg.contract.underlying_id for leg in self.legs}) != 1:
            raise UnsupportedContract("Only a single underlying is supported")
        if (
            min(self.underlying_shares, self.underlying_entry_price, self.stock_fees, self.stock_exit_fees)
            < 0
        ):
            raise ValueError("Negative stock exposure/fees unsupported")
        if not np.isfinite(
            [self.underlying_shares, self.underlying_entry_price, self.stock_fees, self.stock_exit_fees]
        ).all():
            raise ValueError("Nonfinite stock exposure")
        if self.underlying_shares > 0 and self.underlying_entry_price <= 0:
            raise ValueError("Covered stock position needs an actual entry price")
        if self.committed_capital is not None and (
            not np.isfinite(self.committed_capital) or self.committed_capital <= 0 or not self.capital_source
        ):
            raise ValueError("Committed capital needs a positive amount and verification/source")
        if self.committed_capital is not None and all(leg.side == 1 for leg in self.legs):
            paid = (
                self.underlying_shares * self.underlying_entry_price
                + sum(leg.units * leg.premium + leg.contracts * leg.entry_fee for leg in self.legs)
                + self.stock_fees
            )
            if self.committed_capital < paid:
                raise ValueError(
                    "Long-position capital cannot be less than premium/stock paid plus entry fees"
                )

    @property
    def expiration(self):
        return self.legs[0].contract.expiration

    @property
    def initial_net_premium(self):
        """Positive means cash received; excludes fees and stock."""
        return -sum(leg.side * leg.units * leg.premium for leg in self.legs)

    @property
    def transaction_costs(self):
        return (
            self.stock_fees
            + self.stock_exit_fees
            + sum(leg.contracts * (leg.entry_fee + leg.exit_fee) for leg in self.legs)
        )

    @property
    def initial_cash_flow(self):
        return (
            self.initial_net_premium
            - self.underlying_shares * self.underlying_entry_price
            - sum(leg.contracts * leg.entry_fee for leg in self.legs)
            - self.stock_fees
        )

    @property
    def capital(self):
        if self.committed_capital is not None:
            return self.committed_capital
        if all(leg.side == 1 for leg in self.legs):
            return (
                self.underlying_shares * self.underlying_entry_price
                + sum(leg.units * leg.premium + leg.contracts * leg.entry_fee for leg in self.legs)
                + self.stock_fees
            )
        shorts = [leg for leg in self.legs if leg.side == -1]
        if all(leg.contract.option_type == "call" for leg in shorts) and self.underlying_shares >= sum(
            leg.units for leg in shorts
        ):
            return (
                self.underlying_shares * self.underlying_entry_price
                + sum(leg.units * leg.premium for leg in self.legs if leg.side == 1)
                + self.stock_fees
                + sum(leg.contracts * leg.entry_fee for leg in self.legs)
            )
        return None  # Net credit is never a capital denominator; margin must be supplied.

    def payoff(self, spots):
        spots = np.asarray(spots, dtype=float)
        return (
            sum(leg.payoff(spots) for leg in self.legs)
            + self.underlying_shares * (spots - self.underlying_entry_price)
            - self.stock_fees
            - self.stock_exit_fees
        )

    def theoretical(self):
        knots = np.array(sorted({0.0, *(leg.contract.strike for leg in self.legs)}))
        values = self.payoff(knots)
        right_slope = self.underlying_shares + sum(
            leg.side * leg.units for leg in self.legs if leg.contract.option_type == "call"
        )
        max_profit = float("inf") if right_slope > 0 else float(values.max())
        max_loss = float("inf") if right_slope < 0 else max(0.0, float(-values.min()))
        roots, intervals = [], []
        for a, b, ya, yb in zip(knots[:-1], knots[1:], values[:-1], values[1:]):
            if abs(ya) < 1e-8:
                roots.append(float(a))
            if abs(ya) < 1e-8 and abs(yb) < 1e-8:
                intervals.append([float(a), float(b)])
            elif ya * yb < 0:
                roots.append(float(a - ya * (b - a) / (yb - ya)))
        if abs(values[-1]) < 1e-8:
            roots.append(float(knots[-1]))
        if right_slope:
            root = knots[-1] - values[-1] / right_slope
            if root > knots[-1]:
                roots.append(float(root))
        elif abs(values[-1]) < 1e-8:
            intervals.append([float(knots[-1]), float("inf")])
        return {
            "max_profit": max_profit,
            "max_loss": max_loss,
            "break_even": sorted(set(roots)),
            "break_even_intervals": intervals,
            "risk_reward_ratio": max_profit / max_loss
            if np.isfinite(max_profit) and np.isfinite(max_loss) and max_loss > 0 and max_profit > 0
            else None,
        }

    def mark_to_market(
        self, spots, on_date, volatility, rate=0.0, dividend_yield=0.0, liquidation_haircut=0.0
    ):
        on_date = parse_date(on_date)
        if on_date > self.expiration:
            raise ValueError("Cannot value after expiration using this payoff model")
        if on_date == self.expiration:
            return self.payoff(spots)
        if not 0 <= liquidation_haircut < 1:
            raise ValueError("Liquidation haircut must be in [0,1)")
        years = (self.expiration - on_date).days / 365
        result = (
            self.underlying_shares * (np.asarray(spots) - self.underlying_entry_price)
            - self.stock_fees
            - self.stock_exit_fees
        )
        for leg in self.legs:
            leg.contract.require_terms()
            if leg.contract.exercise_style != "european":
                raise UnsupportedContract(
                    "Pre-expiration American valuation/early assignment is not implemented"
                )
            value = bsm(
                spots, leg.contract.strike, years, volatility, rate, leg.contract.option_type, dividend_yield
            )
            value = value * (1 - leg.side * liquidation_haircut)
            result = (
                result
                + leg.side * leg.units * (value - leg.premium)
                - leg.contracts * (leg.entry_fee + leg.exit_fee)
            )
        return result


# Leg specifications refer to contracts sorted by strike, supplied by the user.
TEMPLATES = {
    "Long Call": [("call", 1, 0, 1)],
    "Long Put": [("put", 1, 0, 1)],
    "Covered Call": [("call", -1, 0, 1)],
    "Protective Put": [("put", 1, 0, 1)],
    "Bull Call Spread": [("call", 1, 0, 1), ("call", -1, 1, 1)],
    "Bear Put Spread": [("put", -1, 0, 1), ("put", 1, 1, 1)],
    "Bull Put Spread": [("put", 1, 0, 1), ("put", -1, 1, 1)],
    "Bear Call Spread": [("call", -1, 0, 1), ("call", 1, 1, 1)],
    "Long Straddle": [("call", 1, 0, 1), ("put", 1, 0, 1)],
    "Short Straddle": [("call", -1, 0, 1), ("put", -1, 0, 1)],
    "Long Strangle": [("put", 1, 0, 1), ("call", 1, 0, 1)],
    "Short Strangle": [("put", -1, 0, 1), ("call", -1, 0, 1)],
    "Butterfly Spread": [("call", 1, 0, 1), ("call", -1, 1, 2), ("call", 1, 2, 1)],
    "Iron Condor": [("put", 1, 0, 1), ("put", -1, 1, 1), ("call", -1, 0, 1), ("call", 1, 1, 1)],
    "Collar": [("put", 1, 0, 1), ("call", -1, 0, 1)],
}


def build_strategy(
    name,
    contracts,
    spot,
    on_date,
    quantity=1,
    entry_fee=0.0,
    exit_fee=0.0,
    slippage_pct=0.0,
    committed_capital=None,
    capital_source="",
):
    specification = TEMPLATES[name]
    ordered = {
        kind: sorted([c for c in contracts if c.option_type == kind], key=lambda c: c.strike)
        for kind in ("call", "put")
    }
    legs = []
    for kind, side, index, count in specification:
        if index >= len(ordered[kind]):
            raise ValueError(f"{name}: insufficient {kind} contracts")
        contract = ordered[kind][index]
        if contract.expiration <= parse_date(on_date):
            raise ValueError("Expired or expiration-day opening trades disabled")
        premium = contract.execution_premium(side, quantity * count, slippage_pct, on_date)
        legs.append(OptionLeg(contract, side, premium, quantity * count, entry_fee, exit_fee))
    strikes = [leg.contract.strike for leg in legs]
    if len({leg.contract.multiplier for leg in legs}) > 1:
        raise ValueError("Template requires equal multipliers; use custom quantities for unequal contracts")
    if "Spread" in name and len(set(strikes)) != len(strikes):
        raise ValueError("Spread strikes must be distinct")
    if "Straddle" in name and strikes[0] != strikes[1]:
        raise ValueError("Straddle strikes must be equal")
    if ("Strangle" in name or name == "Collar") and strikes[0] >= strikes[1]:
        raise ValueError("Put strike must be below call strike")
    if name == "Butterfly Spread" and not np.isclose(strikes[1] - strikes[0], strikes[2] - strikes[1]):
        raise ValueError("Butterfly requires equally spaced strikes")
    if name == "Iron Condor" and not strikes[0] < strikes[1] < strikes[2] < strikes[3]:
        raise ValueError("Iron condor requires ordered nonoverlapping wings")
    shares = (
        quantity * legs[0].contract.multiplier if name in ("Covered Call", "Protective Put", "Collar") else 0
    )
    return Strategy(
        name, legs, shares, spot, committed_capital=committed_capital, capital_source=capital_source
    )
