"""Close-of-session features. No backfills, centered windows or future observations."""

import numpy as np
import pandas as pd

MODEL_FEATURES = ["log_return", "volatility_21", "momentum_5", "momentum_21", "rsi_scaled", "volume_change"]


def engineer_features(history, annual_sessions=252):
    frame = history.sort_values("trading_date").copy().reset_index(drop=True)
    for name in ("close", "adjusted_close", "open", "high", "low", "volume", "value", "nav"):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    # Never splice adjusted and unadjusted levels; a partially adjusted series is rejected.
    use_adjusted = "adjusted_close" in frame and frame.adjusted_close.notna().any()
    if use_adjusted and frame.adjusted_close.isna().any():
        raise ValueError("Adjusted price history is incomplete; provide a consistent series")
    price = frame.adjusted_close if use_adjusted else frame.close
    if price.isna().any() or (price <= 0).any():
        raise ValueError("Historical prices must be positive, finite, and complete")
    frame["model_price"] = price
    frame["price_basis"] = "adjusted" if use_adjusted else "unadjusted"
    returns = price.pct_change(fill_method=None)
    frame["simple_return"] = returns
    frame["log_return"] = np.log(price).diff()
    for window in (5, 21):
        frame[f"return_{window}"] = price.pct_change(window, fill_method=None)
        frame[f"momentum_{window}"] = frame[f"return_{window}"]
        frame[f"mean_{window}"] = returns.rolling(window).mean()
        frame[f"std_{window}"] = returns.rolling(window).std()
        frame[f"volatility_{window}"] = frame.log_return.rolling(window).std() * np.sqrt(annual_sessions)
        frame[f"sma_{window}"] = price.rolling(window).mean()
        frame[f"ema_{window}"] = price.ewm(span=window, adjust=False).mean()
    frame["historical_volatility"] = frame.log_return.expanding(min_periods=21).std() * np.sqrt(
        annual_sessions
    )
    frame["ewm_volatility"] = frame.log_return.ewm(span=21, adjust=False).std() * np.sqrt(annual_sessions)
    delta = price.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame["rsi"] = (100 - 100 / (1 + up / down)).where(down.ne(0), 100).where((up + down).ne(0), 50)
    frame["rsi_scaled"] = frame.rsi / 100
    frame["macd"] = price.ewm(span=12, adjust=False).mean() - price.ewm(span=26, adjust=False).mean()
    frame["macd_signal"] = frame.macd.ewm(span=9, adjust=False).mean()
    frame["bollinger_upper"] = frame.sma_21 + 2 * price.rolling(21).std()
    frame["bollinger_lower"] = frame.sma_21 - 2 * price.rolling(21).std()
    if {"high", "low"} <= set(frame):
        # ATR is in actual tradable price units, consistent with the OHLC inputs.
        tr = pd.concat(
            [
                frame.high - frame.low,
                (frame.high - frame.close.shift()).abs(),
                (frame.low - frame.close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        frame["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    volume = frame.get("volume", pd.Series(np.nan, index=frame.index))
    frame["volume_change"] = np.log1p(volume).diff()
    frame["volume_mean"] = volume.rolling(21).mean()
    frame["obv"] = (np.sign(price.diff()).fillna(0) * volume).cumsum()
    frame["drawdown"] = price / price.cummax() - 1
    frame["rolling_drawdown"] = price / price.rolling(63, min_periods=1).max() - 1
    frame["skewness"] = returns.rolling(63).skew()
    frame["kurtosis"] = returns.rolling(63).kurt()
    frame["autocorrelation"] = returns.rolling(63).apply(lambda x: pd.Series(x).autocorr(1), raw=True)
    if "value" in frame:
        frame["amihud_illiquidity"] = returns.abs() / frame.value.replace(0, np.nan)
    if "nav" in frame:
        frame["nav_change"] = frame.nav.pct_change(fill_method=None)
        frame["nav_premium_pct"] = 100 * (frame.close / frame.nav - 1)
    return frame.replace([np.inf, -np.inf], np.nan)
