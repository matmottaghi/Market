# توضیح فایل: دریافت روزانه و معاملات تاریخی TSETMC و تجمیع معاملات به کندل‌های ساعتی.
# معرفی توابع و متدها:
# - _aggregate_hourly: حذف لغو و تکرار، رد تعارض شناسه و تجمیع معاملات یک نماد و روز.
# - get_hourly_day: حداکثر سه پاسخ همان endpoint را با کلید معامله ادغام و با حجم روزانه کنترل می‌کند.
# - _fa_normalize: یکسان‌سازی حروف فارسی و حذف فاصلهٔ اضافی برای تطبیق نماد.
# - __init__: مقداردهی اولیهٔ نمونه و وابستگی‌های آن.
# - _json: درخواست JSON با timeout، فاصلهٔ درخواست‌ها و بررسی ساختار پاسخ.
# - resolve_symbol_code: قبول فقط یک تطبیق دقیق نماد با شناسهٔ معتبر.
# - get_daily: دریافت همهٔ سطرهای روزانهٔ موجود در API برای نماد و تبدیل تاریخ.
# - get_hourly_day: تجمیع معاملات یک روز به OHLCV ساعتی و مقایسهٔ حجم با دادهٔ روزانه.
# - get_stock_history: رابط دریافت روزانه برای یک یا چند نماد.
# - get_hourly_history: رابط دریافت ساعتی یک مجموعهٔ روزانهٔ کوچک.
# - get_stock_datasets: بازگرداندن دادهٔ روزانه و ساعتی با رابط سازگار قبلی.

"""TSETMC daily and per-date intraday trade adapters (no API key)."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from urllib.parse import quote
import logging
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Settings

LOG = logging.getLogger(__name__)
HOURLY_COLUMNS = [
    "insCode", "Symbol", "Date", "Hour", "Open", "High", "Low",
    "Close", "Volume", "Trades", "VWAP", "DailyVolume",
    "VolumeCoverage", "DataQuality", "Source",
]


def _fa_normalize(value: str) -> str:
    return str(value).strip().replace("ي", "ی").replace("ك", "ک").replace("\u200c", "")


class TsetmcClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
        })
        retry = Retry(
            total=settings.retries, backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset({"GET"}),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self._last_call = None

    def _json(self, endpoint: str) -> dict:
        if self._last_call is not None:
            remaining = self.settings.delay - (time.monotonic() - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
        url = f"{self.settings.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.get(url, timeout=self.settings.timeout)
            response.raise_for_status()
            payload = response.json()
        finally:
            self._last_call = time.monotonic()
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object for {endpoint}")
        return payload

    def resolve_symbol_code(self, symbol: str) -> str:
        """Only accept an unambiguous exact ticker match. Review aliases manually."""
        payload = self._json(f"Instrument/GetInstrumentSearch/{quote(symbol, safe='')}")
        rows = payload.get("instrumentSearch")
        if not isinstance(rows, list):
            raise ValueError("instrumentSearch is absent or invalid")
        exact = [r for r in rows if _fa_normalize(r.get("lVal18AFC", "")) == _fa_normalize(symbol)]
        codes = {str(r["insCode"]) for r in exact if r.get("insCode") is not None}
        if len(codes) != 1:
            raise ValueError(f"Symbol {symbol!r}: expected one exact instrument, found {len(codes)}; inspect TSETMC search results")
        return codes.pop()

    def get_daily_since(self,symbol,ins_code,start):
        """Request only enough recent daily rows to cover the incremental boundary."""
        days=max(0,(pd.Timestamp.now(tz='Asia/Tehran').date()-pd.Timestamp(start).date()).days)
        frame=self.get_daily(symbol,ins_code,limit=max(30,days+15))
        return frame[frame.Date>=pd.Timestamp(start)].copy() if not frame.empty else frame

    def get_daily(self, symbol: str, ins_code: str, limit=0) -> pd.DataFrame:
        payload = self._json(f"ClosingPrice/GetClosingPriceDailyList/{ins_code}/{int(limit)}")
        rows = payload.get("closingPriceDaily")
        if not isinstance(rows, list):
            raise ValueError(f"Invalid closingPriceDaily for {symbol}")
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        if "dEven" not in df:
            raise ValueError("Daily response has no dEven")
        df["Date"] = pd.to_datetime(
            pd.to_numeric(df["dEven"], errors="coerce").astype("Int64").astype("string"),
            format="%Y%m%d", errors="coerce",
        )
        df["Symbol"] = symbol
        df["insCode"] = str(ins_code)
        df["Source"] = "tsetmc"
        return df

    def get_hourly_day(
        self,
        symbol: str,
        ins_code: str,
        date: pd.Timestamp,
        daily_volume: float | None,
    ) -> pd.DataFrame:
        """Aggregate one trading day's ticks; no invented candles for missing hours."""
        date = pd.Timestamp(date).normalize()
        ymd = date.strftime("%Y%m%d")
        parts = []
        best = pd.DataFrame(columns=HOURLY_COLUMNS)
        for attempt in range(3):
            try:
                payload = self._json(f"Trade/GetTradeHistory/{ins_code}/{ymd}/false")
            except requests.RequestException:
                if not best.empty:
                    return best
                raise
            rows = payload.get("tradeHistory")
            if not isinstance(rows, list):
                raise ValueError(f"Invalid tradeHistory for {symbol} {ymd}")
            # Some archive responses omit different trade IDs on successive calls.
            # Union only identified trades of this exact instrument-date endpoint.
            if rows and all(row.get('nTran') is not None for row in rows):
                parts.extend(rows)
            elif rows:
                parts = rows
            best = self._aggregate_hourly(symbol, ins_code, date, daily_volume, parts)
            best['ResponseParts'] = attempt + 1
            if not best.empty and best['DataQuality'].iloc[0] == 'matched':
                return best
        return best

    def _aggregate_hourly(self, symbol, ins_code, date, daily_volume, rows):
        """Validate and aggregate identified trades from one instrument-date only."""
        ymd = date.strftime("%Y%m%d")
        if not isinstance(rows, list):
            raise ValueError(f"Invalid tradeHistory for {symbol} {ymd}")
        if not rows:
            return pd.DataFrame(columns=HOURLY_COLUMNS)
        trades = pd.DataFrame(rows)
        required = {"hEven", "pTran", "qTitTran"}
        if not required.issubset(trades.columns):
            raise ValueError(f"Missing trade fields {sorted(required - set(trades.columns))}")
        # TSETMC can repeat identical trade sequences in a single response.
        # De-duplicate only with an explicit trade identity, never just price/time.
        for flag in ('canceled', 'iAnuTran'):
            if flag in trades:
                trades = trades[pd.to_numeric(trades[flag], errors='coerce').fillna(0) == 0]
        if 'nTran' in trades:
            trades = trades.drop_duplicates(['nTran', 'hEven', 'pTran', 'qTitTran'])
            if trades['nTran'].duplicated().any():
                raise ValueError('Conflicting records for the same trade sequence')
        time_vals = pd.to_numeric(trades["hEven"], errors="coerce").astype("Int64").astype("string").str.zfill(6)
        trades["Timestamp"] = pd.to_datetime(ymd + time_vals, format="%Y%m%d%H%M%S", errors="coerce")
        trades["Price"] = pd.to_numeric(trades["pTran"], errors="coerce")
        trades["Quantity"] = pd.to_numeric(trades["qTitTran"], errors="coerce")
        trades = trades.dropna(subset=["Timestamp", "Price", "Quantity"])
        trades = trades[(trades["Price"] > 0) & (trades["Quantity"] > 0)].copy()
        if trades.empty:
            return pd.DataFrame(columns=HOURLY_COLUMNS)
        if "nTran" in trades:
            trades["Sequence"] = pd.to_numeric(trades["nTran"], errors="coerce")
            trades = trades.sort_values(["Timestamp", "Sequence"], kind="stable")
        else:
            trades = trades.sort_values("Timestamp", kind="stable")
        trades["Hour"] = trades["Timestamp"].dt.floor("h")
        trades["Turnover"] = trades["Price"] * trades["Quantity"]
        hourly = trades.groupby("Hour", as_index=False).agg(
            Open=("Price", "first"), High=("Price", "max"),
            Low=("Price", "min"), Close=("Price", "last"),
            Volume=("Quantity", "sum"), Trades=("Price", "size"),
            Turnover=("Turnover", "sum"),
        )
        hourly["VWAP"] = hourly["Turnover"] / hourly["Volume"]
        hourly = hourly.drop(columns="Turnover")
        hourly["insCode"] = str(ins_code)
        hourly["Symbol"] = symbol
        hourly["Date"] = date
        daily_vol = pd.to_numeric(pd.Series([daily_volume]), errors="coerce").iloc[0]
        hourly["DailyVolume"] = daily_vol
        if pd.notna(daily_vol) and daily_vol > 0:
            coverage = float(hourly["Volume"].sum()) / float(daily_vol)
            hourly["VolumeCoverage"] = coverage
            hourly["DataQuality"] = (
                "matched" if abs(coverage - 1) <= self.settings.volume_tolerance else "volume_mismatch"
            )
        else:
            hourly["VolumeCoverage"] = float("nan")
            hourly["DataQuality"] = "unverified"
        hourly["Source"] = "tsetmc"
        return hourly[HOURLY_COLUMNS]


def get_stock_history(ins_codes: str | Sequence[str] | Mapping[str, str], *, client: TsetmcClient | None = None) -> pd.DataFrame:
    """Retained signature; calls the historical daily endpoint for each code."""
    client = client or TsetmcClient(Settings.from_env())
    if isinstance(ins_codes, str):
        entries = [(ins_codes, ins_codes)]
    elif isinstance(ins_codes, Mapping):
        entries = list(ins_codes.items())
    else:
        entries = [(str(code), str(code)) for code in ins_codes]
    parts = [client.get_daily(sym, str(code)) for sym, code in entries]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True)
    return (result.dropna(subset=["Date"])
            .drop_duplicates(["insCode", "Date"], keep="last")
            .sort_values(["insCode", "Date"]).reset_index(drop=True))


def get_hourly_history(
    daily_df: pd.DataFrame,
    *,
    client: TsetmcClient | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Compatibility helper: collect hourly history into one DataFrame.

    For large histories prefer `sync_market`, which incrementally persists
    each daily partition and doesn't keep all historical hourly rows in RAM.
    """
    from .cleaning import clean_stock_data
    client = client or TsetmcClient(Settings.from_env())
    if daily_df.empty:
        return pd.DataFrame(columns=HOURLY_COLUMNS)
    _, valid = clean_stock_data(daily_df)
    if start_date:
        valid = valid.loc[valid["Date"] >= pd.Timestamp(start_date)]
    if end_date:
        valid = valid.loc[valid["Date"] <= pd.Timestamp(end_date)]
    parts = []
    errors = []
    for row in valid[["Symbol", "insCode", "Date", "qTotTran5J"]].itertuples(index=False):
        try:
            part = client.get_hourly_day(row.Symbol, row.insCode, row.Date, row.qTotTran5J)
            if not part.empty:
                parts.append(part)
        except (requests.RequestException, ValueError, KeyError) as exc:
            errors.append((row.insCode, row.Date.strftime("%Y-%m-%d"), str(exc)))
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=HOURLY_COLUMNS)
    result.attrs["failed_requests"] = errors
    return result


def get_stock_datasets(
    ins_codes: str | Sequence[str] | Mapping[str, str],
    *,
    client: TsetmcClient | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience for small tests; do not use for full historical backfill."""
    client = client or TsetmcClient(Settings.from_env())
    daily = get_stock_history(ins_codes, client=client)
    hourly = get_hourly_history(daily, client=client)
    return daily, hourly
