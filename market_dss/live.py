"""Credential-safe live diagnostics and verified public quote fallback."""
from datetime import datetime, time as day_time
from time import monotonic

import math
import pandas as pd
import requests

from .calendar import TEHRAN, parse_date, utc_timestamp
from .historical import HEADERS, TsetmcProvider, HistoricalAPIError
from .ingestion import DataProvider, ImportBatch, load_historical_data


def quote_timestamp(row):
    for field in ("dEven", "date_update", "trading_date", "date", "market_date"):
        value = row.get(field)
        if not value:
            continue
        try:
            day = parse_date(value)
            clock = row.get("hEven") or row.get("time")
            text = str(clock or "").replace(":", "").split(".")[0].zfill(6)
            moment = day_time(int(text[:2]), int(text[2:4]), int(text[4:])) if clock else day_time(0)
            return datetime.combine(day, moment, TEHRAN).isoformat(), "timestamp" if clock else "date_only"
        except (ValueError, TypeError):
            continue
    return None, "unknown"


def freshness(timestamp, max_age_hours=48, now=None):
    if timestamp is None:
        return "UNKNOWN"
    age = (pd.Timestamp(now or utc_timestamp()) - pd.Timestamp(timestamp)).total_seconds() / 3600
    if age < -1:
        return "INVALID_FUTURE_TIMESTAMP"
    return "STALE" if age > max_age_hours else "DELAYED_REST"


def validate_public_quote(raw, identifier, provider):
    if not isinstance(raw, dict):
        raise ValueError("Public quote is not a record")
    reported = str(raw.get("insCode") or "0")
    if reported not in ("0", identifier):
        raise ValueError("Quote identity mismatch")
    close, last = float(raw["pClosing"]), float(raw["pDrCotVal"])
    if not math.isfinite(close) or close <= 0 or not math.isfinite(last) or last <= 0:
        raise ValueError("Nonpositive quote price")
    if reported == "0":
        # Observed API omits identity (insCode=0). Cross-check the dated daily record.
        data_dir = getattr(provider, "data_dir", None)
        if data_dir is not None:
            from .storage import ExcelRepository
            path = data_dir / "historical" / "market.xlsx"
            if path.exists():
                stored = load_historical_data(ExcelRepository(path), identifier, "TSETMC")
                if not stored.empty:
                    match = stored[stored.trading_date.eq(str(parse_date(raw.get("dEven"))))]
                    if not match.empty and abs(float(match.close.iloc[-1]) - close) <= .000001:
                        return "validated_request_path_and_stored_daily_record"
        history = provider.request(f"ClosingPrice/GetClosingPriceDailyList/{identifier}/0", "closingPriceDaily")
        matches = [r for r in history if str(r.get("insCode")) == identifier and r.get("dEven") == raw.get("dEven")]
        if not matches or abs(float(matches[0]["pClosing"]) - close) > .000001:
            raise ValueError("Quote with missing native ID does not match verified daily history")
        return "validated_request_path_and_daily_record"
    return "provider_instrument_id"


def diagnose_market_api(provider, credentials=None, symbols=None, session=None, max_age_hours=48, data_dir="data"):
    """Real requests; return safe metadata only, never response bodies or secret URLs."""
    symbols = symbols or ["اهرم", "موج", "توان"]
    session = session or requests.Session()
    session.headers.update(HEADERS)
    rows = []
    if provider == "BrsApi" and not credentials:
        return [{"provider": provider, "endpoint": "configuration", "status": "AUTHENTICATION_FAILED",
                 "error_category": "MISSING_CREDENTIALS", "safe_details": "کلید قابل استفاده در حساب فعلی موجود نیست.",
                 "retrieved_at": utc_timestamp()}]
    requests_to_make = []
    if provider == "BrsApi":
        requests_to_make = [("AllSymbols.php", {"type": 1}, ""), ("Option.php", {}, "")]
        for symbol in symbols:
            requests_to_make += [("Symbol.php", {"l18": symbol}, symbol), ("Nav.php", {"l18": symbol}, symbol)]
    elif provider == "TSETMC":
        lookup = TsetmcProvider(data_dir, session=session)
        for symbol in symbols:
            try:
                identity = lookup.resolve(symbol)
                requests_to_make.append((f"ClosingPrice/GetClosingPriceInfo/{identity['instrument_id']}", {}, symbol))
            except Exception as exc:
                rows.append({"provider": provider, "symbol": symbol, "endpoint": "identity",
                             "status": "NETWORK_ERROR", "error_category": type(exc).__name__,
                             "safe_details": "شناسایی نماد ناموفق؛ شبکه یا هویت نماد را بررسی کنید.", "retrieved_at": utc_timestamp()})
    else:
        raise ValueError("Unsupported diagnostic provider")
    for path, params, symbol in requests_to_make:
        started = monotonic()
        result = {"provider": provider, "endpoint": path.split("/")[-1] if provider == "BrsApi" else path.split("/")[1],
                  "symbol": symbol, "request_at": utc_timestamp(), "http_status": None,
                  "content_type": None, "record_count": 0, "provider_timestamp": None,
                  "freshness": "UNKNOWN", "validation_status": "FAILED"}
        try:
            base = "https://Api.BrsApi.ir/Tsetmc/" if provider == "BrsApi" else "https://cdn.tsetmc.com/api/"
            response = session.get(base + path, params={"key": credentials, **params} if provider == "BrsApi" else None,
                                   timeout=(8, 25), allow_redirects=False)
            code = response.status_code
            result.update(http_status=code, content_type=response.headers.get("Content-Type", ""))
            if code in (401, 403):
                result.update(status="AUTHENTICATION_FAILED" if provider == "BrsApi" else "GEO_BLOCKED",
                              safe_details="دسترسی رد شد؛ مجوز حساب یا محدودیت IP را بررسی کنید.")
            elif code == 429:
                result.update(status="RATE_LIMITED", safe_details="محدودیت تعداد درخواست؛ بعداً تلاش کنید.")
            elif code >= 500:
                result.update(status="PROVIDER_UNAVAILABLE", safe_details="سرویس ارائه‌دهنده در دسترس نیست.")
            elif code != 200:
                result.update(status="INVALID_RESPONSE", safe_details=f"HTTP {code}; request rejected; check endpoint parameters and account permissions")
            elif "json" not in result["content_type"].lower():
                result.update(status="INVALID_RESPONSE", safe_details="پاسخ JSON نیست؛ صفحه خطا یا محدودیت دسترسی محتمل است.")
            else:
                payload = response.json()
                if provider == "BrsApi":
                    from leveraged_funds_collector import records
                    data = records(payload, path)
                else:
                    value = payload.get("closingPriceInfo") if isinstance(payload, dict) else None
                    if isinstance(value, dict):
                        result["identity_validation"] = validate_public_quote(value, path.rsplit("/", 1)[-1], lookup)
                    data = [value] if isinstance(value, dict) else []
                if not data:
                    result.update(status="EMPTY_RESPONSE", safe_details="هیچ رکوردی دریافت نشد.")
                else:
                    required = {"AllSymbols.php": ("id", "l18"),
                                "Option.php": ("id", "base_id", "price_strike", "size_contract"),
                                "Nav.php": ("predtran",)}.get(path, ())
                    if not all(isinstance(r, dict) and all(k in r for k in required) for r in data):
                        raise ValueError("Response schema mismatch")
                    stamps = [quote_timestamp(r)[0] for r in data]
                    stamps = [stamp for stamp in stamps if stamp]
                    stamp = max(stamps) if stamps else None
                    age = freshness(stamp, max_age_hours)
                    result.update(record_count=len(data), provider_timestamp=stamp, freshness=age,
                                  validation_status="FAILED" if age == "INVALID_FUTURE_TIMESTAMP" else "VALID",
                                  status="INVALID_RESPONSE" if age == "INVALID_FUTURE_TIMESTAMP" else
                                      ("STALE_DATA" if age == "STALE" else "CONNECTED"),
                                  safe_details="زمان دریافت با زمان معامله متفاوت است؛ داده REST تضمین لحظه‌ای ندارد.")
        except requests.RequestException as exc:
            result.update(status="NETWORK_ERROR", error_category=type(exc).__name__,
                          safe_details="درخواست شبکه ناموفق؛ DNS، TLS، proxy و محدودیت IP را بررسی کنید.")
        except Exception as exc:
            result.update(status="INVALID_RESPONSE", error_category=type(exc).__name__,
                          safe_details="ساختار یا محتوای پاسخ معتبر نیست؛ HTTP 200 به‌تنهایی کافی نیست.")
        result.update(duration_seconds=round(monotonic() - started, 3), retrieved_at=utc_timestamp())
        result.setdefault("error_category", None if result["validation_status"] == "VALID" else result["status"])
        rows.append(result)
    return rows


class LiveDataError(HistoricalAPIError):
    pass


class TsetmcLiveProvider(DataProvider):
    name = "TSETMC"

    def __init__(self, data_dir, symbols, provider=None, max_age_hours=48):
        self.public = provider or TsetmcProvider(data_dir)
        self.symbols, self.max_age_hours = symbols, max_age_hours

    def fetch(self, include_history=False):
        snapshots, instruments, issues = [], [], []
        for symbol in self.symbols:
            try:
                identity = self.public.resolve(symbol)
                identifier = identity["instrument_id"]
                request_at = utc_timestamp()
                raw = self.public.request(f"ClosingPrice/GetClosingPriceInfo/{identifier}", "closingPriceInfo")
                identity_basis = validate_public_quote(raw, identifier, self.public)
                stamp, precision = quote_timestamp(raw)
                if stamp is None:
                    raise ValueError("Provider quote timestamp missing")
                close, last = float(raw["pClosing"]), float(raw["pDrCotVal"])
                if not math.isfinite(close) or close <= 0 or not math.isfinite(last) or last <= 0:
                    raise ValueError("Nonpositive quote price")
                state = freshness(stamp, self.max_age_hours)
                if state == "INVALID_FUTURE_TIMESTAMP":
                    raise ValueError("Future quote timestamp")
                row = {**identity, "source": "TSETMC", "retrieved_at": utc_timestamp(),
                       "request_at": request_at, "provider_quote_at": stamp, "timestamp_precision": precision,
                       "available_at": utc_timestamp(), "trading_date": str(parse_date(stamp)),
                       "quote_identity_basis": identity_basis, "provider_instrument_id": str(raw.get("insCode")),
                       "close": close, "last": last, "volume": raw.get("qTotTran5J"),
                       "value": raw.get("qTotCap"), "open": raw.get("priceFirst"), "high": raw.get("priceMax"),
                       "low": raw.get("priceMin"), "freshness": state}
                try:
                    book = self.public.request(f"BestLimits/{identifier}", "bestLimits", allow_empty=True)
                    best = next((r for r in book if r.get("number") == 1), None) if isinstance(book, list) else None
                    if best:
                        row.update(bid=best.get("pMeDem"), ask=best.get("pMeOf"),
                                   bid_quantity=best.get("qTitMeDem"), ask_quantity=best.get("qTitMeOf"),
                                   orderbook_time_verified=False)
                except Exception:
                    issues.append({"symbol": symbol, "issue": "Public order book unavailable"})
                snapshots.append(row)
                instruments.append({**identity, "source": self.name, "retrieved_at": row["retrieved_at"]})
                if state == "STALE":
                    issues.append({"symbol": symbol, "issue": "STALE_DATA", "provider_quote_at": stamp})
            except Exception as exc:
                issues.append({"symbol": symbol, "issue": str(exc) if isinstance(exc, (HistoricalAPIError, ValueError, KeyError)) else "Live quote unavailable",
                               "error_type": type(exc).__name__})
        if not snapshots:
            summary = "; ".join(f"{item.get('symbol')}: {item.get('issue')} ({item.get('error_type')})" for item in issues)
            raise LiveDataError("TSETMC live quotes unavailable: " + summary[:1800])
        return ImportBatch({"snapshots": pd.DataFrame(snapshots), "instruments": pd.DataFrame(instruments)}, issues)
