"""Public TSETMC history: no authentication, strict identity and response validation."""
import json
import logging
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from filelock import FileLock

from .calendar import normalize_text, parse_date, utc_timestamp
from .ingestion import ImportBatch, PriceObservation
from .storage import atomic_bytes

HISTORICAL_API_BASE_URL = "https://cdn.tsetmc.com/api"
HISTORICAL_API_KEY = None
HISTORICAL_API_AUTH_REQUIRED = False
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
           "Accept": "application/json"}


class HistoricalAPIError(RuntimeError):
    """Contains only safe endpoint/error categories, never response bodies or credentials."""


class TsetmcProvider:
    name = "TSETMC"

    def __init__(self, data_dir, session=None, timeout=(8, 30), attempts=3, interval=.4):
        self.data_dir = Path(data_dir)
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)
        self.timeout, self.attempts, self.interval = timeout, attempts, interval
        self.previous = 0.0

    def request(self, path, key, allow_empty=False):
        for attempt in range(self.attempts):
            delay = self.interval - (time.monotonic() - self.previous)
            if delay > 0:
                time.sleep(delay)
            self.previous = time.monotonic()
            try:
                response = self.session.get(f"{HISTORICAL_API_BASE_URL}/{path}",
                                            timeout=self.timeout, allow_redirects=False)
                code = response.status_code
                if code in (429, 500, 502, 503, 504) and attempt + 1 < self.attempts:
                    time.sleep(1.5 * 2 ** attempt)
                    continue
                if code != 200:
                    raise HistoricalAPIError(f"TSETMC HTTP {code}: {path.split('/')[1]}")
                if "json" not in response.headers.get("Content-Type", "").lower():
                    raise HistoricalAPIError("TSETMC non-JSON/HTML response; possible geographic block")
                try:
                    payload = response.json()
                except ValueError:
                    raise HistoricalAPIError("TSETMC invalid JSON response") from None
                if not isinstance(payload, dict) or key not in payload:
                    raise HistoricalAPIError(f"TSETMC schema changed: {key}")
                value = payload[key]
                if not isinstance(value, (list, dict)) or (not value and not allow_empty):
                    raise HistoricalAPIError(f"TSETMC empty/invalid dataset: {key}")
                logging.getLogger(__name__).info("historical_request endpoint=%s status=200", path.split('/')[1])
                return value
            except requests.RequestException as exc:
                if attempt + 1 == self.attempts:
                    raise HistoricalAPIError(f"TSETMC {type(exc).__name__}; check network/IP access") from None
                time.sleep(1.5 * 2 ** attempt)
        raise HistoricalAPIError("TSETMC retries exhausted")

    def resolve(self, symbol):
        symbol = normalize_text(symbol)
        cache_path = self.data_dir / "processed" / "symbol_mapping.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(cache_path) + ".lock"):
            cache = json.loads(cache_path.read_text("utf-8")) if cache_path.exists() else {}
            cached = cache.get(symbol)
            if cached and (symbol not in ("اهرم", "موج", "توان") or cached.get("instrument_type") == "leveraged_etf") and (pd.Timestamp(utc_timestamp()) - pd.Timestamp(cached["verified_at"])).days < 7:
                return cached
        rows = self.request(f"Instrument/GetInstrumentSearch/{quote(symbol, safe='')}", "instrumentSearch")
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise HistoricalAPIError("TSETMC invalid search records")
        matches = {str(r.get("insCode")): r for r in rows
                   if normalize_text(r.get("lVal18AFC")) == symbol and str(r.get("insCode", "")).isdigit()
                   and str(r.get("flow")) in ("1", "2", "4")}
        if len(matches) != 1:
            raise HistoricalAPIError(f"TSETMC exact symbol identity ambiguous or missing: {symbol}")
        identifier, found = next(iter(matches.items()))
        info = self.request(f"Instrument/GetInstrumentInfo/{identifier}", "instrumentInfo")
        if (not isinstance(info, dict) or str(info.get("insCode")) != identifier
                or normalize_text(info.get("lVal18AFC")) != symbol
                or not normalize_text(info.get("lVal30")) or info.get("flow") != found.get("flow")):
            raise HistoricalAPIError("TSETMC search/instrument identity conflict")
        description = normalize_text(" ".join(str(info.get(k) or "") for k in ("lVal30", "faraDesc", "cgrValCotTitle")))
        leveraged = "اهرم" in description and "صندوق" in description
        if symbol in ("اهرم", "موج", "توان") and not leveraged:
            raise HistoricalAPIError("Required symbol is not verified as a leveraged fund")
        result = {"instrument_type": "leveraged_etf" if leveraged else "unknown", "instrument_id": identifier, "symbol": symbol, "name": normalize_text(info["lVal30"]),
                  "market": str(info["flow"]), "isin": info.get("cIsin"), "verified_at": utc_timestamp()}
        with FileLock(str(cache_path) + ".lock"):
            cache = json.loads(cache_path.read_text("utf-8")) if cache_path.exists() else {}
            cache[symbol] = result
            atomic_bytes(cache_path, json.dumps(cache, ensure_ascii=False).encode("utf-8"))
        return result

    def fetch_symbol(self, symbol, latest_date=None):
        identity = self.resolve(symbol)
        identifier, stamp = identity["instrument_id"], utc_timestamp()
        rows = self.request(f"ClosingPrice/GetClosingPriceDailyList/{identifier}/0", "closingPriceDaily")
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise HistoricalAPIError("TSETMC invalid historical records")
        raw_path = self.data_dir / "raw" / f"tsetmc_{identifier}_{time.time_ns()}.json"
        atomic_bytes(raw_path, json.dumps(rows, ensure_ascii=False).encode("utf-8"))
        valid, issues = [], []
        for index, raw in enumerate(rows):
            try:
                if str(raw.get("insCode")) != identifier:
                    raise ValueError("Historical identifier mismatch")
                day = parse_date(raw.get("dEven"))
                if day > pd.Timestamp.now(tz="Asia/Tehran").date():
                    raise ValueError("Future trading date")
                record = {"instrument_id": identifier, "symbol": identity["symbol"],
                          "trading_date": str(day), "source": self.name, "retrieved_at": stamp,
                          "available_at": stamp, "currency": "IRR"}
                for original, canonical in {"priceFirst": "open", "priceMax": "high", "priceMin": "low",
                    "pClosing": "close", "pDrCotVal": "last", "priceYesterday": "previous_close",
                    "qTotTran5J": "volume", "qTotCap": "value", "zTotTran": "transactions"}.items():
                    value = raw.get(original)
                    record[canonical] = float(value) if value is not None else None
                for field in ("close", "open", "high", "low"):
                    if record.get(field) is not None and record[field] <= 0:
                        raise ValueError(f"Nonpositive {field}")
                PriceObservation.model_validate(record)
                valid.append(record)
            except (ValueError, TypeError) as exc:
                detail = str(exc).split("\n")[0][:150]
                issues.append({"row": index, "trading_date_raw": str(raw.get("dEven")), "issue": detail})
        if not valid:
            raise HistoricalAPIError("TSETMC no valid historical prices; previous history preserved")
        frame = pd.DataFrame(valid).sort_values("trading_date").drop_duplicates("trading_date", keep="last")
        # The API returns full history; reconcile revisions as well as append dates.
        issues.append({"issue": "Unadjusted prices; adjustment events are archived, never treated as an adjusted series."})
        try:
            events = self.request(f"ClosingPrice/GetPriceAdjustList/{identifier}", "priceAdjust", allow_empty=True)
            atomic_bytes(self.data_dir / "raw" / f"adjustments_{identifier}.json",
                         json.dumps(events, ensure_ascii=False).encode("utf-8"))
        except HistoricalAPIError as exc:
            issues.append({"issue": str(exc)})
        clients = []
        try:
            activity = self.request(f"ClientType/GetClientTypeHistory/{identifier}", "clientType", allow_empty=True)
            if not isinstance(activity, list):
                raise HistoricalAPIError("TSETMC client history schema changed")
            known_dates = set(frame.trading_date)
            for raw in activity:
                if not isinstance(raw, dict) or str(raw.get("insCode")) != identifier:
                    continue
                day = str(parse_date(raw.get("recDate")))
                if day in known_dates:
                    clients.append({**raw, "instrument_id": identifier, "symbol": identity["symbol"],
                                    "source": self.name, "trading_date": day,
                                    "retrieved_at": stamp, "available_at": stamp})
        except (HistoricalAPIError, ValueError, TypeError) as exc:
            issues.append({"issue": str(exc) if isinstance(exc, HistoricalAPIError) else "Invalid client history dates"})
        tables = {"prices": frame, "instruments": pd.DataFrame([{**identity, "source": self.name,
                    "retrieved_at": stamp}])}
        if clients:
            tables["clients"] = pd.DataFrame(clients)
        return ImportBatch(tables, issues, raw_path,
                           {"api_records": len(rows), "valid_records": len(frame),
                            "removed_records": sum("row" in issue for issue in issues),
                            "duplicate_api_dates": len(valid) - len(frame)})
