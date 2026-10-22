"""Collect daily Iranian leveraged ETF snapshots and related listed options.

User-supplied BrsApi collector, integrated without changing its public workflow.
Requires BRS_API_KEY. Safe to import; it never schedules itself.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://Api.BrsApi.ir/Tsetmc"
DEFAULT_TICKERS = ("اهرم", "توان", "موج", "شتاب", "جهش", "بیدار", "نارنج اهرم", "دوایکس", "پیشران")
ID_FIELDS = {"id", "base_id", "isin", "code_12", "code_5", "code_4", "insCode"}
SHEET_NAMES = (
    "Funds",
    "Details",
    "NAV",
    "Options",
    "PriceHistory",
    "ClientHistory",
    "Assemblies",
    "Metadata",
    "Errors",
)


class MarketDataError(RuntimeError):
    """API or data validation failure; never includes an API key."""


def normalize_persian(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).translate(
        str.maketrans(
            {
                "ي": "ی",
                "ك": "ک",
                "ى": "ی",
                "\u200c": " ",
                **{chr(0x06F0 + i): str(i) for i in range(10)},
                **{chr(0x0660 + i): str(i) for i in range(10)},
            }
        )
    )
    return " ".join(s.split()).strip()


def as_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(normalize_persian(value).replace(",", "").replace("−", "-"))
    except (TypeError, ValueError):
        return None


def records(payload: Any, endpoint: str) -> list[dict[str, Any]]:
    for _ in range(5):
        if isinstance(payload, list):
            if not all(isinstance(row, dict) for row in payload):
                raise MarketDataError(f"{endpoint}: list does not contain records")
            return payload
        if isinstance(payload, dict):
            for key in ("data", "result", "items", "rows", "symbols", "options", "response"):
                if key in payload and isinstance(payload[key], (dict, list)):
                    payload = payload[key]
                    break
            else:
                if any(k in payload for k in ("l18", "date", "predtran", "psubtran", "pc", "id")):
                    return [payload]
                raise MarketDataError(f"{endpoint}: unexpected JSON object")
        else:
            raise MarketDataError(f"{endpoint}: JSON must be an object or array")
    raise MarketDataError(f"{endpoint}: too many nested envelopes")


def normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in row.items():
        if key in ID_FIELDS or key.endswith("_isin"):
            output[key] = None if value is None else str(value)
        elif isinstance(value, (list, dict)):
            output[key] = json.dumps(value, ensure_ascii=False, default=str)
        else:
            output[key] = value
    return output


@dataclass
class CollectionResult:
    file_path: Path
    sheets: dict[str, pd.DataFrame]
    errors: list[dict[str, str]]


class LeveragedFundsCollector:
    def __init__(
        self,
        api_key=None,
        output_dir="data/leveraged",
        tickers=DEFAULT_TICKERS,
        discover_new_funds=True,
        request_interval=0.4,
        session=None,
    ):
        self.api_key = api_key or os.environ.get("BRS_API_KEY", "")
        if not self.api_key:
            raise ValueError("Set BRS_API_KEY environment variable or pass api_key")
        self.output_dir = Path(output_dir)
        self.known_tickers = {normalize_persian(t) for t in tickers}
        self.discover_new_funds = discover_new_funds
        self.request_interval = max(0.35, request_interval)
        self._previous_request_start = 0.0
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
            }
        )

    def _get(self, endpoint, **params):
        for attempt in range(3):
            remaining = self.request_interval - (time.monotonic() - self._previous_request_start)
            if remaining > 0:
                time.sleep(remaining)
            self._previous_request_start = time.monotonic()
            try:
                response = self.session.get(
                    f"{BASE_URL}/{endpoint}", params={"key": self.api_key, **params}, timeout=(10, 45), allow_redirects=False
                )
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(1.5 * 2**attempt)
                    continue
                response.raise_for_status()
                if 300 <= response.status_code < 400:
                    raise MarketDataError(f"{endpoint}: redirect rejected")
                if "text/html" in getattr(response, "headers", {}).get("Content-Type", "").lower():
                    raise MarketDataError(f"{endpoint}: HTML error/block page rejected")
                return records(response.json(), endpoint)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                raise MarketDataError(f"{endpoint}: HTTP {status}") from None
            except requests.RequestException as exc:
                if attempt < 2:
                    time.sleep(1.5 * 2**attempt)
                    continue
                raise MarketDataError(f"{endpoint}: network failure ({type(exc).__name__})") from None
            except ValueError:
                raise MarketDataError(f"{endpoint}: invalid JSON response") from None
        raise MarketDataError(f"{endpoint}: retries exhausted")

    def _select_funds(self, all_symbols):
        selection = []
        for row in all_symbols:
            symbol, name = normalize_persian(row.get("l18")), normalize_persian(row.get("l30"))
            sector = normalize_persian(row.get("cs"))
            identified = self.discover_new_funds and ("صندوق" in sector or "صندوق" in name) and "اهرم" in name
            if symbol and (symbol in self.known_tickers or identified):
                copy = normalized_row(row)
                copy["symbol"] = symbol
                selection.append(copy)
        return sorted({row["symbol"]: row for row in selection}.values(), key=lambda x: x["symbol"])

    @staticmethod
    def _frame(rows):
        return pd.DataFrame.from_records(rows)

    @staticmethod
    def _excel_value(value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, default=str)
        if isinstance(value, int) and abs(value) > 999_999_999_999_999:
            return str(value)
        return value

    def _save_excel(self, path, frames):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.stem}.tmp.xlsx")
        try:
            with pd.ExcelWriter(
                temp_path,
                engine="xlsxwriter",
                engine_kwargs={
                    "options": {
                        "strings_to_formulas": False,
                        "strings_to_urls": False,
                    }
                },
            ) as writer:
                wb = writer.book
                header_fmt = wb.add_format(
                    {
                        "bold": True,
                        "bg_color": "#17365D",
                        "font_color": "#FFFFFF",
                        "border": 0,
                        "valign": "vcenter",
                    }
                )
                numeric_fmt = wb.add_format({"num_format": "#,##0.00"})
                integer_fmt = wb.add_format({"num_format": "#,##0"})
                for sheet_name in SHEET_NAMES:
                    df = frames.get(sheet_name, pd.DataFrame()).copy()
                    if df.empty and not len(df.columns):
                        df = pd.DataFrame(columns=["no_records"])
                    for col in df.columns:
                        df[col] = df[col].map(self._excel_value)
                        if col in ID_FIELDS or col.endswith("_isin"):
                            df[col] = df[col].map(lambda x: None if x is None else str(x))
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    ws = writer.sheets[sheet_name]
                    ws.freeze_panes(1, 0)
                    ws.set_row(0, 24, header_fmt)
                    for number, col in enumerate(df.columns):
                        lengths = df[col].dropna().astype(str).head(500).map(len)
                        width = min(
                            42,
                            max(13, len(str(col)) + 2, int(lengths.max()) + 2 if not lengths.empty else 13),
                        )
                        fmt = (
                            integer_fmt
                            if col in {"tvol", "tval", "tno", "interest_open"}
                            else (numeric_fmt if col in {"pl", "pc", "py", "pmin", "pmax"} else None)
                        )
                        ws.set_column(number, number, width, fmt)
                    if len(df):
                        ws.autofilter(0, 0, len(df), max(0, len(df.columns) - 1))
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def collect_daily(self, include_history=False, fetch_options=True):
        now = datetime.now(ZoneInfo("Asia/Tehran"))
        timestamp, snapshot_date = now.isoformat(timespec="seconds"), now.date().isoformat()
        notes, errors = [], []
        sheets_data = {n: [] for n in SHEET_NAMES}

        def success(name, symbol, count):
            notes.append({"endpoint": name, "symbol": symbol, "status": "ok", "rows": count})

        def failure(name, symbol, exc):
            msg = str(exc)
            LOGGER.warning("%s (%s): %s", name, symbol, msg)
            errors.append({"endpoint": name, "symbol": symbol, "error": msg})
            notes.append({"endpoint": name, "symbol": symbol, "status": "error", "rows": 0})

        universe = self._get("AllSymbols.php", type=1)
        if not universe:
            raise MarketDataError("AllSymbols.php returned zero rows; no snapshot was saved")
        success("AllSymbols", "*", len(universe))
        funds = self._select_funds(universe)
        if not funds:
            raise MarketDataError("No leveraged ETFs matched the configured tickers/name rule")
        present = {r["symbol"] for r in funds}
        for expected in sorted(self.known_tickers - present):
            notes.append(
                {"endpoint": "universe", "symbol": expected, "status": "missing_from_AllSymbols", "rows": 0}
            )
        for fund in funds:
            symbol = fund["symbol"]
            fund.update(
                {
                    "snapshot_ts": timestamp,
                    "snapshot_date": snapshot_date,
                    "market_date": None,
                    "source": "BrsApi/AllSymbols",
                }
            )
            try:
                detail = self._get("Symbol.php", l18=symbol)
                success("Symbol", symbol, len(detail))
                for row in detail:
                    assembly = row.get("assembly", [])
                    if isinstance(assembly, list):
                        for meeting in assembly:
                            if isinstance(meeting, dict):
                                sheets_data["Assemblies"].append(
                                    {"symbol": symbol, "snapshot_ts": timestamp, **normalized_row(meeting)}
                                )
                    flat = normalized_row({k: v for k, v in row.items() if k != "assembly"})
                    flat.update({"symbol": symbol, "snapshot_ts": timestamp})
                    sheets_data["Details"].append(flat)
                    market_date = row.get("date_update") or row.get("date")
                    if market_date:
                        fund["market_date"] = str(market_date)
            except MarketDataError as exc:
                failure("Symbol", symbol, exc)
            try:
                nav_rows = self._get("Nav.php", l18=symbol)
                success("NAV", symbol, len(nav_rows))
                for nav in nav_rows:
                    row = normalized_row(nav)
                    row.update({"symbol": symbol, "snapshot_ts": timestamp})
                    nav_value, price = as_number(nav.get("predtran")), as_number(fund.get("pc"))
                    nav_date, market_date = str(nav.get("date") or ""), str(fund.get("market_date") or "")
                    dates_match = nav_date == market_date if nav_date and market_date else None
                    row["nav_date_matches_market_date"] = dates_match
                    row["premium_to_redemption_nav_pct"] = (
                        round((price / nav_value - 1) * 100, 6)
                        if price is not None
                        and nav_value is not None
                        and nav_value > 0
                        and dates_match is not False
                        else None
                    )
                    sheets_data["NAV"].append(row)
            except MarketDataError as exc:
                failure("NAV", symbol, exc)
            if include_history:
                for api_type, sheet, label in (
                    (0, "PriceHistory", "History/price"),
                    (1, "ClientHistory", "History/client"),
                ):
                    try:
                        history = self._get("History.php", type=api_type, l18=symbol)
                        success(label, symbol, len(history))
                        for row in history:
                            sheets_data[sheet].append(
                                {"symbol": symbol, "snapshot_ts": timestamp, **normalized_row(row)}
                            )
                    except MarketDataError as exc:
                        failure(label, symbol, exc)
            sheets_data["Funds"].append(fund)
        if fetch_options:
            try:
                options = self._get("Option.php")
                success("Option", "*", len(options))
                base_ids = {str(f["id"]) for f in funds if f.get("id") is not None}
                for option in options:
                    underlying = normalize_persian(option.get("base_l18"))
                    if underlying in present or str(option.get("base_id")) in base_ids:
                        sheets_data["Options"].append(
                            {
                                **normalized_row(option),
                                "underlying_symbol": underlying,
                                "snapshot_ts": timestamp,
                                "snapshot_date": snapshot_date,
                            }
                        )
            except MarketDataError as exc:
                failure("Option", "*", exc)
        sheets_data["Metadata"] = [
            {"key": "provider", "value": "BrsApi / TSETMC"},
            {"key": "snapshot_timestamp_tehran", "value": timestamp},
            {"key": "snapshot_date_tehran", "value": snapshot_date},
            {"key": "fund_count", "value": len(funds)},
            {"key": "option_count", "value": len(sheets_data["Options"])},
            {"key": "historical_backfill", "value": include_history},
            {"key": "notes", "value": "Prices are normally IRR; verify endpoint units."},
            {
                "key": "freshness",
                "value": "market_date is from Symbol.php, not assumed from collection time.",
            },
        ] + notes
        sheets_data["Errors"] = errors
        frames = {name: self._frame(rows) for name, rows in sheets_data.items()}
        path = self.output_dir / f"leveraged_market_{snapshot_date}.xlsx"
        self._save_excel(path, frames)
        return CollectionResult(file_path=path, sheets=frames, errors=errors)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = LeveragedFundsCollector().collect_daily(
        include_history=os.getenv("BRS_INCLUDE_HISTORY", "0") == "1"
    )
    print("Saved:", report.file_path)
    print("Funds:", len(report.sheets["Funds"]))
    print("Options:", len(report.sheets["Options"]))
    print("Errors:", len(report.errors))
