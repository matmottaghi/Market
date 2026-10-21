"""Provider boundaries and adapter for the supplied collector workbook schema."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .calendar import normalize_text, parse_date, utc_timestamp
from .storage import ExcelRepository

LOGGER = logging.getLogger(__name__)


class PriceObservation(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    instrument_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    trading_date: str
    retrieved_at: str
    source: str
    close: float = Field(gt=0)
    open: float | None = Field(default=None, gt=0)
    high: float | None = Field(default=None, gt=0)
    low: float | None = Field(default=None, gt=0)
    volume: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_range(self):
        if self.high is not None and self.low is not None:
            if self.high < self.low or not self.low <= self.close <= self.high:
                raise ValueError("Invalid OHLC price range")
            if self.open is not None and not self.low <= self.open <= self.high:
                raise ValueError("Open outside daily range")
        parse_date(self.trading_date)
        utc_timestamp(self.retrieved_at)
        return self


@dataclass
class ImportBatch:
    tables: dict[str, pd.DataFrame]
    issues: list[dict] = field(default_factory=list)
    raw_path: Path | None = None


class DataProvider(ABC):
    name: str

    @abstractmethod
    def fetch(self, include_history=False) -> ImportBatch:
        """Return a validated batch; failure must not mutate the repository."""


PRICE_MAP = {
    "pc": "close",
    "pl": "last",
    "pf": "open",
    "pmin": "low",
    "pmax": "high",
    "tvol": "volume",
    "tval": "value",
    "tno": "transactions",
    "pcc": "price_change",
    "pd1": "bid",
    "po1": "ask",
    "qd1": "bid_quantity",
    "qo1": "ask_quantity",
}
OPTION_MAP = {
    "base_id": "underlying_id",
    "base_l18": "underlying_symbol",
    "type": "option_type",
    "price_strike": "strike",
    "date_end": "expiration",
    "size_contract": "multiplier",
    "interest_open": "open_interest",
}
NUMERIC = set(PRICE_MAP.values()) | {"adjusted_close", "nav", "strike", "multiplier", "open_interest"}


def clean_record(row):
    return {str(k): (None if pd.isna(v) else v) for k, v in row.items()}


class CollectorAdapter:
    """No inferred trading dates, IDs, contract styles, settlement or margins."""

    def __init__(self, source="BrsApi", column_mapping=None):
        self.source = source
        self.column_mapping = column_mapping or {}

    def inspect(self, path):
        return {
            name: {"rows": len(frame), "columns": frame.columns.tolist()}
            for name, frame in pd.read_excel(path, sheet_name=None, dtype=str).items()
        }

    def read(self, path):
        frames = pd.read_excel(path, sheet_name=None, dtype=str)
        result = self.adapt(frames)
        result.raw_path = Path(path)
        return result

    def adapt(self, frames):
        tables = {name: [] for name in ("instruments", "prices", "nav", "clients", "options", "snapshots")}
        issues, identities = [], {}
        frames = {
            name: frame.rename(columns=self.column_mapping.get(name, {})) for name, frame in frames.items()
        }
        for name in ("Funds", "Instruments", "PriceHistory"):
            for raw in frames.get(name, pd.DataFrame()).to_dict("records"):
                row = clean_record(raw)
                symbol = normalize_text(row.get("symbol") or row.get("l18"))
                identifier = row.get("instrument_id") or row.get("id") or row.get("insCode")
                if identifier and symbol:
                    if isinstance(identifier, float) and abs(identifier) >= 1e15:
                        raise ValueError("Large numeric identifier: supply original ID as text")
                    identities[symbol] = str(identifier)
        for sheet, target in (
            ("Funds", "snapshots"),
            ("Instruments", "instruments"),
            ("PriceHistory", "prices"),
            ("NAV", "nav"),
            ("ClientHistory", "clients"),
            ("Options", "options"),
        ):
            for index, raw in enumerate(frames.get(sheet, pd.DataFrame()).to_dict("records")):
                row = clean_record(raw)
                try:
                    symbol = normalize_text(row.get("symbol") or row.get("l18"))
                    identifier = row.get("instrument_id") or row.get("id") or identities.get(symbol)
                    if not identifier or not symbol:
                        raise ValueError("Missing stable identifier/symbol; raw row retained")
                    stamp = row.get("retrieved_at") or row.get("snapshot_ts")
                    # Imports without a retrieval stamp become available now, never retroactively.
                    row.update(
                        instrument_id=str(identifier),
                        symbol=symbol,
                        source=self.source,
                        retrieved_at=utc_timestamp(stamp),
                        available_at=utc_timestamp(row.get("available_at") or stamp),
                    )
                    for original, canonical in PRICE_MAP.items():
                        if canonical not in row and row.get(original) is not None:
                            row[canonical] = row[original]
                    if target == "options":
                        for original, canonical in OPTION_MAP.items():
                            if row.get(canonical) is None and row.get(original) is not None:
                                row[canonical] = row[original]
                        if row.get("expiration"):
                            row["original_expiration"] = str(row["expiration"])
                            row["expiration"] = parse_date(row["expiration"]).isoformat()
                    date_value = row.get("trading_date") or row.get("market_date") or row.get("date")
                    if date_value:
                        row["original_date"] = str(date_value)
                        row["trading_date"] = parse_date(date_value).isoformat()
                    if target == "nav":
                        row["nav"] = row.get("nav") or row.get("predtran")
                    for col in NUMERIC:
                        if row.get(col) is not None:
                            row[col] = float(normalize_text(row[col]).replace(",", ""))
                            if not np.isfinite(row[col]):
                                raise ValueError(f"Nonfinite {col}")
                    if target in ("prices", "nav", "clients") and not row.get("trading_date"):
                        raise ValueError("Missing observation date; snapshot_date is not a trading date")
                    if target == "prices":
                        PriceObservation.model_validate(row)
                    if target == "nav" and (row.get("nav") is None or row["nav"] <= 0):
                        raise ValueError("Missing/invalid redemption NAV")
                    tables[target].append(row)
                    if sheet in ("Funds", "PriceHistory"):
                        tables["instruments"].append(
                            {
                                "instrument_id": str(identifier),
                                "symbol": symbol,
                                "source": self.source,
                                "name": row.get("l30"),
                                "instrument_type": row.get("instrument_type", "unknown"),
                                "retrieved_at": row["retrieved_at"],
                            }
                        )
                    if sheet == "Funds" and row.get("trading_date"):
                        try:
                            PriceObservation.model_validate(row)
                            tables["prices"].append(row.copy())
                        except ValueError:
                            issues.append(
                                {"sheet": sheet, "row": index + 2, "issue": "Snapshot has invalid OHLC"}
                            )
                    if target == "options" and not row.get("trading_date"):
                        issues.append(
                            {
                                "sheet": sheet,
                                "row": index + 2,
                                "issue": "Trading date unknown; execution/backtest disabled for this quote",
                            }
                        )
                except (ValueError, TypeError) as exc:
                    # Validation messages contain fields, not confidential request URLs.
                    issues.append({"sheet": sheet, "row": index + 2, "issue": str(exc)[:500]})
        return ImportBatch({k: pd.DataFrame(v) for k, v in tables.items() if v}, issues)


class ExcelProvider(DataProvider):
    name = "Excel"

    def __init__(self, path, adapter=None):
        self.path, self.adapter = Path(path), adapter or CollectorAdapter()

    def fetch(self, include_history=False):
        return self.adapter.read(self.path)


class BrsApiProvider(DataProvider):
    name = "BrsApi"

    def __init__(self, output_dir, tickers=None):
        self.output_dir, self.tickers = Path(output_dir), tickers

    def fetch(self, include_history=False):
        from leveraged_funds_collector import LeveragedFundsCollector

        kwargs = {"output_dir": self.output_dir}
        if self.tickers:
            kwargs.update(tickers=self.tickers, discover_new_funds=False)
        collector = LeveragedFundsCollector(**kwargs)
        result = collector.collect_daily(include_history=include_history)
        batch = CollectorAdapter().adapt(result.sheets)
        batch.raw_path = result.file_path
        batch.issues.extend(result.errors)
        return batch


def update_market_data(repository: ExcelRepository, provider: DataProvider, raw_dir, include_history=False):
    batch = provider.fetch(include_history=include_history)
    if batch.raw_path:
        repository.archive_file(batch.raw_path, raw_dir)
    if not batch.tables:
        raise ValueError("No valid records; historical database unchanged")
    counts = repository.upsert_many(batch.tables)
    LOGGER.info("update_complete provider=%s counts=%s issues=%s", provider.name, counts, len(batch.issues))
    return {"counts": counts, "issues": batch.issues, "retrieved_at": utc_timestamp()}


def numeric_frame(frame):
    frame = frame.copy()
    for name in NUMERIC & set(frame.columns):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def load_historical_data(repository, symbol, source="BrsApi", as_of=None):
    table = "price_versions" if as_of is not None else "prices"
    data = numeric_frame(repository.read(table))
    if data.empty:
        return data
    data = data[
        (data.symbol.eq(normalize_text(symbol)) | data.instrument_id.eq(str(symbol))) & data.source.eq(source)
    ]
    if as_of is not None:
        data = data[pd.to_datetime(data.available_at, utc=True) <= pd.Timestamp(utc_timestamp(as_of))]
    return (
        data.sort_values("retrieved_at")
        .drop_duplicates("trading_date", keep="last")
        .sort_values("trading_date")
        .reset_index(drop=True)
    )


def load_options_chain(repository, underlying_symbol, source="BrsApi", as_of=None):
    data = numeric_frame(repository.read("options"))
    if data.empty or "underlying_symbol" not in data:
        return pd.DataFrame()
    data = data[
        data.underlying_symbol.map(normalize_text).eq(normalize_text(underlying_symbol))
        & data.source.eq(source)
    ]
    if as_of is not None:
        data = data[pd.to_datetime(data.available_at, utc=True) <= pd.Timestamp(utc_timestamp(as_of))]
    return (
        data.sort_values("retrieved_at").drop_duplicates("instrument_id", keep="last").reset_index(drop=True)
    )


def validate_market_data(repository, calendar=None):
    issues = []
    data = numeric_frame(repository.read("prices"))
    for index, row in data.iterrows():
        try:
            PriceObservation.model_validate(clean_record(row.to_dict()))
        except ValueError as exc:
            issues.append({"table": "prices", "row": index, "issue": str(exc)[:300]})
    if not data.empty and calendar:
        for identifier, group in data.groupby("instrument_id"):
            dates = {parse_date(d) for d in group.trading_date}
            missing = set(calendar.sessions(min(dates), max(dates))) - dates
            for day in sorted(missing):
                issues.append(
                    {
                        "table": "prices",
                        "instrument_id": identifier,
                        "date": str(day),
                        "issue": "Missing session or suspension/unconfigured holiday; never forward filled",
                    }
                )
    return pd.DataFrame(issues)


def export_data_to_excel(repository, destination):
    repository.export(destination)
