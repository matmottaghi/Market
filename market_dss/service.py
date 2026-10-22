"""Application orchestration shared by CLI, scheduler and dashboard."""

import json
import logging
import hashlib

import pandas as pd
from filelock import FileLock

from .calendar import TradingCalendar, utc_timestamp
from .config import Settings
from .ingestion import (
    BrsApiProvider,
    CollectorAdapter,
    ExcelProvider,
    load_historical_data,
    load_options_chain,
    update_market_data,
    validate_market_data,
)
from .logging import configure_logging
from .storage import ExcelRepository, atomic_bytes


class MarketService:
    def __init__(self, settings=None):
        self.settings = settings or Settings.from_env()
        self.settings.initialize()
        configure_logging(self.settings.data_dir / "logs" / "dss.jsonl")
        self.repository = ExcelRepository(self.settings.data_dir / "historical" / "market.xlsx")
        self.calendar = TradingCalendar.from_csv(self.settings.holidays_file)

    def update(self, include_history=False, provider=None, tickers=None):
        provider = provider or BrsApiProvider(self.settings.collector_dir, tickers, self.live_key())
        status_path = self.settings.data_dir / "processed" / "last_update.json"
        # Includes the legacy collector's same-day write; one refresh process at a time.
        with FileLock(str(self.settings.data_dir / "refresh.lock"), timeout=1):
            try:
                report = update_market_data(
                    self.repository, provider, self.settings.data_dir / "raw", include_history
                )
                report["status"] = "ok"
            except Exception as exc:
                # Do not serialize request exception strings, which may carry API keys.
                report = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "retrieved_at": utc_timestamp(),
                }
                atomic_bytes(status_path, json.dumps(report, ensure_ascii=False).encode())
                logging.getLogger(__name__).error("update_failed type=%s", type(exc).__name__)
                raise
            atomic_bytes(status_path, json.dumps(report, ensure_ascii=False, default=str).encode())
            return report

    def import_file(self, path, mapping=None):
        from pathlib import Path

        fingerprint = hashlib.sha256(
            Path(path).read_bytes()
            + json.dumps(mapping, sort_keys=True).encode()
            + self.settings.source.encode()
        ).hexdigest()
        manifest_path = self.settings.data_dir / "processed" / "imports.json"
        with FileLock(str(manifest_path) + ".lock", timeout=10):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            if fingerprint in manifest:
                return {"status": "already_imported", "first_imported_at": manifest[fingerprint]}
            report = self.update(
                provider=ExcelProvider(path, CollectorAdapter(self.settings.source, mapping))
            )
            manifest[fingerprint] = report["retrieved_at"]
            atomic_bytes(manifest_path, json.dumps(manifest).encode())
            return report

    def live_key(self):
        from .credentials import CredentialStore
        return CredentialStore(self.settings.data_dir).get()

    def test_live_connection(self, symbols=None):
        from .live import diagnose_market_api
        from .credentials import CredentialStore
        rows = diagnose_market_api("BrsApi", self.live_key(), symbols=symbols,
                                   max_age_hours=self.settings.live_max_age_hours)
        report = {"credentials": CredentialStore(self.settings.data_dir).status(), "requests": rows,
                  "status": "SUCCESS" if any(r["status"] == "CONNECTED" for r in rows) else "FAILED",
                  "retrieved_at": utc_timestamp()}
        old_path = self.settings.data_dir / "processed/live_connection.json"
        old = json.loads(old_path.read_text("utf-8")) if old_path.exists() else {}
        if report["status"] == "SUCCESS":
            report["last_success_at"] = report["retrieved_at"]
        elif old.get("last_success_at"):
            report["last_success_at"] = old["last_success_at"]
        atomic_bytes(old_path, json.dumps(report, ensure_ascii=False).encode("utf-8"))
        return report

    def refresh_live(self, symbols):
        from .live import TsetmcLiveProvider
        from .credentials import CredentialStore
        from leveraged_funds_collector import MarketDataError
        credential_status = CredentialStore(self.settings.data_dir).status()
        warnings = []
        if self.live_key():
            try:
                report = self.update(False, tickers=symbols)
                if report.get("incoming_counts", {}).get("snapshots", 0) > 0:
                    return {**report, "provider": "BrsApi"}
                warnings.append({"provider": "BrsApi", "issue": "No new valid snapshot records"})
            except Exception as exc:
                warnings.append({"provider": "BrsApi", "error_type": type(exc).__name__,
                                 "issue": str(exc) if isinstance(exc, MarketDataError) else
                                     "Primary live provider failed; inspect connection diagnostics"})
        report = self.update(provider=TsetmcLiveProvider(self.settings.data_dir, symbols,
                            max_age_hours=self.settings.live_max_age_hours))
        if not self.live_key():
            warnings.append({"provider": "BrsApi", "issue": credential_status.get("message", "Live key not configured")})
        return {**report, "provider": "TSETMC", "warnings": warnings, "credentials": credential_status,
                "limitations": "Public stock/ETF quotes only; NAV and listed option chains not provided by this fallback"}

    def latest_quote(self, symbol):
        frame = self.repository.read("snapshots")
        if frame.empty:
            return None
        frame = frame[frame.symbol.eq(symbol)]
        return frame.sort_values("retrieved_at").iloc[-1].to_dict() if not frame.empty else None

    def coverage(self, symbols=None):
        from .features import prepare_model_frame
        rows = []
        for symbol in symbols or ["اهرم", "موج", "توان"]:
            history = self.history(symbol)
            if history.empty:
                rows.append({"symbol": symbol, "stored_records": 0})
                continue
            identifier = str(history.instrument_id.iloc[-1])
            path = self.settings.data_dir / "processed" / f"history_{identifier}.json"
            report = json.loads(path.read_text("utf-8")) if path.exists() else {}
            try:
                ready, _ = prepare_model_frame(history, self.settings.annual_sessions)
                eligible, preparation = len(ready), "VALID"
            except ValueError:
                eligible, preparation = 0, "FEATURE_GAPS"
            days = {pd.Timestamp(d).date() for d in history.trading_date}
            gaps = len(set(self.calendar.sessions(min(days), max(days))) - days)
            rows.append({"symbol": symbol, "instrument_id": identifier, "source": str(history.source.iloc[-1]),
                         "earliest_date": str(history.trading_date.min()), "latest_date": str(history.trading_date.max()),
                         "api_records": report.get("api_records"), "stored_records": len(history),
                         "valid_feature_rows": eligible, "preprocessing_removed": len(history) - eligible,
                         "api_removed": report.get("removed_records"), "missing_values": int(history.isna().sum().sum()),
                         "duplicates": int(history.duplicated(["instrument_id", "trading_date", "source"]).sum()),
                         "potential_session_gaps": gaps, "preparation_status": preparation})
        return pd.DataFrame(rows)

    def update_historical(self, symbol, provider=None):
        from .historical import TsetmcProvider
        provider = provider or TsetmcProvider(self.settings.data_dir)
        previous = load_historical_data(self.repository, symbol, "TSETMC")
        latest = None if previous.empty else str(previous.trading_date.max())
        batch = provider.fetch_symbol(symbol, latest_date=latest)
        incoming = batch.tables["prices"]
        counts = self.repository.upsert_many(batch.tables)
        old_dates = set(previous.trading_date) if not previous.empty else set()
        report = {"symbol": symbol, "status": "SUCCESS", "source": "TSETMC",
                  "retrieved_at": utc_timestamp(), "fetched": len(incoming),
                  "inserted": int((~incoming.trading_date.isin(old_dates)).sum()),
                  "previous_latest": latest, "mode": "backfill" if latest is None else "reconcile",
                  "latest": str(incoming.trading_date.max()), "counts": counts, "issues": batch.issues, **batch.stats}
        atomic_bytes(self.settings.data_dir / "processed" / f"history_{incoming.instrument_id.iloc[0]}.json",
                     json.dumps(report, ensure_ascii=False).encode("utf-8"))
        return report

    def history(self, symbol):
        # Choose the freshest complete series; TSETMC wins ties. Never splice provider levels.
        public = load_historical_data(self.repository, symbol, "TSETMC")
        alternate = load_historical_data(self.repository, symbol, self.settings.source)
        if public.empty:
            return alternate
        if not alternate.empty and str(alternate.trading_date.max()) > str(public.trading_date.max()):
            return alternate
        return public

    def chain(self, symbol, as_of=None):
        return load_options_chain(self.repository, symbol, self.settings.source, as_of)

    def instruments(self):
        frame = self.repository.read("instruments")
        if frame.empty:
            return frame
        frame = frame[frame.source.isin(["TSETMC", self.settings.source])].copy()
        frame["priority"] = frame.source.eq("TSETMC").astype(int)
        return frame.sort_values("priority", ascending=False).drop_duplicates("symbol").drop(columns="priority")

    def models(self, identifier):
        return sorted((self.settings.model_dir / str(identifier)).glob("*/model.json"), reverse=True)

    def quality(self):
        return validate_market_data(self.repository, self.calendar)

    def training_due(self, identifier):
        models = self.models(identifier)
        if not models:
            return True
        info = json.loads(models[0].read_text(encoding="utf-8"))
        age = pd.Timestamp(utc_timestamp()) - pd.Timestamp(info["metadata"]["created_at"])
        return age.days >= self.settings.retrain_days


def schedule_updates(service, hour=16, minute=30, include_history=False, retrain=False):
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="Asia/Tehran")

    def job():
        day = pd.Timestamp.now(tz="Asia/Tehran").date()
        if service.calendar.is_session(day):
            service.update(include_history=include_history)
            if retrain:
                from .forecasting import ForecastEngine, InsufficientData, ModelConfig

                for row in service.instruments().itertuples():
                    if service.training_due(row.instrument_id):
                        try:
                            ForecastEngine(ModelConfig(annual_sessions=service.settings.annual_sessions)).fit(
                                service.history(row.symbol)
                            ).save(service.settings.model_dir)
                        except (InsufficientData, ValueError) as exc:
                            logging.getLogger(__name__).warning(
                                "scheduled_training_skipped symbol=%s type=%s", row.symbol, type(exc).__name__
                            )

    scheduler.add_job(
        job,
        CronTrigger(day_of_week="sat,sun,mon,tue,wed", hour=hour, minute=minute, timezone="Asia/Tehran"),
        id="market_update",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    scheduler.start()
