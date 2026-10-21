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
        provider = provider or BrsApiProvider(self.settings.collector_dir, tickers)
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

    def history(self, symbol):
        return load_historical_data(self.repository, symbol, self.settings.source)

    def chain(self, symbol, as_of=None):
        return load_options_chain(self.repository, symbol, self.settings.source, as_of)

    def instruments(self):
        frame = self.repository.read("instruments")
        return frame[frame.source.eq(self.settings.source)] if not frame.empty else frame

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
