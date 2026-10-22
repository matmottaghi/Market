"""Durable one-click orchestration: public history and authenticated live data are independent."""
from copy import deepcopy
import hashlib
import inspect
import json
from time import monotonic
from uuid import uuid4

import pandas as pd
from filelock import FileLock

from .calendar import normalize_text, utc_timestamp
from .features import MODEL_FEATURES, engineer_features
from .storage import atomic_bytes, excel_bytes


class MarketPipelineService:
    def __init__(self, service):
        self.service = service

    def run_full_pipeline(self, symbols=None, historical_backfill="auto", update_live=True,
                          retrain_policy="auto", update_forecasts=True, update_options=True,
                          progress_callback=None, *, include_history=None, refresh_models=False, collect=None):
        # Compatibility with existing CLI/callers of the initial orchestration API.
        if collect is False:
            historical_backfill, update_live = "never", False
        if refresh_models:
            retrain_policy = "always"
        if retrain_policy not in ("auto", "always", "never") or historical_backfill not in ("auto", "never", "full"):
            raise ValueError("Invalid pipeline policy")
        service = self.service
        symbols = list(dict.fromkeys(normalize_text(s) for s in (symbols or
                    service.settings.default_symbols.replace("،", ",").split(",")) if normalize_text(s)))
        run = {"run_id": uuid4().hex, "status": "RUNNING", "started_at": utc_timestamp(), "stages": [],
               "results": [], "history": [], "symbols": symbols,
               "counts": {"symbols_processed": 0, "historical_inserted": 0, "historical_fetched": 0,
                          "models_updated": 0, "option_contracts": 0, "forecasts_updated": 0}}
        report_path = service.settings.data_dir / "processed" / "pipeline_runs" / f"{run['run_id']}.json"
        started = monotonic()

        def publish():
            run["elapsed_seconds"] = round(monotonic() - started, 2)
            content = json.dumps(run, ensure_ascii=False, default=str).encode("utf-8")
            atomic_bytes(report_path, content)
            atomic_bytes(service.settings.data_dir / "processed" / "last_pipeline.json", content)
            if progress_callback:
                progress_callback(deepcopy(run))

        def step(name, operation, skip=None):
            stage = {"name": name, "status": "PENDING", "started_at": None, "ended_at": None}
            run["stages"].append(stage)
            publish()
            stage.update(status="RUNNING", started_at=utc_timestamp())
            publish()
            try:
                if skip:
                    stage.update(status="SKIPPED", message=skip)
                    return None
                value = operation()
                stage["status"] = "SUCCESS"
                return value
            except Exception as exc:
                from .historical import HistoricalAPIError
                from leveraged_funds_collector import MarketDataError
                safe = str(exc) if isinstance(exc, (HistoricalAPIError, MarketDataError)) else (
                    "عملیات ناموفق؛ داده کافی، تنظیمات مدل یا دسترسی فایل را بررسی کنید.")
                stage.update(status="FAILED", error_type=type(exc).__name__, message=safe)
                return None
            finally:
                stage["ended_at"] = utc_timestamp()
                publish()

        with FileLock(str(service.settings.data_dir / "pipeline.lock"), timeout=0):
            step("اعتبارسنجی تنظیمات — تاریخچه بدون کلید", lambda: True)
            for symbol in symbols:
                if historical_backfill != "never":
                    history_report = step(f"TSETMC: شناسایی نماد، تاریخچه و ثبت اکسل — {symbol}",
                                          lambda: service.update_historical(symbol))
                    if history_report:
                        run["history"].append(history_report)
                        run["counts"]["historical_inserted"] += history_report["inserted"]
                        run["counts"]["historical_fetched"] += history_report["fetched"]
                    elif service.live_key():
                        fallback = step(f"تاریخچه جایگزین BrsApi — {symbol}",
                                        lambda: service.update(True, tickers=[symbol]))
                        if fallback:
                            run["history"].append({"symbol": symbol, "source": "BrsApi", **fallback})
            live = None
            if update_live:
                live = step("قیمت جاری، NAV و اختیار — BrsApi با جایگزین عمومی TSETMC",
                            lambda: service.refresh_live(symbols))
                if live:
                    run["collection"] = live
                    run["counts"]["option_contracts"] = live.get("incoming_counts", {}).get("options", 0)
            else:
                step("دریافت بازار زنده", lambda: None, skip="دریافت زنده در این اجرا خاموش است.")
            step("گزارش کیفیت داده", lambda: atomic_bytes(
                service.settings.data_dir / "processed" / "quality.xlsx",
                excel_bytes({"issues": service.quality()})))
            from .forecasting import ForecastEngine, ModelConfig, TRAINING_SCHEMA_VERSION
            config = ModelConfig(annual_sessions=service.settings.annual_sessions,
                                 epochs=service.settings.training_epochs)
            signature = hashlib.sha256((inspect.getsource(engineer_features) + str(TRAINING_SCHEMA_VERSION) +
                json.dumps(config.model_dump(), sort_keys=True)).encode()).hexdigest()
            for symbol in symbols:
                history = service.history(symbol)
                if history.empty:
                    step(f"مدل و پیش‌بینی — {symbol}", lambda: None, skip="تاریخچه معتبری دریافت نشد؛ خروجی قبلی حفظ شد.")
                    continue
                identifier = str(history.instrument_id.iloc[-1])
                def features():
                    result = engineer_features(history, service.settings.annual_sessions)
                    atomic_bytes(service.settings.data_dir / "features" / f"{identifier}.xlsx",
                                 excel_bytes({"features": result}))
                    return result
                if step(f"ویژگی‌ها — {symbol}", features) is None:
                    continue
                run["counts"]["symbols_processed"] += 1
                if not update_forecasts:
                    continue
                versions = service.models(identifier)
                info = json.loads(versions[0].read_text("utf-8")) if versions else None
                incompatible = bool(info and ([f for f in info.get("features", []) if f != "volume_change"] != [f for f in MODEL_FEATURES if f != "volume_change"] or
                    info["metadata"].get("source") != str(history.source.iloc[-1]) or
                    info["metadata"].get("pipeline_signature") != signature))
                due = not versions or incompatible or service.training_due(identifier)
                model, stale = None, False
                if retrain_policy == "always" or (retrain_policy == "auto" and due):
                    def train():
                        engine = ForecastEngine(config).fit(history)
                        engine.metadata["pipeline_signature"] = signature
                        engine.save(service.settings.model_dir)
                        run["counts"]["models_updated"] += 1
                        run.setdefault("evaluations", []).append({"symbol": symbol, "version": engine.metadata["version"],
                            "metrics": engine.metrics.to_dict("records")})
                        return engine
                    model = step(f"آموزش و ارزیابی زمانی — {symbol}", train)
                    stale = model is None
                if model is None and versions and not incompatible:
                    model = step(f"بارگذاری مدل ذخیره‌شده — {symbol}", lambda: ForecastEngine.load(versions[0].parent))
                    stale = stale or due
                if model is None:
                    step(f"پیش‌بینی — {symbol}", lambda: None, skip="مدل سازگار در دسترس نیست؛ داده کافی یا آموزش لازم است.")
                    continue
                for horizon in (1, 5, 21):
                    if horizon > model.config.horizon:
                        step(f"پیش‌بینی {horizon} — {symbol}", lambda: None, skip="افق مدل کافی نیست.")
                        continue
                    def predict():
                        forecast = model.predict(history, horizon, calendar=service.calendar)
                        forecast.metadata.update(model_stale=stale, pipeline_run_id=run["run_id"])
                        forecast.export(service.settings.data_dir / "forecasts" / f"{identifier}_{horizon}.xlsx")
                        run["results"].append({"symbol": symbol, "horizon": horizon,
                            "version": model.metadata["version"], "cutoff": forecast.metadata["input_data_cutoff"],
                            "model_stale": stale})
                        run["counts"]["forecasts_updated"] += 1
                        return forecast
                    forecast = step(f"پیش‌بینی {horizon} — {symbol}", predict)
                    if forecast is not None and update_options:
                        step(f"سناریوهای اختیار — {symbol} / {horizon}",
                             lambda: self.option_scenarios(symbol, history, forecast, horizon, run["started_at"]),
                             skip=None if live and live.get("provider") == "BrsApi" and not stale else "دریافت زنده معتبر یا مدل تازه موجود نیست.")
            states = {stage["status"] for stage in run["stages"]}
            run.update(status="PARTIAL" if states & {"FAILED", "SKIPPED"} else "SUCCESS", ended_at=utc_timestamp())
            publish()
        return run

    def option_scenarios(self, symbol, history, forecast, horizon, snapshot_since=None):
        from .options import contracts_from_frame, build_strategy
        from .risk import evaluate_strategy
        service = self.service
        chain = service.chain(symbol)
        if snapshot_since and not chain.empty:
            chain = chain[pd.to_datetime(chain.retrieved_at, utc=True) >= pd.Timestamp(snapshot_since)]
        if not chain.empty and "trading_date" in chain:
            ages = (pd.Timestamp.now(tz="Asia/Tehran").tz_localize(None) -
                    pd.to_datetime(chain.trading_date, errors="coerce")).dt.total_seconds() / 3600
            chain = chain[ages.between(0, service.settings.live_max_age_hours)]
        contracts, issues = contracts_from_frame(chain)
        results = []
        for contract in contracts:
            try:
                if contract.underlying_id != forecast.metadata["instrument_id"]:
                    raise ValueError("Underlying identity mismatch")
                strategy = build_strategy("Long Call" if contract.option_type == "call" else "Long Put",
                    [contract], float(history.close.iloc[-1]), history.trading_date.iloc[-1])
                metric, _ = evaluate_strategy(strategy, forecast, horizon)
                results.append({"contract": contract.instrument_id, "forecast_version": forecast.metadata["version"],
                                "data_cutoff": forecast.metadata["input_data_cutoff"],
                                "generated_at": utc_timestamp(), **metric})
            except ValueError as exc:
                issues.append({"contract": contract.instrument_id, "reason": str(exc)})
        if not results:
            raise ValueError("No verified executable contracts")
        destination = service.settings.data_dir / "options" / f"{forecast.metadata['instrument_id']}_{horizon}.xlsx"
        atomic_bytes(destination, excel_bytes({"scenarios": pd.DataFrame(results), "issues": pd.DataFrame(issues)}))
        return len(results)
