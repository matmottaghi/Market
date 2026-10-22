"""Regression checks for orchestration, failure preservation and report visibility."""
import json
from pathlib import Path

from filelock import FileLock, Timeout
import pytest

from market_dss.config import Settings
from market_dss.pipeline import MarketPipelineService
from market_dss.service import MarketService


def make_service(tmp_path):
    return MarketService(Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models",
                                  collector_dir=tmp_path / "collector", source="test_fixture"))


def test_missing_live_key_still_collects_history(tmp_path, monkeypatch):
    monkeypatch.delenv("BRS_API_KEY", raising=False)
    service = make_service(tmp_path)
    calls = []
    monkeypatch.setattr(service, "update", lambda *a, **k: pytest.fail("Must not call live API"))
    def history(symbol):
        calls.append(symbol)
        return {"inserted": 0, "fetched": 0}
    monkeypatch.setattr(service, "update_historical", history)
    monkeypatch.setattr(service, "refresh_live", lambda symbols: {"provider": "TSETMC", "counts": {}})
    run = MarketPipelineService(service).run_full_pipeline(symbols=["TEST_ONLY"])
    assert calls == ["TEST_ONLY"]
    assert run["stages"][1]["status"] == "SUCCESS"
    assert json.loads((service.settings.data_dir / "processed/last_pipeline.json").read_text("utf-8"))["run_id"] == run["run_id"]


def test_overlapping_pipeline_rejected(tmp_path):
    service = make_service(tmp_path)
    with FileLock(str(service.settings.data_dir / "pipeline.lock")):
        with pytest.raises(Timeout):
            MarketPipelineService(service).run_full_pipeline(collect=False)


def test_collection_failure_is_redacted_and_preserves_forecasts(tmp_path, monkeypatch):
    monkeypatch.setenv("BRS_API_KEY", "private-test-secret")
    service = make_service(tmp_path)
    previous = service.settings.data_dir / "forecasts" / "valid.xlsx"
    previous.write_bytes(b"prior valid result")
    def broken(*args, **kwargs):
        raise ConnectionError("https://example.test?key=private-test-secret")
    monkeypatch.setattr(service, "update", broken)
    report = MarketPipelineService(service).run_full_pipeline(historical_backfill="never")
    assert report["status"] == "PARTIAL"
    assert any(s["status"] == "FAILED" for s in report["stages"])
    assert previous.read_bytes() == b"prior valid result"
    assert "private-test-secret" not in json.dumps(report)
    assert "private-test-secret" not in (service.settings.data_dir / "processed/last_pipeline.json").read_text("utf-8")


def test_inference_without_model_does_not_train(tmp_path, monkeypatch, synthetic_history):
    service = make_service(tmp_path)
    service.repository.upsert_many({"prices": synthetic_history, "instruments":
        synthetic_history[["instrument_id", "symbol", "source"]].drop_duplicates()})
    from market_dss.forecasting import ForecastEngine
    monkeypatch.setattr(ForecastEngine, "fit", lambda *a, **k: pytest.fail("No implicit training"))
    report = MarketPipelineService(service).run_full_pipeline(symbols=["TEST_ONLY"], collect=False, retrain_policy="never")
    assert any(s["status"] == "SKIPPED" and "مدل" in s.get("message", "") for s in report["stages"])
    assert (service.settings.data_dir / "features/synthetic-underlying.xlsx").exists()


def test_evaluation_visible_without_inference(tmp_path, monkeypatch, synthetic_history):
    from streamlit.testing.v1 import AppTest
    from market_dss.forecasting import ForecastEngine, ModelConfig
    service = make_service(tmp_path)
    monkeypatch.setenv("DSS_DATA_DIR", str(service.settings.data_dir))
    monkeypatch.setenv("DSS_MODEL_DIR", str(service.settings.model_dir))
    monkeypatch.setenv("DSS_SOURCE", "test_fixture")
    service.repository.upsert_many({"prices": synthetic_history, "instruments":
        synthetic_history[["instrument_id", "symbol", "source"]].drop_duplicates()})
    ForecastEngine(ModelConfig(epochs=1, hidden_size=8, layers=1, horizon=5, window=10)).fit(
        synthetic_history).save(service.settings.model_dir)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=45).run()
    assert not app.exception and not app.error
    assert not any("rmse" in frame.value.columns for frame in app.dataframe)
    app.switch_page("market_dss/pages/forecast.py").run()
    assert any(s.label == "نسخه مدل" for s in app.selectbox)
    assert any("rmse" in frame.value.columns for frame in app.dataframe)
    assert [tab.label for tab in app.tabs] == ["پیش‌بینی قیمت", "منحنی یادگیری", "ارزیابی عملکرد مدل", "مدیریت مدل"]
    assert not app.exception and not app.error

def test_pipeline_auto_trains_once_then_predicts_refreshed_history(tmp_path, synthetic_history):
    service = make_service(tmp_path)
    service.settings.training_epochs = 1
    service.repository.upsert_many({"prices": synthetic_history.head(400), "instruments":
        synthetic_history[["instrument_id", "symbol", "source"]].drop_duplicates()})
    first = MarketPipelineService(service).run_full_pipeline(symbols=["TEST_ONLY"], collect=False)
    assert first["counts"]["models_updated"] == 1
    assert first["counts"]["forecasts_updated"] == 3
    service.repository.upsert_many({"prices": synthetic_history.tail(20)})
    second = MarketPipelineService(service).run_full_pipeline(symbols=["TEST_ONLY"], collect=False)
    assert second["counts"]["models_updated"] == 0
    assert {result["cutoff"] for result in second["results"]} == {synthetic_history.trading_date.iloc[-1]}
    assert second["results"][0]["version"] == first["results"][0]["version"]
    assert len(service.history("TEST_ONLY")) == 420


def test_failed_retraining_preserves_model_and_marks_stale(tmp_path, monkeypatch, synthetic_history):
    from market_dss.forecasting import ForecastEngine
    service = make_service(tmp_path)
    service.settings.training_epochs = 1
    service.repository.upsert_many({"prices": synthetic_history, "instruments":
        synthetic_history[["instrument_id", "symbol", "source"]].drop_duplicates()})
    first = MarketPipelineService(service).run_full_pipeline(symbols=["TEST_ONLY"], collect=False)
    def failure(*args):
        raise RuntimeError("Training unavailable")
    monkeypatch.setattr(ForecastEngine, "fit", failure)
    second = MarketPipelineService(service).run_full_pipeline(symbols=["TEST_ONLY"], collect=False, retrain_policy="always")
    assert second["results"] and all(r["model_stale"] for r in second["results"])
    assert second["results"][0]["version"] == first["results"][0]["version"]
    assert len(service.models("synthetic-underlying")) == 1


def test_pipeline_button_only_runs_once(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("DSS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DSS_MODEL_DIR", str(tmp_path / "models"))
    calls = []
    def execute(self, *args, **kwargs):
        calls.append(kwargs)
        return {"run_id": "test", "status": "SUCCESS", "counts": {}, "stages": [], "results": []}
    monkeypatch.setattr(MarketPipelineService, "run_full_pipeline", execute)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    app.switch_page("market_dss/pages/settings.py").run()
    button = next(b for b in app.button if b.key == "run_full_pipeline")
    assert not button.disabled
    button.click().run()
    app.run()
    assert len(calls) == 1
    assert not app.exception and not app.error


def test_settings_password_field_and_secure_save(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from market_dss.credentials import CredentialStore
    monkeypatch.setenv("DSS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DSS_MODEL_DIR", str(tmp_path / "models"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    app.switch_page("market_dss/pages/settings.py").run()
    field = next(x for x in app.text_input if x.key == "new_live_api_key")
    assert field.proto.type == field.proto.PASSWORD
    field.set_value("fixture-local-api-key")
    next(b for b in app.button if b.label == "ذخیره امن تنظیمات").click().run()
    assert not app.error and not app.exception
    assert CredentialStore(tmp_path / "data").get() == "fixture-local-api-key"
    assert "fixture-local-api-key" not in " ".join(m.value for m in app.markdown)

def test_old_option_snapshot_not_reused(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    service = make_service(tmp_path)
    monkeypatch.setattr(service, "chain", lambda symbol: pd.DataFrame([
        {"retrieved_at": "2024-01-01T00:00:00Z", "instrument_id": "old-contract"}]))
    forecast = SimpleNamespace(metadata={"instrument_id": "underlying"})
    with pytest.raises(ValueError, match="No verified"):
        MarketPipelineService(service).option_scenarios("TEST_ONLY", pd.DataFrame(), forecast, 1,
                                                      "2024-01-02T00:00:00Z")
