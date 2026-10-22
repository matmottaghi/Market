"""Forecast-only diagnostics, saved report navigation and typography regressions."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from market_dss.config import Settings
from market_dss.forecasting import ForecastEngine, ModelConfig
from market_dss.service import MarketService


@pytest.fixture
def three_reports(tmp_path, monkeypatch, synthetic_history):
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models", source="test_fixture")
    service = MarketService(settings)
    monkeypatch.setenv("DSS_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("DSS_MODEL_DIR", str(settings.model_dir))
    monkeypatch.setenv("DSS_SOURCE", "test_fixture")
    frames = []
    for index, symbol in enumerate(["اهرم", "موج", "توان"]):
        history = synthetic_history.assign(symbol=symbol, instrument_id=f"fixture-{index}")
        frames.append(history)
        engine = ForecastEngine(ModelConfig(epochs=1, hidden_size=8, layers=1, horizon=21, window=10)).fit(history)
        engine.save(settings.model_dir)
        for horizon in (1, 5, 21):
            engine.predict(history, horizon=horizon, simulations=100).export(
                settings.data_dir / "forecasts" / f"fixture-{index}_{horizon}.xlsx")
    prices = pd.concat(frames, ignore_index=True)
    service.repository.upsert_many({"prices": prices, "instruments":
        prices[["instrument_id", "symbol", "source"]].drop_duplicates()})
    return service


def app():
    return AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=60).run()


def test_diagnostics_exclusive_to_forecasting_and_no_weight_loading(three_reports, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("Displaying reports must not load weights or train a model")
    monkeypatch.setattr(ForecastEngine, "load", forbidden)
    monkeypatch.setattr(ForecastEngine, "fit", forbidden)
    dashboard = app()
    for slug in ["market", "options", "strategy", "decision", "backtest", "settings"]:
        dashboard.switch_page(f"market_dss/pages/{slug}.py").run()
        assert not dashboard.exception and not dashboard.error, slug
        assert not any("rmse" in frame.value.columns or "validation_mse" in frame.value.columns
                       for frame in dashboard.dataframe), slug
        assert not any("منحنی یادگیری" in header.value or "ارزیابی عملکرد مدل" in header.value
                       for header in dashboard.subheader), slug
    dashboard.switch_page("market_dss/pages/forecast.py").run()
    assert not dashboard.exception and not dashboard.error
    assert [tab.label for tab in dashboard.tabs] == ["پیش‌بینی قیمت", "منحنی یادگیری", "ارزیابی عملکرد مدل", "مدیریت مدل"]
    assert any("rmse" in frame.value.columns for frame in dashboard.dataframe)
    assert any("learning_rate" in frame.value.columns for frame in dashboard.dataframe)


def test_three_symbols_horizons_and_restart_use_saved_artifacts(three_reports, monkeypatch):
    monkeypatch.setattr(ForecastEngine, "load", lambda *a, **k: pytest.fail("Saved forecast should be reused"))
    dashboard = app()
    dashboard.switch_page("market_dss/pages/forecast.py").run()
    for index, symbol in enumerate(["اهرم", "موج", "توان"]):
        next(s for s in dashboard.selectbox if s.key == "forecast_symbol").set_value(symbol).run()
        horizon_selector = next(s for s in dashboard.selectbox if s.key == f"model_horizon_fixture-{index}")
        for horizon in (1, 5, 21):
            horizon_selector.set_value(horizon).run()
            assert not dashboard.exception and not dashboard.error, (symbol, horizon)
            metrics = [frame.value for frame in dashboard.dataframe if "rmse" in frame.value.columns]
            assert metrics and set(metrics[0]["تعداد جلسات"]) == {horizon}
            horizon_selector = next(s for s in dashboard.selectbox if s.key == f"model_horizon_fixture-{index}")
    restarted = app()
    restarted.switch_page("market_dss/pages/forecast.py").run()
    assert any("learning_rate" in frame.value.columns for frame in restarted.dataframe)
    assert not restarted.error and not restarted.exception


def test_b_roya_is_requested_without_copying_restricted_assets():
    from market_dss.ui import font_style, PERSIAN_FONT
    assert PERSIAN_FONT.startswith("B Roya")
    assert 'local("B Roya")' in font_style()
    config = (Path(__file__).parents[1] / ".streamlit/config.toml").read_text("utf-8")
    assert 'font = "B Roya, Tahoma, sans-serif"' in config

def test_walk_forward_report_stays_with_its_instrument(tmp_path, monkeypatch, synthetic_history):
    from market_dss.storage import ExcelRepository
    settings = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models", source="test_fixture")
    monkeypatch.setenv("DSS_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("DSS_MODEL_DIR", str(settings.model_dir))
    monkeypatch.setenv("DSS_SOURCE", "test_fixture")
    prices = pd.concat([synthetic_history.assign(symbol=symbol, instrument_id=identifier)
                        for symbol, identifier in [("FUND_A", "fixture-a"), ("FUND_B", "fixture-b")]], ignore_index=True)
    ExcelRepository(settings.data_dir / "historical/market.xlsx").upsert_many({"prices": prices,
        "instruments": prices[["instrument_id", "symbol", "source"]].drop_duplicates()})
    result = pd.DataFrame([{"horizon":5,"actual":100,"expected":101,"target_date":"2024-01-01"}])
    monkeypatch.setattr("market_dss.forecasting.walk_forward_forecasts", lambda *a, **k: result)
    dashboard = app()
    dashboard.switch_page("market_dss/pages/backtest.py").run()
    next(b for b in dashboard.button if b.label == "اجرای آزمون پیش‌بینی").click().run()
    assert (settings.data_dir / "backtests/forecast_walk_forward_fixture-a.xlsx").exists()
    next(s for s in dashboard.selectbox if s.key == "selected_symbol").set_value("FUND_B").run()
    assert not (settings.data_dir / "backtests/forecast_walk_forward_fixture-b.xlsx").exists()
    assert not dashboard.dataframe and not dashboard.error and not dashboard.exception
