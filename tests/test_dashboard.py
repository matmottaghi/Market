from pathlib import Path

from streamlit.testing.v1 import AppTest

from market_dss.storage import ExcelRepository


def test_all_seven_pages_empty_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DSS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DSS_MODEL_DIR", str(tmp_path / "models"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    for page in ["market", "forecast", "options", "strategy", "decision", "backtest", "settings"]:
        app.switch_page(f"market_dss/pages/{page}.py").run()
        assert not app.exception, page
        assert not app.error, page


def test_dashboard_with_market_history(tmp_path, monkeypatch, synthetic_history):
    monkeypatch.setenv("DSS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DSS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("DSS_SOURCE", "test_fixture")
    repo = ExcelRepository(tmp_path / "data" / "historical" / "market.xlsx")
    repo.upsert_many(
        {
            "prices": synthetic_history,
            "instruments": synthetic_history[["instrument_id", "symbol", "source"]].drop_duplicates(),
        }
    )
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 4


def test_dashboard_forecast_strategy_decision_flow(
    tmp_path, monkeypatch, synthetic_history, contract_factory
):
    import pandas as pd
    from market_dss.calendar import TradingCalendar, parse_date
    from market_dss.forecasting import ForecastEngine, ModelConfig

    monkeypatch.setenv("DSS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DSS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("DSS_SOURCE", "test_fixture")
    repo = ExcelRepository(tmp_path / "data" / "historical" / "market.xlsx")
    cutoff = parse_date(synthetic_history.trading_date.iloc[-1])
    expiry = TradingCalendar().next_sessions(cutoff, 5)[-1]
    c = contract_factory(
        strike=float(synthetic_history.close.iloc[-1]), expiration=expiry, trading_date=cutoff
    )
    chain = pd.DataFrame(
        [
            {
                **c.model_dump(mode="json"),
                "underlying_symbol": "TEST_ONLY",
                "source": "test_fixture",
                "retrieved_at": f"{cutoff}T13:00:00Z",
                "available_at": f"{cutoff}T13:00:00Z",
            }
        ]
    )
    repo.upsert_many(
        {
            "prices": synthetic_history,
            "options": chain,
            "instruments": synthetic_history[["instrument_id", "symbol", "source"]].drop_duplicates(),
        }
    )
    ForecastEngine(ModelConfig(epochs=1, hidden_size=8, layers=1, horizon=5, window=10)).fit(
        synthetic_history
    ).save(tmp_path / "models")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    app.switch_page("market_dss/pages/forecast.py").run()
    next(button for button in app.button if button.label == "تولید توزیع پیش‌بینی").click().run()
    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    app.switch_page("market_dss/pages/options.py").run()
    assert not app.error
    app.switch_page("market_dss/pages/strategy.py").run()
    app.multiselect[0].set_value([c.instrument_id]).run()
    next(button for button in app.button if button.label == "محاسبه استراتژی").click().run()
    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    next(button for button in app.button if button.label == "افزودن به مقایسه تصمیم").click().run()
    app.switch_page("market_dss/pages/decision.py").run()
    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    assert len(app.dataframe) >= 2
