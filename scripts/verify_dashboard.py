"""Read-only UI acceptance using actual saved market observations and model artifacts."""
from pathlib import Path
import json
from streamlit.testing.v1 import AppTest
from market_dss.service import MarketService

app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=60).run()
report = []
for slug in ["market", "forecast", "options", "strategy", "decision", "backtest", "settings"]:
    app.switch_page(f"market_dss/pages/{slug}.py").run()
    metrics = any("rmse" in frame.value.columns for frame in app.dataframe)
    report.append({"page": slug, "exceptions": len(app.exception), "errors": [e.value for e in app.error],
                   "tables": len(app.dataframe), "charts": len(app.get("plotly_chart")),
                   "has_model_evaluation": metrics})
assert all(not row["exceptions"] and not row["errors"] for row in report)
assert all(not row["has_model_evaluation"] for row in report if row["page"] != "forecast")

service = MarketService()
selections = []
app.switch_page("market_dss/pages/forecast.py").run()
for symbol in ["اهرم", "موج", "توان"]:
    history = service.history(symbol)
    if history.empty or not service.models(str(history.instrument_id.iloc[-1])):
        continue
    identifier = str(history.instrument_id.iloc[-1])
    next(s for s in app.selectbox if s.key == "forecast_symbol").set_value(symbol).run()
    for horizon in (1, 5, 21):
        next(s for s in app.selectbox if s.key == f"model_horizon_{identifier}").set_value(horizon).run()
        selections.append({"symbol": symbol, "horizon": horizon, "exceptions": len(app.exception),
                           "errors": [e.value for e in app.error], "tabs": [tab.label for tab in app.tabs]})
assert all(not row["exceptions"] and not row["errors"] for row in selections)
result = {"pages": report, "saved_forecast_selections": selections}
print(json.dumps(result, ensure_ascii=True))
Path("data/processed/dashboard_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
