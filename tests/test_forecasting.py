import numpy as np
import pytest

from market_dss.features import engineer_features
from market_dss.forecasting import ForecastEngine, InsufficientData, ModelConfig, chronological_origins


def test_purged_chronology():
    groups, boundaries = chronological_origins(400, 30, 21)
    assert groups["train"][-1] + 21 < groups["validation"][0] + 1
    assert groups["validation"][-1] + 21 < groups["calibration"][0] + 1
    assert groups["calibration"][-1] + 21 < groups["test"][0] + 1
    assert boundaries[-1] == 400


def test_lstm_end_to_end_train_only_scaler_ordered_positive_paths(
    tmp_path, synthetic_history, contract_factory
):
    engine = ForecastEngine(
        ModelConfig(epochs=2, patience=1, hidden_size=8, layers=1, horizon=5, window=10)
    ).fit(synthetic_history)
    features = engineer_features(synthetic_history).dropna(subset=engine.feature_names).reset_index(drop=True)
    end = int(len(features) * 0.6)
    np.testing.assert_allclose(
        engine.scaler.mean_, features[engine.feature_names].iloc[:end].mean().to_numpy()
    )
    assert engine.metadata["training_target_end"] < engine.metadata["test_start"]
    forecast = engine.predict(synthetic_history, simulations=100)
    assert np.isfinite(forecast.paths).all() and (forecast.paths > 0).all()
    quantiles = np.quantile(forecast.paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    assert (np.diff(quantiles, axis=0) >= 0).all()
    assert set(engine.metrics.model) == {"lstm", "persistence", "historical_drift"}
    path = engine.save(tmp_path)
    assert (path / "training_history.xlsx").exists()
    assert len(engine.losses) == engine.metadata["epochs_completed"]
    assert engine.losses.epoch.tolist() == list(range(1, len(engine.losses) + 1))
    assert "learning_rate" in engine.losses
    assert engine.metadata["best_epoch"] == int(engine.losses.validation_mse.idxmin()) + 1
    parts = engine.metadata["partitions"]
    assert sum(part["observations"] for part in parts.values()) == engine.metadata["eligible_feature_rows"]
    assert parts["train"]["last_target"] < parts["validation"]["first_target"]
    from market_dss.forecasting import load_model_artifacts
    artifacts = load_model_artifacts(path)
    assert artifacts["training"].learning_rate.tolist() == engine.losses.learning_rate.tolist()
    loaded = ForecastEngine.load(path)
    np.testing.assert_allclose(loaded.predict(synthetic_history, simulations=100).paths, forecast.paths)
    with pytest.raises(ValueError, match="after"):
        loaded.predict(synthetic_history.iloc[:-10], simulations=100)
    forecast.export(tmp_path / "forecast.xlsx")
    from market_dss.forecasting import Forecast
    restored = Forecast.from_excel(tmp_path / "forecast.xlsx")
    np.testing.assert_allclose(restored.paths, forecast.paths)
    assert restored.dates == forecast.dates
    assert restored.metadata["instrument_id"] == forecast.metadata["instrument_id"]
    from market_dss.calendar import parse_date
    from market_dss.options import OptionLeg, Strategy
    from market_dss.risk import evaluate_strategy

    contract = contract_factory(
        strike=forecast.metadata["spot"],
        expiration=forecast.dates[-1],
        trading_date=parse_date(forecast.metadata["input_data_cutoff"]),
    )
    strategy = Strategy(
        "forecast integration", [OptionLeg(contract, 1, contract.ask, entry_fee=2, exit_fee=1)]
    )
    metrics, pnl = evaluate_strategy(strategy, forecast, 5)
    np.testing.assert_allclose(pnl, strategy.payoff(forecast.paths[:, -1]))
    assert metrics["expected_return_pct"] == pytest.approx(100 * pnl.mean() / strategy.capital)
    assert metrics["valuation"] == "expiration"


def test_insufficient_data_explicit(synthetic_history):
    with pytest.raises(InsufficientData):
        ForecastEngine().fit(synthetic_history.head(50))
