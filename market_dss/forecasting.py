"""Direct multi-session LSTM with MC dropout and held-out residual-vector resampling.

Residual vectors retain cross-horizon dependence and heavy tails. They mix market
innovation and model error; dropout is only an approximate epistemic component.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import monotonic
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .calendar import TradingCalendar, parse_date, utc_timestamp
from .features import prepare_model_frame
from .storage import atomic_bytes, excel_bytes


TRAINING_SCHEMA_VERSION = 2


class InsufficientData(ValueError):
    pass


class ModelConfig(BaseModel):
    window: int = Field(default=30, ge=5)
    horizon: int = Field(default=21, ge=1, le=252)
    hidden_size: int = Field(default=32, ge=4)
    layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.15, gt=0, lt=1)
    attention: bool = False
    learning_rate: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.0001, ge=0)
    lr_patience: int = Field(default=3, ge=1)
    lr_factor: float = Field(default=0.5, gt=0, lt=1)
    epochs: int = Field(default=60, ge=1)
    patience: int = Field(default=8, ge=1)
    batch_size: int = Field(default=32, ge=1)
    seed: int = 42
    annual_sessions: int = Field(default=252, ge=1)
    device: str = "cpu"


class ReturnLSTM(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        self.lstm = nn.LSTM(
            features,
            config.hidden_size,
            config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0,
        )
        self.attention = nn.Linear(config.hidden_size, 1) if config.attention else None
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.hidden_size, config.horizon))

    def forward(self, x):
        sequence, _ = self.lstm(x)
        hidden = (
            (self.attention(sequence).softmax(dim=1) * sequence).sum(dim=1)
            if self.attention is not None
            else sequence[:, -1]
        )
        return self.head(hidden)


def chronological_origins(n, window, horizon):
    boundaries = [int(n * 0.60), int(n * 0.70), int(n * 0.85), n]
    result, start = {}, window
    for name, end in zip(("train", "validation", "calibration", "test"), boundaries):
        # Origin o sees rows <= o; every target belongs entirely to this partition.
        origins = np.arange(max(window - 1, start - 1), end - horizon)
        if len(origins) < (20 if name == "train" else 8):
            raise InsufficientData(f"Insufficient {name} samples for window={window}, horizon={horizon}")
        result[name] = origins
        start = end
    return result, boundaries


def probabilistic_metrics(actual, samples, spot):
    """Samples: [observation, simulation]; empirical CRPS in price units."""
    actual, samples, spot = np.asarray(actual), np.asarray(samples), np.asarray(spot)
    mean = samples.mean(axis=1)
    q05, q95 = np.quantile(samples, [0.05, 0.95], axis=1)
    ordered = np.sort(samples, axis=1)
    m = samples.shape[1]
    crps = (
        np.abs(samples - actual[:, None]).mean(axis=1)
        - (ordered * (2 * np.arange(1, m + 1) - m - 1)).sum(axis=1) / m**2
    )
    quantile_loss = []
    calibration = {}
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        error = actual - np.quantile(samples, q, axis=1)
        quantile_loss.append(np.maximum(q * error, (q - 1) * error).mean())
        calibration[f"coverage_q{q:g}"] = float((error <= 0).mean())
    return {
        "mae": float(np.abs(actual - mean).mean()),
        "rmse": float(np.sqrt(((actual - mean) ** 2).mean())),
        "directional_accuracy": float((np.sign(mean - spot) == np.sign(actual - spot)).mean()),
        "interval_90_coverage": float(((actual >= q05) & (actual <= q95)).mean()),
        "interval_90_width": float((q95 - q05).mean()),
        "quantile_loss": float(np.mean(quantile_loss)),
        "crps": float(crps.mean()),
        **calibration,
    }


@dataclass
class Forecast:
    paths: np.ndarray
    dates: list
    metadata: dict
    epistemic_log_return_std: np.ndarray

    def summary(self, confidence=0.90, target=None):
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between 0 and 1")
        lower, upper = (1 - confidence) / 2, (1 + confidence) / 2
        spot = self.metadata["spot"]
        data = {
            "date": [str(d) for d in self.dates],
            "expected": self.paths.mean(axis=0),
            "median": np.median(self.paths, axis=0),
            "expected_return_pct": 100 * (self.paths.mean(axis=0) / spot - 1),
            "median_return_pct": 100 * (np.median(self.paths, axis=0) / spot - 1),
            "lower": np.quantile(self.paths, lower, axis=0),
            "upper": np.quantile(self.paths, upper, axis=0),
            "probability_up": (self.paths > spot).mean(axis=0),
            "probability_down": (self.paths < spot).mean(axis=0),
            "cumulative_log_return_std": np.log(self.paths / spot).std(axis=0),
            "epistemic_log_return_std": self.epistemic_log_return_std,
        }
        for q in (0.05, 0.25, 0.75, 0.95):
            data[f"q{int(q * 100):02d}"] = np.quantile(self.paths, q, axis=0)
        if target is not None:
            data["probability_above_target"] = (self.paths > target).mean(axis=0)
        return pd.DataFrame(data)

    def export(self, destination):
        atomic_bytes(
            destination,
            excel_bytes(
                {
                    "forecast": self.summary(),
                    "metadata": pd.DataFrame([self.metadata]),
                    "paths": pd.DataFrame(self.paths.T, index=self.dates).reset_index(),
                }
            ),
        )

    @classmethod
    def from_excel(cls, path):
        reports = pd.read_excel(path, sheet_name=None, dtype={"instrument_id": str})
        metadata = reports["metadata__0000"].iloc[0].to_dict()
        metadata["instrument_id"] = str(metadata["instrument_id"])
        summary = reports["forecast__0000"]
        paths_frame = reports["paths__0000"]
        paths = paths_frame.iloc[:, 1:].to_numpy(dtype=float).T
        if paths.shape[1] != len(summary) or not np.isfinite(paths).all() or (paths <= 0).any():
            raise ValueError("Invalid persisted forecast paths")
        return cls(paths, [parse_date(d) for d in summary.date], metadata,
                   summary.epistemic_log_return_std.to_numpy(dtype=float))


class ForecastEngine:
    def __init__(self, config=None):
        self.config = config or ModelConfig()
        self.model = None
        self.metadata = {}

    def _seed(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _prepare(self, history):
        return prepare_model_frame(history, self.config.annual_sessions, getattr(self, "feature_names", None))

    def fit(self, history):
        started = monotonic()
        self._seed()
        frame, self.feature_names = self._prepare(history)
        c = self.config
        if len(frame) < 180:
            raise InsufficientData("At least 180 complete feature rows required; longer horizons need more")
        groups, boundaries = chronological_origins(len(frame), c.window, c.horizon)
        self.scaler = StandardScaler().fit(frame[self.feature_names].iloc[: boundaries[0]])
        transformed = self.scaler.transform(frame[self.feature_names]).astype(np.float32)
        returns = frame.log_return.to_numpy()
        self.target_scale = max(float(np.std(returns[: boundaries[0]])), 1e-5)

        def tensors(origins):
            x = np.stack([transformed[o - c.window + 1 : o + 1] for o in origins])
            y = np.stack([returns[o + 1 : o + 1 + c.horizon] for o in origins]) / self.target_scale
            return torch.tensor(x), torch.tensor(y, dtype=torch.float32)

        train_x, train_y = tensors(groups["train"])
        val_x, val_y = tensors(groups["validation"])
        self.model = ReturnLSTM(len(self.feature_names), c).to(c.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=c.lr_factor, patience=c.lr_patience, min_lr=1e-6)
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=c.batch_size, shuffle=False)
        best_loss, best_state, stale, losses = float("inf"), None, 0, []
        for epoch in range(c.epochs):
            self.model.train()
            train_loss_sum, train_count = 0.0, 0
            epoch_lr = optimizer.param_groups[0]["lr"]
            for x, y in loader:
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.model(x.to(c.device)), y.to(c.device))
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss_sum += loss.item() * len(x)
                train_count += len(x)
            self.model.eval()
            with torch.no_grad():
                val_loss = nn.functional.mse_loss(self.model(val_x.to(c.device)), val_y.to(c.device)).item()
            losses.append(
                {"epoch": epoch + 1, "train_mse": train_loss_sum / train_count,
                 "validation_mse": val_loss, "learning_rate": epoch_lr}
            )
            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= c.patience:
                break
        training_duration = monotonic() - started
        self.model.load_state_dict(best_state)
        cal_x, cal_y = tensors(groups["calibration"])
        self.model.eval()
        with torch.no_grad():
            self.residuals = (
                (cal_y.numpy() - self.model(cal_x.to(c.device)).cpu().numpy()) * self.target_scale
            ).astype(np.float64)
        self.train_returns = returns[: boundaries[0]].copy()
        timestamp = datetime.now(timezone.utc)
        self.metadata = {
            "version": timestamp.strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8],
            "instrument_id": str(frame.instrument_id.iloc[-1]),
            "symbol": str(frame.symbol.iloc[-1]),
            "source": str(frame.source.iloc[-1]),
            "created_at": timestamp.isoformat(),
            "input_data_cutoff": str(frame.trading_date.iloc[-1]),
            "price_basis": str(frame.price_basis.iloc[-1]),
            "scaler_fit_end": str(frame.trading_date.iloc[boundaries[0] - 1]),
            "training_target_end": str(frame.trading_date.iloc[groups["train"][-1] + c.horizon]),
            "calibration_end": str(frame.trading_date.iloc[boundaries[2] - 1]),
            "test_start": str(frame.trading_date.iloc[boundaries[2]]),
            "horizon": c.horizon,
            "data_sha256": hashlib.sha256(history.to_csv(index=False).encode()).hexdigest(),
            "uncertainty_method": "MC dropout + held-out empirical residual vectors",
            "residual_vectors": len(self.residuals),
            "test_status": "out_of_sample_chronological",
            "training_schema_version": TRAINING_SCHEMA_VERSION,
            "input_observations": len(history), "eligible_feature_rows": len(frame),
            "preprocessing_removed": len(history) - len(frame),
            "best_epoch": int(np.argmin([row["validation_mse"] for row in losses])) + 1,
            "epochs_completed": len(losses), "best_validation_loss": best_loss,
            "early_stopping_epoch": len(losses) if len(losses) < c.epochs else None,
            "training_duration_seconds": training_duration,
            "loss_description": "MSE on standardized future log returns; dimensionless, not rial",
            "partitions": {
                name: {"start": str(frame.trading_date.iloc[start]),
                       "end": str(frame.trading_date.iloc[end - 1]),
                       "observations": end - start, "samples": len(groups[name]),
                       "observation_pct": 100 * (end - start) / len(frame),
                       "first_target": str(frame.trading_date.iloc[groups[name][0] + 1]),
                       "last_target": str(frame.trading_date.iloc[groups[name][-1] + c.horizon])}
                for name, start, end in zip(groups, [0, *boundaries[:-1]], boundaries)
            },
        }
        flat = self.residuals.ravel()
        normal_parameters = stats.norm.fit(flat)
        student_parameters = stats.t.fit(flat)
        self.distribution_diagnostics = {
            "residual_skew": float(stats.skew(flat)),
            "residual_excess_kurtosis": float(stats.kurtosis(flat)),
            "normality_pvalue": float(stats.normaltest(flat).pvalue)
            if len(flat) >= 8 and np.std(flat) > 0
            else None,
            "normal_bic": float(
                2 * np.log(len(flat)) - 2 * stats.norm.logpdf(flat, *normal_parameters).sum()
            ),
            "student_t_bic": float(
                3 * np.log(len(flat)) - 2 * stats.t.logpdf(flat, *student_parameters).sum()
            ),
            "note": "Diagnostic only: overlapping vectors are dependent; empirical distribution is used regardless.",
        }
        test_x, test_y = tensors(groups["test"])
        rng = np.random.default_rng(c.seed)
        metrics, predictions = [], []
        # Test distribution uses dropout, exactly as live forecasting does.
        draws = []
        self.model.train()
        with torch.no_grad():
            for _ in range(30):
                draws.append(self.model(test_x.to(c.device)).cpu().numpy() * self.target_scale)
        self.model.eval()
        draws = np.stack(draws, axis=1)
        steps = (
            draws[:, rng.integers(0, 30, 200)]
            + self.residuals[rng.integers(0, len(self.residuals), (len(test_x), 200))]
        )
        spots = frame.model_price.iloc[groups["test"]].to_numpy()
        actual = spots[:, None] * np.exp(np.cumsum(test_y.numpy() * self.target_scale, axis=1))
        paths = spots[:, None, None] * np.exp(np.cumsum(steps, axis=2))
        # Baseline physical distributions: persistence bootstrap and historical drift bootstrap.
        bootstrap = rng.choice(self.train_returns, size=(len(spots), 200, c.horizon))
        baseline_paths = {
            "persistence": spots[:, None, None]
            * np.exp(np.cumsum(bootstrap - self.train_returns.mean(), axis=2)),
            "historical_drift": spots[:, None, None] * np.exp(np.cumsum(bootstrap, axis=2)),
        }
        for horizon in sorted({1, min(5, c.horizon), min(21, c.horizon), c.horizon}):
            for name, candidate in {"lstm": paths, **baseline_paths}.items():
                result = probabilistic_metrics(actual[:, horizon - 1], candidate[:, :, horizon - 1], spots)
                # Naive point persistence is exactly the cutoff price, separately from its bootstrap intervals.
                if name == "persistence":
                    result["mae"] = float(np.abs(actual[:, horizon - 1] - spots).mean())
                    result["rmse"] = float(np.sqrt(((actual[:, horizon - 1] - spots) ** 2).mean()))
                    result["directional_accuracy"] = float((actual[:, horizon - 1] == spots).mean())
                metrics.append({"model": name, "horizon": horizon, **result})
            for i, origin in enumerate(groups["test"]):
                predictions.append(
                    {
                        "origin": str(frame.trading_date.iloc[origin]),
                        "target_date": str(frame.trading_date.iloc[origin + horizon]),
                        "horizon": horizon,
                        "actual": actual[i, horizon - 1],
                        "expected": paths[i, :, horizon - 1].mean(),
                        "q05": np.quantile(paths[i, :, horizon - 1], 0.05),
                        "q95": np.quantile(paths[i, :, horizon - 1], 0.95),
                        "split": "out_of_sample",
                        "price_basis": self.metadata["price_basis"],
                    }
                )
        self.metrics, self.predictions, self.losses = (
            pd.DataFrame(metrics),
            pd.DataFrame(predictions),
            pd.DataFrame(losses),
        )
        self.metadata["fit_duration_seconds"] = monotonic() - started
        return self

    def predict(
        self, history, horizon=None, simulations=2000, volatility_scale=1.0, calendar=None, seed=None
    ):
        if self.model is None:
            raise ValueError("Train or load a model first")
        c = self.config
        horizon = horizon or c.horizon
        if not 1 <= horizon <= c.horizon or not 100 <= simulations <= 50000 or volatility_scale <= 0:
            raise ValueError("Invalid horizon, simulation count or volatility scenario")
        frame, _ = self._prepare(history)
        if len(frame) < c.window:
            raise InsufficientData("Insufficient prediction window")
        if (
            str(frame.instrument_id.iloc[-1]) != self.metadata["instrument_id"]
            or str(frame.source.iloc[-1]) != self.metadata["source"]
            or str(frame.price_basis.iloc[-1]) != self.metadata["price_basis"]
        ):
            raise ValueError("Instrument, source or price basis does not match this model")
        if str(frame.trading_date.iloc[-1]) < self.metadata["input_data_cutoff"]:
            raise ValueError("Model uses information after requested prediction cutoff")
        seed = c.seed if seed is None else seed
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        x = torch.tensor(
            self.scaler.transform(frame[self.feature_names].iloc[-c.window :]), dtype=torch.float32
        )
        self.model.train()
        with torch.no_grad():
            # 128 independent dropout draws; residual sampling adds market-return variation.
            means = self.model(x[None].repeat(128, 1, 1).to(c.device)).cpu().numpy() * self.target_scale
        self.model.eval()
        steps = (
            means[rng.integers(0, len(means), simulations)]
            + self.residuals[rng.integers(0, len(self.residuals), simulations)]
        )
        center = means.mean(axis=0) + self.residuals.mean(axis=0)
        steps = center + (steps - center) * volatility_scale
        # Anchor adjusted return scenarios to the actual latest tradable underlying price.
        spot = float(frame.close.iloc[-1])
        paths = spot * np.exp(np.cumsum(steps[:, :horizon], axis=1))
        if not np.isfinite(paths).all() or (paths <= 0).any():
            raise ValueError("Nonfinite simulated path; model/scenario is numerically invalid")
        calendar = calendar or TradingCalendar()
        dates = calendar.next_sessions(frame.trading_date.iloc[-1], horizon)
        metadata = {
            **self.metadata,
            "generated_at": utc_timestamp(),
            "input_data_cutoff": str(frame.trading_date.iloc[-1]),
            "spot": spot,
            "horizon": horizon,
            "simulations": simulations,
            "seed": seed,
            "volatility_scale": volatility_scale,
            "calendar": calendar.description,
            "distribution": "physical predictive scenarios; not risk neutral",
        }
        return Forecast(paths, dates, metadata, np.cumsum(means[:, :horizon], axis=1).std(axis=0))

    def save(self, root):
        import re

        if not re.fullmatch(r"[\w-]{1,128}", self.metadata["instrument_id"]):
            raise ValueError("Instrument ID is not a safe checkpoint directory name")
        directory = Path(root) / self.metadata["instrument_id"] / self.metadata["version"]
        directory.mkdir(parents=True, exist_ok=False)
        # State dictionaries only. Load uses weights_only=True; no arbitrary pickle model objects.
        from io import BytesIO

        buffer = BytesIO()
        torch.save({k: v.cpu() for k, v in self.model.state_dict().items()}, buffer)
        atomic_bytes(directory / "weights.pt", buffer.getvalue())
        info = {
            "config": self.config.model_dump(),
            "metadata": self.metadata,
            "features": self.feature_names,
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "target_scale": self.target_scale,
            "residuals": self.residuals.tolist(),
            "train_returns": self.train_returns.tolist(),
            "distribution_diagnostics": self.distribution_diagnostics,
        }
        atomic_bytes(
            directory / "evaluation.xlsx",
            excel_bytes({"metrics": self.metrics, "predictions": self.predictions, "training": self.losses}),
        )
        atomic_bytes(directory / "training_history.xlsx", excel_bytes({"training_history": self.losses.rename(
            columns={"train_mse": "train_loss", "validation_mse": "validation_loss"})}))
        # Publish the discovery marker last; incomplete directories never appear as loadable versions.
        atomic_bytes(directory / "model.json", json.dumps(info, ensure_ascii=False, indent=2).encode("utf-8"))
        return directory

    @classmethod
    def load(cls, directory, device="cpu"):
        directory = Path(directory)
        info = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        engine = cls(ModelConfig(**{**info["config"], "device": device}))
        engine.feature_names, engine.metadata = info["features"], info["metadata"]
        engine.scaler = StandardScaler()
        engine.scaler.mean_, engine.scaler.scale_ = np.array(info["mean"]), np.array(info["scale"])
        engine.scaler.var_ = engine.scaler.scale_**2
        engine.scaler.n_features_in_ = len(engine.feature_names)
        engine.scaler.feature_names_in_ = np.array(engine.feature_names, dtype=object)
        engine.target_scale, engine.residuals = info["target_scale"], np.array(info["residuals"])
        engine.train_returns = np.array(info["train_returns"])
        engine.distribution_diagnostics = info["distribution_diagnostics"]
        engine.model = ReturnLSTM(len(engine.feature_names), engine.config).to(device)
        engine.model.load_state_dict(
            torch.load(directory / "weights.pt", map_location=device, weights_only=True)
        )
        reports = pd.read_excel(directory / "evaluation.xlsx", sheet_name=None)
        engine.metrics, engine.predictions, engine.losses = (
            reports["metrics__0000"],
            reports["predictions__0000"],
            reports["training__0000"],
        )
        return engine


def walk_forward_forecasts(history, config, initial=300, step=21, simulations=200):
    records = []
    for end in range(initial, len(history) - config.horizon + 1, step):
        training = history.iloc[:end].copy()
        engine = ForecastEngine(config).fit(training)
        forecast = engine.predict(training, simulations=simulations)
        realized = history.iloc[end : end + config.horizon]
        for h in range(config.horizon):
            records.append(
                {
                    "origin": str(training.trading_date.iloc[-1]),
                    "horizon": h + 1,
                    "target_date": str(realized.trading_date.iloc[h]),
                    "actual": float(realized.close.iloc[h]),
                    "expected": forecast.paths[:, h].mean(),
                    "q05": np.quantile(forecast.paths[:, h], 0.05),
                    "q95": np.quantile(forecast.paths[:, h], 0.95),
                    "split": "walk_forward_out_of_sample",
                }
            )
    if not records:
        raise InsufficientData("No complete walk-forward folds")
    return pd.DataFrame(records)


def load_model_artifacts(directory):
    """Read durable reports/metadata only; no checkpoint loading or model inference."""
    directory = Path(directory)
    info = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    reports = pd.read_excel(directory / "evaluation.xlsx", sheet_name=None)
    losses = reports.get("training__0000", pd.DataFrame())
    history_path = directory / "training_history.xlsx"
    if history_path.exists():
        losses = pd.read_excel(history_path, sheet_name=0).rename(
            columns={"train_loss": "train_mse", "validation_loss": "validation_mse"})
    return {**info, "metrics": reports["metrics__0000"],
            "predictions": reports["predictions__0000"], "training": losses}
