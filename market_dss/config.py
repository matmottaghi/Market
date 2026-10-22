from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DSS_DATA_DIR", "data")))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("DSS_MODEL_DIR", "models")))
    collector_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DSS_COLLECTOR_DIR", "data/leveraged"))
    )
    holidays_file: Path = field(
        default_factory=lambda: Path(os.getenv("DSS_HOLIDAYS_FILE", "config/holidays.csv"))
    )
    source: str = field(default_factory=lambda: os.getenv("DSS_SOURCE", "BrsApi"))
    annual_sessions: int = field(default_factory=lambda: int(os.getenv("DSS_ANNUAL_SESSIONS", "252")))
    retrain_days: int = field(default_factory=lambda: int(os.getenv("DSS_RETRAIN_DAYS", "7")))

    default_symbols: str = field(default_factory=lambda: os.getenv("DSS_SYMBOLS", "اهرم,موج,توان"))
    training_epochs: int = field(default_factory=lambda: int(os.getenv("DSS_TRAINING_EPOCHS", "30")))

    live_max_age_hours: int = field(default_factory=lambda: int(os.getenv("DSS_LIVE_MAX_AGE_HOURS", "48")))

    @classmethod
    def from_env(cls):
        load_dotenv(encoding="utf-8-sig")
        return cls()

    def initialize(self):
        for name in (
            "raw",
            "historical",
            "processed",
            "features",
            "forecasts",
            "options",
            "backtests",
            "exports",
            "logs",
        ):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
