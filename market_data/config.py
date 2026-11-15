# توضیح فایل: خواندن تنظیمات منبع داده و مسیر CSV از محیط و اعتبارسنجی آن‌ها.
# معرفی توابع و متدها:
# - from_env: ساخت و بررسی تنظیمات از متغیرهای محیطی و فایل env.

"""Configuration read from environment variables, with optional .env support."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    root: Path
    base_url: str
    timeout: float
    delay: float
    retries: int
    reconcile_days: int
    volume_tolerance: float
    hourly_max_days: int
    symbols: tuple[str, ...]

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Settings":
        load_dotenv(dotenv_path=env_file, override=False)
        symbols = tuple(
            x.strip() for x in os.getenv("MARKET_SYMBOLS", "اهرم,موج,توان").split(",")
            if x.strip()
        )
        result = cls(
            root=Path(os.getenv("MARKET_CSV_ROOT", "data/market_csv")).expanduser(),
            base_url=os.getenv("TSETMC_BASE_URL", "https://cdn.tsetmc.com/api").rstrip("/"),
            timeout=float(os.getenv("TSETMC_TIMEOUT", "40")),
            delay=float(os.getenv("TSETMC_REQUEST_DELAY", "0.6")),
            retries=int(os.getenv("TSETMC_RETRIES", "3")),
            reconcile_days=int(os.getenv("MARKET_RECONCILE_DAYS", "14")),
            volume_tolerance=float(os.getenv("MARKET_VOLUME_TOLERANCE", "0.001")),
            hourly_max_days=int(os.getenv("MARKET_HOURLY_MAX_DAYS", "0")),
            symbols=symbols,
        )
        if not 0 <= result.volume_tolerance < 1:
            raise ValueError("MARKET_VOLUME_TOLERANCE must be in [0,1)")
        if result.reconcile_days < 0 or result.hourly_max_days < 0 or result.delay < 0:
            raise ValueError("Reconciliation days, hourly limit and delay must be >= 0")
        return result
