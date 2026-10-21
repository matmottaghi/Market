import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler


class JsonFormatter(logging.Formatter):
    def format(self, record):
        # Provider exceptions are sanitized at their boundary; never log requests/URLs/keys.
        return json.dumps(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            },
            ensure_ascii=False,
        )


def configure_logging(path, level="INFO"):
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=4, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("market_dss")
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
