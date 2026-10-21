"""Atomic, locked Excel snapshots. Raw inputs are immutable, content-addressed archives."""

from contextlib import suppress
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tempfile

from filelock import FileLock
import pandas as pd

KEYS = {
    "instruments": ["instrument_id", "source"],
    "prices": ["instrument_id", "trading_date", "source"],
    "price_versions": ["instrument_id", "trading_date", "source", "retrieved_at"],
    "nav": ["instrument_id", "trading_date", "source"],
    "clients": ["instrument_id", "trading_date", "source"],
    "options": ["instrument_id", "retrieved_at", "source"],
    "snapshots": ["instrument_id", "retrieved_at", "source"],
}


def excel_bytes(tables, max_rows=1_000_000):
    if not 1 <= max_rows <= 1_048_575:
        raise ValueError("Excel rows per sheet must be between 1 and 1,048,575")
    stream = BytesIO()
    with pd.ExcelWriter(
        stream,
        engine="xlsxwriter",
        engine_kwargs={
            "options": {
                "strings_to_formulas": False,
                "strings_to_urls": False,
            }
        },
    ) as writer:
        for name, frame in tables.items():
            frame = frame.copy()
            for col in frame:
                frame[col] = frame[col].map(
                    lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                )
                if frame[col].map(lambda v: isinstance(v, str) and len(v) > 32767).any():
                    raise ValueError(f"Excel cell too long: {name}.{col}; raw archive remains intact")
            for part, start in enumerate(range(0, max(len(frame), 1), max_rows)):
                sheet = f"{name[:24]}__{part:04d}"
                frame.iloc[start : start + max_rows].to_excel(writer, sheet_name=sheet, index=False)
                writer.sheets[sheet].freeze_panes(1, 0)
    return stream.getvalue()


def atomic_bytes(path, content, backup=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if backup and path.exists():
            # The backup is also atomic, so interruption never destroys the previous backup.
            atomic_bytes(path.with_suffix(path.suffix + ".bak"), path.read_bytes())
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


class ExcelRepository:
    def __init__(self, path, max_rows=1_000_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path) + ".lock", timeout=60)
        self.max_rows = max_rows

    def _read_all(self):
        if not self.path.exists():
            return {}
        result = {}
        # Strings preserve identifiers and original representations exactly.
        for sheet, frame in pd.read_excel(self.path, sheet_name=None, dtype=str).items():
            name = sheet.rsplit("__", 1)[0]
            result[name] = pd.concat([result.get(name, pd.DataFrame()), frame], ignore_index=True)
        return result

    def read_all(self):
        with self.lock:
            return self._read_all()

    def read(self, name):
        return self.read_all().get(name, pd.DataFrame())

    def upsert_many(self, tables):
        with self.lock:
            current = self._read_all()
            for name, frame in tables.items():
                if frame.empty:
                    continue
                keys = KEYS[name]
                if any(k not in frame or frame[k].isna().any() for k in keys):
                    raise ValueError(f"Missing primary key in {name}: {keys}")
                incoming = frame.copy()
                for key in keys:
                    incoming[key] = incoming[key].astype(str)
                if name == "prices":
                    if "retrieved_at" not in incoming:
                        raise ValueError("Prices require a retrieval timestamp for vintage tracking")
                    versions = pd.concat(
                        [current.get("price_versions", current.get("prices", pd.DataFrame())), incoming],
                        ignore_index=True,
                    )
                    current["price_versions"] = versions.drop_duplicates(KEYS["price_versions"], keep="last")
                merged = pd.concat([current.get(name, pd.DataFrame()), incoming], ignore_index=True)
                # Sparse refreshes cannot erase previously observed optional fields.
                current[name] = merged.groupby(keys, dropna=False, sort=False, as_index=False).last()
            if current:
                atomic_bytes(self.path, excel_bytes(current, self.max_rows), backup=True)
        return {name: len(frame) for name, frame in current.items()}

    def export(self, destination):
        with self.lock:
            if not self.path.exists():
                raise FileNotFoundError("No market data imported")
            atomic_bytes(destination, self.path.read_bytes())

    def archive_file(self, source, raw_dir):
        content = Path(source).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        target = Path(raw_dir) / f"{digest}{Path(source).suffix}"
        if not target.exists():
            atomic_bytes(target, content)
        return target

    def restore_backup(self):
        with self.lock:
            backup = self.path.with_suffix(self.path.suffix + ".bak")
            pd.read_excel(backup, engine="openpyxl", nrows=1)
            shutil.copyfile(backup, self.path.with_suffix(".recovery.xlsx"))
            os.replace(self.path.with_suffix(".recovery.xlsx"), self.path)
