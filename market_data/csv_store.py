# توضیح فایل: ذخیرهٔ اتمیک CSV پارتیشن‌بندی‌شده و وضعیت دریافت برای جلوگیری از تخریب تاریخچه.
# معرفی توابع و متدها:
# - _code: اعتبارسنجی شناسه برای استفادهٔ امن در مسیر فایل.
# - _write_bytes_atomic: نوشتن متن در فایل موقت و جایگزینی اتمیک.
# - _csv_atomic: ذخیرهٔ CSV با فایل موقت، flush و جایگزینی اتمیک.
# - __init__: مقداردهی اولیهٔ نمونه و وابستگی‌های آن.
# - daily_path: ساخت مسیر پارتیشن ماهانهٔ روزانه.
# - hourly_path: ساخت مسیر پارتیشن یک روز ساعتی.
# - state_path: ساخت مسیر فایل وضعیت شناسه.
# - read_state: خواندن آخرین وضعیت دریافت نماد.
# - update_state: ادغام و ثبت اتمیک وضعیت دریافت.
# - upsert_daily: افزودن روزهای جدید و اصلاح کلیدهای موجود بدون حذف سایر تاریخچه.
# - hourly_failure_path: ساخت مسیر گزارش شکست دریافت یک روز.
# - record_hourly_failure: ثبت علت شکست و زمان مجاز تلاش مجدد.
# - should_fetch_hourly: تصمیم دریافت مجدد با بررسی فایل، کیفیت و مهلت retry.
# - hourly_exists: بررسی وجود پارتیشن ساعتی و در صورت درخواست کیفیت آن.
# - save_hourly_day: ثبت اتمیک کندل‌ها و فرادادهٔ کیفیت یک روز.
# - _iter_csv: خواندن تکه‌ای پارتیشن‌های مرتبط با پنجرهٔ زمانی.
# - iter_daily: پیمایش تکه‌های روزانه.
# - iter_hourly: پیمایش تکه‌های ساعتی.
# - load_daily: بارگذاری روزانه در DataFrame.
# - load_hourly: بارگذاری ساعتی در DataFrame.

"""CSV-only persistence using small partitions and atomic upserts.

Daily: one CSV per instrument/month (unique key insCode, Date)
Hourly: one CSV per instrument/day (unique key insCode, Date, Hour)
Only touched partitions are read during updates.
"""
from __future__ import annotations
from collections.abc import Iterator
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import os
import re
import tempfile
import time

import pandas as pd


def _code(ins_code: str) -> str:
    code = str(ins_code)
    if not re.fullmatch(r"\d{5,25}", code):
        raise ValueError("insCode must contain 5–25 digits")
    return code


def _replace_atomic(temporary,path):
    # Windows readers/antivirus may briefly deny replacement; preserve the old file and retry.
    for attempt in range(8):
        try:
            os.replace(temporary,path)
            return
        except PermissionError:
            if attempt==7:raise
            time.sleep(.05*(attempt+1))

def _write_bytes_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp_", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_atomic(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
            df.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_atomic(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CsvMarketStore:
    """Storage adapter; swap this class for SQL/Parquet later."""
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def daily_path(self, ins_code: str, month: str) -> Path:
        if not re.fullmatch(r"\d{4}-\d{2}", month):
            raise ValueError("Invalid year-month")
        return self.root / "daily" / _code(ins_code) / f"{month}.csv"

    def hourly_path(self, ins_code: str, date: pd.Timestamp) -> Path:
        return self.root / "hourly" / _code(ins_code) / f"{pd.Timestamp(date):%Y-%m-%d}.csv"

    def state_path(self, ins_code: str) -> Path:
        return self.root / "state" / f"{_code(ins_code)}.json"

    def read_state(self, ins_code: str) -> dict:
        path = self.state_path(ins_code)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def update_state(self, ins_code: str, **updates) -> None:
        state = self.read_state(ins_code)
        state.update(updates)
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_bytes_atomic(self.state_path(ins_code), json.dumps(state, ensure_ascii=False, indent=2))

    def upsert_daily(self, rows: pd.DataFrame) -> dict:
        """Merge only the months in `rows`; never load every historical CSV.

        For each touched month, the latest validated API observations win.
        """
        if rows.empty:
            return {"months_touched": 0, "incoming_rows": 0, "new_rows": 0}
        input_df = rows.copy()
        input_df["insCode"] = input_df["insCode"].astype("string")
        input_df["Date"] = pd.to_datetime(input_df["Date"]).dt.normalize()
        input_df = input_df.dropna(subset=["Date", "insCode"])
        input_df = input_df.drop_duplicates(["insCode", "Date"], keep="last")
        input_df["_month"] = input_df["Date"].dt.strftime("%Y-%m")
        touched, new_count = 0, 0
        for (code, month), part in input_df.groupby(["insCode", "_month"], sort=True):
            path = self.daily_path(str(code), str(month))
            new_part = part.drop(columns="_month")
            if path.exists():
                old = pd.read_csv(path, dtype={"insCode": "string"}, parse_dates=["Date"])
                old_keys = set(zip(old["insCode"].astype(str), old["Date"].dt.strftime("%Y-%m-%d")))
                new_count += sum((str(code), d) not in old_keys for d in new_part["Date"].dt.strftime("%Y-%m-%d"))
                combined = pd.concat([old, new_part], ignore_index=True)
            else:
                new_count += len(new_part)
                combined = new_part
            combined["Date"] = pd.to_datetime(combined["Date"])
            combined = combined.drop_duplicates(["insCode", "Date"], keep="last")
            combined = combined.sort_values(["insCode", "Date"]).reset_index(drop=True)
            _csv_atomic(combined, path)
            touched += 1
        return {"months_touched": touched, "incoming_rows": len(input_df), "new_rows": new_count}

    def hourly_failure_path(self, ins_code: str, date: pd.Timestamp) -> Path:
        return self.hourly_path(ins_code, date).with_suffix(".error.json")

    def record_hourly_failure(self, ins_code: str, date: pd.Timestamp, error: str, *, retry_after_hours: int = 24) -> None:
        path = self.hourly_failure_path(ins_code, date)
        payload = {
            "error": error[:500],
            "tried_at_utc": datetime.now(timezone.utc).isoformat(),
            "retry_after_utc": (datetime.now(timezone.utc) + timedelta(hours=retry_after_hours)).isoformat(),
        }
        _write_bytes_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2))

    def should_fetch_hourly(self, ins_code: str, date: pd.Timestamp, *, retry_incomplete: bool = False) -> bool:
        if self.hourly_exists(ins_code, date, retry_incomplete=retry_incomplete):
            return False
        if retry_incomplete:
            return True
        failure = self.hourly_failure_path(ins_code, date)
        if failure.exists():
            try:
                retry_at = datetime.fromisoformat(json.loads(failure.read_text(encoding="utf-8"))["retry_after_utc"])
                if datetime.now(timezone.utc) < retry_at:
                    return False
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                pass
        return True

    def hourly_exists(self, ins_code: str, date: pd.Timestamp, *, retry_incomplete: bool = False) -> bool:
        """True when the day can be skipped during the next fetch run."""
        path = self.hourly_path(ins_code, date)
        meta = path.with_suffix(".meta.json")
        if not (path.exists() and meta.exists()):
            return False
        if not retry_incomplete:
            return True
        try:
            return json.loads(meta.read_text(encoding="utf-8")).get("data_quality") == "matched"
        except (json.JSONDecodeError, OSError):
            return False

    def save_hourly_day(self, rows: pd.DataFrame) -> None:
        if rows.empty:
            return
        codes = rows["insCode"].astype(str).unique()
        dates = pd.to_datetime(rows["Date"]).dt.normalize().unique()
        if len(codes) != 1 or len(dates) != 1:
            raise ValueError("save_hourly_day expects exactly one instrument-day")
        path = self.hourly_path(codes[0], pd.Timestamp(dates[0]))
        existing_meta = path.with_suffix('.meta.json')
        if path.exists() and existing_meta.exists():
            old_quality = json.loads(existing_meta.read_text(encoding='utf-8')).get('data_quality')
            if old_quality == 'matched' and rows['DataQuality'].iloc[0] != 'matched':
                raise ValueError('Refusing to replace a matched hourly day with invalid volume coverage')
        saved = rows.copy()
        saved["Hour"] = pd.to_datetime(saved["Hour"])
        saved["Date"] = pd.to_datetime(saved["Date"]).dt.normalize()
        saved = (saved.drop_duplicates(["insCode", "Date", "Hour"], keep="last")
                 .sort_values(["insCode", "Date", "Hour"]))
        _csv_atomic(saved, path)
        meta = {
            "response_parts": int(saved['ResponseParts'].iloc[0]) if 'ResponseParts' in saved else 1,
            "data_quality": str(saved["DataQuality"].iloc[0]),
            "volume_coverage": None if pd.isna(saved["VolumeCoverage"].iloc[0]) else float(saved["VolumeCoverage"].iloc[0]),
            "rows": len(saved),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_bytes_atomic(path.with_suffix(".meta.json"), json.dumps(meta, ensure_ascii=False, indent=2))
        self.hourly_failure_path(codes[0], pd.Timestamp(dates[0])).unlink(missing_ok=True)

    def _iter_csv(
        self,
        frequency: str,
        ins_code: str,
        *,
        start: str | None = None,
        end: str | None = None,
        chunksize: int = 10000,
    ) -> Iterator[pd.DataFrame]:
        ins_code = _code(ins_code)
        folder = self.root / frequency / ins_code
        if not folder.exists():
            return
        start_ts = pd.Timestamp(start).normalize() if start else None
        end_ts = pd.Timestamp(end).normalize() if end else None
        for path in sorted(folder.glob("*.csv")):
            if frequency == "daily":
                # Skip unrelated month files before ever opening them.
                month = pd.Timestamp(path.stem + "-01")
                month_end = month + pd.offsets.MonthEnd(0)
                if start_ts is not None and month_end < start_ts:
                    continue
                if end_ts is not None and month > end_ts:
                    continue
            else:
                day = pd.Timestamp(path.stem)
                if start_ts is not None and day < start_ts:
                    continue
                if end_ts is not None and day > end_ts:
                    continue
            parse = ["Date", "Hour"] if frequency == "hourly" else ["Date"]
            for chunk in pd.read_csv(
                path, chunksize=chunksize,
                dtype={"insCode": "string", "Symbol": "string"},
                parse_dates=parse,
            ):
                mask = pd.Series(True, index=chunk.index)
                if start_ts is not None:
                    mask &= chunk["Date"] >= start_ts
                if end_ts is not None:
                    mask &= chunk["Date"] <= end_ts
                filtered = chunk.loc[mask]
                if not filtered.empty:
                    yield filtered

    def iter_daily(self, ins_code: str, **kwargs) -> Iterator[pd.DataFrame]:
        return self._iter_csv("daily", ins_code, **kwargs)

    def iter_hourly(self, ins_code: str, **kwargs) -> Iterator[pd.DataFrame]:
        return self._iter_csv("hourly", ins_code, **kwargs)

    def load_daily(self, ins_code: str, **kwargs) -> pd.DataFrame:
        parts = list(self.iter_daily(ins_code, **kwargs))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def load_hourly(self, ins_code: str, **kwargs) -> pd.DataFrame:
        parts = list(self.iter_hourly(ins_code, **kwargs))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
