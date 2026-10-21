from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import re

import jdatetime
import pandas as pd

TEHRAN = ZoneInfo("Asia/Tehran")


def normalize_text(value):
    if value is None or pd.isna(value):
        return ""
    return " ".join(
        str(value)
        .translate(str.maketrans("يك٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "یک01234567890123456789"))
        .replace("\u200c", " ")
        .split()
    )


def parse_date(value) -> date:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.date()
    if isinstance(value, date):
        return value
    text = normalize_text(value)
    if re.fullmatch(r"\d{8}(\.0)?", text):
        text = text[:8]
        parts = [int(text[:4]), int(text[4:6]), int(text[6:8])]
    else:
        match = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:$|[ T])", text)
        if not match:
            raise ValueError(f"Unrecognized date: {text!r}")
        parts = [int(p) for p in match.groups()]
    return jdatetime.date(*parts).togregorian() if parts[0] < 1700 else date(*parts)


def jalali(value):
    return jdatetime.date.fromgregorian(date=parse_date(value)).strftime("%Y/%m/%d")


def utc_timestamp(value=None):
    timestamp = pd.Timestamp(value or datetime.now(TEHRAN))
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Tehran")
    return timestamp.tz_convert("UTC").isoformat()


@dataclass
class TradingCalendar:
    holidays: set[date] = field(default_factory=set)
    verified: bool = False

    @classmethod
    def from_csv(cls, path):
        if not path.exists():
            return cls()
        frame = pd.read_csv(path)
        return cls({parse_date(d) for d in frame["date"]}, verified=False)

    @property
    def description(self):
        return "Saturday–Wednesday; supplied closures applied; future sessions are estimates"

    def is_session(self, day):
        day = parse_date(day)
        return day.weekday() not in (3, 4) and day not in self.holidays

    def sessions(self, start, end):
        day, end = parse_date(start), parse_date(end)
        result = []
        while day <= end:
            if self.is_session(day):
                result.append(day)
            day += timedelta(days=1)
        return result

    def next_sessions(self, after, count):
        day, result = parse_date(after), []
        while len(result) < count:
            day += timedelta(days=1)
            if self.is_session(day):
                result.append(day)
        return result
