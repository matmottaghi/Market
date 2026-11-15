# توضیح فایل: پاکسازی تاریخ و قیمت روزانه، حذف تکرار و ساخت بازده فقط روی جلسات معتبر.
# معرفی توابع و متدها:
# - prepare_columns: کنترل ستون‌های لازم و تبدیل نوع تاریخ و مقادیر عددی.
# - remove_duplicates: نگهداری یک رکورد برای هر شناسه و روز.
# - clean_prices: علامت‌گذاری روز معاملاتی و نامعتبر کردن قیمت‌های غیرقابل استفاده.
# - select_trading_days: انتخاب جلسات دارای معامله و قیمت معتبر.
# - calculate_returns: محاسبهٔ بازده و فاصلهٔ تقویمی هر نماد به ترتیب زمان.
# - make_cleaning_pipeline: ساخت زنجیرهٔ پاکسازی سازگار با scikit-learn.
# - clean_stock_data: بازگرداندن دادهٔ پاک و زیرمجموعهٔ معتبر آموزش.

"""Daily cleaning with the same public function names as the user's notebook.

Important: calculate_returns must run on a chronologically complete requested
training window, not only on new rows in an incremental update.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

NUMERIC_COLS = [
    "priceChange", "priceMin", "priceMax", "priceYesterday",
    "priceFirst", "pClosing", "pDrCotVal", "zTotTran",
    "qTotTran5J", "qTotCap",
]


def prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    missing = set(NUMERIC_COLS + ["dEven", "insCode"]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing daily columns: {sorted(missing)}")
    data[NUMERIC_COLS] = data[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    date_text = pd.to_numeric(data["dEven"], errors="coerce").astype("Int64").astype("string")
    data["Date"] = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    data["insCode"] = data["insCode"].astype("string")
    if "Symbol" in data.columns:
        data["Symbol"] = data["Symbol"].astype("string")
    data = data.replace([np.inf, -np.inf], np.nan)
    return data.dropna(subset=["Date", "insCode"]).reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    # The daily grain is one row per instrument per trading date.
    if "insCode" not in data or "Date" not in data:
        raise ValueError("Daily data must contain insCode and Date")
    data = data.sort_values(["insCode", "Date"], kind="stable")
    return data.drop_duplicates(["insCode", "Date"], keep="last").reset_index(drop=True)


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["HasTrade"] = (data["qTotTran5J"] > 0) & (data["zTotTran"] > 0)
    price_cols = ["priceFirst", "priceMin", "priceMax", "pDrCotVal", "pClosing"]
    data[price_cols] = data[price_cols].mask(data[price_cols] <= 0)
    trade_cols = ["priceFirst", "priceMin", "priceMax", "pDrCotVal"]
    data.loc[~data["HasTrade"], trade_cols] = np.nan
    return data


def select_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    required_prices = ["pClosing", "priceFirst", "priceMin", "priceMax", "pDrCotVal"]
    data = data.loc[
        data["HasTrade"] & data[required_prices].notna().all(axis=1)
    ].copy()
    return data.reset_index(drop=True)


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.sort_values(["insCode", "Date"], kind="stable").copy()
    data["DailyReturn"] = data.groupby("insCode")["pClosing"].pct_change(fill_method=None)
    data["GapDays"] = data.groupby("insCode")["Date"].diff().dt.days
    return data.reset_index(drop=True)


def make_cleaning_pipeline(trading_only: bool = True):
    """Optional sklearn-compatible pipeline for existing calling code."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    steps = [
        ("prepare", FunctionTransformer(prepare_columns)),
        ("duplicates", FunctionTransformer(remove_duplicates)),
        ("prices", FunctionTransformer(clean_prices)),
    ]
    if trading_only:
        steps.extend([
            ("trading_days", FunctionTransformer(select_trading_days)),
            ("returns", FunctionTransformer(calculate_returns)),
        ])
    return Pipeline(steps)


def clean_stock_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (all_clean_daily, valid_sessions_for_model)."""
    if df.empty:
        return df.copy(), df.copy()
    clean = clean_prices(remove_duplicates(prepare_columns(df)))
    model = calculate_returns(select_trading_days(clean))
    return clean, model
