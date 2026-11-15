# توضیح فایل: آزمون دریافت افزایشی، کلیدهای یکتا، تجمیع ساعتی و ادامه پس از خطای منبع.
# معرفی توابع و متدها:
# - settings: تابع کمکی دادهٔ آزمایشی: settings
# - sample_daily: تابع کمکی دادهٔ آزمایشی: sample_daily
# - __init__: مقداردهی اولیهٔ نمونه و وابستگی‌های آن.
# - get_daily: دریافت همهٔ سطرهای روزانهٔ موجود در API برای نماد و تبدیل تاریخ.
# - get_hourly_day: تجمیع معاملات یک روز به OHLCV ساعتی و مقایسهٔ حجم با دادهٔ روزانه.
# - test_first_run_and_resume_no_duplicate: آزمون test_first_run_and_resume_no_duplicate
# - test_new_date_only_updates_new_partition: آزمون test_new_date_only_updates_new_partition
# - test_cleaning_returns_only_trading_days: آزمون test_cleaning_returns_only_trading_days
# - test_load_filter_and_hourly_join: آزمون test_load_filter_and_hourly_join
# - test_bad_hourly_dates_do_not_create_fake_candles: آزمون test_bad_hourly_dates_do_not_create_fake_candles
# - raise_for_status: تابع کمکی دادهٔ آزمایشی: raise_for_status
# - json: تابع کمکی دادهٔ آزمایشی: json
# - __init__: مقداردهی اولیهٔ نمونه و وابستگی‌های آن.
# - mount: تابع کمکی دادهٔ آزمایشی: mount
# - get: تابع کمکی دادهٔ آزمایشی: get
# - test_failed_latest_hourly_day_does_not_block_next_batch: آزمون test_failed_latest_hourly_day_does_not_block_next_batch
# - get_hourly_day: تجمیع معاملات یک روز به OHLCV ساعتی و مقایسهٔ حجم با دادهٔ روزانه.

from pathlib import Path
from datetime import timedelta
import pandas as pd
import pytest

from market_data import Settings, CsvMarketStore, sync_market, clean_stock_data, load_for_model
from market_data.provider import TsetmcClient

CODE = "17914401175772326"


def settings(root):
    return Settings(root=Path(root), base_url="https://cdn.tsetmc.com/api",
                    timeout=5, delay=0, retries=0, reconcile_days=14,
                    volume_tolerance=0.001, hourly_max_days=0, symbols=("اهرم",))


def sample_daily(dates=("20261010", "20261011", "20261012")):
    rows=[]
    for i, date in enumerate(dates):
        rows.append({"dEven": int(date), "priceChange": 10, "priceMin": 100+i*10,
                     "priceMax": 130+i*10, "priceYesterday": 110+i*10,
                     "priceFirst": 110+i*10, "pClosing": 120+i*10,
                     "pDrCotVal": 118+i*10, "zTotTran": 3,
                     "qTotTran5J": 60, "qTotCap": 7200,
                     "Symbol": "اهرم", "insCode": CODE, "Source": "tsetmc"})
    return pd.DataFrame(rows)


class FakeClient:
    def __init__(self, daily):
        self.daily=daily
        self.hourly_calls=[]

    def get_daily(self, symbol, ins_code):
        return self.daily.copy()

    def get_hourly_day(self, symbol, ins_code, date, daily_volume):
        self.hourly_calls.append(pd.Timestamp(date).strftime("%Y-%m-%d"))
        hourly = pd.DataFrame([{
            "insCode": ins_code, "Symbol": symbol, "Date": pd.Timestamp(date),
            "Hour": pd.Timestamp(date)+pd.Timedelta(hours=9),
            "Open": 100, "High": 120, "Low": 100, "Close": 110,
            "Volume": 60, "Trades": 2, "VWAP": 108.0,
            "DailyVolume": 60, "VolumeCoverage": 1.0,
            "DataQuality": "matched", "Source": "tsetmc",
        }])
        return hourly


def test_first_run_and_resume_no_duplicate(tmp_path):
    cfg=settings(tmp_path)
    client=FakeClient(sample_daily())
    store=CsvMarketStore(cfg.root)
    a=sync_market({"اهرم": CODE}, settings=cfg, client=client, store=store)
    assert a["اهرم"]["daily"]["new_rows"]==3
    assert a["اهرم"]["hourly_days_saved"]==3
    b=sync_market({"اهرم": CODE}, settings=cfg, client=client, store=store)
    assert b["اهرم"]["daily"]["new_rows"]==0
    assert b["اهرم"]["hourly_days_saved"]==0
    assert len(client.hourly_calls)==3
    assert len(store.load_daily(CODE))==3
    assert len(store.load_hourly(CODE))==3


def test_new_date_only_updates_new_partition(tmp_path):
    cfg=settings(tmp_path)
    client=FakeClient(sample_daily())
    store=CsvMarketStore(cfg.root)
    sync_market({"اهرم":CODE}, settings=cfg, client=client, store=store)
    client.daily=sample_daily(("20261010","20261011","20261012","20261013"))
    b=sync_market({"اهرم":CODE}, settings=cfg, client=client, store=store)
    assert b["اهرم"]["daily"]["new_rows"]==1
    assert client.hourly_calls[-1]=="2026-10-13"
    assert len(store.load_daily(CODE))==4
    assert len(store.load_hourly(CODE))==4


def test_cleaning_returns_only_trading_days():
    raw=sample_daily()
    raw.loc[1, "zTotTran"]=0
    cleaned, model=clean_stock_data(raw)
    assert len(cleaned)==3
    assert cleaned.loc[1,"HasTrade"] == False
    assert len(model)==2
    assert pd.isna(model.loc[0,"DailyReturn"])
    assert model.loc[1,"DailyReturn"] == pytest.approx(140/120-1)


def test_load_filter_and_hourly_join(tmp_path):
    cfg=settings(tmp_path)
    store=CsvMarketStore(cfg.root)
    sync_market({"اهرم":CODE}, settings=cfg, store=store, client=FakeClient(sample_daily()))
    d=store.load_daily(CODE,start="2026-10-11",end="2026-10-12")
    h=store.load_hourly(CODE,start="2026-10-11",end="2026-10-12")
    assert len(d)==2 and len(h)==2
    merged=d.merge(h,on=["insCode","Date"],validate="one_to_many")
    assert len(merged)==2
    model=load_for_model(CODE,store=store)
    assert len(model)==3
    assert model["DailyReturn"].iloc[-1]==pytest.approx(140/130-1)


def test_bad_hourly_dates_do_not_create_fake_candles(tmp_path):
    # No real quote data needed; verify complete synthetic day aggregations.
    cfg=settings(tmp_path)
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"tradeHistory": [
                {"hEven":90000,"pTran":100,"qTitTran":10,"nTran":1},
                {"hEven":93000,"pTran":120,"qTitTran":20,"nTran":2},
                {"hEven":100000,"pTran":110,"qTitTran":30,"nTran":3},
            ]}
    class Session:
        def __init__(self): self.headers={}
        def mount(self,*args): pass
        def get(self, url, timeout): return Response()
    c=TsetmcClient(cfg,session=Session())
    h=c.get_hourly_day("اهرم",CODE,pd.Timestamp("2026-10-10"),60)
    assert len(h)==2
    assert h.loc[0,"Open"]==100
    assert h.loc[0,"Close"]==120
    assert h.loc[0,"Volume"]==30
    assert h.loc[1,"Volume"]==30
    assert h.loc[0,"DataQuality"]=="matched"
    assert h.loc[0,"VWAP"]==pytest.approx((100*10+120*20)/30)


def test_failed_latest_hourly_day_does_not_block_next_batch(tmp_path):
    cfg=settings(tmp_path)
    store=CsvMarketStore(cfg.root)
    class FailingOnceClient(FakeClient):
        def get_hourly_day(self, symbol, ins_code, date, daily_volume):
            self.hourly_calls.append(pd.Timestamp(date).strftime("%Y-%m-%d"))
            if pd.Timestamp(date) == pd.Timestamp("2026-10-12"):
                return pd.DataFrame()  # Newest day is attempted first; failed days back off.
            return super().get_hourly_day(symbol, ins_code, date, daily_volume)
    client=FailingOnceClient(sample_daily())
    first=sync_market({"اهرم": CODE},settings=cfg,client=client,store=store,max_hourly_days=1)
    assert first["اهرم"]["status"]=="partial_success"
    second=sync_market({"اهرم": CODE},settings=cfg,client=client,store=store,max_hourly_days=1)
    assert second["اهرم"]["hourly_days_saved"]==1
    assert "2026-10-11" in client.hourly_calls
