# توضیح فایل: آزمون مهاجرت بدون حذف CSV، upsert یکتا و حفظ روز معتبر در برابر پاسخ نامعتبر.
import json
import pandas as pd
import pytest
from market_data.csv_store import CsvMarketStore
from market_data.sqlite_store import migrate,make_store,SqliteMarketStore
from test_market_csv import sample_daily,FakeClient,CODE

def test_migrate_and_repeat(tmp_path):
    original=CsvMarketStore(tmp_path)
    daily=sample_daily();daily['Date']=pd.to_datetime(daily.dEven.astype(str))
    original.upsert_daily(daily);original.update_state(CODE,symbol='اهرم')
    hour=FakeClient(daily).get_hourly_day('اهرم',CODE,pd.Timestamp('2026-10-10'),60)
    original.save_hourly_day(hour)
    report=migrate(tmp_path);assert all(r['verified'] for r in report)
    target=make_store(tmp_path);assert isinstance(target,SqliteMarketStore)
    assert len(target.load_daily(CODE))==3
    target.upsert_daily(daily);assert len(target.load_daily(CODE))==3
    assert original.daily_path(CODE,'2026-10').exists()
    assert migrate(tmp_path)==report
    broken=hour.copy();broken['DataQuality']='volume_mismatch'
    with pytest.raises(ValueError):target.save_hourly_day(broken)
    assert target.day_quality(CODE,'2026-10-10')=='matched'

def test_sqlite_quality_retry(tmp_path):
    target=SqliteMarketStore(tmp_path)
    target.record_hourly_failure(CODE,'2026-10-10','timeout')
    assert not target.should_fetch_hourly(CODE,'2026-10-10')
    assert target.should_fetch_hourly(CODE,'2026-10-10',retry_incomplete=True)
