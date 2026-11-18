# توضیح فایل: آزمون آرشیو مستقل، ترتیب/محدودیت پیش‌نمایش، کنترل مجدد حجم و دریافت بدون آموزش؛ هیچ درخواست شبکه‌ای انجام نمی‌دهد.
# archive مخزن سه روزه می‌سازد؛ آزمون‌های زیر از نمایش نمادهای بدون داده و عدم دریافت تصادفی همهٔ نمادها محافظت می‌کنند.
import os
from dataclasses import replace
import pandas as pd
import pytest

@pytest.fixture
def archive(tmp_path,monkeypatch):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from market_data.sqlite_store import SqliteMarketStore
    from market_data.config import Settings
    from forecast import data_views
    store=SqliteMarketStore(tmp_path/'archive')
    store.update_state('12345',symbol='اهرم');store.update_state('99999',symbol='نماد تازه')
    daily=pd.DataFrame([dict(insCode='12345',Symbol='اهرم',Date=pd.Timestamp(f'2025-01-0{d}'),pClosing=10+d,pDrCotVal=11+d,priceFirst=10,priceMin=9,priceMax=15,qTotTran5J=100,zTotTran=2) for d in [1,2,3]])
    store.upsert_daily(daily)
    for day,volume in [(1,50),(2,20)]:
        store.save_hourly_day(pd.DataFrame([dict(insCode='12345',Symbol='اهرم',Date=pd.Timestamp(f'2025-01-0{day}'),Hour=pd.Timestamp(f'2025-01-0{day} {h}:00'),Open=10,High=15,Low=9,Close=11,Volume=volume,Trades=1,DataQuality='matched') for h in [9,10]]))
    monkeypatch.setattr(data_views,'settings_store',lambda:(replace(Settings.from_env(),root=store.root),store))
    monkeypatch.setattr(data_views,'instruments',store.list_instruments)
    monkeypatch.setattr(data_views,'list_runs',lambda:[])
    from django.test import Client
    return store,Client()

def test_bounded_preview_and_current_quality(archive):
    store,_=archive
    rows=store.latest_rows('12345','daily',2)
    assert [r['Date'] for r in rows]==['2025-01-03','2025-01-02']
    assert store.latest_rows('12345','hourly',1)[0]['Hour']=='2025-01-02T10:00:00'
    with pytest.raises(ValueError):store.latest_rows('12345','bad',10)
    with pytest.raises(ValueError):store.latest_rows('12345','daily',999)
    coverage=store.coverage_summary('12345')
    assert (coverage['active_days'],coverage['matched_days'],coverage['missing_days'],coverage['invalid_days'])==(3,1,1,1)
    assert coverage['last_matched_hour']=='2025-01-01T10:00:00'

def test_data_page_shows_inventory_latest_rows_and_missing(archive):
    _,client=archive
    response=client.get('/data/',dict(symbol='اهرم',limit='10000'))
    assert response.status_code==200
    text=response.content.decode()
    assert 'نماد تازه' in text and '2025-01-03' in text and '2025-01-02 10:00' in text
    assert 'حجم نامنطبق' in text and 'روز فعال بدون کندل ساعتی' in text
    assert '۱۰ ردیف آخر' in text and '<script src="/static/forecast/data_panel.js?v=' in text
    empty=client.get('/data/',dict(symbol='نماد تازه')).content.decode()
    assert 'هنوز دادهٔ روزانه‌ای' in empty and 'هنوز کندل ساعتی نداریم' in empty
    assert client.get('/data/',dict(symbol='../unknown')).status_code==400

def test_data_refresh_preserves_selection_and_never_trains(archive,monkeypatch,tmp_path):
    store,client=archive
    from forecast import views
    monkeypatch.setattr(views,'RUNS',tmp_path/'runs')
    monkeypatch.setattr(views,'instruments',store.list_instruments)
    calls=[];monkeypatch.setattr(views.subprocess,'Popen',lambda args,**kwargs:calls.append(args))
    data=dict(next='/data/',view_symbol='اهرم',action='refresh',symbols=['اهرم'],hourly_days='21')
    response=client.post('/run/',data)
    assert response.status_code==302 and response.url.startswith('/data/?symbol=')
    assert '--refresh-only' in calls[0] and '--offline' not in calls[0]
    assert calls[0][calls[0].index('--hourly-days')+1]=='21'
    data.pop('symbols');assert client.post('/run/',data).status_code==400
    assert len(calls)==1
