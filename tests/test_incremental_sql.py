# توضیح فایل: آزمون دریافت افزایشی SQLite، مرز نماد مرجع، عدم ایجاد CSV و جلوگیری از بازنویسی سطر یکسان؛ بدون شبکه.
import pandas as pd
from market_data.sqlite_store import SqliteMarketStore,make_store
from market_data.pipeline import sync_market
from test_market_csv import CODE,FakeClient,sample_daily,settings
from forecast.table_io import write_table,read_table,table_path
from market_data.cleaning import clean_stock_data

def test_resume_is_bounded_and_does_not_write_csv(tmp_path,monkeypatch):
    monkeypatch.delenv('MARKET_HISTORY_START',raising=False)
    cfg=settings(tmp_path);store=SqliteMarketStore(tmp_path)
    raw,_=clean_stock_data(sample_daily());store.upsert_daily(raw);store.update_state(CODE,symbol='اهرم',last_daily_date='2026-10-12')
    class Client(FakeClient):
        def get_daily_since(self,symbol,code,start):
            self.since=start
            return self.daily[self.daily.Date>=pd.Timestamp(start)].copy()
    client=Client(raw)
    first=sync_market({'اهرم':CODE},settings=cfg,store=store,client=client)
    before=store.latest_rows(CODE,'daily',1)[0]['_ingested_at']
    sync_market({'اهرم':CODE},settings=cfg,store=store,client=client)
    assert len(client.hourly_calls)==3
    assert str(pd.Timestamp(client.since).date())=='2026-09-28'
    assert first['اهرم']['history_start'] is None
    assert first['اهرم']['history_policy']=='per_instrument_full_archive'
    assert store.latest_rows(CODE,'daily',1)[0]['_ingested_at']==before
    assert not list(tmp_path.rglob('*.csv'))
    assert isinstance(make_store(tmp_path),SqliteMarketStore)

def test_json_model_artifacts_roundtrip_and_legacy_read(tmp_path):
    data=pd.DataFrame(dict(insCode=['12345'],value=[.123456789],timestamp=['2026-10-12']))
    path=tmp_path/'metrics.csv';write_table(data,path)
    assert not path.exists() and table_path(path).suffix=='.json'
    pd.testing.assert_frame_equal(read_table(path),data)
    legacy=tmp_path/'old.csv';data.to_csv(legacy,index=False)
    assert read_table(legacy,dtype={'insCode':str}).insCode.iloc[0]=='12345'


def test_full_history_backfills_older_target_despite_young_context(tmp_path,monkeypatch):
    monkeypatch.delenv('MARKET_HISTORY_START',raising=False)
    cfg=settings(tmp_path);store=SqliteMarketStore(tmp_path)
    raw,_=clean_stock_data(sample_daily())
    store.upsert_daily(raw.iloc[-1:])
    young=raw.iloc[-1:].copy();young['insCode']='22222';young['Symbol']='جوان'
    store.upsert_daily(young)
    store.update_state(CODE,symbol='اهرم',last_daily_date='2026-10-12')
    store.update_state('22222',symbol='جوان',last_daily_date='2026-10-12')
    client=FakeClient(raw)
    report=sync_market({'اهرم':CODE},settings=cfg,store=store,client=client,full_reconcile=True)
    assert len(store.load_daily(CODE))==3
    assert len(client.hourly_calls)==3
    assert report['اهرم']['history_start'] is None


def test_history_limit_only_applies_when_explicit(tmp_path,monkeypatch):
    from market_data.history_scope import history_start
    store=SqliteMarketStore(tmp_path);cfg=settings(tmp_path)
    monkeypatch.setenv('MARKET_HISTORY_START','2020-01-01')
    assert history_start(store,cfg)=='2020-01-01'
    monkeypatch.delenv('MARKET_HISTORY_START')
    assert history_start(store,cfg) is None
