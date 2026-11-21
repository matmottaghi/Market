import gzip,json,os
import pytest
from market_data.sqlite_store import SqliteMarketStore
from market_data.cleaning import clean_stock_data
from test_market_csv import sample_daily,CODE


def test_delete_archives_raw_data_and_keeps_other_symbols(tmp_path):
    store=SqliteMarketStore(tmp_path)
    daily,_=clean_stock_data(sample_daily());store.upsert_daily(daily);store.update_state(CODE,symbol='حذف')
    store.update_state('22222',symbol='باقی')
    store.record_hourly_failure(CODE,'2026-10-10','empty')
    archive=store.delete_instrument(CODE)
    assert store.list_instruments()=={'باقی':'22222'}
    assert store.load_daily(CODE).empty and not store.state_path(CODE).exists()
    with gzip.open(archive,'rt',encoding='utf8') as f:backup=json.load(f)
    assert len(backup['tables']['daily_bars'])==3
    assert len(backup['tables']['ingestion_issues'])==1
    assert backup['tables']['instruments'][0][0]==CODE


def test_delete_endpoint_checks_identity_and_requires_post(tmp_path,monkeypatch):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from forecast import data_views
    from django.test import Client
    store=SqliteMarketStore(tmp_path/'archive');store.update_state(CODE,symbol='حذف')
    runs=tmp_path/'runs';runs.mkdir()
    monkeypatch.setattr(data_views,'RUNS',runs)
    monkeypatch.setattr(data_views,'settings_store',lambda:(None,store))
    c=Client()
    assert c.get('/data/delete/').status_code==405
    assert c.post('/data/delete/',dict(code=CODE,symbol='اشتباه')).status_code==400
    assert 'حذف' in store.list_instruments()
    assert Client(enforce_csrf_checks=True).post('/data/delete/',dict(code=CODE,symbol='حذف')).status_code==403
    assert c.post('/data/delete/',dict(code=CODE,symbol='حذف')).status_code==302
    assert not store.list_instruments()


@pytest.mark.parametrize('mode,names,expected', [('selected',['اهرم'],['اهرم']),('none',[],[])])
def test_launch_freezes_explicit_context_selection(tmp_path,monkeypatch,mode,names,expected):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from forecast import views
    from django.test import RequestFactory
    monkeypatch.setattr(views,'RUNS',tmp_path)
    monkeypatch.setattr(views,'instruments',lambda:{'موج':'11111','اهرم':'22222'})
    calls=[];monkeypatch.setattr(views.subprocess,'Popen',lambda args,**kw:calls.append(args))
    response=views.launch(RequestFactory().post('/run/',dict(action='train',target_symbol='موج',context_mode=mode,context_symbols=names)))
    assert response.status_code==302
    args=calls[0]
    assert args[args.index('--context-symbols')+1:]==expected
    assert '--offline' in args


def test_training_rejects_unknown_context(tmp_path,monkeypatch):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from forecast import views
    from django.test import RequestFactory
    monkeypatch.setattr(views,'instruments',lambda:{'موج':'11111'})
    response=views.launch(RequestFactory().post('/run/',dict(action='train',target_symbol='موج',context_mode='selected',context_symbols=['ناشناس'])))
    assert response.status_code==400
