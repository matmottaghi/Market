# توضیح فایل: آزمون انتخاب پیش‌بینی بر اساس نماد، اجرای درحال‌آموزش و خروجی تاریخی SQLite با وزن حذف‌شده؛ بدون شبکه و CSV.
# workspace دو اجرای مستقل می‌سازد؛ آزمون‌ها از نمایش صفحهٔ خالی یا اجرای مدل سهم دیگر جلوگیری می‌کنند.
import json,os
from dataclasses import replace
import pandas as pd
import pytest
from market_data.sqlite_store import SqliteMarketStore
from market_data.config import Settings
from forecast.table_io import write_table
from forecast.availability import catalog,choose

@pytest.fixture
def availability_workspace(tmp_path,monkeypatch):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from forecast import views,options_views
    root=tmp_path/'runs';root.mkdir();store=SqliteMarketStore(tmp_path/'db')
    known={'اهرم':'11111','موج':'22222','نماد تازه':'33333'}
    for symbol,code in known.items():store.update_state(code,symbol=symbol)
    predictions=[]
    for rid,symbol,code in [('20260101-old','اهرم','11111'),('20260102-new','موج','22222')]:
        p=dict(run_id=rid,symbol=symbol,insCode=code,horizon=13,target_schema='hourly_path_v1',target_day=1,target_hour=12,current_price=100,mu_log=.01,sigma_log=.1,as_of='2026-01-01 13:00:00')
        store.record_run(dict(run_id=rid,state='complete',label=rid,model_kind='lstm'))
        store.record_results(rid,[p],[]);predictions.append(p)
    # Only the newest model's local artifacts remain, just as after pruning.
    folder=root/'20260102-new';folder.mkdir()
    (folder/'status.json').write_text(json.dumps(dict(run_id=folder.name,state='complete',label='wave',model_kind='lstm')))
    write_table(pd.DataFrame([predictions[1]]),folder/'predictions.json')
    for module in [views,options_views]:
        monkeypatch.setattr(module,'RUNS',root)
        monkeypatch.setattr(module,'instruments',lambda:known)
        monkeypatch.setattr(module,'settings_store',lambda:(replace(Settings.from_env(),root=store.root),store))
    monkeypatch.setattr(options_views,'list_runs',lambda:views.list_runs(root))
    from django.test import Client
    return Client(),store,root

def test_market_switches_to_correct_symbol_and_loads_archived_sql(availability_workspace):
    client,_,_=availability_workspace
    response=client.get('/market/',dict(symbol='اهرم',run='20260102-new'))
    text=response.content.decode()
    assert response.status_code==200
    assert 'قیمت مورد انتظار' in text and 'خروجی تاریخی ذخیره‌شده' in text
    assert 'اجرای انتخابی برای این نماد خروجی نداشت' in text
    assert 'value="20260101-old" selected' in text
    assert 'disabled title="وزن مدل' in text

def test_new_symbol_has_clear_missing_forecast_message(availability_workspace):
    client,_,_=availability_workspace
    text=client.get('/market/',dict(symbol='نماد تازه')).content.decode()
    assert 'داشتن داده یا حضور در زمینهٔ بازار' in text
    assert 'نماد تازه — بدون پیش‌بینی' in text

def test_archived_output_cannot_launch_inference(availability_workspace):
    client,_,_=availability_workspace
    response=client.post('/forecast/refresh/',dict(symbol='اهرم',run='20260101-old'))
    assert response.status_code==400 and 'وزن مدل این نماد موجود نیست' in response.content.decode()

def test_options_uses_the_same_symbol_specific_run(availability_workspace):
    client,_,_=availability_workspace
    text=client.get('/options/',dict(symbol='اهرم',run='20260102-new')).content.decode()
    assert 'value="20260101-old" selected' in text and 'جلسه 1' in text

def test_running_new_run_does_not_hide_finished_forecasts(availability_workspace):
    _,store,root=availability_workspace
    rows=catalog([dict(run_id='20260103-running',state='running')],root,store)
    assert choose(rows,'11111')['run_id']=='20260101-old'
    assert choose(rows,'22222')['run_id']=='20260102-new'
    assert choose(rows,'33333') is None


def test_training_form_has_one_target_dropdown_and_no_refresh_controls(availability_workspace):
    client,_,_=availability_workspace
    from html.parser import HTMLParser
    class Controls(HTMLParser):
        def __init__(self):
            super().__init__();self.inside=False;self.tags=[]
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if tag=='form' and attrs.get('id')=='target-training-form':self.inside=True
            if self.inside:self.tags.append((tag,attrs))
        def handle_endtag(self,tag):
            if tag=='form':self.inside=False
    response=client.get('/training/',dict(symbol='اهرم'))
    assert response.status_code==200
    controls=Controls();controls.feed(response.content.decode())
    selects=[attrs for tag,attrs in controls.tags if tag=='select']
    assert len(selects)==1 and selects[0]['name']=='target_symbol' and 'multiple' not in selects[0]
    assert any(tag=='option' and attrs.get('value')=='اهرم' and 'selected' in attrs for tag,attrs in controls.tags)
    assert not any((attrs.get('type')=='checkbox' and attrs.get('name')!='context_symbols') or attrs.get('value')=='refresh' or attrs.get('name') in ['symbols','offline','new_symbols'] for tag,attrs in controls.tags)
    assert any(attrs.get('name')=='epochs' and attrs.get('value')=='200' for tag,attrs in controls.tags)


def test_market_slicer_receives_full_horizon_even_with_old_days_filter(availability_workspace):
    client,store,_=availability_workspace
    from html.parser import HTMLParser
    class Payload(HTMLParser):
        def __init__(self):
            super().__init__(); self.active=False; self.data=[]; self.ids=set()
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs); self.ids.add(attrs.get("id"))
            if tag=="script" and attrs.get("id")=="market-forecast-data":self.active=True
        def handle_endtag(self,tag):
            if tag=="script":self.active=False
        def handle_data(self,value):
            if self.active:self.data.append(value)
    rows=[]
    for step,day,hour in [(1,1,9),(2,1,10),(80,21,12)]:
        rows.append(dict(run_id='20260101-old',symbol='اهرم',insCode='11111',horizon=step,target_schema='hourly_returns_v3',target_day=day,target_hour=hour,current_price=100,mu_log=.01,sigma_log=.1,as_of='2026-01-01 13:00:00',scenario_prices=[90,100,110,120]))
    store.record_results('20260101-old',rows,[])
    response=client.get('/market/',dict(symbol='اهرم',days=1))
    soup=Payload();soup.feed(response.content.decode())
    payload=json.loads(''.join(soup.data))
    assert max(r['target_day'] for r in payload['rows'])==21
    assert {'forecast-start','forecast-end'}<=soup.ids
    assert {'forecast-point','forecast-reset'}<=soup.ids
    assert 'days-slider' not in soup.ids
    assert 'scenario_prices' in payload['rows'][0]


def test_market_slicer_not_in_training(availability_workspace):
    client,_,_=availability_workspace
    text=client.get('/training/',dict(symbol='موج')).content.decode()
    assert 'id="forecast-start"' not in text and 'id="days-slider"' in text


def test_training_report_uses_selected_model_universe(availability_workspace):
    client,_,root=availability_workspace
    from test_training_report import metadata
    folder=root/'20260102-new'
    (folder/'22222_preprocessing.json').write_text(json.dumps(metadata()),encoding='utf8')
    text=client.get('/training/',dict(symbol='موج',run='20260102-new')).content.decode()
    assert 'id="training-report"' in text
    assert '2025-01-01' in text and '2025-07-01 12:00:00' in text
    assert '240 × 84 × 16' in text and 'زمینه قدیمی · زمینه' in text
    assert 'موج · سهم هدف' in text
    assert 'تعداد نمونه' in text and 'کالیبراسیون' in text
    market=client.get('/market/',dict(symbol='موج')).content.decode()
    assert 'id="training-report"' not in market


def test_invalid_v2_forecast_is_not_relabelled_as_full_horizon(availability_workspace):
    client,store,_=availability_workspace
    row=dict(run_id='20260101-old',symbol='اهرم',insCode='11111',horizon=1,target_schema='hourly_returns_v2',target_day=1,target_hour=13,current_price=100,mu_log=.01,sigma_log=.1,as_of='2026-01-01 13:00:00')
    store.record_results('20260101-old',[row],[])
    text=client.get('/market/',dict(symbol='اهرم')).content.decode()
    assert 'برنامهٔ ساعت‌های تاریخی نادرست' in text
    assert 'آموزش دهید' in text
