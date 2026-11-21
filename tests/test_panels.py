# توضیح فایل: آزمون نگهداری انتخابی مدل، تقویم شمسی، SSE و دو تب با دادهٔ واقعی/ساختگی.
# توابع test_retention حفظ منتخب، test_jalali مرز ماه، test_stream قرارداد استریم و test_panels مسیرهای پنل را بررسی می‌کنند.
# آزمون‌های training انتخاب دقیقاً یک هدف ثبت‌شده و خواندن صرفاً از دیتابیس را کنترل می‌کنند؛ درخواست refresh از آموزش رد می‌شود.
import json
import os
import pandas as pd
import pytest
from datetime import date
from forecast.registry import prune, jalali_month
from forecast.table_io import write_table

def test_retention(tmp_path):
    for name, saved in [('old',False),('selected',True),('latest',False)]:
        p=tmp_path/name;p.mkdir()
        (p/'status.json').write_text(json.dumps(dict(run_id=name,state='complete')))
        if saved:(p/'saved.json').write_text('{}')
    prune('latest',tmp_path)
    assert not (tmp_path/'old').exists()
    assert (tmp_path/'selected').exists() and (tmp_path/'latest').exists()

def test_jalali():
    assert jalali_month(date(2024,3,20))==1
    assert jalali_month(date(2025,3,20))==12
    assert jalali_month(date(2025,9,23))==7

def test_only_selected_symbol_survives(tmp_path):
    old=tmp_path/'old';old.mkdir()
    (old/'status.json').write_text(json.dumps(dict(run_id='old',state='complete')))
    (old/'saved.json').write_text(json.dumps(dict(codes=['11111'])))
    (old/'11111_lstm.keras').write_text('selected')
    (old/'22222_lstm.keras').write_text('discarded')
    prune('new',tmp_path)
    assert (old/'11111_lstm.keras').exists()
    assert not (old/'22222_lstm.keras').exists()

def test_save_selected_symbol(monkeypatch,tmp_path):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from forecast import views
    from django.test import RequestFactory
    p=tmp_path/'run';p.mkdir()
    (p/'status.json').write_text(json.dumps(dict(run_id='run',state='complete',model_kind='lstm')))
    (p/'11111_lstm.keras').write_text('fixture')
    (p/'11111_preprocessing.json').write_text(json.dumps({'calibration_sigma_factor':[1]}))
    write_table(pd.DataFrame({'insCode':['11111']}),p/'predictions.json')
    monkeypatch.setattr(views,'RUNS',tmp_path)
    monkeypatch.setattr(views,'instruments',lambda:{'توان':'11111'})
    response=views.retain(RequestFactory().post('/retain/',dict(run='run',scope='symbol',symbol='توان',action='save',name='منتخب')))
    assert response.status_code==302
    assert json.loads((p/'saved.json').read_text(encoding='utf8'))['codes']==['11111']

def test_stream(monkeypatch,tmp_path):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from forecast import views
    from django.test import RequestFactory
    monkeypatch.setattr(views,'RUNS',tmp_path)
    response=views.stream(RequestFactory().get('/stream/'))
    assert response['Content-Type']=='text/event-stream'
    content=next(iter(response.streaming_content)).decode()
    assert content.startswith('data: ') and content.endswith('\n\n')
    assert json.loads(content[6:])['status']['state']=='idle'
    response.close()

def test_panels():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from django.test import Client
    for path in ['/training/','/market/']:
        result=Client().get(path)
        assert result.status_code==200
        assert 'آموزش و ارزیابی مدل'.encode() in result.content

def test_training_page_cannot_refresh_data(monkeypatch,tmp_path):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from forecast import views
    from django.test import RequestFactory
    monkeypatch.setattr(views,'RUNS',tmp_path)
    monkeypatch.setattr(views,'instruments',lambda:{'توان':'11111'})
    calls=[]
    monkeypatch.setattr(views.subprocess,'Popen',lambda args,**kwargs:calls.append(args))
    response=views.launch(RequestFactory().post('/run/',dict(epochs=200, symbols=['توان'],offline='on',action='refresh')))
    assert response.status_code==400 and calls==[]


@pytest.mark.parametrize('fields',[
    {}, {'target_symbol':['توان','اهرم']}, {'symbols':['توان','اهرم']},
    {'target_symbol':'ناشناس'}, {'target_symbol':'توان','new_symbols':'موج'},
])
def test_training_rejects_missing_multiple_or_unregistered_targets(monkeypatch,tmp_path,fields):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from forecast import views
    from django.test import RequestFactory
    monkeypatch.setattr(views,'RUNS',tmp_path)
    monkeypatch.setattr(views,'instruments',lambda:{'توان':'11111','اهرم':'22222'})
    calls=[]
    monkeypatch.setattr(views.subprocess,'Popen',lambda args,**kwargs:calls.append(args))
    response=views.launch(RequestFactory().post('/run/',dict(action='train',**fields)))
    assert response.status_code==400 and calls==[]


@pytest.mark.parametrize('field',['target_symbol','symbols'])
def test_single_target_training_always_uses_database(monkeypatch,tmp_path,field):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from forecast import views
    from django.test import RequestFactory
    monkeypatch.setattr(views,'RUNS',tmp_path)
    monkeypatch.setattr(views,'instruments',lambda:{'توان':'11111','اهرم':'22222'})
    calls=[]
    monkeypatch.setattr(views.subprocess,'Popen',lambda args,**kwargs:calls.append(args))
    response=views.launch(RequestFactory().post('/run/',dict(action='train',**{field:'اهرم'})))
    assert response.status_code==302 and response.url.startswith('/training/?symbol=')
    args=calls[0]
    assert args[args.index('--symbols')+1:]==['اهرم','--offline']
    assert '--refresh-only' not in args


def test_data_refresh_can_explicitly_backfill_full_history(monkeypatch,tmp_path):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from forecast import views
    from django.test import RequestFactory
    monkeypatch.setattr(views,'RUNS',tmp_path)
    monkeypatch.setattr(views,'instruments',lambda:{'اهرم':'11111'})
    calls=[]
    monkeypatch.setattr(views.subprocess,'Popen',lambda args,**kwargs:calls.append(args))
    response=views.launch(RequestFactory().post('/run/',dict(next='/data/',action='refresh',symbols=['اهرم'],hourly_days=0,full_history='1')))
    assert response.status_code==302
    assert '--full-history' in calls[0] and '--refresh-only' in calls[0]
    assert calls[0][calls[0].index('--hourly-days')+1]=='0'
    assert '--offline' not in calls[0]
