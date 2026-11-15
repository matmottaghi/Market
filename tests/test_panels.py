# توضیح فایل: آزمون نگهداری انتخابی مدل، تقویم شمسی، SSE و دو تب با دادهٔ واقعی/ساختگی.
# توابع test_retention حفظ منتخب، test_jalali مرز ماه، test_stream قرارداد استریم و test_panels مسیرهای پنل را بررسی می‌کنند.
import json
import os
from datetime import date
from forecast.registry import prune, jalali_month

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

def test_refresh_forces_online(monkeypatch,tmp_path):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from forecast import views
    from django.test import RequestFactory
    monkeypatch.setattr(views,'RUNS',tmp_path)
    monkeypatch.setattr(views,'instruments',lambda:{'توان':'11111'})
    calls=[]
    monkeypatch.setattr(views.subprocess,'Popen',lambda args,**kwargs:calls.append(args))
    response=views.launch(RequestFactory().post('/run/',dict(epochs=200, symbols=['توان'],offline='on',action='refresh')))
    assert response.status_code==302
    assert '--refresh-only' in calls[0] and '--offline' not in calls[0]
