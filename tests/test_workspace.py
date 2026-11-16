# توضیح فایل: آزمون مسیر ساعتی پویا، بستهٔ مدل و سناریوی آپشن بدون درخواست شبکه یا تغییر دادهٔ واقعی.
# fixture workspace مخزن و مدل مستقل می‌سازد؛ سه آزمون زیر قرارداد صفحه و محاسبه را بررسی می‌کنند.
import os,json,io,zipfile
from dataclasses import replace
import pandas as pd
import pytest

@pytest.fixture
def workspace(tmp_path,monkeypatch):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from forecast import views,options_views
    from market_data.config import Settings
    from market_data.sqlite_store import SqliteMarketStore
    store=SqliteMarketStore(tmp_path/'db');store.update_state('12345',symbol='اهرم')
    folder=tmp_path/'runs'/'demo';folder.mkdir(parents=True)
    (folder/'status.json').write_text(json.dumps(dict(run_id='demo',state='complete',model_kind='lstm',label='test')))
    rows=[]
    for day in range(1,22):
        for hour in [9,10,11,12]:
            rows.append(dict(symbol='اهرم',insCode='12345',horizon=(day-1)*24+hour+1,target_schema='hourly_path_v1',target_day=day,target_hour=hour,current_price=100,mu_log=.01,sigma_log=.1,as_of='2026-11-15 13:00:00'))
    pd.DataFrame(rows).to_csv(folder/'predictions.csv',index=False)
    for module in [views,options_views]:
        monkeypatch.setattr(module,'RUNS',folder.parent)
        monkeypatch.setattr(module,'instruments',lambda:{'اهرم':'12345'})
        monkeypatch.setattr(module,'settings_store',lambda:(replace(Settings.from_env(),root=store.root),store))
    monkeypatch.setattr(options_views,'list_runs',lambda:[dict(run_id='demo',state='complete',label='test')])
    from django.test import Client
    return Client(),store,folder

def test_dynamic_report_days(workspace):
    client,store,folder=workspace
    response=client.get('/market/',dict(run='demo',symbol='اهرم',days=3))
    assert response.status_code==200
    text=response.content.decode()
    assert 'بازده پایان جلسه 3' in text and 'بازده پایان جلسه 4' not in text
    assert 'جلسه 3 — ساعت 12' in text and 'جلسه 4 — ساعت' not in text

def test_download_contains_scaler_and_hashes(workspace):
    client,store,folder=workspace
    (folder/'12345_lstm.keras').write_bytes(b'keras-fixture')
    (folder/'12345_preprocessing.json').write_text('{}')
    response=client.get('/models/download/',dict(run='demo',symbol='اهرم'))
    assert response.status_code==200
    with zipfile.ZipFile(io.BytesIO(b''.join(response.streaming_content))) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        assert len(manifest['sha256']['12345_lstm.keras'])==64
        assert archive.read('12345_preprocessing.json')==b'{}'
    response.close()

def test_option_template_real_quote_and_expected_profit(workspace):
    client,store,folder=workspace
    snapshot=dict(spot=100,contracts=[dict(code='call',symbol='test-call',kind='call',strike=100,expiry='2026-11-18',multiplier=1000,bid=9,ask=10,bid_volume=10,ask_volume=10)])
    with store.connect() as db:
        db.execute('INSERT INTO snapshots(kind,underlying,retrieved_at,payload) VALUES(?,?,?,?)',('options','12345','2026-11-15',json.dumps(snapshot)))
    response=client.post('/options/',dict(symbol='اهرم',run='demo',strategy='long_call',build='template',expiry='2026-11-18',step=61,use_forecast='on',confirm_alignment='on',fees=100,capital=10100,rate=.3,dividend=.01))
    assert response.status_code==200
    content=response.content.decode()
    assert '10000' in content and 'امید ریاضی سود' in content
    assert 'name="rate" type="number" step=".01" value="0.3"' in content
    assert 'ابتدا snapshot' not in content and 'منطبق نیست' not in content
