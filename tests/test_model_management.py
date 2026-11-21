import gzip,json,os
import pytest
from forecast.model_management import model_identity,new_model_identity,model_records,delete_model_run
from market_data.sqlite_store import SqliteMarketStore
from market_data.cleaning import clean_stock_data
from test_market_csv import sample_daily,CODE


def test_identity_uses_tehran_target_and_excludes_target_from_context():
    assert model_identity('2026-11-20T16:00:00+00:00',['اهرم'],{'اهرم':2})=='2026-11-20 | 19:30:00 تهران | اهرم | 2 سهم زمینه'
    identity=new_model_identity(['اهرم'],['اهرم','موج','موج','فملی'])
    assert '-اهرم-C2-' in identity['run_id']
    assert identity['context_counts']=={'اهرم':2}


def test_delete_removes_all_model_references_without_touching_market_data(tmp_path):
    root=tmp_path/'runs';root.mkdir();folder=root/'model';folder.mkdir()
    (folder/'network.keras').write_bytes(b'weights')
    (folder/'status.json').write_text(json.dumps(dict(run_id='model',state='complete',job_kind='training')))
    store=SqliteMarketStore(tmp_path/'db')
    raw,_=clean_stock_data(sample_daily());store.upsert_daily(raw);store.update_state(CODE,symbol='اهرم')
    store.record_run(dict(run_id='model',state='complete',job_kind='training'))
    store.record_run(dict(run_id='data',state='complete',job_kind='data'))
    store.record_results('model',[dict(insCode=CODE,horizon=1)],[])
    store.record_epoch('model',CODE,dict(epoch=1,loss=1))
    with store.connect() as db:db.execute('INSERT INTO model_registry VALUES(?,?,?)',('active:'+CODE,'active',json.dumps(dict(run_id='model'))))
    archive=delete_model_run(root,store,'model')
    assert not folder.exists() and (archive/'artifacts/network.keras').read_bytes()==b'weights'
    with gzip.open(archive/'database.json.gz','rt',encoding='utf8') as f:assert json.load(f)['tables']['forecasts']
    with store.connect() as db:
        for table in ['forecasts','epoch_metrics','model_registry']:assert db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]==0
        assert db.execute('SELECT id FROM training_runs').fetchall()==[('data',)]
    assert len(store.load_daily(CODE))==3
    assert model_records(root,store,[],store.list_instruments())==[]
    with pytest.raises(ValueError):delete_model_run(root,store,'../outside')
    with pytest.raises(ValueError):delete_model_run(root,store,'data')


def test_management_lists_sql_only_models_and_ignores_data_jobs(tmp_path):
    store=SqliteMarketStore(tmp_path/'db')
    store.record_run(dict(run_id='old',state='complete',job_kind='training',started_at='2026-11-20T16:00:00+00:00',target_symbols=['اهرم'],context_counts={'اهرم':2}))
    store.record_run(dict(run_id='download',state='complete',job_kind='data'))
    rows=model_records(tmp_path/'runs',store,[],{})
    assert len(rows)==1 and rows[0]['context_description']=='2'
    assert '19:30:00' in rows[0]['display_label'] and not rows[0]['has_weights']


def test_delete_view_and_training_only_management(tmp_path,monkeypatch):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
    import django;django.setup()
    from forecast import views
    from django.test import Client
    root=tmp_path/'runs';root.mkdir();folder=root/'run';folder.mkdir()
    status=dict(run_id='run',state='complete',job_kind='training',model_kind='lstm')
    (folder/'status.json').write_text(json.dumps(status))
    store=SqliteMarketStore(tmp_path/'db');store.record_run(status)
    monkeypatch.setattr(views,'RUNS',root)
    monkeypatch.setattr(views,'settings_store',lambda:(None,store))
    monkeypatch.setattr(views,'instruments',lambda:{})
    c=Client()
    assert 'id="model-management"' in c.get('/training/').content.decode()
    assert 'id="model-management"' not in c.get('/market/').content.decode()
    assert c.get('/models/delete/').status_code==405
    assert Client(enforce_csrf_checks=True).post('/models/delete/',dict(run='run')).status_code==403
    assert c.post('/models/delete/',dict(run='unknown')).status_code==400
    assert c.post('/models/delete/',dict(run='run')).status_code==302
    assert not folder.exists()


def test_model_delete_lock_allows_data_jobs_but_blocks_model_jobs(tmp_path):
    from filelock import FileLock,Timeout
    from forecast.model_management import model_mutation_lock
    root=tmp_path/'runs';root.mkdir();job=root/'job';job.mkdir()
    status=job/'status.json'
    status.write_text(json.dumps(dict(run_id='job',state='running',job_kind='data',started_at='2026-11-20T16:00:00+00:00')))
    with FileLock(str(root/'.pipeline.lock'),timeout=0):
        with model_mutation_lock(root):pass
        status.write_text(json.dumps(dict(run_id='job',state='running',job_kind='training',started_at='2026-11-20T16:00:00+00:00')))
        with pytest.raises(Timeout):
            with model_mutation_lock(root):pass
