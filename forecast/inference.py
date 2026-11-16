# توضیح فایل: پیش‌بینی تازه بدون بازآموزی؛ ترتیب نمادهای زمینه و scaler منجمد مدل حفظ می‌شود.
# predict_saved بازخوانی keras و ثبت snapshot پیش‌بینی در SQLite؛ هیچ وزن یا آمار آموزش تغییر نمی‌کند.
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from filelock import FileLock
from market_data.csv_store import _write_bytes_atomic
from .pipeline import RUNS
from .registry import settings_store,list_runs
from .hourly_data import prepare_hourly_sequences,clean_context
from .neural_data import price_distribution

def predict_saved(run_id,symbol):
    job=dict(state='running',label='پیش‌بینی با مدل ثابت',run_id=run_id,symbol=symbol,started_at=datetime.now(timezone.utc).isoformat())
    with FileLock(str(RUNS/'.pipeline.lock'),timeout=0):
        _write_bytes_atomic(RUNS/'inference_status.json',json.dumps(job,ensure_ascii=False))
        try:
            _,store=settings_store();known=store.list_instruments();code=known[symbol]
            if run_id not in {r['run_id'] for r in list_runs()}:raise ValueError('Unknown run')
            folder=RUNS/run_id
            config=json.loads((folder/f'{code}_preprocessing.json').read_text(encoding='utf8'))
            if config.get('target_schema')!='hourly_path_v1' or config.get('feature_schema')!='cross_asset_v2':raise ValueError('Hourly model required')
            codes=config.get('context_codes') or [known[n] for n in config.get('context_universe',[])]
            contexts=[clean_context(store.load_hourly(c),store.load_daily(c),code=c) for c in codes]
            _,_,_,_,x,mask,latest=prepare_hourly_sequences(store.load_hourly(code),store.load_daily(code),contexts,allow_unlabeled=True,latest_only=True)
            scaling=config['scaling']
            x=((x-np.array(scaling['feature_mean']))/np.array(scaling['feature_scale'])).astype('float32');x[~mask]=0
            import keras
            from . import neural_network
            from .neural_training import decode
            model=keras.models.load_model(folder/f'{code}_lstm.keras',compile=False)
            mu,sigma=decode(model([x,mask],training=False).numpy(),np.array(scaling['target_scale']),np.array(config['calibration_sigma_factor']))
            evaluated=pd.read_csv(folder/'metrics.csv',dtype={'insCode':str})
            supported=set(evaluated[evaluated.insCode==code].horizon)
            rows=[]
            for step in supported:
                index=int(step)-1;day,hour=index//24+1,index%24
                if hour not in latest['future_hours']:continue
                row=price_distribution(latest['current_price'],float(mu[0,index]),float(sigma[0,index]))
                row.update(run_id=run_id,insCode=code,symbol=symbol,model_kind='lstm',frequency='hourly',horizon=int(step),horizon_unit='hourly_slot',target_schema='hourly_path_v1',target_day=day,target_hour=hour,as_of=latest['as_of'],current_price=latest['current_price'],mu_log=float(mu[0,index]),sigma_log=float(sigma[0,index]),lookback_sessions=21)
                rows.append(row)
            store.record_results(run_id,rows,[])
            with store.connect() as db:
                db.execute('CREATE TABLE IF NOT EXISTS forecast_snapshots(run TEXT,code TEXT,asof TEXT,step INTEGER,payload TEXT,PRIMARY KEY(run,code,asof,step))')
                db.executemany('INSERT OR REPLACE INTO forecast_snapshots VALUES(?,?,?,?,?)',[(run_id,code,r['as_of'],r['horizon'],json.dumps(r,ensure_ascii=False)) for r in rows])
            job.update(state='complete',as_of=latest['as_of'],points=len(rows))
        except Exception as exc:
            job.update(state='failed',error=str(exc))
        job['finished_at']=datetime.now(timezone.utc).isoformat()
        _write_bytes_atomic(RUNS/'inference_status.json',json.dumps(job,ensure_ascii=False))
        return job
