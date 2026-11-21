# توضیح فایل: پیش‌بینی تازه بدون بازآموزی؛ ترتیب نمادهای زمینه و scaler منجمد مدل حفظ می‌شود.
# predict_saved بازخوانی keras و ثبت snapshot پیش‌بینی در SQLite؛ هیچ وزن یا آمار آموزش تغییر نمی‌کند.
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from .table_io import read_table,write_table,table_path,learning_tables
from filelock import FileLock
from market_data.csv_store import _write_bytes_atomic
from .pipeline import RUNS
from .registry import settings_store,list_runs
from .hourly_data import prepare_hourly_sequences,clean_context
from .neural_data import price_distribution

def predict_saved(run_id,symbol):
    job=dict(state='running',label='پیش‌بینی با مدل ثابت',run_id=run_id,symbol=symbol,started_at=datetime.now(timezone.utc).isoformat())
    with FileLock(str(RUNS/'.pipeline.lock'),timeout=0),FileLock(str(RUNS/'.models.lock'),timeout=0):
        _write_bytes_atomic(RUNS/'inference_status.json',json.dumps(job,ensure_ascii=False))
        try:
            _,store=settings_store();known=store.list_instruments();code=known[symbol]
            if run_id not in {r['run_id'] for r in list_runs()}:raise ValueError('Unknown run')
            folder=RUNS/run_id
            config=json.loads((folder/f'{code}_preprocessing.json').read_text(encoding='utf8'))
            if config.get('target_schema')=='hourly_returns_v2':raise ValueError('This model uses an invalid historical-hour schedule; retrain with hourly_returns_v3')
            if config.get('target_schema') not in ('hourly_path_v1','hourly_returns_v3'):raise ValueError('Hourly model required')
            codes=config.get('context_codes') or [known[n] for n in config.get('context_universe',[])]
            missing=[c for c in codes if str(c) not in set(known.values())]
            if missing:raise ValueError('Required context data was removed; restore/download the missing symbol or retrain with the current selection: '+', '.join(missing))
            floor=config.get('history_start')
            contexts=[clean_context(store.load_hourly(c,start=floor),store.load_daily(c,start=floor),code=c,include_closed_hours=config.get('target_schema')=='hourly_returns_v3') for c in codes]
            prepare=prepare_hourly_sequences
            if config.get('target_schema')=='hourly_returns_v3':
                from .return_data import prepare_returns
                prepare=prepare_returns
            _,_,_,_,x,mask,latest=prepare(store.load_hourly(code,start=floor),store.load_daily(code,start=floor),contexts,allow_unlabeled=True,latest_only=True,alignment="lagged" if config.get("feature_schema")=="cross_asset_v2" else "exact",**({"trading_hours":config["latest"]["future_hours"]} if config.get("target_schema")=="hourly_returns_v3" else {}))
            scaling=config['scaling']
            x=((x-np.array(scaling['feature_mean']))/np.array(scaling['feature_scale'])).astype('float32');x[~mask]=0
            import keras
            from . import neural_network
            from .neural_training import decode
            model=keras.models.load_model(folder/f'{code}_lstm.keras',compile=False)
            raw=model([x,mask],training=False).numpy()
            if config.get('target_schema')=='hourly_returns_v3':
                if len(latest['forecast_slots'])>config.get('supported_return_steps',0):raise ValueError('Saved model does not support every hour through 21 future sessions; retraining is required')
                mu=np.cumsum(raw*np.array(scaling['target_scale']),axis=1)
                sigma=np.sqrt(np.cumsum(np.array(config['return_residual_scale'])**2))[None,:]
            else:mu,sigma=decode(raw,np.array(scaling['target_scale']),np.array(config['calibration_sigma_factor']))
            evaluated=read_table(folder/'metrics.json',dtype={'insCode':str})
            supported=range(1,len(latest['forecast_slots'])+1) if config.get('target_schema')=='hourly_returns_v3' else set(evaluated[evaluated.insCode==code].horizon)
            paths=None
            if config.get('target_schema')=='hourly_returns_v3':
                from .scenarios import scenario_paths,scenario_summary
                with np.load(folder/f'{code}_residuals.npz') as archive:
                    paths=scenario_paths(raw[0]*np.array(scaling['target_scale']),archive['residuals'],latest['current_price'])
            rows=[]
            for step in supported:
                index=int(step)-1;day,hour=index//24+1,index%24
                if config.get('target_schema')=='hourly_returns_v3':
                    if index>=len(latest['forecast_slots']):continue
                    day=latest['forecast_slots'][index]['target_day'];hour=latest['forecast_slots'][index]['target_hour']
                    if day>21:continue
                if hour not in latest['future_hours']:continue
                row=price_distribution(latest['current_price'],float(mu[0,index]),float(sigma[0,index]))
                row.update(run_id=run_id,insCode=code,symbol=symbol,model_kind='lstm',frequency='hourly',horizon=int(step),horizon_unit='hourly_slot',target_schema=config['target_schema'],target_day=day,target_hour=hour,as_of=latest['as_of'],current_price=latest['current_price'],mu_log=float(mu[0,index]),sigma_log=float(sigma[0,index]),lookback_sessions=21)
                if paths is not None:row.update(scenario_summary(paths[:,index],latest['current_price']))
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
