"""Display identity and recoverable deletion of model runs, including SQL-only history."""
import gzip
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path


def model_identity(started, targets, counts):
    at=datetime.fromisoformat(str(started).replace('Z','+00:00'))
    if at.tzinfo is None:at=at.replace(tzinfo=timezone.utc)
    at=at.astimezone(ZoneInfo('Asia/Tehran'))
    target='، '.join(targets) if targets else 'هدف ثبت نشده'
    context=' / '.join(str(counts[t]) if counts.get(t) is not None else 'نامشخص' for t in targets) if targets else 'نامشخص'
    return f'{at:%Y-%m-%d} | {at:%H:%M:%S} تهران | {target} | {context} سهم زمینه'


def new_model_identity(targets, contexts):
    now=datetime.now(timezone.utc)
    contexts=list(dict.fromkeys(contexts))
    counts={t:len([n for n in contexts if n!=t]) for t in targets}
    local=now.astimezone(ZoneInfo('Asia/Tehran'))
    target=re.sub(r'[^\w-]+','-',targets[0] if len(targets)==1 else 'multi')[:40]
    count=str(counts[targets[0]]) if len(targets)==1 else 'multi'
    run_id=f'{local:%Y%m%d-%H%M%S}-{target}-C{count}-{uuid.uuid4().hex[:6]}'
    return dict(run_id=run_id,label=model_identity(now.isoformat(),targets,counts),started_at=now.isoformat(),target_symbols=targets,context_counts=counts)


def model_records(root,store,local_rows,known):
    rows={r['run_id']:dict(r) for r in local_rows if r.get('job_kind')!='data'}
    active=set()
    if hasattr(store,'connect'):
        with store.connect() as db:
            for rid,payload in db.execute('SELECT id,manifest FROM training_runs'):
                row=json.loads(payload)
                if row.get('job_kind')!='data':rows.setdefault(rid,row)
            for payload, in db.execute('SELECT manifest FROM model_registry'):
                active.add(json.loads(payload).get('run_id'))
            data_runs={rid for rid,payload in db.execute('SELECT id,manifest FROM training_runs') if json.loads(payload).get('job_kind')=='data'}
            for table in ('forecasts','forecast_metrics','epoch_metrics','forecast_snapshots'):
                if db.execute('SELECT 1 FROM sqlite_master WHERE name=?',(table,)).fetchone():
                    for rid, in db.execute(f'SELECT DISTINCT run FROM {table}'):
                        if rid not in data_runs:rows.setdefault(rid,dict(run_id=rid,state='complete',model_kind='lstm'))
    inverse={code:name for name,code in known.items()}
    result=[]
    for rid,row in rows.items():
        folder=Path(root)/rid
        targets=row.get('target_symbols') or list(dict.fromkeys(s['symbol'] for s in row.get('stages',[]) if s.get('stage') in ('prepared','trained','training_failed') and s.get('symbol')))
        counts=dict(row.get('context_counts',{}))
        configs=[]
        if folder.exists():
            for path in folder.glob('*_preprocessing.json'):
                try:configs.append((path.name.split('_')[0],json.loads(path.read_text(encoding='utf8'))))
                except (OSError,ValueError):continue
        for code,config in configs:
            name=config.get('target_symbol') or inverse.get(code)
            if name and name not in targets:targets.append(name)
            if name and 'context_universe' in config:counts[name]=len(config['context_universe'])
        if not targets and row.get('training_progress',{}).get('symbol'):targets=[row['training_progress']['symbol']]
        try:identity=model_identity(row.get('started_at'),targets,counts)
        except (ValueError,TypeError):identity=row.get('label',rid)
        marker=folder/'saved.json';saved=marker.exists()
        alias=''
        if saved:
            try:alias=json.loads(marker.read_text(encoding='utf8')).get('name','')
            except (OSError,ValueError):pass
        row.update(identity=identity,display_label=identity+(' · '+alias if alias and alias!=identity else ''),saved=saved,active=rid in active,targets=targets,context_description=' / '.join(str(counts[t]) if counts.get(t) is not None else 'ثبت نشده' for t in targets) or 'ثبت نشده',has_weights=any(folder.glob('*.keras')) if folder.exists() else False)
        result.append(row)
    return sorted(result,key=lambda r:r.get('started_at') or r['run_id'],reverse=True)


def delete_model_run(root,store,run_id):
    root=Path(root).resolve();source=(root/run_id).resolve()
    if not run_id or source.parent!=root or source==root:raise ValueError('Invalid model run path')
    archive_root=(root.parent/'deleted_models').resolve()
    if archive_root.parent!=root.parent:raise ValueError('Invalid recovery path')
    archive_root.mkdir(exist_ok=True)
    archive=archive_root/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'-'+uuid.uuid4().hex[:6]);archive.mkdir()
    moved=False
    try:
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            model_run=db.execute('SELECT manifest FROM training_runs WHERE id=?',(run_id,)).fetchone()
            if model_run and json.loads(model_run[0]).get('job_kind')=='data':raise ValueError('Data jobs cannot be deleted as models')
            tables={}
            existing={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ('forecasts','forecast_metrics','epoch_metrics','forecast_snapshots'):
                if table in existing:tables[table]=db.execute(f'SELECT * FROM {table} WHERE run=?',(run_id,)).fetchall()
            tables['training_runs']=db.execute('SELECT * FROM training_runs WHERE id=?',(run_id,)).fetchall()
            registry=[r for r in db.execute('SELECT * FROM model_registry') if json.loads(r[2]).get('run_id')==run_id]
            tables['model_registry']=registry
            with gzip.open(archive/'database.json.gz','wt',encoding='utf8') as output:
                json.dump(dict(run_id=run_id,tables=tables),output,ensure_ascii=False)
            if source.exists():
                if not source.is_dir():raise ValueError('Expected model run directory')
                shutil.move(str(source),str(archive/'artifacts'));moved=True
            for table in ('forecasts','forecast_metrics','epoch_metrics','forecast_snapshots'):
                if table in existing:db.execute(f'DELETE FROM {table} WHERE run=?',(run_id,))
            db.execute('DELETE FROM training_runs WHERE id=?',(run_id,))
            for row in registry:db.execute('DELETE FROM model_registry WHERE id=?',(row[0],))
    except BaseException:
        if moved and not source.exists():shutil.move(str(archive/'artifacts'),str(source))
        raise
    try:
        status=root/'inference_status.json'
        if status.exists() and json.loads(status.read_text(encoding='utf8')).get('run_id')==run_id:
            shutil.move(str(status),str(archive/'inference_status.json'))
    except (OSError,ValueError):pass
    return archive


from contextlib import contextmanager
from filelock import FileLock, Timeout

@contextmanager
def model_mutation_lock(root):
    """Serialize model deletion; allow a confirmed data-only downloader to continue."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(root/'.models.lock'),timeout=0):
        pipeline=FileLock(str(root/'.pipeline.lock'),timeout=0)
        acquired=False
        try:
            try:pipeline.acquire();acquired=True
            except Timeout:
                from .registry import list_runs
                rows=list_runs(root)
                if not rows or rows[0].get('job_kind')!='data' or rows[0].get('state')!='running':raise
            yield
        finally:
            if acquired:pipeline.release()
