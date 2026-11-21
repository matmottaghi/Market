# توضیح فایل: catalog موجودی پیش‌بینی هر نماد را از فایل‌های مدل و SQLite جمع می‌کند؛ choose انتخاب سازگار، load_rows خواندن خروجی و active_run مدل فعال را انجام می‌دهند.
import json
from .table_io import table_path,read_table

def catalog(rows,root,store):
    result={r['run_id']:dict(r,ready_codes=set()) for r in rows}
    if hasattr(store,'connect'):
        with store.connect() as db:
            available=db.execute('SELECT DISTINCT run,code FROM forecasts').fetchall()
            manifests={rid:json.loads(payload) for rid,payload in db.execute('SELECT id,manifest FROM training_runs') if any(run==rid for run,_ in available)}
        for run,code in available:
            if run not in result:
                row=manifests.get(run,dict(run_id=run,label='خروجی تاریخی — '+run[:15],state='complete',model_kind='lstm'))
                result[run]=dict(row,saved=False,ready_codes=set())
            result[run]['ready_codes'].add(str(code))
    for run,row in result.items():
        path=table_path(root/run/'predictions.json')
        if path.exists():
            frame=read_table(path,dtype={'insCode':str})
            if 'insCode' in frame:row['ready_codes'].update(frame.insCode)
    return sorted(result.values(),key=lambda r:r['run_id'],reverse=True)

def active_run(store,code):
    if not code or not hasattr(store,'connect'):return None
    with store.connect() as db:row=db.execute('SELECT manifest FROM model_registry WHERE id=?',('active:'+code,)).fetchone()
    return json.loads(row[0]).get('run_id') if row else None

def choose(rows,code,requested=None,active=None):
    ready=[r for r in rows if code in r.get('ready_codes',set())]
    return next((r for r in ready if r['run_id']==requested),None) or next((r for r in ready if r['run_id']==active),None) or (ready[0] if ready else None)

def load_rows(run,code,root,store,kind='forecasts'):
    if not run or not code:return []
    if hasattr(store,'connect'):
        if kind not in ('forecasts','forecast_metrics'):raise ValueError('Unknown result kind')
        with store.connect() as db:rows=db.execute(f'SELECT payload FROM {kind} WHERE run=? AND code=? ORDER BY step',(run['run_id'],code)).fetchall()
        if rows:return [json.loads(r[0]) for r in rows]
    path=table_path(root/run['run_id']/('metrics.json' if kind=='forecast_metrics' else 'predictions.json'))
    if not path.exists():return []
    frame=read_table(path,dtype={'insCode':str})
    return frame[frame.insCode==code].to_dict('records')

def reusable(root,run,code):
    if not run or not code:return False
    folder=root/run['run_id']
    config=folder/f'{code}_preprocessing.json'
    if not (folder/f'{code}_lstm.keras').exists() or not config.exists():return False
    try:return json.loads(config.read_text(encoding='utf8')).get('target_schema')!='hourly_returns_v2'
    except (OSError,ValueError):return False
