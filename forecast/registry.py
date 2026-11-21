# توابع instruments نگاشت نمادها و jalali_month ماه شمسی تاریخ میلادی را محاسبه می‌کنند.
# توضیح فایل: فهرست نمادها و نگهداری انتخابی مدل؛ list_runs خواندن، prune حذف امن آزمایش‌های قدیمی و settings_store مسیر داده را می‌سازد.
from dataclasses import replace
from pathlib import Path
import json
import shutil
import pandas as pd
from .table_io import read_table,write_table,table_path,learning_tables
from market_data import Settings, CsvMarketStore
from market_data.sqlite_store import make_store
from .pipeline import BASE, RUNS

def settings_store():
    settings = Settings.from_env(BASE/'.env')
    if not settings.root.is_absolute():
        settings = replace(settings, root=BASE/settings.root)
    return settings, make_store(settings.root)

def instruments():
    _, store = settings_store()
    if hasattr(store,'list_instruments'):return store.list_instruments()
    return {store.read_state(p.stem)['symbol']: p.stem for p in (store.root/'state').glob('*.json') if store.read_state(p.stem).get('symbol')}

def list_runs(root=None):
    root = Path(root or RUNS)
    rows = []
    for p in sorted(root.glob('*/status.json'), reverse=True):
        row = json.loads(p.read_text(encoding='utf8'))
        row['saved'] = (p.parent/'saved.json').exists()
        row['label'] = row.get('label', f"شبکه عصبی — {row.get('started_at', row['run_id'])[:16].replace('T',' ')}")
        rows.append(row)
    return rows

def prune(keep, root=None):
    root = Path(root or RUNS).resolve()
    for row in list_runs(root):
        if row['run_id'] == keep or row['state'] == 'running':
            continue
        path = (root/row['run_id']).resolve()
        if path.parent != root:
            raise ValueError('Invalid artifact path')
        if row['saved']:
            selected=json.loads((path/'saved.json').read_text(encoding='utf8')).get('codes')
            if selected:
                for file in path.iterdir():
                    prefix=file.name.split('_')[0]
                    if file.is_file() and prefix.isdigit() and prefix not in selected:
                        file.unlink()
                for name in ('predictions.json','metrics.json'):
                    target=table_path(path/name)
                    if target.exists():
                        frame=read_table(target,dtype={'insCode':str})
                        if 'insCode' in frame:
                            from .table_io import write_table
                            write_table(frame[frame.insCode.isin(selected)],target)
        else:
            shutil.rmtree(path)

def jalali_month(date):
    gy, gm, gd = date.year-1600, date.month-1, date.day-1
    days = 365*gy+(gy+3)//4-(gy+99)//100+(gy+399)//400
    days += sum([31,28,31,30,31,30,31,31,30,31,30,31][:gm])+gd
    if gm>1 and (date.year%4==0 and (date.year%100!=0 or date.year%400==0)):
        days += 1
    days -= 79
    days %= 12053
    days %= 1461
    if days >= 366:
        days = (days-1)%365
    return days//31+1 if days<186 else (days-186)//30+7
