# توضیح فایل: حذف CSVهای قدیمی فقط بعد از تطبیق کلیدهای داده با SQLite، تبدیل آثار مدل به JSON و تأیید نسخهٔ فشردهٔ قابل بازگشت.
# main قفل‌ها و دامنهٔ مسیر را کنترل می‌کند؛ digest صحت هر عضو پشتیبان را پیش از حذف می‌سنجد. هیچ پوشه‌ای بازگشتی حذف نمی‌شود.
import csv
import hashlib
import json
import sys
import zipfile
from contextlib import ExitStack
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from filelock import FileLock

BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE))
from forecast.registry import settings_store
from forecast.table_io import write_table,read_table

def digest(content):return hashlib.sha256(content).hexdigest()

def main():
    data=(BASE/'data').resolve();_,store=settings_store();runs=data/'training_runs'
    if not store.database.resolve().is_relative_to(data):raise ValueError('Database outside expected workspace data directory')
    with ExitStack() as locks:
        locks.enter_context(FileLock(str(data/'.csv_retirement.lock'),timeout=0))
        if '--online-data-refresh' in sys.argv:
            # Only the inspected SQLite-only download may run concurrently; training can prune artifacts.
            latest=sorted(runs.glob('*/status.json'))[-1]
            status=json.loads(latest.read_text(encoding='utf8'))
            environment=json.loads((latest.parent/'environment.json').read_text(encoding='utf8'))
            source=(latest.parent/'neural_training.py').read_text(encoding='utf8')
            if status.get('job_kind')!='data' or environment.get('feature_schema')!='cross_asset_exact_v3' or 'history_start' not in source:
                raise ValueError('Concurrent process is not a verified SQLite-only data refresh')
        else:
            locks.enter_context(FileLock(str(runs/'.pipeline.lock'),timeout=0))
            locks.enter_context(FileLock(str(store.root/'.market_sync.lock'),timeout=0))
        paths=sorted(data.rglob('*.csv'))
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(data):raise ValueError('Unsafe CSV path')
        with store.connect() as db:
            if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Database integrity check failed')
            daily=set(db.execute('SELECT code,date FROM daily_bars'))
            hourly=set(db.execute('SELECT code,hour FROM hourly_bars'))
        checked=0
        for path in paths:
            if not path.is_relative_to(store.root/'daily') and not path.is_relative_to(store.root/'hourly'):continue
            frequency=path.relative_to(store.root).parts[0]
            with path.open(encoding='utf-8-sig',newline='') as stream:
                for row in csv.DictReader(stream):
                    code=row['insCode']
                    key=(code,str(pd.Timestamp(row['Date']).date())) if frequency=='daily' else (code,pd.Timestamp(row['Hour']).isoformat())
                    if key not in (daily if frequency=='daily' else hourly):raise ValueError(f'Unmigrated row in {path.relative_to(BASE)}: {key}')
                    checked+=1
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        archive=data/'backups'/f'csv_retirement_{stamp}.zip';archive.parent.mkdir(parents=True,exist_ok=True)
        manifest=[]
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as backup:
            for path in paths:
                content=path.read_bytes();name=path.relative_to(BASE).as_posix()
                manifest.append(dict(path=name,sha256=digest(content),bytes=len(content)))
                backup.writestr(name,content)
            backup.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        with zipfile.ZipFile(archive) as backup:
            for row in manifest:
                if digest(backup.read(row['path']))!=row['sha256']:raise ValueError('Archive verification failed')
        converted=0
        for path in paths:
            if not path.is_relative_to(runs):continue
            try:frame=pd.read_csv(path,dtype={'insCode':str})
            except pd.errors.EmptyDataError:frame=pd.DataFrame()
            target=path.with_suffix('.json')
            if not target.exists():write_table(frame,target);converted+=1
            restored=read_table(target,dtype={'insCode':str})
            pd.testing.assert_frame_equal(frame,restored,check_dtype=False,check_like=True,rtol=1e-9,atol=1e-9)
        removed=0
        for path,row in zip(paths,manifest):
            # Recheck containment and bytes immediately before the explicitly authorized deletion.
            if not path.resolve().is_relative_to(data) or digest(path.read_bytes())!=row['sha256']:
                raise ValueError('CSV changed during cleanup; deletion stopped')
            path.unlink();removed+=1
        report=dict(removed_csv=removed,market_rows_verified=checked,model_tables_converted=converted,archive=str(archive),remaining_csv=len(list(data.rglob('*.csv'))))
        (data/'csv_retirement_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(report,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
