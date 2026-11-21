# توضیح فایل: مخزن اصلی SQLite با WAL، کلید یکتا و نگهداری تمام ستون‌های منبع در JSON.
# توابع make_store انتخاب backend، migrate انتقال و تطبیق؛ متدهای SqliteMarketStore خواندن، upsert، وضعیت و خطا را پیاده می‌کنند.
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import sqlite3
from contextlib import contextmanager
import pandas as pd
from .csv_store import CsvMarketStore, _code, _write_bytes_atomic


class SqliteMarketStore(CsvMarketStore):
    def __init__(self, root):
        super().__init__(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root/'market.sqlite3'
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS instruments(code TEXT PRIMARY KEY,symbol TEXT,state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS daily_bars(code TEXT NOT NULL,date TEXT NOT NULL,close REAL,volume REAL,payload TEXT NOT NULL,ingested_at TEXT NOT NULL,PRIMARY KEY(code,date));
            CREATE TABLE IF NOT EXISTS hourly_bars(code TEXT NOT NULL,date TEXT NOT NULL,hour TEXT NOT NULL,close REAL,volume REAL,quality TEXT,payload TEXT NOT NULL,ingested_at TEXT NOT NULL,PRIMARY KEY(code,hour));
            CREATE INDEX IF NOT EXISTS hourly_day ON hourly_bars(code,date);
            CREATE TABLE IF NOT EXISTS ingestion_issues(code TEXT,date TEXT,error TEXT,retry_after TEXT,PRIMARY KEY(code,date));
            CREATE TABLE IF NOT EXISTS snapshots(id INTEGER PRIMARY KEY,kind TEXT,underlying TEXT,retrieved_at TEXT,payload TEXT);
            CREATE TABLE IF NOT EXISTS model_registry(id TEXT PRIMARY KEY,label TEXT,manifest TEXT);
            CREATE TABLE IF NOT EXISTS training_runs(id TEXT PRIMARY KEY,label TEXT,state TEXT,manifest TEXT);
            CREATE TABLE IF NOT EXISTS epoch_metrics(run TEXT,code TEXT,epoch INTEGER,payload TEXT,PRIMARY KEY(run,code,epoch));
            CREATE TABLE IF NOT EXISTS forecasts(run TEXT,code TEXT,step INTEGER,payload TEXT,PRIMARY KEY(run,code,step));
            CREATE TABLE IF NOT EXISTS forecast_metrics(run TEXT,code TEXT,step INTEGER,payload TEXT,PRIMARY KEY(run,code,step));
            ''')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.database,timeout=30)
        try:
            db.execute('PRAGMA busy_timeout=30000')
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA foreign_keys=ON')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def list_instruments(self):
        with self.connect() as db:
            return {symbol:code for code,symbol in db.execute('SELECT code,symbol FROM instruments WHERE symbol IS NOT NULL ORDER BY symbol')}

    def inventory(self):
        """Live row counts without deserializing complete histories for the dashboard."""
        with self.connect() as db:
            rows=db.execute('''SELECT i.symbol,
                (SELECT COUNT(*) FROM daily_bars d WHERE d.code=i.code),
                (SELECT COUNT(*) FROM hourly_bars h WHERE h.code=i.code),
                (SELECT MAX(date) FROM daily_bars d WHERE d.code=i.code),
                (SELECT MAX(date) FROM hourly_bars h WHERE h.code=i.code),
                (SELECT COUNT(*) FROM ingestion_issues e WHERE e.code=i.code),i.code,
                (SELECT MAX(hour) FROM hourly_bars h WHERE h.code=i.code),
                (SELECT MIN(date) FROM daily_bars d WHERE d.code=i.code),
                (SELECT MIN(date) FROM hourly_bars h WHERE h.code=i.code)
                FROM instruments i ORDER BY i.symbol''').fetchall()
        return [dict(zip(['symbol','daily','hourly','daily_end','hourly_end','issues','code','last_hour','daily_start','hourly_start'],row)) for row in rows]

    def delete_instrument(self,code):
        """Remove active symbol data transactionally after writing a recovery archive."""
        import gzip
        code=_code(str(code))
        archive_dir=self.root/'deleted_symbols';archive_dir.mkdir(exist_ok=True)
        archive=archive_dir/(code+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json.gz')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            tables={name:db.execute(f'SELECT * FROM {name} WHERE code=?',(code,)).fetchall()
                    for name in ('instruments','daily_bars','hourly_bars','ingestion_issues')}
            if not tables['instruments']:raise ValueError('Unknown instrument')
            tables['snapshots']=db.execute('SELECT * FROM snapshots WHERE underlying=?',(code,)).fetchall()
            tables['model_registry']=db.execute('SELECT * FROM model_registry WHERE id=?',('active:'+code,)).fetchall()
            with gzip.open(archive,'wt',encoding='utf8') as output:
                json.dump(dict(version=1,code=code,deleted_at=datetime.now(timezone.utc).isoformat(),tables=tables),output,ensure_ascii=False)
            for name in ('daily_bars','hourly_bars','ingestion_issues','instruments'):
                db.execute(f'DELETE FROM {name} WHERE code=?',(code,))
            db.execute('DELETE FROM snapshots WHERE underlying=?',(code,))
            db.execute('DELETE FROM model_registry WHERE id=?',('active:'+code,))
        self.state_path(code).unlink(missing_ok=True)
        return archive

    def latest_rows(self,code,frequency,limit=10):
        """Read a bounded newest-first preview; never load complete history for a table."""
        if frequency not in ('daily','hourly'):raise ValueError('Unknown frequency')
        if not 1<=int(limit)<=50:raise ValueError('Preview limit must be 1 to 50')
        column='date' if frequency=='daily' else 'hour'
        with self.connect() as db:
            rows=db.execute(f'SELECT payload,ingested_at FROM {frequency}_bars WHERE code=? ORDER BY {column} DESC LIMIT ?',(_code(code),int(limit))).fetchall()
        return [dict(json.loads(payload),_ingested_at=at) for payload,at in rows]

    def coverage_summary(self,code):
        """Count daily active sessions with absent or volume-matched hourly candles."""
        matched="h.bars IS NOT NULL AND h.qmin='matched' AND h.qmax='matched' AND ABS(h.total/d.volume-1)<=0.001"
        with self.connect() as db:
            row=db.execute(f'''WITH h AS (
                SELECT date,COUNT(*) bars,SUM(volume) total,MIN(COALESCE(quality,'')) qmin,
                MAX(COALESCE(quality,'')) qmax,MAX(hour) last_hour FROM hourly_bars WHERE code=? GROUP BY date)
                SELECT COUNT(*),SUM(CASE WHEN {matched} THEN 1 ELSE 0 END),
                SUM(CASE WHEN h.bars IS NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN h.bars IS NOT NULL AND NOT({matched}) THEN 1 ELSE 0 END),
                MAX(CASE WHEN {matched} THEN h.last_hour END)
                FROM daily_bars d LEFT JOIN h ON h.date=d.date
                WHERE d.code=? AND d.volume>0 AND COALESCE(json_extract(d.payload,'$.zTotTran'),0)>0''',(_code(code),_code(code))).fetchone()
        return dict(active_days=row[0],matched_days=row[1] or 0,missing_days=row[2] or 0,invalid_days=row[0]-(row[1] or 0)-(row[2] or 0),last_matched_hour=row[4])

    def read_state(self, ins_code):
        with self.connect() as db:
            row=db.execute('SELECT state FROM instruments WHERE code=?',(_code(ins_code),)).fetchone()
        return json.loads(row[0]) if row else {}

    def update_state(self,ins_code,**updates):
        code=_code(ins_code);state=self.read_state(code);state.update(updates)
        state['updated_at_utc']=datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute('INSERT INTO instruments VALUES(?,?,?) ON CONFLICT(code) DO UPDATE SET symbol=excluded.symbol,state=excluded.state',(code,state.get('symbol'),json.dumps(state,ensure_ascii=False)))
        _write_bytes_atomic(self.state_path(code),json.dumps(state,ensure_ascii=False,indent=2))

    def upsert_daily(self,rows):
        if rows.empty:return dict(months_touched=0,incoming_rows=0,new_rows=0)
        records=json.loads(rows.to_json(orient='records',date_format='iso',double_precision=15))
        added=0;now=datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            for row in records:
                code=_code(str(row['insCode']));date=str(pd.Timestamp(row['Date']).date())
                added+=not bool(db.execute('SELECT 1 FROM daily_bars WHERE code=? AND date=?',(code,date)).fetchone())
                row['Date']=date
                db.execute('INSERT INTO daily_bars VALUES(?,?,?,?,?,?) ON CONFLICT(code,date) DO UPDATE SET close=excluded.close,volume=excluded.volume,payload=excluded.payload,ingested_at=excluded.ingested_at WHERE daily_bars.payload<>excluded.payload',(code,date,row.get('pClosing'),row.get('qTotTran5J'),json.dumps(row,ensure_ascii=False,allow_nan=False),now))
        return dict(months_touched=len({str(r['Date'])[:7] for r in records}),incoming_rows=len(records),new_rows=int(added))

    def day_quality(self,code,date):
        with self.connect() as db:
            key=(_code(str(code)),str(pd.Timestamp(date).date()))
            rows=db.execute('SELECT DISTINCT quality FROM hourly_bars WHERE code=? AND date=?',key).fetchall()
            quality=rows[0][0] if len(rows)==1 else 'missing' if not rows else 'mixed'
            if quality=='matched':
                total=db.execute('SELECT SUM(volume) FROM hourly_bars WHERE code=? AND date=?',key).fetchone()[0]
                reference=db.execute('SELECT volume FROM daily_bars WHERE code=? AND date=?',key).fetchone()
                if not reference or not reference[0] or reference[0]<=0:quality='unverified'
                elif abs(total/reference[0]-1)>.001:quality='volume_mismatch'
        return quality

    def day_error(self,code,date):
        with self.connect() as db:
            row=db.execute('SELECT error FROM ingestion_issues WHERE code=? AND date=?',(_code(str(code)),str(pd.Timestamp(date).date()))).fetchone()
        return row[0] if row else ''

    def record_run(self,status):
        with self.connect() as db:
            db.execute('INSERT INTO training_runs VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET label=excluded.label,state=excluded.state,manifest=excluded.manifest',(status['run_id'],status.get('label'),status['state'],json.dumps(status,ensure_ascii=False)))

    def record_epoch(self,run,code,row):
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO epoch_metrics VALUES(?,?,?,?)',(run,code,row['epoch'],json.dumps(row,ensure_ascii=False)))

    def record_results(self,run,predictions,metrics):
        with self.connect() as db:
            for table,records in [('forecasts',predictions),('forecast_metrics',metrics)]:
                for row in records:
                    db.execute(f'INSERT OR REPLACE INTO {table} VALUES(?,?,?,?)',(run,str(row['insCode']),int(row['horizon']),json.dumps(row,ensure_ascii=False,allow_nan=False)))

    def hourly_exists(self,ins_code,date,*,retry_incomplete=False):
        quality=self.day_quality(ins_code,date)
        return quality=='matched' if retry_incomplete else quality!='missing'

    def record_hourly_failure(self,ins_code,date,error,*,retry_after_hours=24):
        with self.connect() as db:
            db.execute('INSERT INTO ingestion_issues VALUES(?,?,?,?) ON CONFLICT(code,date) DO UPDATE SET error=excluded.error,retry_after=excluded.retry_after',(_code(ins_code),str(pd.Timestamp(date).date()),str(error)[:1000],(datetime.now(timezone.utc)+timedelta(hours=retry_after_hours)).isoformat()))

    def should_fetch_hourly(self,ins_code,date,*,retry_incomplete=False):
        if self.hourly_exists(ins_code,date,retry_incomplete=True):return False
        if retry_incomplete:return True
        with self.connect() as db:
            row=db.execute('SELECT retry_after FROM ingestion_issues WHERE code=? AND date=?',(_code(ins_code),str(pd.Timestamp(date).date()))).fetchone()
        return not row or datetime.now(timezone.utc)>=datetime.fromisoformat(row[0])

    def save_hourly_day(self,rows):
        if rows.empty:return
        codes=rows.insCode.astype(str).unique();dates=pd.to_datetime(rows.Date).dt.normalize().unique()
        if len(codes)!=1 or len(dates)!=1:raise ValueError('Expected one instrument-day')
        code=_code(codes[0]);date=str(pd.Timestamp(dates[0]).date())
        if self.day_quality(code,date)=='matched' and not rows.DataQuality.eq('matched').all():
            raise ValueError('Refusing invalid replacement of matched day')
        records=json.loads(rows.to_json(orient='records',date_format='iso',double_precision=15))
        with self.connect() as db:
            db.execute('DELETE FROM hourly_bars WHERE code=? AND date=?',(code,date))
            for row in records:
                row['Date']=date;row['Hour']=pd.Timestamp(row['Hour']).isoformat()
                db.execute('INSERT INTO hourly_bars VALUES(?,?,?,?,?,?,?,?)',(code,date,row['Hour'],row.get('Close'),row.get('Volume'),row.get('DataQuality'),json.dumps(row,ensure_ascii=False,allow_nan=False),datetime.now(timezone.utc).isoformat()))
            db.execute('DELETE FROM ingestion_issues WHERE code=? AND date=?',(code,date))

    def _load(self,table,code,start=None,end=None,**kwargs):
        query=f'SELECT payload FROM {table} WHERE code=?';args=[_code(code)]
        if start:query+=' AND date>=?';args.append(str(pd.Timestamp(start).date()))
        if end:query+=' AND date<=?';args.append(str(pd.Timestamp(end).date()))
        query+=' ORDER BY '+('hour' if table=='hourly_bars' else 'date')
        with self.connect() as db:records=[json.loads(row[0]) for row in db.execute(query,args)]
        frame=pd.DataFrame(records)
        for column in ('Date','Hour'):
            if column in frame:frame[column]=pd.to_datetime(frame[column])
        return frame

    def load_daily(self,code,**kwargs):return self._load('daily_bars',code,**kwargs)
    def load_hourly(self,code,**kwargs):return self._load('hourly_bars',code,**kwargs)
    def iter_daily(self,code,**kwargs):
        frame=self.load_daily(code,**kwargs)
        if not frame.empty:yield frame
    def iter_hourly(self,code,**kwargs):
        frame=self.load_hourly(code,**kwargs)
        if not frame.empty:yield frame

def make_store(root):
    if not (Path(root)/'storage.json').exists() and not (Path(root)/'market.sqlite3').exists() and any((Path(root)/'daily').glob('*/*.csv')):
        migrate(root)
    return SqliteMarketStore(root)

def migrate(root):
    if (Path(root)/'storage.json').exists():
        report=Path(root)/'migration_report.json'
        return json.loads(report.read_text(encoding='utf8')) if report.exists() else []
    source=CsvMarketStore(root);target=SqliteMarketStore(root);report=[]
    for path in sorted((Path(root)/'state').glob('*.json')):
        code=path.stem;state=source.read_state(code)
        daily=source.load_daily(code);hourly=source.load_hourly(code)
        target.upsert_daily(daily)
        if not hourly.empty:
            for _,day in hourly.groupby('Date'):target.save_hourly_day(day)
        target.update_state(code,**state)
        for frequency,original in [('daily',daily),('hourly',hourly)]:
            restored=getattr(target,f'load_{frequency}')(code)
            keys=['insCode','Date']+(['Hour'] if frequency=='hourly' else [])
            if original.empty:assert restored.empty;continue
            original=original.sort_values(keys).reset_index(drop=True)
            restored=restored.reindex(columns=original.columns).sort_values(keys).reset_index(drop=True)
            for column in ['insCode','Symbol']:
                if column in original:
                    original[column]=original[column].astype('string');restored[column]=restored[column].astype('string')
            pd.testing.assert_frame_equal(original,restored,check_dtype=False,rtol=1e-10,atol=1e-10)
            report.append(dict(code=code,symbol=state.get('symbol'),table=frequency,rows=len(original),verified=True))
    _write_bytes_atomic(Path(root)/'migration_report.json',json.dumps(report,ensure_ascii=False,indent=2))
    _write_bytes_atomic(Path(root)/'storage.json',json.dumps(dict(backend='sqlite',database='market.sqlite3',timezone='Asia/Tehran')))
    return report
