# توضیح فایل: دریافت واقعی نمادهای نمای بازار و صندوق‌های بخشی در SQLite؛ ثبت پیشرفت و خطای هر نماد.
import sys,json,time,threading
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from forecast.registry import settings_store
from market_data import TsetmcClient,sync_market
from market_data.csv_store import _write_bytes_atomic
settings,store=settings_store()
if not hasattr(store,'database'):raise RuntimeError('Migration must finish before expansion')
settings=replace(settings,timeout=8,retries=0,delay=.6)
candidates=['فولاد','فملی','فارس','شپنا','وبملت','خودرو','شستا','یاقوت','استیل','پتروآگاه','دارونو','سیمانو']
starts=[store.load_daily(code).Date.min() for name,code in store.list_instruments().items() if name in ['توان','اهرم','موج']]
import pandas as pd
start=str((min(starts)-pd.Timedelta(days=60)).date())
path=settings.root/'universe_expansion.json'
report=json.loads(path.read_text(encoding='utf8')) if path.exists() else {'hourly_start':start,'daily_scope':'all API history','symbols':{},'progress':{}}
path=settings.root/'universe_expansion.json'
def save():_write_bytes_atomic(path,json.dumps(report,ensure_ascii=False,indent=2))
def fetch(symbol):
 client=TsetmcClient(settings)
 try:
  code=client.resolve_symbol_code(symbol)
  # Network workers never write SQLite concurrently; per-day commits are short and serialized by SQLite.
  def progress(details):
   report['progress'][symbol]=details
   try:save()
   except PermissionError:pass
  result=sync_market({symbol:code},settings=settings,client=client,store=store,max_hourly_days=0,workers=6,full_reconcile=True,hourly_start=start,progress=progress)
  return symbol,result[symbol]
 finally:client.session.close()
save()
# sync_market has a global ingestion lock: symbols are sequential, requests within each symbol are bounded-parallel.
for symbol in candidates:
 try:
  for attempt in range(18000):
   try:
    name,result=fetch(symbol);report['symbols'][name]=result;break
   except RuntimeError as exc:
    if 'already running' not in str(exc):raise
    time.sleep(.2)
 except Exception as exc:report['symbols'][symbol]={'status':'failed','error':str(exc)}
 save()
print(json.dumps(report,ensure_ascii=True))
