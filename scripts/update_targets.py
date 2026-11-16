# توضیح فایل: دریافت تازهٔ سه دارایی هدف، منتظر آزاد شدن قفل دریافت نماد زمینه می‌ماند.
import sys,time,json
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from forecast.registry import settings_store,instruments
from market_data import sync_market
settings,store=settings_store();settings=replace(settings,timeout=10,retries=1,delay=.6)
report={}
for name in ['توان','اهرم','موج']:
 for attempt in range(18000):
  try:
   code=instruments()[name]; report[name]=sync_market({name:code},settings=settings,store=store,workers=4,max_hourly_days=10);break
  except RuntimeError as e:
   if 'already running' not in str(e):raise
   time.sleep(.2)
  except Exception as e:report[name]={'error':str(e)};break
Path('data/target_refresh.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
