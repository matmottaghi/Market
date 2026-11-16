# توضیح فایل: دریافت واقعی snapshot پایه و زنجیره و ثبت آن در دیتابیس.
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from forecast.registry import settings_store,instruments
from market_data.live_options import fetch_snapshot
settings,store=settings_store()
symbol=sys.argv[1] if len(sys.argv)>1 else 'اهرم'
s=fetch_snapshot(settings,store,instruments()[symbol])
print(json.dumps({k:v for k,v in s.items() if k!='contracts'},ensure_ascii=True));print('contracts',len(s['contracts']))
