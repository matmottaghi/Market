# توضیح فایل: یک دور تلاش مجدد برای خطاهای موقت دریافت؛ روزهای واقعاً خالی دوباره درخواست نمی‌شوند.
# fetch دریافت روز با client مستقل؛ main ثبت تدریجی نتیجه و گزارش قابل بازبینی تحت قفل دریافت.
import sys,json
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor,as_completed
from filelock import FileLock
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from forecast.registry import settings_store
from market_data import TsetmcClient
from market_data.csv_store import _write_bytes_atomic

def main():
 settings,store=settings_store();settings=replace(settings,timeout=20,retries=1,delay=.6)
 report=dict(attempted=0,saved=0,errors=[])
 with FileLock(str(settings.root/'.market_sync.lock'),timeout=0):
  with store.connect() as db:
   jobs=db.execute("SELECT i.symbol,e.code,e.date,d.volume FROM ingestion_issues e JOIN instruments i ON i.code=e.code JOIN daily_bars d ON d.code=e.code AND d.date=e.date WHERE e.error!='empty tradeHistory' ORDER BY e.date DESC").fetchall()
  def fetch(job):
   name,code,date,volume=job;client=TsetmcClient(settings)
   try:return job,client.get_hourly_day(name,code,date,volume),None
   except Exception as exc:return job,None,str(exc)
   finally:client.session.close()
  with ThreadPoolExecutor(max_workers=6) as executor:
   for future in as_completed([executor.submit(fetch,j) for j in jobs]):
    (name,code,date,volume),frame,error=future.result();report['attempted']+=1
    if error or frame.empty:
     error=error or 'empty tradeHistory';store.record_hourly_failure(code,date,error)
     report['errors'].append(dict(symbol=name,date=date,error=error))
    else:
     store.save_hourly_day(frame);report['saved']+=1
    _write_bytes_atomic(settings.root/'retry_report.json',json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps(report,ensure_ascii=True))

if __name__=='__main__':main()
