# توضیح فایل: صورت‌وضعیت واقعی SQLite پس از مهاجرت و تلاش مجدد؛ هیچ دادهٔ قیمتی را تغییر نمی‌دهد.
# main شمار رکوردها، دامنهٔ تاریخ و پوشش ساعت روزهای فعال را با دادهٔ روزانه تطبیق می‌دهد.
import sys,json
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from forecast.registry import settings_store
from market_data.csv_store import _write_bytes_atomic

def main():
 settings,store=settings_store();rows=[]
 original={'اطلس','عیار','اهرم','توان','موج'}
 with store.connect() as db: integrity=db.execute('PRAGMA integrity_check').fetchone()[0]
 for name,code in store.list_instruments().items():
  daily=store.load_daily(code);hourly=store.load_hourly(code)
  if daily.empty:continue
  active=daily[(daily.qTotTran5J>0)&(daily.zTotTran>0)]
  dates=active.Date if name in original else active.loc[active.Date>=pd.Timestamp('2021-10-21'),'Date']
  coverage=pd.DataFrame([dict(Symbol=name,insCode=code,Date=date,quality=store.day_quality(code,date),error=store.day_error(code,date)) for date in dates])
  rows.append(dict(symbol=name,code=code,daily_rows=len(daily),hourly_rows=len(hourly),daily_start=str(daily.Date.min().date()),daily_end=str(daily.Date.max().date()),hourly_start=str(hourly.Date.min().date()) if not hourly.empty else None,hourly_end=str(hourly.Date.max().date()) if not hourly.empty else None,expected_days=len(dates),coverage={str(k):int(v) for k,v in coverage.quality.value_counts().items()}))
 report=dict(integrity=integrity,daily_scope='all available API history',new_assets_hourly_requested_start='2021-10-21',daily_rows=sum(r['daily_rows'] for r in rows),hourly_rows=sum(r['hourly_rows'] for r in rows),instruments=rows)
 _write_bytes_atomic(settings.root/'data_audit.json',json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps(report,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
