# توضیح فایل: data_page پنل مستقل آرشیو، پیش‌نمایش آخرین رکوردها و کنترل دریافت را بدون آموزش یا درخواست شبکه می‌سازد.
# preview_rows مقادیر منبع را با حفظ تفاوت قیمت پایانی/آخرین معامله برای جدول آماده می‌کند.
from urllib.parse import urlencode
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponseBadRequest
from .registry import settings_store,instruments,list_runs

QUALITY={'matched':'حجم همخوان','volume_mismatch':'حجم نامنطبق','missing':'بدون کندل','unverified':'تأییدنشده','mixed':'کیفیت ترکیبی'}

def preview_rows(rows,frequency):
    columns=(['Date','priceFirst','priceMax','priceMin','pClosing','pDrCotVal','qTotTran5J','zTotTran'] if frequency=='daily'
             else ['Hour','Open','High','Low','Close','Volume','Trades'])
    result=[]
    for row in rows:
        values=[]
        for key in columns:
            value=row.get(key)
            if value is None or pd.isna(value):value='—'
            elif key in ('Date','Hour'):value=str(value).replace('T',' ')[:10 if key=='Date' else 16]
            elif isinstance(value,(int,float)):value=f'{value:,.2f}'.rstrip('0').rstrip('.') if float(value)%1 else f'{value:,.0f}'
            values.append(value)
        quality=row.get('DataQuality','unverified')
        result.append(dict(values=values,quality=QUALITY.get(quality,'تأییدنشده'),matched=quality=='matched',received=row.get('_ingested_at','')))
    return result

def data_page(request):
    _,store=settings_store();known=instruments()
    symbol=request.GET.get('symbol','اهرم' if 'اهرم' in known else next(iter(known),''))
    if symbol and symbol not in known:return HttpResponseBadRequest('نماد در آرشیو ثبت نشده است')
    try:limit=int(request.GET.get('limit','10'))
    except ValueError:limit=10
    if limit not in (10,25,50):limit=10
    inventory=store.inventory() if hasattr(store,'inventory') else []
    latest_daily=max((r['daily_end'] for r in inventory if r['daily_end']),default=None)
    latest_hour=max((r.get('last_hour') for r in inventory if r.get('last_hour')),default=None)
    for row in inventory:
        row['url']='/data/?'+urlencode(dict(symbol=row['symbol'],limit=limit))
        row['selected']=row['symbol']==symbol
        row['status']='بدون داده' if not row['daily'] else 'هم‌تاریخ با آخرین روز آرشیو' if row['daily_end']==latest_daily else 'قدیمی‌تر از آخرین روز آرشیو'
    code=known.get(symbol);daily=[];hourly=[];coverage={};selected={}
    if code:
        selected=next((r for r in inventory if r['symbol']==symbol),{})
        if hasattr(store,'latest_rows'):
            daily=preview_rows(store.latest_rows(code,'daily',limit),'daily')
            raw_hourly=store.latest_rows(code,'hourly',limit)
            quality_by_date={date:store.day_quality(code,date) for date in {row['Date'] for row in raw_hourly}}
            for row in raw_hourly:row['DataQuality']=quality_by_date[row['Date']]
            hourly=preview_rows(raw_hourly,'hourly')
            coverage=store.coverage_summary(code)
    jobs=[r for r in list_runs() if r.get('job_kind')=='data']
    return render(request,'forecast/data.html',dict(symbols=known,symbol=symbol,inventory=inventory,selected=selected,
        daily_rows=daily,hourly_rows=hourly,coverage=coverage,limit=limit,
        totals=dict(symbols=len(known),daily=sum(r['daily'] for r in inventory),hourly=sum(r['hourly'] for r in inventory),issues=sum(r['issues'] for r in inventory)),
        latest_daily=latest_daily,latest_hour=latest_hour,latest_job=jobs[0] if jobs else None))
