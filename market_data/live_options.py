# توضیح فایل: snapshot عمومی TSETMC؛ هویت، تاریخ و قیمت پایانی کنترل می‌شوند. قیمت آپشن بدون timestamp منبع، زندهٔ تضمینی نامیده نمی‌شود.
# search_symbols جستجوی نماد؛ fetch_snapshot ذخیرهٔ زنجیره و قیمت در SQLite با زمان دریافت و زمان مستقل منبع.
from datetime import datetime, timezone
import json
from urllib.parse import quote
import pandas as pd
from .provider import TsetmcClient, _fa_normalize

def search_symbols(settings,query):
    client=TsetmcClient(settings)
    try:
        rows=client._json('Instrument/GetInstrumentSearch/'+quote(query,safe='')).get('instrumentSearch',[])
        results=[];seen=set()
        for row in rows:
            code=str(row.get('insCode',''));symbol=str(row.get('lVal18AFC','')).strip()
            if not code.isdigit() or code=='0' or code in seen:continue
            seen.add(code);results.append(dict(code=code,symbol=symbol,name=row.get('lVal30','')))
        return sorted(results,key=lambda row:(_fa_normalize(row['symbol'])!=_fa_normalize(query),_fa_normalize(row['name']).startswith('اختیار'),len(row['symbol']),row['symbol']))[:40]
    finally:client.session.close()

def fetch_snapshot(settings,store,code):
    client=TsetmcClient(settings)
    try:
        info=client._json(f'Instrument/GetInstrumentInfo/{code}').get('instrumentInfo',{})
        if str(info.get('insCode'))!=str(code):raise ValueError('Underlying identity mismatch')
        raw=client._json(f'ClosingPrice/GetClosingPriceInfo/{code}').get('closingPriceInfo',{})
        daily=client.get_daily(info['lVal18AFC'].strip(),str(code))
        date=pd.to_datetime(str(int(raw['dEven'])),format='%Y%m%d')
        reference=daily[pd.to_datetime(daily.Date).dt.normalize().eq(date)]
        if reference.empty or abs(float(reference.pClosing.iloc[-1])-float(raw['pClosing']))>max(1,float(raw['pClosing'])*.0001):
            raise ValueError('Quote does not match same-day daily close; snapshot rejected')
        if str(raw.get('insCode')) not in ('0',str(code)):raise ValueError('Wrong quote instrument')
        h=str(int(raw.get('lastHEven') or raw.get('hEven') or 0)).zfill(6)
        quoted=pd.to_datetime(date.strftime('%Y%m%d')+h,format='%Y%m%d%H%M%S').tz_localize('Asia/Tehran')
        payload=client._json('Instrument/GetInstrumentOptionMarketWatch/1')
        pairs=payload.get('instrumentOptMarketWatch')
        if not isinstance(pairs,list):raise ValueError('Invalid option market response')
        contracts=[]
        for pair in pairs:
            if str(pair.get('uaInsCode'))!=str(code):continue
            expiry=pd.to_datetime(str(pair['endDate']),format='%Y%m%d').date().isoformat()
            size=int(pair.get('contractSize') or 0)
            if size<=0:continue
            for suffix,kind in [('C','call'),('P','put')]:
                option_code=str(pair.get('insCode_'+suffix) or '')
                if not option_code.isdigit() or option_code=='0':continue
                contracts.append(dict(code=option_code,symbol=pair.get('lVal18AFC_'+suffix),kind=kind,
                    underlying=str(code),strike=float(pair['strikePrice']),expiry=expiry,multiplier=size,
                    bid=float(pair.get('pMeDem_'+suffix) or 0),ask=float(pair.get('pMeOf_'+suffix) or 0),
                    bid_volume=float(pair.get('qTitMeDem_'+suffix) or 0),ask_volume=float(pair.get('qTitMeOf_'+suffix) or 0),
                    last=float(pair.get('pDrCotVal_'+suffix) or 0),open_interest=float(pair.get('oP_'+suffix) or 0),
                    provider_quote_at=None))
        book={}
        try:
            levels=client._json(f'BestLimits/{code}').get('bestLimits',[])
            top=next((r for r in levels if r.get('number')==1),None)
            if top:
                book=dict(bid=float(top.get('pMeDem') or 0),ask=float(top.get('pMeOf') or 0),bid_volume=float(top.get('qTitMeDem') or 0),ask_volume=float(top.get('qTitMeOf') or 0),provider_quote_at=None)
        except Exception:
            pass # Chain remains useful; stock legs require explicit manual prices if order book is unavailable.
        snapshot=dict(underlying=str(code),symbol=info['lVal18AFC'].strip(),spot=float(raw['pDrCotVal']),
                      closing=float(raw['pClosing']),provider_quote_at=quoted.isoformat(),
                      retrieved_at=datetime.now(timezone.utc).isoformat(),contracts=contracts,
                      source='TSETMC public snapshot',option_quote_time_known=False,
                      execution_guaranteed=False)
        snapshot['underlying_book']=book
        with store.connect() as db:
            cursor=db.execute('INSERT INTO snapshots(kind,underlying,retrieved_at,payload) VALUES(?,?,?,?)',('options',str(code),snapshot['retrieved_at'],json.dumps(snapshot,ensure_ascii=False)))
            snapshot['id']=cursor.lastrowid
        return snapshot
    finally:client.session.close()

def latest_snapshot(store,code):
    with store.connect() as db:
        row=db.execute("SELECT id,payload FROM snapshots WHERE kind='options' AND underlying=? ORDER BY id DESC LIMIT 1",(str(code),)).fetchone()
    if not row:return None
    result=json.loads(row[1]);result['id']=row[0]
    return result
