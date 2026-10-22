import json
from pathlib import Path
import requests
r=json.loads(Path('data/processed/connectivity_probe.json').read_text('utf-8'))[0]
identifier=str(r['exact_matches'][0]['insCode'])
s=requests.Session()
s.headers.update({'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36','Accept':'application/json'})
report=[]
for path,key in [(f'Instrument/GetInstrumentInfo/{identifier}','instrumentInfo'),(f'ClosingPrice/GetClosingPriceDailyList/{identifier}/0','closingPriceDaily'),(f'ClosingPrice/GetPriceAdjustList/{identifier}','priceAdjust'),(f'ClientType/GetClientTypeHistory/{identifier}','clientType')]:
    entry={'endpoint':path.split('/')[1]}
    try:
        res=s.get('https://cdn.tsetmc.com/api/'+path,timeout=(8,20),allow_redirects=False)
        entry['http_status']=res.status_code
        if res.status_code==200:
            payload=res.json()
            entry['keys']=list(payload)
            value=payload.get(key)
            entry['rows']=len(value) if isinstance(value,list) else 1
            entry['sample']=value[:1] if isinstance(value,list) else value
    except Exception as exc:
        entry['error_type']=type(exc).__name__
    report.append(entry)
Path('data/processed/history_schema_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True)[:10000])
