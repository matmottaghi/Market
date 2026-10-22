"""Safe live connectivity probe: outputs status/schema, never credentials or request URLs."""
import json
import requests
from pathlib import Path
from market_dss.service import MarketService
from market_dss.historical import HEADERS

service = MarketService()
key = service.live_key()
report = {"key_configured": bool(key), "requests": []}
session = requests.Session()
session.headers.update(HEADERS)
paths = []
if key:
    for endpoint, params in [('AllSymbols.php', {'type': 1}), ('Symbol.php', {'l18': 'اهرم'}),
                             ('Option.php', {}), ('Nav.php', {'l18': 'اهرم'})]:
        paths.append(('BrsApi', 'https://Api.BrsApi.ir/Tsetmc/' + endpoint, {'key': key, **params}))
identifier = '17914401175772326'
for endpoint in [f'ClosingPrice/GetClosingPriceInfo/{identifier}', f'BestLimits/{identifier}',
                 f'ClientType/GetClientType/{identifier}/1/0']:
    paths.append(('TSETMC', 'https://cdn.tsetmc.com/api/' + endpoint, None))
for provider, url, params in paths:
    item = {'provider': provider, 'endpoint': url.split('/')[-1] if provider == 'BrsApi' else url.split('/')[-2]}
    try:
        response = session.get(url, params=params, timeout=(8,20), allow_redirects=False)
        item.update(http_status=response.status_code, content_type=response.headers.get('Content-Type'), seconds=response.elapsed.total_seconds())
        if response.status_code == 200 and 'json' in response.headers.get('Content-Type',''):
            payload=response.json()
            item['payload_type']=type(payload).__name__
            if isinstance(payload,dict):
                item['envelope_fields']=list(payload)
                for field, value in payload.items():
                    if isinstance(value,dict):
                        item['record_fields']=list(value)
                        item['date_fields']={k:v for k,v in value.items() if k in ['dEven','date','date_update','hEven']}
                    elif isinstance(value,list):
                        item['record_count']=len(value)
                        item['record_fields']=list(value[0]) if value and isinstance(value[0],dict) else []
            elif isinstance(payload,list):
                item['record_count']=len(payload)
                item['record_fields']=list(payload[0]) if payload and isinstance(payload[0],dict) else []
    except Exception as exc:
        item['error_type']=type(exc).__name__
    report['requests'].append(item)
Path('data/processed/live_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True))
