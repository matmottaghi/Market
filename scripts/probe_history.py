import json
import requests
from urllib.parse import quote
from pathlib import Path

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36', 'Accept': 'application/json'})
report = []
for base in ['https://cdn.tsetmc.com/api']:
    path = '/Instrument/GetInstrumentSearch/' + quote('اهرم')
    entry = {'provider': base, 'endpoint': 'search'}
    try:
        response = session.get(base + path, timeout=(8, 15), allow_redirects=False)
        entry.update(http_status=response.status_code, content_type=response.headers.get('Content-Type'))
        if response.status_code == 200:
            payload = response.json()
            entry['keys'] = list(payload) if isinstance(payload, dict) else []
            rows = payload.get('instrumentSearch', [])
            entry['row_count'] = len(rows)
            entry['sample_fields'] = list(rows[0]) if rows else []
            exact = [r for r in rows if str(r.get('lVal18AFC','')).replace('ي','ی').replace('ك','ک').strip() == 'اهرم']
            entry['exact_matches'] = [{k:r.get(k) for k in ('insCode','lVal18AFC','lVal30','flow')} for r in exact]
    except Exception as exc:
        entry['error_type'] = type(exc).__name__
    report.append(entry)
print(json.dumps(report, ensure_ascii=True))
Path('data/processed').mkdir(parents=True, exist_ok=True)
Path('data/processed/connectivity_probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
