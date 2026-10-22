import json
from market_dss.live import TsetmcLiveProvider, diagnose_market_api
p=TsetmcLiveProvider('data',['اهرم','موج','توان'])
batch=p.fetch()
print(json.dumps({'counts':{k:len(v) for k,v in batch.tables.items()},'issues':batch.issues},ensure_ascii=True))
rows=diagnose_market_api('TSETMC',symbols=['اهرم','موج','توان'])
print(json.dumps(rows,ensure_ascii=True))
