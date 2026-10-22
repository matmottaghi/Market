import json
from market_dss.historical import TsetmcProvider
p=TsetmcProvider('data')
rows=[]
for symbol in ['اهرم','موج','توان']:
    identity=p.resolve(symbol)
    quote=p.request(f"ClosingPrice/GetClosingPriceInfo/{identity['instrument_id']}",'closingPriceInfo')
    instrument=quote.get('instrument')
    rows.append({'symbol':symbol,'resolved_id':identity['instrument_id'],'quote_id':quote.get('insCode'),
                 'nested_instrument': {k:v for k,v in (instrument or {}).items() if k in ['insCode','lVal18AFC','lVal30','cIsin']}})
print(json.dumps(rows,ensure_ascii=True))
