# توضیح فایل: ثبت شناسه و نوع صندوق/صنعت از منبع؛ دادهٔ تاریخی را جعل یا جایگزین نمی‌کند.
import sys,json
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from forecast.registry import settings_store
from market_data import TsetmcClient
settings,store=settings_store();settings=replace(settings,timeout=8,retries=0,delay=.3)
client=TsetmcClient(settings)
result=[]
for name in ['فولاد','فملی','فارس','شپنا','وبملت','خودرو','شستا','یاقوت','استیل','پتروآگاه','فارما','سیمانو']:
 try:
  code=client.resolve_symbol_code(name);info=client._json(f'Instrument/GetInstrumentInfo/{code}')['instrumentInfo']
  if str(info['insCode'])!=code:raise ValueError('identity mismatch')
  if not store.read_state(code):store.update_state(code,symbol=info['lVal18AFC'].strip(),name=info['lVal30'],instrument_type=info.get('faraDesc'),sector=info.get('sector'))
  result.append({'symbol':name,'code':code,'name':info['lVal30'],'type':info.get('faraDesc')})
 except Exception as exc:result.append({'symbol':name,'error':str(exc)})
client.session.close()
print(json.dumps(result,ensure_ascii=True,indent=2))
