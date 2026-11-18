# توضیح فایل: تب snapshot بازار و زنجیرهٔ آپشن؛ ساخت سناریوی آزاد، ارزیابی سود و نمایش توزیع پیش‌بینی.
# options_page خواندن و محاسبه؛ refresh_option_data دریافت منبع با CSRF؛ symbol_search جستجوی دقیق/جزئی؛ register_symbol ثبت شناسهٔ تأییدشده.
from dataclasses import replace
import json
from urllib.parse import urlencode
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from django.shortcuts import render,redirect
from django.http import JsonResponse,HttpResponseBadRequest
from django.views.decorators.http import require_POST
from market_data.live_options import search_symbols,fetch_snapshot,latest_snapshot
from market_data.provider import TsetmcClient,_fa_normalize
from .registry import settings_store,instruments,list_runs
from .pipeline import RUNS
from .options_math import STRATEGIES,template_legs,evaluate_strategy
from .views import plot

def options_page(request):
    settings,store=settings_store();known=instruments()
    data=request.POST if request.method=='POST' else request.GET
    symbol=data.get('symbol','اهرم' if 'اهرم' in known else next(iter(known),''));code=known.get(symbol)
    snapshot=latest_snapshot(store,code) if code and hasattr(store,'connect') else None
    error='';result=None;chart='';distribution=None;forecast=None
    template=data.get('strategy','long_call')
    if template not in STRATEGIES:template='custom'
    contracts=snapshot['contracts'] if snapshot else []
    expiries=sorted({c['expiry'] for c in contracts})
    expiry=data.get('expiry',expiries[0] if expiries else '')
    runs=[r for r in list_runs() if (RUNS/r['run_id']/'predictions.csv').exists()]
    run=next((r for r in runs if r['run_id']==data.get('run')),runs[0] if runs else None)
    forecasts=[];forecast_metrics=None
    if run and code:
        frame=pd.read_csv(RUNS/run['run_id']/'predictions.csv',dtype={'insCode':str})
        forecasts=frame[frame.insCode==code].to_dict('records')
        if hasattr(store,'connect'):
            with store.connect() as db:
                rows=db.execute('SELECT payload FROM forecasts WHERE run=? AND code=? ORDER BY step',(run['run_id'],code)).fetchall()
            if rows:forecasts=[json.loads(r[0]) for r in rows]
    try:step=int(data.get('step',forecasts[-1]['horizon'] if forecasts else 1))
    except ValueError:step=1
    if forecasts and step not in {p['horizon'] for p in forecasts}:step=forecasts[-1]['horizon']
    forecast=next((p for p in forecasts if p['horizon']==step),None)
    if run and code and (RUNS/run['run_id']/'metrics.csv').exists():
        scored=pd.read_csv(RUNS/run['run_id']/'metrics.csv',dtype={'insCode':str})
        scored=scored[(scored.insCode==code)&(scored.horizon==step)]
        if not scored.empty:
            forecast_metrics=scored.iloc[0].to_dict()
            forecast_metrics.update(mae_percent=100*forecast_metrics['mae'],accuracy_percent=100*forecast_metrics['direction_accuracy'],coverage_percent=100*forecast_metrics['coverage90'])
    rawlegs=data.get('legs','')
    fees=data.get('fees','0');capital=data.get('capital','')
    if request.method=='POST' and data.get('build')!='select_model':
        try:
            if not snapshot:raise ValueError('ابتدا snapshot و زنجیره را دریافت کنید.')
            legs=template_legs(template,contracts,snapshot['spot'],expiry) if data.get('build')=='template' else json.loads(rawlegs) if rawlegs.strip() else template_legs(template,contracts,snapshot['spot'],expiry)
            if data.get('inverse'):
                for leg in legs:leg['quantity']=-float(leg['quantity'])
            rawlegs=json.dumps(legs,ensure_ascii=False,indent=2)
            if data.get('use_forecast'):
                if not forecast:raise ValueError('برای این دارایی پایه و افق، مدل پیش‌بینی موجود نیست.')
                if not data.get('confirm_alignment'):raise ValueError('تطابق افق پیش‌بینی با تاریخ ارزیابی و زمان مبنای مدل باید بررسی و تأیید شود.')
                asof=pd.Timestamp(forecast['as_of']).date();end=pd.Timestamp(expiry).date()
                if end<=asof:raise ValueError('تاریخ ارزیابی باید بعد از زمان مبنای مدل باشد.')
                if (end-asof).days>45:raise ValueError('سررسید خارج از محدوده تقریبی ۲۱ جلسه است؛ توزیع این مدل قابل استفاده نیست.')
                holidays_path=settings.root/'holidays.json'
                holidays=set(json.loads(holidays_path.read_text(encoding='utf8'))) if holidays_path.exists() else set()
                sessions=[d for d in pd.date_range(pd.Timestamp(asof)+pd.Timedelta(days=1),pd.Timestamp(end)) if d.weekday() not in (3,4) and str(d.date()) not in holidays]
                if forecast.get('target_schema')!='hourly_path_v1':raise ValueError('برای امید سود از مدل ساعتی جدید استفاده کنید.')
                if len(sessions)!=int(forecast['target_day']):raise ValueError('تعداد جلسه تا تاریخ ارزیابی با افق انتخابی منطبق نیست؛ تقویم تعطیلات و افق را اصلاح کنید.')
                hours=[p['target_hour'] for p in forecasts if p.get('target_day')==forecast['target_day']]
                if forecast['target_hour']!=max(hours):raise ValueError('برای سود سررسید، آخرین ساعت جلسهٔ متناظر را انتخاب کنید.')
                distribution=(forecast['current_price'],forecast['mu_log'],forecast['sigma_log'])
            result=evaluate_strategy(legs,contracts,snapshot['spot'],expiry,float(fees),distribution,float(capital) if capital else None,float(data.get('rate','0')),float(data.get('dividend','0')),underlying_book=snapshot.get('underlying_book'))
            fig=go.Figure(go.Scatter(x=result['x'],y=result['pnl'],name='سود/زیان در تاریخ ارزیابی'))
            fig.add_hline(y=0,line_dash='dot')
            if forecast and distribution:
                predicted=forecast.get('predicted_price',forecast['current_price']*np.exp(forecast['mu_log']+.5*forecast['sigma_log']**2))
                fig.add_vline(x=predicted,line_color='#67e6ba',annotation_text=f"قیمت مورد انتظار مدل — جلسه {forecast['target_day']}، ساعت {int(forecast['target_hour']):02d}")
            chart=plot(fig,'عایدی خالص استراتژی — همه قیمت‌ها در واحد یکسان منبع')
        except (ValueError,KeyError,TypeError,OverflowError) as exc:error=str(exc)
    return render(request,'forecast/options.html',dict(symbols=known,symbol=symbol,snapshot=snapshot,contracts=contracts,
        strategies=STRATEGIES,strategy=template,expiries=expiries,expiry=expiry,legs=rawlegs,result=result,chart=chart,error=error,
        forecasts=forecasts,forecast=forecast,forecast_metrics=forecast_metrics,step=step,runs=runs,run=run,fees=fees,capital=capital,
        rate=data.get('rate','0'),dividend=data.get('dividend','0'),use_forecast=bool(data.get('use_forecast')),confirm_alignment=bool(data.get('confirm_alignment'))))

@require_POST
def refresh_option_data(request):
    settings,store=settings_store();known=instruments();symbol=request.POST.get('symbol')
    if symbol not in known:return HttpResponseBadRequest('نماد ثبت نشده است')
    try:
        fetch_snapshot(replace(settings,timeout=12,retries=1),store,known[symbol])
        return redirect('/options/?symbol='+symbol)
    except Exception as exc:
        return render(request,'forecast/provider_error.html',dict(error=str(exc),symbol=symbol),status=502)

def symbol_search(request):
    query=request.GET.get('q','').strip()
    if not 2<=len(query)<=60:return JsonResponse({'results':[]})
    known=instruments()
    results=[dict(code=c,symbol=s,name='ثبت‌شده در دیتابیس',registered=True) for s,c in known.items() if _fa_normalize(query) in _fa_normalize(s)]
    if request.GET.get('remote')=='1':
        try:
            settings,_=settings_store()
            remote=search_symbols(replace(settings,timeout=6,retries=0),query)
            seen={r['code'] for r in results};results += [r for r in remote if r['code'] not in seen]
        except Exception as exc:return JsonResponse(dict(results=results,error=str(exc)),status=200)
    return JsonResponse(dict(results=results))

@require_POST
def register_symbol(request):
    code=request.POST.get('code','')
    if not code.isdigit() or code=='0':return HttpResponseBadRequest('شناسه نامعتبر')
    settings,store=settings_store();client=TsetmcClient(replace(settings,timeout=8,retries=0))
    try:
        info=client._json(f'Instrument/GetInstrumentInfo/{code}').get('instrumentInfo',{})
        if str(info.get('insCode'))!=code:raise ValueError('Identity mismatch')
        symbol=info['lVal18AFC'].strip();known=instruments()
        if symbol in known and known[symbol]!=code:raise ValueError('این نماد با شناسهٔ دیگری ثبت شده؛ ابتدا هویت نماد را بررسی کنید.')
        store.update_state(code,symbol=symbol,name=info.get('lVal30'),sector=info.get('sector',{}),instrument_type=info.get('faraDesc'))
        target='/data/' if request.POST.get('next')=='/data/' else '/training/'
        return redirect(target+'?'+urlencode(dict(symbol=symbol)))
    except Exception as exc:return HttpResponseBadRequest(str(exc))
    finally:client.session.close()
