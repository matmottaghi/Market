# توضیح فایل: دو پنل آموزش و بازار، جریان SSE، انتخاب نماد/اجرا، ذخیره مدل و دریافت داده.
# توابع plot قالب نمودار و events مولد رویداد SSE هستند.
# توابع: dashboard ساخت فیلترها و نمودارها؛ stream ارسال پیشرفت زنده؛ launch شروع worker؛ retain ذخیره/پاکسازی مدل.
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import lognorm
from django.http import StreamingHttpResponse, HttpResponseBadRequest, HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST
from filelock import FileLock, Timeout
from .pipeline import BASE, RUNS
from .registry import list_runs, prune, instruments, settings_store, jalali_month
from .neural_data import price_distribution
from market_data.csv_store import _write_bytes_atomic

def plot(fig, title):
    fig.update_layout(title=title, template='plotly_dark', height=350, margin=dict(t=55,b=45,l=50,r=20))
    return fig.to_html(full_html=False, include_plotlyjs=False)

def dashboard(request):
    runs = [r for r in list_runs(RUNS) if r.get('model_kind')=='lstm' and r.get('job_kind')!='data']
    current = next((r for r in runs if r['run_id']==request.GET.get('run')), runs[0] if runs else None)
    known = instruments()
    symbol = request.GET.get('symbol', next(iter(known), 'توان'))
    horizon = int(request.GET.get('horizon','1')) if request.GET.get('horizon','1') in ('1','5','21') else 1
    try:
        confidence = float(request.GET.get('confidence','90'))/100
        if not np.isfinite(confidence) or not .5<=confidence<=.99: raise ValueError()
        confidence_error=''
    except (ValueError, TypeError):
        confidence,confidence_error=.9,'بازه باید بین ۵۰ و ۹۹ درصد باشد.'
    predictions,metrics,learning_charts,charts,forecast_charts,cards = [],[],[],[],[],[]
    decision=[]; decision_source=request.GET.get('decision_set','calibration')
    if decision_source not in ('test','calibration'):decision_source='calibration'
    if current:
        folder=RUNS/current['run_id']
        pp=folder/'predictions.csv'; mm=folder/'metrics.csv'
        if pp.exists():
            frame=pd.read_csv(pp,dtype={'insCode':str});predictions=frame[frame.symbol==symbol].to_dict('records')
        for row in predictions:
            row.update(price_distribution(row['current_price'],row['mu_log'],row['sigma_log'],confidence))
            row['return_percent']=row['predicted_return']*100
            row['probability_up_percent']=row['probability_up']*100
            row['probability_down_percent']=row['probability_down']*100
        if mm.exists():
            frame=pd.read_csv(mm);metrics=frame[(frame.symbol==symbol)&(frame.horizon==horizon)].to_dict('records')
            if metrics:
                m=metrics[0]
                cards=[('دقت جهت آزمون',f"{100*m['direction_accuracy']:.1f}%"),('MAE بازده',f"{100*m['mae']:.2f}%"),('RMSE',f"{100*m['rmse']:.2f}%"),('Brier',f"{m['brier']:.3f}"),('پوشش بازه ۹۰٪',f"{100*m['coverage90']:.1f}%"),('تعداد آزمون',str(m['test_rows']))]
                baseline=folder/'baseline_metrics.csv'
                if baseline.exists():
                    old=pd.read_csv(baseline)
                    selected=old[(old.symbol==symbol)&(old.horizon==horizon)]
                    if not selected.empty:
                        gain=100*(selected.iloc[0].mae-m['mae'])
                        cards.append(('بهبود MAE نسبت اجرای قبل',f'{gain:+.2f} واحد درصد'))
        code=known.get(symbol)
        if code:
            path=folder/f'{code}_learning.csv'
            if path.exists():
                df=pd.read_csv(path)
                for cols,title in [(['loss','val_loss'],'زیان ترکیبی آموزش / اعتبارسنجی'),(['standardized_mae','val_standardized_mae'],'خطای استانداردشده')]:
                    fig=go.Figure()
                    for col in cols:
                        if col in df:fig.add_scatter(x=df.epoch,y=df[col],name=col)
                    config=folder/f'{code}_preprocessing.json'
                    if config.exists():
                        best=json.loads(config.read_text(encoding='utf8')).get('best_epoch')
                        if best:fig.add_vline(x=best,line_dash='dot',annotation_text='بهترین epoch')
                    learning_charts.append(plot(fig,f'{symbol} — {title}'))
            path=folder/f'{code}_hourly_{horizon}_{decision_source}.csv'
            if path.exists():
                df=pd.read_csv(path);decision=[dict(p=float(r.probability_up),actual=int(r.actual>0)) for r in df.itertuples()]
            path=folder/f'{code}_hourly_{horizon}_test.csv'
            if path.exists():
                df=pd.read_csv(path);fig=go.Figure()
                fig.add_scatter(x=df.timestamp,y=df.actual*100,name='بازده واقعی')
                fig.add_scatter(x=df.timestamp,y=df.predicted*100,name='بازده پیش‌بینی')
                charts.append(plot(fig,f'{symbol} — بازده آزمون افق {horizon} روز'))
    chosen=next((p for p in predictions if p['horizon']==horizon),None)
    if predictions:
        fig=go.Figure()
        rows=sorted(predictions,key=lambda p:p['horizon'])
        for key,label in [('upper_price','حد بالا'),('lower_price','حد پایین'),('predicted_price','قیمت مورد انتظار')]:
            fig.add_scatter(x=[p['horizon'] for p in rows],y=[p[key] for p in rows],name=label,fill='tonexty' if key=='lower_price' else None)
        forecast_charts.append(plot(fig,f'{symbol} — بازه قیمت {confidence*100:g}٪'))
    if chosen:
        x=np.linspace(lognorm.ppf(.005,chosen['sigma_log'],scale=chosen['current_price']*np.exp(chosen['mu_log'])),lognorm.ppf(.995,chosen['sigma_log'],scale=chosen['current_price']*np.exp(chosen['mu_log'])),250)
        fig=go.Figure(go.Scatter(x=x,y=lognorm.pdf(x,chosen['sigma_log'],scale=chosen['current_price']*np.exp(chosen['mu_log'])),fill='tozeroy',name='چگالی قیمت'))
        fig.add_vline(x=chosen['lower_price'],line_dash='dot');fig.add_vline(x=chosen['upper_price'],line_dash='dot')
        forecast_charts.append(plot(fig,f'توزیع قیمت در پایان {horizon} روز معاملاتی'))
    seasonality=[]; history=[]
    _,store=settings_store()
    if symbol in known:
        daily=store.load_daily(known[symbol])
        if not daily.empty:
            daily=daily[(daily.qTotTran5J>0)&(daily.zTotTran>0)&(daily.pClosing>0)].sort_values('Date').copy()
            daily['ret']=daily.pClosing.pct_change(fill_method=None)*100
            history.append(plot(go.Figure(go.Scatter(x=daily.Date,y=daily.pClosing,name='قیمت پایانی')),f'تاریخچه {symbol} — قیمت تعدیل‌نشده'))
            daily['weekday']=pd.to_datetime(daily.Date).dt.weekday
            fig=go.Figure()
            order=[5,6,0,1,2,3,4];names=['شنبه','یکشنبه','دوشنبه','سه‌شنبه','چهارشنبه','پنجشنبه','جمعه']
            for up,label in [(True,'میانگین روزهای مثبت'),(False,'میانگین روزهای منفی')]:
                group=daily[daily.ret>0] if up else daily[daily.ret<0]
                means=group.groupby('weekday').ret.mean();counts=group.groupby('weekday').ret.count()
                fig.add_bar(x=names,y=[means.get(i,None) for i in order],name=label,customdata=[counts.get(i,0) for i in order],hovertemplate='%{x}: %{y:.2f}% — تعداد %{customdata}')
            seasonality.append(plot(fig,'میانگین شرطی بازده مثبت/منفی به تفکیک روز هفته'))
            daily['month']=pd.to_datetime(daily.Date).map(jalali_month)
            group=daily.groupby('month').ret.agg(['mean','count']).reindex(range(1,13))
            seasonality.append(plot(go.Figure(go.Bar(x=['فروردین','اردیبهشت','خرداد','تیر','مرداد','شهریور','مهر','آبان','آذر','دی','بهمن','اسفند'],y=group['mean'],customdata=group['count'],hovertemplate='%{x}: %{y:.2f}% — تعداد %{customdata}')),'میانگین بازده روزانه در ماه‌های شمسی (همه سال‌ها)'))
    return render(request,'forecast/dashboard.html',dict(page='market' if request.path.startswith('/market') else 'train',runs=runs,current=current,symbols=known,symbol=symbol,horizon=horizon,confidence_percent=confidence*100,confidence_error=confidence_error,predictions=predictions,chosen=chosen,metrics=metrics,cards=cards,learning_charts=learning_charts,charts=charts,forecast_charts=forecast_charts,seasonality=seasonality,history=history,decision=decision,decision_source=decision_source))

def stream(request):
    def events():
        for _ in range(1800):
            rows=list_runs(RUNS)
            state=rows[0] if rows else {'state':'idle'}
            curves={}
            if rows:
                for p in (RUNS/rows[0]['run_id']).glob('*_learning.csv'):
                    curves[p.stem.replace('_learning','')]=pd.read_csv(p).fillna(0).to_dict('records')
            yield 'data: '+json.dumps(dict(status=state,curves=curves),ensure_ascii=False)+'\n\n'
            time.sleep(2)
    response=StreamingHttpResponse(events(),content_type='text/event-stream')
    response['Cache-Control']='no-cache';response['X-Accel-Buffering']='no'
    return response

@require_POST
def launch(request):
    try:
        epochs=int(request.POST.get('epochs','200'))
        if not 1<=epochs<=200:raise ValueError()
        symbols=request.POST.getlist('symbols') or list(instruments())
        extra=request.POST.get('new_symbols','').replace('،',',').split(',')
        symbols=list(dict.fromkeys(symbols+[s.strip() for s in extra if s.strip()]))
        if not symbols or any(len(s)>60 or '/' in s or '\\' in s for s in symbols):raise ValueError()
    except ValueError:return HttpResponseBadRequest('ورودی نامعتبر')
    RUNS.mkdir(parents=True,exist_ok=True)
    try:
        with FileLock(str(RUNS/'.pipeline.lock'),timeout=0):
            args=[sys.executable,str(BASE/'manage.py'),'train_forecast','--epochs',str(epochs),'--symbols',*symbols]
            if request.POST.get('action')!='refresh' and request.POST.get('offline') and not any(s not in instruments() for s in symbols):args.append('--offline')
            if request.POST.get('action')=='refresh':args.append('--refresh-only')
            with (RUNS/'worker.log').open('a',encoding='utf8') as log:
                subprocess.Popen(args,cwd=BASE,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    except Timeout:return HttpResponse('آموزش یا دریافت دیگری در حال اجراست.',status=409)
    return redirect('dashboard')

@require_POST
def retain(request):
    rows=list_runs(RUNS);row=next((r for r in rows if r['run_id']==request.POST.get('run')),None)
    if not row:return HttpResponseBadRequest('اجرا یافت نشد')
    try:
        with FileLock(str(RUNS/'.pipeline.lock'),timeout=0):
            if request.POST.get('action')=='save':
                if row['state'] not in ('complete','partial') or not list((RUNS/row['run_id']).glob('*.keras')):return HttpResponseBadRequest('مدل آماده نیست')
                name=request.POST.get('name','').strip()[:100] or row['label']
                marker=RUNS/row['run_id']/'saved.json'
                old=json.loads(marker.read_text(encoding='utf8')) if marker.exists() else {}
                if request.POST.get('scope')=='symbol':
                    code=instruments().get(request.POST.get('symbol'))
                    if not code or not (RUNS/row['run_id']/f'{code}_lstm.keras').exists():return HttpResponseBadRequest('مدل این نماد موجود نیست')
                    codes=sorted(set(old.get('codes',[])+[code]))
                else:
                    codes=[p.name.split('_')[0] for p in (RUNS/row['run_id']).glob('*_lstm.keras')]
                _write_bytes_atomic(marker,json.dumps({'name':name,'codes':codes,'saved_at':datetime.now(timezone.utc).isoformat()},ensure_ascii=False))
                status=json.loads((RUNS/row['run_id']/'status.json').read_text(encoding='utf8'));status['label']=name
                _write_bytes_atomic(RUNS/row['run_id']/'status.json',json.dumps(status,ensure_ascii=False))
            elif request.POST.get('action')=='discard':
                (RUNS/row['run_id']/'saved.json').unlink(missing_ok=True)
            latest_model=next((r['run_id'] for r in rows if r.get('job_kind')!='data'),rows[0]['run_id'])
            prune(latest_model,RUNS)
    except Timeout:return HttpResponse('تا پایان آموزش منتظر بمانید.',status=409)
    return redirect('dashboard')
