# توضیح فایل: دو پنل آموزش و بازار، جریان SSE، انتخاب نماد/اجرا، ذخیره مدل و دریافت داده.
# توابع plot قالب نمودار و events مولد رویداد SSE هستند.
# توابع: dashboard نمودار؛ stream پیشرفت؛ launch آموزش تک‌هدف آفلاین یا دریافت مستقل از پنل داده؛ retain نگهداری؛ download_model بستهٔ وزن و scaler؛ refresh_prediction/prediction_status پیش‌بینی تازه و وضعیت.
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlencode
import numpy as np
import pandas as pd
from .table_io import read_table,write_table,table_path,learning_tables
import plotly.graph_objects as go
from django.http import StreamingHttpResponse, HttpResponseBadRequest, HttpResponse, FileResponse, JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST
from filelock import FileLock, Timeout
from .pipeline import BASE, RUNS
from .registry import list_runs, prune, instruments, settings_store, jalali_month
from .neural_data import price_distribution
from .availability import catalog,choose,load_rows,active_run,reusable
from .training_report import training_report
from .model_management import model_records,delete_model_run,model_mutation_lock
from django.views.decorators.csrf import csrf_protect
from market_data.csv_store import _write_bytes_atomic

def plot(fig, title):
    fig.update_layout(title=title, template='plotly_dark', height=350, margin=dict(t=55,b=45,l=50,r=20))
    return fig.to_html(full_html=False, include_plotlyjs=False)

def dashboard(request):
    runs = [r for r in list_runs(RUNS) if r.get('model_kind')=='lstm' and r.get('job_kind')!='data']
    current = next((r for r in runs if r['run_id']==request.GET.get('run')), runs[0] if runs else None)
    known = instruments()
    symbol = request.GET.get('symbol', next(iter(known), 'توان'))
    if not request.GET.get('symbol') and current:
        path=table_path(RUNS/current['run_id']/'predictions.json')
        if path.exists():
            available=read_table(path,usecols=['symbol'])
            if not available.empty:symbol=available.symbol.iloc[0]
        else:symbol=current.get('training_progress',{}).get('symbol','توان' if 'توان' in known else symbol)
    _,availability_store=settings_store()
    runs=catalog(runs,RUNS,availability_store)
    current=next((r for r in runs if r['run_id']==request.GET.get('run')),current)
    notice=''
    if request.path.startswith('/market'):
        requested=request.GET.get('run')
        current=choose(runs,known.get(symbol),requested,active_run(availability_store,known.get(symbol)))
        if current and requested and current['run_id']!=requested:
            notice='اجرای انتخابی برای این نماد خروجی نداشت؛ آخرین اجرای دارای پیش‌بینی این نماد نمایش داده شد.'
    else:
        ready=choose(runs,known.get(symbol),request.GET.get('run'))
        preparing=current and current.get('state')=='running' and any(s.get('symbol')==symbol for s in current.get('stages',[]))
        if ready and not preparing and (not current or known.get(symbol) not in current.get('ready_codes',set())):
            current=ready;notice='اجرای دارای خروجی این نماد برای ارزیابی انتخاب شد.'
    management=model_records(RUNS,availability_store,runs,known)
    labels={r['run_id']:r['display_label'] for r in management}
    for r in runs:r['display_label']=labels.get(r['run_id'],r.get('label',r['run_id']))
    can_predict=reusable(RUNS,current,known.get(symbol))
    if current and not can_predict and known.get(symbol) in current.get('ready_codes',set()) and not (RUNS/current['run_id']/f"{known.get(symbol)}_lstm.keras").exists():
        notice+=' خروجی تاریخی ذخیره‌شده نمایش داده می‌شود؛ وزن این مدل نگهداری نشده و پیش‌بینی تازه با آن ممکن نیست.'
    empty_reason='' if current and known.get(symbol) in current.get('ready_codes',set()) else 'برای این نماد در اجرای انتخابی هنوز پیش‌بینی آماده نیست. داشتن داده یا حضور در زمینهٔ بازار، به معنی آموزش مدل مستقل آن نماد نیست.'
    symbol_options=[dict(name=n,ready=any(c in r.get('ready_codes',set()) for r in runs)) for n,c in known.items()]
    try:horizon=max(1,min(504,int(request.GET.get('horizon','1'))))
    except ValueError:horizon=1
    try:
        confidence = float(request.GET.get('confidence','90'))/100
        if not np.isfinite(confidence) or not .5<=confidence<=.99: raise ValueError()
        confidence_error=''
    except (ValueError, TypeError):
        confidence,confidence_error=.9,'بازه باید بین ۵۰ و ۹۹ درصد باشد.'
    try:days=max(1,min(21,int(request.GET.get('days','21'))))
    except ValueError:days=21
    prediction_steps=[];day_forecasts=[]
    predictions,metrics,learning_charts,charts,forecast_charts,cards = [],[],[],[],[],[]
    decision=[]; decision_source=request.GET.get('decision_set','calibration')
    if decision_source not in ('test','calibration'):decision_source='calibration'
    if current:
        folder=RUNS/current['run_id']
        pp=table_path(folder/'predictions.json'); mm=table_path(folder/'metrics.json')
        if request.path.startswith('/market'):
            predictions=load_rows(current,known.get(symbol),RUNS,availability_store)
        elif pp.exists():
            frame=read_table(pp,dtype={'insCode':str});predictions=frame[frame.symbol==symbol].to_dict('records')
        if request.path.startswith('/market') and any(p.get('target_schema')=='hourly_returns_v2' for p in predictions):
            predictions=[]
            empty_reason='این مدل با برنامهٔ ساعت‌های تاریخی نادرست آموزش دیده و افق کامل ۲۱ جلسه ندارد؛ برای پیش‌بینی معتبر باید مدل نسخهٔ جدید را دوباره آموزش دهید.'
            can_predict=False
        if predictions and predictions[0].get('target_schema') in ('hourly_path_v1','hourly_returns_v2','hourly_returns_v3'):
            predictions=[p for p in predictions if p['target_day']<= (21 if request.path.startswith('/market') else days)]
            prediction_steps=[dict(value=p['horizon'],label=f"جلسه {p['target_day']} — ساعت {int(p['target_hour']):02d}") for p in predictions]
            if horizon not in [p['horizon'] for p in predictions]:horizon=predictions[-1]['horizon'] if predictions else 1
        for row in predictions:
            if row.get('scenario_prices'):
                from .scenarios import scenario_summary
                row.update(scenario_summary(row['scenario_prices'],row['current_price'],confidence))
            else:row.update(price_distribution(row['current_price'],row['mu_log'],row['sigma_log'],confidence))
            row['return_percent']=row['predicted_return']*100
            row['probability_up_percent']=row['probability_up']*100
            row['probability_down_percent']=row['probability_down']*100
        metric_rows=load_rows(current,known.get(symbol),RUNS,availability_store,kind='forecast_metrics')
        if metric_rows:
            metrics=[m for m in metric_rows if m['horizon']==horizon]
            if metrics:
                m=metrics[0]
                cards=[('دقت جهت آزمون',f"{100*m['direction_accuracy']:.1f}%"),('MAE بازده',f"{100*m['mae']:.2f}%"),('RMSE',f"{100*m['rmse']:.2f}%"),('Brier',f"{m['brier']:.3f}"),('پوشش بازه ۹۰٪',f"{100*m['coverage90']:.1f}%"),('تعداد آزمون',str(m['test_rows']))]
                baseline=table_path(folder/'baseline_metrics.json')
                if baseline.exists():
                    old=read_table(baseline)
                    selected=old[(old.symbol==symbol)&(old.horizon==horizon)]
                    if not selected.empty:
                        gain=100*(selected.iloc[0].mae-m['mae'])
                        cards.append(('بهبود MAE نسبت اجرای قبل',f'{gain:+.2f} واحد درصد'))
        if predictions and predictions[0].get('target_schema') in ('hourly_path_v1','hourly_returns_v2','hourly_returns_v3'):
            for day in range(1,days+1):
                rows=[p for p in predictions if p['target_day']==day]
                if rows:day_forecasts.append(max(rows,key=lambda p:p['target_hour']))
        code=known.get(symbol)
        if code:
            path=table_path(folder/f'{code}_learning.json')
            df=read_table(path) if path.exists() else pd.DataFrame()
            if df.empty and hasattr(availability_store,'connect'):
                with availability_store.connect() as db:
                    rows=db.execute('SELECT payload FROM epoch_metrics WHERE run=? AND code=? ORDER BY epoch',(current['run_id'],code)).fetchall()
                df=pd.DataFrame([json.loads(r[0]) for r in rows])
            if not df.empty:
                for cols,title in [(['loss','val_loss'],'زیان ترکیبی آموزش / اعتبارسنجی'),(['standardized_mae','val_standardized_mae','return_mae','val_return_mae'],'خطای استانداردشده')]:
                    fig=go.Figure()
                    for col in cols:
                        if col in df:fig.add_scatter(x=df.epoch,y=df[col],name=col)
                    config=folder/f'{code}_preprocessing.json'
                    best=None
                    if config.exists():
                        best=json.loads(config.read_text(encoding='utf8')).get('best_epoch')
                    else:
                        best=next((s.get('best_epoch') for s in current.get('stages',[]) if s.get('stage')=='trained' and s.get('symbol')==symbol),None)
                    if best:fig.add_vline(x=best,line_dash='dot',annotation_text='بهترین epoch')
                    learning_charts.append(plot(fig,f'{symbol} — {title}'))
            path=table_path(folder/f'{code}_hourly_{horizon}_{decision_source}.json')
            if path.exists():
                df=read_table(path);decision=[dict(p=float(r.probability_up),actual=int(r.actual>0),t=str(r.timestamp),price=float(r.current_price) if hasattr(r,'current_price') else None) for r in df.itertuples()]
            path=table_path(folder/f'{code}_hourly_{horizon}_test.json')
            if path.exists():
                df=read_table(path);fig=go.Figure()
                fig.add_scatter(x=df.timestamp,y=df.actual*100,name='بازده واقعی')
                fig.add_scatter(x=df.timestamp,y=df.predicted*100,name='بازده پیش‌بینی')
                charts.append(plot(fig,f'{symbol} — بازده تاریخی افق انتخابی'))
                price_fig=go.Figure(go.Scatter(x=df.timestamp,y=df.current_price,name='قیمت مبنای تصمیم')) if 'current_price' in df else go.Figure()
                if 'current_price' in df:
                    good=(df.probability_up>=.5)==(df.actual>0)
                    for correct,color,label in [(True,'#46dfaa','تشخیص درست'),(False,'#ff657c','تشخیص غلط')]:
                        part=df[good==correct];price_fig.add_scatter(x=part.timestamp,y=part.current_price,mode='markers',marker=dict(color=color,size=8),name=label)
                    charts.append(plot(price_fig,'محل تصمیم‌های خارج از آموزش روی قیمت'))
    chosen=next((p for p in predictions if p['horizon']==horizon),None)
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
    report=training_report(RUNS/current['run_id'],known.get(symbol),symbol,current) if current and not request.path.startswith('/market') else None
    market_forecast=dict(symbol=symbol,horizon=horizon,confidence=confidence,rows=sorted(predictions,key=lambda p:p['horizon']))
    inventory=store.inventory() if hasattr(store,'inventory') else []
    return render(request,'forecast/dashboard.html',dict(model_management=management,model_deleted=request.GET.get('model_deleted'),training_report=report,market_forecast=market_forecast,notice=notice,empty_reason=empty_reason,can_predict=can_predict,symbol_options=symbol_options,inventory=inventory,days=days,prediction_steps=prediction_steps,day_forecasts=day_forecasts,page='market' if request.path.startswith('/market') else 'train',runs=runs,current=current,symbols=known,symbol=symbol,horizon=horizon,confidence_percent=confidence*100,confidence_error=confidence_error,predictions=predictions,chosen=chosen,metrics=metrics,cards=cards,learning_charts=learning_charts,charts=charts,forecast_charts=forecast_charts,seasonality=seasonality,history=history,decision=decision,decision_source=decision_source))

def stream(request):
    def events():
        for _ in range(1800):
            rows=list_runs(RUNS)
            state=rows[0] if rows else {'state':'idle'}
            curves={}
            if rows:
                for p in learning_tables(RUNS/rows[0]['run_id']):
                    curves[p.stem.replace('_learning','')]=read_table(p).fillna(0).to_dict('records')
            yield 'data: '+json.dumps(dict(status=state,curves=curves),ensure_ascii=False)+'\n\n'
            time.sleep(2)
    response=StreamingHttpResponse(events(),content_type='text/event-stream')
    response['Cache-Control']='no-cache';response['X-Accel-Buffering']='no'
    return response

@require_POST
def launch(request):
    try:
        data_request=request.POST.get('next')=='/data/' and request.POST.get('action')=='refresh'
        epochs=int(request.POST.get('epochs','200'))
        if not 1<=epochs<=200:raise ValueError()
        if data_request:
            symbols=request.POST.getlist('symbols')
        else:
            if request.POST.get('action','train')!='train':
                return HttpResponseBadRequest('به‌روزرسانی داده فقط از تب داده‌های من انجام می‌شود.')
            symbols=request.POST.getlist('target_symbol') or request.POST.getlist('symbols')
            if len(symbols)!=1 or request.POST.get('new_symbols','').strip():
                return HttpResponseBadRequest('برای هر آموزش دقیقاً یک سهم هدف ثبت‌شده انتخاب کنید.')
            if symbols[0] not in instruments():return HttpResponseBadRequest('ابتدا نماد را در تب داده‌های من ثبت و داده‌های آن را دریافت کنید.')
            context_mode=request.POST.get('context_mode','all')
            if context_mode not in ('all','selected','none'):raise ValueError()
            context_symbols=list(dict.fromkeys(request.POST.getlist('context_symbols'))) if context_mode=='selected' else []
            if context_mode=='selected' and not context_symbols:return HttpResponseBadRequest('حداقل یک سهم زمینه انتخاب کنید یا حالت فقط سهم هدف را بزنید.')
            if any(name not in instruments() for name in context_symbols):raise ValueError()
        if not symbols or any(len(s)>60 or '/' in s or '\\' in s for s in symbols):raise ValueError()
        if data_request:
            if any(s not in instruments() for s in symbols):raise ValueError()
            hourly_days=int(request.POST.get('hourly_days','21'))
            if hourly_days not in (0,21,63):raise ValueError()
    except ValueError:return HttpResponseBadRequest('ورودی نامعتبر')
    RUNS.mkdir(parents=True,exist_ok=True)
    try:
        with FileLock(str(RUNS/'.pipeline.lock'),timeout=0):
            args=[sys.executable,str(BASE/'manage.py'),'train_forecast','--epochs',str(epochs),'--symbols',*symbols]
            if data_request:args.append('--refresh-only')
            else:
                args.append('--offline')
                if context_mode!='all':args.extend(['--context-symbols',*context_symbols])
            if data_request:args.extend(['--hourly-days',str(hourly_days),'--workers','4'])
            if data_request and request.POST.get('full_history')=='1':args.append('--full-history')
            with (RUNS/'worker.log').open('a',encoding='utf8') as log:
                subprocess.Popen(args,cwd=BASE,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    except Timeout:return HttpResponse('آموزش یا دریافت دیگری در حال اجراست.',status=409)
    if data_request:
        symbol=request.POST.get('view_symbol')
        return redirect('/data/?'+urlencode(dict(symbol=symbol if symbol in instruments() else symbols[0])))
    return redirect('/training/?'+urlencode(dict(symbol=symbols[0])))

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
                prediction_file=table_path(RUNS/row['run_id']/'predictions.json')
                ready=set(read_table(prediction_file,dtype={'insCode':str}).insCode) if prediction_file.exists() else set()
                if request.POST.get('scope')=='symbol':
                    code=instruments().get(request.POST.get('symbol'))
                    if not code or code not in ready or not (RUNS/row['run_id']/f'{code}_lstm.keras').exists():return HttpResponseBadRequest('مدل این نماد ارزیابی کامل ندارد')
                    codes=sorted(set(old.get('codes',[])+[code]))
                else:
                    codes=[p.name.split('_')[0] for p in (RUNS/row['run_id']).glob('*_lstm.keras') if p.name.split('_')[0] in ready]
                for code in codes:
                    preprocessing=RUNS/row['run_id']/f'{code}_preprocessing.json'
                    if not preprocessing.exists() or 'calibration_sigma_factor' not in json.loads(preprocessing.read_text(encoding='utf8')):
                        return HttpResponseBadRequest('مقیاس‌بندی یا کالیبراسیون مدل ناقص است')
                _write_bytes_atomic(marker,json.dumps({'name':name,'codes':codes,'saved_at':datetime.now(timezone.utc).isoformat()},ensure_ascii=False))
                if request.POST.get('activate'):
                    _,store=settings_store()
                    if hasattr(store,'connect'):
                        with store.connect() as db:
                            for code in codes:
                                manifest=dict(run_id=row['run_id'],code=code,model=f'{code}_lstm.keras',preprocessing=f'{code}_preprocessing.json')
                                db.execute('INSERT INTO model_registry VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET label=excluded.label,manifest=excluded.manifest',('active:'+code,name,json.dumps(manifest)))
                status=json.loads((RUNS/row['run_id']/'status.json').read_text(encoding='utf8'));status['label']=name
                _write_bytes_atomic(RUNS/row['run_id']/'status.json',json.dumps(status,ensure_ascii=False))
            elif request.POST.get('action')=='discard':
                (RUNS/row['run_id']/'saved.json').unlink(missing_ok=True)
                _,store=settings_store()
                if hasattr(store,'connect'):
                    with store.connect() as db:
                        for key,manifest in db.execute('SELECT id,manifest FROM model_registry').fetchall():
                            if json.loads(manifest).get('run_id')==row['run_id']:db.execute('DELETE FROM model_registry WHERE id=?',(key,))
            latest_model=next((r['run_id'] for r in rows if r.get('job_kind')!='data'),rows[0]['run_id'])
            prune(latest_model,RUNS)
    except Timeout:return HttpResponse('تا پایان آموزش منتظر بمانید.',status=409)
    return redirect('dashboard')

def download_model(request):
    import io,zipfile,hashlib
    rows=list_runs(RUNS);row=next((r for r in rows if r['run_id']==request.GET.get('run')),None)
    code=instruments().get(request.GET.get('symbol'))
    if not row or not code:return HttpResponseBadRequest('مدل یافت نشد')
    folder=RUNS/row['run_id']
    required=[folder/f'{code}_lstm.keras',folder/f'{code}_preprocessing.json']
    if not all(p.exists() for p in required):return HttpResponseBadRequest('مدل یا مقیاس‌بندی ناقص است')
    files=required+[p for p in [folder/'neural_network.py',folder/'neural_data.py',folder/'hourly_data.py',folder/'return_data.py',folder/'scenarios.py',folder/f'{code}_residuals.npz',folder/'environment.json',folder/f'{code}_architecture.txt'] if p.exists()]
    buffer=io.BytesIO();manifest={}
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            content=p.read_bytes();archive.writestr(p.name,content);manifest[p.name]=hashlib.sha256(content).hexdigest()
        archive.writestr('manifest.json',json.dumps(dict(run=row['run_id'],symbol=request.GET.get('symbol'),sha256=manifest),ensure_ascii=False,indent=2))
    buffer.seek(0)
    return FileResponse(buffer,as_attachment=True,filename=f"model_{request.GET.get('symbol')}_{row['run_id'][:15]}.zip")

@require_POST
def refresh_prediction(request):
    run=request.POST.get('run');symbol=request.POST.get('symbol')
    known=instruments();_,store=settings_store()
    row=next((r for r in catalog(list_runs(RUNS),RUNS,store) if r['run_id']==run),None)
    if row is None or symbol not in known:return HttpResponseBadRequest('مدل یا نماد نامعتبر')
    config_path=RUNS/run/f'{known[symbol]}_preprocessing.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf8')).get('target_schema')=='hourly_returns_v2':
        return HttpResponseBadRequest('برنامهٔ ساعت‌های این مدل ناسازگار است؛ برای افق کامل ۲۱ جلسه، مدل جدید را بازآموزی کنید.')
    if not reusable(RUNS,row,known[symbol]):return HttpResponseBadRequest('وزن مدل این نماد موجود نیست یا آموزش کامل نشده است؛ خروجی تاریخی قابل مشاهده است ولی برای پیش‌بینی تازه باید مدل این نماد آموزش داده و ذخیره شود.')
    try:
        with FileLock(str(RUNS/'.pipeline.lock'),timeout=0):
            with (RUNS/'inference.log').open('a',encoding='utf8') as log:
                subprocess.Popen([sys.executable,str(BASE/'manage.py'),'predict_market','--run',run,'--symbol',symbol],cwd=BASE,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    except Timeout:return HttpResponse('پردازش دیگری فعال است',status=409)
    return redirect('/market/?symbol='+symbol+'&run='+run)

def prediction_status(request):
    path=RUNS/'inference_status.json'
    return JsonResponse(json.loads(path.read_text(encoding='utf8')) if path.exists() else {'state':'idle'})


@require_POST
@csrf_protect
def delete_model(request):
    run_id=request.POST.get('run','')
    try:
        with model_mutation_lock(RUNS):
            _,store=settings_store()
            rows=model_records(RUNS,store,list_runs(RUNS),instruments())
            row=next((r for r in rows if r['run_id']==run_id),None)
            if row is None:return HttpResponseBadRequest('مدل یافت نشد.')
            if row.get('state')=='running':return HttpResponse('مدل در حال اجرا قابل حذف نیست.',status=409)
            delete_model_run(RUNS,store,run_id)
    except Timeout:return HttpResponse('تا پایان آموزش یا پیش‌بینی جاری منتظر بمانید.',status=409)
    except ValueError:return HttpResponseBadRequest('شناسهٔ مدل نامعتبر است.')
    return redirect('/training/?model_deleted=1')
