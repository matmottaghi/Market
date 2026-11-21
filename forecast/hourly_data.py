# توضیح فایل: هدف مستقیم همه ساعت‌های ۲۱ جلسه؛ ماسک هدف مفقود و زمینهٔ بازار با تاریخ/ساعت دقیق؛ حالت lagged فقط برای مدل‌های قدیمی محفوظ است.
# prepare_hourly_sequences ورودی/هدف؛ clean_context کنترل یکسان زمینه در آموزش/پیش‌بینی؛ scale_hourly مقیاس فقط آموزشی.
import numpy as np
import pandas as pd
from .neural_data import prepare_sequences

STEPS=21*24


def clean_context(frame,daily,symbol='',code='',include_closed_hours=False):
    """Accept only complete, finite OHLCV sessions, identically at train and inference."""
    if not frame.empty:
        frame=frame.copy()
        numeric=['Open','High','Low','Close','Volume']
        frame[numeric]=frame[numeric].apply(pd.to_numeric,errors='coerce')
        dates=pd.to_datetime(frame.Date).dt.normalize()
        reference=dict(zip(pd.to_datetime(daily.Date).dt.normalize(),pd.to_numeric(daily.qTotTran5J,errors='coerce')))
        now=pd.Timestamp.now(tz='Asia/Tehran');today=now.tz_localize(None).normalize()
        if include_closed_hours:
            frame=frame[(dates<today)|(pd.to_datetime(frame.Hour)+pd.Timedelta(hours=1)<=now.tz_localize(None))].copy()
            dates=pd.to_datetime(frame.Date).dt.normalize()
        valid=(frame.DataQuality.eq('matched') & np.isfinite(frame[numeric]).all(axis=1)
               & (frame[numeric]>0).all(axis=1) & ~frame.Hour.duplicated(keep=False)
               & (frame.High>=frame[['Open','Close','Low']].max(axis=1))
               & (frame.Low<=frame[['Open','Close']].min(axis=1))
               & pd.to_datetime(frame.Hour).dt.normalize().eq(dates)
               & ((dates<today)|((dates==today)&((now.hour>=18) | (include_closed_hours & (pd.to_datetime(frame.Hour)+pd.Timedelta(hours=1)<=now.tz_localize(None)))))))
        sums=frame.Volume.groupby(dates).sum();totals=pd.Series(sums.index.map(reference),index=sums.index,dtype=float)
        volume_ok=(sums/totals-1).abs()<=.001
        if include_closed_hours:volume_ok |= (sums.index==today)&(sums<=totals*1.001)
        accepted=valid.groupby(dates).all() & (totals>0) & volume_ok
        frame=frame[dates.isin(accepted[accepted].index)].copy()
    frame.attrs.update(symbol=symbol,code=str(code))
    return frame

def market_context(frames):
    parts=[]
    for frame in frames:
        if frame.empty:continue
        f=frame[frame.DataQuality.eq('matched')].copy().sort_values('Hour')
        f['Hour']=pd.to_datetime(f.Hour)
        f['r']=np.log(pd.to_numeric(f.Close,errors='coerce')/pd.to_numeric(f.Close,errors='coerce').shift(1))
        f.loc[f.Hour.diff()>pd.Timedelta(days=4),'r']=np.nan
        parts.append(f[['Hour','r']].dropna())
    if not parts:return pd.DataFrame(columns=['mean_return','breadth','coverage'])
    combined=pd.concat(parts)
    groups=combined.groupby('Hour').r
    return pd.DataFrame(dict(mean_return=groups.mean(),breadth=groups.apply(lambda r:float((r>0).mean())),coverage=groups.count()/max(len(frames),1)))

def prepare_hourly_sequences(hourly,daily,context_frames=(),allow_unlabeled=False,latest_only=False,alignment="exact"):
    x,mask,_,meta,lx,lm,latest=prepare_sequences(hourly,daily,require_all_targets=False,allow_unlabeled=allow_unlabeled,latest_only=latest_only)
    active=daily[(daily.qTotTran5J>0)&(daily.zTotTran>0)].copy()
    calendar=pd.DatetimeIndex(sorted(pd.to_datetime(active.Date).dt.normalize().unique()))
    volume=dict(zip(pd.to_datetime(active.Date).dt.normalize(),active.qTotTran5J))
    valid={}
    now=pd.Timestamp.now(tz='Asia/Tehran')
    for date,g in hourly.groupby(pd.to_datetime(hourly.Date).dt.normalize()):
        g=g.sort_values('Hour')
        closed=date.date()<now.date() or (date.date()==now.date() and now.hour>=18)
        if closed and g.DataQuality.eq('matched').all() and (g.Close>0).all() and volume.get(date,0)>0 and abs(g.Volume.sum()/volume[date]-1)<=.001:
            valid[date]=g
    y=np.full((len(meta),STEPS),np.nan,dtype='float32')
    for i,row in enumerate(meta.itertuples()):
        end=calendar.get_loc(pd.Timestamp(row.as_of))
        for day in range(1,22):
            date=calendar[end+day]
            if date not in valid:continue
            for bar in valid[date].itertuples():
                slot=(day-1)*24+pd.Timestamp(bar.Hour).hour
                y[i,slot]=np.log(bar.Close/row.current_price)
    individual=[]
    for frame in context_frames:
        if frame.empty:
            individual.append(pd.DataFrame(columns=['r','session_return','available','source_time']))
        else:
            f=frame[frame.DataQuality.eq('matched')].copy().sort_values('Hour')
            f['Hour']=pd.to_datetime(f.Hour)
            f['Date']=f.Hour.dt.normalize()
            f['r']=np.log(f.Close/f.Close.shift(1))
            f.loc[f.Hour.diff()>pd.Timedelta(days=4),'r']=np.nan
            previous=f.groupby('Date').Close.last().shift(1)
            f['session_return']=np.log(f.Close/f.Date.map(previous))
            f['available']=np.isfinite(f.r).astype(float)
            f['source_time']=f.Hour
            individual.append(f.set_index('Hour')[['r','session_return','available','source_time']])
    if context_frames:
        def augment(values,valid_mask,rows):
            result=np.zeros((*values.shape[:2],values.shape[2]+3+4*len(individual)),dtype='float32')
            result[:,:,:values.shape[2]]=values
            for i,row in enumerate(rows):
                dates=calendar[(calendar>=pd.Timestamp(row['window_start']))&(calendar<=pd.Timestamp(row['as_of']).normalize())]
                hours=pd.DatetimeIndex(np.concatenate([pd.to_datetime(valid[d].Hour).to_numpy() for d in dates]))
                prior=hours-pd.Timedelta(hours=1) if alignment=='lagged' else hours
                returns=[];sessions=[];availability=[]
                for j,series in enumerate(individual):
                    aligned=series.reindex(prior,method='ffill',tolerance=pd.Timedelta(days=4)) if alignment=='lagged' and not series.empty else series.reindex(prior)
                    r=aligned.r.fillna(0).to_numpy(dtype='float32')
                    sr=aligned.session_return.fillna(0).to_numpy(dtype='float32')
                    available=aligned.available.fillna(0).to_numpy(dtype='float32')
                    source=pd.DatetimeIndex(pd.to_datetime(aligned.source_time))
                    age=np.nan_to_num(np.asarray((prior-source)/pd.Timedelta(hours=96),dtype=float),nan=1).clip(0,1)
                    offset=values.shape[2]+3+j*4
                    result[i,:len(hours),offset:offset+4]=np.column_stack([r,sr,available,age])
                    returns.append(r*available);sessions.append((sr>0)*available);availability.append(available)
                count=np.maximum(np.sum(availability,axis=0),1)
                result[i,:len(hours),values.shape[2]:values.shape[2]+3]=np.column_stack([np.sum(returns,axis=0)/count,np.sum(sessions,axis=0)/count,np.sum(availability,axis=0)/len(individual)])
            return result
        x=augment(x,mask,meta.to_dict('records'))
        lx=augment(lx,lm,[latest])
    last=pd.Timestamp(latest['as_of']).normalize()
    latest['future_hours']=sorted(pd.to_datetime(valid[last].Hour).dt.hour.unique().astype(int).tolist())
    latest['as_of']=str(pd.Timestamp(latest['as_of'])+pd.Timedelta(hours=1))
    latest['target_schema']='hourly_path_v1'
    latest['context_assets']=len(context_frames)
    latest['feature_schema']='cross_asset_v2' if alignment=='lagged' else 'cross_asset_exact_v3'
    return x,mask,y,meta,lx,lm,latest

def scale_hourly(x,mask,lx,lm,y,splits):
    train=splits['train'];observed=x[train][mask[train]]
    mean=observed.mean(axis=0);scale=np.maximum(observed.std(axis=0),1e-6)
    counts=np.isfinite(y[train]).sum(axis=0)
    target_scale=np.array([max(float(np.nanstd(y[train,j])),.001) if counts[j] else 1 for j in range(y.shape[1])])
    xs=((x-mean)/scale).astype('float32');xs[~mask]=0
    ls=((lx-mean)/scale).astype('float32');ls[~lm]=0
    labels=np.concatenate([np.nan_to_num(y/target_scale),np.isfinite(y).astype('float32')],axis=1).astype('float32')
    return xs,ls,labels,dict(feature_mean=mean.tolist(),feature_scale=scale.tolist(),target_scale=target_scale.tolist(),target_counts=counts.tolist())
