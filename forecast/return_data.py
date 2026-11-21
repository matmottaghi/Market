"""Hourly decision windows, eight features per asset, next-hour return paths."""
import numpy as np
import pandas as pd
from .hourly_data import clean_context, STEPS
from .neural_data import FEATURE_NAMES

SCHEMA = 'hourly_returns_v3'

def trading_schedule(frame):
    """Modal complete-session pattern in the most recent 60 verified sessions."""
    from collections import Counter
    today=pd.Timestamp.now(tz='Asia/Tehran').tz_localize(None).normalize()
    closed=frame[pd.to_datetime(frame.Hour).dt.normalize()<today]
    patterns=[tuple(sorted(pd.to_datetime(g.Hour).dt.hour.unique().tolist())) for _,g in closed.groupby(pd.to_datetime(closed.Hour).dt.normalize(),sort=True)]
    if not patterns:raise ValueError('No closed sessions available to determine trading hours')
    recent=patterns[-60:];counts=Counter(recent)
    return list(max(reversed(recent),key=lambda pattern:counts[pattern]))

def forecast_slots(hours, origin_hour):
    # Session zero is only the remainder of the current session; then 21 full sessions.
    return ([dict(target_day=0,target_hour=int(h)) for h in hours if h>origin_hour]+
            [dict(target_day=d,target_hour=int(h)) for d in range(1,22) for h in hours])

def prepare_returns(hourly, daily, context_frames=(), allow_unlabeled=False, latest_only=False, trading_hours=None, **kwargs):
    frame = clean_context(hourly, daily, include_closed_hours=True).sort_values('Hour').reset_index(drop=True)
    if frame.empty:
        raise ValueError('No verified hourly history')
    frame['Hour'] = pd.to_datetime(frame.Hour)
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(daily.loc[(daily.qTotTran5J>0)&(daily.zTotTran>0),'Date']).unique()))
    groups = {d:g for d,g in frame.groupby(frame.Hour.dt.normalize())}
    hours = list(trading_hours) if trading_hours is not None else trading_schedule(frame)
    if not hours or hours!=sorted(set(hours)) or any(h<0 or h>23 for h in hours):raise ValueError('Invalid frozen trading schedule')
    # Rare out-of-session bars never become recurring future trading hours.
    frame=frame[frame.Hour.dt.hour.isin(hours)].copy()
    groups = {d:g for d,g in frame.groupby(frame.Hour.dt.normalize())}
    # Fixed slots retain missing candles rather than inventing prices.
    slots = pd.DatetimeIndex([d+pd.Timedelta(hours=int(h)) for d in dates for h in hours])
    indexed = frame.set_index('Hour')
    assets = [frame] + list(context_frames)
    features=[]
    for asset in assets:
        a=asset.copy();a['Hour']=pd.to_datetime(a.Hour);a=a.set_index('Hour').reindex(slots)
        blocks=np.column_stack([np.log(a[c].to_numpy(dtype=float)) for c in ['Open','High','Low','Close']] + [np.log1p(a.Volume.to_numpy(dtype=float)), np.sin(2*np.pi*slots.hour/24),np.cos(2*np.pi*slots.hour/24),np.zeros(len(slots))])
        features.append(blocks)
    values=np.concatenate(features,axis=1)
    records=[];windows=[];targets=[];latest=None
    ends=range(len(slots)-1,-1,-1) if latest_only else range(len(slots))
    for end in ends:
        date=slots[end].normalize();di=dates.get_loc(date)
        if di<20 or slots[end] not in indexed.index:continue
        past=dates[di-20:di+1]
        if any(d not in groups for d in past):continue
        start=slots.searchsorted(past[0]);window=values[start:end+1].copy()
        own=np.isfinite(window[:,:5]).all(axis=1)
        window=window[own]
        if not len(window):continue
        for j in range(len(assets)):
            ref=window[:,j*8][np.isfinite(window[:,j*8])]
            if len(ref):window[:,j*8:j*8+4]-=ref[0]
            window[:,j*8+7]=np.linspace(0,1,len(window))
        window=np.nan_to_num(window).astype('float32')
        price=float(indexed.loc[slots[end],'Close'])
        latest=dict(sequence=window,as_of=str(slots[end]+pd.Timedelta(hours=1)),window_start=str(past[0].date()),current_price=price,future_hours=hours,target_schema=SCHEMA,feature_schema='assets_eight_v1',context_assets=len(context_frames))
        if latest_only:break
        if di+21>=len(dates):continue
        y=np.full(STEPS,np.nan,dtype='float32')
        future=slots[end+1:slots.searchsorted(dates[di+21]+pd.Timedelta(days=1))]
        previous=price
        for k,t in enumerate(future[:STEPS]):
            if t in indexed.index:
                close=float(indexed.loc[t,'Close'])
                if previous is not None:y[k]=np.log(close/previous)
                previous=close
            else:previous=None
        records.append(dict(as_of=str(slots[end]),window_start=str(past[0].date()),current_price=price,target_end_21=str(dates[di+21]+pd.Timedelta(days=1))))
        windows.append(window);targets.append(y)
    if latest is None or (not windows and not allow_unlabeled):raise ValueError('Insufficient verified hourly windows')
    length=max([len(w) for w in windows]+[len(latest['sequence'])]);width=8*len(assets)
    x=np.zeros((len(windows),length,width),dtype='float32');mask=np.zeros(x.shape[:2],bool)
    for i,w in enumerate(windows):x[i,:len(w)]=w;mask[i,:len(w)]=True
    lx=np.zeros((1,length,width),dtype='float32');lm=np.zeros((1,length),bool);w=latest.pop('sequence');lx[0,:len(w)]=w;lm[0,:len(w)]=True
    origin_hour=(pd.Timestamp(latest['as_of'])-pd.Timedelta(hours=1)).hour
    latest['forecast_slots']=forecast_slots(hours,origin_hour)
    if len(latest['forecast_slots'])>STEPS:raise ValueError('Network output capacity cannot cover 21 complete sessions for this schedule')
    latest['schedule_schema']='recent_modal_v1'
    return x,mask,np.asarray(targets,dtype='float32').reshape(-1,STEPS),pd.DataFrame(records),lx,lm,latest
