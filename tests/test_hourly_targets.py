# توضیح فایل: آزمون برچسب مستقیم هر ساعت، ماسک ساعت بدون معامله و جداسازی مقیاس آموزشی.
import numpy as np
import pandas as pd
from test_neural_data import sample
from forecast.hourly_data import prepare_hourly_sequences,scale_hourly
from forecast.neural_data import chronological_splits

def test_hourly_targets_and_masks():
    hourly,daily=sample()
    x,mask,y,meta,lx,lm,latest=prepare_hourly_sequences(hourly,daily,[hourly.copy()])
    assert y.shape[1]==504 and x.shape[-1]==15
    assert np.isnan(y[0,0]) # No midnight candle.
    first_future=pd.Timestamp(meta.target_end_1.iloc[0])
    expected=np.log(hourly[(hourly.Date==first_future)&(hourly.Hour.dt.hour==9)].Close.iloc[0]/meta.current_price.iloc[0])
    assert y[0,9]==np.float32(expected)
    split=chronological_splits(meta)
    xs,ls,ys,config=scale_hourly(x,mask,lx,lm,y,split)
    assert ys.shape[1]==1008 and np.isfinite(ys).all()
    assert (ys[:,504]==0).all()
    assert latest['target_schema']=='hourly_path_v1'

def test_missing_future_does_not_discard_known_short_targets():
    hourly,daily=sample()
    missing=daily.Date.iloc[41]
    hourly=hourly[hourly.Date!=missing]
    _,_,y,meta,*_=prepare_hourly_sequences(hourly,daily)
    assert meta.as_of.iloc[0]==str(daily.Date.iloc[20].date())
    assert np.isfinite(y[0,9])
    assert np.isnan(y[0,480:]).all()

def test_inference_requires_only_21_past_sessions():
    hourly,daily=sample();daily=daily.iloc[-21:]
    hourly=hourly[hourly.Date.isin(daily.Date)]
    x,mask,y,meta,latest,latest_mask,info=prepare_hourly_sequences(hourly,daily,allow_unlabeled=True)
    assert len(x)==0 and len(meta)==0
    assert latest.shape[0]==1 and latest_mask.sum()==len(hourly)

def test_later_market_close_never_leaks_into_earlier_target():
    hourly,daily=sample()
    gold=hourly.copy();gold['Hour']+=pd.Timedelta(hours=6)
    original=prepare_hourly_sequences(hourly,daily,[gold])
    cutoff=daily.Date.iloc[20]
    for col in ['Open','High','Low','Close']:
        gold.loc[gold.Date>=cutoff,col]*=2
    changed=prepare_hourly_sequences(hourly,daily,[gold])
    np.testing.assert_array_equal(original[0][0],changed[0][0])
    last=int(original[1][0].sum())-1
    assert original[0][0,last,13]==1 # Prior completed observation is available.
    assert original[0][0,last,14]>0  # Staleness is explicit.


def test_context_rejects_stale_volume_and_invalid_ohlc():
    from forecast.hourly_data import clean_context
    hourly,daily=sample()
    bad=hourly.copy();date=daily.Date.iloc[1]
    bad.loc[bad.Date==date,'High']=0
    daily.loc[2,'qTotTran5J']*=2
    result=clean_context(bad,daily,code='123')
    assert date not in result.Date.values and daily.Date.iloc[2] not in result.Date.values
    assert result.attrs['code']=='123'


def test_latest_only_matches_full_preparation():
    hourly,daily=sample()
    full=prepare_hourly_sequences(hourly,daily,[hourly])
    fast=prepare_hourly_sequences(hourly,daily,[hourly],allow_unlabeled=True,latest_only=True)
    n=full[5].sum()
    np.testing.assert_allclose(full[4][0,:n],fast[4][0,:n])
    assert fast[6]==full[6] and len(fast[0])==0
