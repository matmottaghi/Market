import numpy as np
from test_neural_data import sample
from forecast.return_data import prepare_returns
from forecast.scenarios import scenario_paths,scenario_summary

def test_hourly_origins_and_next_return():
    h,d=sample()
    x,m,y,meta,*_=prepare_returns(h,d,[h])
    assert x.shape[-1]==16
    assert meta.as_of.str.startswith('2020-01-29').sum()>1
    first=h[h.Hour>meta.as_of.iloc[0]].iloc[0]
    np.testing.assert_allclose(y[0,0],np.log(first.Close/meta.current_price.iloc[0]),rtol=1e-5)

def test_scenarios_compound_and_diverge():
    residual=np.tile([-.01,.01],(100,1)).reshape(100,2)
    residual[:,0]=np.linspace(-.02,.02,100);residual[:,1]=residual[:,0]
    paths=scenario_paths([.01,-.01],residual,100)
    assert np.std(paths[:,1])>np.std(paths[:,0])
    row=scenario_summary(paths[:,1],100)
    assert row['lower_price']<row['predicted_price']<row['upper_price']
def test_hourly_context_has_no_future_leakage():
    h,d=sample()
    original=prepare_returns(h,d,[h.copy()])
    cutoff=original[3].as_of.iloc[0]
    changed=h.copy()
    changed.loc[changed.Hour>cutoff,['Open','High','Low','Close']]*=3
    updated=prepare_returns(h,d,[changed])
    np.testing.assert_array_equal(original[0][0],updated[0][0])
    assert original[3].target_end_21.iloc[0]>cutoff


def test_younger_context_does_not_truncate_target_history():
    h,d=sample()
    short=h[h.Date>=d.Date.iloc[300]].copy()
    own=prepare_returns(h,d)
    contextual=prepare_returns(h,d,[short])
    assert contextual[0].shape[-1]==16
    assert len(contextual[3])==len(own[3])
    assert contextual[3].as_of.iloc[0]==own[3].as_of.iloc[0]
    assert contextual[3].window_start.iloc[0]==str(d.Date.iloc[0].date())


def test_schedule_ignores_rare_afterhours_and_spans_21_full_sessions():
    import pandas as pd
    from forecast.return_data import trading_schedule,forecast_slots
    days=pd.bdate_range('2024-01-01',periods=60)
    rows=[dict(Hour=d+pd.Timedelta(hours=h)) for d in days for h in [9,10,11,12]]
    rows += [dict(Hour=days[-3]+pd.Timedelta(hours=h)) for h in [13,14,15,21]]
    hours=trading_schedule(pd.DataFrame(rows))
    assert hours==[9,10,11,12]
    slots=forecast_slots(hours,12)
    assert len(slots)==84
    assert slots[0]==dict(target_day=1,target_hour=9)
    assert slots[-1]==dict(target_day=21,target_hour=12)
    assert set(s['target_day'] for s in slots)==set(range(1,22))
    intraday=forecast_slots(hours,10)
    assert intraday[:2]==[dict(target_day=0,target_hour=11),dict(target_day=0,target_hour=12)]
    assert intraday[-1]==dict(target_day=21,target_hour=12)


def test_complete_regular_targets_keep_21_session_evaluation():
    import pandas as pd
    h,d=sample()
    h=h[h.Hour.dt.hour.isin([9,10])].copy()
    d=d.copy();d['qTotTran5J']=h.groupby('Date').Volume.sum().reindex(d.Date).to_numpy()
    *_,latest=prepare_returns(h,d)
    x,mask,y,meta,*_=prepare_returns(h,d)
    assert latest['future_hours']==[9,10]
    assert len(latest['forecast_slots'])==42
    assert np.isfinite(np.cumsum(y[:,:42],axis=1)).all()
