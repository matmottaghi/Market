# توضیح فایل: آزمون پنجرهٔ ۲۱ روز واقعی، جلوگیری از نشت آینده و تبدیل توزیع شبکه به بازهٔ قیمت.
# توابع: sample دادهٔ ساعتی با تعداد کندل متغیر می‌سازد؛ آزمون‌ها تقویم، مقیاس‌بندی، برچسب و سطح اطمینان را بررسی می‌کنند.
import numpy as np
import pandas as pd
import pytest
from forecast.neural_data import prepare_sequences, chronological_splits, scale_sequences, price_distribution


def sample():
    dates = pd.bdate_range('2020-01-01', periods=500)
    rows = []
    for day, date in enumerate(dates):
        for hour in range(9, 11+day % 3):
            price = 100*np.exp(day*.001+(hour-9)*.0001)
            rows.append(dict(Date=date, Hour=date+pd.Timedelta(hours=hour), Open=price, High=price*1.01,
                             Low=price*.99, Close=price, Volume=100+day, DataQuality='matched'))
    totals = pd.DataFrame(rows).groupby('Date').Volume.sum().reindex(dates).to_numpy()
    daily = pd.DataFrame(dict(Date=dates, qTotTran5J=totals, zTotTran=1))
    return pd.DataFrame(rows), daily


def test_actual_sessions_and_target_alignment():
    hourly, daily = sample()
    x, mask, y, meta, latest, latest_mask, info = prepare_sequences(hourly, daily)
    assert meta.context_sessions.eq(21).all()
    assert mask[0].sum() == hourly[hourly.Date <= daily.Date.iloc[20]].shape[0]
    expected = np.log(hourly.loc[hourly.Date.eq(daily.Date.iloc[41]), 'Close'].iloc[-1]/meta.current_price.iloc[0])
    assert y[0, 2] == pytest.approx(expected)
    assert meta.target_end_21.iloc[0] == str(daily.Date.iloc[41].date())
    assert latest_mask.sum() == hourly[hourly.Date >= daily.Date.iloc[-21]].shape[0]
    assert np.isfinite(x).all()


def test_missing_session_does_not_shorten_lookback():
    hourly, daily = sample()
    missing = daily.Date.iloc[40]
    hourly = hourly[hourly.Date != missing]
    _, _, _, meta, *_ = prepare_sequences(hourly, daily)
    assert not ((pd.to_datetime(meta.window_start) <= missing) & (pd.to_datetime(meta.as_of) >= missing)).any()


def test_stale_quality_flag_is_rechecked_against_daily_volume():
    hourly, daily = sample()
    day = daily.Date.iloc[40]
    daily.loc[daily.Date.eq(day), 'qTotTran5J'] *= 2
    _, _, _, meta, *_ = prepare_sequences(hourly, daily)
    assert not ((pd.to_datetime(meta.window_start) <= day) & (pd.to_datetime(meta.as_of) >= day)).any()


def test_purge_and_scaler_train_only():
    hourly, daily = sample()
    x, mask, y, meta, lx, lm, _ = prepare_sequences(hourly, daily)
    split = chronological_splits(meta)
    for left, right in [('train', 'validation'), ('validation', 'calibration'), ('calibration', 'test')]:
        assert meta.iloc[split[left]].target_end_21.max() < meta.iloc[split[right]].as_of.min()
    _, _, _, config = scale_sequences(x, mask, lx, lm, y, split)
    x[split['test']] += 999
    _, _, _, changed = scale_sequences(x, mask, lx, lm, y, split)
    assert config == changed


def test_future_changes_do_not_change_earlier_inputs():
    hourly, daily = sample()
    original = prepare_sequences(hourly, daily)
    for column in ['Open', 'High', 'Low', 'Close']:
        hourly.loc[hourly.Date >= daily.Date.iloc[150], column] *= 2
    changed = prepare_sequences(hourly, daily)
    before = pd.to_datetime(original[3].as_of) < daily.Date.iloc[150]
    np.testing.assert_array_equal(original[0][before], changed[0][before])


def test_confidence_changes_only_bounds():
    narrow = price_distribution(100, .01, .05, .80)
    wide = price_distribution(100, .01, .05, .95)
    assert 0 < wide['lower_price'] < narrow['lower_price']
    assert wide['upper_price'] > narrow['upper_price']
    assert wide['predicted_price'] == narrow['predicted_price']
    assert wide['probability_up']+wide['probability_down'] == pytest.approx(1)
    with pytest.raises(ValueError):
        price_distribution(100, 0, .1, .999)
