# توضیح فایل: آماده‌سازی ورودی ساعتی شبکه با پنجرهٔ دقیق ۲۱ جلسهٔ معاملاتی و هدف بازده لگاریتمی.
# توابع: prepare_sequences پنجره و برچسب می‌سازد؛ chronological_splits بخش‌های بدون نشت می‌سازد؛
# scale_sequences فقط از آموزش پارامتر مقیاس را می‌آموزد؛ price_distribution خروجی قیمت و احتمال را محاسبه می‌کند.
# تابع داخلی window همهٔ کندل‌های ۲۱ جلسهٔ پیوسته را بدون حذف روز نامعتبر کنار هم می‌چیند.
import numpy as np
import pandas as pd
from scipy.stats import norm

LOOKBACK = 21
HORIZONS = (1, 5, 21)
FEATURE_NAMES = ['log_open', 'log_high', 'log_low', 'log_close', 'log_volume',
                 'hour_sin', 'hour_cos', 'session_position']


def prepare_sequences(hourly, daily, require_all_targets=True, allow_unlabeled=False, latest_only=False):
    """One observation at each complete session close; never bridge a missing session."""
    daily = daily.copy()
    daily['Date'] = pd.to_datetime(daily.Date).dt.normalize()
    daily = daily.drop_duplicates('Date', keep='last')
    daily_volume = dict(zip(daily.Date, pd.to_numeric(daily.qTotTran5J, errors='coerce')))
    active = (pd.to_numeric(daily.qTotTran5J, errors='coerce') > 0) & (pd.to_numeric(daily.zTotTran, errors='coerce') > 0)
    calendar = pd.DatetimeIndex(sorted(daily.loc[active, 'Date'].unique()))
    data = hourly.copy()
    data['Date'] = pd.to_datetime(data.Date).dt.normalize()
    data['Hour'] = pd.to_datetime(data.Hour)
    if data.duplicated(['Date', 'Hour']).any():
        raise ValueError('Duplicate hourly keys')
    numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    data[numeric] = data[numeric].apply(pd.to_numeric, errors='coerce')
    groups = {}
    for date, group in data.groupby('Date', sort=True):
        group = group.sort_values('Hour')
        valid = (group.DataQuality.eq('matched').all() and group[numeric].notna().all().all()
                 and np.isfinite(group[numeric]).all().all()
                 and (group[numeric] > 0).all().all() and (group.High >= group.Low).all()
                 and (group.High >= group[['Open', 'Close']].max(axis=1)).all()
                 and (group.Low <= group[['Open', 'Close']].min(axis=1)).all()
                 and group.Hour.dt.normalize().eq(date).all()
                 and daily_volume.get(date, 0) > 0
                 and abs(group.Volume.sum()/daily_volume[date]-1) <= .001)
        # Current session can still gain hours even when the volume snapshot matches.
        now = pd.Timestamp.now(tz='Asia/Tehran')
        closed = date.date() < now.date() or (date.date() == now.date() and now.hour >= 18)
        if valid and closed:
            groups[date] = group

    day_values = {}
    for date, part in groups.items():
        hours = part.Hour.dt.hour.to_numpy() + part.Hour.dt.minute.to_numpy()/60
        day_values[date] = np.column_stack([
            np.log(part[column].to_numpy()) for column in ['Open', 'High', 'Low', 'Close']
        ] + [np.log1p(part.Volume.to_numpy()), np.sin(2*np.pi*hours/24),
             np.cos(2*np.pi*hours/24), np.zeros(len(part))])

    def window(end):
        dates = calendar[end-LOOKBACK+1:end+1]
        if len(dates) != LOOKBACK or any(date not in groups for date in dates):
            return None
        blocks = []
        reference = np.log(float(groups[dates[0]].Open.iloc[0]))
        for offset, date in enumerate(dates):
            values = day_values[date].copy()
            values[:, :4] -= reference
            values[:, -1] = offset/(LOOKBACK-1)
            blocks.append(values)
        return np.concatenate(blocks).astype('float32')

    sequences, labels, records = [], [], []
    latest = None
    endpoints=range(len(calendar)-1,LOOKBACK-2,-1) if latest_only else range(LOOKBACK-1,len(calendar))
    for end in endpoints:
        sequence = window(end)
        if sequence is None:
            continue
        date = calendar[end]
        price = float(groups[date].Close.iloc[-1])
        latest = dict(sequence=sequence, as_of=str(groups[date].Hour.iloc[-1]),
                      window_start=str(calendar[end-LOOKBACK+1].date()), current_price=price)
        if latest_only:break
        if end+max(HORIZONS) >= len(calendar):
            continue
        targets = [calendar[end+h] for h in HORIZONS]
        if require_all_targets and any(target not in groups for target in targets):
            continue
        if not require_all_targets and not any(calendar[end+h] in groups for h in range(1,22)):
            continue
        labels.append([np.log(float(groups[target].Close.iloc[-1])/price) if target in groups else np.nan for target in targets])
        sequences.append(sequence)
        record = dict(as_of=str(date.date()), window_start=str(calendar[end-LOOKBACK+1].date()),
                      context_sessions=LOOKBACK, bars=len(sequence), current_price=price)
        record.update({f'target_end_{h}': str(target.date()) for h, target in zip(HORIZONS, targets)})
        records.append(record)
    if latest is None or (not sequences and not allow_unlabeled):
        raise ValueError('No complete 21-session hourly windows with valid future targets')
    length = max(max(map(len, sequences),default=0), len(latest['sequence']))
    x = np.zeros((len(sequences), length, len(FEATURE_NAMES)), dtype='float32')
    mask = np.zeros(x.shape[:2], dtype=bool)
    for i, sequence in enumerate(sequences):
        x[i, :len(sequence)] = sequence
        mask[i, :len(sequence)] = True
    latest_x = np.zeros((1, length, len(FEATURE_NAMES)), dtype='float32')
    latest_mask = np.zeros((1, length), dtype=bool)
    sequence = latest.pop('sequence')
    latest_x[0, :len(sequence)] = sequence
    latest_mask[0, :len(sequence)] = True
    columns=['as_of','window_start','context_sessions','bars','current_price']+[f'target_end_{h}' for h in HORIZONS]
    return x, mask, np.asarray(labels, dtype='float32'), pd.DataFrame(records,columns=columns), latest_x, latest_mask, latest


def chronological_splits(metadata):
    n = len(metadata)
    cuts = [0, int(n*.60), int(n*.75), int(n*.85), n]
    result = {}
    for i, name in enumerate(['train', 'validation', 'calibration', 'test']):
        index = np.arange(cuts[i], cuts[i+1])
        if i < 3:
            boundary = metadata.iloc[cuts[i+1]].as_of
            index = index[metadata.iloc[index]['target_end_21'].to_numpy() < boundary]
        if len(index) < 20:
            raise ValueError(f'Not enough purged {name} samples: {len(index)}')
        result[name] = index
    return result


def scale_sequences(x, mask, latest_x, latest_mask, y, splits):
    train = splits['train']
    observed = x[train][mask[train]]
    mean, scale = observed.mean(axis=0), np.maximum(observed.std(axis=0), 1e-6)
    target_scale = np.maximum(y[train].std(axis=0), .001)
    transformed = ((x-mean)/scale).astype('float32')
    transformed[~mask] = 0
    latest = ((latest_x-mean)/scale).astype('float32')
    latest[~latest_mask] = 0
    config = dict(feature_mean=mean.tolist(), feature_scale=scale.tolist(), target_scale=target_scale.tolist())
    return transformed, latest, (y/target_scale).astype('float32'), config


def price_distribution(current_price, mu, sigma, confidence=.90):
    if not .50 <= confidence <= .99:
        raise ValueError('Confidence must be between 0.50 and 0.99')
    if current_price <= 0 or sigma <= 0 or not np.isfinite([current_price, mu, sigma]).all():
        raise ValueError('Invalid log-return distribution')
    z = norm.ppf((1+confidence)/2)
    probability = float(norm.cdf(mu/sigma))
    return dict(probability_up=probability, probability_down=1-probability,
                predicted_return=float(np.expm1(mu+.5*sigma*sigma)),
                predicted_price=float(current_price*np.exp(mu+.5*sigma*sigma)),
                lower_price=float(current_price*np.exp(mu-z*sigma)),
                upper_price=float(current_price*np.exp(mu+z*sigma)), confidence=confidence)
