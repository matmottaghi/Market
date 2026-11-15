# توضیح فایل: اجرای آموزش واقعی LSTM، ثبت منحنی هر epoch، کالیبراسیون مستقل و ذخیرهٔ مدل و دادهٔ آماده.
# توابع: decode تبدیل خروجی شبکه به پارامتر بازده؛ train_symbol آموزش یک نماد با سه افق؛
# run_neural_pipeline هماهنگی دریافت، ثبت وضعیت و آموزش نمادهای انتخابی؛ EpochRecorder.on_epoch_end ثبت زندهٔ منحنی‌ها.
# row_without_epoch مقادیر عددی منحنی را برای ذخیرهٔ پیشرفت از شمارهٔ epoch جدا می‌کند؛ update وضعیت اتمیک اجرا را ثبت می‌کند.
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sys
import uuid

import numpy as np
import pandas as pd
import scipy
from filelock import FileLock
from scipy.special import ndtr

from market_data import Settings, CsvMarketStore, TsetmcClient, sync_market
from market_data.csv_store import _write_bytes_atomic, _csv_atomic
from .pipeline import BASE, RUNS, save_status
from .registry import instruments, prune, list_runs
from .neural_data import prepare_sequences, chronological_splits, scale_sequences, price_distribution, HORIZONS, FEATURE_NAMES


def decode(raw, target_scale, calibration=None):
    raw = np.asarray(raw)
    if not np.isfinite(raw).all():
        raise ValueError('Network produced non-finite parameters')
    mu = raw[:, :3]*target_scale
    sigma = (np.logaddexp(0, raw[:, 3:])+.05)*target_scale
    if calibration is not None:
        sigma *= calibration
    return mu, np.maximum(sigma, 1e-6)


def train_symbol(hourly, daily, symbol, code, folder, epochs, update):
    import keras
    from .neural_network import build_network
    keras.backend.clear_session()
    x, mask, y, metadata, latest_x, latest_mask, latest_info = prepare_sequences(hourly, daily)
    splits = chronological_splits(metadata)
    xs, latest_scaled, ys, scaling = scale_sequences(x, mask, latest_x, latest_mask, y, splits)
    metadata['split'] = 'purged'
    for name, indices in splits.items():
        metadata.loc[indices, 'split'] = name
    _csv_atomic(metadata, folder / f'{code}_windows.csv')
    np.savez_compressed(folder / f'{code}_prepared.npz', x=x, mask=mask, y_log_return=y,
                        latest_x=latest_x, latest_mask=latest_mask, **splits)
    _csv_atomic(hourly, folder / f'{code}_hourly_dataset.csv')
    details = dict(scaling=scaling, horizons=list(HORIZONS), lookback_sessions=21,
                   feature_names=FEATURE_NAMES, latest=latest_info,
                   split_counts={name: len(indices) for name, indices in splits.items()},
                   input_shape=list(x.shape), model_kind='lstm',
                   split_boundaries={name: dict(first=str(metadata.iloc[idx[0]].as_of),
                                                last=str(metadata.iloc[idx[-1]].as_of),
                                                last_target=str(metadata.iloc[idx[-1]].target_end_21))
                                     for name, idx in splits.items()})
    _write_bytes_atomic(folder / f'{code}_preprocessing.json', json.dumps(details, ensure_ascii=False, indent=2))
    update('prepared', symbol=symbol, input_shape=list(x.shape), splits=details['split_counts'],
           latest_as_of=latest_info['as_of'], source_last_session=str(pd.to_datetime(daily.Date).max().date()))
    model = build_network(len(FEATURE_NAMES))
    summary = []
    model.summary(print_fn=lambda message, **kwargs: summary.append(message))
    _write_bytes_atomic(folder / f'{code}_architecture.txt', '\n'.join(summary))
    rows = []

    class EpochRecorder(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            row = dict(epoch=epoch+1, **{name: float(value) for name, value in (logs or {}).items()})
            rows.append(row)
            _csv_atomic(pd.DataFrame(rows), folder / f'{code}_learning.csv')
            update('epoch', symbol=symbol, epoch=epoch+1, max_epochs=epochs, **row_without_epoch(row))

    train, validation = splits['train'], splits['validation']
    model.fit([xs[train], mask[train]], ys[train],
              validation_data=([xs[validation], mask[validation]], ys[validation]),
              epochs=epochs, batch_size=32, shuffle=False, verbose=0,
              callbacks=[EpochRecorder(), keras.callbacks.TerminateOnNaN(), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, factor=.5, min_lr=1e-6),
                         keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, min_delta=.001,
                                                       restore_best_weights=True)])
    if not rows or not np.isfinite([row['loss'] for row in rows]).all():
        raise ValueError('Training did not complete with finite loss')
    model.save(folder / f'{code}_lstm.keras')
    scale = np.array(scaling['target_scale'])
    calibration = splits['calibration']
    mu_cal, sigma_cal = decode(model.predict([xs[calibration], mask[calibration]], verbose=0), scale)
    factors = np.maximum(np.sqrt(np.mean(((y[calibration]-mu_cal)/sigma_cal)**2, axis=0)), .05)
    details.update(calibration_sigma_factor=factors.tolist(), epochs_completed=len(rows),
                   best_epoch=int(np.argmin([row['val_loss'] for row in rows]))+1)
    _write_bytes_atomic(folder / f'{code}_preprocessing.json', json.dumps(details, ensure_ascii=False, indent=2))
    test = splits['test']
    mu, sigma = decode(model.predict([xs[test], mask[test]], verbose=0), scale, factors)
    latest_mu, latest_sigma = decode(model.predict([latest_scaled, latest_mask], verbose=0), scale, factors)
    predictions, metrics = [], []
    for column, horizon in enumerate(HORIZONS):
        base = dict(model_kind='lstm', symbol=symbol, insCode=code, frequency='hourly', horizon=horizon,
                    horizon_unit='trading_days', lookback_sessions=21)
        forecast = price_distribution(latest_info['current_price'], float(latest_mu[0, column]), float(latest_sigma[0, column]))
        forecast.update(base, as_of=latest_info['as_of'], current_price=latest_info['current_price'],
                        mu_log=float(latest_mu[0, column]), sigma_log=float(latest_sigma[0, column]),
                        distribution='lognormal_price', direction='صعودی' if forecast['probability_up'] >= .5 else 'نزولی')
        forecast['lower90'] = forecast['lower_price']/forecast['current_price']-1
        forecast['upper90'] = forecast['upper_price']/forecast['current_price']-1
        predictions.append(forecast)
        actual = np.expm1(y[test, column])
        expected = np.expm1(mu[:, column]+.5*sigma[:, column]**2)
        probability = ndtr(mu[:, column]/sigma[:, column])
        lower, upper = np.expm1(mu[:, column]-1.6448536269514722*sigma[:, column]), np.expm1(mu[:, column]+1.6448536269514722*sigma[:, column])
        history = pd.DataFrame(dict(timestamp=metadata.iloc[test].as_of.to_numpy(), actual=actual,
                                    predicted=expected, probability_up=probability, probability_down=1-probability,
                                    lower90=lower, upper90=upper, mu_log=mu[:, column], sigma_log=sigma[:, column],
                                    current_price=metadata.iloc[test].current_price.to_numpy()))
        history['expanding_mae'] = abs(history.actual-history.predicted).expanding().mean()
        cal_sigma = sigma_cal[:, column]*factors[column]
        _csv_atomic(pd.DataFrame(dict(timestamp=metadata.iloc[calibration].as_of.to_numpy(),
            actual=np.expm1(y[calibration, column]), probability_up=ndtr(mu_cal[:, column]/cal_sigma))),
            folder / f'{code}_hourly_{horizon}_calibration.csv')
        _csv_atomic(history, folder / f'{code}_hourly_{horizon}_test.csv')
        scores = dict(base, mae=float(np.mean(abs(actual-expected))), rmse=float(np.sqrt(np.mean((actual-expected)**2))),
                      direction_accuracy=float(np.mean((probability >= .5) == (actual > 0))),
                      brier=float(np.mean((probability-(actual > 0))**2)), coverage90=float(np.mean((actual >= lower)&(actual <= upper))),
                      baseline_zero_mae=float(np.mean(abs(actual))),
                      baseline_drift_mae=float(np.mean(abs(actual-np.expm1(y[train, column]).mean()))),
                      **{f'{name}_rows': len(idx) for name, idx in splits.items()})
        metrics.append(scores)
    update('trained', symbol=symbol, epochs_completed=len(rows), best_epoch=details['best_epoch'])
    return predictions, metrics


def row_without_epoch(row):
    return {name: value for name, value in row.items() if name != 'epoch'}


def run_neural_pipeline(refresh=True, max_hourly_days=0, full_history=False, workers=1, epochs=200, symbols=None, refresh_only=False):
    if not 1 <= epochs <= 200:
        raise ValueError('Epochs must be between 1 and 200')
    RUNS.mkdir(parents=True, exist_ok=True)
    with FileLock(str(RUNS / '.pipeline.lock'), timeout=0):
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-nn-'+uuid.uuid4().hex[:6]
        previous = next((r for r in list_runs() if r.get('model_kind')=='lstm' and r.get('job_kind')!='data' and r['state'] in ('complete','partial')), None)
        folder = RUNS / run_id
        folder.mkdir()
        if previous and (RUNS/previous['run_id']/'metrics.csv').exists():
            shutil.copyfile(RUNS/previous['run_id']/'metrics.csv', folder/'baseline_metrics.csv')
        status = dict(label=('دریافت داده — ' if refresh_only else 'DeepLSTM 96×48 — ')+pd.Timestamp.now(tz='Asia/Tehran').strftime('%Y-%m-%d %H:%M تهران'), run_id=run_id, job_kind='data' if refresh_only else 'training', model_kind='lstm', state='running', stages=[], errors=[], warnings=[], started_at=datetime.now(timezone.utc).isoformat())

        def update(stage, **details):
            if stage == 'prepared' and details['latest_as_of'][:10] < details['source_last_session']:
                status['warnings'].append(f"{details['symbol']}: latest complete context {details['latest_as_of']}; daily archive through {details['source_last_session']}")
            if stage == 'epoch':
                status['training_progress'] = details
            elif stage == 'download_progress':
                status['download_progress'] = details
            else:
                status['stages'].append(dict(stage=stage, **details))
            save_status(folder, status)

        save_status(folder, status)
        results, metrics = [], []
        try:
            import tensorflow as tf
            import keras
            tf.config.threading.set_intra_op_parallelism_threads(2)
            tf.config.threading.set_inter_op_parallelism_threads(2)
            for name in ['neural_data.py', 'neural_network.py', 'neural_training.py']:
                shutil.copyfile(Path(__file__).with_name(name), folder / name)
            _write_bytes_atomic(folder / 'environment.json', json.dumps(dict(python=sys.version,
                packages=dict(tensorflow=tf.__version__, keras=keras.__version__, numpy=np.__version__,
                              pandas=pd.__version__, scipy=scipy.__version__),
                model='LSTM96 -> LSTM48 -> Dense64 -> Dense32 -> Gaussian heads', direction_loss_weight=.2,
                epochs=epochs, seed=42, batch_size=32, lookback_sessions=21, horizons=list(HORIZONS),
                split='60/15/10/15 with max-horizon purge; calibration independent of early stopping',
                refresh=refresh), indent=2))
            settings = Settings.from_env(BASE / '.env')
            if not settings.root.is_absolute():
                settings = replace(settings, root=BASE / settings.root)
            store = CsvMarketStore(settings.root)
            selected = symbols or list(instruments())
            for symbol in selected:
                try:
                    codes = [p.stem for p in (settings.root/'state').glob('*.json') if store.read_state(p.stem).get('symbol') == symbol]
                    if not codes and not refresh:
                        raise ValueError('No cached instrument; enable data refresh')
                    if codes:
                        code = codes[0]
                    else:
                        resolver = TsetmcClient(settings)
                        try:
                            code = resolver.resolve_symbol_code(symbol)
                        finally:
                            resolver.session.close()
                    if refresh:
                        try:
                            report = sync_market({symbol: code}, settings=settings, store=store,
                                max_hourly_days=max_hourly_days, full_reconcile=full_history,
                                retry_incomplete_hourly=full_history, workers=workers,
                                progress=lambda details: update('download_progress', symbol=symbol, **details))
                            update('download', symbol=symbol, report=report)
                            if report[symbol]['status'] != 'ok':
                                status['errors'].append(f'{symbol}: incomplete market archive; see coverage')
                        except Exception as exc:
                            status['errors'].append(f'{symbol}: refresh failed; training uses cached valid data: {exc}')
                    if refresh_only:
                        update('data_ready', symbol=symbol)
                        continue
                    predictions, scores = train_symbol(store.load_hourly(code), store.load_daily(code), symbol, code, folder, epochs, update)
                    for row in predictions:
                        row['run_id'] = run_id
                    results.extend(predictions)
                    metrics.extend(scores)
                    _csv_atomic(pd.DataFrame(results), folder/'predictions.csv')
                    _csv_atomic(pd.DataFrame(metrics), folder/'metrics.csv')
                except Exception as exc:
                    status['errors'].append(f'{symbol}: {exc}')
                    update('training_failed', symbol=symbol, error=str(exc))
            status['state'] = ('partial' if status['errors'] else 'complete') if results or refresh_only else 'failed'
        except Exception as exc:
            status['state'] = 'failed'
            status['errors'].append(str(exc))
        status['finished_at'] = datetime.now(timezone.utc).isoformat()
        save_status(folder, status)
        if results:
            prune(run_id)
        return status
