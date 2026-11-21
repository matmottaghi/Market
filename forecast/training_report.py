"""Training-panel report from the selected run's frozen artifacts only."""
import json
from .table_io import table_path, read_table

SPLITS = [('train', 'آموزش'), ('validation', 'اعتبارسنجی'), ('calibration', 'کالیبراسیون'), ('test', 'آزمون مستقل')]


def training_report(folder, code, symbol, run):
    if not run or not code:
        return None
    path = folder / f'{code}_preprocessing.json'
    config = {}
    warning = ''
    if path.exists():
        try:
            config = json.loads(path.read_text(encoding='utf8'))
        except (OSError, ValueError):
            warning = 'خواندن مشخصات ذخیره‌شدهٔ این مدل ممکن نشد.'
    stage = next((s for s in reversed(run.get('stages', [])) if s.get('stage') == 'prepared' and s.get('symbol') == symbol), {})
    if not config and not stage:
        return dict(available=False, warning=warning)
    counts = config.get('split_counts', stage.get('splits', {}))
    boundaries = config.get('split_boundaries', {})
    periods = config.get('split_input_periods', {})
    windows = table_path(folder / f'{code}_windows.json')
    if windows.exists() and not periods:
        try:
            frame = read_table(windows)
            if 'split' in frame and 'window_start' in frame and 'as_of' in frame:
                for name, _ in SPLITS:
                    part = frame[frame['split'] == name]
                    if not part.empty:
                        periods[name] = dict(first=str(part.window_start.min()), last=str(part.as_of.max()))
        except (OSError, ValueError, KeyError):
            warning = 'بخشی از جزئیات پنجره‌های این اجرا قابل خواندن نیست.'
    split_rows = []
    for name, label in SPLITS:
        boundary = boundaries.get(name, {})
        split_rows.append(dict(name=name, label=label, count=counts.get(name), first=boundary.get('first'), last=boundary.get('last'), target_end=boundary.get('last_target')))
    shape = config.get('input_shape', stage.get('input_shape', []))
    universe = config.get('context_universe')
    assets = None if universe is None else [dict(name=symbol, target=True)] + [dict(name=n, target=False) for n in universe]
    horizons = config.get('horizons', [])
    schema = config.get('target_schema')
    hourly = schema in ('hourly_returns_v2', 'hourly_returns_v3', 'hourly_path_v1')
    return dict(available=True, warning=warning, splits=split_rows, input_period=periods.get('train'),
                train_boundary=boundaries.get('train', {}), shape=shape,
                shape_text=' × '.join(str(v) for v in shape) if len(shape) == 3 else None,
                steps=shape[1] if len(shape) == 3 else None, features=shape[2] if len(shape) == 3 else None,
                samples=shape[0] if len(shape) == 3 else None,
                train_samples=counts.get('train'), assets=assets,
                asset_count=len(assets) if assets is not None else None,
                lookback=config.get('lookback_sessions'),
                forecast_sessions=config.get('forecast_sessions', 21 if hourly else None),
                output_count=len(horizons) if horizons else None,
                hourly=hourly, schema=schema, history_start=config.get('history_start'))
