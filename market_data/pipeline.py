# توضیح فایل: همگام‌سازی افزایشی تاریخچه؛ دریافت کنترل‌شدهٔ موازی و گزارش پوشش روزبه‌روز.
# معرفی توابع و متدها:
# - resolve_symbols: یافتن شناسهٔ دقیق هر نماد از منبع.
# - sync_market: قفل‌گذاری و همگام‌سازی دادهٔ روزانه و ساعتی.
# - _run_locked: دریافت و ذخیرهٔ تاریخچه زیر قفل و تولید گزارش پوشش.
# - fetch: دریافت یک روز ساعتی با session مستقل برای هر worker؛ بازگرداندن خطا بدون توقف کل اجرا.
# - persist: ثبت نتیجهٔ یک روز در نخ اصلی و به‌روزرسانی شمارنده‌ها و خطاها.
# - load_for_model: خواندن پارتیشن‌های روزانه و پاکسازی کل پنجرهٔ زمانی برای مدل.

"""One callable incremental sync for CLI, Streamlit or scheduled jobs."""
from __future__ import annotations
from collections.abc import Mapping
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local
import logging
import pandas as pd
from filelock import FileLock, Timeout

from .cleaning import clean_stock_data
from .config import Settings
from .csv_store import CsvMarketStore
from .sqlite_store import make_store
from .csv_store import _csv_atomic
import json
from .provider import TsetmcClient
from .history_scope import history_start,archive_status
from datetime import datetime,timezone

LOG = logging.getLogger(__name__)


def resolve_symbols(
    symbols: list[str] | tuple[str, ...],
    *,
    client: TsetmcClient,
) -> dict[str, str]:
    return {symbol: client.resolve_symbol_code(symbol) for symbol in symbols}


def sync_market(
    instruments: Mapping[str, str],
    *,
    settings: Settings | None = None,
    client: TsetmcClient | None = None,
    store: CsvMarketStore | None = None,
    full_reconcile: bool = False,
    retry_incomplete_hourly: bool = False,
    max_hourly_days: int | None = None,
    workers: int = 1,
    progress=None,
    hourly_start=None,
) -> dict[str, dict]:
    """Incrementally append/reconcile SQLite daily and hourly series.

    `full_reconcile` revisits *all* daily rows (for historical corrections), but
    normal updates revisit just `MARKET_RECONCILE_DAYS` calendar days.
    Normal updates request a bounded recent daily list; SQLite remains the canonical store.
    """
    settings = settings or Settings.from_env()
    client = client or TsetmcClient(settings)
    store = store or make_store(settings.root)
    if max_hourly_days is None:
        max_hourly_days = settings.hourly_max_days
    if max_hourly_days < 0:
        raise ValueError("max_hourly_days must be nonnegative")
    if not 1 <= workers <= 8:
        raise ValueError('workers must be between 1 and 8')

    settings.root.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(settings.root / ".market_sync.lock"), timeout=0):
            return _run_locked(instruments, settings, client, store, full_reconcile,
                               retry_incomplete_hourly, max_hourly_days, workers, progress, hourly_start)
    except Timeout as exc:
        raise RuntimeError("Another market sync is already running") from exc


def _run_locked(instruments, settings, client, store, full_reconcile,
                retry_incomplete_hourly, max_hourly_days, workers, progress, hourly_start):
    reports = {}
    floor=hourly_start or history_start(store,settings)
    for symbol, ins_code in instruments.items():
        ins_code = str(ins_code)
        state=store.read_state(ins_code)
        persisted_max=state.get('last_daily_date')
        since=(pd.Timestamp(persisted_max)-timedelta(days=settings.reconcile_days)) if persisted_max and not full_reconcile else pd.Timestamp(floor) if floor else None
        if floor and since is not None:since=max(since,pd.Timestamp(floor))
        raw=client.get_daily_since(symbol,ins_code,since) if since is not None and hasattr(client,'get_daily_since') else client.get_daily(symbol,ins_code)
        if floor and not raw.empty:raw=raw[raw.Date>=pd.Timestamp(floor)].copy()
        if raw.empty:
            reports[symbol] = {"status": "empty_daily_response", "insCode": ins_code}
            continue
        clean, model = clean_stock_data(raw)
        if clean.empty:
            reports[symbol] = {"status": "no_valid_daily_dates", "insCode": ins_code}
            continue

        state = store.read_state(ins_code)
        persisted_max = state.get("last_daily_date")
        if persisted_max and not full_reconcile:
            cutoff = pd.Timestamp(persisted_max) - timedelta(days=settings.reconcile_days)
            changed = clean.loc[clean["Date"] >= cutoff]
        else:
            changed = clean
        daily_stats = store.upsert_daily(changed)
        current_max = clean["Date"].max().strftime("%Y-%m-%d")
        # Advance the checkpoint only after successful persistence.
        store.update_state(
            ins_code,
            symbol=symbol,
            last_daily_date=max(persisted_max or current_max, current_max),
        )

        # A transient shortened API response must not hide already known dates.
        _, model = clean_stock_data(store.load_daily(ins_code,start=floor))

        hourly_dates = model[["Date", "qTotTran5J"]].drop_duplicates("Date").sort_values("Date", ascending=False)
        if floor:
            hourly_dates=hourly_dates[hourly_dates.Date>=pd.Timestamp(floor)]
        qualities,failures=archive_status(store,ins_code) if hasattr(store,'connect') else ({},{})
        outstanding = []
        waiting_retry = 0
        for row in hourly_dates.itertuples(index=False):
            # A matched intraday snapshot can still gain new hours later today.
            refresh_today = (pd.Timestamp(row.Date).date() == pd.Timestamp.now(tz='Asia/Tehran').date()
                             and store.hourly_exists(ins_code, row.Date))
            key=str(pd.Timestamp(row.Date).date())
            if hasattr(store,'connect'):
                failure=failures.get(key)
                due=not failure or datetime.now(timezone.utc)>=datetime.fromisoformat(failure[1])
                fetch_needed=qualities.get(key)!='matched' and (retry_incomplete_hourly or due)
                if qualities.get(key)!='matched' and not (retry_incomplete_hourly or due):waiting_retry+=1
            else:fetch_needed=store.should_fetch_hourly(ins_code,row.Date,retry_incomplete=retry_incomplete_hourly)
            if refresh_today or fetch_needed:
                outstanding.append((row.Date, row.qTotTran5J))
        pending_total = len(outstanding)
        if max_hourly_days:
            outstanding = outstanding[:max_hourly_days]

        processed, errors, mismatches = 0, [], 0
        thread_state = local()
        worker_clients = []
        def fetch(item):
            trading_date, daily_volume = item
            try:
                if workers == 1:
                    worker_client = client
                else:
                    if not hasattr(thread_state, 'client'):
                        thread_state.client = TsetmcClient(settings)
                        worker_clients.append(thread_state.client)
                    worker_client = thread_state.client
                return trading_date, worker_client.get_hourly_day(symbol, ins_code, trading_date, daily_volume), None
            except Exception as exc:
                return trading_date, None, str(exc)

        def persist(result):
            nonlocal processed, mismatches
            trading_date, hourly, error = result
            try:
                if error:
                    raise ValueError(error)
                if hourly.empty:
                    store.record_hourly_failure(ins_code, trading_date, "empty tradeHistory")
                    errors.append({"date": pd.Timestamp(trading_date).strftime("%Y-%m-%d"), "error": "empty tradeHistory"})
                else:
                    store.save_hourly_day(hourly)
                    processed += 1
                    if hourly["DataQuality"].iloc[0] != "matched":
                        mismatches += 1
            except (ValueError, KeyError, OSError, ConnectionError) as exc:
                store.record_hourly_failure(ins_code, trading_date, str(exc))
                errors.append({"date": pd.Timestamp(trading_date).strftime("%Y-%m-%d"), "error": str(exc)})
                LOG.warning("Failed hourly %s %s: %s", symbol, trading_date, exc)
            except Exception as exc:
                # Preserve the daily download and continue other days/symbols.
                store.record_hourly_failure(ins_code, trading_date, str(exc))
                errors.append({"date": pd.Timestamp(trading_date).strftime("%Y-%m-%d"), "error": str(exc)})
                LOG.exception("Failed hourly %s %s", symbol, trading_date)
            if progress:
                progress(dict(total=len(outstanding), saved=processed, failed=len(errors), mismatched=mismatches,
                              attempted=processed+len(errors), date=str(pd.Timestamp(trading_date).date())))

        if progress:
            progress(dict(total=len(outstanding), saved=0, failed=0, attempted=0))
        try:
            if workers == 1:
                for item in outstanding:
                    persist(fetch(item))
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(fetch, item) for item in outstanding]
                    for future in as_completed(futures):
                        persist(future.result())
        finally:
            for worker_client in worker_clients:
                worker_client.session.close()

        coverage=[]
        if hasattr(store,'connect'):qualities,failures=archive_status(store,ins_code)
        for row in hourly_dates.itertuples(index=False):
            key=str(pd.Timestamp(row.Date).date())
            if hasattr(store,'connect'):
                quality=qualities.get(key,'missing');reason=failures.get(key,('',))[0]
            else:
                meta=store.hourly_path(ins_code,row.Date).with_suffix('.meta.json')
                quality=json.loads(meta.read_text(encoding='utf8')).get('data_quality','unverified') if meta.exists() else 'missing'
                reason=''
            coverage.append(dict(Symbol=symbol,insCode=ins_code,Date=row.Date,quality=quality,error=reason))
        coverage_df=pd.DataFrame(coverage)
        if not hasattr(store,'connect'):_csv_atomic(coverage_df,settings.root/'coverage'/f'{ins_code}.csv')
        counts = coverage_df.quality.value_counts().to_dict() if not coverage_df.empty else {}
        reports[symbol] = {
            "status": "ok" if not errors and counts.get('matched', 0) == len(hourly_dates) else "partial_success",
            "coverage": {str(key): int(value) for key, value in counts.items()},
            "expected_hourly_days": len(hourly_dates),
            "insCode": ins_code,
            "api_daily_rows": len(raw),
            "history_start": floor,
            "history_policy": "explicit_start" if floor else "per_instrument_full_archive",
            "waiting_retry_days": waiting_retry,
            "daily_request_since": str(since) if since is not None else None,
            "eligible_model_rows": len(model),
            "daily": daily_stats,
            "hourly_days_saved": processed,
            "hourly_mismatched_days": mismatches,
            "hourly_pending_candidates": pending_total - processed,
            "errors": errors,
        }
    return reports


def load_for_model(
    ins_code: str,
    *,
    store: CsvMarketStore,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load only the desired partitions; calculate returns on the full selected window.

    To compute the first return in a narrow window, include the prior
    trading observation in start, then trim after feature engineering.
    """
    daily = store.load_daily(ins_code, start=start, end=end)
    if daily.empty:
        return daily
    _, model = clean_stock_data(daily)
    return model
