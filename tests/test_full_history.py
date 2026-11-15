# توضیح فایل: آزمون دریافت موازی و گزارش پوشش کامل تاریخچه بدون تماس شبکه.
# معرفی توابع:
# - test_parallel_history_coverage: بررسی session مستقل workerها، ذخیرهٔ همهٔ روزها و گزارش پوشش.
# - Session.close: بستن ساختگی session در آزمون؛ Worker.__init__: ساخت worker مستقل بدون شبکه.
# - test_invalid_refresh_preserves_valid_day: حفظ روز معتبر در برابر پاسخ تازهٔ نامنطبق.
import pandas as pd
import pytest
from market_data import CsvMarketStore, sync_market
from test_market_csv import CODE, FakeClient, sample_daily, settings


def test_parallel_history_coverage(tmp_path, monkeypatch):
    cfg = settings(tmp_path)
    store = CsvMarketStore(cfg.root)
    progress = []
    clients = []

    class Session:
        def close(self):
            pass

    class Worker(FakeClient):
        def __init__(self, config):
            super().__init__(sample_daily())
            self.session = Session()
            clients.append(self)

    monkeypatch.setattr('market_data.pipeline.TsetmcClient', Worker)
    report = sync_market({'اهرم': CODE}, settings=cfg, store=store,
                         client=FakeClient(sample_daily()), max_hourly_days=0,
                         full_reconcile=True, retry_incomplete_hourly=True,
                         workers=2, progress=progress.append)
    assert report['اهرم']['coverage'] == {'matched': 3}
    assert report['اهرم']['status'] == 'ok'
    assert progress[-1]['attempted'] == progress[-1]['total'] == 3
    assert 1 <= len(clients) <= 2
    coverage = pd.read_csv(tmp_path / 'coverage' / f'{CODE}.csv')
    assert len(coverage) == 3
    assert store.load_hourly(CODE).Hour.nunique() == 3


def test_invalid_refresh_preserves_valid_day(tmp_path):
    store = CsvMarketStore(tmp_path)
    client = FakeClient(sample_daily())
    day = client.get_hourly_day('اهرم', CODE, pd.Timestamp('2026-10-10'), 60)
    store.save_hourly_day(day)
    broken = day.copy()
    broken['Volume'] = 1
    broken['DataQuality'] = 'volume_mismatch'
    with pytest.raises(ValueError, match='Refusing'):
        store.save_hourly_day(broken)
    assert store.load_hourly(CODE).Volume.sum() == day.Volume.sum()
