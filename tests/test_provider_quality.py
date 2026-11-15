# توضیح فایل: آزمون کیفیت معاملات TSETMC و جلوگیری از دوباره‌شماری حجم.
# معرفی توابع:
# - make_client: آماده‌سازی پاسخ ساختگی شبکه.
# - test_duplicate_and_cancelled_trades: بررسی حذف تکرار و معاملهٔ لغوشده و تطابق حجم.
# - test_conflicting_trade_sequence_fails: بررسی رد دو رکورد متعارض با شمارهٔ معاملهٔ یکسان.
# - test_partial_responses_merge_by_trade_id: تکمیل پاسخ‌های ناقص با شناسهٔ معامله بدون دوباره‌شماری.
import pandas as pd
import pytest
from market_data import TsetmcClient
from test_market_csv import settings


def make_client(tmp_path, rows):
    client = TsetmcClient(settings(tmp_path))
    client._json = lambda endpoint: {'tradeHistory': rows}
    return client


def test_duplicate_and_cancelled_trades(tmp_path):
    trade = dict(nTran=1, hEven=93000, pTran=100, qTitTran=10, canceled=0)
    cancelled = dict(nTran=2, hEven=93100, pTran=200, qTitTran=20, canceled=1)
    client = make_client(tmp_path, [trade, trade.copy(), cancelled])
    result = client.get_hourly_day('اهرم', '17914401175772326', pd.Timestamp('2026-10-10'), 10)
    assert result.Volume.sum() == 10
    assert result.Trades.sum() == 1
    assert result.DataQuality.iloc[0] == 'matched'
    client.session.close()


def test_conflicting_trade_sequence_fails(tmp_path):
    first = dict(nTran=1, hEven=93000, pTran=100, qTitTran=10)
    second = dict(nTran=1, hEven=93000, pTran=100, qTitTran=20)
    client = make_client(tmp_path, [first, second])
    with pytest.raises(ValueError, match='Conflicting'):
        client.get_hourly_day('اهرم', '17914401175772326', pd.Timestamp('2026-10-10'), 30)
    client.session.close()


def test_partial_responses_merge_by_trade_id(tmp_path):
    first = dict(nTran=1, hEven=93000, pTran=100, qTitTran=10)
    second = dict(nTran=2, hEven=93100, pTran=110, qTitTran=20)
    responses = iter([{'tradeHistory': [first]}, {'tradeHistory': [second, first]}])
    client = make_client(tmp_path, [])
    client._json = lambda endpoint: next(responses)
    result = client.get_hourly_day('اهرم', '17914401175772326', pd.Timestamp('2026-10-10'), 30)
    assert result.Volume.sum() == 30
    assert result.Trades.sum() == 2
    assert result.DataQuality.iloc[0] == 'matched'
    assert result.ResponseParts.iloc[0] == 2
    client.session.close()
