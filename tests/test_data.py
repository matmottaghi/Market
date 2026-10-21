import pandas as pd
import pytest

from market_dss.calendar import TradingCalendar, parse_date
from market_dss.features import engineer_features
from market_dss.ingestion import CollectorAdapter, DataProvider, update_market_data
from market_dss.storage import ExcelRepository, atomic_bytes, excel_bytes


def test_excel_incremental_dedup_backup_and_shards(tmp_path, synthetic_history):
    repo = ExcelRepository(tmp_path / "market.xlsx", max_rows=3)
    first = synthetic_history.head(5)
    repo.upsert_many({"prices": first})
    repo.upsert_many({"prices": first.iloc[2:]})
    assert len(repo.read("prices")) == 5
    assert len([s for s in pd.ExcelFile(repo.path).sheet_names if s.startswith("prices__")]) == 2
    assert repo.path.with_suffix(".xlsx.bak").exists()
    assert repo.read("prices").instrument_id.iloc[0] == "synthetic-underlying"


def test_failure_preserves_valid_excel(tmp_path, synthetic_history):
    class BrokenProvider(DataProvider):
        name = "broken"

        def fetch(self, include_history=False):
            raise ConnectionError("Unavailable")

    repo = ExcelRepository(tmp_path / "market.xlsx")
    repo.upsert_many({"prices": synthetic_history.head(3)})
    before = repo.path.read_bytes()
    with pytest.raises(ConnectionError):
        update_market_data(repo, BrokenProvider(), tmp_path / "raw")
    assert repo.path.read_bytes() == before


def test_bad_batch_is_atomic(tmp_path, synthetic_history):
    repo = ExcelRepository(tmp_path / "market.xlsx")
    repo.upsert_many({"prices": synthetic_history.head(3)})
    before = repo.path.read_bytes()
    with pytest.raises(ValueError):
        repo.upsert_many({"prices": synthetic_history.tail(2), "options": pd.DataFrame([{"bad": 1}])})
    assert repo.path.read_bytes() == before


def test_supplied_collector_schema_and_unknown_option_date():
    frames = {
        "Funds": pd.DataFrame(
            [
                {
                    "id": "17914401175772326",
                    "symbol": "TEST",
                    "pc": 100,
                    "market_date": "1405/07/29",
                    "snapshot_ts": "2026-10-21T16:00:00+03:30",
                }
            ]
        ),
        "PriceHistory": pd.DataFrame([{"symbol": "TEST", "date": "1405/07/28", "pc": 99, "tvol": 5}]),
        "Options": pd.DataFrame(
            [
                {
                    "id": "2839111977611376",
                    "l18": "TEST_CALL",
                    "base_l18": "TEST",
                    "base_id": "17914401175772326",
                    "type": "call",
                    "date_end": "1405/09/10",
                    "price_strike": 100,
                    "size_contract": 10,
                    "snapshot_date": "2026-10-21",
                }
            ]
        ),
    }
    result = CollectorAdapter().adapt(frames)
    assert len(result.tables["prices"]) == 2
    assert result.tables["prices"].instrument_id.iloc[1] == "17914401175772326"
    assert "trading_date" not in result.tables["options"]
    assert result.issues


def test_dates_and_calendar():
    assert str(parse_date("۱۴۰۳/۰۱/۰۱")) == "2024-03-20"
    assert str(parse_date(20240320)) == "2024-03-20"
    calendar = TradingCalendar({parse_date("2024-03-23")})
    assert str(calendar.next_sessions("2024-03-20", 1)[0]) == "2024-03-24"


def test_features_do_not_use_future(synthetic_history):
    first = engineer_features(synthetic_history.iloc[:200])
    all_data = engineer_features(synthetic_history)
    pd.testing.assert_frame_equal(first, all_data.iloc[:200].reset_index(drop=True))


def test_excel_formula_text_and_long_identifier(tmp_path):
    target = tmp_path / "safe.xlsx"
    atomic_bytes(target, excel_bytes({"data": pd.DataFrame([{"id": "17914401175772326", "symbol": "=1+1"}])}))
    from openpyxl import load_workbook

    sheet = load_workbook(target).active
    assert sheet["B2"].data_type == "s"
    assert sheet["A2"].value == "17914401175772326"


def test_reject_partial_adjustments(synthetic_history):
    history = synthetic_history.copy()
    history["adjusted_close"] = history.close
    history.loc[5, "adjusted_close"] = float("nan")
    with pytest.raises(ValueError, match="incomplete"):
        engineer_features(history)


def test_price_revisions_preserve_point_in_time_vintage(tmp_path, synthetic_history):
    from market_dss.ingestion import load_historical_data

    repo = ExcelRepository(tmp_path / "market.xlsx")
    first = synthetic_history.head(1).copy()
    repo.upsert_many({"prices": first})
    revised = first.copy()
    revised["close"] = first.close * 2
    revised["retrieved_at"] = "2099-01-01T00:00:00Z"
    revised["available_at"] = "2099-01-01T00:00:00Z"
    repo.upsert_many({"prices": revised})
    historic = load_historical_data(repo, "TEST_ONLY", "test_fixture", "2025-01-01")
    assert historic.close.iloc[0] == pytest.approx(first.close.iloc[0])
    assert len(repo.read("price_versions")) == 2
