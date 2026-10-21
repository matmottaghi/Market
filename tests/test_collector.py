import pandas as pd

from leveraged_funds_collector import LeveragedFundsCollector, MarketDataError
from market_dss.ingestion import CollectorAdapter


def test_existing_collector_workbook_and_dynamic_discovery(tmp_path):
    class FixtureCollector(LeveragedFundsCollector):
        def _get(self, endpoint, **params):
            if endpoint == "AllSymbols.php":
                return [
                    {
                        "id": "17914401175772326",
                        "l18": "TEST_NEW_FUND",
                        "l30": "صندوق اهرمی آزمایشی",
                        "pc": 100,
                    }
                ]
            if endpoint == "Symbol.php":
                return [{"id": "17914401175772326", "date": "1405-07-29"}]
            if endpoint == "Nav.php":
                raise MarketDataError("Nav.php: HTTP 503")
            if endpoint == "History.php":
                return [{"date": "1405-07-28", "pc": 99, "tvol": 20}]
            if endpoint == "Option.php":
                return []
            raise AssertionError(endpoint)

    collector = FixtureCollector(api_key="synthetic_test_key", output_dir=tmp_path, tickers=[])
    result = collector.collect_daily(include_history=True)
    assert result.file_path.exists()
    assert len(result.sheets["Funds"]) == 1
    assert result.errors[0]["endpoint"] == "NAV"
    sheets = pd.read_excel(result.file_path, sheet_name=None, dtype=str)
    assert set(sheets) == {
        "Funds",
        "Details",
        "NAV",
        "Options",
        "PriceHistory",
        "ClientHistory",
        "Assemblies",
        "Metadata",
        "Errors",
    }
    assert sheets["Funds"].id.iloc[0] == "17914401175772326"
    batch = CollectorAdapter().read(result.file_path)
    assert len(batch.tables["prices"]) == 2
