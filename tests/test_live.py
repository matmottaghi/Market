import json
from pathlib import Path

import pytest
import requests

from market_dss.live import TsetmcLiveProvider, diagnose_market_api, freshness


class Response:
    def __init__(self, code=200, payload=None, mime="application/json"):
        self.status_code = code
        self.payload = payload
        self.headers = {"Content-Type": mime}
    def json(self):
        return self.payload


class Session:
    def __init__(self, response):
        self.headers = {}
        self.response = response
    def get(self, *a, **k):
        return self.response


@pytest.mark.parametrize("response,status", [
    (Response(401), "AUTHENTICATION_FAILED"), (Response(429), "RATE_LIMITED"),
    (Response(503), "PROVIDER_UNAVAILABLE"), (Response(200, {}, "text/html"), "INVALID_RESPONSE"),
    (Response(200, []), "EMPTY_RESPONSE"), (Response(200, {"unexpected": "private-secret"}), "INVALID_RESPONSE"),
    (Response(200, [{"id":"1", "l18":"نماد", "date":"2000-01-01", "base_id":"2",
                      "price_strike":100, "size_contract":10, "predtran":100}]), "STALE_DATA"),
])
def test_live_diagnostics_categories_and_no_secret_leak(response, status):
    rows = diagnose_market_api("BrsApi", credentials="private-secret", symbols=["نماد"], session=Session(response))
    assert rows and all(row["status"] == status for row in rows)
    assert "private-secret" not in json.dumps(rows)
    assert all("request_at" in r and "duration_seconds" in r and "endpoint" in r for r in rows)


def test_diagnostics_timeout_is_safe():
    class Broken(Session):
        def get(self, *a, **k):
            raise requests.Timeout("https://provider/?key=private-secret")
    rows = diagnose_market_api("BrsApi", "private-secret", symbols=["نماد"], session=Broken(None))
    assert all(row["status"] == "NETWORK_ERROR" for row in rows)
    assert "private-secret" not in json.dumps(rows)


class PublicProvider:
    def resolve(self, symbol):
        return {"instrument_id": "123", "symbol": symbol, "name": "صندوق اهرمی"}
    def request(self, path, key, **kwargs):
        if "DailyList" in path:
            return [{"insCode":"123", "dEven":20261021, "pClosing":100}]
        if "BestLimits" in path:
            return [{"number":1,"pMeDem":99,"pMeOf":101,"qTitMeDem":10,"qTitMeOf":20}]
        return {"insCode":"0", "dEven":20261021, "hEven":123002, "pClosing":100,
                "pDrCotVal":100, "qTotTran5J":1000, "qTotCap":100000}


def test_public_quote_missing_native_id_validates_against_daily_history(tmp_path):
    batch = TsetmcLiveProvider(tmp_path, ["اهرم"], provider=PublicProvider()).fetch()
    quote = batch.tables["snapshots"].iloc[0]
    assert quote.instrument_id == "123" and quote.provider_instrument_id == "0"
    assert quote.quote_identity_basis == "validated_request_path_and_daily_record"
    assert quote.provider_quote_at != quote.retrieved_at
    assert quote.source == "TSETMC" and not quote.orderbook_time_verified


def test_public_quote_inconsistent_id_or_price_rejected(tmp_path):
    class BadProvider(PublicProvider):
        def request(self, path, key, **kwargs):
            result = super().request(path, key, **kwargs)
            if "DailyList" in path:
                result[0]["pClosing"] = 999
            return result
    from market_dss.live import LiveDataError
    with pytest.raises(LiveDataError, match="does not match"):
        TsetmcLiveProvider(tmp_path, ["اهرم"], provider=BadProvider()).fetch()


def test_freshness_and_protected_store_failure_are_explicit(tmp_path, monkeypatch):
    from market_dss.credentials import CredentialStore
    assert freshness("2024-01-01T00:00:00Z", now="2024-01-04T00:00:00Z") == "STALE"
    monkeypatch.setenv("BRS_API_KEY", "fixture-key")
    store = CredentialStore(tmp_path)
    store.path.parent.mkdir()
    store.path.write_bytes(b"test-encrypted-placeholder")
    original = Path.read_bytes
    def denied(path):
        if path == store.path:
            raise PermissionError("access denied")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", denied)
    assert store.get() == "fixture-key"
    assert store.status()["status"] == "PROTECTED_STORE_UNAVAILABLE"
    assert "fixture-key" not in json.dumps(store.status())


def test_quote_identity_uses_trusted_stored_history_during_history_api_outage(tmp_path):
    import pandas as pd
    from market_dss.storage import ExcelRepository
    ExcelRepository(tmp_path / "historical/market.xlsx").upsert_many({"prices": pd.DataFrame([
        {"instrument_id":"123", "symbol":"اهرم", "source":"TSETMC", "trading_date":"2026-10-21",
         "retrieved_at":"2026-10-21T13:00:00Z", "close":100}])})
    class CachedProvider(PublicProvider):
        data_dir = tmp_path
        def request(self, path, key, **kwargs):
            if "DailyList" in path:
                raise requests.Timeout("History API unavailable")
            return super().request(path, key, **kwargs)
    batch = TsetmcLiveProvider(tmp_path, ["اهرم"], provider=CachedProvider()).fetch()
    assert len(batch.tables["snapshots"]) == 1
    assert batch.tables["snapshots"].quote_identity_basis.iloc[0] == "validated_request_path_and_stored_daily_record"
