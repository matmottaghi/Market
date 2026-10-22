import json

import pytest
import requests

from market_dss.historical import HistoricalAPIError, TsetmcProvider
from market_dss.config import Settings
from market_dss.service import MarketService


class Response:
    def __init__(self, value, code=200, content_type="application/json"):
        self.value, self.status_code = value, code
        self.headers = {"Content-Type": content_type}
    def json(self):
        return self.value


class Session:
    def __init__(self):
        self.headers, self.calls = {}, []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if "Search" in url:
            return Response({"instrumentSearch": [{"insCode": "12345678901234567", "lVal18AFC": "نماد", "flow": 1}]})
        if "InstrumentInfo" in url:
            return Response({"instrumentInfo": {"insCode": "12345678901234567", "lVal18AFC": "نماد", "lVal30": "شرکت", "flow": 1}})
        if "DailyList" in url:
            return Response({"closingPriceDaily": [{"insCode": "12345678901234567", "dEven": 20240101,
                "pClosing": 100, "priceFirst": 99, "priceMin": 98, "priceMax": 101,
                "pDrCotVal": 100, "priceYesterday": 99, "qTotTran5J": 30, "qTotCap": 3000, "zTotTran": 2}]})
        if "Adjust" in url:
            return Response({"priceAdjust": []})
        return Response({"clientType": []})


def test_public_history_mapping_cache_and_automatic_excel(tmp_path, monkeypatch):
    monkeypatch.delenv("BRS_API_KEY", raising=False)
    session = Session()
    provider = TsetmcProvider(tmp_path, session=session, interval=0)
    settings = Settings(data_dir=tmp_path, model_dir=tmp_path / "models")
    service = MarketService(settings)
    first = service.update_historical("نماد", provider)
    assert first["mode"] == "backfill" and first["inserted"] == 1
    history = service.history("نماد")
    assert history.instrument_id.iloc[0] == "12345678901234567"
    assert history.close.iloc[0] == 100 and history.volume.iloc[0] == 30
    assert history.value.iloc[0] == 3000 and history.trading_date.iloc[0] == "2024-01-01"
    assert service.repository.path.exists()
    second = service.update_historical("نماد", provider)
    assert second["mode"] == "reconcile" and second["inserted"] == 0
    assert len(service.history("نماد")) == 1
    assert sum("Search" in url for url, _ in session.calls) == 1
    assert all("key=" not in url and kwargs["allow_redirects"] is False for url, kwargs in session.calls)
    assert json.loads((tmp_path / "processed/symbol_mapping.json").read_text("utf-8"))["نماد"]["market"] == "1"


@pytest.mark.parametrize("response", [Response({},403), Response({},302), Response({},200,"text/html"),
                                        Response({"closingPriceDaily":[]}), Response({"other":[]})])
def test_blocked_or_empty_responses_rejected(tmp_path, response):
    session = Session()
    session.get = lambda *a, **k: response
    provider = TsetmcProvider(tmp_path, session=session, attempts=1, interval=0)
    with pytest.raises(HistoricalAPIError):
        provider.request("ClosingPrice/GetClosingPriceDailyList/1/0", "closingPriceDaily")


def test_history_failure_preserves_workbook(tmp_path):
    service = MarketService(Settings(data_dir=tmp_path, model_dir=tmp_path / "models"))
    provider = TsetmcProvider(tmp_path, session=Session(), interval=0)
    service.update_historical("نماد", provider)
    previous = service.repository.path.read_bytes()
    provider.session.get = lambda *a, **k: Response({}, 403)
    with pytest.raises(HistoricalAPIError):
        service.update_historical("نماد", provider)
    assert service.repository.path.read_bytes() == previous


def test_ambiguous_symbol_rejected(tmp_path):
    session = Session()
    session.get = lambda *a, **k: Response({"instrumentSearch": [
        {"insCode": "1", "lVal18AFC": "نماد", "flow": 1},
        {"insCode": "2", "lVal18AFC": "نماد", "flow": 2}]})
    with pytest.raises(HistoricalAPIError, match="ambiguous"):
        TsetmcProvider(tmp_path, session=session, interval=0).resolve("نماد")


def test_timeout_retry_is_bounded(tmp_path, monkeypatch):
    session = Session()
    attempts = []
    def timeout(*a, **k):
        attempts.append(1)
        raise requests.Timeout("private response must not leak")
    session.get = timeout
    monkeypatch.setattr("market_dss.historical.time.sleep", lambda *a: None)
    with pytest.raises(HistoricalAPIError, match="Timeout") as error:
        TsetmcProvider(tmp_path, session=session, interval=0).request("Instrument/Search/x", "instrumentSearch")
    assert len(attempts) == 3 and "private response" not in str(error.value)


def test_credential_store_encrypted_and_environment_compatible(tmp_path, monkeypatch):
    import os
    from market_dss.credentials import CredentialStore
    monkeypatch.setenv("BRS_API_KEY", "fixture-env-key")
    store = CredentialStore(tmp_path)
    assert store.get() == "fixture-env-key"
    if os.name != "nt":
        pytest.skip("Windows DPAPI integration")
    store.save("fixture-local-key")
    assert store.get() == "fixture-local-key"
    assert b"fixture-local-key" not in store.path.read_bytes()
