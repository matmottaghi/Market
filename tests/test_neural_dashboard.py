# توضیح فایل: آزمون بازهٔ قابل تنظیم و نمودار epoch در رابط Django با دادهٔ ساختگی ذخیره‌شده.
# تابع test_neural_dashboard_confidence_and_curves تغییر بازه بدون آموزش و بازگشت امن ورودی نامعتبر را بررسی می‌کند.
# تابع داخلی capture ورودی قالب را برای بررسی عددی بازه‌ها و نمودارها ثبت می‌کند.
import json
import os
import pandas as pd


def test_neural_dashboard_confidence_and_curves(tmp_path, monkeypatch):
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    import django
    django.setup()
    from django.test import RequestFactory
    from forecast import views
    folder = tmp_path/'run'
    folder.mkdir()
    (folder/'status.json').write_text(json.dumps(dict(run_id='run', state='complete', model_kind='lstm', stages=[], errors=[])))
    pd.DataFrame([dict(model_kind='lstm', symbol='توان', frequency='hourly', horizon=1,
                       horizon_unit='trading_days', as_of='2020-01-01', current_price=100,
                       mu_log=.01, sigma_log=.1, predicted_return=.01, probability_up=.5)]).to_csv(folder/'predictions.csv', index=False)
    pd.DataFrame([dict(epoch=1, loss=.3, val_loss=.4, standardized_mae=.2, val_standardized_mae=.3)]).to_csv(folder/'11111_learning.csv', index=False)
    monkeypatch.setattr(views, 'RUNS', tmp_path)
    monkeypatch.setattr(views, 'instruments', lambda: {'توان':'11111'})
    class Store:
        def load_daily(self, code): return pd.DataFrame()
    monkeypatch.setattr(views, 'settings_store', lambda: (None, Store()))
    captured = []
    from django.shortcuts import render

    def capture(request, template, context):
        captured.append(context)
        return render(request, template, context)

    monkeypatch.setattr(views, 'render', capture)
    factory = RequestFactory()
    assert views.dashboard(factory.get('/?confidence=80')).status_code == 200
    assert views.dashboard(factory.get('/?confidence=95')).status_code == 200
    narrow, wide = captured[0]['predictions'][0], captured[1]['predictions'][0]
    assert wide['lower_price'] < narrow['lower_price']
    assert wide['upper_price'] > narrow['upper_price']
    assert wide['predicted_price'] == narrow['predicted_price']
    assert len(captured[1]['learning_charts']) == 2
    assert views.dashboard(factory.get('/?confidence=NaN')).status_code == 200
    assert captured[-1]['confidence_error']
