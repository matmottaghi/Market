# توضیح فایل: آزمون هزینه و ضریب قرارداد، سود سررسید، امید سود تحلیلی و جلوگیری از قیمت جعلی/سررسید ناسازگار.
import numpy as np
import pytest
from scipy.stats import norm
from forecast.options_math import evaluate_strategy,template_legs

def contracts():
    return [dict(code='call',kind='call',strike=100,expiry='2026-12-01',multiplier=1000,bid=9,ask=10,bid_volume=10,ask_volume=10),dict(code='put',kind='put',strike=100,expiry='2026-12-01',multiplier=1000,bid=9,ask=10,bid_volume=10,ask_volume=10)]

def test_long_call_expected_profit():
    result=evaluate_strategy([dict(contract='call',quantity=1)],contracts(),100,'2026-12-01',fees=100,distribution=(100,0,.2),capital=10100)
    mean=100*np.exp(.02)
    expected=(mean*norm.cdf(.2)-100*.5)*1000-10100
    assert result['entry_debit']==10100
    assert result['expected_profit']==pytest.approx(expected)
    assert result['expected_profit']!=pytest.approx(result['profit_at_expected_price'])
    assert not result['unbounded_upside_loss']
    assert result['expected_roi']==pytest.approx(expected/10100)

def test_short_call_and_missing_quote():
    result=evaluate_strategy([dict(contract='call',quantity=-1)],contracts(),100,'2026-12-01')
    assert result['entry_credit']==9000 and result['unbounded_upside_loss']
    broken=contracts();broken[0]['ask']=0
    with pytest.raises(ValueError,match='price'):evaluate_strategy([dict(contract='call',quantity=1)],broken,100,'2026-12-01')

def test_expiry_and_volume_guards():
    with pytest.raises(ValueError,match='expires before'):evaluate_strategy([dict(contract='call',quantity=1)],contracts(),100,'2026-12-02')
    with pytest.raises(ValueError,match='IV'):evaluate_strategy([dict(contract='call',quantity=1)],contracts(),100,'2026-11-30')
    with pytest.raises(ValueError,match='volume'):evaluate_strategy([dict(contract='call',quantity=20)],contracts(),100,'2026-12-01')

def test_covered_call_template():
    legs=template_legs('covered_call',contracts(),100,'2026-12-01')
    assert legs[0]['quantity']==1000
    legs[0]['price']=100
    result=evaluate_strategy(legs,contracts(),100,'2026-12-01')
    assert not result['unbounded_upside_loss']
