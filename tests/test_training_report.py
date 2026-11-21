"""Frozen training metadata, split counts and training-only report rendering."""
import json
from forecast.training_report import training_report


def metadata():
    return dict(target_schema='hourly_returns_v2',lookback_sessions=21,forecast_sessions=21,
                context_universe=['زمینه قدیمی'],input_shape=[240,84,16],horizons=list(range(1,505)),
                split_counts=dict(train=100,validation=30,calibration=20,test=40),
                split_boundaries=dict(train=dict(first='2025-02-01 09:00:00',last='2025-07-01 12:00:00',last_target='2025-07-30')),
                split_input_periods=dict(train=dict(first='2025-01-01',last='2025-07-01 12:00:00')))


def test_report_preserves_training_input_dates_and_counts(tmp_path):
    (tmp_path/'111_preprocessing.json').write_text(json.dumps(metadata()),encoding='utf8')
    report=training_report(tmp_path,'111','هدف',dict(run_id='run'))
    assert report['input_period']['first']=='2025-01-01'
    assert report['train_boundary']['first']=='2025-02-01 09:00:00'
    assert [s['count'] for s in report['splits']]==[100,30,20,40]
    assert report['shape_text']=='240 × 84 × 16'
    assert report['asset_count']==2 and report['assets'][1]['name']=='زمینه قدیمی'
    assert report['forecast_sessions']==21 and report['lookback']==21


def test_missing_metadata_does_not_invent_counts_or_assets(tmp_path):
    assert training_report(tmp_path,'111','هدف',dict(run_id='run'))['available'] is False
    run=dict(stages=[dict(stage='prepared',symbol='هدف',input_shape=[20,80,8],splits={'train':0})])
    report=training_report(tmp_path,'111','هدف',run)
    assert report['splits'][0]['count']==0
    assert report['splits'][1]['count'] is None
    assert report['assets'] is None and report['lookback'] is None


def test_invalid_metadata_has_explicit_unavailable_message(tmp_path):
    (tmp_path/'111_preprocessing.json').write_text('{',encoding='utf8')
    report=training_report(tmp_path,'111','هدف',dict(run_id='run'))
    assert not report['available'] and report['warning']
