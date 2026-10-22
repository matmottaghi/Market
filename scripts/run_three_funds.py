"""Run actual three-fund acceptance without printing credentials or full API URLs."""
import json
from market_dss.service import MarketService
from market_dss.pipeline import MarketPipelineService
from market_dss.live import diagnose_market_api
from market_dss.storage import atomic_bytes, excel_bytes

service = MarketService()
symbols = ['اهرم', 'موج', 'توان']
report = MarketPipelineService(service).run_full_pipeline(symbols=symbols)
coverage = service.coverage(symbols)
atomic_bytes(service.settings.data_dir / 'processed' / 'historical_coverage.xlsx', excel_bytes({'coverage':coverage}))
diagnostics = diagnose_market_api('TSETMC', symbols=symbols, data_dir=service.settings.data_dir)
diagnostics += diagnose_market_api('BrsApi', credentials=service.live_key(), symbols=symbols)
atomic_bytes(service.settings.data_dir / 'processed/live_diagnostics.json', json.dumps(diagnostics, ensure_ascii=False).encode('utf-8'))
print(json.dumps({'run_id':report['run_id'], 'status':report['status'], 'counts':report['counts'],
                  'coverage':coverage.to_dict('records'), 'live_provider':report.get('collection',{}).get('provider'),
                  'failed_stages':[s for s in report['stages'] if s['status']=='FAILED']}, ensure_ascii=True, default=str))
