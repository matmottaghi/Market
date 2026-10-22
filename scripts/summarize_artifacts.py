"""Summarize saved artifacts without loading model weights or using API credentials."""
import json
from market_dss.service import MarketService
from market_dss.forecasting import load_model_artifacts, Forecast
from market_dss.storage import atomic_bytes

service = MarketService()
summary = {'coverage':service.coverage().to_dict('records'), 'models':[], 'quotes':[]}
for symbol in ['اهرم','موج','توان']:
    history=service.history(symbol)
    if history.empty:
        continue
    marker=service.models(str(history.instrument_id.iloc[-1]))[0]
    artifact=load_model_artifacts(marker.parent)
    metadata=artifact['metadata']
    summary['models'].append({'symbol':symbol,'directory':str(marker.parent), 'metadata':metadata,
        'metrics':artifact['metrics'].to_dict('records'), 'epoch_rows':len(artifact['training']),
        'minimum_learning_rate':float(artifact['training'].learning_rate.min()) if 'learning_rate' in artifact['training'] else None})
    quote=service.latest_quote(symbol)
    if quote:
        summary['quotes'].append({k:quote.get(k) for k in ['symbol','source','close','last','provider_quote_at','retrieved_at','freshness','quote_identity_basis']})
    for horizon in [1,5,21]:
        path=service.settings.data_dir / 'forecasts' / f'{history.instrument_id.iloc[-1]}_{horizon}.xlsx'
        if path.exists():
            Forecast.from_excel(path).export(path)
atomic_bytes(service.settings.data_dir/'processed/final_artifact_summary.json',json.dumps(summary,ensure_ascii=False,default=str).encode('utf-8'))
print(json.dumps({'coverage':summary['coverage'], 'training':[{'symbol':m['symbol'],'version':m['metadata']['version'],
    'epochs':m['epoch_rows'],'best_epoch':m['metadata'].get('best_epoch'), 'min_lr':m['minimum_learning_rate'],
    'train_samples':m['metadata'].get('partitions',{}).get('train',{}).get('samples')} for m in summary['models']],
    'quote_count':len(summary['quotes'])},ensure_ascii=True))
