# توضیح فایل: آزمون خارج از آموزش با مبناهای بدون نشت و بازنمونه‌گیری بلوکی ۲۱ جلسه؛ کیفیت معاملات آپشن را ادعا نمی‌کند.
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from forecast.pipeline import RUNS
from forecast.table_io import read_table,table_path
root=RUNS/(sys.argv[1] if len(sys.argv)>1 else sorted(p.name for p in RUNS.glob('*-nn-*') if table_path(p/'metrics.json').exists())[-1])
metrics=read_table(root/'metrics.json',dtype={'insCode':str});results=[]
rng=np.random.default_rng(42)
for symbol,group in metrics.groupby('symbol'):
 for day in (1,5,21):
  rows=group[group.target_day==day]
  if rows.empty:continue
  row=rows.sort_values('target_hour').iloc[-1]
  history=read_table(root/f'{row.insCode}_hourly_{int(row.horizon)}_test.json')
  delta=abs(history.actual.to_numpy())-abs(history.actual.to_numpy()-history.predicted.to_numpy())
  n=len(delta);block=min(21,n);bootstrap=[]
  for _ in range(1000):
   starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
   indices=np.concatenate([np.arange(s,s+block) for s in starts])[:n]
   bootstrap.append(float(delta[indices].mean()))
  z=(np.log1p(history.actual.to_numpy())-history.mu_log.to_numpy())/history.sigma_log.to_numpy()
  result=dict(symbol=symbol,day=day,hour=int(row.target_hour),test_rows=n,first=str(history.timestamp.iloc[0]),last=str(history.timestamp.iloc[-1]),mae=float(row.mae),rmse=float(row.rmse),direction_accuracy=float(row.direction_accuracy),baseline_direction=float(row.get('baseline_direction_accuracy',np.nan)),brier=float(row.brier),baseline_brier=float(row.baseline_brier),coverage90=float(row.coverage90),coverage95=float(np.mean(abs(z)<=1.959963984540054)),zero_mae=float(row.baseline_zero_mae),drift_mae=float(row.baseline_drift_mae),mae_improvement_ci95=np.quantile(bootstrap,[.025,.975]).tolist())
  results.append(result)
report=dict(run=root.name,scope='held-out price/return predictions, no option trading backtest',bootstrap='moving blocks of 21 sessions, 1000 resamples; overlapping targets are dependent',rows=results)
(root/'benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
lines=['<!-- توضیح فایل: معیار آزمون خارج از آموزش و عدم‌قطعیت مقایسه با مبنا؛ خروجی benchmark_hourly.py و فاقد تابع اجرایی. -->','# آزمون مدل ساعتی','',f'اجرا: {root.name}','','تقسیم زمانی مستقل؛ معیارها بدون هزینهٔ معامله‌اند. بازهٔ بهبود MAE از bootstrap بلوکی ۲۱ جلسه است و با نمونهٔ کوچک قطعیت ندارد. هدف‌های متوالی همپوشانی دارند؛ این آزمون بک‌تست آپشن نیست.','','| نماد | روز / ساعت | نمونه | دقت جهت | جهت ثابت | MAE | MAE صفر | پوشش ۹۰٪ | بهبود MAE: بازه۹۵٪ (واحد درصد) |','|---|---|---:|---:|---:|---:|---:|---:|---|']
for r in results:
 ci=r['mae_improvement_ci95']
 lines.append(f"| {r['symbol']} | {r['day']} / {r['hour']} | {r['test_rows']} | {r['direction_accuracy']:.1%} | {r['baseline_direction']:.1%} | {r['mae']:.2%} | {r['zero_mae']:.2%} | {r['coverage90']:.1%} | [{100*ci[0]:+.3f}, {100*ci[1]:+.3f}] |")
lines+=['','مبنای جهت و احتمال فقط از آموزش آموخته شده است. بهبود MAE مثبت یعنی مدل از بازده صفر بهتر است؛ بازهٔ شامل صفر، برتری روشن را نشان نمی‌دهد. Brier، مبنای میانگین بازده، پوشش۹۵٪ و تاریخ نمونه‌ها در benchmark.json موجودند.','', 'موج در این تقسیم بهتر از مبناست؛ برای توان کیفیت جهت ضعیف است و اهرم در جلسه۲۱ از جهت ثابت ضعیف‌تر است. پیشنهاد: تعدیل قیمت، walk-forward، آزمایش حذف/افزودن زمینه و سنجش کالیبراسیون در رژیم‌های مختلف پیش از توسعهٔ بیشتر اندازهٔ شبکه.']
(root/'BENCHMARK_FA.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps(report,ensure_ascii=True,indent=2))
