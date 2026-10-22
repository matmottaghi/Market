# Iranian Market Decision Support System

راهنمای فارسی نصب، معماری، آدرس صفحات و بررسی نتایج: [راهنمای سیستم](docs/LOCAL_GUIDE_FA.md).

اجرای ویندوز: `Start-Market.cmd -Install` برای نصب اولیه؛ سپس `Start-Market.cmd`. وضعیت واقعی این سیستم: [گزارش اعتبارسنجی](docs/VALIDATION_REPORT.md).

ناوبری هفت صفحه در بالای داشبورد قرار دارد. در «تنظیمات و داده‌ها»، نمادها و کلید اختیاری BrsApi را تنظیم کنید و «اجرای کامل پایپلاین» را بزنید. تاریخچه TSETMC بدون کلید دریافت و اکسل خودکار ساخته می‌شود؛ مدل اولیه و بازآموزی دوره‌ای خودکار است. ارزیابی، منحنی یادگیری و مدیریت مدل فقط در صفحه «پیش‌بینی» و چهار تب اختصاصی آن نمایش داده می‌شوند.

در بررسی واقعی همین سیستم، ۲۶۰۷ مشاهده معتبر برای اهرم، موج و توان ذخیره شد. سه مدل با منحنی یادگیری واقعی و ۹ پیش‌بینی آماده‌اند. قیمت جاری عمومی هر سه صندوق از TSETMC دریافت شده؛ BrsApi احراز هویت‌شده، NAV و اختیار هنوز در این محیط تأیید نشده‌اند. ۶۹ تست خودکار و Ruff موفق‌اند. جزئیات تست‌ها و محدودیت‌ها در گزارش اعتبارسنجی آمده است.

راهنماهای تکمیلی: [تشخیص API](docs/LIVE_API_DIAGNOSTICS.md)، [پوشش تاریخچه](docs/HISTORICAL_COVERAGE.md)، [آموزش و ارزیابی LSTM](docs/LSTM_TRAINING_VALIDATION.md)، [فونت و چیدمان](docs/FRONTEND_TYPOGRAPHY.md).

```powershell
.\.venv\Scripts\python.exe -m market_dss pipeline --symbols 'اهرم' 'موج' 'توان'
# Public historical workflow only:
.\.venv\Scripts\python.exe -m market_dss pipeline --symbols 'اهرم' 'موج' 'توان' --no-live
```

A runnable Python/Streamlit DSS for Iranian stocks, ETFs and listed options. The existing Django starter (`manage.py`, `Stocks/`, `DataAnalysis/`) remains independent. No external database or order execution is required.

The seven Persian RTL pages cover market history, probabilistic forecasts, option chains, multi-leg strategies, forecast-based strategy ranking, backtesting and data management. All money is stored in **rial**; the dashboard optionally displays toman (10 rial).

## Install and run (PowerShell, Python 3.11+)

```powershell
cd E:\Work\Market
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env and set your own BRS_API_KEY if live collection is needed.
.\.venv\Scripts\python.exe -m market_dss init
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`. A fresh installation has no seeded market observations: **no demo prices, contracts or performance metrics are mixed with market observations**. Tests generate synthetic fixtures only inside temporary test directories. A CPU PyTorch installation is sufficient; a compatible CUDA build can be installed separately when needed.

`requirements-lock.txt` is an inherited environment snapshot, not evidence of installation on this workstation. Current installation and verification limits are recorded in `docs/VALIDATION_REPORT.md`.

## Connect the supplied collector

`leveraged_funds_collector.py` preserves the supplied `LeveragedFundsCollector`, `collect_daily()`, `CollectionResult`, workbook names and nine output sheets. The retry delay is changed to exponential backoff. The DSS adds a process lock and archives every collected workbook before canonical storage updates. The collector also remains runnable by itself:

```powershell
$env:BRS_API_KEY = 'YOUR_PERSONAL_KEY'
.\.venv\Scripts\python.exe -m market_dss update --history
# Ordinary stocks / other ETFs use the same documented provider fields:
.\.venv\Scripts\python.exe -m market_dss update --history --symbols 'فولاد' 'اهرم'
```

Without `--symbols`, the supplied ticker seeds and dynamic fund-name discovery are used. New funds whose names identify them as leveraged can be discovered; a seed list is not the sole discovery mechanism. The original standalone collector does not automatically read `.env`; the DSS commands do.

Import an existing workbook without credentials:

```powershell
.\.venv\Scripts\python.exe -m market_dss inspect 'C:\path\leveraged_market.xlsx'
.\.venv\Scripts\python.exe -m market_dss import 'C:\path\leveraged_market.xlsx'
.\.venv\Scripts\python.exe -m market_dss validate
.\.venv\Scripts\python.exe -m market_dss export 'data\exports\market.xlsx'
```

The dashboard offers the same import, refresh, validation and export actions, plus a blank input workbook. Canonical columns are accepted alongside the supplied collector columns. Alternate names can be mapped with `import ... --mapping mapping.json`, e.g. `{"PriceHistory":{"ClosingPrice":"close","InsCode":"instrument_id"}}`. Read [the storage contract](docs/data-storage.md) before importing other sources.

## Train, forecast and validate

```powershell
.\.venv\Scripts\python.exe -m market_dss train 'اهرم' --horizon 21 --window 30 --epochs 60
.\.venv\Scripts\python.exe -m market_dss train all --horizon 21 --epochs 60
.\.venv\Scripts\python.exe -m market_dss predict 'اهرم' --horizon 5 --paths 2000
.\.venv\Scripts\python.exe -m market_dss walk-forward 'اهرم' --horizon 5 --initial 300 --step 21
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check market_dss tests app.py leveraged_funds_collector.py
```

`train` also accepts `--attention` and `--device cuda`. Training requires at least 180 complete feature rows, plus enough observations for each purged horizon partition; a 21-session model normally needs several hundred observations. Insufficient data produces an explicit error. Versions live in `models/<instrument_id>/<version>/`, with tensor weights, configuration/scaler/residual metadata, `evaluation.xlsx` and `training_history.xlsx`. Only locally trusted checkpoints should be used; weights are loaded with PyTorch's restricted `weights_only=True` mode.

The LSTM predicts a vector of future log returns. Training, early-stopping validation, residual calibration and test partitions occupy 60/10/15/15 percent of feature rows, with targets purged across boundaries. The scaler and return normalization are fit only on training rows. Dropout draws and held-out residual vectors produce coherent positive price paths. Returned metadata identifies the instrument, source, cutoff, model version, random seed, horizon and generation time.

The report includes MAE, RMSE, directional accuracy, quantile loss, CRPS, 90% interval coverage/width and empirical quantile calibration. Persistence and a historical drift/bootstrap baseline are evaluated on the same test origins. Normal/Student-t residual diagnostics are descriptive; the predictor uses empirical residuals regardless of those diagnostics. **No live model performance has been established until real history is imported and evaluated.**

## Forecast → options → decisions

1. Import/collect a real history and option chain, then inspect the data-quality report.
2. Train a model and generate its physical price distribution on the forecasting page.
3. Supply verified contract terms and quote dates where the provider omits them.
4. Select contracts of one underlying and expiration, then use a predefined or custom strategy.
5. Add strategies to the comparison basket and open Decision Support.
6. Compare expected monetary P&L, percentage returns where capital is known, profit probability and tail loss at a common horizon.

All 15 requested templates are implemented. Templates require matching strikes/widths where appropriate and compatible multipliers; custom legs allow different multipliers. Buys use asks, sells use bids, subject to quoted depth and optional slippage. Fees are **rial per contract per side**. Credit received is never used as invested capital. Strategies containing shorts need user-supplied, sourced margin/collateral, except stock-covered calls/collars whose committed stock capital is known.

The supplied collector's `Option.php` rows do **not** verify the option trading date, exercise style, settlement rules or margin. Such rows can be inspected, but execution-based decisions remain unavailable until those fields are provided. A retrieval date must not be relabeled as a trading date. The custom builder has an explicitly hypothetical mode for user-entered European cash-settled scenarios; these are excluded from actionable rankings and historical backtests.

Historical options execution additionally requires a verified `execution_at` after the information used by the signal was available. Daily after-close collector snapshots alone do not establish this, so the system disables those backtests instead of assuming earlier fills. Stock hedge execution backtests are also unsupported; covered-strategy payoff/scenario calculations are available.

Early-horizon evaluation uses European BSM under explicit volatility, continuous rate, dividend-yield and liquidation-cost assumptions. Expiration uses intrinsic payoff only at the actual requested expiration date. Forecasts after expiry, mixed expirations, unsupported American assignment and missing expiry observations are rejected. See [financial assumptions and limits](docs/assumptions.md).

## Scheduled operation

```powershell
# Foreground, blocking scheduler; Tehran time; one job at a time.
.\.venv\Scripts\python.exe -m market_dss schedule --history --retrain
```

Set `DSS_UPDATE_HOUR`, `DSS_UPDATE_MINUTE` and `DSS_RETRAIN_DAYS` in `.env`. Default update time is 16:30 Asia/Tehran on Saturday–Wednesday; configurable holiday exclusions also apply. Training failures are logged by instrument. Copy `config/holidays.example.csv` to `config/holidays.csv` and enter verified exchange closure dates. An empty/missing file does **not** mean the calendar is complete. Sessions are never forward-filled into price history.

For Windows Task Scheduler, create a daily task with program `E:\Work\Market\.venv\Scripts\python.exe`, arguments `-m market_dss update --history`, and **Start in** `E:\Work\Market`. Configure the trigger in the workstation's local timezone and avoid overlap with the APScheduler process. No system task is installed automatically.

## Module map

| Module | Responsibility |
|---|---|
| `ingestion.py` | Abstract provider, BrsApi/Excel adapters, schema validation, public data functions |
| `storage.py` | File locking, atomic Excel replacement, backups, shards and raw archives |
| `calendar.py`, `features.py` | Persian/Jalali normalization, session estimates and causal features |
| `forecasting.py` | PyTorch model, persistence, uncertainty, evaluation and walk-forward forecasts |
| `options.py` | Contract checks, BSM/IV, templates, custom strategies and exact payoff extrema |
| `risk.py` | Physical-scenario P&L metrics, VaR/CVaR, rankings and Pareto frontier |
| `backtesting.py` | Point-in-time historical option execution and realized trade results |
| `service.py`, `__main__.py` | Application API, CLI, update status and scheduler |
| `dashboard.py` | Seven Persian pages and Plotly/Excel/HTML outputs |

## Operational status

This is a tested local implementation with production-oriented data protections, not a claim of production deployment or profitable forecasting. The workspace initially contained no market datasets or API credentials. Authenticated live responses, actual exchange contract terms, a complete holiday calendar and real options backtest results remain unverified. The public TSETMC history integration is implemented and live-tested; its undocumented service can still be blocked or change schema. Direct Codal integration remains unavailable. Provider references and the verification boundary are recorded in [sources.md](docs/sources.md).

Excel remains the source of truth. A write replaces the whole canonical workbook; this favors correctness and portability over large-scale throughput. Worksheets split before Excel's row limit, but very large universes can exceed practical memory/file-size limits. Raw archives and backups require disk monitoring and an external retention/backup policy.
