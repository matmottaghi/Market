<!-- توضیح فایل: شروع سریع سه پنل، مسیر داده و آموزش؛ ریاضی کامل در GUIDE_FA.md است. -->
# آزمایشگاه محلی بازار و آپشن

Django و Keras با Python سراسری اجرا می‌شوند؛ هیچ کتابخانه‌ای نصب نشده است.

```powershell
python manage.py runserver 127.0.0.1:8000
```

- `/training/`: انتخاب هدف، دریافت داده، آموزش تا ۲۰۰ epoch با توقف زودهنگام، SSE، منحنی یادگیری، KPI و آستانه‌های تصمیم روی احتمال و قیمت.
- `/market/`: مسیر ساعتی تا ۲۱ جلسه، تعداد روز پویا، بازهٔ پیش‌بینی ۵۰–۹۹٪، توزیع قیمت، تاریخچه و فصل‌گرایی.
- `/options/`: snapshot عمومی قیمت و زنجیره، ۳۱ الگو + ترکیب آزاد/معکوس، bid/ask، هزینه، نمودار عایدی، امید سود و ROI مشروط به تطابق افق و سرمایهٔ مشخص.

## داده و بازگشت

نقطهٔ بازگشت قبل از تغییرات: `9b84239`. داده و مدل قدیم خارج Git در `data/rollback_9b84239` محفوظ‌اند؛ راهنمای بازگشت در همان پوشه است. CSVهای اولیه پس از تطبیق با دیتابیس از پوشهٔ فعال حذف و در ZIP پشتیبان data/backups نگهداری شده‌اند.

مخزن فعال `data/market_csv/market.sqlite3` است. کلیدهای نماد/تاریخ و نماد/ساعت یکتا هستند. migration_report تطبیق تمام ستون‌های ۸۰۱۸ سطر روزانه و ۲۳۳۴۹ سطر ساعتی اولیه را ثبت کرده است. پس از توسعهٔ سبد: ۱۷ نماد، ۴۱٬۲۸۱ روزانه و ۷۱٬۱۲۶ ساعتی؛ گزارش سلامت/پوشش در `data_audit.json` و `coverage` است.

نمادهای تازه: فولاد، فملی، فارس، شپنا، وبملت، خودرو، شستا، یاقوت، استیل، پتروآگاه، دارونو و سیمانو. روزانه تمام تاریخ API و ساعتی از ۲۰۲۱-۱۰-۲۱ دریافت شده است؛ این تاریخچهٔ ساعتی تمام عمر سهام قدیمی نیست. روزهای ناموجود/نامنطبق کنار گذاشته و ثبت می‌شوند.

## آموزش و استفاده

```powershell
python manage.py train_forecast --offline --symbols توان اهرم موج --epochs 200
python manage.py train_forecast --symbols توان اهرم موج --full-history --workers 4
python manage.py train_forecast --symbols فولاد --refresh-only
python manage.py predict_market --run RUN_ID --symbol اهرم
python scripts/refresh_options.py اهرم
python scripts/audit_data.py
```

نمادهای انتخابی هدف‌اند؛ بقیهٔ نمادهای دیتابیس زمینهٔ هر شبکه‌اند. ورودی ۲۱ جلسه، معماری LSTM96→48→Dense64→32 و خروجی پارامترهای نرمال بازده برای ۵۰۴ مختصات بالقوهٔ جلسه/ساعت است. فقط ساعت‌های واقعی معامله با نمونهٔ کافی نمایش داده می‌شوند؛ در برنامهٔ فعلی سه هدف، ۸۴ نقطه برای هر نماد داریم. ترتیب ۱۶ نماد زمینه و scaler هر مدل ثابت ذخیره می‌شود. افزودن زمینهٔ جدید نیاز به بازآموزی دارد.

مدل منتخب را در پنل آموزش نام‌گذاری و ذخیره کنید؛ فعال‌کردن برای بازار اختیاری است. ZIP وزن، preprocessing، محیط و SHA256 دارد. فقط آخرین آزمایش ذخیره‌نشده و مدل‌های انتخاب‌شده باقی می‌مانند. پیش‌بینی تازه از وزن ثابت، نتایج آزمون اصلی را بازنویسی نمی‌کند.

## ارزیابی واقعی

اجرای `20261115T211855-nn-98b150` با سه هدف کامل شد: توان ۲۷ epoch (بهترین۱۵)، اهرم۱۵ (بهترین۳)، موج۲۳ (بهترین۱۱). گزارش مستقل و bootstrap بلوکی در `BENCHMARK_FA.md` و `benchmark.json` همان اجرا است. نتیجهٔ هر نماد با مبنا فرق دارد؛ اجرای موفق تضمین دقت نیست. این آزمون بک‌تست معاملات آپشن نیست.

```powershell
python -m pytest tests -q --basetemp=data/test_checks
python manage.py check
node --check forecast/static/forecast/panel.js
node --check forecast/static/forecast/options_builder.js
```

[راهنمای کامل منطق صفحات، پاکسازی، ریاضی شبکه، خطا و تمام استراتژی‌ها](GUIDE_FA.md)

قیمت‌ها فعلاً تعدیل‌نشده‌اند. منبع عمومی snapshot است؛ زمان دریافت با زمان معامله فرق دارد و زمان مستقل quote آپشن موجود نیست. تعطیلات رسمی و قواعد اعمال/وجه تضمین باید کنترل شوند. هیچ سفارش واقعی ارسال نمی‌شود.


اصلاح دریافت افزایشی: دادهٔ جدید بازار فقط SQLite و آثار جدولی مدل جدید JSON هستند؛ CSV قدیمی فقط خوانده می‌شود. تاریخچه برای هر سهم مستقل است و محدودیت خودکار بر اساس اهرم یا جدیدترین سهم ندارد؛ فقط MARKET_HISTORY_START در صورت تنظیم صریح، تاریخ شروع را محدود می‌کند. ورودی جدید نمادها را بر تاریخ/ساعت دقیق هم‌تراز می‌کند (cross_asset_exact_v3) و برای استفاده از آن باید بازآموزی انجام شود. تاریخچهٔ قدیمی موجود حذف نشده است.


پاک‌سازی CSV انجام شد: تمام جدول‌های قدیمی مدل به JSON منتقل شدند و هیچ CSV مستقلی در پروژه باقی نمانده است. نسخهٔ اصلی فایل‌های حذف‌شده، همراه manifest و SHA256، در data/backups/csv_retirement_20261119T075810Z.zip محفوظ است. گزارش در data/csv_retirement_report.json قرار دارد.


### Hourly return model (v2)
New training runs use `hourly_returns_v2`: an observation at each verified hourly close,
eight features per asset, and signed next-hour log returns. Returns compound into prices
for up to 21 trading sessions (only trading hours; no fabricated overnight candles).
Each instrument uses its entire available history; younger context instruments do not
truncate older target history. `MARKET_HISTORY_START` explicitly overrides the start. Missing target hours remain masked.
Chronological splits purge the full 21-session target interval. Calibration residuals
are saved separately and sampled into 256 reproducible branching price paths; chart
bands and the price histogram use the same paths. Direction weight remains 0.2.
Old distribution models remain readable; retraining is required to use the new target.
Refresh: `python manage.py train_forecast --refresh-only --full-history --workers 2`.
Offline training: `python manage.py train_forecast --offline`.
