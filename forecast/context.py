# توضیح فایل: assets نسخهٔ فایل‌های نمایشی را از زمان/اندازهٔ فایل‌ها می‌سازد تا مرورگر بعد از اصلاح رابط فایل قدیمی نشان ندهد.
from pathlib import Path
from hashlib import sha256

def assets(request):
    root=Path(__file__).parent/'static'/'forecast'
    fingerprint='|'.join(f'{p.stat().st_mtime_ns}:{p.stat().st_size}' for p in [root/'workspace.css',root/'panel.js',root/'options_builder.js'])
    return {'asset_version':sha256(fingerprint.encode()).hexdigest()[:12]}
