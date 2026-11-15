# توضیح فایل: مسیرهای مشترک و ثبت اتمیک وضعیت؛ مدل Ridge حذف شده است.
# save_status وضعیت اجرا را بدون فایل نیمه‌نوشته ثبت می‌کند.
from pathlib import Path
import json
from market_data.csv_store import _write_bytes_atomic
BASE = Path(__file__).resolve().parent.parent
RUNS = BASE / 'data' / 'training_runs'
def save_status(folder, status):
    _write_bytes_atomic(folder / 'status.json', json.dumps(status, ensure_ascii=False, indent=2))
