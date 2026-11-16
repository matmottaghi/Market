# توضیح فایل: مهاجرت غیرمخرب CSV به SQLite با قفل و نسخهٔ بازگشت مدل‌های فعلی.
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from filelock import FileLock
from forecast.registry import settings_store
from market_data.sqlite_store import migrate
from forecast.pipeline import BASE,RUNS
settings,_=settings_store()
backup=BASE/'data'/'rollback_9b84239'
with FileLock(str(RUNS/'.pipeline.lock'),timeout=0),FileLock(str(settings.root/'.market_sync.lock'),timeout=0):
    if not backup.exists():
        backup.mkdir()
        shutil.copytree(RUNS,backup/'training_runs',ignore=shutil.ignore_patterns('*.lock','*.log'))
    print(json.dumps(migrate(settings.root),ensure_ascii=True,indent=2))
