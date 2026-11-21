# توضیح فایل: جدول‌های آثار مدل فقط JSON نوشته می‌شوند؛ table_path/read_table صرفاً برای خواندن مدل‌های قدیمی CSV سازگاری دارند.
# write_table ثبت اتمیک، read_table حفظ نوع ستون‌ها و learning_tables انتخاب منحنی بدون تکرار نسخه‌های قدیمی را انجام می‌دهد.
from pathlib import Path
from io import StringIO
import pandas as pd
from market_data.csv_store import _write_bytes_atomic

def table_path(path):
    path=Path(path);modern=path.with_suffix('.json')
    legacy=path.with_suffix('.csv')
    return modern if modern.exists() else legacy if legacy.exists() else modern

def write_table(frame,path):
    _write_bytes_atomic(Path(path).with_suffix('.json'),frame.to_json(orient='table',index=False,date_format='iso',double_precision=15))

def read_table(path,dtype=None,usecols=None,**kwargs):
    path=table_path(path)
    frame=pd.read_json(StringIO(path.read_text(encoding='utf8')),orient='table') if path.suffix=='.json' else pd.read_csv(path,dtype=dtype,**kwargs)
    if dtype:
        for col,kind in dtype.items():
            if col in frame:frame[col]=frame[col].astype(kind)
    return frame[list(usecols)] if usecols is not None else frame

def learning_tables(folder):
    rows={p.stem:p for p in Path(folder).glob('*_learning.csv')}
    rows.update({p.stem:p for p in Path(folder).glob('*_learning.json')})
    return rows.values()
