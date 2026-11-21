# توضیح فایل: history_start فقط مرز صریح کاربر را اعمال می‌کند؛ archive_status وضعیت گروهی روزهای ساعتی را بدون خواندن CSV می‌سازد.
import os
import pandas as pd

def history_start(store,settings):
    explicit=os.getenv('MARKET_HISTORY_START')
    if explicit:return str(pd.Timestamp(explicit).date())
    # A young context asset must never truncate another instrument's archive.
    # None means use each instrument's entire available history.
    return None


def archive_status(store,code):
    with store.connect() as db:
        rows=db.execute('''SELECT h.date,SUM(h.volume),MIN(h.quality),MAX(h.quality),d.volume
            FROM hourly_bars h LEFT JOIN daily_bars d ON d.code=h.code AND d.date=h.date
            WHERE h.code=? GROUP BY h.date''',(str(code),)).fetchall()
        failures={date:(error,retry) for date,error,retry in db.execute('SELECT date,error,retry_after FROM ingestion_issues WHERE code=?',(str(code),))}
    quality={date:('matched' if lo==hi=='matched' and ref and total is not None and abs(total/ref-1)<=.001 else 'volume_mismatch') for date,total,lo,hi,ref in rows}
    return quality,failures
