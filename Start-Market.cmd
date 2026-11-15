@rem توضیح فایل: تغییر مسیر به پوشه پروژه و اجرای سرور محلی Django با Python سراسری؛ تابع مستقلی ندارد.
@echo off
cd /d "%~dp0"
python manage.py runserver 127.0.0.1:8000
