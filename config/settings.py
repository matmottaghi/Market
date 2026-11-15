# توضیح فایل: تنظیمات اجرای محلی Django، قالب‌ها، فایل‌های ثابت و منطقهٔ زمانی تهران.
# تابع مستقلی در این فایل تعریف نشده است.

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'local-market-research-only')
DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1', 'testserver']
ROOT_URLCONF = 'config.urls'
INSTALLED_APPS = ['django.contrib.staticfiles', 'forecast']
MIDDLEWARE = ['django.middleware.security.SecurityMiddleware', 'django.middleware.common.CommonMiddleware', 'django.middleware.csrf.CsrfViewMiddleware']
TEMPLATES = [{'BACKEND': 'django.template.backends.django.DjangoTemplates', 'DIRS': [], 'APP_DIRS': True, 'OPTIONS': {}}]
STATIC_URL = 'static/'
TIME_ZONE = 'Asia/Tehran'
USE_TZ = True
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
