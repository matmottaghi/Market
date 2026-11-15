# توضیح فایل: ورودی WSGI برای بارگذاری برنامهٔ Django.
# تابع مستقلی در این فایل تعریف نشده است.

import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
application = get_wsgi_application()
