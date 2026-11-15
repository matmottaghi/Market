# توضیح فایل: مسیرهای دو پنل، استریم و مدیریت مدل؛ تابع مستقلی ندارد.
from django.urls import path
from forecast.views import dashboard, launch, stream, retain
urlpatterns=[path('',dashboard,name='dashboard'),path('training/',dashboard,name='training'),path('market/',dashboard,name='market'),path('run/',launch,name='launch'),path('stream/',stream,name='stream'),path('retain/',retain,name='retain')]
