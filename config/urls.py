# توضیح فایل: مسیرهای سه پنل، استریم، پیش‌بینی تازه، جستجوی نماد و مدیریت مدل؛ تابع مستقلی ندارد.
from django.urls import path
from forecast.views import dashboard, launch, stream, retain
urlpatterns=[path('',dashboard,name='dashboard'),path('training/',dashboard,name='training'),path('market/',dashboard,name='market'),path('run/',launch,name='launch'),path('stream/',stream,name='stream'),path('retain/',retain,name='retain')]

from forecast.options_views import options_page,refresh_option_data,symbol_search,register_symbol
urlpatterns += [path('options/',options_page,name='options'),path('options/refresh/',refresh_option_data),path('symbols/search/',symbol_search),path('symbols/register/',register_symbol)]

from forecast.views import download_model
urlpatterns += [path("models/download/",download_model)]

from forecast.views import refresh_prediction,prediction_status
urlpatterns += [path("forecast/refresh/",refresh_prediction),path("forecast/status/",prediction_status)]
