# توضیح فایل: فرمان پیش‌بینی با مدل ذخیره‌شده بدون تغییر وزن؛ handle نتیجه را به‌صورت JSON برمی‌گرداند.
import json
from django.core.management.base import BaseCommand
from forecast.inference import predict_saved
class Command(BaseCommand):
    def add_arguments(self,parser):
        parser.add_argument('--run',required=True)
        parser.add_argument('--symbol',required=True)
    def handle(self,*args,**options):
        self.stdout.write(json.dumps(predict_saved(options['run'],options['symbol']),ensure_ascii=True))
