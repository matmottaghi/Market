# توضیح فایل: فرمان دریافت تاریخچه و آموزش؛ حالت آفلاین، دریافت کامل و تعداد worker قابل تنظیم است.
# معرفی توابع و متدها:
# - add_arguments: تعریف گزینه‌های فرمان خط فرمان.
# - handle: اعتبارسنجی گزینه‌ها و اجرای پایپ‌لاین.

import json
from django.core.management.base import BaseCommand
from forecast.neural_training import run_neural_pipeline


class Command(BaseCommand):
    help = 'Update daily/hourly CSVs, clean, train and evaluate forecasts'

    def add_arguments(self, parser):
        parser.add_argument('--offline', action='store_true')
        parser.add_argument('--hourly-days', type=int, default=0, help='Per-symbol missing days per run; 0 = all')
        parser.add_argument('--full-history', action='store_true', help='Reconcile all daily rows; retry missing/incomplete hours')
        parser.add_argument('--workers', type=int, default=1, choices=range(1, 9))
        parser.add_argument('--model', choices=['lstm'], default='lstm')
        parser.add_argument('--epochs', type=int, default=200)

        parser.add_argument('--symbols', nargs='+')
        parser.add_argument('--refresh-only', action='store_true')

    def handle(self, *args, **options):
        if options['hourly_days'] < 0:
            raise ValueError('hourly-days must be nonnegative')
        result = run_neural_pipeline(not options['offline'], options['hourly_days'], options['full_history'], options['workers'], options['epochs'], symbols=options['symbols'], refresh_only=options['refresh_only'])
        self.stdout.write(json.dumps(result, ensure_ascii=True, indent=2))
