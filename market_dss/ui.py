"""Shared presentation tokens; scoped styling for the existing Streamlit shell."""
import base64
import os
from pathlib import Path
import streamlit as st

PERSIAN_FONT = "B Roya, Tahoma, Arial, sans-serif"


def font_style():
    # local() uses the installed font without distributing a restricted font file.
    source = 'local("B Roya"), local("BRoya")'
    configured = os.getenv("DSS_B_ROYA_FONT_PATH", "")
    if configured and os.getenv("DSS_B_ROYA_EMBED_AUTHORIZED") == "1":
        path = Path(configured)
        if path.is_file() and path.suffix.lower() in (".ttf", ".woff", ".woff2"):
            mime = {".ttf": "font/ttf", ".woff": "font/woff", ".woff2": "font/woff2"}[path.suffix.lower()]
            source += f', url("data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}")'
    return '<style>@font-face {font-family:"B Roya";src:' + source + ';font-weight:400;font-display:swap;}</style>'


def typography_status():
    installed = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "BROYA.TTF"
    return "B Roya نصب‌شده سیستم؛ بدون توزیع فایل فونت" if installed.exists() else (
        "B Roya برای مرورگر درخواست شده؛ در صورت نصب نبودن، Tahoma استفاده می‌شود. مجوز جاسازی فایل باید جداگانه تأیید شود.")

CSS = """
<style>
.stApp {direction:rtl;font-family:"B Roya",Tahoma,Arial,sans-serif;font-size:1.1rem;line-height:1.8;}
.stApp button,.stApp input,.stApp textarea,.stApp select,.stApp [role="tab"],
[data-testid="stSidebar"], [data-testid="stTopNav"] {font-family:"B Roya",Tahoma,Arial,sans-serif;}
.dss-hero h2 {font-family:"B Roya",Tahoma,Arial,sans-serif;}
[data-testid="stMainBlockContainer"] {max-width:1550px;padding-top:2rem;}
[data-testid="stSidebar"] {direction:rtl;border-left:1px solid #24344d;}
[data-testid="stMetric"] {background:linear-gradient(135deg,#15243b,#101c30);padding:1.2rem;
 border:1px solid #293a53;border-radius:16px;min-height:122px;}
[data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;color:#72e6cc;}
[data-testid="stDataFrame"], [data-testid="stCode"] {direction:ltr;}
.dss-hero {padding:1.5rem 1.8rem;border:1px solid #29425c;border-radius:20px;
 background:linear-gradient(110deg,#142b40,#102034 65%,#163a3d);margin-bottom:1.5rem;}
.dss-hero .eyebrow {font:600 12px Arial;letter-spacing:3px;color:#73e3cb;direction:ltr;text-align:right;}
.dss-hero h2 {margin:.5rem 0;color:#edf5ff;}
.dss-hero p {margin:0;color:#b4c6dc;line-height:1.9;}
@media(max-width:700px) {.dss-hero {padding:1rem;} [data-testid="stMainBlockContainer"] {padding:1rem;}}
</style>
"""

DESCRIPTIONS = {
    "نمای بازار": "نبض معاملات، روند قیمت و کیفیت مشاهدات؛ از داده واقعی تا تحلیل قابل بررسی.",
    "پیش‌بینی": "توزیع احتمالات قیمت، عدم‌قطعیت و مقایسه عملکرد خارج از نمونه با مدل‌های پایه.",
    "بازار آپشن": "زنجیره قراردادها، عمق مظنه و مشخصات قابل اعتبارسنجی بازار اختیار.",
    "ساخت استراتژی": "ساخت سبد چندپایه، بررسی نقطه سربه‌سر و مشاهده ساختار سود و زیان.",
    "پشتیبانی تصمیم": "مقایسه بازده احتمالی و ریسک دنباله با سرمایه و قراردادهای تأییدشده.",
    "آزمون تاریخی": "بررسی عملکرد در طول زمان با رعایت زمان دسترسی به اطلاعات.",
    "تنظیمات و داده‌ها": "مرکز کنترل دریافت داده، اجرای پایپلاین و بررسی سلامت پایگاه اکسل.",
}


def shell(page):
    st.markdown(font_style() + CSS, unsafe_allow_html=True)
    st.markdown(f'<section class="dss-hero"><div class="eyebrow">MARKET / DECISION INTELLIGENCE</div>'
                f'<h2>{page}</h2><p>{DESCRIPTIONS[page]}</p></section>', unsafe_allow_html=True)
