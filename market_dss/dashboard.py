"""Persian Streamlit presentation. Financial/model logic lives in independent modules."""

import hashlib
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import streamlit as st

from .calendar import jalali, parse_date
from .features import engineer_features
from .options import (
    OptionLeg,
    Strategy,
    TEMPLATES,
    UnsupportedContract,
    build_strategy,
    contracts_from_frame,
    implied_volatility,
)
from .risk import OBJECTIVES, evaluate_strategy, rank_strategies
from .service import MarketService
from .storage import atomic_bytes, excel_bytes

PAGES = [
    "نمای بازار",
    "پیش‌بینی",
    "بازار آپشن",
    "ساخت استراتژی",
    "پشتیبانی تصمیم",
    "آزمون تاریخی",
    "تنظیمات و داده‌ها",
]
LABELS = {
    "symbols_processed": "نمادهای پردازش‌شده",
    "historical_inserted": "مشاهدات تاریخی جدید",
    "historical_fetched": "مشاهدات معتبر دریافت‌شده",
    "models_updated": "مدل‌های آموزش‌دیده",
    "option_contracts": "قراردادهای ذخیره‌شده",
    "forecasts_updated": "پیش‌بینی‌های تازه",
    "api_records": "رکورد API",
    "stored_records": "رکورد ذخیره‌شده",
    "valid_feature_rows": "ردیف آماده ویژگی",
    "preprocessing_removed": "حذف آماده‌سازی",
    "api_removed": "رکورد نامعتبر API",
    "earliest_date": "ابتدای تاریخچه",
    "latest_date": "انتهای تاریخچه",
    "potential_session_gaps": "شکاف احتمالی جلسات",
    "missing_values": "مقادیر مفقود",
    "duplicates": "رکورد تکراری",
    "symbol": "نماد",
    "underlying_symbol": "نماد پایه",
    "trading_date": "تاریخ میلادی معامله",
    "close": "قیمت پایانی",
    "last": "آخرین قیمت",
    "volume": "حجم",
    "value": "ارزش معاملات",
    "strike": "قیمت اعمال",
    "expiration": "سررسید",
    "multiplier": "اندازه قرارداد",
    "bid": "خرید",
    "ask": "فروش",
    "option_type": "نوع اختیار",
    "open_interest": "موقعیت باز",
    "instrument_id": "شناسه",
    "retrieved_at": "زمان دریافت",
    "source": "منبع",
    "nav": "خالص ارزش دارایی",
    "nav_premium_pct": "حباب درصد",
    "strategy": "استراتژی",
    "horizon": "تعداد جلسات",
    "target_date": "تاریخ هدف",
    "model": "مدل",
    "expected_pnl": "سود و زیان مورد انتظار",
    "expected_return_pct": "بازده مورد انتظار ٪",
    "median_return_pct": "میانه بازده ٪",
    "probability_profit": "احتمال سود",
    "probability_loss": "احتمال زیان",
    "var_loss": "ارزش در معرض خطر",
    "cvar_loss": "میانگین زیان دنباله",
    "capital": "سرمایه مبنا",
    "expected_downside_loss": "زیان انتظاری",
    "pareto": "مرز پارتو",
    "weighted_score": "امتیاز وزنی",
    "realized_pnl": "سود و زیان محقق‌شده",
    "realized_return_pct": "بازده محقق‌شده ٪",
    "entry_date": "تاریخ ورود",
    "exit_date": "تاریخ خروج",
    "date": "تاریخ",
    "issue": "مسئله",
    "reason": "دلیل",
    "split": "بخش ارزیابی",
    "implied_volatility": "نوسان ضمنی",
    "return_q05_pct": "صدک ۵ بازده ٪",
    "return_q95_pct": "صدک ۹۵ بازده ٪",
    "return_std_pct": "انحراف معیار بازده ٪",
    "probability_target_return": "احتمال عبور از بازده هدف",
    "probability_loss_threshold": "احتمال زیان بیش از آستانه",
}
MONEY = {
    "close",
    "last",
    "open",
    "high",
    "low",
    "bid",
    "ask",
    "strike",
    "nav",
    "value",
    "expected_pnl",
    "median_pnl",
    "var_loss",
    "cvar_loss",
    "capital",
    "pnl_std",
    "expected_downside_loss",
    "max_theoretical_loss",
    "realized_pnl",
    "expected",
    "median",
    "lower",
    "upper",
    "q05",
    "q25",
    "q75",
    "q95",
    "actual",
    "model_price",
    "sma_5",
    "sma_21",
    "ema_5",
    "ema_21",
    "bollinger_upper",
    "bollinger_lower",
    "atr",
    "adjusted_close",
}
STRATEGY_LABELS = dict(
    zip(
        TEMPLATES,
        [
            "خرید اختیار خرید",
            "خرید اختیار فروش",
            "اختیار خرید پوشش‌دار",
            "اختیار فروش حفاظتی",
            "اسپرد صعودی خرید",
            "اسپرد نزولی فروش",
            "اسپرد صعودی فروش",
            "اسپرد نزولی خرید",
            "استرادل خرید",
            "استرادل فروش",
            "استرانگل خرید",
            "استرانگل فروش",
            "پروانه",
            "کندور آهنی",
            "کالر",
        ],
    )
)


def show_table(frame, unit=1):
    display = frame.copy()
    for col in MONEY & set(display):
        display[col] = pd.to_numeric(display[col], errors="coerce") / unit
    st.dataframe(display.rename(columns=LABELS), width="stretch", hide_index=True)


def download(frame, name, key):
    st.download_button(
        "دریافت جدول اکسل — مبالغ به ریال",
        excel_bytes({"results": frame}),
        f"{name}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,
    )


def chart(figure, key):
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#101c30",
        colorway=["#54d6b6", "#7daeff", "#f4bb72", "#e989a5"], hovermode="x unified",
        font={"family": "B Roya, Tahoma, Arial, sans-serif", "size": 16},
        legend={"orientation": "h"},
        margin={"l": 25, "r": 25, "t": 45, "b": 30},
    )
    st.plotly_chart(
        figure,
        width="stretch",
        key=key,
        config={"displaylogo": False, "toImageButtonOptions": {"format": "png"}},
    )
    with st.expander("خروجی تعاملی نمودار"):
        st.download_button(
            "دریافت HTML",
            figure.to_html(include_plotlyjs=True).encode("utf-8"),
            f"{key}.html",
            "text/html",
            key=f"html_{key}",
        )


def require_history(service, symbol):
    history = service.history(symbol) if symbol else pd.DataFrame()
    if history.empty:
        st.info("داده تاریخی موجود نیست. برای دریافت خودکار اطلاعات و آماده‌سازی مدل، اجرای کامل پایپلاین را انتخاب کنید.")
        return None
    age = (pd.Timestamp.now(tz="Asia/Tehran").date() - parse_date(history.trading_date.iloc[-1])).days
    if age > 3:
        st.warning(f"آخرین مشاهده {age} روز تقویمی قبل است؛ نتایج با این تاریخ محاسبه می‌شوند.")
    return history


def market_page(service, symbol, unit):
    history = require_history(service, symbol)
    if history is None:
        return
    quote = service.latest_quote(symbol)
    if quote:
        from .live import freshness, quote_timestamp
        stamp = quote.get("provider_quote_at") or quote_timestamp(quote)[0]
        state = freshness(stamp, service.settings.live_max_age_hours)
        st.caption(f"آخرین snapshot: {quote['source']} | زمان معامله: {stamp or 'نامشخص'} | "
                   f"زمان دریافت: {quote['retrieved_at']} | وضعیت: {state}")
        st.write("قیمت جاری دریافت‌شده:", f"{float(quote.get('last') or quote.get('close')) / unit:,.0f}")
        if state in ("STALE", "UNKNOWN"):
            st.warning("زمان مظنه قدیمی یا نامشخص است؛ این داده به‌عنوان قیمت اجرایی لحظه‌ای تأیید نشده است.")
    nav = service.repository.read("nav")
    if not nav.empty:
        nav = nav[
            nav.instrument_id.eq(str(history.instrument_id.iloc[-1])) & nav.source.eq(service.settings.source)
        ]
        # Same-date NAV only, no nearest-date filling.
        history = history.merge(
            nav[["trading_date", "nav"]].drop_duplicates("trading_date"), on="trading_date", how="left"
        )
    features = engineer_features(history, service.settings.annual_sessions)
    last = features.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("قیمت پایانی", f"{last.close / unit:,.2f}")
    cols[1].metric(
        "بازده روزانه", f"{last.simple_return * 100:.2f}٪" if pd.notna(last.simple_return) else "ناموجود"
    )
    cols[2].metric(
        "نوسان سالانه تاریخی",
        f"{last.historical_volatility * 100:.1f}٪" if pd.notna(last.historical_volatility) else "ناموجود",
    )
    premium = last.get("nav_premium_pct")
    cols[3].metric("حباب نسبت به NAV هم‌تاریخ", f"{premium:.2f}٪" if pd.notna(premium) else "ناموجود")
    st.caption(
        f"تاریخ مشاهده: {jalali(last.trading_date)} | منبع: {last.source} | مبنای بازده: {last.price_basis}"
    )
    details = []
    for field, label in (("volume", "حجم معاملات"), ("value", "ارزش معاملات"), ("nav", "NAV")):
        value = last.get(field)
        if value is not None and pd.notna(value):
            divisor = unit if field in ("value", "nav") else 1
            details.append(f"{label}: {float(value) / divisor:,.0f}")
    if details:
        st.caption(" | ".join(details))
    lookback = st.slider(
        "تعداد مشاهدات نمودار",
        20,
        max(21, len(features)),
        min(252, max(20, len(features))),
        key="market_lookback",
    )
    view = features.tail(lookback)
    figure = go.Figure()
    for col, label in (
        ("model_price", "قیمت مبنای تحلیل"),
        ("sma_21", "میانگین ۲۱ جلسه"),
        ("bollinger_upper", "باند بالا"),
        ("bollinger_lower", "باند پایین"),
    ):
        figure.add_scatter(x=view.trading_date, y=view[col] / unit, name=label)
    if {"open", "high", "low", "close"} <= set(view) and view[["open", "high", "low", "close"]].notna().all().all():
        figure.add_trace(go.Candlestick(x=view.trading_date, open=view.open / unit,
                                       high=view.high / unit, low=view.low / unit,
                                       close=view.close / unit, name="کندل مشاهدات واقعی"))
    figure.update_layout(xaxis_rangeslider_visible=False)
    chart(figure, "market_history")
    if "volume" in view:
        chart(px.bar(view, x="trading_date", y="volume", labels=LABELS, title="حجم معاملات"), "market_volume")
    choices = service.instruments().symbol.unique().tolist()
    comparison = st.multiselect("مقایسه نمادها — قیمت ابتدای بازه برابر ۱۰۰", choices, default=[item for item in ("اهرم", "موج", "توان") if item in choices] or [symbol])
    combined = []
    for selected in comparison:
        data = service.history(selected).tail(lookback)
        if not data.empty:
            data = engineer_features(service.history(selected), service.settings.annual_sessions).tail(lookback)
            data = data.assign(normalized=data.close / data.close.iloc[0] * 100)
            combined.append(data)
    if combined:
        chart(
            px.line(pd.concat(combined), x="trading_date", y="normalized", color="symbol", labels=LABELS),
            "market_comparison",
        )
        with st.expander("مقایسه بازده، نوسان، حجم و افت سرمایه صندوق‌ها"):
            for field, title in (("simple_return", "بازده روزانه"), ("volatility_21", "نوسان غلتان سالانه"),
                                 ("volume", "حجم معامله"), ("drawdown", "افت از اوج تاریخی")):
                chart(px.line(pd.concat(combined), x="trading_date", y=field, color="symbol", title=title),
                      f"fund_compare_{field}")
            st.caption("NAV تنها در صورت وجود مشاهده معتبر هم‌تاریخ مقایسه می‌شود؛ مقادیر مفقود پر نمی‌شوند.")
    show_table(features.tail(30), unit)
    download(features, "indicators", "market_export")


def forecast_page(service, symbol, unit):
    from .forecasting import Forecast, ForecastEngine, ModelConfig, load_model_artifacts

    choices = list(dict.fromkeys([*(service.instruments().symbol.tolist() if not service.instruments().empty else []),
                                  "اهرم", "موج", "توان"]))
    symbol = st.selectbox("نماد مدل", choices, index=choices.index(symbol) if symbol in choices else 0,
                          key="forecast_symbol")
    tabs = st.tabs(["پیش‌بینی قیمت", "منحنی یادگیری", "ارزیابی عملکرد مدل", "مدیریت مدل"])
    history = service.history(symbol)
    if history.empty:
        for tab in tabs:
            with tab:
                st.info("تاریخچه این نماد موجود نیست؛ در تنظیمات و داده‌ها، پایپلاین کامل را اجرا کنید.")
        return
    identifier = str(history.instrument_id.iloc[-1])
    versions = service.models(identifier)
    selected = st.selectbox("نسخه مدل", versions, format_func=lambda path: path.parent.name,
                            key=f"model_version_{identifier}") if versions else None
    artifact = load_model_artifacts(selected.parent) if selected else None
    max_horizon = artifact["config"]["horizon"] if artifact else 21
    horizons = [h for h in (1, 5, 21) if h <= max_horizon]
    horizon = st.selectbox("افق پیش‌بینی (جلسه)", horizons, index=len(horizons)-1, key=f"model_horizon_{identifier}",
                           format_func=lambda h: {1:"۱ جلسه — روزانه", 5:"۵ جلسه — هفتگی", 21:"۲۱ جلسه — ماهانه"}[h])
    with tabs[3]:
        st.subheader("مدیریت مدل")
        st.caption(f"نماد: {symbol} | افق انتخابی: {horizon} | آخرین معامله داده: {jalali(history.trading_date.iloc[-1])}")
        st.caption(f"تعداد مشاهدات ورودی: {len(history):,} | آخرین دریافت: {history.retrieved_at.max()}")
        if artifact:
            metadata = artifact["metadata"]
            st.caption(f"نسخه: {metadata['version']} | آموزش: {metadata['created_at']} | برش داده: {metadata['input_data_cutoff']}")
            st.write("وضعیت بازآموزی:", "موعد بازآموزی رسیده" if service.training_due(identifier) else "مدل ذخیره‌شده موجود است")
            st.write("مشاهدات آماده ویژگی:", metadata.get("eligible_feature_rows", "در نسخه قدیمی ثبت نشده"))
            training = metadata.get("partitions", {}).get("train", {})
            st.write("مشاهدات بخش آموزش:", training.get("observations", "ثبت نشده"),
                     " | پنجره‌های آموزشی:", training.get("samples", "ثبت نشده"))
            show_table(pd.DataFrame([artifact["config"]]))
            st.success("گزارش‌ها و مشخصات مدل از فایل ذخیره‌شده خوانده شد؛ بارگذاری وزن‌ها فقط هنگام پیش‌بینی انجام می‌شود.")
        cols = st.columns(4)
        train_horizon = cols[0].number_input("حداکثر افق مدل (جلسه)", 1, 126, 21)
        window = cols[1].number_input("طول پنجره", 5, 200, 30)
        epochs = cols[2].number_input("حداکثر دوره آموزش", 1, 500, service.settings.training_epochs)
        hidden = cols[3].number_input("اندازه لایه پنهان", 4, 128, 32)
        attention = st.checkbox("لایه توجه")
        st.caption("تمام تاریخچه معتبر استفاده می‌شود؛ بخش آزمون زمانی وارد آموزش نمی‌شود.")
        if st.button("آموزش مجدد مدل", type="primary", key="forecast_retrain"):
            with st.spinner("آموزش، اعتبارسنجی زمانی و ذخیره منحنی یادگیری…"):
                engine = ForecastEngine(ModelConfig(horizon=train_horizon, window=window, epochs=epochs,
                    hidden_size=hidden, attention=attention, annual_sessions=service.settings.annual_sessions)).fit(history)
                engine.save(service.settings.model_dir)
            st.session_state.pop("forecast", None)
            st.rerun()
    if not artifact:
        for tab in tabs[:3]:
            with tab:
                st.info("مدل ذخیره‌شده‌ای موجود نیست؛ در مدیریت مدل آموزش را اجرا کنید.")
        return
    with tabs[1]:
        learning_curve(artifact, selected.parent)
    with tabs[2]:
        model_evaluation(service, symbol, artifact, horizon, selected.parent)
    with tabs[0]:
        cols = st.columns(3)
        confidence = cols[0].slider("سطح بازه پیش‌بینی", 0.50, 0.99, 0.90)
        simulations = cols[1].number_input("تعداد مسیر", 100, 50000, 2000, step=100)
        vol_scale = cols[2].slider("ضریب سناریوی نوسان", 0.25, 3.0, 1.0)
        target = st.number_input("قیمت هدف (واحد نمایش)", min_value=.01,
                                 value=float(history.close.iloc[-1] / unit)) * unit
        if st.button("تولید توزیع پیش‌بینی"):
            with st.spinner("بارگذاری وزن‌های نسخه انتخابی و تولید سناریوهای مدل…"):
                engine = ForecastEngine.load(selected.parent)
                st.session_state["forecast"] = engine.predict(history, horizon, simulations, vol_scale, service.calendar)
        forecast = st.session_state.get("forecast")
        if forecast is not None and (forecast.metadata["instrument_id"] != identifier or
            forecast.metadata["version"] != artifact["metadata"]["version"] or forecast.metadata["horizon"] != horizon or
            forecast.metadata["input_data_cutoff"] != str(history.trading_date.iloc[-1])):
            forecast = None
        saved = service.settings.data_dir / "forecasts" / f"{identifier}_{horizon}.xlsx"
        if forecast is None and saved.exists():
            stored = Forecast.from_excel(saved)
            if (stored.metadata["version"] == artifact["metadata"]["version"] and
                    str(stored.metadata["input_data_cutoff"]) == str(history.trading_date.iloc[-1])):
                forecast = stored
                st.session_state["forecast"] = stored
        if forecast is None:
            st.info("برای این نسخه و افق خروجی مطابق آخرین داده موجود نیست؛ تولید توزیع پیش‌بینی را انتخاب کنید.")
            return
        if forecast.metadata.get("model_stale"):
            st.warning("خروجی مدل قدیمی است؛ وضعیت بازآموزی را بررسی کنید.")
        st.caption(f"تولید: {forecast.metadata['generated_at']} | نسخه: {forecast.metadata['version']} | "
                   f"برش داده: {forecast.metadata['input_data_cutoff']} | افق: {horizon} جلسه")
        st.info("نمودار آینده پیش‌بینی مدل است؛ تاریخ جلسات آینده تا تأیید کامل تقویم برآوردی است.")
        forecast_outputs(history, forecast, confidence, target, unit, service.settings.annual_sessions)


def forecast_outputs(history, forecast, confidence, target, unit, annual_sessions):
    summary = forecast.summary(confidence, target)
    figure = go.Figure()
    figure.add_scatter(
        x=history.trading_date.tail(100),
        y=history.close.tail(100) / unit,
        name="قیمت مشاهده‌شده",
        line_color="#7daeff",
    )
    figure.add_scatter(x=summary.date, y=summary.upper / unit, name="حد بالا", line_width=0)
    figure.add_scatter(
        x=summary.date, y=summary.lower / unit, name="بازه پیش‌بینی", fill="tonexty", line_width=0
    )
    figure.add_scatter(x=summary.date, y=summary.q75 / unit, name="صدک ۷۵", line_width=0)
    figure.add_scatter(x=summary.date, y=summary.q25 / unit, name="بازه میانی", fill="tonexty", line_width=0)
    figure.add_scatter(x=summary.date, y=summary.expected / unit, name="امید قیمت", line_color="#00a58b")
    figure.add_scatter(x=summary.date, y=summary["median"] / unit, name="میانه", line_dash="dash")
    chart(figure, "forecast_fan")

    with st.expander("مسیرهای مونت‌کارلو"):
        figure = go.Figure()
        for path in forecast.paths[:100]:
            figure.add_scatter(
                x=forecast.dates, y=path / unit, line={"width": 0.6}, opacity=0.25, showlegend=False
            )
        chart(figure, "forecast_paths")
    with st.expander("توزیع قیمت آینده و احتمال تجمعی"):
        terminal = forecast.paths[:, -1] / unit
        figure = px.histogram(
            x=terminal, nbins=60, histnorm="probability density", title="چگالی قیمت پایانی شبیه‌سازی"
        )
        if np.std(terminal) > 1e-8:
            grid = np.linspace(terminal.min(), terminal.max(), 200)
            figure.add_scatter(x=grid, y=gaussian_kde(terminal)(grid), name="چگالی تجربی هموار")
        chart(figure, "terminal_density")
        ordered = np.sort(terminal)
        chart(
            go.Figure(
                go.Scatter(x=ordered, y=np.arange(1, len(ordered) + 1) / len(ordered), name="توزیع تجمعی")
            ),
            "terminal_cdf",
        )
    with st.expander("توزیع بازده و نوسان"):
        chart(
            px.histogram(
                x=100 * (forecast.paths[:, -1] / forecast.metadata["spot"] - 1),
                nbins=50,
                title="بازده تجمعی سناریوها (درصد)",
            ),
            "return_histogram",
        )
        historical = engineer_features(history, annual_sessions)
        figure = go.Figure(
            go.Scatter(
                x=historical.trading_date.tail(100),
                y=historical.volatility_21.tail(100),
                name="نوسان تاریخی سالانه",
            )
        )
        figure.add_scatter(
            x=summary.date,
            y=summary.cumulative_log_return_std
            * np.sqrt(annual_sessions / np.arange(1, len(summary) + 1)),
            name="نوسان ضمنی توزیع پیش‌بینی سالانه",
        )
        chart(figure, "forecast_volatility")
        st.caption("خطای باقیمانده شامل نوآوری بازار و خطای مدل است؛ سهم dropout فقط تقریب عدم‌قطعیت مدل است.")
    show_table(summary, unit)
    st.download_button(
        "دریافت پیش‌بینی و مشخصات — ریال",
        excel_bytes({"forecast": summary, "metadata": pd.DataFrame([forecast.metadata])}),
        "forecast.xlsx",
        key="forecast_download",
    )



def learning_curve(artifact, directory):
    st.subheader("منحنی یادگیری مدل LSTM")
    losses, metadata = artifact["training"], artifact["metadata"]
    if losses.empty:
        st.info("تاریخچه آموزش برای این نسخه موجود نیست.")
        return
    best = int(metadata.get("best_epoch", losses.loc[losses.validation_mse.idxmin(), "epoch"]))
    figure = go.Figure()
    figure.add_scatter(x=losses.epoch, y=losses.train_mse, name="زیان آموزش", line_color="#54d6b6")
    figure.add_scatter(x=losses.epoch, y=losses.validation_mse, name="زیان اعتبارسنجی", line_color="#f4bb72")
    figure.add_vline(x=best, annotation_text="بهترین دوره", line_dash="dash")
    stopping = metadata.get("early_stopping_epoch")
    if stopping:
        figure.add_vline(x=stopping, annotation_text="توقف زودهنگام", line_color="#e989a5")
    figure.update_layout(xaxis_title="دوره آموزش", yaxis_title="MSE بازده لگاریتمی استاندارد — بدون واحد")
    chart(figure, "model_learning_curve")
    cols = st.columns(4)
    cols[0].metric("دوره‌های انجام‌شده", len(losses))
    cols[1].metric("بهترین دوره", best)
    cols[2].metric("کمترین زیان اعتبارسنجی", f"{losses.validation_mse.min():.5f}")
    cols[3].metric("زیان نهایی آموزش", f"{losses.train_mse.iloc[-1]:.5f}")
    st.caption("زیان استانداردشده است و خطای قیمت به ریال نیست. وزن‌های ذخیره‌شده مربوط به بهترین اعتبارسنجی هستند.")
    if metadata.get("partitions"):
        st.caption("پنجره‌های هر بخش: " + " | ".join(f"{name}: {part['samples']}" for name, part in metadata["partitions"].items()))
    if metadata.get("training_schema_version"):
        st.caption("توقف زودهنگام فعال شد." if stopping else "سقف دوره‌های تنظیم‌شده تکمیل شد.")
    if "learning_rate" in losses:
        chart(px.line(losses, x="epoch", y="learning_rate", title="تاریخچه واقعی نرخ یادگیری"), "model_learning_rate")
    else:
        st.info("نسخه قدیمی تاریخچه نرخ یادگیری را ذخیره نکرده است؛ برای دیدن منحنی‌های موجود نیازی به آموزش مجدد نیست.")
    st.caption(f"مدت آموزش (ثانیه): {metadata.get('training_duration_seconds', 'ثبت نشده')} | "
               f"پنجره: {artifact['config']['window']} | batch: {artifact['config']['batch_size']} | "
               f"لایه‌ها: {artifact['config']['layers']} | hidden: {artifact['config']['hidden_size']}")
    show_table(losses)
    path = directory / "training_history.xlsx"
    if path.exists():
        st.download_button("دریافت تاریخچه آموزش", path.read_bytes(), "training_history.xlsx", key="training_history_download")


def model_evaluation(service, symbol, artifact, horizon, directory):
    st.subheader("ارزیابی عملکرد مدل")
    metrics = artifact["metrics"]
    view = metrics[metrics.horizon.eq(horizon)]
    st.caption("معیارهای آزمون زمانی؛ MAE، RMSE، CRPS و خطای چندکی در واحد مبنای قیمت آموزش (ریال) هستند. "
               "دقت جهت و پوشش بین صفر و یک‌اند؛ پوشش بازه ۹۰٪ نزدیک ۰٫۹ مطلوب است.")
    show_table(view)
    chart(px.bar(view, x="model", y="rmse", color="model", title="LSTM و مدل‌های پایه — RMSE ریال"), "evaluation_comparison")
    out = artifact["predictions"]
    out = out[out.horizon.eq(horizon)]
    if not out.empty:
        chart(px.line(out, x="target_date", y=["actual", "expected", "q05", "q95"],
                      title="قیمت واقعی و پیش‌بینی در مجموعه آزمون"), "forecast_oos")
    partitions = artifact["metadata"].get("partitions")
    if partitions:
        st.markdown("**مرزها و نمونه‌های تقسیم زمانی**")
        show_table(pd.DataFrame([{"partition": name, **values} for name, values in partitions.items()]))
    else:
        metadata = artifact["metadata"]
        st.caption(f"آخر هدف آموزش: {metadata.get('training_target_end')} | "
                   f"آخر کالیبراسیون: {metadata.get('calibration_end')} | شروع آزمون: {metadata.get('test_start')}")
        st.info("تعداد دقیق نمونه هر بخش در این نسخه قدیمی ثبت نشده است.")
    st.caption("افت نمونه‌ها در مرز بخش‌ها برای حذف هم‌پوشانی اهداف است. مقیاس‌گذار فقط با آموزش برازش شده است.")
    from .forecasting import load_model_artifacts
    compare = st.multiselect("مقایسه گزارش‌های ذخیره‌شده نمادها", ["اهرم", "موج", "توان"],
                            default=[symbol] if symbol in ["اهرم", "موج", "توان"] else [])
    rows = []
    for item in compare:
        history = service.history(item)
        if history.empty:
            continue
        versions = service.models(str(history.instrument_id.iloc[-1]))
        if versions:
            report = artifact if item == symbol else load_model_artifacts(versions[0].parent)
            selected_metrics = report["metrics"][report["metrics"].horizon.eq(horizon)].copy()
            selected_metrics["symbol"] = item
            selected_metrics["version"] = report["metadata"]["version"]
            rows.append(selected_metrics)
    if rows:
        st.caption("خطای ریالی نمادهای با سطح قیمت متفاوت قابل رتبه‌بندی مستقیم نیست؛ هر مدل را با پایه همان نماد مقایسه کنید.")
        show_table(pd.concat(rows, ignore_index=True))
    with st.expander("تشخیص توزیع باقیمانده و کالیبراسیون"):
        st.json(artifact.get("distribution_diagnostics", {}))
    st.download_button("دریافت گزارش کامل ارزیابی", (directory / "evaluation.xlsx").read_bytes(),
                       "evaluation.xlsx", key="evaluation_report_download")
    walk_path = service.settings.data_dir / "backtests" / f"forecast_walk_forward_{artifact['metadata']['instrument_id']}.xlsx"
    if walk_path.exists():
        with st.expander("نتایج آزمون گام‌به‌گام پیش‌بینی"):
            show_table(pd.read_excel(walk_path, sheet_name=0))


def option_page(service, symbol, unit):
    chain = service.chain(symbol) if symbol else pd.DataFrame()
    if chain.empty:
        st.info("زنجیره اختیار معامله موجود نیست. فایل گردآورنده یا داده معتبر وارد کنید.")
        return
    kinds = st.multiselect(
        "نوع قرارداد",
        ["call", "put"],
        default=["call", "put"],
        format_func=lambda x: "خرید" if x == "call" else "فروش",
    )
    if "option_type" in chain:
        chain = chain[chain.option_type.isin(kinds)]
    expirations = sorted(chain.expiration.dropna().unique()) if "expiration" in chain else []
    dates = st.multiselect("سررسید", expirations, default=expirations, format_func=jalali)
    if expirations:
        chain = chain[chain.expiration.isin(dates)]
    min_volume = st.number_input("حداقل حجم", 0, value=0)
    if "volume" in chain and min_volume:
        chain = chain[chain.volume >= min_volume]
    history = service.history(symbol)
    contracts, issues = contracts_from_frame(chain)
    iv, reliability = {}, {}
    rate = st.number_input("نرخ بدون ریسک پیوسته سالانه برای IV", 0.0, 2.0, 0.30)
    if not history.empty:
        cutoff, spot = parse_date(history.trading_date.iloc[-1]), float(history.close.iloc[-1])
        for contract in contracts:
            try:
                if contract.trading_date != cutoff or contract.bid is None or contract.ask is None:
                    raise ValueError("قیمت هم‌تاریخ یا خرید/فروش موجود نیست")
                iv[contract.instrument_id] = implied_volatility(
                    contract, (contract.bid + contract.ask) / 2, spot, cutoff, rate
                )
            except ValueError as exc:
                reliability[contract.instrument_id] = str(exc)
    chain["implied_volatility"] = chain.instrument_id.map(iv)
    chain["iv_status"] = chain.instrument_id.map(reliability)
    if "expiration" in chain:
        chain["calendar_days_remaining"] = chain.expiration.map(
            lambda x: (parse_date(x) - pd.Timestamp.now(tz="Asia/Tehran").date()).days
        )
    if {"bid", "ask"} <= set(chain):
        chain["relative_spread"] = (chain.ask - chain.bid) / ((chain.ask + chain.bid) / 2).replace(0, np.nan)
    st.warning(
        "تاریخ دریافت، تاریخ معامله نیست. نوع اعمال، تسویه، اندازه قرارداد و عمق سفارش باید معتبر باشند."
    )
    show_table(chain, unit)
    if issues:
        show_table(pd.DataFrame(issues))
    if {"strike", "last", "option_type"} <= set(chain):
        view = chain.copy()
        view[["strike", "last"]] /= unit
        chart(
            px.scatter(
                view,
                x="strike",
                y="last",
                color="option_type",
                hover_name="symbol",
                labels=LABELS,
                title="آخرین معامله؛ تضمین اجرای سفارش نیست",
            ),
            "option_premiums",
        )
    download(chain, "option_chain", "chain_download")


def strategy_page(service, symbol, unit):
    history = require_history(service, symbol)
    if history is None:
        return
    contracts, issues = contracts_from_frame(service.chain(symbol))
    if not contracts:
        st.info("قرارداد دارای قیمت اعمال، سررسید و اندازه معتبر برای ساخت استراتژی موجود نیست.")
        if issues:
            show_table(pd.DataFrame(issues))
        return
    spot, cutoff = float(history.close.iloc[-1]), parse_date(history.trading_date.iloc[-1])
    contracts = [c for c in contracts if c.expiration > cutoff]
    if not contracts:
        st.info("قرارداد منقضی‌نشده موجود نیست.")
        return
    expiration = st.selectbox("سررسید مشترک", sorted({c.expiration for c in contracts}), format_func=jalali)
    candidates = [c for c in contracts if c.expiration == expiration]
    selected_ids = st.multiselect(
        "قراردادهای مورد استفاده",
        [c.instrument_id for c in candidates],
        format_func=lambda identifier: next(c.symbol for c in candidates if c.instrument_id == identifier),
    )
    selected = [c for c in candidates if c.instrument_id in selected_ids]
    name = st.selectbox(
        "الگوی استراتژی", list(TEMPLATES) + ["Custom"], format_func=lambda x: STRATEGY_LABELS.get(x, "سفارشی")
    )
    cols = st.columns(4)
    quantity = cols[0].number_input("تعداد قرارداد پایه", 1, 100000, 1)
    entry_fee = cols[1].number_input("کارمزد ورود هر قرارداد (ریال)", 0.0, value=0.0)
    exit_fee = cols[2].number_input("کارمزد خروج/اعمال هر قرارداد (ریال)", 0.0, value=0.0)
    slippage = cols[3].slider("لغزش اجرای سفارش (درصد)", 0.0, 10.0, 0.0) / 100
    capital = st.number_input("سرمایه/وثیقه تأییدشده (ریال؛ صفر یعنی ناموجود)", 0.0, value=0.0)
    capital_source = st.text_input("منبع مبلغ وثیقه یا سرمایه متعهدشده")
    hypothetical = st.checkbox("سناریوی فرضی با مفروضات کاربر؛ خارج از رتبه‌بندی و آزمون واقعی")
    rows = []
    if name == "Custom" and selected:
        rows = st.data_editor(
            pd.DataFrame(
                [
                    {
                        "id": c.instrument_id,
                        "symbol": c.symbol,
                        "side": 1,
                        "contracts": quantity,
                        "premium": c.ask if c.ask is not None else 0.0,
                        "strike": c.strike,
                    }
                    for c in selected
                ]
            ),
            disabled=["id", "symbol"],
            hide_index=True,
            width="stretch",
            key="leg_editor",
        ).to_dict("records")
    if st.button("محاسبه استراتژی", type="primary"):
        if not selected:
            raise ValueError("حداقل یک قرارداد انتخاب کنید")
        if name == "Custom":
            legs = []
            for row in rows:
                contract = next(c for c in selected if c.instrument_id == row["id"])
                if hypothetical:
                    contract = contract.model_copy(
                        update={
                            "strike": float(row["strike"]),
                            "exercise_style": "european",
                            "settlement": "cash",
                            "terms_verified": True,
                            "terms_source": "user hypothetical European cash assumption",
                        }
                    )
                    premium = float(row["premium"])
                else:
                    if float(row["strike"]) != contract.strike:
                        raise ValueError("تغییر قیمت اعمال فقط در سناریوی فرضی مجاز است")
                    premium = contract.execution_premium(
                        int(row["side"]), int(row["contracts"]), slippage, cutoff
                    )
                legs.append(
                    OptionLeg(contract, int(row["side"]), premium, int(row["contracts"]), entry_fee, exit_fee)
                )
            strategy = Strategy(
                "سفارشی",
                legs,
                committed_capital=capital or None,
                capital_source=capital_source,
                scenario_only=hypothetical,
            )
        else:
            if hypothetical:
                raise ValueError("برای مفروضات فرضی از سازنده سفارشی استفاده کنید")
            strategy = build_strategy(
                name,
                selected,
                spot,
                cutoff,
                quantity,
                entry_fee,
                exit_fee,
                slippage,
                capital or None,
                capital_source,
            )
        st.session_state["strategy"] = strategy
    strategy = st.session_state.get("strategy")
    if strategy is None or strategy.legs[0].contract.underlying_id != str(history.instrument_id.iloc[-1]):
        return
    st.caption("سناریوی فرضی کاربر" if strategy.scenario_only else "قیمت ورود از سمت قابل اجرای خرید/فروش")
    details = strategy.theoretical()
    cols = st.columns(4)
    cols[0].metric("جریان نقد اولیه", f"{strategy.initial_cash_flow / unit:,.1f}")
    cols[1].metric("بیشینه زیان نظری", f"{details['max_loss'] / unit:,.1f}")
    cols[2].metric("بیشینه سود نظری", f"{details['max_profit'] / unit:,.1f}")
    cols[3].metric("سرمایه مبنا", f"{strategy.capital / unit:,.1f}" if strategy.capital else "ناموجود")
    st.write("نقاط سربه‌سر:", [round(value / unit, 2) for value in details["break_even"]])
    if not strategy.capital:
        st.info(
            "برای موقعیت دارای فروش، سرمایه یا وجه تضمین تأییدشده لازم است؛ اعتبار دریافتی مخرج بازده نیست."
        )
    cols = st.columns(4)
    price_change = cols[0].slider("تغییر قیمت پایه ٪", -80.0, 100.0, 0.0)
    volatility = cols[1].slider("نوسان سالانه ٪", 1.0, 200.0, 30.0) / 100
    shock = cols[2].slider("شوک نوسان (واحد درصد)", -29.0, 100.0, 0.0) / 100
    rate = cols[3].number_input("نرخ بدون ریسک پیوسته", 0.0, 2.0, 0.30, key="builder_rate")
    days = st.slider(
        "روز تقویمی تا سررسید",
        0,
        max(1, (strategy.expiration - cutoff).days),
        max(0, (strategy.expiration - cutoff).days),
    )
    grid = np.linspace(max(0.01, spot * 0.2), spot * 2, 250)
    figure = go.Figure()
    for i, leg in enumerate(strategy.legs):
        figure.add_scatter(x=grid / unit, y=leg.payoff(grid) / unit, name=f"پای {i + 1}", line_dash="dot")
    figure.add_scatter(
        x=grid / unit, y=strategy.payoff(grid) / unit, name="سود و زیان خالص سررسید", line_width=3
    )
    figure.add_hline(y=0)
    for value in details["break_even"]:
        figure.add_vline(x=value / unit, line_dash="dash")
    chart(figure, "payoff")
    from datetime import timedelta

    scenario_date = strategy.expiration - timedelta(days=days)
    try:
        value = strategy.mark_to_market(
            spot * (1 + price_change / 100), scenario_date, max(0.001, volatility + shock), rate
        )
        st.metric("سود و زیان ارزش‌گذاری سناریو", f"{float(value) / unit:,.1f}")
        day_grid = np.unique(np.linspace(0, max(1, (strategy.expiration - cutoff).days), 20).astype(int))
        heat = np.array(
            [
                strategy.mark_to_market(
                    grid, strategy.expiration - timedelta(days=int(d)), max(0.001, volatility + shock), rate
                )
                for d in day_grid
            ]
        )
        chart(
            go.Figure(go.Heatmap(x=grid / unit, y=day_grid, z=heat / unit, colorscale="RdYlGn", zmid=0)),
            "price_time_heatmap",
        )
        vols = np.linspace(0.05, 1.5, 25)
        heat = np.array([strategy.mark_to_market(grid, scenario_date, vol, rate) for vol in vols])
        chart(
            go.Figure(go.Heatmap(x=grid / unit, y=vols, z=heat / unit, colorscale="RdYlGn", zmid=0)),
            "price_vol_heatmap",
        )
        st.caption(
            "ارزش‌گذاری قبل از سررسید: BSM اروپایی، نرخ و نوسان ثابت، بازده نقدی صفر؛ اجرای بازار تضمین نمی‌شود."
        )
    except UnsupportedContract as exc:
        st.warning(str(exc))
    if st.button("افزودن به مقایسه تصمیم"):
        basket = st.session_state.setdefault("strategies", [])
        basket.append(strategy)
        st.success(f"{len(basket)} استراتژی در سبد مقایسه")
    download(
        pd.DataFrame({"underlying_price": grid, "pnl": strategy.payoff(grid)}), "payoff", "payoff_download"
    )


def decision_page(service, symbol, unit):
    history = service.history(symbol) if symbol else pd.DataFrame()
    previous_forecast = st.session_state.get("forecast")
    if not history.empty and (previous_forecast is None or
                             previous_forecast.metadata["instrument_id"] != str(history.instrument_id.iloc[-1])):
        from .forecasting import Forecast
        saved = service.settings.data_dir / "forecasts" / f"{history.instrument_id.iloc[-1]}_21.xlsx"
        if saved.exists():
            candidate = Forecast.from_excel(saved)
            if candidate.metadata["input_data_cutoff"] == str(history.trading_date.iloc[-1]):
                st.session_state["forecast"] = candidate
    if not history.empty:
        saved_scenarios = service.settings.data_dir / "options" / f"{history.instrument_id.iloc[-1]}_21.xlsx"
        if saved_scenarios.exists():
            with st.expander("آخرین سناریوهای ذخیره‌شده پایپلاین"):
                st.caption("گزارش ذخیره‌شده است؛ تاریخ داده و مفروضات را بررسی کنید.")
                show_table(pd.read_excel(saved_scenarios, sheet_name="scenarios__0000"), unit)
    forecast = st.session_state.get("forecast")
    strategies = st.session_state.get("strategies", [])
    if forecast is None or not strategies:
        st.info("ابتدا در پیش‌بینی توزیع تولید کنید و در سازنده، استراتژی را به مقایسه اضافه کنید.")
        return
    history = service.history(symbol) if symbol else pd.DataFrame()
    if history.empty or str(history.instrument_id.iloc[-1]) != forecast.metadata["instrument_id"]:
        st.info("پیش‌بینی ذخیره‌شده متعلق به نماد انتخابی نیست.")
        return
    cols = st.columns(4)
    budget = cols[0].number_input("بودجه سرمایه (ریال)", 1.0, value=100_000_000.0)
    volatility = cols[1].number_input("نوسان ارزش‌گذاری سالانه", 0.01, 3.0, 0.30)
    rate = cols[2].number_input("نرخ پیوسته سالانه", 0.0, 2.0, 0.30, key="decision_rate")
    haircut = cols[3].slider("هزینه نقدشوندگی ارزش مدل ٪", 0.0, 50.0, 1.0) / 100
    target = st.number_input("بازده هدف ٪", value=10.0)
    loss_threshold = st.number_input("آستانه زیان ٪", min_value=0.0, value=10.0)
    objective = st.selectbox("معیار رتبه‌بندی", list(OBJECTIVES), format_func=lambda x: LABELS.get(x, x))
    weights = (0.4, 0.3, 0.3)
    if objective == "weighted_score":
        a = st.slider("وزن بازده", 0.0, 1.0, 0.4)
        b = st.slider("وزن احتمال سود", 0.0, 1.0 - a, min(0.3, 1.0 - a))
        weights = (a, b, 1.0 - a - b)
    results, distributions = [], {}
    errors = []
    for index, strategy in enumerate(strategies):
        horizons = {h for h in (1, 5, 21) if h <= len(forecast.dates)}
        if strategy.expiration in forecast.dates:
            horizons.add(forecast.dates.index(strategy.expiration) + 1)
        else:
            errors.append(
                {
                    "strategy": strategy.name,
                    "reason": "توزیع هم‌تاریخ سررسید موجود نیست؛ بازده سررسید محاسبه نشد",
                }
            )
        for horizon in sorted(horizons):
            try:
                result, pnl = evaluate_strategy(
                    strategy,
                    forecast,
                    horizon,
                    volatility,
                    rate,
                    liquidation_haircut=haircut,
                    target_return=target,
                    loss_threshold=loss_threshold,
                )
                result["strategy"] = f"{index + 1}. {STRATEGY_LABELS.get(strategy.name, strategy.name)}"
                result["feasible"] = not strategy.scenario_only
                results.append(result)
                distributions[(result["strategy"], horizon)] = pnl
            except ValueError as exc:
                errors.append({"strategy": strategy.name, "horizon": horizon, "reason": str(exc)})
    st.caption(
        f"توزیع فیزیکی مدل: {forecast.metadata['version']} | برش داده: {forecast.metadata['input_data_cutoff']}"
    )
    st.warning("رتبه‌ها وابسته به مدل، مفروضات نقدشوندگی و ارزش‌گذاری هستند و تضمین عملکرد نیستند.")
    if errors:
        with st.expander("محدودیت‌ها و افق‌های ناموجود"):
            show_table(pd.DataFrame(errors))
    if not results:
        st.info("هیچ ارزیابی معتبر با داده‌های فعلی ممکن نیست.")
        return
    all_results = pd.DataFrame(results)
    horizon = st.selectbox("افق مقایسه (جلسه)", sorted(all_results.horizon.unique()))
    ranking = rank_strategies(
        all_results[all_results.horizon.eq(horizon)].to_dict("records"), objective, budget, weights
    )
    show_table(ranking, unit)
    if not ranking.empty:
        view = ranking.copy()
        view.cvar_loss /= unit
        chart(
            px.scatter(
                view,
                x="cvar_loss",
                y="expected_return_pct",
                color="pareto",
                hover_name="strategy",
                labels=LABELS,
            ),
            "risk_return",
        )
        selected = st.selectbox("توزیع سود و زیان استراتژی", ranking.strategy.tolist())
        chart(
            px.histogram(x=distributions[(selected, horizon)] / unit, nbins=60, title="توزیع سود و زیان مدل"),
            "strategy_distribution",
        )
    else:
        st.info("استراتژی با سرمایه مشخص و در محدوده بودجه یافت نشد.")
    with st.expander("تمام افق‌ها و سناریوهای فرضی"):
        show_table(all_results, unit)
    download(all_results, "decision_analysis", "decision_download")


def backtest_page(service, symbol, unit):
    from .backtesting import backtest_options
    from .forecasting import ForecastEngine, ModelConfig, walk_forward_forecasts

    history = require_history(service, symbol)
    if history is None:
        return
    mode = st.radio("نوع آزمون", ["پیش‌بینی قیمت", "اختیار معامله با آرشیو واقعی"], horizontal=True)
    horizon = st.number_input("افق نگهداری (جلسه)", 1, 63, 5, key="backtest_horizon")
    initial = st.number_input("حداقل تاریخچه آموزشی", 180, max(1000, len(history)), 300)
    step = st.number_input("فاصله بازآموزی (جلسه)", 1, 126, 21)
    epochs = st.number_input("دوره‌های آموزش هر گام", 1, 300, 20, key="backtest_epochs")
    config = ModelConfig(horizon=horizon, epochs=epochs, annual_sessions=service.settings.annual_sessions)
    if mode == "پیش‌بینی قیمت":
        st.caption(
            "آزمون بازآموزی گام‌به‌گام بر تاریخچه واردشده؛ اصلاحات بعدی قیمت و سوگیری بقا ممکن است باقی بمانند."
        )
        if st.button("اجرای آزمون پیش‌بینی"):
            with st.spinner("آموزش مجدد در هر برش زمانی…"):
                result = walk_forward_forecasts(history, config, initial, step)
                identifier = str(history.instrument_id.iloc[-1])
                destination = service.settings.data_dir / "backtests" / f"forecast_walk_forward_{identifier}.xlsx"
                atomic_bytes(destination, excel_bytes({"predictions": result}))
                st.session_state["walk_forward"] = result
                st.session_state["walk_forward_id"] = identifier
        if st.session_state.get("walk_forward_id") == str(history.instrument_id.iloc[-1]):
            st.info("گزارش آزمون پیش‌بینی ذخیره شد؛ آن را در صفحه پیش‌بینی، تب ارزیابی عملکرد مدل ببینید.")
        return
    options = service.repository.read("options")
    if options.empty or not {"available_at", "trading_date", "execution_at"} <= set(options):
        st.warning(
            "آرشیو زنجیره اختیار با زمان اجرای تأییدشده پس از دسترسی به اطلاعات موجود نیست؛ آزمون تاریخی غیرفعال است."
        )
        return
    options = options[
        options.underlying_id.eq(str(history.instrument_id.iloc[-1]))
        & options.source.eq(service.settings.source)
    ]
    st.caption(
        "فعلاً انتخاب از خرید ساده اختیار خرید/فروش؛ تسویه نقدی اروپایی تأییدشده، ورود فروشنده و خروج خریدار. مسیر سرمایه بر مبنای خروج‌هاست."
    )
    capital = st.number_input("سرمایه اولیه (ریال)", 1.0, value=100_000_000.0)
    fee = st.number_input("کارمزد هر سمت هر قرارداد (ریال)", 0.0, value=0.0)
    slippage = st.slider("لغزش هر سمت ٪", 0.0, 10.0, 0.0, key="backtest_slippage") / 100
    if st.button("اجرای آزمون اختیار"):

        def forecast_fn(known):
            if len(known) < initial:
                raise ValueError("Insufficient point-in-time history")
            return (
                ForecastEngine(config).fit(known).predict(known, simulations=300, calendar=service.calendar)
            )

        def select(known, chain, forecast, budget):
            parsed, _ = contracts_from_frame(chain)
            choices = []
            for contract in parsed:
                try:
                    strategy = build_strategy(
                        "Long Call" if contract.option_type == "call" else "Long Put",
                        [contract],
                        float(known.close.iloc[-1]),
                        known.trading_date.iloc[-1],
                        entry_fee=fee,
                        exit_fee=fee,
                        slippage_pct=slippage,
                    )
                    metric, _ = evaluate_strategy(strategy, forecast, horizon)
                    if strategy.capital <= budget:
                        choices.append((strategy, metric))
                except ValueError:
                    continue
            if not choices:
                raise ValueError("No feasible historical candidate")
            return max(choices, key=lambda candidate: candidate[1]["expected_pnl"])

        with st.spinner("آزمون اطلاعات قابل دسترس و قیمت‌های تاریخی…"):
            result = backtest_options(
                history,
                options,
                forecast_fn,
                select,
                capital,
                horizon,
                step,
                slippage,
                price_versions=service.repository.read("price_versions"),
            )
            result.save(service.settings.data_dir / "backtests" / "options_backtest.xlsx")
            st.session_state["option_backtest"] = result
    result = st.session_state.get("option_backtest")
    if result:
        st.json(result.metrics)
        show_table(result.trades, unit)
        if not result.equity.empty:
            view = result.equity.copy()
            view.equity /= unit
            chart(px.line(view, x="date", y="equity", title="سرمایه در تاریخ خروج از معاملات"), "equity")
            chart(
                px.scatter(result.trades, x="expected_return_pct", y="realized_return_pct", labels=LABELS),
                "expected_realized",
            )
        show_table(result.skipped)
        download(result.trades, "backtest_trades", "backtest_download")


def advanced_data_page(service, symbol, unit):
    st.caption(
        "ورود فایل‌های Funds / Details / NAV / Options / PriceHistory / ClientHistory / Metadata؛ فایل اصلی بدون تغییر آرشیو می‌شود."
    )
    uploaded = st.file_uploader("فایل اکسل گردآورنده", type=["xlsx"])
    if uploaded is not None and st.button("اعتبارسنجی و ورود فایل"):
        content = uploaded.getvalue()
        path = service.settings.data_dir / "raw" / f"{hashlib.sha256(content).hexdigest()}.xlsx"
        atomic_bytes(path, content)
        report = service.import_file(path)
        st.success("ورود داده انجام شد؛ گزارش رکوردهای ناقص را بررسی کنید.")
        st.json(report)
    tables = service.repository.read_all()
    show_table(
        pd.DataFrame(
            [
                {"table": name, "rows": len(frame), "missing_cells": int(frame.isna().sum().sum())}
                for name, frame in tables.items()
            ]
        )
    )
    if st.button("گزارش کیفیت و جلسات مفقود"):
        st.session_state["quality_report"] = service.quality()
    if "quality_report" in st.session_state:
        show_table(st.session_state["quality_report"])
    st.caption("تقویم: شنبه تا چهارشنبه با تعطیلات واردشده؛ کامل بودن فهرست تعطیلات تأیید نشده است.")
    if service.repository.path.exists():
        st.download_button(
            "دریافت پایگاه اکسل — ریال",
            service.repository.path.read_bytes(),
            "market.xlsx",
            key="database_download",
        )
    template = {
        "Instruments": pd.DataFrame(columns=["instrument_id", "symbol", "instrument_type"]),
        "PriceHistory": pd.DataFrame(
            columns=[
                "instrument_id",
                "symbol",
                "trading_date",
                "close",
                "open",
                "high",
                "low",
                "adjusted_close",
                "volume",
                "available_at",
                "retrieved_at",
            ]
        ),
        "Options": pd.DataFrame(
            columns=[
                "instrument_id",
                "symbol",
                "underlying_id",
                "underlying_symbol",
                "option_type",
                "strike",
                "expiration",
                "multiplier",
                "bid",
                "ask",
                "bid_quantity",
                "ask_quantity",
                "volume",
                "open_interest",
                "trading_date",
                "available_at",
                "retrieved_at",
                "exercise_style",
                "settlement",
                "terms_verified",
                "terms_source",
                "execution_at",
            ]
        ),
    }
    # Collector input sheets deliberately have their canonical names (no repository shard suffix).
    from io import BytesIO

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for name, frame in template.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    st.download_button(
        "قالب خالی داده و مشخصات تأییدشده", buffer.getvalue(), "input_template.xlsx", key="template_download"
    )
    log = service.settings.data_dir / "logs" / "dss.jsonl"
    if log.exists():
        with st.expander("گزارش رویدادهای برنامه"):
            st.code("\n".join(log.read_text(encoding="utf-8").splitlines()[-30:]), language="json")





def pipeline_controls(service, symbols_text):
    from .pipeline import MarketPipelineService
    from filelock import Timeout

    st.subheader("اجرای پایپلاین")
    st.caption("تاریخچه بدون کلید ← اکسل ← ویژگی‌ها ← آموزش خودکار / بارگذاری مدل ← پیش‌بینی ← اختیار")
    refresh = st.checkbox("بازآموزی اجباری مدل‌ها (زمان‌بر)", key="pipeline_retrain")
    full, data_only, inference = st.columns(3)
    run_full = full.button("اجرای کامل پایپلاین", key="run_full_pipeline", type="primary")
    run_data = data_only.button("دریافت تمام داده‌های تاریخی", key="run_data_only")
    run_inference = inference.button("فقط اجرای پیش‌بینی", key="run_inference")
    live_col, train_col = st.columns(2)
    run_live = live_col.button("به‌روزرسانی داده جاری", key="run_live_only")
    run_train = train_col.button("آموزش مجدد مدل‌ها", key="run_train_only")
    if run_full or run_data or run_inference or run_live or run_train:
        progress, current, table = st.progress(0), st.empty(), st.empty()
        def update(run):
            stages = run["stages"]
            progress.progress(min(.95, len(stages) / (8 + len(stages))), text=stages[-1]["name"])
            current.caption(f"زمان: {run['elapsed_seconds']} ثانیه | " + " | ".join(
                f"{LABELS.get(k, k)}: {v}" for k, v in run["counts"].items()))
            table.dataframe(pd.DataFrame(stages), hide_index=True, width="stretch")
        try:
            symbols = [s.strip() for s in symbols_text.replace("،", ",").split(",") if s.strip()]
            report = MarketPipelineService(service).run_full_pipeline(
                symbols or None, historical_backfill="auto" if run_full or run_data else "never",
                update_live=bool(run_full or run_live), update_forecasts=not (run_data or run_live),
                retrain_policy="always" if refresh or run_train else "auto", progress_callback=update)
            st.session_state["pipeline_report"] = report
            for key in ("forecast", "quality_report", "walk_forward", "option_backtest", "strategy"):
                st.session_state.pop(key, None)
            st.cache_data.clear()
            st.rerun()
        except Timeout:
            st.warning("اجرای دیگری فعال است؛ تا پایان آن صبر کنید.")
    path = service.settings.data_dir / "processed" / "last_pipeline.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        st.write(f"آخرین اجرا: {report['status']} | شناسه: {report['run_id']}")
        show_table(pd.DataFrame([report["counts"]]))
        show_table(pd.DataFrame(report["stages"]))
        collection = report.get("collection", {})
        if collection:
            st.caption(f"منبع snapshot این اجرا: {collection.get('provider', 'نامشخص')}")
            if collection.get("warnings"):
                with st.expander("هشدارهای اتصال داده جاری"):
                    show_table(pd.DataFrame(collection["warnings"]))
            if collection.get("limitations"):
                st.info(collection["limitations"])
        st.download_button("دریافت گزارش اجرای پایپلاین", path.read_bytes(), "pipeline.json")


def data_page(service, symbol, unit):
    from .credentials import CredentialStore
    from .historical import TsetmcProvider, HistoricalAPIError
    st.subheader("تنظیمات و اطلاعات داده‌ها")
    st.markdown("**اتصالات داده**")
    st.info("تاریخچه TSETMC عمومی است و به کلید نیاز ندارد. کلید BrsApi فقط برای داده زنده و جایگزین اختیاری است.")
    symbols_text = st.text_input("نمادهای پایپلاین با ویرگول", value=service.settings.default_symbols,
                                  key="pipeline_symbols")
    if st.button("آزمون اتصال تاریخچه — بدون کلید"):
        try:
            selected = symbols_text.replace("،", ",").split(",")[0].strip()
            provider = TsetmcProvider(service.settings.data_dir)
            identity = provider.resolve(selected)
            provider.request(f"Instrument/GetInstrumentInfo/{identity['instrument_id']}", "instrumentInfo")
            result = {"status": "SUCCESS", "tested_at": pd.Timestamp.now(tz="UTC").isoformat(), **identity}
            st.success(f"اتصال و هویت نماد تأیید شد: {identity['symbol']}")
        except HistoricalAPIError as exc:
            result = {"status": "FAILED", "message": str(exc)}
            st.warning(str(exc))
        atomic_bytes(service.settings.data_dir / "processed/historical_connection.json",
                     json.dumps(result, ensure_ascii=False).encode("utf-8"))
    with st.form("live_key_configuration", clear_on_submit=True):
        entered = st.text_input("کلید API بازار زنده BrsApi", type="password", key="new_live_api_key")
        save = st.form_submit_button("ذخیره امن تنظیمات")
        if save:
            try:
                CredentialStore(service.settings.data_dir).save(entered)
                st.success("کلید به‌صورت رمز‌شده برای حساب ویندوز ذخیره شد.")
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
    key_available = bool(service.live_key())
    st.caption("BrsApi: کلید تنظیم شده" if key_available else "BrsApi: کلید زنده تنظیم نشده؛ تاریخچه همچنان قابل دریافت است.")
    from .credentials import CredentialStore
    credential_status = CredentialStore(service.settings.data_dir).status()
    if credential_status.get("message"):
        st.warning(credential_status["message"])
    if st.button("آزمون اتصال بازار زنده", disabled=not key_available):
        try:
            report = service.test_live_connection()
            if report["status"] == "SUCCESS":
                st.success("اتصال و ساختار پاسخ تأیید شد؛ تازگی هر endpoint را در جدول بررسی کنید.")
            else:
                st.warning("اتصال بازار زنده تأیید نشد؛ علت در گزارش اتصال ثبت شده است.")
        except Exception as exc:
            from leveraged_funds_collector import MarketDataError
            message = str(exc) if isinstance(exc, MarketDataError) else "اتصال تأیید نشد؛ کلید، مجوز سرویس و شبکه را بررسی کنید."
            status_path = service.settings.data_dir / "processed/live_connection.json"
            previous = json.loads(status_path.read_text("utf-8")) if status_path.exists() else {}
            previous.update(status="FAILED", message=message, last_attempt_at=pd.Timestamp.now(tz="UTC").isoformat())
            atomic_bytes(status_path, json.dumps(previous, ensure_ascii=False).encode("utf-8"))
            st.warning(message)
    if st.button("بررسی اتصال داده جاری", key="diagnose_live"):
        from .live import diagnose_market_api
        symbols = [item.strip() for item in symbols_text.replace("،", ",").split(",") if item.strip()]
        results = diagnose_market_api("BrsApi", service.live_key(), symbols=symbols,
                                     max_age_hours=service.settings.live_max_age_hours)
        results += diagnose_market_api("TSETMC", symbols=symbols, data_dir=service.settings.data_dir,
                                      max_age_hours=service.settings.live_max_age_hours)
        atomic_bytes(service.settings.data_dir / "processed/live_diagnostics.json",
                     json.dumps(results, ensure_ascii=False).encode("utf-8"))
    diagnostic = service.settings.data_dir / "processed/live_diagnostics.json"
    if diagnostic.exists():
        st.markdown("**وضعیت API داده جاری**")
        show_table(pd.DataFrame(json.loads(diagnostic.read_text("utf-8"))))
    from .ui import typography_status
    st.caption(typography_status())
    for filename, label in (("historical_connection.json", "وضعیت اتصال تاریخچه"),
                            ("live_connection.json", "آخرین اتصال موفق بازار زنده")):
        path = service.settings.data_dir / "processed" / filename
        if path.exists():
            with st.expander(label):
                st.json(json.loads(path.read_text("utf-8")))
    st.divider()
    pipeline_controls(service, symbols_text)
    st.divider()
    st.subheader("وضعیت داده‌ها")
    coverage = service.coverage()
    show_table(coverage)
    download(coverage, "historical_coverage", "coverage_download")
    st.caption("شکاف جلسات برآوردی شامل تعطیلات تأییدنشده و توقف نماد هم می‌شود؛ تاریخچه به‌صورت مصنوعی پر نمی‌شود.")
    with st.expander("علت حذف رکوردهای API و محدودیت‌های داده"):
        removed = []
        for identifier in coverage.get("instrument_id", pd.Series(dtype=str)).dropna():
            path = service.settings.data_dir / "processed" / f"history_{identifier}.json"
            if path.exists():
                report = json.loads(path.read_text("utf-8"))
                issues = pd.DataFrame(report.get("issues", []))
                if not issues.empty:
                    issues["symbol"] = report["symbol"]
                    removed.append(issues)
        if removed:
            show_table(pd.concat(removed, ignore_index=True))
    tables = service.repository.read_all()
    prices = tables.get("prices", pd.DataFrame())
    if prices.empty:
        st.info("داده تاریخی موجود نیست. برای دریافت خودکار اطلاعات و آماده‌سازی مدل، اجرای کامل پایپلاین را انتخاب کنید.")
    else:
        cols = st.columns(4)
        cols[0].metric("مشاهدات تاریخی", f"{len(prices):,}")
        cols[1].metric("نمادهای تحت پوشش", prices.instrument_id.nunique())
        cols[2].metric("ابتدای تاریخچه", jalali(prices.trading_date.min()))
        cols[3].metric("آخرین معامله تاریخی", jalali(prices.trading_date.max()))
        st.caption(f"آخرین دریافت تاریخچه: {prices.retrieved_at.max()}")
    snapshots = tables.get("snapshots", pd.DataFrame())
    if not snapshots.empty:
        st.caption(f"آخرین دریافت snapshot زنده: {snapshots.retrieved_at.max()}؛ زمان دریافت، زمان معامله نیست.")
    versions = list(service.settings.model_dir.glob("*/*/model.json"))
    st.caption(f"نسخه‌های آموزش‌دیده: {len(versions)} | پایگاه: {service.repository.path}")
    results = list((service.settings.data_dir / "forecasts").glob("*.xlsx"))
    if results:
        st.caption(f"فایل‌های پیش‌بینی: {len(results)} | آخرین ثبت: " +
                   str(pd.Timestamp(max(p.stat().st_mtime for p in results), unit="s", tz="UTC")))
    with st.expander("ابزارهای پیشرفته: بازیابی اکسل، کنترل کیفیت و خروجی"):
        advanced_data_page(service, symbol, unit)


PAGE_PATHS = ["market", "forecast", "options", "strategy", "decision", "backtest", "settings"]


def render_page(index):
    from .ui import shell
    service = MarketService()
    symbol = st.session_state.get("selected_symbol")
    unit = 10 if st.session_state.get("display_unit") == "تومان" else 1
    shell(PAGES[index])
    try:
        [market_page, forecast_page, option_page, strategy_page, decision_page, backtest_page, data_page][index](service, symbol, unit)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"عملیات انجام نشد ({type(exc).__name__}). گزارش اجرا را بررسی کنید.")


def main():
    from pathlib import Path
    st.set_page_config(page_title="سامانه تصمیم‌یار بازار ایران", page_icon="📊", layout="wide")
    service = MarketService()
    st.sidebar.title("فیلترهای تحلیل")
    instruments = service.instruments()
    choices = sorted(instruments.symbol.unique()) if not instruments.empty else [
        s.strip() for s in service.settings.default_symbols.split(",") if s.strip()]
    if st.session_state.get("selected_symbol") not in choices:
        st.session_state["selected_symbol"] = choices[0] if choices else None
    if choices:
        st.sidebar.selectbox("نماد", choices, key="selected_symbol")
    st.sidebar.radio("واحد نمایش مبالغ", ["ریال", "تومان"], horizontal=True, key="display_unit")
    st.sidebar.caption("هر تومان = ۱۰ ریال؛ ذخیره‌سازی و ورودی سرمایه همواره ریال است.")
    pages = [st.Page(str(Path(__file__).parent / "pages" / f"{slug}.py"), title=title,
                     url_path=slug, default=i == 0) for i, (slug, title) in enumerate(zip(PAGE_PATHS, PAGES))]
    st.navigation(pages, position="top").run()
