"""Persian Streamlit presentation. Financial/model logic lives in independent modules."""

import hashlib
import json
import os

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
    "بازار اختیار معامله",
    "ساخت استراتژی",
    "پشتیبانی تصمیم",
    "آزمون تاریخی",
    "مدیریت داده",
]
LABELS = {
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
        template="plotly_white",
        font={"family": "Tahoma, Arial", "size": 13},
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
        st.info("داده تاریخی موجود نیست. در مدیریت داده، فایل گردآورنده را وارد یا تاریخچه را دریافت کنید.")
        return None
    age = (pd.Timestamp.now(tz="Asia/Tehran").date() - parse_date(history.trading_date.iloc[-1])).days
    if age > 3:
        st.warning(f"آخرین مشاهده {age} روز تقویمی قبل است؛ نتایج با این تاریخ محاسبه می‌شوند.")
    return history


def market_page(service, symbol, unit):
    history = require_history(service, symbol)
    if history is None:
        return
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
    chart(figure, "market_history")
    if "volume" in view:
        chart(px.bar(view, x="trading_date", y="volume", labels=LABELS, title="حجم معاملات"), "market_volume")
    choices = service.instruments().symbol.unique().tolist()
    comparison = st.multiselect("مقایسه نمادها — قیمت ابتدای بازه برابر ۱۰۰", choices, default=[symbol])
    combined = []
    for selected in comparison:
        data = service.history(selected).tail(lookback)
        if not data.empty:
            data = data.assign(normalized=data.close / data.close.iloc[0] * 100)
            combined.append(data)
    if combined:
        chart(
            px.line(pd.concat(combined), x="trading_date", y="normalized", color="symbol", labels=LABELS),
            "market_comparison",
        )
    show_table(features.tail(30), unit)
    download(features, "indicators", "market_export")


def forecast_page(service, symbol, unit):
    from .forecasting import ForecastEngine, ModelConfig

    history = require_history(service, symbol)
    if history is None:
        return
    identifier = str(history.instrument_id.iloc[-1])
    with st.expander("آموزش مدل", expanded=not bool(service.models(identifier))):
        cols = st.columns(4)
        horizon_train = cols[0].number_input("حداکثر افق مدل (جلسه)", 1, 126, 21)
        window = cols[1].number_input("طول پنجره", 5, 200, 30)
        epochs = cols[2].number_input("حداکثر دوره آموزش", 1, 500, 60)
        lookback = cols[3].number_input(
            "مشاهدات مورد استفاده", 50, max(500, len(history)), min(1000, max(50, len(history)))
        )
        attention = st.checkbox("لایه توجه")
        if st.button("آموزش و ارزیابی زمانی", type="primary"):
            with st.spinner("آموزش و محاسبه نتایج خارج از نمونه…"):
                model = ForecastEngine(
                    ModelConfig(
                        horizon=horizon_train,
                        window=window,
                        epochs=epochs,
                        attention=attention,
                        annual_sessions=service.settings.annual_sessions,
                    )
                ).fit(history.tail(lookback))
                model.save(service.settings.model_dir)
            st.success("مدل و گزارش ارزیابی ذخیره شد.")
    versions = service.models(identifier)
    if not versions:
        st.info("هنوز مدل آموزش‌دیده‌ای برای این نماد وجود ندارد. داده کم، پیش‌بینی تولید نمی‌کند.")
        return
    selected = st.selectbox("نسخه مدل", versions, format_func=lambda path: path.parent.name)
    model = ForecastEngine.load(selected.parent)
    cols = st.columns(4)
    horizon = cols[0].number_input(
        "افق پیش‌بینی (۱ روز، ۵ هفته، ۲۱ ماه)", 1, model.config.horizon, model.config.horizon
    )
    confidence = cols[1].slider("سطح بازه پیش‌بینی", 0.50, 0.99, 0.90)
    simulations = cols[2].number_input("تعداد مسیر", 100, 50000, 2000, step=100)
    vol_scale = cols[3].slider("ضریب سناریوی نوسان", 0.25, 3.0, 1.0)
    target = (
        st.number_input("قیمت هدف (واحد نمایش)", min_value=0.01, value=float(history.close.iloc[-1] / unit))
        * unit
    )
    if st.button("تولید توزیع پیش‌بینی"):
        with st.spinner("نمونه‌گیری از مدل و خطاهای کالیبراسیون…"):
            st.session_state["forecast"] = model.predict(
                history, horizon, simulations, vol_scale, service.calendar
            )
    forecast = st.session_state.get("forecast")
    if forecast is None or forecast.metadata["instrument_id"] != identifier:
        return
    st.caption(
        f"نسخه نتیجه: {forecast.metadata['version']} | تولید: {forecast.metadata['generated_at']} | "
        f"برش داده: {forecast.metadata['input_data_cutoff']} | افق: {forecast.metadata['horizon']} جلسه"
    )
    st.info("تاریخ جلسات آینده برآوردی است؛ تعطیلات باید در فایل تقویم ثبت شوند. مسیرها سناریوهای مدل هستند.")
    if model.metadata["version"] != forecast.metadata["version"]:
        st.warning("نتیجه نمایش‌داده‌شده متعلق به نسخه قبلی است؛ برای نسخه انتخابی دوباره تولید کنید.")
    summary = forecast.summary(confidence, target)
    figure = go.Figure()
    figure.add_scatter(
        x=history.trading_date.tail(100),
        y=history.close.tail(100) / unit,
        name="قیمت مشاهده‌شده",
        line_color="#183b56",
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
    tabs = st.tabs(["مسیرها", "چگالی و توزیع تجمعی", "بازده و نوسان", "ارزیابی خارج از نمونه"])
    with tabs[0]:
        figure = go.Figure()
        for path in forecast.paths[:100]:
            figure.add_scatter(
                x=forecast.dates, y=path / unit, line={"width": 0.6}, opacity=0.25, showlegend=False
            )
        chart(figure, "forecast_paths")
    with tabs[1]:
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
    with tabs[2]:
        chart(
            px.histogram(
                x=100 * (forecast.paths[:, -1] / forecast.metadata["spot"] - 1),
                nbins=50,
                title="بازده تجمعی سناریوها (درصد)",
            ),
            "return_histogram",
        )
        historical = engineer_features(history, service.settings.annual_sessions)
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
            * np.sqrt(service.settings.annual_sessions / np.arange(1, len(summary) + 1)),
            name="نوسان ضمنی توزیع پیش‌بینی سالانه",
        )
        chart(figure, "forecast_volatility")
        st.caption("خطای باقیمانده شامل نوآوری بازار و خطای مدل است؛ سهم dropout فقط تقریب عدم‌قطعیت مدل است.")
    with tabs[3]:
        evaluated_model = ForecastEngine.load(
            service.settings.model_dir / identifier / forecast.metadata["version"]
        )
        show_table(evaluated_model.metrics)
        st.caption("خطاها در واحد قیمت مبنای آموزش هستند؛ هیچ برتری بر مدل پایه بدون این نتایج ادعا نمی‌شود.")
        out = evaluated_model.predictions
        metric_horizon = st.selectbox("افق گزارش تاریخی", sorted(out.horizon.unique()))
        out = out[out.horizon.eq(metric_horizon)]
        chart(
            px.line(
                out,
                x="target_date",
                y=["actual", "expected", "q05", "q95"],
                title="مشاهده و پیش‌بینی خارج از نمونه",
            ),
            "forecast_oos",
        )
        st.json(evaluated_model.distribution_diagnostics)
        download(evaluated_model.metrics, "model_metrics", "metrics_download")
    show_table(summary, unit)
    st.download_button(
        "دریافت پیش‌بینی و مشخصات — ریال",
        excel_bytes({"forecast": summary, "metadata": pd.DataFrame([forecast.metadata])}),
        "forecast.xlsx",
        key="forecast_download",
    )


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
                st.session_state["walk_forward"] = walk_forward_forecasts(history, config, initial, step)
        result = st.session_state.get("walk_forward")
        if result is not None:
            show_table(result)
            chart(
                px.line(
                    result[result.horizon.eq(horizon)],
                    x="target_date",
                    y=["actual", "expected"],
                    title="نتایج خارج از نمونه",
                ),
                "walk_forward",
            )
            download(result, "walk_forward", "walk_forward_download")
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


def data_page(service, symbol, unit):
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
    include_history = st.checkbox("دریافت تاریخچه کامل قیمت و حقیقی/حقوقی")
    symbols_text = st.text_input("نمادهای دلخواه با ویرگول؛ خالی برای کشف صندوق‌های اهرمی")
    key_available = bool(os.environ.get("BRS_API_KEY"))
    st.write(
        "BrsApi:",
        "کلید تنظیم شده؛ دسترسی در زمان درخواست آزموده می‌شود" if key_available else "نیازمند BRS_API_KEY",
    )
    if st.button("به‌روزرسانی یک‌کلیکی", disabled=not key_available):
        with st.spinner("دریافت و ثبت اتمی مشاهدات…"):
            symbols = [s.strip() for s in symbols_text.replace("،", ",").split(",") if s.strip()]
            report = service.update(include_history, tickers=symbols or None)
        st.json(report)
        st.success("به‌روزرسانی پایان یافت. برای مشاهده نمادهای جدید صفحه را بازخوانی کنید.")
    status = service.settings.data_dir / "processed" / "last_update.json"
    if status.exists():
        st.json(json.loads(status.read_text(encoding="utf-8")))
    st.info(
        "TSETMC مستقیم از این محیط تأیید نشد و فعال نیست. قرارداد یا تاریخ ناموجود حدس زده نمی‌شود. دریافت مستقیم گزارش‌های کدال در این نسخه فعال نیست."
    )
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


def main():
    st.set_page_config(page_title="سامانه تصمیم‌یار بازار ایران", page_icon="📊", layout="wide")
    st.markdown(
        """<style>
        .stApp {direction:rtl;font-family:Tahoma,Arial,sans-serif;}
        [data-testid="stSidebar"] {direction:rtl;background:#f3f6fa;}
        h1,h2,h3,p,label {font-family:Tahoma,Arial,sans-serif!important;}
        [data-testid="stMetric"] {background:#f3f6fa;padding:1rem;border-radius:10px;}
        [data-testid="stDataFrame"],.js-plotly-plot,pre,code {direction:ltr;}
        </style>""",
        unsafe_allow_html=True,
    )
    service = MarketService()
    st.sidebar.title("تصمیم‌یار بازار ایران")
    page = st.sidebar.radio("بخش سامانه", PAGES)
    instruments = service.instruments()
    symbol = (
        st.sidebar.selectbox("نماد", sorted(instruments.symbol.unique())) if not instruments.empty else None
    )
    unit_name = st.sidebar.radio("واحد نمایش مبالغ", ["ریال", "تومان"], horizontal=True)
    unit = 10 if unit_name == "تومان" else 1
    st.sidebar.caption("هر تومان = ۱۰ ریال. ذخیره‌سازی و ورودی کارمزد/سرمایه همواره ریال است.")
    st.title(page)
    st.caption("مشاهدات واقعی، پیش‌بینی خارج از نمونه و سناریوهای مدل با برچسب جداگانه نمایش داده می‌شوند.")
    pages = dict(
        zip(
            PAGES,
            [market_page, forecast_page, option_page, strategy_page, decision_page, backtest_page, data_page],
        )
    )
    try:
        pages[page](service, symbol, unit)
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        st.error(str(exc))
    except Exception as exc:
        # Never echo networking exceptions/credentials into the UI.
        st.error(f"عملیات انجام نشد ({type(exc).__name__}). گزارش خطا و دسترسی فایل/شبکه را بررسی کنید.")
