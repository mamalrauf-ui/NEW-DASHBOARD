# -*- coding: utf-8 -*-
"""
داشبورد بازار اختیار معامله (آپشن) بورس تهران
با استفاده از کتابخانه oxtapus  (https://github.com/yghaderi/oxtapus)

نصب پیش‌نیازها:
    pip install oxtapus streamlit pandas plotly xlsxwriter

اجرا:
    streamlit run option_dashboard.py
"""

import io
import re
import math
import inspect
import datetime as dt

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="داشبورد اختیار معامله", layout="wide")

# ============================================================================
# 1) اتصال به oxtapus
# ============================================================================
@st.cache_resource
def get_client():
    from oxtapus import TSETMC
    return TSETMC()


def find_option_method(client):
    """
    متدهای مرتبط با آپشن را پیدا می‌کند و آن‌هایی که بدون آرگومان اجباری
    قابل فراخوانی هستند (یعنی مناسب دیده‌بان کل بازار، نه یک نماد خاص) را
    در اولویت قرار می‌دهد. متدهایی مثل specific_option_data که به ins_id
    نیاز دارند، در انتهای لیست می‌آیند و علامت‌گذاری می‌شوند.
    """
    candidates = [m for m in dir(client) if not m.startswith("_")]
    option_like = [m for m in candidates if re.search(r"option", m, re.I)]

    no_arg, needs_arg = [], []
    for name in option_like:
        try:
            sig = inspect.signature(getattr(client, name))
            required = [
                p for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            ]
            (needs_arg if required else no_arg).append(name)
        except (TypeError, ValueError):
            no_arg.append(name)  # اگر امضا قابل بررسی نبود، خوش‌بینانه نگه‌اش می‌داریم

    # اول متدهای بدون‌آرگومان (مناسب Watchlist)، بعد بقیه با برچسب هشدار
    ordered = no_arg + [f"{m}  (نیاز به آرگومان اضافه دارد)" for m in needs_arg]
    return ordered, candidates, no_arg


@st.cache_data(ttl=60)
def load_option_data(method_name: str):
    client = get_client()
    method = getattr(client, method_name)
    df = method()
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df


# ============================================================================
# 2) شناسایی خودکار ستون‌های خام
# ============================================================================
COLUMN_GUESSES = {
    "symbol":            r"symbol|l18|ticker|نماد$|^نماد|ins_code",
    "underlying":        r"base|underlying|پایه",
    "contract_type":     r"type|call|put|نوع|ua_type",
    "underlying_close":  r"(base|underlying).*(close|price|pl)|py_underlying",
    "option_close":      r"^(close|pl|pdc|py)$|option.*close",
    "strike":            r"strike|exercise|اعمال",
    "maturity_date":     r"maturity|expire|end_date|سررسید",
    "days_remaining":    r"day.*(remain|left|maturity)|remain.*day|روز.*ماند|dtm",
    "volume":            r"^vol|volume|حجم",
    "value":             r"value|turnover|ارزش.*معام",
    "oi":               r"\boi\b|open_interest|موقعیت.*باز",
}


def guess_columns(df):
    guesses = {}
    for key, pattern in COLUMN_GUESSES.items():
        match = next((c for c in df.columns if re.search(pattern, c, re.I)), None)
        guesses[key] = match
    return guesses


# ============================================================================
# 3) توابع بلک-شولز (بدون نیاز به scipy)
# ============================================================================
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def norm_pdf(x):
    return math.exp(-x ** 2 / 2) / math.sqrt(2 * math.pi)


def bs_price(S, K, T, r, sigma, option_type):
    if not all([S, K, T, sigma]) or T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "Call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_greeks(S, K, T, r, sigma, option_type):
    empty = {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}
    if not all([S, K, T, sigma]) or T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return empty
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = norm_pdf(d1)
    gamma = pdf_d1 / (S * sigma * math.sqrt(T))
    vega = S * pdf_d1 * math.sqrt(T) / 100
    if option_type == "Call":
        delta = norm_cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm_cdf(d2)) / 365
        rho = K * T * math.exp(-r * T) * norm_cdf(d2) / 100
    else:
        delta = norm_cdf(d1) - 1
        theta = (-(S * pdf_d1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm_cdf(-d2)) / 365
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2) / 100
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


def implied_vol(price, S, K, T, r, option_type, tol=1e-4, max_iter=60):
    if not all([price, S, K, T]) or price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    low, high = 1e-4, 5.0
    mid = 0.3
    for _ in range(max_iter):
        mid = (low + high) / 2
        est = bs_price(S, K, T, r, mid, option_type)
        if est is None:
            return None
        if abs(est - price) < tol:
            return mid
        if est > price:
            high = mid
        else:
            low = mid
    return mid


# ============================================================================
# 4) رابط کاربری — سایدبار
# ============================================================================
st.title("📊 داشبورد بازار اختیار معامله (بورس تهران)")
st.caption("منبع داده: کتابخانه‌ی oxtapus — tsetmc.com")

client = get_client()
option_like, all_methods, no_arg_methods = find_option_method(client)

with st.sidebar:
    st.header("تنظیمات")

    if not option_like:
        st.error("هیچ متدی با کلمه‌ی 'option' پیدا نشد؛ از لیست پایین انتخاب کن.")
        method_label = st.selectbox("انتخاب متد دستی", all_methods)
    else:
        method_label = st.selectbox(
            "متد داده‌ی آپشن", option_like, index=0,
            help="متدهایی که برچسب «نیاز به آرگومان اضافه دارد» دارند، برای یک نماد خاص هستند "
                 "و برای Watchlist کل بازار مناسب نیستند."
        )
    # حذف برچسب هشدار برای رسیدن به نام واقعی متد
    method_name = method_label.split("  (")[0]
    if method_label != method_name:
        st.warning(
            "این متد به آرگومان اضافه (مثل شناسه‌ی یک نماد خاص) نیاز دارد و با کلیک روی "
            "«به‌روزرسانی داده» خطا می‌دهد. برای دیده‌بان کل بازار، یکی از متدهای بدون برچسب را انتخاب کن."
        )

    if st.button("🔄 به‌روزرسانی داده", use_container_width=True):
        st.cache_data.clear()

    st.divider()
    risk_free_rate = st.number_input(
        "نرخ بدون ریسک سالانه (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0,
        help="برای محاسبه‌ی نوسان ضمنی، Greeks و پوت-کال پاریتی استفاده می‌شود."
    ) / 100

    with st.expander("همه‌ی متدهای موجود در TSETMC()"):
        st.write(all_methods)

# ============================================================================
# 5) دریافت داده خام
# ============================================================================
try:
    raw = load_option_data(method_name)
except Exception as e:
    st.error(
        f"خطا در فراخوانی متد `{method_name}`: {e}\n\n"
        f"برای دیدن امضای دقیق تابع: `help(TSETMC().{method_name})`"
    )
    st.stop()

if raw is None or len(raw) == 0:
    st.warning("داده‌ای برگردانده نشد (ممکن است بازار بسته باشد).")
    st.stop()

guesses = guess_columns(raw)

with st.sidebar:
    st.divider()
    st.subheader("نگاشت ستون‌ها")
    st.caption("اگر تشخیص خودکار اشتباه بود، اینجا اصلاحش کن.")
    col_options = ["— ندارد —"] + list(raw.columns)
    mapping = {}
    for key, label in [
        ("symbol", "نماد"), ("underlying", "دارایی پایه"),
        ("contract_type", "نوع قرارداد (Call/Put)"),
        ("underlying_close", "قیمت پایانی دارایی پایه"),
        ("option_close", "قیمت پایانی اختیار"),
        ("strike", "قیمت اعمال"), ("maturity_date", "تاریخ سررسید"),
        ("days_remaining", "روز مانده"), ("volume", "حجم"),
        ("value", "ارزش معاملات"), ("oi", "OI"),
    ]:
        default = guesses.get(key)
        idx = col_options.index(default) if default in col_options else 0
        chosen = st.selectbox(label, col_options, index=idx, key=f"map_{key}")
        mapping[key] = None if chosen == "— ندارد —" else chosen

# ============================================================================
# 6) ساخت دیتافریم اصلی «core» (نام‌های انگلیسی داخلی) + محاسبات مشتق‌شده
# ============================================================================
core = pd.DataFrame(index=raw.index)
for key, col in mapping.items():
    if col:
        core[key] = raw[col]

if "contract_type" in core.columns:
    def norm_type(v):
        s = str(v).lower()
        if re.search(r"call|خرید|c$", s):
            return "Call"
        if re.search(r"put|فروش|p$", s):
            return "Put"
        return str(v)
    core["contract_type"] = core["contract_type"].apply(norm_type)

for key in ["underlying_close", "option_close", "strike", "days_remaining", "volume", "value", "oi"]:
    if key in core.columns:
        core[key] = pd.to_numeric(core[key], errors="coerce")

has_pricing = {"underlying_close", "strike", "option_close", "contract_type"} <= set(core.columns)
if has_pricing:
    def intrinsic(row):
        if row["contract_type"] == "Call":
            return max(row["underlying_close"] - row["strike"], 0)
        if row["contract_type"] == "Put":
            return max(row["strike"] - row["underlying_close"], 0)
        return None
    core["intrinsic_value"] = core.apply(intrinsic, axis=1)
    core["time_value"] = core["option_close"] - core["intrinsic_value"]

    def moneyness(row):
        if pd.isna(row["intrinsic_value"]):
            return None
        if row["intrinsic_value"] > 0:
            return "ITM"
        if row["strike"] and abs(row["underlying_close"] - row["strike"]) / row["strike"] < 0.02:
            return "ATM"
        return "OTM"
    core["moneyness"] = core.apply(moneyness, axis=1)

has_yield_inputs = {"option_close", "strike", "underlying_close", "days_remaining"} <= set(core.columns)
if has_yield_inputs:
    def ytm(row):
        if not row["days_remaining"] or row["days_remaining"] <= 0 or not row["underlying_close"]:
            return None
        base = (row["option_close"] + row["strike"] - row["underlying_close"]) / row["underlying_close"]
        return base * (365 / row["days_remaining"]) * 100
    core["yield_to_maturity"] = core.apply(ytm, axis=1)

# --- نوسان ضمنی و Greeks -----------------------------------------------------
has_bs_inputs = {"underlying_close", "strike", "option_close", "days_remaining", "contract_type"} <= set(core.columns)
if has_bs_inputs:
    def compute_iv(row):
        T = (row["days_remaining"] or 0) / 365
        return implied_vol(row["option_close"], row["underlying_close"], row["strike"], T,
                            risk_free_rate, row["contract_type"])
    core["iv"] = core.apply(compute_iv, axis=1)

    def compute_greeks(row):
        T = (row["days_remaining"] or 0) / 365
        g = bs_greeks(row["underlying_close"], row["strike"], T, risk_free_rate, row["iv"], row["contract_type"])
        return pd.Series(g)
    greeks_df = core.apply(compute_greeks, axis=1)
    core = pd.concat([core, greeks_df], axis=1)

DISPLAY_NAMES = {
    "symbol": "نماد", "underlying": "دارایی پایه", "contract_type": "نوع قرارداد",
    "underlying_close": "ق. پایانی دارایی پایه", "option_close": "ق. پایانی اختیار",
    "strike": "قیمت اعمال", "maturity_date": "سررسید", "days_remaining": "روز مانده",
    "volume": "حجم", "value": "ارزش معاملات", "oi": "OI",
    "intrinsic_value": "ارزش ذاتی", "time_value": "ارزش زمانی",
    "moneyness": "Moneyness", "yield_to_maturity": "بازده تا سررسید (%)",
    "iv": "نوسان ضمنی (IV)", "delta": "دلتا", "gamma": "گاما",
    "theta": "تتا (روزانه)", "vega": "وگا (به‌ازای ۱٪)", "rho": "رو",
}

if core.empty or "symbol" not in core.columns:
    st.warning("ستون «نماد» شناسایی نشد. از سایدبار، نگاشت ستون‌ها را دستی تنظیم کن.")
    st.stop()

# ============================================================================
# 7) تب‌ها
# ============================================================================
tab_watch, tab_chain, tab_greeks, tab_parity, tab_ratio = st.tabs(
    ["📋 Watchlist", "🔗 زنجیره اختیار", "📐 نوسان ضمنی و Greeks", "⚖️ پوت-کال پاریتی", "📊 نسبت Put/Call"]
)

# ---------------------------------------------------------------- Watchlist
with tab_watch:
    df = core.rename(columns={k: v for k, v in DISPLAY_NAMES.items() if k in core.columns})

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        search_text = st.text_input("🔍 جستجوی نماد", "", key="wl_search")
    with f2:
        if "دارایی پایه" in df.columns:
            assets = ["همه"] + sorted(df["دارایی پایه"].dropna().unique().tolist())
            selected_asset = st.selectbox("دارایی پایه", assets, key="wl_asset")
        else:
            selected_asset = "همه"
    with f3:
        if "نوع قرارداد" in df.columns:
            types = ["همه"] + sorted(df["نوع قرارداد"].dropna().unique().tolist())
            selected_type = st.selectbox("Call / Put", types, key="wl_type")
        else:
            selected_type = "همه"
    with f4:
        if "سررسید" in df.columns:
            maturities = ["همه"] + sorted(df["سررسید"].dropna().astype(str).unique().tolist())
            selected_maturity = st.selectbox("سررسید", maturities, key="wl_maturity")
        else:
            selected_maturity = "همه"

    filtered = df.copy()
    if search_text:
        filtered = filtered[filtered["نماد"].astype(str).str.contains(search_text, case=False, na=False)]
    if "دارایی پایه" in filtered.columns and selected_asset != "همه":
        filtered = filtered[filtered["دارایی پایه"] == selected_asset]
    if "نوع قرارداد" in filtered.columns and selected_type != "همه":
        filtered = filtered[filtered["نوع قرارداد"] == selected_type]
    if "سررسید" in filtered.columns and selected_maturity != "همه":
        filtered = filtered[filtered["سررسید"].astype(str) == selected_maturity]

    st.caption(f"تعداد نتایج: {len(filtered)}")

    sc1, sc2 = st.columns([3, 1])
    with sc1:
        sort_by = st.selectbox("مرتب‌سازی بر اساس ستون", ["بدون مرتب‌سازی"] + list(filtered.columns), key="wl_sort")
    with sc2:
        sort_dir = st.radio("جهت", ["صعودی", "نزولی"], horizontal=True, key="wl_dir")
    if sort_by != "بدون مرتب‌سازی":
        filtered = filtered.sort_values(by=sort_by, ascending=(sort_dir == "صعودی"))

    st.dataframe(filtered, use_container_width=True, height=440)

    def to_excel(dataframe):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            dataframe.to_excel(writer, index=False, sheet_name="Watchlist")
        return buffer.getvalue()

    st.download_button(
        "⬇️ خروجی Excel", data=to_excel(filtered),
        file_name=f"option_watchlist_{dt.date.today().isoformat()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        with st.expander("📈 نمودار"):
            y_col = st.selectbox("ستون برای نمایش نموداری", numeric_cols, key="wl_chart_col")
            fig = px.bar(filtered.head(40), x="نماد", y=y_col, title=f"{y_col} بر اساس نماد")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------- Option Chain
with tab_chain:
    st.caption("نمایش استاندارد زنجیره‌ی اختیار: Callها و Putها کنار هم بر اساس قیمت اعمال.")
    needed = {"underlying", "maturity_date", "strike", "contract_type"}
    if not needed <= set(core.columns):
        st.warning("برای این نما به ستون‌های «دارایی پایه»، «سررسید»، «قیمت اعمال» و «نوع قرارداد» نیاز است.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            underlyings = sorted(core["underlying"].dropna().unique().tolist())
            sel_underlying = st.selectbox("دارایی پایه", underlyings, key="chain_underlying")
        subset = core[core["underlying"] == sel_underlying]
        with c2:
            mats = sorted(subset["maturity_date"].dropna().astype(str).unique().tolist())
            sel_maturity = st.selectbox("سررسید", mats, key="chain_maturity") if mats else None

        if sel_maturity:
            subset = subset[subset["maturity_date"].astype(str) == sel_maturity]

        calls = subset[subset["contract_type"] == "Call"].set_index("strike")
        puts = subset[subset["contract_type"] == "Put"].set_index("strike")

        show_cols = [c for c in ["symbol", "volume", "oi", "option_close", "iv"] if c in subset.columns]
        calls_view = calls[show_cols].add_prefix("Call ") if show_cols else pd.DataFrame()
        puts_view = puts[show_cols].add_prefix("Put ") if show_cols else pd.DataFrame()

        chain = pd.concat([calls_view, puts_view], axis=1).sort_index()
        chain.index.name = "قیمت اعمال"
        st.dataframe(chain, use_container_width=True, height=440)

# ---------------------------------------------------------- Greeks & IV
with tab_greeks:
    if not has_bs_inputs:
        st.warning("داده‌ی کافی برای محاسبه‌ی نوسان ضمنی و Greeks موجود نیست (نیاز به قیمت پایه، اعمال، اختیار و روز مانده).")
    else:
        st.caption(
            f"محاسبه‌شده با مدل بلک-شولز و نرخ بدون ریسک {risk_free_rate*100:.0f}٪ "
            "(از سایدبار قابل تغییر است). این محاسبات تقریبی هستند."
        )
        g_cols = ["symbol", "underlying", "contract_type", "strike", "days_remaining",
                  "option_close", "iv", "delta", "gamma", "theta", "vega", "rho"]
        g_cols = [c for c in g_cols if c in core.columns]
        gdf = core[g_cols].rename(columns={k: v for k, v in DISPLAY_NAMES.items() if k in g_cols})

        gf1, gf2 = st.columns(2)
        with gf1:
            if "دارایی پایه" in gdf.columns:
                assets_g = ["همه"] + sorted(gdf["دارایی پایه"].dropna().unique().tolist())
                sel_asset_g = st.selectbox("دارایی پایه", assets_g, key="greeks_asset")
            else:
                sel_asset_g = "همه"
        with gf2:
            if "نوع قرارداد" in gdf.columns:
                types_g = ["همه"] + sorted(gdf["نوع قرارداد"].dropna().unique().tolist())
                sel_type_g = st.selectbox("Call / Put", types_g, key="greeks_type")
            else:
                sel_type_g = "همه"

        if sel_asset_g != "همه":
            gdf = gdf[gdf["دارایی پایه"] == sel_asset_g]
        if sel_type_g != "همه":
            gdf = gdf[gdf["نوع قرارداد"] == sel_type_g]

        fmt = {c: "{:.4f}" for c in ["نوسان ضمنی (IV)", "دلتا", "گاما", "تتا (روزانه)", "وگا (به‌ازای ۱٪)", "رو"] if c in gdf.columns}
        st.dataframe(gdf.style.format(fmt, na_rep="—"), use_container_width=True, height=440)

# ----------------------------------------------------------- Parity/Arbitrage
with tab_parity:
    st.caption(
        "پوت-کال پاریتی: C − P باید برابر با S − K·e^(−rT) باشد. اختلاف زیاد می‌تواند نشانه‌ی فرصت آربیتراژ "
        "یا خطای داده باشد — قبل از تصمیم واقعی، هزینه‌ی معاملات و نقدشوندگی را هم در نظر بگیر."
    )
    needed = {"underlying", "strike", "maturity_date", "contract_type", "option_close",
              "underlying_close", "days_remaining"}
    if not needed <= set(core.columns):
        st.warning("برای این تحلیل به ستون‌های دارایی پایه، قیمت اعمال، سررسید، نوع قرارداد، قیمت‌ها و روز مانده نیاز است.")
    else:
        calls = core[core["contract_type"] == "Call"]
        puts = core[core["contract_type"] == "Put"]
        merged = calls.merge(
            puts, on=["underlying", "strike", "maturity_date"], suffixes=("_call", "_put")
        )
        if merged.empty:
            st.info("جفت Call/Put هم‌سررسید و هم‌اعمال پیدا نشد.")
        else:
            def parity_calc(row):
                T = (row["days_remaining_call"] or row["days_remaining_put"] or 0) / 365
                if T <= 0:
                    return None, None
                S = row["underlying_close_call"]
                K = row["strike"]
                lhs = row["option_close_call"] - row["option_close_put"]
                rhs = S - K * math.exp(-risk_free_rate * T)
                return lhs - rhs, rhs
            parity_results = merged.apply(parity_calc, axis=1, result_type="expand")
            merged["اختلاف پاریتی"] = parity_results[0]
            merged["مقدار نظری (S - K·e^-rT)"] = parity_results[1]
            merged["اختلاف (%دارایی پایه)"] = (
                merged["اختلاف پاریتی"] / merged["underlying_close_call"] * 100
            )

            out = merged[[
                "underlying", "strike", "maturity_date",
                "symbol_call", "symbol_put", "option_close_call", "option_close_put",
                "underlying_close_call", "اختلاف پاریتی", "اختلاف (%دارایی پایه)",
            ]].rename(columns={
                "underlying": "دارایی پایه", "strike": "قیمت اعمال", "maturity_date": "سررسید",
                "symbol_call": "نماد Call", "symbol_put": "نماد Put",
                "option_close_call": "ق. Call", "option_close_put": "ق. Put",
                "underlying_close_call": "ق. دارایی پایه",
            })

            threshold = st.slider("حداقل اختلاف برای نمایش (٪ از قیمت دارایی پایه)", 0.0, 20.0, 1.0, 0.5)
            flagged = out[out["اختلاف (%دارایی پایه)"].abs() >= threshold]
            flagged = flagged.sort_values("اختلاف (%دارایی پایه)", key=abs, ascending=False)

            st.caption(f"تعداد جفت‌های با اختلاف بیش از {threshold}٪: {len(flagged)} از {len(out)}")
            st.dataframe(flagged, use_container_width=True, height=440)

# --------------------------------------------------------------- Put/Call Ratio
with tab_ratio:
    st.caption("نسبت حجم و ارزش معاملات Put به Call — شاخصی از احساسات بازار (بدبینی/خوش‌بینی).")
    if not {"underlying", "contract_type"} <= set(core.columns):
        st.warning("برای این تحلیل به ستون‌های دارایی پایه و نوع قرارداد نیاز است.")
    else:
        agg_cols = [c for c in ["volume", "value"] if c in core.columns]
        if not agg_cols:
            st.warning("ستون حجم یا ارزش معاملات شناسایی نشد.")
        else:
            grouped = core.groupby(["underlying", "contract_type"])[agg_cols].sum().reset_index()
            pivoted = grouped.pivot(index="underlying", columns="contract_type", values=agg_cols)
            pivoted.columns = [f"{a} {b}" for a, b in pivoted.columns]
            pivoted = pivoted.fillna(0)

            if "volume Call" in pivoted.columns and "volume Put" in pivoted.columns:
                pivoted["نسبت حجمی Put/Call"] = pivoted["volume Put"] / pivoted["volume Call"].replace(0, pd.NA)
            if "value Call" in pivoted.columns and "value Put" in pivoted.columns:
                pivoted["نسبت ارزشی Put/Call"] = pivoted["value Put"] / pivoted["value Call"].replace(0, pd.NA)

            pivoted = pivoted.reset_index().rename(columns={"underlying": "دارایی پایه"})

            total_call_vol = grouped[grouped["contract_type"] == "Call"]["volume"].sum() if "volume" in agg_cols else None
            total_put_vol = grouped[grouped["contract_type"] == "Put"]["volume"].sum() if "volume" in agg_cols else None

            if total_call_vol and total_put_vol is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("مجموع حجم Call", f"{total_call_vol:,.0f}")
                m2.metric("مجموع حجم Put", f"{total_put_vol:,.0f}")
                m3.metric("نسبت کل بازار (Put/Call)", f"{(total_put_vol / total_call_vol):.2f}" if total_call_vol else "—")

            st.dataframe(pivoted, use_container_width=True, height=380)

            if "نسبت حجمی Put/Call" in pivoted.columns:
                fig = px.bar(
                    pivoted.sort_values("نسبت حجمی Put/Call", ascending=False).head(20),
                    x="دارایی پایه", y="نسبت حجمی Put/Call",
                    title="نسبت حجمی Put/Call به تفکیک دارایی پایه (۲۰ مورد برتر)",
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

st.caption(
    "⚠️ این داشبورد صرفاً ابزار نمایشی/تحلیلی است و مبنای تصمیم سرمایه‌گذاری نیست. "
    "نوسان ضمنی، Greeks و پوت-کال پاریتی بر پایه‌ی مدل بلک-شولز و نرخ بدون ریسک فرضی محاسبه شده‌اند."
)
