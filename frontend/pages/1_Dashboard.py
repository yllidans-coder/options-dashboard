import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import warnings
import requests
import os
from scipy.stats import norm
from datetime import datetime, date, timedelta

warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard | Dr. Marcus",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2d3250; border-radius: 12px;
        padding: 16px; text-align: center; margin: 4px;
    }
    .metric-label { color: #8b92a5; font-size: 11px; font-weight: 600;
                    letter-spacing: 1px; text-transform: uppercase; }
    .metric-value { color: #e0e6ff; font-size: 24px; font-weight: 700; }
    .section-title {
        color: #7c83ff; font-size: 17px; font-weight: 700;
        border-left: 4px solid #7c83ff;
        padding-left: 10px; margin: 24px 0 12px 0;
    }
    .div-card {
        background: linear-gradient(135deg, #1a2535, #1e2d40);
        border: 1px solid #2563eb44; border-radius: 12px;
        padding: 16px; margin: 6px 0;
    }
    .alert-card {
        background: linear-gradient(135deg, #2d1f00, #3d2800);
        border: 1px solid #f59e0b; border-radius: 10px;
        padding: 12px 16px; margin: 6px 0;
    }
    .regime-high {
        background: linear-gradient(135deg, #2d0f1a, #3d1525);
        border: 1px solid #ff4b6e; border-radius: 10px; padding: 12px;
    }
    .regime-low {
        background: linear-gradient(135deg, #0f2d1a, #153d25);
        border: 1px solid #00d4aa; border-radius: 10px; padding: 12px;
    }
    .regime-mid {
        background: linear-gradient(135deg, #2d2500, #3d3200);
        border: 1px solid #f59e0b; border-radius: 10px; padding: 12px;
    }
    .stTabs [data-baseweb="tab"] { color: #8b92a5; }
    .stTabs [aria-selected="true"] { color: #7c83ff !important; }
    .fonte-badge {
        font-size: 10px; padding: 2px 7px; border-radius: 5px;
        font-weight: 700; letter-spacing: 0.5px;
    }
    .fonte-oplab  { background:#0d3326; color:#00d4aa; border:1px solid #00d4aa55; }
    .fonte-bcb    { background:#1a2d0d; color:#7ec850; border:1px solid #7ec85055; }
    .fonte-calc   { background:#2d2500; color:#f59e0b; border:1px solid #f59e0b55; }
    .fonte-yf     { background:#1a1f35; color:#7c83ff; border:1px solid #7c83ff55; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR — NAVEGAÇÃO
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 Painel de Opções")
    st.markdown("---")
    st.markdown("**Navegação**")
    st.page_link("Home.py",                          label="🏠 Home")
    st.page_link("pages/1_Dashboard.py",             label="📊 Dashboard")
    st.page_link("pages/4_Calculadora_de_Opcoes.py", label="🧮 Calculadora")
    st.markdown("---")


# ══════════════════════════════════════════════════════════════════
#  TOKEN OPLAB
# ══════════════════════════════════════════════════════════════════

OPLAB_TOKEN  = st.secrets.get("OPLAB_TOKEN", os.getenv("OPLAB_TOKEN", ""))
OPLAB_BASE   = "https://api.oplab.com.br/v3"
OPLAB_HEADER = {"Access-Token": OPLAB_TOKEN}


# ══════════════════════════════════════════════════════════════════
#  BSM ENGINE
# ══════════════════════════════════════════════════════════════════

def bs_price(S, K, T, r, sigma, option_type='call', q=0.0):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, option_type='call', q=0.0):
    if T <= 0 or sigma <= 0 or S <= 0:
        return {}
    d1   = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2   = d1 - sigma * np.sqrt(T)
    pdf1 = norm.pdf(d1)
    sqT  = np.sqrt(T)

    gamma = pdf1 * np.exp(-q*T) / (S * sigma * sqT)
    vega  = S * np.exp(-q*T) * pdf1 * sqT / 100

    if option_type == 'call':
        delta = np.exp(-q*T) * norm.cdf(d1)
        theta = (-(S * np.exp(-q*T) * pdf1 * sigma / (2*sqT))
                 - r * K * np.exp(-r*T) * norm.cdf(d2)
                 + q * S * np.exp(-q*T) * norm.cdf(d1)) / 365
        rho   =  K * T * np.exp(-r*T) * norm.cdf(d2)  / 100
    else:
        delta = np.exp(-q*T) * (norm.cdf(d1) - 1)
        theta = (-(S * np.exp(-q*T) * pdf1 * sigma / (2*sqT))
                 + r * K * np.exp(-r*T) * norm.cdf(-d2)
                 - q * S * np.exp(-q*T) * norm.cdf(-d1)) / 365
        rho   = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100

    vanna = -np.exp(-q*T) * pdf1 * d2 / sigma
    vomma = vega * d1 * d2 / sigma

    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega,
                rho=rho, vanna=vanna, vomma=vomma)


# ══════════════════════════════════════════════════════════════════
#  SELIC — BCB
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def get_selic_bcb() -> float:
    try:
        url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados/ultimos/1?formato=json"
        r   = requests.get(url, timeout=5)
        val = float(r.json()[0]["valor"].replace(",", "."))
        return val / 100
    except:
        return 0.1075


# ══════════════════════════════════════════════════════════════════
#  OPLAB — COTAÇÃO
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def get_oplab_quote(ticker: str) -> dict | None:
    if not OPLAB_TOKEN:
        return None
    try:
        symbol = ticker.replace(".SA", "").upper()
        r = requests.get(
            f"{OPLAB_BASE}/market/stocks/{symbol}",
            headers=OPLAB_HEADER,
            timeout=8
        )
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None


# ══════════════════════════════════════════════════════════════════
#  HISTÓRICO — yfinance
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def get_historical_yf(ticker: str):
    try:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="2y", auto_adjust=True)
        if hist.empty:
            return None, None
        close  = hist["Close"].squeeze().dropna()
        volume = hist["Volume"].squeeze().dropna()
        return close, volume
    except:
        return None, None


# ══════════════════════════════════════════════════════════════════
#  DIVIDENDOS — yfinance
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def get_dividends_yf(ticker: str) -> pd.DataFrame:
    try:
        tk   = yf.Ticker(ticker)
        divs = tk.dividends
        if divs.empty:
            return pd.DataFrame()
        dh = divs.reset_index()
        dh.columns = ["Data", "Dividendo"]
        dh["Data"] = pd.to_datetime(dh["Data"]).dt.tz_localize(None)
        return dh.sort_values("Data", ascending=False).reset_index(drop=True)
    except:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
#  VENCIMENTOS B3
# ══════════════════════════════════════════════════════════════════

def get_vencimentos_b3() -> list[date]:
    hoje    = date.today()
    venc    = []
    ano_ref = hoje.year
    for delta_mes in range(0, 13):
        mes          = (hoje.month - 1 + delta_mes) % 12 + 1
        ano          = ano_ref + (hoje.month - 1 + delta_mes) // 12
        primeiro_dia = date(ano, mes, 1)
        seg          = primeiro_dia + timedelta(days=(0 - primeiro_dia.weekday()) % 7)
        terceira_seg = seg + timedelta(weeks=2)
        if terceira_seg >= hoje:
            venc.append(terceira_seg)
    return sorted(venc)[:6]


def get_proximo_vencimento_dias() -> tuple[date, int]:
    hoje = date.today()
    for v in get_vencimentos_b3():
        if v > hoje:
            return v, (v - hoje).days
    return hoje + timedelta(days=30), 30


# ══════════════════════════════════════════════════════════════════
#  FETCH PRINCIPAL
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def fetch_asset_data(ticker: str):
    try:
        quote = get_oplab_quote(ticker)

        if not quote:
            tk    = yf.Ticker(ticker)
            info  = tk.info
            quote = {
                "close":            info.get("regularMarketPrice", 0),
                "variation":        info.get("regularMarketChangePercent", 0),
                "volume":           info.get("regularMarketVolume", 0),
                "financial_volume": 0,
                "name":             info.get("shortName", ticker),
                "beta_ibov":        info.get("beta", None),
                "iv_current":       None,
                "iv_1y_rank":       None,
                "iv_1y_percentile": None,
            }

        price   = float(quote.get("close") or quote.get("last_price") or 0)
        chg_pct = float(quote.get("variation") or 0)
        vol_day = float(quote.get("financial_volume") or quote.get("volume") or 0)
        name    = (quote.get("name") or ticker)
        beta    = float(quote.get("beta_ibov") or 1.0) or None
        mkt_cap = quote.get("market_cap", None)

        iv_current_raw = quote.get("iv_current")
        if iv_current_raw and float(iv_current_raw) > 0:
            iv_real  = float(iv_current_raw) / 100
            iv_fonte = "OpLab"
        else:
            iv_real  = None
            iv_fonte = "HV×1.25 (proxy)"

        iv_rank_1y = quote.get("iv_1y_rank")
        iv_pct_1y  = quote.get("iv_1y_percentile")
        iv_rank_6m = quote.get("iv_6m_rank")
        iv_pct_6m  = quote.get("iv_6m_percentile")
        ewma       = quote.get("ewma_current")
        garch      = quote.get("garch11_1y")

        if iv_rank_1y is not None:
            iv_rank = float(iv_rank_1y)
            iv_pct  = float(iv_pct_1y) if iv_pct_1y else iv_rank
        elif iv_rank_6m is not None:
            iv_rank = float(iv_rank_6m)
            iv_pct  = float(iv_pct_6m) if iv_pct_6m else iv_rank
        else:
            iv_rank = 50.0
            iv_pct  = 50.0

        close, volume = get_historical_yf(ticker)

        if close is None or len(close) < 30:
            st.error(f"Histórico indisponível para {ticker}")
            return None

        log_r = np.log(close / close.shift(1)).dropna()

        def hv(n):
            s = log_r.rolling(n, min_periods=max(5, n // 2)).std()
            return float(s.iloc[-1]) * np.sqrt(252) * 100 if len(s) >= 5 else None

        hv10  = hv(10)
        hv21  = hv(21)
        hv63  = hv(63)
        hv252 = hv(252)

        if iv_real is None:
            iv_real  = (hv21 * 1.25 / 100) if hv21 else 0.35
            iv_fonte = "HV×1.25 (proxy)"

        if iv_rank == 50.0:
            hv21_s  = log_r.rolling(21, min_periods=10).std().dropna() * np.sqrt(252) * 100
            hv_min  = float(hv21_s.min())
            hv_max  = float(hv21_s.max())
            hv_now  = float(hv21_s.iloc[-1])
            iv_rank = (hv_now - hv_min) / (hv_max - hv_min) * 100 if hv_max != hv_min else 50.0
            iv_pct  = iv_rank

        vol_med = (float(volume.rolling(20).mean().iloc[-1])
                   if len(volume) >= 20 else float(volume.mean()))

        div_history = get_dividends_yf(ticker)
        div_yield   = 0.0
        annual_div  = 0.0

        if not div_history.empty:
            now_ts     = pd.Timestamp.now()
            mask_1y    = div_history["Data"] >= now_ts - pd.DateOffset(years=1)
            annual_div = float(div_history[mask_1y]["Dividendo"].sum())
            div_yield  = annual_div / price * 100 if price > 0 else 0.0

        prox_venc, dias_venc = get_proximo_vencimento_dias()
        selic                = get_selic_bcb()

        hv21_series = (log_r.rolling(21, min_periods=10).std().dropna()
                       * np.sqrt(252) * 100)

        return {
            "price":        price,
            "change_pct":   chg_pct,
            "hv10":         hv10,
            "hv21":         hv21,
            "hv63":         hv63,
            "hv252":        hv252,
            "iv_real":      iv_real,
            "iv_fonte":     iv_fonte,
            "iv_rank":      iv_rank,
            "iv_pct":       iv_pct,
            "iv_rank_hv":   iv_rank,
            "iv_rank_1y":   float(iv_rank_1y) if iv_rank_1y else None,
            "iv_pct_1y":    float(iv_pct_1y)  if iv_pct_1y  else None,
            "iv_rank_6m":   float(iv_rank_6m) if iv_rank_6m else None,
            "iv_pct_6m":    float(iv_pct_6m)  if iv_pct_6m  else None,
            "iv_1y_max":    float(quote.get("iv_1y_max") or 0),
            "iv_1y_min":    float(quote.get("iv_1y_min") or 0),
            "ewma":         float(ewma)  if ewma  else None,
            "garch":        float(garch) if garch else None,
            "short_trend":  quote.get("short_term_trend"),
            "mid_trend":    quote.get("middle_term_trend"),
            "entropy":      quote.get("entropy"),
            "oplab_score":  quote.get("oplab_score"),
            "oplab_instr":  quote,
            "selic":        selic,
            "prox_venc":    prox_venc,
            "dias_venc":    dias_venc,
            "div_yield":    div_yield,
            "annual_div":   annual_div,
            "div_history":  div_history,
            "earnings":     None,
            "name":         name,
            "sector":       quote.get("category", "—"),
            "industry":     quote.get("segment",  "—"),
            "beta":         beta,
            "mkt_cap":      mkt_cap,
            "close":        close,
            "log_ret":      log_r,
            "volume":       volume,
            "vol_med":      vol_med,
            "hv21_series":  hv21_series,
            "vol_day":      vol_day,
        }

    except Exception as e:
        st.error(f"Erro ao carregar {ticker}: {e}")
        return None


@st.cache_data(ttl=60)
def fetch_vix():
    try:
        v = yf.download("^VIX", period="5d", progress=False)
        return float(v["Close"].iloc[-1])
    except:
        return None


@st.cache_data(ttl=60)
def fetch_ibov():
    try:
        q = get_oplab_quote("BVSP")
        if q:
            return float(q.get("close", 0)), float(q.get("variation", 0))
        ib = yf.download("^BVSP", period="5d", progress=False)
        p  = float(ib["Close"].iloc[-1])
        pp = float(ib["Close"].iloc[-2])
        return p, (p - pp) / pp * 100
    except:
        return None, None


def fmt_mktcap(v):
    if v is None: return "—"
    if v >= 1e12: return f"R$ {v/1e12:.2f}T"
    if v >= 1e9:  return f"R$ {v/1e9:.2f}B"
    if v >= 1e6:  return f"R$ {v/1e6:.2f}M"
    return f"R$ {v:,.0f}"


def regime_label(ivr):
    if ivr >= 70: return "ALTA",  "#ff4b6e", "regime-high", "📈 IV Elevada — Vender Volatilidade"
    if ivr <= 30: return "BAIXA", "#00d4aa", "regime-low",  "📉 IV Baixa — Comprar Volatilidade"
    return "MÉDIA", "#f59e0b", "regime-mid", "↔️ IV Neutra — Estratégias Neutras"


def badge_fonte(fonte: str) -> str:
    cls = {"OpLab": "fonte-oplab", "BCB": "fonte-bcb",
           "Calculado": "fonte-calc"}.get(fonte, "fonte-yf")
    return f"<span class='fonte-badge {cls}'>⬤ {fonte}</span>"


PLOTLY_BASE = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(14,17,23,0.8)',
    margin=dict(l=20, r=20, t=20, b=40),
    legend=dict(bgcolor='rgba(30,33,46,0.8)')
)


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR — CONTROLES
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🏦 Dashboard Institucional")

    if OPLAB_TOKEN:
        st.markdown("<span class='fonte-badge fonte-oplab'>✅ OpLab PRO Conectado</span>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<span class='fonte-badge fonte-calc'>⚠️ OpLab: Token não configurado</span>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Ativo Principal")
    main_ticker = st.text_input(
        "Ticker",
        value=st.session_state.get("bs_asset", "PETR4.SA"),
        placeholder="PETR4.SA, VALE3.SA, AAPL"
    )

    st.markdown("### 📡 Watchlist")
    watchlist_raw = st.text_area(
        "Tickers (um por linha)",
        value="VALE3.SA\nITUB4.SA\nBBDC4.SA\nBOVA11.SA\nWEGE3.SA",
        height=130
    )
    watchlist = [t.strip().upper() for t in watchlist_raw.split("\n") if t.strip()]

    st.markdown("### ⚙️ Parâmetros — Auto + Manual")

    selic_auto = get_selic_bcb()
    st.markdown(f"**Taxa Selic** {badge_fonte('BCB')}", unsafe_allow_html=True)
    dash_r = st.slider("Taxa (%)", 0.0, 25.0,
                        round(selic_auto * 100, 2), 0.25) / 100

    prox_v, dias_v = get_proximo_vencimento_dias()
    st.markdown(f"**Vencimento B3** {badge_fonte('Calculado')}", unsafe_allow_html=True)
    st.caption(f"📅 Próximo: {prox_v.strftime('%d/%m/%Y')} ({dias_v}d)")
    dash_T_dias = st.slider("Dias até Venc.", 1, 365, dias_v)
    dash_T      = dash_T_dias / 365.0

    st.markdown("**IV** — preenchida automaticamente via OpLab", unsafe_allow_html=True)
    iv_default = float(st.session_state.get("bs_iv", 35.0))
    if iv_default <= 5:
        iv_default = iv_default * 100
    dash_iv = st.slider("IV (%)", 1.0, 150.0,
                        float(np.clip(round(iv_default, 1), 1.0, 150.0)),
                        0.5) / 100

    st.button("🔄 Atualizar Tudo", use_container_width=True,
              on_click=lambda: st.cache_data.clear())

    st.markdown("---")
    vix_val          = fetch_vix()
    ibov_p, ibov_chg = fetch_ibov()

    if vix_val:
        vix_color = "#ff4b6e" if vix_val > 25 else "#00d4aa" if vix_val < 15 else "#f59e0b"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>VIX</div>
            <div style='color:{vix_color};font-size:26px;font-weight:700'>{vix_val:.2f}</div>
        </div>""", unsafe_allow_html=True)

    if ibov_p:
        ibov_color = "#00d4aa" if ibov_chg >= 0 else "#ff4b6e"
        st.markdown(f"""
        <div class='metric-card' style='margin-top:8px'>
            <div class='metric-label'>IBOVESPA</div>
            <div style='color:{ibov_color};font-size:20px;font-weight:700'>
                {ibov_p:,.0f} <span style='font-size:14px'>({ibov_chg:+.2f}%)</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='margin-top:12px;padding:8px;border-radius:8px;
                background:#1a2535;border:1px solid #2d3250'>
        <div style='color:#8b92a5;font-size:10px'>FONTES DE DADOS</div>
        <div style='margin-top:4px'>
            {badge_fonte('OpLab')} Preço · IV · IV Rank · Beta<br>
            {badge_fonte('BCB')} Selic<br>
            {badge_fonte('Calculado')} HV · Vol Cone<br>
            {badge_fonte('yfinance')} Histórico · Dividendos
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='color:#7c83ff;margin-bottom:0'>🏦 Painel Institucional</h1>
<p style='color:#8b92a5;margin-top:4px'>
    Análise completa · OpLab PRO · API BCB · B3 Real · Vol Surface
</p>
<hr style='border-color:#2d3250'>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  CARREGA ATIVO PRINCIPAL
# ══════════════════════════════════════════════════════════════════

with st.spinner(f"Carregando {main_ticker}..."):
    mkt = fetch_asset_data(main_ticker)

if not mkt:
    st.error(f"❌ Não foi possível carregar dados de **{main_ticker}**.")
    st.stop()

iv_auto         = mkt["iv_real"]
dash_iv_efetivo = dash_iv if abs(dash_iv - iv_auto) > 0.01 else iv_auto
iv_rank         = mkt["iv_rank"]
reg_label, reg_color, reg_class, reg_rec = regime_label(iv_rank)


# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════

st.session_state.update({
    "bs_asset":      main_ticker,
    "bs_name":       mkt["name"],
    "bs_sector":     mkt["sector"],
    "bs_industry":   mkt["industry"],
    "bs_S":          round(mkt["price"], 2),
    "bs_iv":         round(iv_auto * 100, 2),
    "bs_iv_pct":     round(iv_auto * 100, 2),
    "bs_r":          round(mkt["selic"], 6),
    "bs_T_dias":     int(mkt["dias_venc"]),
    "bs_T":          int(mkt["dias_venc"]),
    "bs_K":          round(mkt["price"], 2),
    "bs_iv_rank":    round(iv_rank, 2),
    "bs_iv_pct_1y":  round(mkt["iv_pct"], 2),
    "bs_iv_rank_1y": round(mkt["iv_rank_1y"], 2) if mkt.get("iv_rank_1y") else round(iv_rank, 2),
    "bs_iv_52h":     round(mkt["iv_1y_max"], 4),
    "bs_iv_52l":     round(mkt["iv_1y_min"], 4),
    "bs_hv21":       round(mkt["hv21"], 4) if mkt.get("hv21") else None,
    "bs_hv63":       round(mkt["hv63"], 4) if mkt.get("hv63") else None,
    "bs_q":          round(mkt["div_yield"] / 100, 6),
    "bs_div_yield":  round(mkt["div_yield"], 4),
    "bs_beta_ibov":  mkt["beta"],
    "bs_earnings":   mkt["earnings"],
    "bs_ewma":       mkt.get("ewma"),
    "bs_garch":      mkt.get("garch"),
})


# ══════════════════════════════════════════════════════════════════
#  KPIs
# ══════════════════════════════════════════════════════════════════

chg_color = "#00d4aa" if mkt["change_pct"] >= 0 else "#ff4b6e"
chg_arrow = "▲" if mkt["change_pct"] >= 0 else "▼"

st.markdown(f"""
<h2 style='color:#e0e6ff;margin-bottom:4px'>
    {mkt['name']} &nbsp;
    <span style='color:#8b92a5;font-size:16px'>{main_ticker}</span>
    {badge_fonte(mkt['iv_fonte'])}
</h2>
<p style='color:#8b92a5;font-size:13px'>
    {mkt['sector']} | {mkt['industry']} &nbsp;|&nbsp;
    Selic: <b style='color:#7ec850'>{mkt['selic']*100:.2f}%</b> (BCB automático) &nbsp;|&nbsp;
    Próx. Venc: <b style='color:#7c83ff'>{mkt['prox_venc'].strftime('%d/%m/%Y')}</b>
    ({mkt['dias_venc']}d)
</p>
""", unsafe_allow_html=True)

k1,k2,k3,k4,k5,k6,k7,k8,k9,k10 = st.columns(10)

iv_hv_diff = (mkt['iv_real'] - (mkt['hv21'] or 0) / 100) * 100 if mkt['hv21'] else None

for col, label, val, color in [
    (k1,  "PREÇO",       f"R$ {mkt['price']:.2f}",                        "#e0e6ff"),
    (k2,  "VARIAÇÃO",    f"{chg_arrow} {mkt['change_pct']:+.2f}%",        chg_color),
    (k3,  "IV REAL",     f"{mkt['iv_real']*100:.1f}%",                    "#7c83ff"),
    (k4,  "HV 21d",      f"{mkt['hv21']:.1f}%" if mkt['hv21'] else "—",  "#f59e0b"),
    (k5,  "IV−HV",
          f"{iv_hv_diff:+.1f}%" if iv_hv_diff is not None else "—",
          "#ff4b6e" if iv_hv_diff and iv_hv_diff > 0 else "#00d4aa"),
    (k6,  "IV RANK 1Y",  f"{mkt['iv_rank']:.0f}",                         reg_color),
    (k7,  "IV PERC. 1Y", f"{mkt['iv_pct']:.0f}",                          reg_color),
    (k8,  "DIV YIELD",   f"{mkt['div_yield']:.2f}%",                      "#00d4aa"),
    (k9,  "BETA IBOV",
          f"{mkt['beta']:.2f}" if mkt['beta'] else "—",                    "#a78bfa"),
    (k10, "SELIC",       f"{mkt['selic']*100:.2f}%",                      "#7ec850"),
]:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div style='color:{color};font-size:16px;font-weight:700'>{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

trend_short = mkt.get("short_trend")
trend_mid   = mkt.get("mid_trend")
trend_txt   = ("📈 Tendência Curto Prazo: " +
               ("Alta" if trend_short == 1 else "Baixa" if trend_short == -1 else "Neutro") +
               " | Médio Prazo: " +
               ("Alta" if trend_mid == 1 else "Baixa" if trend_mid == -1 else "Neutro")
               if trend_short is not None else "")

st.markdown(f"""
<div class='{reg_class}' style='padding:14px 20px;margin:8px 0;border-radius:12px'>
    <span style='color:{reg_color};font-size:18px;font-weight:700'>
        REGIME: {reg_label}
    </span>
    &nbsp;|&nbsp;
    <span style='color:#e0e6ff;font-size:15px'>{reg_rec}</span>
    &nbsp;|&nbsp;
    <span style='color:#8b92a5;font-size:13px'>
        IV Rank 1Y = {mkt['iv_rank']:.0f} &nbsp;·&nbsp;
        IV Perc. 1Y = {mkt['iv_pct']:.0f} &nbsp;·&nbsp;
        IV atual = {mkt['iv_real']*100:.1f}% (OpLab) &nbsp;·&nbsp;
        IV máx 1Y = {mkt['iv_1y_max']:.1f}% &nbsp;·&nbsp;
        IV mín 1Y = {mkt['iv_1y_min']:.1f}%
    </span>
    {"<br><span style='color:#7c83ff;font-size:12px'>" + trend_txt + "</span>" if trend_txt else ""}
</div>
""", unsafe_allow_html=True)

if mkt.get("ewma") or mkt.get("garch"):
    e1, e2, e3 = st.columns(3)
    if mkt.get("ewma"):
        with e1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>VOL EWMA (OpLab)</div>
                <div style='color:#7c83ff;font-size:18px;font-weight:700'>
                    {mkt['ewma']:.2f}%</div>
            </div>""", unsafe_allow_html=True)
    if mkt.get("garch"):
        with e2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>VOL GARCH 1Y (OpLab)</div>
                <div style='color:#a78bfa;font-size:18px;font-weight:700'>
                    {mkt['garch']:.2f}%</div>
            </div>""", unsafe_allow_html=True)
    if mkt.get("entropy") is not None:
        with e3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>ENTROPIA (OpLab)</div>
                <div style='color:#f59e0b;font-size:18px;font-weight:700'>
                    {mkt['entropy']:.4f}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Preço e Volume",
    "💰 Dividendos",
    "🌡️ Volatilidade",
    "📡 Watchlist",
    "🎯 Cenários de Opções",
])


# ── TAB 1 — PREÇO & VOLUME ────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<div class='section-title'>📈 Histórico de Preços — 2 anos</div>",
                    unsafe_allow_html=True)

        close = mkt["close"]
        ma20  = close.rolling(20).mean()
        ma50  = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(
            x=close.index,
            open=close, high=close*1.002, low=close*0.998, close=close,
            name="Preço",
            increasing_line_color='#00d4aa',
            decreasing_line_color='#ff4b6e',
            showlegend=False
        ))
        for ma, name_ma, color in [
            (ma20,  'MA20',  '#f59e0b'),
            (ma50,  'MA50',  '#7c83ff'),
            (ma200, 'MA200', '#ff4b6e')
        ]:
            fig_price.add_trace(go.Scatter(
                x=ma.index, y=ma, mode='lines', name=name_ma,
                line=dict(color=color, width=1.5,
                          dash='solid' if name_ma != 'MA200' else 'dash')
            ))

        dh = mkt["div_history"]
        if not dh.empty:
            cl_df = pd.DataFrame({
                "Data":  close.index.tz_localize(None),
                "Close": close.values
            }).sort_values("Data")
            dh_m = pd.merge_asof(
                dh.sort_values("Data"), cl_df,
                on="Data", direction="nearest"
            )
            if not dh_m.empty:
                fig_price.add_trace(go.Scatter(
                    x=dh_m["Data"], y=dh_m["Close"],
                    mode='markers', name='Dividendo',
                    marker=dict(symbol='triangle-up', size=10,
                                color='#00d4aa',
                                line=dict(width=1, color='white'))
                ))

        fig_price.update_layout(**PLOTLY_BASE, height=420,
                                xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_price, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>📊 Estatísticas</div>",
                    unsafe_allow_html=True)

        p52h    = float(close.tail(252).max())
        p52l    = float(close.tail(252).min())
        pct_52h = ((mkt["price"] - p52l) / (p52h - p52l) * 100
                   if p52h != p52l else 50)

        for label, val, color in [
            ("Máx 52 sem",      f"R$ {p52h:.2f}",                        "#00d4aa"),
            ("Mín 52 sem",      f"R$ {p52l:.2f}",                        "#ff4b6e"),
            ("% no Range 52s",  f"{pct_52h:.1f}%",                       "#f59e0b"),
            ("IV Real (OpLab)", f"{mkt['iv_real']*100:.1f}%",             "#7c83ff"),
            ("IV Rank 1Y",      f"{mkt['iv_rank']:.0f}",                  reg_color),
            ("IV Perc. 1Y",     f"{mkt['iv_pct']:.0f}",                   reg_color),
            ("IV Máx 1Y",       f"{mkt['iv_1y_max']:.1f}%",               "#ff4b6e"),
            ("IV Mín 1Y",       f"{mkt['iv_1y_min']:.1f}%",               "#00d4aa"),
            ("EWMA Vol",
             f"{mkt['ewma']:.2f}%" if mkt.get('ewma') else "—", "#7c83ff"),
            ("GARCH 1Y",
             f"{mkt['garch']:.2f}%" if mkt.get('garch') else "—", "#a78bfa"),
            ("Selic (BCB)",     f"{mkt['selic']*100:.2f}%",               "#7ec850"),
            ("Prox. Venc B3",   mkt['prox_venc'].strftime('%d/%m/%Y'),    "#7c83ff"),
            ("Beta IBOV",
             f"{mkt['beta']:.2f}" if mkt['beta'] else "—", "#a78bfa"),
            ("Vol Fin. Dia",
             f"R$ {mkt['vol_day']/1e6:.1f}M"
             if mkt['vol_day'] > 1e6
             else f"R$ {mkt['vol_day']:,.0f}", "#7c83ff"),
            ("Div Yield",       f"{mkt['div_yield']:.2f}%",               "#00d4aa"),
        ]:
            st.markdown(f"""
            <div class='metric-card'
                 style='margin-bottom:5px;text-align:left;padding:8px 14px'>
                <span style='color:#8b92a5;font-size:11px'>{label}</span>
                <span style='color:{color};font-size:14px;font-weight:700;
                             float:right'>{val}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📊 Volume Histórico</div>",
                unsafe_allow_html=True)
    vol_s  = mkt["volume"].tail(120)
    vol_ma = vol_s.rolling(20).mean()
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=vol_s.index, y=vol_s,
        name='Volume', marker_color='rgba(124,131,255,0.5)'
    ))
    fig_vol.add_trace(go.Scatter(
        x=vol_ma.index, y=vol_ma, mode='lines',
        name='MA20', line=dict(color='#f59e0b', width=2)
    ))
    fig_vol.update_layout(**PLOTLY_BASE, height=220)
    st.plotly_chart(fig_vol, use_container_width=True)


# ── TAB 2 — DIVIDENDOS ───────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-title'>💰 Análise Completa de Dividendos</div>",
                unsafe_allow_html=True)
    dh = mkt["div_history"]

    if dh.empty:
        st.warning("⚠️ Nenhum dividendo encontrado para este ativo.")
    else:
        d1,d2,d3,d4,d5,d6 = st.columns(6)
        now      = pd.Timestamp.now()
        total_5y = float(dh[dh["Data"] >= now-pd.DateOffset(years=5)]["Dividendo"].sum())
        total_3y = float(dh[dh["Data"] >= now-pd.DateOffset(years=3)]["Dividendo"].sum())
        total_1y = float(dh[dh["Data"] >= now-pd.DateOffset(years=1)]["Dividendo"].sum())
        ultimo   = float(dh["Dividendo"].iloc[0])
        count_1y = len(dh[dh["Data"] >= now-pd.DateOffset(years=1)])
        dy_atual = total_1y / mkt["price"] * 100 if mkt["price"] > 0 else 0

        for col, label, val, color in [
            (d1, "Último Div.",  f"R$ {ultimo:.4f}",   "#00d4aa"),
            (d2, "Total 1 Ano",  f"R$ {total_1y:.4f}", "#7c83ff"),
            (d3, "Total 3 Anos", f"R$ {total_3y:.4f}", "#a78bfa"),
            (d4, "Total 5 Anos", f"R$ {total_5y:.4f}", "#f59e0b"),
            (d5, "Pagtos/Ano",   f"{count_1y}x",        "#00d4aa"),
            (d6, "Yield Atual",  f"{dy_atual:.2f}%",    "#00d4aa"),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{label}</div>
                    <div style='color:{color};font-size:20px;font-weight:700'>{val}</div>
                </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            dh_plot = dh.sort_values("Data").copy()
            fig_div = go.Figure()
            fig_div.add_trace(go.Bar(
                x=dh_plot["Data"], y=dh_plot["Dividendo"],
                name="Proventos",
                marker=dict(color=dh_plot["Dividendo"],
                            colorscale="Greens", showscale=False),
                text=[f"R$ {v:.4f}" for v in dh_plot["Dividendo"]],
                textposition="outside",
                textfont=dict(color='#e0e6ff', size=10)
            ))
            if len(dh_plot) >= 4:
                div_ma = dh_plot["Dividendo"].rolling(4, min_periods=1).mean()
                fig_div.add_trace(go.Scatter(
                    x=dh_plot["Data"], y=div_ma,
                    mode='lines', name='Média 4 pag.',
                    line=dict(color='#f59e0b', width=2, dash='dash')
                ))
            fig_div.update_layout(**PLOTLY_BASE, height=360)
            st.plotly_chart(fig_div, use_container_width=True)

        with col2:
            dh_display              = dh.copy()
            dh_display["Data"]      = dh_display["Data"].dt.strftime("%d/%m/%Y")
            dh_display["Dividendo"] = dh_display["Dividendo"].map(
                lambda x: f"R$ {x:.4f}"
            )
            dh_display["Yield%"] = [
                f"{(float(d.replace('R$ ','')) / mkt['price'] * 100):.3f}%"
                for d in dh_display["Dividendo"]
            ]
            st.dataframe(dh_display, use_container_width=True,
                         hide_index=True, height=400)


# ── TAB 3 — VOLATILIDADE ─────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-title'>🌡️ Painel de Volatilidade — OpLab + Calculado</div>",
                unsafe_allow_html=True)

    iv_ref = mkt["iv_real"] * 100

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col, label, val in [
        (c1, "IV Real (OpLab)",  iv_ref),
        (c2, "EWMA (OpLab)",     mkt.get("ewma")),
        (c3, "GARCH 1Y (OpLab)", mkt.get("garch")),
        (c4, "HV 10d",           mkt["hv10"]),
        (c5, "HV 21d",           mkt["hv21"]),
        (c6, "HV 63d",           mkt["hv63"]),
    ]:
        if val:
            diff  = iv_ref - float(val)
            color = ("#ff4b6e" if diff > 5
                     else "#00d4aa" if diff < -2
                     else "#f59e0b")
            badge = ("OpLab" if label in ("IV Real (OpLab)", "EWMA (OpLab)", "GARCH 1Y (OpLab)")
                     else f"IV−HV: {diff:+.1f}%")
        else:
            val, color, badge = 0.0, "#4b5563", "Sem dados"
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div style='color:{color};font-size:20px;font-weight:700'>
                    {float(val):.1f}%</div>
                <div style='color:{color};font-size:10px'>{badge}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    r1,r2,r3,r4,r5,r6 = st.columns(6)
    for col, label, val, color in [
        (r1, "IV Rank 1Y",  mkt.get("iv_rank_1y"), reg_color),
        (r2, "IV Perc. 1Y", mkt.get("iv_pct_1y"),  reg_color),
        (r3, "IV Rank 6M",  mkt.get("iv_rank_6m"), "#f59e0b"),
        (r4, "IV Perc. 6M", mkt.get("iv_pct_6m"),  "#f59e0b"),
        (r5, "IV Máx 1Y",   mkt.get("iv_1y_max"),  "#ff4b6e"),
        (r6, "IV Mín 1Y",   mkt.get("iv_1y_min"),  "#00d4aa"),
    ]:
        v_fmt = f"{float(val):.1f}" if val is not None else "—"
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div style='color:{color};font-size:20px;font-weight:700'>{v_fmt}</div>
                <div style='color:#8b92a5;font-size:10px'>OpLab</div>
            </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        log_r  = mkt["log_ret"]
        hv21_s = log_r.rolling(21, min_periods=10).std() * np.sqrt(252) * 100
        hv63_s = log_r.rolling(63, min_periods=30).std() * np.sqrt(252) * 100

        fig_vhist = go.Figure()
        fig_vhist.add_trace(go.Scatter(
            x=hv21_s.index, y=hv21_s, mode='lines',
            name='HV 21d', line=dict(color='#f59e0b', width=2)))
        fig_vhist.add_trace(go.Scatter(
            x=hv63_s.index, y=hv63_s, mode='lines',
            name='HV 63d', line=dict(color='#a78bfa', width=2)))
        fig_vhist.add_hline(
            y=iv_ref, line_dash="solid",
            line_color="#00d4aa", line_width=2,
            annotation_text=f"IV Real OpLab {iv_ref:.1f}%",
            annotation_font_color="#00d4aa")
        if mkt.get("ewma"):
            fig_vhist.add_hline(
                y=mkt["ewma"], line_dash="dot",
                line_color="#7c83ff", line_width=1.5,
                annotation_text=f"EWMA {mkt['ewma']:.1f}%",
                annotation_font_color="#7c83ff")
        if mkt.get("garch"):
            fig_vhist.add_hline(
                y=mkt["garch"], line_dash="dot",
                line_color="#a78bfa", line_width=1.5,
                annotation_text=f"GARCH {mkt['garch']:.1f}%",
                annotation_font_color="#a78bfa")
        fig_vhist.update_layout(**PLOTLY_BASE, height=380)
        st.plotly_chart(fig_vhist, use_container_width=True)

    with col2:
        fig_ivr = go.Figure(go.Indicator(
            mode="gauge+number",
            value=iv_rank,
            title={'text': "IV Rank 1Y\n(OpLab)",
                   'font': {'color': 'white', 'size': 13}},
            number={'font': {'color': reg_color, 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                'bar':  {'color': reg_color},
                'steps': [
                    {'range': [0,  30],  'color': 'rgba(0,212,170,0.15)'},
                    {'range': [30, 70],  'color': 'rgba(245,158,11,0.15)'},
                    {'range': [70, 100], 'color': 'rgba(255,75,110,0.15)'},
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 3},
                    'thickness': 0.8, 'value': iv_rank
                }
            }
        ))
        fig_ivr.update_layout(
            template='plotly_dark', height=280,
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_ivr, use_container_width=True)

        st.markdown(f"""
        <div class='{reg_class}' style='padding:12px;border-radius:10px;margin-top:8px'>
            <div style='color:{reg_color};font-size:16px;font-weight:700'>{reg_label}</div>
            <div style='color:#e0e6ff;font-size:13px;margin-top:4px'>{reg_rec}</div>
            <div style='color:#8b92a5;font-size:11px;margin-top:6px'>
                IV: {mkt['iv_real']*100:.1f}% &nbsp;·&nbsp;
                Máx: {mkt['iv_1y_max']:.1f}% &nbsp;·&nbsp;
                Mín: {mkt['iv_1y_min']:.1f}%
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📐 Vol Cone Real</div>",
                unsafe_allow_html=True)
    periodos  = [10, 21, 30, 63, 90, 120, 252]
    cone_data = {}
    for p in periodos:
        s = log_r.rolling(p, min_periods=max(5, p//2)).std().dropna() * np.sqrt(252) * 100
        if len(s) >= 20:
            cone_data[p] = {pct: float(np.percentile(s, pct))
                            for pct in [10, 25, 50, 75, 90]}
            cone_data[p]['current'] = float(s.iloc[-1])

    if cone_data:
        pds      = sorted(cone_data.keys())
        fig_cone = go.Figure()
        for pct, clr, lbl, dsh in [
            (10, '#ff4b6e', 'P10',    'dot'),
            (25, '#f59e0b', 'P25',    'dot'),
            (50, '#e0e6ff', 'Median', 'solid'),
            (75, '#f59e0b', 'P75',    'dot'),
            (90, '#ff4b6e', 'P90',    'dot'),
        ]:
            fig_cone.add_trace(go.Scatter(
                x=pds, y=[cone_data[p][pct] for p in pds],
                mode='lines+markers', name=lbl,
                line=dict(color=clr, width=2, dash=dsh),
                marker=dict(size=5)
            ))
        fig_cone.add_trace(go.Scatter(
            x=pds, y=[cone_data[p]['current'] for p in pds],
            mode='lines+markers', name='HV Atual',
            line=dict(color='#00d4aa', width=3),
            marker=dict(size=9)
        ))
        fig_cone.add_hline(
            y=iv_ref,
            line_dash="solid", line_color="#7c83ff", line_width=2,
            annotation_text=f"IV Real OpLab {iv_ref:.1f}%",
            annotation_font_color="#7c83ff")
        fig_cone.update_layout(
            **PLOTLY_BASE, height=380,
            xaxis_title="Período (dias)",
            yaxis_title="Vol Realizada (%)")
        st.plotly_chart(fig_cone, use_container_width=True)


# ── TAB 4 — WATCHLIST ────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-title'>📡 Watchlist — IV Rank & Regime (OpLab)</div>",
                unsafe_allow_html=True)

    rows_wl = []
    prog    = st.progress(0)
    for i, ticker in enumerate(watchlist):
        prog.progress((i+1)/len(watchlist), text=f"Carregando {ticker}...")
        d = fetch_asset_data(ticker)
        if d:
            ivr = d["iv_rank"]
            rlb, rclr, _, rrec = regime_label(ivr)
            rows_wl.append({
                "Ticker":     ticker,
                "Nome":       d["name"][:22],
                "Preço":      f"R$ {d['price']:.2f}",
                "Var %":      f"{d['change_pct']:+.2f}%",
                "IV Real":    f"{d['iv_real']*100:.1f}%",
                "Fonte IV":   d["iv_fonte"],
                "HV 21d":     f"{d['hv21']:.1f}%" if d['hv21'] else "—",
                "IV Rank 1Y": f"{d['iv_rank']:.0f}",
                "IV Perc 1Y": f"{d['iv_pct']:.0f}",
                "Regime":     rlb,
                "EWMA":       f"{d['ewma']:.1f}%" if d.get('ewma') else "—",
                "GARCH":      f"{d['garch']:.1f}%" if d.get('garch') else "—",
                "Beta IBOV":  f"{d['beta']:.2f}" if d['beta'] else "—",
                "Selic":      f"{d['selic']*100:.2f}%",
                "Prox Vnc":   d["prox_venc"].strftime("%d/%m/%Y"),
                "DY":         f"{d['div_yield']:.2f}%",
            })
    prog.empty()

    if rows_wl:
        st.dataframe(pd.DataFrame(rows_wl), use_container_width=True,
                     hide_index=True, height=420)

        st.markdown("<div class='section-title'>IV Rank 1Y Comparativo (OpLab)</div>",
                    unsafe_allow_html=True)
        ivranks = [float(r["IV Rank 1Y"]) for r in rows_wl]
        tickers = [r["Ticker"]            for r in rows_wl]
        colors  = ["#ff4b6e" if v > 70
                   else "#00d4aa" if v < 30
                   else "#f59e0b"
                   for v in ivranks]

        fig_wl = go.Figure(go.Bar(
            x=tickers, y=ivranks,
            marker_color=colors,
            text=[f"{v:.0f}" for v in ivranks],
            textposition='outside',
            textfont=dict(color='white')
        ))
        fig_wl.add_hline(y=70, line_dash="dash", line_color="#ff4b6e",
                         annotation_text="Vender Vol",
                         annotation_font_color="#ff4b6e")
        fig_wl.add_hline(y=30, line_dash="dash", line_color="#00d4aa",
                         annotation_text="Comprar Vol",
                         annotation_font_color="#00d4aa")
        fig_wl.update_layout(**PLOTLY_BASE, height=340,
                             yaxis=dict(range=[0, 110]))
        st.plotly_chart(fig_wl, use_container_width=True)


# ── TAB 5 — CENÁRIOS ─────────────────────────────────────────────
with tab5:
    st.markdown(
        "<div class='section-title'>🎯 Cenários de Opções — BSM com IV Real OpLab</div>",
        unsafe_allow_html=True)

    S_c  = mkt["price"]
    T_c  = dash_T
    r_c  = mkt["selic"]
    iv_c = mkt["iv_real"]
    q_c  = mkt["div_yield"] / 100 if mkt["div_yield"] else 0

    st.markdown(f"""
    <div class='div-card' style='margin-bottom:12px'>
        <b style='color:#7c83ff'>Parâmetros BSM Automáticos</b> &nbsp;
        {badge_fonte('OpLab')} {badge_fonte('BCB')} {badge_fonte('Calculado')}
        <br><br>
        <span style='color:#8b92a5'>Spot:</span>
        <b style='color:#e0e6ff'>R$ {S_c:.2f}</b> &nbsp;|&nbsp;
        <span style='color:#8b92a5'>IV (OpLab):</span>
        <b style='color:#7c83ff'>{iv_c*100:.1f}%</b> &nbsp;|&nbsp;
        <span style='color:#8b92a5'>IV Rank 1Y:</span>
        <b style='color:{reg_color}'>{mkt['iv_rank']:.0f}</b> &nbsp;|&nbsp;
        <span style='color:#8b92a5'>r (Selic):</span>
        <b style='color:#7ec850'>{r_c*100:.2f}%</b> &nbsp;|&nbsp;
        <span style='color:#8b92a5'>T:</span>
        <b style='color:#a78bfa'>{dash_T_dias}d</b> &nbsp;|&nbsp;
        <span style='color:#8b92a5'>q:</span>
        <b style='color:#00d4aa'>{q_c*100:.3f}%</b>
    </div>
    """, unsafe_allow_html=True)

    em_1sd = S_c * iv_c * np.sqrt(T_c)
    st.info(
        f"📌 **Expected Move (±1σ):** R$ {em_1sd:.2f}  |  "
        f"**Up:** R$ {S_c+em_1sd:.2f}  |  "
        f"**Down:** R$ {S_c-em_1sd:.2f}")

    K_c = st.number_input("Strike (K)", value=round(S_c, 2),
                          step=0.50, format="%.2f")

    cenarios = [
        ("🐂 Bull +10%",  S_c*1.10, iv_c*0.85),
        ("📈 Bull  +5%",  S_c*1.05, iv_c*0.90),
        ("↔️  Base",      S_c,       iv_c),
        ("📉 Bear  -5%",  S_c*0.95, iv_c*1.10),
        ("🐻 Bear -10%",  S_c*0.90, iv_c*1.20),
        ("💥 Crash -20%", S_c*0.80, iv_c*1.50),
    ]

    for opt_t in ['call', 'put']:
        emoji = '📗' if opt_t == 'call' else '📕'
        st.markdown(f"#### {emoji} {opt_t.upper()} — Strike K = R$ {K_c:.2f}")

        p0     = bs_price(S_c, K_c, T_c, r_c, iv_c, opt_t, q_c)
        greeks = bs_greeks(S_c, K_c, T_c, r_c, iv_c, opt_t, q_c)

        g1,g2,g3,g4,g5,g6 = st.columns(6)
        for col, gk, gl, gc in [
            (g1, "delta", "Δ Delta",   "#00d4aa"),
            (g2, "gamma", "Γ Gamma",   "#7c83ff"),
            (g3, "theta", "Θ Theta/d", "#ff4b6e"),
            (g4, "vega",  "ν Vega/1%", "#f59e0b"),
            (g5, "rho",   "ρ Rho/1%",  "#a78bfa"),
            (g6, "vanna", "Vanna",      "#e0e6ff"),
        ]:
            v = greeks.get(gk, 0)
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{gl}</div>
                    <div style='color:{gc};font-size:16px;font-weight:700'>{v:.4f}</div>
                </div>""", unsafe_allow_html=True)

        st.caption(f"Preço BSM: **R$ {p0:.4f}**")

        rows_c = []
        for label, S_s, iv_s in cenarios:
            p       = bs_price(S_s, K_c, T_c, r_c, iv_s, opt_t, q_c)
            pnl     = p - p0
            pnl_pct = pnl / p0 * 100 if p0 > 0 else 0
            rows_c.append({
                "Cenário":     label,
                "Spot":        f"R$ {S_s:.2f}",
                "IV":          f"{iv_s*100:.1f}%",
                "Preço Opção": f"R$ {p:.4f}",
                "P&L":         f"R$ {pnl:+.4f}",
                "P&L %":       f"{pnl_pct:+.1f}%",
            })
        st.dataframe(pd.DataFrame(rows_c), use_container_width=True, hide_index=True)
        st.markdown("")


# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<hr style='border-color:#2d3250;margin-top:30px'>
<p style='text-align:center;color:#4b5563;font-size:12px'>
    OpLab PRO · BCB API · yfinance · BSM+q · Greeks 1ª/2ª Ordem<br>
    IV Real · IV Rank 1Y · EWMA · GARCH · Vol Cone · Term Structure
</p>
""", unsafe_allow_html=True)

