# Home.py
import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Painel de Opções",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CSS GLOBAL
# ─────────────────────────────────────────
st.markdown("""
<style>
    /* Fundo geral */
    .stApp { background-color: #0e1117; color: #e0e0e0; }

    /* Cards de métricas customizados */
    .card {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #2e3a4e;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .card-label {
        font-size: 0.78rem;
        color: #8899aa;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #e0e0e0;
    }
    .card-delta-up   { font-size: 0.85rem; color: #00e676; margin-top: 4px; }
    .card-delta-down { font-size: 0.85rem; color: #ff5252; margin-top: 4px; }
    .card-delta-neu  { font-size: 0.85rem; color: #8899aa; margin-top: 4px; }

    /* Cards de navegação */
    .nav-card {
        background: linear-gradient(135deg, #1a1f2e, #1e2d40);
        border: 1px solid #2e5080;
        border-radius: 14px;
        padding: 28px 24px;
        text-align: center;
        cursor: pointer;
        transition: all 0.25s;
    }
    .nav-card:hover {
        border-color: #4a9eff;
        box-shadow: 0 0 20px rgba(74,158,255,0.25);
        transform: translateY(-4px);
    }
    .nav-card h3 { color: #4a9eff; font-size: 1.2rem; margin: 10px 0 6px; }
    .nav-card p  { color: #8899aa; font-size: 0.85rem; margin: 0; }
    .nav-icon    { font-size: 2.2rem; }

    /* Divider estilizado */
    .divider {
        border: none;
        border-top: 1px solid #2e3a4e;
        margin: 24px 0;
    }

    /* Header */
    .header-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .header-sub {
        font-size: 0.95rem;
        color: #8899aa;
        margin-top: -6px;
    }
    .badge {
        display: inline-block;
        background: #1e2d40;
        border: 1px solid #2e5080;
        color: #4a9eff;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        margin-left: 10px;
        vertical-align: middle;
    }

    /* Esconde botões padrão do Streamlit nos metrics */
    div[data-testid="metric-container"] {
        background: transparent;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1e2a3a;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# FUNÇÕES DE DADOS
# ─────────────────────────────────────────

@st.cache_data(ttl=300)
def get_market_data():
    """Busca IBOV, VIX, USDBRL, PETR4, VALE3 via yfinance"""
    tickers = {
        "IBOV":   "^BVSP",
        "VIX":    "^VIX",
        "USDBRL": "BRL=X",
        "PETR4":  "PETR4.SA",
        "VALE3":  "VALE3.SA",
    }
    resultado = {}
    for nome, ticker in tickers.items():
        try:
            dados = yf.Ticker(ticker).history(period="5d", interval="1d")
            if len(dados) >= 2:
                ultimo   = float(dados["Close"].iloc[-1])
                anterior = float(dados["Close"].iloc[-2])
                variacao = ((ultimo - anterior) / anterior) * 100
                resultado[nome] = {"valor": ultimo, "var": variacao}
            elif len(dados) == 1:
                ultimo = float(dados["Close"].iloc[-1])
                resultado[nome] = {"valor": ultimo, "var": 0.0}
            else:
                resultado[nome] = {"valor": None, "var": 0.0}
        except Exception:
            resultado[nome] = {"valor": None, "var": 0.0}
    return resultado


@st.cache_data(ttl=3600)
def get_selic():
    """Busca Selic atual via API do Banco Central"""
    try:
        url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"
        resp = requests.get(url, timeout=5)
        dados = resp.json()
        return float(dados[0]["valor"].replace(",", "."))
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_historical(ticker_yf: str, days: int = 30):
    """Retorna histórico para mini-gráfico"""
    try:
        end   = datetime.today()
        start = end - timedelta(days=days)
        df = yf.download(ticker_yf, start=start, end=end, interval="1d", progress=False)
        df = df[["Close"]].dropna()
        df.columns = ["Close"]
        return df
    except Exception:
        return pd.DataFrame()


def mini_chart(df: pd.DataFrame, cor: str = "#4a9eff", height: int = 80):
    """Gráfico sparkline minimalista"""
    if df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines",
        line=dict(color=cor, width=2),
        fill="tozeroy",
        fillcolor=cor.replace(")", ",0.08)").replace("rgb", "rgba") if "rgb" in cor else cor + "14",
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def fmt_valor(nome: str, valor: float):
    """Formata valor conforme o ativo"""
    if valor is None:
        return "—"
    if nome == "IBOV":
        return f"{valor:,.0f}".replace(",", ".")
    if nome in ("PETR4", "VALE3"):
        return f"R$ {valor:.2f}"
    if nome == "USDBRL":
        return f"R$ {valor:.4f}"
    return f"{valor:.2f}"


def delta_html(var: float):
    sinal = "▲" if var > 0 else ("▼" if var < 0 else "—")
    cls   = "card-delta-up" if var > 0 else ("card-delta-down" if var < 0 else "card-delta-neu")
    return f'<div class="{cls}">{sinal} {abs(var):.2f}%</div>'


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Painel de Opções")
    st.markdown("---")
    st.markdown("**Navegação**")
    st.page_link("Home.py",                              label="🏠 Home")
    st.page_link("pages/1_Dashboard.py",                 label="📊 Dashboard")
    st.page_link("pages/2_Calculadora_de_Opcoes.py",     label="🧮 Calculadora")  # ← CORRIGIDO
    st.markdown("---")
    st.markdown(
        "<small style='color:#556677'>Dados via Yahoo Finance & BCB<br>"
        "Atualização: 5 min</small>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<small style='color:#445566'>🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style='padding: 10px 0 20px 0;'>
    <div class='header-title'>
        📈 Painel de Opções
        <span class='badge'>BETA</span>
    </div>
    <div class='header-sub'>
        Análise de derivativos · Black-Scholes · Greeks · Mercado ao vivo
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DADOS DE MERCADO
# ─────────────────────────────────────────
with st.spinner("Carregando dados de mercado..."):
    mercado = get_market_data()
    selic   = get_selic()

st.markdown("### 🌐 Mercado Agora")

# ── Linha 1: IBOV · VIX · USD/BRL · SELIC ──
c1, c2, c3, c4 = st.columns(4)

def render_card(col, label, nome_key, selic_val=None):
    with col:
        if selic_val is not None:
            st.markdown(f"""
            <div class='card'>
                <div class='card-label'>🏦 Selic a.a.</div>
                <div class='card-value'>{f"{selic_val:.2f}%" if selic_val else "—"}</div>
                <div class='card-delta-neu'>Banco Central do Brasil</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            d = mercado.get(nome_key, {})
            valor = d.get("valor")
            var   = d.get("var", 0.0)
            st.markdown(f"""
            <div class='card'>
                <div class='card-label'>{label}</div>
                <div class='card-value'>{fmt_valor(nome_key, valor)}</div>
                {delta_html(var)}
            </div>
            """, unsafe_allow_html=True)

render_card(c1, "🇧🇷 IBOVESPA",  "IBOV")
render_card(c2, "😨 VIX (Medo)", "VIX")
render_card(c3, "💵 USD / BRL",  "USDBRL")
render_card(c4, "",              "",      selic_val=selic)

st.markdown("<br>", unsafe_allow_html=True)

# ── Linha 2: PETR4 · VALE3 ──
c5, c6, _, _ = st.columns(4)
render_card(c5, "🛢️ PETR4", "PETR4")
render_card(c6, "⛏️ VALE3", "VALE3")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# MINI-GRÁFICOS HISTÓRICOS
# ─────────────────────────────────────────
st.markdown("### 📉 Histórico 30 dias")

gc1, gc2, gc3 = st.columns(3)

charts = [
    (gc1, "IBOV",   "^BVSP",  "#4a9eff", "IBOVESPA"),
    (gc2, "VIX",    "^VIX",   "#ff6b6b", "VIX"),
    (gc3, "USDBRL", "BRL=X",  "#ffd166", "USD/BRL"),
]

for col, nome, ticker_yf, cor, titulo in charts:
    with col:
        st.markdown(f"**{titulo}**")
        df_hist = get_historical(ticker_yf, days=30)
        fig = mini_chart(df_hist, cor=cor, height=120)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Dados indisponíveis")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# CARDS DE NAVEGAÇÃO
# ─────────────────────────────────────────
st.markdown("### 🧭 Navegar")

n1, n2 = st.columns(2)

with n1:
    st.markdown("""
    <div class='nav-card'>
        <div class='nav-icon'>📊</div>
        <h3>Dashboard</h3>
        <p>Análise completa de volatilidade, Greeks, skew e term structure de qualquer ativo</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Ir para o Dashboard →", use_container_width=True, key="btn_dash"):
        st.switch_page("pages/1_Dashboard.py")

with n2:
    st.markdown("""
    <div class='nav-card'>
        <div class='nav-icon'>🧮</div>
        <h3>Calculadora de Opções</h3>
        <p>Precificação Black-Scholes, Greeks completos, payoff e análise de estratégias</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Ir para a Calculadora →", use_container_width=True, key="btn_calc"):
        st.switch_page("pages/2_Calculadora_de_Opcoes.py")  # ← CORRIGIDO

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#445566; font-size:0.78rem; padding: 20px 0 10px;'>
    Dados fornecidos por Yahoo Finance e Banco Central do Brasil · Apenas fins educacionais<br>
    Não constitui recomendação de investimento · Use com responsabilidade
</div>
""", unsafe_allow_html=True)
