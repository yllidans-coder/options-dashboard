import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Calculadora de Opções", page_icon="📊", layout="wide")
st.title("📊 Calculadora de Opções — Black-Scholes")

col1, col2 = st.columns(2)

with col1:
    S = st.number_input("Preço do Ativo (S)", value=100.0, step=1.0)
    K = st.number_input("Strike (K)", value=100.0, step=1.0)
    T = st.number_input("Tempo até vencimento (dias)", value=30, step=1) / 365
    r = st.number_input("Taxa livre de risco % (r)", value=10.75, step=0.25) / 100
    sigma = st.number_input("Volatilidade Implícita % (σ)", value=30.0, step=1.0) / 100

def black_scholes(S, K, T, r, sigma, tipo="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == "call":
        preco = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        preco = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if tipo=="call" else -d2)) / 365
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return preco, delta, gamma, theta, vega

with col2:
    st.subheader("Resultados")
    if T > 0:
        preco_call, delta_c, gamma, theta_c, vega = black_scholes(S, K, T, r, sigma, "call")
        preco_put, delta_p, _, theta_p, _ = black_scholes(S, K, T, r, sigma, "put")

        st.metric("💰 Call", f"R$ {preco_call:.4f}")
        st.metric("💰 Put", f"R$ {preco_put:.4f}")
        st.divider()
        st.metric("Δ Delta Call", f"{delta_c:.4f}")
        st.metric("Δ Delta Put", f"{delta_p:.4f}")
        st.metric("Γ Gamma", f"{gamma:.6f}")
        st.metric("Θ Theta Call/dia", f"{theta_c:.4f}")
        st.metric("ν Vega /1%σ", f"{vega:.4f}")
    else:
        st.warning("Tempo até vencimento deve ser maior que 0.")
