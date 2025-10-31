import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq

st.set_page_config(layout="wide")
st.title("📈 Monte Carlo Option Simulator — Call vs Put Dashboard with Live Updates")

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Live Stock Inputs")
ticker = st.sidebar.text_input("Ticker (e.g. AAPL)", value="AAPL").upper()
history_period = st.sidebar.selectbox("Price History Period", ["1mo","3mo","6mo","1y","2y"], index=1)
interval = st.sidebar.selectbox("History Interval", ["1d","1wk"], index=0)
refresh_interval = st.sidebar.number_input("Auto-refresh interval (seconds, 0=off)", 0, 300, 30, 5)

st.sidebar.markdown("---")
st.sidebar.header("Simulation Settings")
T = st.sidebar.slider("Time to Maturity (years)", 0.01, 5.0, 1.0, 0.01)
n_steps = st.sidebar.number_input("Time steps (per path)", 10, 252, 50, 1)
num_sim = st.sidebar.number_input("Total number of simulations", 50, 5000, 400, 50)
batch_size = st.sidebar.number_input("Simulations per animation wave", 5, 200, 20, 5)
K = st.sidebar.slider("Strike Price (K). If 0, defaults to spot", 0.0, 5000.0, 0.0, 0.5)
animation_sample_size = st.sidebar.slider("Max paths displayed for animation", 10, 500, 100, 10)
animation_speed = st.sidebar.slider("Animation speed (ms per frame)", 50, 1000, 250, 50)

# ---------------------------
# Fetch Stock Data
# ---------------------------
@st.cache_data(ttl=refresh_interval)
def fetch_data_live(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    return df

data = fetch_data_live(ticker, history_period, interval)
if data is None:
    st.error(f"Could not fetch data for {ticker}.")
    st.stop()

S0 = float(data["Close"].iloc[-1])
if K == 0.0:
    K = S0

# ---------------------------
# Historical volatility
# ---------------------------
log_returns = np.log(data["Close"]/data["Close"].shift(1)).dropna()
sigma = float(log_returns.std() * np.sqrt(252))

# ---------------------------
# Risk-free rate
# ---------------------------
def fetch_risk_free_rate():
    try:
        url = "https://api.stlouisfed.org/fred/series/observations?series_id=DGS1&api_key=YOUR_FRED_API_KEY&file_type=json"
        r = requests.get(url).json()
        last_val = float(r['observations'][-1]['value'])
        if last_val <= 0:
            return 0.03
        return last_val/100
    except:
        return 0.03
r = fetch_risk_free_rate()

# ---------------------------
# Black-Scholes function
# ---------------------------
def black_scholes_price(S,K,T,r,sigma,option_type="call"):
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type=="call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def implied_volatility(target_price, S,K,T,r,option_type="call"):
    try:
        func = lambda sigma: black_scholes_price(S,K,T,r,sigma,option_type)-target_price
        return brentq(func,1e-6,5.0)
    except:
        return np.nan

# ---------------------------
# Monte Carlo Simulation
# ---------------------------
def monte_carlo_paths(S0, sigma, r, T, n_steps, num_sim):
    dt = T/n_steps
    Z = np.random.default_rng(seed=42).standard_normal((num_sim, n_steps))
    increments = (r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    log_paths = np.concatenate([np.full((num_sim,1), np.log(S0)), np.cumsum(increments, axis=1)+np.log(S0)], axis=1)
    price_paths = np.exp(log_paths)
    return price_paths

price_paths = monte_carlo_paths(S0, sigma, r, T, n_steps, num_sim)
ST = price_paths[:,-1]

payoffs_call = np.exp(-r*T)*np.maximum(ST-K,0)
payoffs_put  = np.exp(-r*T)*np.maximum(K-ST,0)

mc_call = float(np.mean(payoffs_call))
mc_put  = float(np.mean(payoffs_put))
mc_call_se = float(np.std(payoffs_call)/np.sqrt(num_sim))
mc_put_se  = float(np.std(payoffs_put)/np.sqrt(num_sim))
iv_call = implied_volatility(mc_call, S0,K,T,r,"call")
iv_put  = implied_volatility(mc_put, S0,K,T,r,"put")
bs_call = black_scholes_price(S0,K,T,r,sigma,"call")
bs_put  = black_scholes_price(S0,K,T,r,sigma,"put")

# ---------------------------
# Summary Metrics
# ---------------------------
st.subheader(f"{ticker} — Summary")
col1, col2 = st.columns(2)
with col1:
    st.write(f"- Spot (S₀): ${S0:.2f}")
    st.write(f"- Strike (K): ${K:.2f}")
    st.write(f"- Time to Maturity (T): {T} yrs")
with col2:
    st.write(f"- Volatility (σ): {sigma:.2%}")
    st.write(f"- Risk-free Rate (r): {r:.2%}")
st.write(f"- Black–Scholes Call Price: ${bs_call:.6f}")
st.write(f"- Black–Scholes Put Price: ${bs_put:.6f}")
st.write(f"- MC Call Price: ${mc_call:.6f} ± {mc_call_se:.6f}")
st.write(f"- MC Put Price: ${mc_put:.6f} ± {mc_put_se:.6f}")
st.write(f"- Implied Vol Call: {iv_call:.2%}, Implied Vol Put: {iv_put:.2%}")

# ---------------------------
# Tabs for Plots
# ---------------------------
tab1, tab2, tab3 = st.tabs(["MC Paths", "Terminal Histogram", "MC Convergence"])

# ----- Tab1: Animated Paths -----
with tab1:
    subset_size = min(animation_sample_size, num_sim)
    subset_indices = np.random.choice(num_sim, subset_size, replace=False)
    subset_paths = price_paths[subset_indices]
    time_axis = list(range(n_steps+1))
    batch_inds = list(range(batch_size, subset_size+1, batch_size))
    if batch_inds[-1] != subset_size:
        batch_inds.append(subset_size)

    fig_anim = go.Figure(layout=go.Layout(
        xaxis=dict(title="Time Step"),
        yaxis=dict(title="Simulated Price"),
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None, {"frame":{"duration":animation_speed,"redraw":True},
                                                     "fromcurrent":True,"transition":{"duration":0}}])])],
        height=600
    ))

    fig_anim.add_trace(go.Scatter(x=time_axis, y=[None]*(n_steps+1), mode="lines", line=dict(width=3), name="Mean Path"))
    for i in range(subset_size):
        fig_anim.add_trace(go.Scatter(x=time_axis, y=subset_paths[i], mode="lines", line=dict(width=1),
                                      visible=(i<batch_inds[0]), opacity=0.6, hoverinfo="skip"))

    frames = []
    for m in batch_inds:
        mean_path = np.mean(subset_paths[:m,:], axis=0)
        frame_data = [go.Scatter(x=time_axis, y=mean_path, mode="lines", line=dict(width=3))]
        for i in range(subset_size):
            y = subset_paths[i] if i<m else [None]*(n_steps+1)
            frame_data.append(go.Scatter(x=time_axis, y=y, mode="lines", line=dict(width=1)))
        frames.append(go.Frame(data=frame_data, name=str(m),
                               layout=go.Layout(title=f"Revealed {m}/{subset_size} paths")))
    fig_anim.frames = frames
    st.plotly_chart(fig_anim, use_container_width=True)

# ----- Tab2: Terminal Histogram -----
with tab2:
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=ST, nbinsx=40, name="Terminal Price", opacity=0.6, marker_color='blue'))
    hist_fig.update_layout(height=450, xaxis_title="S_T", yaxis_title="Count")
    st.plotly_chart(hist_fig, use_container_width=True)

# ----- Tab3: MC Convergence -----
with tab3:
    cumulative_call = np.cumsum(payoffs_call)/np.arange(1,num_sim+1)
    cumulative_put  = np.cumsum(payoffs_put)/np.arange(1,num_sim+1)
    sim_indices = np.arange(1,num_sim+1)
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(x=sim_indices, y=cumulative_call, mode='lines', name='MC Call', line=dict(color='blue')))
    fig_conv.add_trace(go.Scatter(x=sim_indices, y=cumulative_put, mode='lines', name='MC Put', line=dict(color='green')))
    fig_conv.add_trace(go.Scatter(x=[1,num_sim], y=[bs_call,bs_call], mode='lines', name='BS Call', line=dict(color='red', dash='dash')))
    fig_conv.add_trace(go.Scatter(x=[1,num_sim], y=[bs_put,bs_put], mode='lines', name='BS Put', line=dict(color='orange', dash='dash')))
    fig_conv.update_layout(xaxis_title="Number of Simulations", yaxis_title="Option Price", height=450)
    st.plotly_chart(fig_conv, use_container_width=True)
