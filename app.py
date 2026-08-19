import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import timedelta

# --- Page Configuration ---
st.set_page_config(page_title="MarketSight: Equity Analysis", layout="wide")

# --- UI/UX: Corporate CSS ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1, h2, h3 {color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    </style>
    """, unsafe_allow_html=True)

st.title("MarketSight: Equity Trend Analysis")
st.markdown("Quantitative market analysis and trend forecasting dashboard.")

# --- Pre-defined Company Roster ---
COMPANY_TICKERS = {
    "Reliance Industries (India)": "RELIANCE.NS",
    "Tata Consultancy Services (India)": "TCS.NS",
    "HDFC Bank (India)": "HDFCBANK.NS",
    "Infosys (India)": "INFY.NS",
    "Apple Inc. (Global)": "AAPL",
    "Microsoft Corp. (Global)": "MSFT",
    "NVIDIA Corp. (Global)": "NVDA",
    "Tesla Inc. (Global)": "TSLA"
}

# Approximate USD to INR exchange rate (July 2026)
USD_TO_INR = 95.70

# --- Sidebar Controls ---
st.sidebar.header("Analysis Parameters")
selected_company = st.sidebar.selectbox("Select Asset:", list(COMPANY_TICKERS.keys()))
prediction_days = st.sidebar.slider("Forecast Horizon (Days):", 1, 30, 7)

ticker_symbol = COMPANY_TICKERS[selected_company]
is_global = "(Global)" in selected_company

if ticker_symbol:
    try:
        # Fetching Data
        with st.spinner("Retrieving market data..."):
            stock = yf.Ticker(ticker_symbol)
            data = stock.history(period="1y")
            
        if data.empty:
            st.error("Market data currently unavailable for this asset.")
        else:
            # Convert global stock prices from USD to INR
            if is_global:
                data[['Open', 'High', 'Low', 'Close']] = data[['Open', 'High', 'Low', 'Close']] * USD_TO_INR
            
            # --- KPI Metrics Section ---
            current_price = data['Close'].iloc[-1]
            previous_price = data['Close'].iloc[-2]
            price_change = current_price - previous_price
            pct_change = (price_change / previous_price) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"₹{current_price:,.2f}", f"₹{price_change:,.2f} ({pct_change:.2f}%)")
            col2.metric("52-Week High", f"₹{data['High'].max():,.2f}")
            col3.metric("Trading Volume", f"{data['Volume'].iloc[-1]:,}")
            
            st.markdown("---")
            
            # --- Machine Learning Model (Linear Regression) ---
            df = data[['Close']].copy()
            df['Day_Index'] = np.arange(len(df))
            X = df[['Day_Index']].values
            y = df['Close'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecasting
            last_index = df['Day_Index'].iloc[-1]
            future_X = np.arange(last_index + 1, last_index + 1 + prediction_days).reshape(-1, 1)
            future_y = model.predict(future_X)
            
            future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=prediction_days)
            
            # --- Tabbed Interface ---
            tab1, tab2 = st.tabs(["Historical Performance", "Trend Forecast"])
            
            with tab1:
                st.subheader("Asset Price History")
                # Professional Candlestick Chart (Light Theme)
                fig_hist = go.Figure(data=[go.Candlestick(x=data.index,
                                open=data['Open'], high=data['High'],
                                low=data['Low'], close=data['Close'])])
                fig_hist.update_layout(xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with tab2:
                st.subheader(f"{prediction_days}-Day Moving Trend Forecast")
                # Clean Line Chart for Predictions
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Price', line=dict(color='#2980b9', width=2)))
                fig_pred.add_trace(go.Scatter(x=df.index, y=model.predict(X), mode='lines', name='Model Fit', line=dict(color='#7f8c8d', dash='dash')))
                fig_pred.add_trace(go.Scatter(x=future_dates, y=future_y, mode='lines+markers', name='Forecasted Trend', line=dict(color='#27ae60', width=3)))
                
                fig_pred.update_layout(template="plotly_white", hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Clean, professional projection text
                st.info(f"Projected trend value by {future_dates[-1].strftime('%Y-%m-%d')}: ₹{future_y[-1]:,.2f}")

    except Exception as e:
        st.error(f"System error: {e}")