from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

app = FastAPI(title="MarketSight Analytics API")

# Enable CORS so your React frontend can talk to your Python backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact React local URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COMPANY_DICTIONARY = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS",
    "SBIN": "SBIN.NS",
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "NVDA": "NVDA",
    "TSLA": "TSLA"
}

@app.get("/api/forecast/{ticker_id}")
async def get_forecast(ticker_id: str, days: int = 7):
    ticker = ticker_id.upper()
    if ticker not in COMPANY_DICTIONARY:
        raise HTTPException(status_code=404, detail="Ticker asset not supported.")
        
    symbol = COMPANY_DICTIONARY[ticker]
    
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1y")
        
        # THE SHIELD: Deletes blank market days so the math doesn't crash
        data = data.dropna() 
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data available.")
            
        # Metrics Calculations
        current_price = float(data['Close'].iloc[-1])
        previous_price = float(data['Close'].iloc[-2])
        price_change = current_price - previous_price
        pct_change = (price_change / previous_price) * 100
        
        # Linear Regression Modeling
        df = data[['Close']].copy()
        df['Index'] = np.arange(len(df))
        
        X = df[['Index']].values
        y = df['Close'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast Horizon
        last_index = df['Index'].iloc[-1]
        future_X = np.arange(last_index + 1, last_index + 1 + days).reshape(-1, 1)
        future_y = model.predict(future_X)
        
        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days)
        
        # Format Response Arrays for Frontend Charts
        historical_points = [{"date": str(d.date()), "price": round(float(p), 2)} for d, p in zip(data.index, data['Close'])]
        forecast_points = [{"date": str(d.date()), "price": round(float(p), 2)} for d, p in zip(future_dates, future_y)]
        
        return {
            "ticker": ticker,
            "currency": "₹" if symbol.endswith(".NS") else "$",
            "metrics": {
                "currentPrice": round(current_price, 2),
                "change": round(price_change, 2),
                "pctChange": round(pct_change, 2),
                "high52": round(float(data['High'].max()), 2),
                "low52": round(float(data['Low'].min()), 2),
                "volume": int(data['Volume'].iloc[-1])
            },
            "historicalData": historical_points,
            "forecastData": forecast_points
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))