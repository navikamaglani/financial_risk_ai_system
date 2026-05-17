from fastapi import FastAPI
import joblib
import yfinance as yf
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Financial Risk Intelligence API",
    description="AI-powered financial risk prediction system",
    version="1.0"
)


# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/risk_classifier.pkl")


# -----------------------------
# Risk Mapping
# -----------------------------
RISK_MAP = {
    0: "LOW RISK",
    1: "MEDIUM RISK",
    2: "HIGH RISK"
}


# -----------------------------
# Download Data
# -----------------------------
def download_latest_data(ticker):

    df = yf.download(
        ticker,
        period="6mo",
        auto_adjust=True
    )

    df.reset_index(inplace=True)

    # Fix multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


# -----------------------------
# Feature Engineering
# -----------------------------
def create_features(df):

    df["Daily_Return"] = df["Close"].pct_change(fill_method=None)

    df["Rolling_Volatility"] = (
        df["Daily_Return"]
        .rolling(window=21)
        .std()
    )

    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    rsi = RSIIndicator(close=df["Close"], window=14)

    df["RSI"] = rsi.rsi()

    macd = MACD(close=df["Close"])

    df["MACD"] = macd.macd()

    df["MACD_Signal"] = macd.macd_signal()

    bb = BollingerBands(close=df["Close"], window=20)

    df["BB_High"] = bb.bollinger_hband()

    df["BB_Low"] = bb.bollinger_lband()

    df["Momentum_10"] = (
        df["Close"] - df["Close"].shift(10)
    )

    df = df.dropna()

    return df


# -----------------------------
# Home Route
# -----------------------------
@app.get("/")
def home():

    return {
        "message": "Financial Risk Intelligence API is running"
    }


# -----------------------------
# Prediction Route
# -----------------------------
@app.get("/predict-risk/{ticker}")
def predict_risk(ticker: str):

    ticker = ticker.upper()

    df = download_latest_data(ticker)

    df = create_features(df)

    latest_row = df.iloc[-1]

    feature_columns = [
        "Daily_Return",
        "Rolling_Volatility",
        "SMA_20",
        "SMA_50",
        "RSI",
        "MACD",
        "MACD_Signal",
        "BB_High",
        "BB_Low",
        "Momentum_10"
    ]

    X = pd.DataFrame([latest_row[feature_columns]])

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    return {
        "ticker": ticker,
        "predicted_risk": RISK_MAP[prediction],
        "probabilities": {
            "LOW_RISK": round(float(probabilities[0]) * 100, 2),
            "MEDIUM_RISK": round(float(probabilities[1]) * 100, 2),
            "HIGH_RISK": round(float(probabilities[2]) * 100, 2)
    }
}