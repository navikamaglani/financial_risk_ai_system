import sys
import joblib
import yfinance as yf
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("models/risk_classifier.pkl")


# -----------------------------
# Risk Labels
# -----------------------------
RISK_MAP = {
    0: "LOW RISK",
    1: "MEDIUM RISK",
    2: "HIGH RISK"
}


# -----------------------------
# Download Latest Data
# -----------------------------
def download_latest_data(ticker):

    df = yf.download(
        ticker,
        period="6mo",
        auto_adjust=True
    )

    df.reset_index(inplace=True)

    # Flatten multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


# -----------------------------
# Feature Engineering
# -----------------------------
def create_features(df):

    # Daily Return
    df["Daily_Return"] = df["Close"].pct_change(fill_method=None)

    # Rolling Volatility
    df["Rolling_Volatility"] = (
        df["Daily_Return"]
        .rolling(window=21)
        .std()
    )

    # Moving Averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # RSI
    rsi = RSIIndicator(close=df["Close"], window=14)

    df["RSI"] = rsi.rsi()

    # MACD
    macd = MACD(close=df["Close"])

    df["MACD"] = macd.macd()

    df["MACD_Signal"] = macd.macd_signal()

    # Bollinger Bands
    bb = BollingerBands(close=df["Close"], window=20)

    df["BB_High"] = bb.bollinger_hband()

    df["BB_Low"] = bb.bollinger_lband()

    # Momentum
    df["Momentum_10"] = (
        df["Close"] - df["Close"].shift(10)
    )

    # Drop missing rows
    df = df.dropna()

    return df


# -----------------------------
# Prediction Pipeline
# -----------------------------
def predict_risk(ticker):

    print(f"\nAnalyzing {ticker}...\n")

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

    X = latest_row[feature_columns].values.reshape(1, -1)

    prediction = model.predict(X)[0]

    print(f"Predicted Risk Level: {RISK_MAP[prediction]}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python predict_risk.py <TICKER>")
        sys.exit()

    ticker = sys.argv[1].upper()

    predict_risk(ticker)