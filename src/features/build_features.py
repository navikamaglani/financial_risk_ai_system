import pandas as pd
import numpy as np
from pathlib import Path

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)


def engineer_features(file_path):

    ticker = file_path.stem

    print(f"Processing {ticker}...")

    # Read CSV
    df = pd.read_csv(file_path)

    # Convert date column
    df["Date"] = pd.to_datetime(df["Date"])

    # Convert numeric columns
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort properly
    df = df.sort_values("Date")

    # -----------------------------
    # Daily Returns
    # -----------------------------
    df["Daily_Return"] = df["Close"].pct_change()

    # -----------------------------
    # Rolling Volatility (21-day)
    # -----------------------------
    df["Rolling_Volatility"] = (
        df["Daily_Return"]
        .rolling(window=21)
        .std()
    )

    # -----------------------------
    # Moving Averages
    # -----------------------------
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # -----------------------------
    # RSI
    # -----------------------------
    rsi = RSIIndicator(close=df["Close"], window=14)

    df["RSI"] = rsi.rsi()

    # -----------------------------
    # MACD
    # -----------------------------
    macd = MACD(close=df["Close"])

    df["MACD"] = macd.macd()

    df["MACD_Signal"] = macd.macd_signal()

    # -----------------------------
    # Bollinger Bands
    # -----------------------------
    bb = BollingerBands(close=df["Close"], window=20)

    df["BB_High"] = bb.bollinger_hband()

    df["BB_Low"] = bb.bollinger_lband()

    # -----------------------------
    # Momentum
    # -----------------------------
    df["Momentum_10"] = (
        df["Close"] - df["Close"].shift(10)
    )

    # Drop missing values
    df = df.dropna()

    # Save processed file
    output_path = PROCESSED_DATA_PATH / f"{ticker}_features.csv"

    df.to_csv(output_path, index=False)

    print(f"Saved -> {output_path}")


if __name__ == "__main__":

    csv_files = RAW_DATA_PATH.glob("*.csv")

    for file_path in csv_files:
        engineer_features(file_path)

    print("Feature engineering completed.")