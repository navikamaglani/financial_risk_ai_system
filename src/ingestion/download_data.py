import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = [
    "AAPL",
    "TSLA",
    "NVDA",
    "AMZN",
    "MSFT",
    "META",
    "GOOGL",
    "SPY",
    "QQQ"
]

START_DATE = "2020-01-01"
END_DATE = "2026-01-01"

RAW_DATA_PATH = Path("data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

def download_stock_data(ticker):
    print(f"Downloading {ticker}...")
    
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True
    )

    df.reset_index(inplace=True)

    df["Ticker"] = ticker

    file_path = RAW_DATA_PATH / f"{ticker}.csv"

    df.to_csv(file_path, index=False)

    print(f"Saved -> {file_path}")

if __name__ == "__main__":
    for ticker in TICKERS:
        download_stock_data(ticker)

    print("All downloads completed.")