"""
fetch_data.py
-------------
Fetches historical stock data for Allcargo Logistics from Yahoo Finance.
Saves raw CSV to data/raw/ and prints a quick summary.
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# ---------------- CONFIG ----------------
TICKER = "ALLCARGO.NS"          # NSE ticker. Use "532749.BO" for BSE
PERIOD = "5y"                   # 5 years of data
INTERVAL = "1d"                 # daily candles
RAW_DIR = os.path.join("data", "raw")


def fetch_stock_data(ticker: str = TICKER, period: str = PERIOD, interval: str = INTERVAL) -> pd.DataFrame:
    """Download stock data from Yahoo Finance and return as DataFrame."""
    print(f"📥 Fetching {ticker} data ({period}, {interval})...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)

    if df.empty:
        raise ValueError(f"❌ No data returned for {ticker}. Check the ticker symbol.")

    # Flatten multi-level columns (yfinance returns tuples when single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    print(f"✅ Fetched {len(df)} rows from {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


def save_raw(df: pd.DataFrame, ticker: str = TICKER) -> str:
    """Save DataFrame as CSV in data/raw/."""
    os.makedirs(RAW_DIR, exist_ok=True)
    safe_ticker = ticker.replace(".", "_")
    filepath = os.path.join(RAW_DIR, f"{safe_ticker}.csv")
    df.to_csv(filepath, index=False)
    print(f"💾 Saved → {filepath}")
    return filepath


def quick_summary(df: pd.DataFrame) -> None:
    """Print a quick overview of the data."""
    print("\n" + "=" * 55)
    print("📊 DATA SUMMARY")
    print("=" * 55)
    print(f"Rows          : {len(df)}")
    print(f"Columns       : {list(df.columns)}")
    print(f"Date range    : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"Latest Close  : ₹{df['Close'].iloc[-1]:.2f}")
    print(f"52W High      : ₹{df['Close'].tail(252).max():.2f}")
    print(f"52W Low       : ₹{df['Close'].tail(252).min():.2f}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print("=" * 55)
    print("\n🔎 Latest 5 rows:")
    print(df.tail().to_string(index=False))


def main():
    df = fetch_stock_data()
    save_raw(df)
    quick_summary(df)


if __name__ == "__main__":
    main()