"""
preprocess.py
-------------
Loads raw CSV, cleans it, engineers the target variable (next-day Close),
and provides train/test split helpers.
"""

import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")


# ---------------- LOAD ----------------
def load_raw(ticker: str = "ALLCARGO.NS") -> pd.DataFrame:
    """Load raw CSV from data/raw/."""
    safe_ticker = ticker.replace(".", "_")
    filepath = os.path.join(RAW_DIR, f"{safe_ticker}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ {filepath} not found. Run fetch_data.py first.")
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"✅ Loaded {len(df)} rows from {filepath}")
    return df


# ---------------- CLEAN ----------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean: drop duplicates, forward-fill nulls, remove bad rows."""
    before = len(df)
    df = df.drop_duplicates(subset=["Date"]).copy()

    # Forward-fill small gaps (holidays etc.)
    df = df.ffill()

    # Drop rows where essential columns are still NaN
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    # Remove zero-volume days (market closed)
    df = df[df["Volume"] > 0]

    print(f"🧹 Cleaned: {before} → {len(df)} rows")
    return df.reset_index(drop=True)


# ---------------- TARGET ----------------
def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Add target column = Close price N days in future.
    horizon=1 → predict tomorrow's close.
    """
    df = df.copy()
    df["Target"] = df["Close"].shift(-horizon)
    df = df.dropna(subset=["Target"]).reset_index(drop=True)
    print(f"🎯 Added target (predicting {horizon}-day-ahead Close). Rows: {len(df)}")
    return df


# ---------------- SPLIT ----------------
def train_test_split_time(df: pd.DataFrame, test_size: float = 0.2):
    """
    Time-series split (NO shuffling!). Returns train_df, test_df.
    """
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    print(f"📚 Train: {len(train)} rows | Test: {len(test)} rows")
    print(f"   Train range: {train['Date'].min().date()} → {train['Date'].max().date()}")
    print(f"   Test range : {test['Date'].min().date()} → {test['Date'].max().date()}")
    return train, test


# ---------------- SAVE ----------------
def save_processed(df: pd.DataFrame, name: str = "cleaned") -> str:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filepath = os.path.join(PROCESSED_DIR, f"{name}.csv")
    df.to_csv(filepath, index=False)
    print(f"💾 Saved → {filepath}")
    return filepath


# ---------------- MAIN ----------------
def main():
    df = load_raw()
    df = clean_data(df)
    df = add_target(df, horizon=1)
    train, test = train_test_split_time(df, test_size=0.2)

    save_processed(train, "train")
    save_processed(test, "test")

    print("\n" + "=" * 55)
    print("📊 PREPROCESSING SUMMARY")
    print("=" * 55)
    print(f"Total usable rows : {len(df)}")
    print(f"Train rows        : {len(train)}")
    print(f"Test rows         : {len(test)}")
    print(f"Latest Close      : ₹{df['Close'].iloc[-1]:.2f}")
    print(f"Latest Target     : ₹{df['Target'].iloc[-1]:.2f}")
    print("=" * 55)
    print("\n🔎 Sample rows (train tail):")
    print(train.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()