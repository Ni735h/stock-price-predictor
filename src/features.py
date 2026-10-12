"""
features.py — v2 (STATIONARY MODE)
----------------------------------
Uses relative/ratio features only, avoids unscaled absolutes.
"""

import os
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")


# ---------------- INDICATORS ----------------
def add_moving_averages(df):
    for w in (5, 10, 20, 50):
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
        df[f"SMA_{w}_ratio"] = df["Close"] / (df[f"SMA_{w}"] + 1e-9)
    return df


def add_ema(df):
    for s in (12, 26):
        df[f"EMA_{s}"] = df["Close"].ewm(span=s, adjust=False).mean()
        df[f"EMA_{s}_ratio"] = df["Close"] / (df[f"EMA_{s}"] + 1e-9)
    return df


def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    df["MACD"] = macd / (df["Close"] + 1e-9)
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger(df, window=20, num_std=2):
    sma = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    df["BB_Upper_ratio"] = df["Close"] / (upper + 1e-9)
    df["BB_Lower_ratio"] = df["Close"] / (lower + 1e-9)
    df["BB_Width"] = (upper - lower) / (sma + 1e-9)
    return df


def add_returns_volatility(df):
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["Volatility_10"] = df["Return_1d"].rolling(10).std()
    df["Volatility_20"] = df["Return_1d"].rolling(20).std()
    return df


def add_volume_features(df):
    df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_SMA_20"] + 1e-9)
    return df


def add_lag_returns(df, lags=(1, 2, 3, 5, 10)):
    for lag in lags:
        df[f"Return_Lag_{lag}"] = df["Return_1d"].shift(lag)
    return df


# ---------------- PIPELINE ----------------
def build_features(df):
    df = df.copy()
    df = add_moving_averages(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_returns_volatility(df)
    df = add_volume_features(df)
    df = add_lag_returns(df)
    return df


def clean_features(df):
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    print(f"🧹 Feature cleaning: {before} → {len(df)} rows")
    return df


def save_features(df, name):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filepath = os.path.join(PROCESSED_DIR, f"{name}_features.csv")
    df.to_csv(filepath, index=False)
    print(f"💾 Saved → {filepath}")
    return filepath


# ---------------- MAIN ----------------
def main():
    for split in ["train", "test"]:
        path = os.path.join(PROCESSED_DIR, f"{split}.csv")
        df = pd.read_csv(path, parse_dates=["Date"])
        df = build_features(df)
        df = clean_features(df)
        save_features(df, split)

    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train_features.csv"))
    print("\n" + "=" * 55)
    print("📊 FEATURE SUMMARY (STATIONARY MODE)")
    print("=" * 55)
    print(f"Total columns : {train.shape[1]}")
    print("=" * 55)


if __name__ == "__main__":
    main()