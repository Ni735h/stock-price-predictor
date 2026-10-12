"""
train_model.py — v4 (RETURN-BASED)
----------------------------------
Predicts next-day RETURN (relative change), not absolute price.
This solves the regime-shift issue where test prices are very different
from train prices.

Reconstruction: Predicted_Price = Close * (1 + Predicted_Return)
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
REPORTS_DIR = "reports"

# Columns we do NOT use as features
DROP_COLS = [
    "Date", "Target",
    "Adj Close", "Close", "High", "Low", "Open",
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    "EMA_12", "EMA_26",
    "BB_Upper", "BB_Lower",
    "Volume_SMA_20", "Volume",
]


# ---------------- HELPERS ----------------
def load_features(split):
    return pd.read_csv(os.path.join(PROCESSED_DIR, f"{split}_features.csv"),
                       parse_dates=["Date"])


def prepare_xy(df):
    """
    Build X (features) and y (next-day RETURN).
    Return = (Target - Close) / Close
    """
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].values
    y_return = ((df["Target"] - df["Close"]) / df["Close"]).values
    close_today = df["Close"].values
    actual_next = df["Target"].values
    return X, y_return, close_today, actual_next, feature_cols


def evaluate_price(name, actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    print(f"\n📊 {name}")
    print(f"   RMSE : ₹{rmse:.3f}")
    print(f"   MAE  : ₹{mae:.3f}")
    print(f"   R²   : {r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


# ---------------- BASELINE ----------------
def baseline_naive(actual_next, close_today):
    """Predict tomorrow = today's Close."""
    return evaluate_price("Naive (yesterday = next)", actual_next, close_today)


# ---------------- MODELS ----------------
def train_linear(X_train, y_train, X_test, y_test, close_test, actual_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(Xtr, y_train)
    pred_returns = model.predict(Xte)

    # Reconstruct prices
    pred_prices = close_test * (1 + pred_returns)
    metrics = evaluate_price("Linear Regression", actual_test, pred_prices)

    joblib.dump(model, os.path.join(MODELS_DIR, "linear.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "linear_scaler.pkl"))
    return model, pred_prices, metrics


def train_rf(X_train, y_train, X_test, y_test, close_test, actual_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    model.fit(Xtr, y_train)
    pred_returns = model.predict(Xte)
    pred_prices = close_test * (1 + pred_returns)

    metrics = evaluate_price("Random Forest", actual_test, pred_prices)

    joblib.dump(model, os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "rf_scaler.pkl"))
    return model, pred_prices, metrics


def train_lstm(X_train, y_train, X_test, y_test, close_test, actual_test, window=30):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        print("⚠️  TensorFlow not installed. Skipping LSTM.")
        return None, None, None

    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    def make_seq(X, y, w):
        Xs, ys = [], []
        for i in range(w, len(X)):
            Xs.append(X[i - w:i]); ys.append(y[i])
        return np.array(Xs), np.array(ys)

    Xtr, ytr = make_seq(X_train_s, y_train_s, window)
    Xte, yte = make_seq(X_test_s, y_test_s, window)

    model = Sequential([
        Input(shape=(window, X_train.shape[1])),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(Xtr, ytr, epochs=50, batch_size=32, validation_split=0.1,
              callbacks=[es], verbose=0)

    pred_returns_s = model.predict(Xte, verbose=0).ravel()
    pred_returns = scaler_y.inverse_transform(pred_returns_s.reshape(-1, 1)).ravel()

    # Reconstruct prices
    close_aligned = close_test[window:]
    actual_aligned = actual_test[window:]
    pred_prices = close_aligned * (1 + pred_returns)

    metrics = evaluate_price("LSTM", actual_aligned, pred_prices)

    model.save(os.path.join(MODELS_DIR, "lstm.keras"))
    joblib.dump(scaler_X, os.path.join(MODELS_DIR, "lstm_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(MODELS_DIR, "lstm_scaler_y.pkl"))
    return model, pred_prices, metrics


# ---------------- PLOT ----------------
def plot_predictions(actual, preds_dict, dates):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    plt.figure(figsize=(14, 6))
    plt.plot(dates, actual, label="Actual", color="black", linewidth=2)
    colors = {"Linear Regression": "blue", "Random Forest": "green", "LSTM": "red"}
    for name, preds in preds_dict.items():
        if preds is None:
            continue
        n = min(len(preds), len(dates))
        plt.plot(dates[-n:], preds, label=name,
                 color=colors.get(name, "gray"), alpha=0.8)
    plt.title("Allcargo Logistics — Actual vs Predicted Close (Return-based)")
    plt.xlabel("Date"); plt.ylabel("Price (₹)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "predictions.png"), dpi=120)
    print(f"\n🖼️  Saved → {REPORTS_DIR}/predictions.png")
    plt.close()


# ---------------- MAIN ----------------
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("📚 Loading features...")
    train_df = load_features("train")
    test_df = load_features("test")

    X_train, y_train_r, _, _, feature_cols = prepare_xy(train_df)
    X_test, y_test_r, close_test, actual_test, _ = prepare_xy(test_df)

    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"   Features ({len(feature_cols)}): {feature_cols}")
    print(f"   Train target (return) range: {y_train_r.min():.4f} → {y_train_r.max():.4f}")
    print(f"   Test  target (return) range: {y_test_r.min():.4f} → {y_test_r.max():.4f}\n")

    # Baseline
    baseline_naive(actual_test, close_test)

    # Models
    _, preds_lr, m_lr = train_linear(X_train, y_train_r, X_test, y_test_r,
                                     close_test, actual_test)
    _, preds_rf, m_rf = train_rf(X_train, y_train_r, X_test, y_test_r,
                                 close_test, actual_test)
    _, preds_lstm, m_lstm = train_lstm(X_train, y_train_r, X_test, y_test_r,
                                       close_test, actual_test)

    print("\n" + "=" * 55)
    print("🏆 MODEL COMPARISON")
    print("=" * 55)
    results = [m for m in [m_lr, m_rf, m_lstm] if m]
    results_df = pd.DataFrame(results).sort_values("rmse")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]["model"]
    print(f"\n🥇 Best model: {best}")
    results_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

    plot_predictions(actual_test, {
        "Linear Regression": preds_lr,
        "Random Forest": preds_rf,
        "LSTM": preds_lstm,
    }, test_df["Date"].values)


if __name__ == "__main__":
    main()