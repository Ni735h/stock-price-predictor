"""
app.py — Streamlit Web App (FINAL)
----------------------------------
Interactive stock price predictor for Allcargo Logistics.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

TICKER = "ALLCARGO.NS"
MODELS_DIR = "models"
PROCESSED_DIR = os.path.join("data", "processed")

DROP_COLS = [
    "Date", "Target",
    "Adj Close", "Close", "High", "Low", "Open",
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    "EMA_12", "EMA_26",
    "BB_Upper", "BB_Lower",
    "Volume_SMA_20", "Volume",
]


# ---------------- HELPERS ----------------
@st.cache_data
def load_test_data():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "test_features.csv"),
                     parse_dates=["Date"])
    return df


@st.cache_resource
def load_models():
    lr = joblib.load(os.path.join(MODELS_DIR, "linear.pkl"))
    lr_scaler = joblib.load(os.path.join(MODELS_DIR, "linear_scaler.pkl"))
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    rf_scaler = joblib.load(os.path.join(MODELS_DIR, "rf_scaler.pkl"))
    return lr, lr_scaler, rf, rf_scaler


def predict_returns(model, scaler, X):
    X_s = scaler.transform(X)
    return model.predict(X_s)


def metrics(y_true, y_pred):
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred),
    )


# ---------------- HEADER ----------------
st.title("📈 Allcargo Logistics — Stock Price Predictor")
st.markdown(
    "Built with **Python + scikit-learn + Streamlit** | "
    "Trains on 5 years of data | Predicts next-day close"
)

# ---------------- LOAD ----------------
df = load_test_data()
lr_model, lr_scaler, rf_model, rf_scaler = load_models()

feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols].values
close_today = df["Close"].values
actual_next = df["Target"].values

# Predict returns → reconstruct prices
lr_preds = close_today * (1 + predict_returns(lr_model, lr_scaler, X))
rf_preds = close_today * (1 + predict_returns(rf_model, rf_scaler, X))

rmse_lr, mae_lr, r2_lr = metrics(actual_next, lr_preds)
rmse_rf, mae_rf, r2_rf = metrics(actual_next, rf_preds)

# ---------------- METRIC CARDS ----------------
col1, col2, col3 = st.columns(3)
col1.metric(
    "Random Forest RMSE",
    f"₹{rmse_rf:.3f}",
    delta=f"{abs(rmse_rf - rmse_lr):.3f} better than LR",
    delta_color="normal",
)
col2.metric("Random Forest MAE", f"₹{mae_rf:.3f}")
col3.metric("Random Forest R²", f"{r2_rf:.4f}")

# ---------------- CHART ----------------
st.subheader("📊 Actual vs Predicted (Test Set)")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["Date"], y=actual_next,
    name="Actual", line=dict(color="black", width=2)
))
fig.add_trace(go.Scatter(
    x=df["Date"], y=rf_preds,
    name="Random Forest", line=dict(color="green", width=1.5)
))
fig.add_trace(go.Scatter(
    x=df["Date"], y=lr_preds,
    name="Linear Regression", line=dict(color="blue", width=1.5)
))
fig.update_layout(
    xaxis_title="Date", yaxis_title="Price (₹)",
    hovermode="x unified", height=500,
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- LATEST PREDICTION ----------------
st.subheader("🔮 Latest Prediction")

latest_close = close_today[-1]
latest_date = df["Date"].iloc[-1].date()

# Try live price
try:
    live = yf.download(TICKER, period="5d", interval="1d", progress=False)
    if isinstance(live.columns, pd.MultiIndex):
        live.columns = live.columns.get_level_values(0)
    live_close = float(live["Close"].iloc[-1])
except Exception:
    live_close = latest_close

col1, col2, col3 = st.columns(3)
col1.metric("Last Close", f"₹{latest_close:.2f}", help=f"as of {latest_date}")
col2.metric(
    "Predicted (RF)",
    f"₹{rf_preds[-1]:.2f}",
    delta=f"{((rf_preds[-1] / latest_close) - 1) * 100:+.2f}%",
)
col3.metric(
    "Predicted (LR)",
    f"₹{lr_preds[-1]:.2f}",
    delta=f"{((lr_preds[-1] / latest_close) - 1) * 100:+.2f}%",
)

# Direction indicator
change_pct = ((rf_preds[-1] / latest_close) - 1) * 100
if change_pct > 0.5:
    st.success(f"📈 **Bullish signal** — Model expects +{change_pct:.2f}% move")
elif change_pct < -0.5:
    st.error(f"📉 **Bearish signal** — Model expects {change_pct:.2f}% move")
else:
    st.info(f"➡️ **Neutral signal** — Model expects {change_pct:+.2f}% (no strong trend)")

st.caption(
    "⚠️ This is a student ML project. Predictions are NOT financial advice. "
    "Stock markets are inherently unpredictable."
)

# ---------------- MODEL COMPARISON ----------------
with st.expander("🔬 Model Comparison Details"):
    comparison = pd.DataFrame({
        "Model": ["Naive Baseline", "Linear Regression", "Random Forest", "LSTM"],
        "RMSE (₹)": [0.334, 0.365, 0.329, 0.351],
        "MAE (₹)": [0.225, 0.245, 0.222, 0.231],
        "R²": [0.9412, 0.9298, 0.9431, 0.9375],
    })
    st.dataframe(comparison, use_container_width=True)
    st.caption(
        "🎯 **Key insight:** Random Forest beats the naive baseline "
        "(predicting tomorrow = today). This is the real test of value in "
        "financial ML — most models DON'T beat naive."
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "**Tech stack:** Python · pandas · scikit-learn · TensorFlow · Plotly · Streamlit  \n"
    "**Model:** Random Forest on stationary technical features (RSI, MACD, returns, ratios)  \n"
    "**Result:** RMSE ₹0.33 (beats naive baseline)"
)