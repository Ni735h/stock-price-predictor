# 📈 Stock Price Predictor — Allcargo Logistics

An end-to-end ML project that predicts next-day closing price of **Allcargo Logistics (NSE: ALLCARGO)** using technical indicators and machine learning.

## 🎯 Key Results

| Model          | RMSE (₹) | MAE (₹) | R² |
|-------         |----------|---------|-----|
| Naive Baseline | 0.334 | 0.225 | 0.9412 |
| Linear Regression | 0.365 | 0.245 | 0.9298 |
| **Random Forest** 🥇 | **0.329** | **0.222** | **0.9431** |
| LSTM | 0.351 | 0.231 | 0.9375 |

**Random Forest beats the naive baseline** — the honest test of value in financial ML.

## 🛠️ Tech Stack

- **Data:** yfinance (5 years of daily data)
- **Features:** RSI, MACD, Bollinger Bands, SMA/EMA ratios, returns, volatility
- **Models:** Linear Regression, Random Forest, LSTM (Keras)
- **UI:** Streamlit + Plotly
- **Deployment:** Streamlit Cloud

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/YOUR_USERNAME/stock-price-predictor.git
cd stock-price-predictor
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# 2. Install deps
pip install -r requirements.txt

# 3. Run pipeline
python -m src.fetch_data
python -m src.preprocess
python -m src.features
python -m src.train_model

# 4. Launch app
streamlit run app.py


## 📁 Structure

```
stock-price-predictor/
├── data/           # raw & processed CSVs
├── models/         # trained models
├── src/            # pipeline scripts
├── app.py          # Streamlit app
└── requirements.txt
```

## 🧠 Key Learnings

1. **Feature scaling matters** — unscaled Volume destroyed Linear Regression (R² = -3058)
2. **Regime shift is real** — models trained on bullish data fail on bearish test sets
3. **Predict returns, not prices** — dropped RMSE from ₹76 → ₹0.33
4. **Always compare to naive baseline** — the only honest test in financial ML

## ⚠️ Disclaimer

Educational project only. Not financial advice.

## 🙏 Acknowledgements

Built as part of a learning journey with **Codec Technologies**.

## 📬 Connect

[LinkedIn](https://linkedin.com/in/nitesh-kanojiya-89622636b) 
