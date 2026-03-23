import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
import threading

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="AI Bitcoin PRO", layout="wide")
st.title("🚀 AI Bitcoin Trading PRO")

# =============================
# TELEGRAM FAST
# =============================
def send_async(token, chat_id, text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=3
        )
    except:
        pass

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    data = yf.download("BTC-USD", start="2018-01-01")
    data.dropna(inplace=True)
    return data

data = load_data()

# =============================
# INDICATORS
# =============================
data["MA50"] = data["Close"].rolling(50).mean()
data["MA200"] = data["Close"].rolling(200).mean()

# RSI
delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

# MACD
exp1 = data["Close"].ewm(span=12, adjust=False).mean()
exp2 = data["Close"].ewm(span=26, adjust=False).mean()

data["MACD"] = exp1 - exp2
data["Signal_MACD"] = data["MACD"].ewm(span=9, adjust=False).mean()

data.dropna(inplace=True)

# =============================
# DATA TABLE
# =============================
st.subheader("📊 Data")
st.dataframe(data.tail(10), use_container_width=True)

# =============================
# FEATURES (PRO)
# =============================
features = data[["Close","RSI","MACD","Signal_MACD"]]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features)

training_len = int(len(scaled_data)*0.8)
train_data = scaled_data[:training_len]

X, y = [], []

for i in range(60, len(train_data)):
    X.append(train_data[i-60:i])
    y.append(train_data[i,0])

X, y = np.array(X), np.array(y)

# =============================
# LSTM MODEL PRO
# =============================
@st.cache_resource
def train_model(X,y):

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(64))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    model.fit(X,y,epochs=15,batch_size=32,verbose=0)

    return model

model = train_model(X,y)

# =============================
# TEST
# =============================
test_data = scaled_data[training_len-60:]

X_test = []

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i])

X_test = np.array(X_test)

predictions = model.predict(X_test, verbose=0)

# inverse
pred_close = []

for i in range(len(predictions)):
    row = test_data[60+i].copy()
    row[0] = predictions[i][0]
    inv = scaler.inverse_transform([row])
    pred_close.append(inv[0][0])

pred_close = np.array(pred_close)

real_values = data["Close"][training_len:].values[:len(pred_close)]

# =============================
# ACCURACY + CONFIDENCE
# =============================
mape = np.mean(np.abs((real_values - pred_close)/real_values))*100
accuracy = 100 - mape
confidence = accuracy / 100

# =============================
# RANDOM FOREST
# =============================
rf_data = data[["Close"]].copy()

rf_data["lag1"] = rf_data["Close"].shift(1)
rf_data["lag2"] = rf_data["Close"].shift(2)
rf_data["lag3"] = rf_data["Close"].shift(3)

rf_data.dropna(inplace=True)

X_rf = rf_data[["lag1","lag2","lag3"]]
y_rf = rf_data["Close"]

split = int(len(rf_data)*0.8)

X_train_rf = X_rf[:split]
X_test_rf = X_rf[split:]

y_train_rf = y_rf[:split]
y_test_rf = y_rf[split:]

@st.cache_resource
def train_rf(X,y):
    model = RandomForestRegressor(n_estimators=100,max_depth=10)
    model.fit(X,y)
    return model

rf_model = train_rf(X_train_rf,y_train_rf)

rf_pred = rf_model.predict(X_test_rf)

rf_mape = np.mean(np.abs((y_test_rf-rf_pred)/y_test_rf))*100
rf_accuracy = 100 - rf_mape

# =============================
# SIGNAL PRO
# =============================
last_price = float(data["Close"].iloc[-1])
pred_price = float(pred_close[-1])
last_rsi = float(data["RSI"].iloc[-1])

change = (pred_price - last_price) / last_price

signal = "HOLD"

if change > 0.02 and last_rsi < 65:
    signal = "BUY"
elif change < -0.02 and last_rsi > 35:
    signal = "SELL"

# =============================
# DASHBOARD PRO
# =============================
st.subheader("🤖 AI Dashboard PRO")

c1,c2,c3,c4,c5 = st.columns(5)

c1.metric("Price", f"${last_price:,.0f}")
c2.metric("Prediction", f"${pred_price:,.0f}")
c3.metric("RSI", f"{last_rsi:.2f}")
c4.metric("Accuracy", f"{accuracy:.2f}%")
c5.metric("Confidence", f"{confidence:.2f}")

# =============================
# SIGNAL UI
# =============================
st.subheader("📢 Trading Signal")

if signal == "BUY":
    st.success("🟢 STRONG BUY")
elif signal == "SELL":
    st.error("🔴 STRONG SELL")
else:
    st.warning("🟡 WAIT")

# =============================
# MODEL COMPARISON
# =============================
st.subheader("🏆 Model Comparison")

compare_df = pd.DataFrame({
    "Model": ["LSTM PRO", "Random Forest"],
    "Accuracy": [round(accuracy,2), round(rf_accuracy,2)]
})

st.dataframe(compare_df, use_container_width=True)

# =============================
# CHARTS PRO
# =============================
st.subheader("📈 Price + AI")

fig = plt.figure(figsize=(12,6))
plt.plot(data["Close"], label="Price")
plt.plot(data["MA50"], label="MA50")
plt.plot(data["MA200"], label="MA200")
plt.legend()
st.pyplot(fig)

st.subheader("📊 AI vs Real")

fig2 = plt.figure(figsize=(12,6))
plt.plot(real_values, label="Real")
plt.plot(pred_close, label="LSTM")
plt.legend()
st.pyplot(fig2)

# =============================
# TELEGRAM
# =============================
st.sidebar.header("📲 Telegram")

token = st.sidebar.text_input("Token", type="password")
chat_id = st.sidebar.text_input("Chat ID")

msg = f"""
🚀 AI PRO SIGNAL

Signal: {signal}
Price: ${last_price:,.2f}
Predict: ${pred_price:,.2f}

RSI: {last_rsi:.2f}
Acc: {accuracy:.2f}%
"""

if st.sidebar.button("Send"):
    if token and chat_id:
        threading.Thread(target=send_async, args=(token, chat_id, msg)).start()
        st.sidebar.success("Sent instantly 🚀")
    else:
        st.sidebar.warning("Missing info")
