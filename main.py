import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
import threading

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# =============================
# CONFIG WEB
# =============================
st.set_page_config(page_title="AI Bitcoin Trading", layout="wide")
st.title("📈 Hệ Thống Dự Đoán Bitcoin Bằng AI")

# =============================
# TELEGRAM (FAST)
# =============================
def send_async(token, chat_id, text):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={
            "chat_id": chat_id,
            "text": text
        }, timeout=3)
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

st.subheader("📊 Dữ liệu Bitcoin gần nhất")
st.dataframe(data.tail(10).style.format("{:.2f}"), use_container_width=True)

# =============================
# INDICATORS
# =============================
data["MA50"] = data["Close"].rolling(50).mean()
data["MA200"] = data["Close"].rolling(200).mean()

delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

# =============================
# AI DATA
# =============================
dataset = data["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

training_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:training_len]

X, y = [], []

for i in range(60, len(train_data)):
    X.append(train_data[i-60:i])
    y.append(train_data[i])

X, y = np.array(X), np.array(y)

# =============================
# MODEL (NO RETRAIN)
# =============================
@st.cache_resource
def train_model(X, y):

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(LSTM(64))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(X, y, epochs=15, batch_size=32, verbose=0)

    return model

model = train_model(X,y)

# =============================
# TEST
# =============================
test_data = scaled_data[training_len - 60:]

X_test = []
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i])

X_test = np.array(X_test)

predictions = model.predict(X_test, verbose=0)
predictions = scaler.inverse_transform(predictions)

real_values = data["Close"][training_len:].values[:len(predictions)]

# =============================
# ACCURACY
# =============================
mape = np.mean(np.abs((real_values - predictions.flatten()) / real_values)) * 100
accuracy = 100 - mape

# =============================
# FUTURE
# =============================
last_60 = scaled_data[-60:]
future = []

for i in range(30):
    x = last_60.reshape(1,60,1)
    pred = model.predict(x, verbose=0)
    future.append(pred[0][0])
    last_60 = np.append(last_60[1:], pred)

future_prices = scaler.inverse_transform(np.array(future).reshape(-1,1))

# =============================
# SIGNAL
# =============================
last_price = float(data["Close"].iloc[-1])
pred_price = float(future_prices[0][0])
profit_percent = ((pred_price - last_price) / last_price) * 100
last_rsi = float(data["RSI"].iloc[-1])

if profit_percent > 2 and last_rsi < 70:
    signal = "BUY"
elif profit_percent < -2 and last_rsi > 30:
    signal = "SELL"
else:
    signal = "HOLD"

# =============================
# DASHBOARD
# =============================
st.subheader("🤖 Kết quả AI")

col1,col2,col3,col4,col5 = st.columns(5)
col1.metric("Giá hiện tại", f"${last_price:,.0f}")
col2.metric("Dự đoán", f"${pred_price:,.0f}")
col3.metric("RSI", f"{last_rsi:.2f}")
col4.metric("Accuracy", f"{accuracy:.2f}%")
col5.metric("Profit %", f"{profit_percent:.2f}%")

# =============================
# SIGNAL UI
# =============================
st.subheader("📢 Tín hiệu giao dịch")

if signal == "BUY":
    st.success(f"🟢 BUY (+{profit_percent:.2f}%)")
elif signal == "SELL":
    st.error(f"🔴 SELL ({profit_percent:.2f}%)")
else:
    st.warning(f"🟡 HOLD ({profit_percent:.2f}%)")

# =============================
# TELEGRAM
# =============================
st.sidebar.header("📲 Telegram")

tele_token = st.sidebar.text_input("Bot Token", type="password")
tele_chat_id = st.sidebar.text_input("Chat ID")

msg = f"""
🚀 AI Bitcoin Signal

Signal: {signal}
Price: ${last_price:,.2f}
Prediction: ${pred_price:,.2f}
Change: {profit_percent:.2f}%

RSI: {last_rsi:.2f}
Accuracy: {accuracy:.2f}%
"""

if st.sidebar.button("🚀 Gửi ngay"):
    if tele_token and tele_chat_id:

        threading.Thread(
            target=send_async,
            args=(tele_token, tele_chat_id, msg)
        ).start()

        st.sidebar.success("✅ Đã gửi ngay!")

    else:
        st.sidebar.warning("Nhập Token + Chat ID")

# =============================
# CHART
# =============================
st.subheader("📈 Biểu đồ giá")
fig = plt.figure(figsize=(12,6))
plt.plot(data["Close"], label="Price")
plt.plot(data["MA50"], label="MA50")
plt.plot(data["MA200"], label="MA200")
plt.legend()
st.pyplot(fig)
