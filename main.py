import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# =============================
# CONFIG WEB
# =============================
st.set_page_config(page_title="AI Bitcoin Trading", layout="wide")

st.title("📈 Hệ Thống Dự Đoán Bitcoin Bằng AI")

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

table = data.tail(10)
styled_table = table.style.format("{:.2f}")
st.dataframe(styled_table, use_container_width=True)

# =============================
# TECHNICAL INDICATORS
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

# Bollinger
data["MA20"] = data["Close"].rolling(20).mean()
data["STD"] = data["Close"].rolling(20).std()

data["Upper"] = data["MA20"] + 2 * data["STD"]
data["Lower"] = data["MA20"] - 2 * data["STD"]

# =============================
# AI DATA PREPARE
# =============================

dataset = data["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

training_len = int(len(scaled_data) * 0.8)

train_data = scaled_data[:training_len]

X = []
y = []

for i in range(60, len(train_data)):
    X.append(train_data[i-60:i])
    y.append(train_data[i])

X = np.array(X)
y = np.array(y)

# =============================
# TRAIN MODEL (CACHE)
# =============================

@st.cache_resource
def train_model(X, y):

    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(LSTM(64))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    return model

model = train_model(X,y)

# =============================
# TEST DATA
# =============================

test_data = scaled_data[training_len - 60:]

X_test = []

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i])

X_test = np.array(X_test)

predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)

real_price = data["Close"][training_len:]

# =============================
# AI ACCURACY
# =============================

real_values = real_price.values[:len(predictions)]

mape = np.mean(np.abs((real_values - predictions.flatten()) / real_values)) * 100
accuracy = 100 - mape

# =============================
# FUTURE PREDICTION
# =============================

future_days = 30

last_60 = scaled_data[-60:]

future = []

for i in range(future_days):

    x = last_60.reshape(1,60,1)

    pred = model.predict(x, verbose=0)

    future.append(pred[0][0])

    last_60 = np.append(last_60[1:], pred)

future_prices = scaler.inverse_transform(
    np.array(future).reshape(-1,1)
)

# =============================
# TRADING SIGNAL
# =============================

last_price = float(data["Close"].iloc[-1])
pred_price = float(future_prices[0][0])

change = (pred_price - last_price) / last_price
profit_percent = change * 100

last_rsi = float(data["RSI"].iloc[-1])

signal = "HOLD"

if change > 0.02 and last_rsi < 70:
    signal = "BUY"

elif change < -0.02 and last_rsi > 30:
    signal = "SELL"

# =============================
# DASHBOARD
# =============================

st.subheader("🤖 Kết quả AI")

col1,col2,col3,col4,col5 = st.columns(5)

col1.metric("Giá hiện tại", f"${last_price:,.0f}")
col2.metric("Dự đoán ngày mai", f"${pred_price:,.0f}")
col3.metric("RSI", f"{last_rsi:.2f}")
col4.metric("AI Accuracy", f"{accuracy:.2f}%")
col5.metric("Profit %", f"{profit_percent:.2f}%")

# =============================
# SIGNAL
# =============================

st.subheader("📢 Tín hiệu giao dịch")

if signal == "BUY":
    st.success("🟢 BUY")

elif signal == "SELL":
    st.error("🔴 SELL")

else:
    st.warning("🟡 HOLD")

# =============================
# INVESTMENT SIMULATION
# =============================

st.subheader("💰 Mô phỏng đầu tư")

investment = st.slider(
    "Số tiền đầu tư ($)",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100
)

profit_money = investment * profit_percent / 100

st.write(f"Nếu đầu tư {investment}$")
st.write(f"Lợi nhuận dự kiến: {profit_money:.2f}$")

# =============================
# PRICE CHART
# =============================

st.subheader("📈 Biểu đồ giá Bitcoin")

fig = plt.figure(figsize=(12,6))

plt.plot(data["Close"], label="Price")
plt.plot(data["MA50"], label="MA50")
plt.plot(data["MA200"], label="MA200")
plt.plot(data["Upper"], label="Bollinger Upper")
plt.plot(data["Lower"], label="Bollinger Lower")

plt.legend()

st.pyplot(fig)

# =============================
# AI VS REAL
# =============================

st.subheader("📊 So sánh AI vs Giá thật")

fig2 = plt.figure(figsize=(12,6))

plt.plot(real_values, label="Real Price")
plt.plot(predictions.flatten(), label="AI Prediction")

plt.legend()

st.pyplot(fig2)

# =============================
# RSI
# =============================

st.subheader("📊 Chỉ báo RSI")

fig3 = plt.figure(figsize=(12,4))

plt.plot(data["RSI"], label="RSI")

plt.axhline(70)
plt.axhline(30)

plt.legend()

st.pyplot(fig3)

# =============================
# MACD
# =============================

st.subheader("📊 Chỉ báo MACD")

fig4 = plt.figure(figsize=(12,4))

plt.plot(data["MACD"], label="MACD")
plt.plot(data["Signal_MACD"], label="Signal")

plt.legend()

st.pyplot(fig4)

# =============================
# FORECAST
# =============================

st.subheader("🔮 Dự đoán 30 ngày")

future_dates = pd.date_range(data.index[-1], periods=future_days+1)[1:]

fig5 = plt.figure(figsize=(12,5))

plt.plot(future_dates, future_prices, label="Future Price")

plt.legend()

st.pyplot(fig5)

# =============================
# AI EXPLANATION
# =============================

st.subheader("🤖 Giải thích mô hình")

st.write("""
Mô hình sử dụng **LSTM (Long Short Term Memory)** để học xu hướng giá Bitcoin.

Dữ liệu đầu vào:  
60 ngày giá gần nhất

Kết hợp các chỉ báo:

• RSI  
• MACD  
• Moving Average  
• Bollinger Bands  

để đưa ra tín hiệu **BUY / SELL / HOLD**.
""")