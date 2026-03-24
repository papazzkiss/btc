import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
import threading

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# =============================
# TELEGRAM (ASYNC)
# =============================

def send_telegram_async(token, chat_id, text):
    def send():
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": text},
                timeout=3
            )
        except:
            pass
    threading.Thread(target=send).start()


# =============================
# CONFIG
# =============================

st.set_page_config(page_title="AI Bitcoin Trading", layout="wide")
st.title("📈 Hệ Thống Dự Đoán Bitcoin Bằng AI")


# =============================
# LOAD DATA
# =============================

@st.cache_data
def load_data():
    data = yf.download("BTC-USD", start="2022-01-01")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.dropna(inplace=True)
    return data

data = load_data()


# =============================
# DATA TABLE
# =============================

st.subheader("📊 Dữ liệu Bitcoin")
rows = st.slider("Số dòng hiển thị",5,100,10)
st.dataframe(data.tail(rows),use_container_width=True)


# =============================
# TECHNICAL INDICATORS
# =============================

filtered_data = data.copy()

filtered_data["MA50"] = filtered_data["Close"].rolling(50).mean()
filtered_data["MA200"] = filtered_data["Close"].rolling(200).mean()

delta = filtered_data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
filtered_data["RSI"] = 100 - (100 / (1 + rs))

exp1 = filtered_data["Close"].ewm(span=12, adjust=False).mean()
exp2 = filtered_data["Close"].ewm(span=26, adjust=False).mean()

filtered_data["MACD"] = exp1 - exp2
filtered_data["Signal_MACD"] = filtered_data["MACD"].ewm(span=9, adjust=False).mean()

filtered_data.dropna(inplace=True)


# =============================
# FEATURES
# =============================

features = filtered_data[
["Close","MA50","MA200","RSI","MACD","Signal_MACD"]
]

dataset = features.values


# =============================
# SCALE
# =============================

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

training_len = int(len(scaled_data)*0.8)
train_data = scaled_data[:training_len]


# =============================
# SEQUENCE
# =============================

window = 90
X=[]
y=[]

for i in range(window,len(train_data)):
    X.append(train_data[i-window:i])
    y.append(train_data[i,0])

X=np.array(X)
y=np.array(y)


# =============================
# TRAIN LSTM MODEL
# =============================

def train_model(X,y):
    model=Sequential()
    model.add(LSTM(128,return_sequences=True,input_shape=(X.shape[1],X.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dense(1))

    model.compile(optimizer="adam",loss="mse")
    model.fit(X,y,epochs=5,batch_size=32,verbose=0)

    return model

model=train_model(X,y)


# =============================
# TEST LSTM
# =============================

test_data=scaled_data[training_len-window:]

X_test=[]
for i in range(window,len(test_data)):
    X_test.append(test_data[i-window:i])

X_test=np.array(X_test)

predictions=model.predict(X_test,verbose=0)

pred_close=[]
for i in range(len(predictions)):
    row=test_data[window+i].copy()
    row[0]=predictions[i][0]
    inv=scaler.inverse_transform([row])
    pred_close.append(inv[0][0])

pred_close=np.array(pred_close)
real_values=filtered_data["Close"][training_len:].values[:len(pred_close)]


# =============================
# RANDOM FOREST
# =============================

rf_data = filtered_data[["Close"]].copy()

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

rf_model = RandomForestRegressor(n_estimators=200,max_depth=10,random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

rf_pred = rf_model.predict(X_test_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test_rf, rf_pred))


# =============================
# ACCURACY
# =============================

recent=150

real_recent=real_values[-recent:]
pred_recent=pred_close[-recent:]

mape=np.mean(np.abs((real_recent-pred_recent)/real_recent))*100
accuracy=100-mape

rf_recent = min(150, len(rf_pred))
rf_real_recent = y_test_rf.values[-rf_recent:]
rf_pred_recent = rf_pred[-rf_recent:]

rf_mape = np.mean(np.abs((rf_real_recent-rf_pred_recent)/rf_real_recent))*100
rf_accuracy = 100-rf_mape


# =============================
# SIGNAL LOGIC
# =============================

last_price=float(filtered_data["Close"].iloc[-1])
pred_price=float(pred_close[-1])
last_rsi=float(filtered_data["RSI"].iloc[-1])

profit_percent = ((pred_price - last_price) / last_price) * 100

if profit_percent > 2 and last_rsi < 65:
    signal = "BUY"
elif profit_percent < -2 and last_rsi > 35:
    signal = "SELL"
else:
    signal = "HOLD"


# =============================
# DASHBOARD
# =============================

# =============================
# AI DASHBOARD
# =============================
st.subheader("🤖 AI Dashboard")

col1, col2, col3, col4, col5, col6 = st.columns(6)

last_price = float(filtered_data["Close"].iloc[-1])
pred_price = float(pred_close[-1])
last_rsi = float(filtered_data["RSI"].iloc[-1])

col1.metric("Giá hiện tại", f"${last_price:,.0f}")
col2.metric("LSTM Predict", f"${pred_price:,.0f}")
col3.metric("RSI", f"{last_rsi:.2f}")
col4.metric("LSTM Accuracy", f"{accuracy:.2f}%")
col5.metric("RF RMSE", f"{rf_rmse:.0f}")
col6.metric("RF Accuracy", f"{rf_accuracy:.2f}%")

# =============================
# PRICE CHART
# =============================
st.subheader("📈 Price Chart")

fig = plt.figure(figsize=(12, 6))

plt.plot(filtered_data["Close"], label="Price")
plt.plot(filtered_data["MA50"], label="MA50")
plt.plot(filtered_data["MA200"], label="MA200")

plt.legend()
st.pyplot(fig)

# =============================
# RSI CHART
# =============================
st.subheader("📉 RSI Indicator")

fig_rsi = plt.figure(figsize=(12, 4))

plt.plot(filtered_data["RSI"], label="RSI")

plt.axhline(70, linestyle="--")
plt.axhline(30, linestyle="--")

plt.legend()
st.pyplot(fig_rsi)

# =============================
# MACD CHART
# =============================
st.subheader("📊 MACD Indicator")

fig_macd = plt.figure(figsize=(12, 4))

plt.plot(filtered_data["MACD"], label="MACD")
plt.plot(filtered_data["Signal_MACD"], label="Signal")

plt.legend()
st.pyplot(fig_macd)

# =============================
# AI MODEL COMPARISON
# =============================
st.subheader("🤖 AI Model Comparison")

fig_compare = plt.figure(figsize=(12, 6))

plt.plot(real_values, label="Real Price")
plt.plot(pred_close, label="LSTM Prediction")

# Random Forest prediction alignment
plt.plot(
    range(len(real_values) - len(rf_pred), len(real_values)),
    rf_pred,
    label="Random Forest"
)

plt.legend()
st.pyplot(fig_compare)

# =============================
# MODEL PERFORMANCE TABLE
# =============================
st.subheader("🏆 AI Model Performance")

compare_df = pd.DataFrame({
    "Model": ["LSTM", "Random Forest"],
    "Accuracy": [accuracy, rf_accuracy]
})

st.dataframe(compare_df, use_container_width=True)

if signal == "BUY":
    col7.metric("Signal","🟢 BUY")
elif signal == "SELL":
    col7.metric("Signal","🔴 SELL")
else:
    col7.metric("Signal","🟡 HOLD")


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
# TELEGRAM UI
# =============================

st.sidebar.header("Telegram Bot")

tele_token=st.sidebar.text_input("Bot Token",type="password")
tele_chat_id=st.sidebar.text_input("Chat ID")

msg=f"""
🚀 AI Bitcoin Signal

📍 Signal: {signal}
💰 Price: ${last_price:,.2f}
🔮 Prediction: ${pred_price:,.2f}
📈 Change: {profit_percent:.2f}%

📊 RSI: {last_rsi:.2f}

🎯 LSTM: {accuracy:.2f}%
🤖 RF: {rf_accuracy:.2f}%
"""

if st.sidebar.button("Send Signal"):
    if tele_token and tele_chat_id:
        send_telegram_async(tele_token, tele_chat_id, msg)
        st.sidebar.success("✅ Gửi ngay")
    else:
        st.sidebar.warning("Nhập Token và Chat ID")
