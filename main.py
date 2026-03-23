import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="AI Bitcoin Trading PRO", layout="wide")
st.title("🚀 AI Bitcoin Trading PRO")

# =============================
# TELEGRAM
# =============================
def send_telegram_message(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    return requests.post(url, data=payload).json()

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
# BẢNG 1: DATA GỐC
# =============================
st.subheader("📊 Bảng 1: Dữ liệu gốc")
st.dataframe(data.tail(10), use_container_width=True)

# =============================
# INDICATORS
# =============================
data["MA50"] = data["Close"].rolling(50).mean()
data["MA200"] = data["Close"].rolling(200).mean()

# RSI
delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

rs = gain.rolling(14).mean() / loss.rolling(14).mean()
data["RSI"] = 100 - (100/(1+rs))

# MACD
exp1 = data["Close"].ewm(span=12).mean()
exp2 = data["Close"].ewm(span=26).mean()
data["MACD"] = exp1 - exp2
data["Signal"] = data["MACD"].ewm(span=9).mean()

data.dropna(inplace=True)

# =============================
# BẢNG 2: INDICATORS
# =============================
st.subheader("📊 Bảng 2: Indicators")
st.dataframe(data[["Close","MA50","MA200","RSI","MACD"]].tail(10), use_container_width=True)

# =============================
# LSTM
# =============================
dataset = data["Close"].values.reshape(-1,1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(dataset)

train_len = int(len(scaled)*0.8)

X,y=[],[]

for i in range(60,train_len):
    X.append(scaled[i-60:i])
    y.append(scaled[i])

X,y=np.array(X),np.array(y)

model=Sequential()
model.add(LSTM(64,return_sequences=True,input_shape=(60,1)))
model.add(LSTM(64))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")

model.fit(X,y,epochs=5,batch_size=32,verbose=0)

# TEST
test = scaled[train_len-60:]
X_test=[]

for i in range(60,len(test)):
    X_test.append(test[i-60:i])

X_test=np.array(X_test)

pred=model.predict(X_test)
pred=scaler.inverse_transform(pred)

real=data["Close"][train_len:].values[:len(pred)]

# =============================
# RANDOM FOREST
# =============================
rf = data[["Close"]].copy()
rf["lag1"]=rf["Close"].shift(1)
rf["lag2"]=rf["Close"].shift(2)
rf["lag3"]=rf["Close"].shift(3)
rf.dropna(inplace=True)

X_rf=rf[["lag1","lag2","lag3"]]
y_rf=rf["Close"]

split=int(len(rf)*0.8)

X_train_rf=X_rf[:split]
X_test_rf=X_rf[split:]
y_train_rf=y_rf[:split]
y_test_rf=y_rf[split:]

model_rf=RandomForestRegressor(n_estimators=100)
model_rf.fit(X_train_rf,y_train_rf)

rf_pred=model_rf.predict(X_test_rf)

# FIX LỖI Ở ĐÂY
y_test_rf = y_test_rf.values

rf_rmse=np.sqrt(mean_squared_error(y_test_rf,rf_pred))
rf_mape=np.mean(np.abs((y_test_rf-rf_pred)/y_test_rf))*100
rf_acc=100-rf_mape

# =============================
# LSTM ACC
# =============================
mape=np.mean(np.abs((real-pred.flatten())/real))*100
lstm_acc=100-mape

# =============================
# BẢNG 3: DỰ ĐOÁN
# =============================
df_pred=pd.DataFrame({
    "Real": real,
    "LSTM": pred.flatten()
})

st.subheader("📊 Bảng 3: LSTM Prediction")
st.dataframe(df_pred.tail(10), use_container_width=True)

# =============================
# BẢNG 4: RANDOM FOREST
# =============================
df_rf=pd.DataFrame({
    "Real": y_test_rf,
    "RF": rf_pred
})

st.subheader("📊 Bảng 4: Random Forest")
st.dataframe(df_rf.tail(10), use_container_width=True)

# =============================
# BẢNG 5: SO SÁNH MODEL
# =============================
compare=pd.DataFrame({
    "Model":["LSTM","Random Forest"],
    "Accuracy":[lstm_acc,rf_acc]
})

st.subheader("📊 Bảng 5: So sánh AI")
st.dataframe(compare, use_container_width=True)

# =============================
# SIGNAL
# =============================
last_price=float(data["Close"].iloc[-1])
pred_price=float(pred[-1])

change=(pred_price-last_price)/last_price
rsi=float(data["RSI"].iloc[-1])

if change>0.02 and rsi<70:
    signal="BUY"
elif change<-0.02 and rsi>30:
    signal="SELL"
else:
    signal="HOLD"

st.subheader("📢 Tín hiệu")

if signal=="BUY":
    st.success("🟢 BUY")
elif signal=="SELL":
    st.error("🔴 SELL")
else:
    st.warning("🟡 HOLD")

# =============================
# TELEGRAM
# =============================
st.sidebar.header("Telegram")

token=st.sidebar.text_input("Token")
chat_id=st.sidebar.text_input("Chat ID")

msg=f"""
AI SIGNAL

Signal: {signal}
Price: {last_price}
Pred: {pred_price}

LSTM: {lstm_acc:.2f}%
RF: {rf_acc:.2f}%
"""

if st.sidebar.button("Send"):
    if token and chat_id:
        send_telegram_message(token,chat_id,msg)
        st.sidebar.success("Sent")
