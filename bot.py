from flask import Flask, render_template_string
import requests, pandas as pd, time, os
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

PAIRS = ["B-DOGE_USDT","B-MATIC_USDT","B-SHIB_USDT"]

model = SGDClassifier(loss="log_loss")
scaler = StandardScaler()

signals = []
equity = []
balance = 1000

# ================= DATA =================
def get_data(pair):
    url = "https://public.coindcx.com/market_data/candles"
    params = {"pair": pair, "interval":"5m","limit":100}
    df = pd.DataFrame(requests.get(url, params=params).json())
    df.columns = [c.lower() for c in df.columns]
    df["close"] = pd.to_numeric(df["close"])
    return df

# ================= FEATURE =================
def features(df):
    df["ret"] = df["close"].pct_change()
    df["ema"] = df["close"].ewm(span=10).mean()
    df["y"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)
    return df

# ================= TRAIN =================
def train(df):
    X = df[["ret","ema"]]
    y = df["y"]
    scaler.fit(X)
    X = scaler.transform(X)
    model.partial_fit(X,y,classes=[0,1])

# ================= SIGNAL =================
def predict(df):
    X = df[["ret","ema"]].iloc[-1:]
    X = scaler.transform(X)
    p = model.predict_proba(X)[0][1]

    if p > 0.6:
        return "BUY", p
    elif p < 0.4:
        return "SELL", p
    else:
        return "HOLD", p

# ================= TRADE =================
def trade(signal, price):
    global balance
    if signal == "BUY":
        balance *= 1.01
    elif signal == "SELL":
        balance *= 0.99

    equity.append(balance)
    if len(equity) > 50:
        equity.pop(0)

# ================= BOT =================
def run():
    while True:
        for pair in PAIRS:
            try:
                df = get_data(pair)
                df = features(df)
                train(df)

                signal, prob = predict(df)
                price = df["close"].iloc[-1]

                if signal != "HOLD":
                    trade(signal, price)

                signals.append({
                    "pair":pair,
                    "signal":signal,
                    "price":round(price,4)
                })

                if len(signals)>10:
                    signals.pop(0)

            except:
                pass

        time.sleep(60)

# ================= UI =================
HTML = """
<h1>🤖 AI Trading Bot</h1>

<h2>Signals</h2>
{% for s in signals %}
<p>{{s.signal}} - {{s.pair}} @ {{s.price}}</p>
{% endfor %}

<h2>Balance</h2>
<p>{{balance}}</p>

<h2>Equity</h2>
<p>{{equity}}</p>
"""

@app.route("/")
def home():
    return render_template_string(HTML, signals=signals, balance=balance, equity=equity)

import threading

threading.Thread(target=run, daemon=True).start()

app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
