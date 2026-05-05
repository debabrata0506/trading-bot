from flask import Flask, render_template_string, send_file
import requests, pandas as pd, time, os
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

app = Flask(__name__)

# ================= COINS =================
PAIRS = [
    "B-BTC_USDT",
    "B-ETH_USDT",
    "B-SUI_USDT",
    "B-DOGE_USDT"
]

# ================= MODELS =================
model1 = SGDClassifier(loss="log_loss")
model2 = LogisticRegression()
scaler = StandardScaler()

model_ready = False

signals = []
equity = []
balance = 1000

MODEL_FILE = "model.pkl"

# ================= TELEGRAM =================
TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg):
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TG_CHAT, "text": msg})
    except:
        pass

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
    global model_ready
    X = df[["ret","ema"]]
    y = df["y"]

    scaler.fit(X)
    X = scaler.transform(X)

    model1.partial_fit(X, y, classes=[0,1])
    model2.fit(X, y)

    model_ready = True

# ================= ENSEMBLE =================
def predict(df):
    if not model_ready:
        return "HOLD", 0.5

    X = df[["ret","ema"]].iloc[-1:]
    X = scaler.transform(X)

    p1 = model1.predict_proba(X)[0][1]
    p2 = model2.predict_proba(X)[0][1]

    p = (p1 + p2) / 2

    if p > 0.6:
        return "BUY", p
    elif p < 0.4:
        return "SELL", p
    else:
        return "HOLD", p

# ================= TRADE =================
def trade(signal):
    global balance

    if signal == "BUY":
        balance *= 1.01
    elif signal == "SELL":
        balance *= 0.99

    equity.append(round(balance,2))
    if len(equity) > 50:
        equity.pop(0)

# ================= SAVE =================
def save_model():
    joblib.dump((model1, model2, scaler), MODEL_FILE)

def load_model():
    global model1, model2, scaler, model_ready
    if os.path.exists(MODEL_FILE):
        model1, model2, scaler = joblib.load(MODEL_FILE)
        model_ready = True

# ================= BOT =================
def run():
    load_model()

    while True:
        for pair in PAIRS:
            try:
                df = get_data(pair)
                df = features(df)

                train(df)
                signal, prob = predict(df)

                price = df["close"].iloc[-1]

                if signal != "HOLD":
                    trade(signal)

                    send_telegram(
                        f"{signal} {pair} @ {round(price,4)} | prob:{round(prob,2)}"
                    )

                signals.append({
                    "pair":pair,
                    "signal":f"{signal} ({round(prob,2)})",
                    "price":round(price,4)
                })

                if len(signals)>10:
                    signals.pop(0)

            except Exception as e:
                print("Error:", e)

        save_model()
        time.sleep(90)   # 🔥 optimized for 512MB

# ================= CHART =================
@app.route("/chart")
def chart():
    if not equity:
        return "No data yet"

    plt.figure()
    plt.plot(equity)
    plt.title("Equity Curve")
    plt.savefig("chart.png")
    plt.close()

    return send_file("chart.png", mimetype="image/png")

# ================= UI =================
HTML = """
<h1>🤖 AI Trading Bot</h1>

<h2>Signals</h2>
{% for s in signals %}
<p>{{s.signal}} - {{s.pair}} @ {{s.price}}</p>
{% endfor %}

<h2>Balance</h2>
<p>{{balance}}</p>

<h2><a href="/chart">📊 View Equity Chart</a></h2>
"""

@app.route("/")
def home():
    return render_template_string(HTML, signals=signals, balance=balance)

# ================= RUN =================
if __name__ == "__main__":
    import threading
    threading.Thread(target=run, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
