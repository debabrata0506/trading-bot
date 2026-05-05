from flask import Flask, render_template_string, send_file
import requests, pandas as pd, time, os
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

app = Flask(__name__)

PAIRS = ["B-BTC_USDT","B-ETH_USDT","B-SUI_USDT","B-DOGE_USDT"]

model1 = SGDClassifier(loss="log_loss")
model2 = LogisticRegression(max_iter=100)
scaler = StandardScaler()

model_ready = False

signals = []
equity = []
balance = 1000

wins = 0
losses = 0

MODEL_FILE = "model.pkl"
TRADE_LOG = "trade_log.csv"
last_sent = {}

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
    params = {"pair": pair, "interval":"5m","limit":80}
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

    if not model_ready:
        scaler.fit(X)

    X = scaler.transform(X)

    model1.partial_fit(X, y, classes=[0,1])
    model2.fit(X, y)

    model_ready = True

# ================= PREDICT =================
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

# ================= SIMULATION =================
def simulate(signal):
    global balance, wins, losses

    old_balance = balance

    if signal == "BUY":
        balance *= 1.01
    elif signal == "SELL":
        balance *= 0.99

    if balance > old_balance:
        wins += 1
    else:
        losses += 1

    equity.append(round(balance,2))
    if len(equity) > 50:
        equity.pop(0)

# ================= COOLDOWN =================
def cooldown(pair, signal, minutes=15):
    key = (pair, signal)
    now = time.time()

    if key in last_sent and (now - last_sent[key]) < (minutes * 60):
        return False

    last_sent[key] = now
    return True

# ================= LOG =================
def log_trade(pair, signal, prob, price, balance):
    row = pd.DataFrame([{
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pair": pair,
        "signal": signal,
        "prob": round(prob, 4),
        "price": round(price, 4),
        "balance": round(balance, 2)
    }])

    header = not os.path.exists(TRADE_LOG)
    row.to_csv(TRADE_LOG, mode="a", header=header, index=False)

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

    last_report = datetime.now().day

    while True:
        for pair in PAIRS:
            try:
                df = get_data(pair)
                df = features(df)

                train(df)
                signal, prob = predict(df)
                price = df["close"].iloc[-1]

                if signal != "HOLD" and prob > 0.65 and cooldown(pair, signal):
                    simulate(signal)
                    log_trade(pair, signal, prob, price, balance)

                    send_telegram(
                        f"📊 {pair}\n📈 {signal}\n💰 {round(price,4)}\n🧠 {round(prob,2)}\n💼 Balance: {round(balance,2)}"
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

        # daily summary
        today = datetime.now().day
        if today != last_report:
            total = wins + losses
            winrate = round((wins/total)*100,2) if total > 0 else 0

            send_telegram(
                f"📊 DAILY REPORT\n💰 Balance: {round(balance,2)}\n📈 Winrate: {winrate}%"
            )

            last_report = today

        save_model()
        time.sleep(120)

# ================= ROUTES =================
@app.route("/")
def home():
    total = wins + losses
    winrate = round((wins/total)*100,2) if total > 0 else 0

    return f"""
    <h1>🤖 AI Bot PRO</h1>
    <p>Balance: {balance}</p>
    <p>Winrate: {winrate}%</p>
    <p><a href='/chart'>📊 Chart</a></p>
    <p><a href='/logs'>🧾 Logs</a></p>
    """

@app.route("/logs")
def logs():
    if not os.path.exists(TRADE_LOG):
        return "No logs yet"
    df = pd.read_csv(TRADE_LOG)
    return df.tail(20).to_html()

@app.route("/chart")
def chart():
    if not equity:
        return "No data yet"

    plt.figure(figsize=(8,4))
    plt.plot(equity)
    plt.title("Equity Curve")
    plt.savefig("chart.png")
    plt.close()

    return send_file("chart.png", mimetype="image/png")

# ================= RUN =================
if __name__ == "__main__":
    import threading
    threading.Thread(target=run, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
