from flask import Flask, render_template_string
import threading
import requests
import pandas as pd
import time
import os

app = Flask(__name__)

signals = []
stats = {"wins": 0, "losses": 0}

PAIRS = ["B-DOGE_USDT", "B-MATIC_USDT", "B-SHIB_USDT"]

# ================= FETCH DATA =================
def get_price(pair):
    try:
        url = "https://public.coindcx.com/market_data/candles"
        params = {"pair": pair, "interval": "5m", "limit": 50}
        data = requests.get(url, params=params).json()

        df = pd.DataFrame(data)
        df.columns = [c.lower() for c in df.columns]
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

        return df["close"].iloc[-1]
    except:
        return None

# ================= BOT =================
def bot_loop():
    while True:
        for pair in PAIRS:
            try:
                price = get_price(pair)

                if price is None:
                    continue

                # simple demo logic
                signal = "BUY" if price % 2 == 0 else "SELL"

                signals.append({
                    "pair": pair,
                    "signal": signal,
                    "price": round(price, 6),
                    "time": time.strftime("%H:%M:%S")
                })

                if len(signals) > 10:
                    signals.pop(0)

            except Exception as e:
                print("Error:", e)

        time.sleep(60)

# ================= UI =================
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Trading Dashboard</title>
<meta http-equiv="refresh" content="5">
<style>
body { font-family: Arial; background: #0f172a; color: white; text-align:center; }
.card { background:#1e293b; padding:15px; margin:10px; border-radius:10px; }
.buy { color: #22c55e; }
.sell { color: #ef4444; }
</style>
</head>
<body>

<h1>🚀 Live Trading Dashboard</h1>

<div class="card">
<h2>📊 Signals</h2>
{% for s in signals %}
<p class="{{'buy' if s.signal=='BUY' else 'sell'}}">
{{s.signal}} - {{s.pair}} @ {{s.price}} ({{s.time}})
</p>
{% endfor %}
</div>

<div class="card">
<h2>📈 Stats</h2>
<p>Wins: {{stats.wins}}</p>
<p>Losses: {{stats.losses}}</p>
</div>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML, signals=signals, stats=stats)

# ================= RUN =================
if __name__ == "__main__":
    threading.Thread(target=bot_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
