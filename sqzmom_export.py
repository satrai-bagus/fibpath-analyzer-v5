"""
Recreate Squeeze Momentum Indicator [LazyBear] from Pine Script,
fetch ETHUSDT 1h candles from Binance, compute SQZMOM, export to Excel.

Pine reference parameters:
    length   = 20   (BB Length)
    mult     = 2.0  (BB MultFactor)        <- not actually used in BB dev (LazyBear's original quirk)
    lengthKC = 20   (KC Length)
    multKC   = 1.5  (KC MultFactor)        <- used for both BB dev and KC range
    useTrueRange = true
"""

import argparse
import json
import ssl
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# CLI args
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Export ETHUSDT 1h + SQZMOM [LazyBear] to Excel")
parser.add_argument("--start", type=str, default=None,
                    help="Filter start (UTC), e.g. 2026-05-14 or 2026-05-14T00:00")
parser.add_argument("--end", type=str, default=None,
                    help="Filter end (UTC, inclusive), e.g. 2026-05-17 or 2026-05-17T23:00")
parser.add_argument("--symbol", type=str, default="ETHUSDT")
parser.add_argument("--interval", type=str, default="1h")
parser.add_argument("--out", type=str, default=None, help="Output xlsx path")
args = parser.parse_args()


def parse_dt(s):
    if s is None:
        return None
    # accept date-only or with hour
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


# ----------------------------------------------------------------------
# 1) Fetch ETHUSDT 1h klines from Binance Spot (with mirror fallback + retry)
# ----------------------------------------------------------------------
BINANCE_HOSTS = [
    "data-api.binance.vision",   # public market-data mirror, usually unblocked
    "api.binance.com",
    "api1.binance.com",
    "api2.binance.com",
    "api3.binance.com",
    "api4.binance.com",
]


def fetch_klines(symbol="ETHUSDT", interval="1h", start_ms=None, end_ms=None, limit=1000):
    last_err = None
    ctx = ssl.create_default_context()
    for host in BINANCE_HOSTS:
        url = f"https://{host}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        if start_ms is not None:
            url += f"&startTime={start_ms}"
        if end_ms is not None:
            url += f"&endTime={end_ms}"
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                    return json.loads(resp.read())
            except Exception as e:
                last_err = e
                print(f"  [{host}] attempt {attempt+1} failed: {e}")
                time.sleep(1.5)
    raise RuntimeError(f"All Binance hosts failed. Last error: {last_err}")


_INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
    "6h": 21_600_000, "8h": 28_800_000, "12h": 43_200_000,
    "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000,
}


def fetch_klines_range(symbol, interval, start_ms, end_ms):
    """Fetch ALL klines in [start_ms, end_ms], chunking around Binance's 1000-bar limit."""
    step = _INTERVAL_MS.get(interval, 3_600_000)
    all_data = []
    cursor = start_ms
    iterations = 0
    while cursor < end_ms:
        iterations += 1
        if iterations > 50:
            raise RuntimeError("fetch_klines_range exceeded 50 iterations")
        chunk = fetch_klines(symbol=symbol, interval=interval,
                             start_ms=cursor, end_ms=end_ms, limit=1000)
        if not chunk:
            break
        all_data.extend(chunk)
        last_open = chunk[-1][0]
        if last_open + step > end_ms or len(chunk) < 1000:
            break
        cursor = last_open + step
    return all_data


filter_start = parse_dt(args.start)
filter_end = parse_dt(args.end)
# If --end is date-only (e.g. 2026-05-17), include the whole day up to 23:00.
if filter_end is not None and filter_end.hour == 0 and filter_end.minute == 0 and len(args.end) <= 10:
    filter_end = filter_end + pd.Timedelta(hours=23)

# Fetch window: enough warm-up before filter_start (or last 40 days if no filter)
end_dt = (filter_end.to_pydatetime() if filter_end is not None else datetime.now(timezone.utc))
start_dt = end_dt - timedelta(days=40)  # 40 days * 24h = 960 bars (< 1000 limit)
if filter_start is not None and filter_start < pd.Timestamp(start_dt):
    start_dt = (filter_start - pd.Timedelta(days=40)).to_pydatetime()

# Extend fetch end by 50h so TP/SL lookahead window is complete for bars at filter_end
fetch_end_dt = end_dt + timedelta(hours=50)
end_ms = int(fetch_end_dt.timestamp() * 1000)
start_ms = int(start_dt.timestamp() * 1000)

print(f"Fetching {args.symbol} {args.interval} from {start_dt:%Y-%m-%d %H:%M} UTC to {fetch_end_dt:%Y-%m-%d %H:%M} UTC ...")
raw = fetch_klines_range(args.symbol, args.interval, start_ms, end_ms)
print(f"Received {len(raw)} candles.")

df = pd.DataFrame(raw, columns=[
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "qav", "num_trades", "tb_base", "tb_quote", "ignore",
])
df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
for col in ["open", "high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col])


# ----------------------------------------------------------------------
# 1b) Fetch BTC for cross-asset / BTC-correlation features
# ----------------------------------------------------------------------
# BTC features applied to every dataset (including BTC itself = self-features).
# For altcoins: alts often lag/lead BTC -> these features are highly informative.
if args.symbol.upper() == "BTCUSDT":
    print("(BTC=main ticker, reusing main data for BTC features)")
    df_btc = df[["datetime", "open", "high", "low", "close"]].copy()
else:
    print("Fetching BTCUSDT for cross-asset features...")
    raw_btc = fetch_klines_range("BTCUSDT", args.interval, start_ms, end_ms)
    df_btc = pd.DataFrame(raw_btc, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "tb_base", "tb_quote", "ignore",
    ])
    df_btc["datetime"] = pd.to_datetime(df_btc["open_time"], unit="ms", utc=True)
    for c_ in ["open", "high", "low", "close"]:
        df_btc[c_] = pd.to_numeric(df_btc[c_])
    df_btc = df_btc[["datetime", "open", "high", "low", "close"]]
    print(f"  BTC candles: {len(df_btc)}")

# Compute BTC features
btc_c = df_btc["close"]
btc_o = df_btc["open"]
btc_h = df_btc["high"]
btc_l = df_btc["low"]
btc_prev_close = btc_c.shift(1)
btc_tr1 = (btc_h - btc_l).abs()
btc_tr2 = (btc_h - btc_prev_close).abs()
btc_tr3 = (btc_l - btc_prev_close).abs()
btc_tr = pd.concat([btc_tr1, btc_tr2, btc_tr3], axis=1).max(axis=1)
btc_atr = btc_tr.ewm(alpha=1.0/14, adjust=False).mean()
btc_range = btc_h - btc_l

btc_feat = pd.DataFrame({
    "datetime":         df_btc["datetime"],
    "BTC Ret 1h":       btc_c.pct_change().round(5),
    "BTC Ret 4h":       btc_c.pct_change(4).round(5),
    "BTC Ret 24h":      btc_c.pct_change(24).round(5),
    "BTC Body Dir":     np.sign(btc_c - btc_o).astype("int8"),
    "BTC Range/ATR":    (btc_range / btc_atr).round(4),
    "BTC ATR Pct100":   btc_atr.rolling(100).rank(pct=True).round(3),
})

# Merge BTC features into main df on datetime
df = df.merge(btc_feat, on="datetime", how="left")
print(f"  After merge with BTC features: {len(df)} rows, {df['BTC Ret 4h'].notna().sum()} have BTC data")


# ----------------------------------------------------------------------
# 2) Squeeze Momentum Indicator [LazyBear]
# ----------------------------------------------------------------------
length = 20
mult = 2.0          # kept for completeness; LazyBear's code uses multKC for BB dev
lengthKC = 20
multKC = 1.5
use_true_range = True

src = df["close"]
high = df["high"]
low = df["low"]

# True Range
prev_close = src.shift(1)
tr1 = (high - low).abs()
tr2 = (high - prev_close).abs()
tr3 = (low - prev_close).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
rng = tr if use_true_range else (high - low)

# Bollinger Bands (LazyBear quirk: BB dev uses multKC, NOT mult — preserved as-is)
basis = src.rolling(length).mean()
dev = multKC * src.rolling(length).std(ddof=0)   # Pine stdev uses ddof=0
upperBB = basis + dev
lowerBB = basis - dev

# Keltner Channels
ma = src.rolling(lengthKC).mean()
rangema = rng.rolling(lengthKC).mean()
upperKC = ma + rangema * multKC
lowerKC = ma - rangema * multKC

# Squeeze state
sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
noSqz = (~sqzOn) & (~sqzOff)

# Momentum core:
#   val = linreg( source - avg( avg(highest(high,n), lowest(low,n)), sma(close,n) ), n, 0 )
highest_high = high.rolling(lengthKC).max()
lowest_low = low.rolling(lengthKC).min()
mid = ((highest_high + lowest_low) / 2.0 + ma) / 2.0
m = src - mid


def rolling_linreg(arr, n, offset=0):
    """Pine-compatible linreg(arr, n, offset) = intercept + slope * (n-1-offset)
    where regression is y = a + b*x over x = 0..n-1 (x=0 is the oldest bar
    in the window, x=n-1 is the current bar)."""
    arr = np.asarray(arr, dtype=float)
    out = np.full(arr.shape, np.nan)
    x = np.arange(n, dtype=float)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denom = n * sum_x2 - sum_x * sum_x
    for i in range(n - 1, len(arr)):
        y = arr[i - n + 1 : i + 1]
        if np.isnan(y).any():
            continue
        sum_y = y.sum()
        sum_xy = (x * y).sum()
        b = (n * sum_xy - sum_x * sum_y) / denom
        a = (sum_y - b * sum_x) / n
        out[i] = a + b * (n - 1 - offset)
    return out


val = rolling_linreg(m.values, lengthKC, 0)
df["SQZMOM"] = val


# Momentum colour (matches Pine's bcolor logic)
def momentum_color(v, v_prev):
    if np.isnan(v) or np.isnan(v_prev):
        return ""
    if v > 0:
        return "lime" if v > v_prev else "green"
    else:
        return "red" if v < v_prev else "maroon"


prev = df["SQZMOM"].shift(1)
df["Momentum Color"] = [momentum_color(v, p) for v, p in zip(df["SQZMOM"], prev)]

# Squeeze status (scolor logic)
df["Squeeze Status"] = np.where(noSqz, "No Squeeze (blue)",
                        np.where(sqzOn, "Squeeze ON (black)", "Squeeze OFF (gray)"))


# ----------------------------------------------------------------------
# 3) Bar strength classification (Fibonacci zones)
# ----------------------------------------------------------------------
# For GREEN bar (close > open): fib drawn from open (0) -> high (1).
#   Position = (close - open) / (high - open). Zone 6 = close at high = strongest bull.
# For RED bar (close < open):   fib drawn from open (0) -> low (1).
#   Position = (open - close) / (open - low). Zone 6 = close at low = strongest bear.
# DOJI (close == open): zone undefined.
o = df["open"]; h = df["high"]; l = df["low"]; c = df["close"]

bar_color = np.where(c > o, "Green", np.where(c < o, "Red", "Doji"))
df["Bar Color"] = bar_color

pos = pd.Series(np.nan, index=df.index, dtype=float)
mask_g = c > o
mask_r = c < o
pos.loc[mask_g] = (c[mask_g] - o[mask_g]) / (h[mask_g] - o[mask_g])
pos.loc[mask_r] = (o[mask_r] - c[mask_r]) / (o[mask_r] - l[mask_r])
df["Fib Position"] = pos.round(4)

fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
df["Fib Zone"] = pd.cut(pos, bins=fib_levels, labels=[1, 2, 3, 4, 5, 6],
                        include_lowest=True).astype("Int64")

# Bar shape ratios
rng_bar = (h - l)
body = (c - o).abs()
upper_wick = h - np.maximum(o, c)
lower_wick = np.minimum(o, c) - l
df["Body %"] = (body / rng_bar).where(rng_bar > 0, 0).round(4)
df["Upper Wick %"] = (upper_wick / rng_bar).where(rng_bar > 0, 0).round(4)
df["Lower Wick %"] = (lower_wick / rng_bar).where(rng_bar > 0, 0).round(4)
df["Range"] = rng_bar.round(4)

# ATR(14) using Wilder smoothing (Pine-compatible)
atr_period = 14
atr = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean()
df["Range/ATR"] = (rng_bar / atr).round(4)


# ----------------------------------------------------------------------
# 3a-extra) Volume, multi-bar context, time-of-day, volatility regime
# ----------------------------------------------------------------------
vol = df["volume"]
vol_ma20 = vol.rolling(20).mean()
df["Volume"] = vol.round(2)
df["Vol/MA20"] = (vol / vol_ma20).round(3)
df["Vol Pct100"] = vol.rolling(100).rank(pct=True).round(3)  # 0-1 percentile last 100 bars

# Multi-bar context — what happened in the recent past leading into this bar
body_pct_of_price = (c - o).abs() / c
df["Body% MA5"] = body_pct_of_price.rolling(5).mean().round(5)

# Streak: how many consecutive Green or Red bars (signed)
direction_sign = np.sign(c - o).fillna(0)
streak = direction_sign.groupby((direction_sign != direction_sign.shift()).cumsum()).cumcount() + 1
df["Streak"] = (streak * direction_sign).astype("int16")  # +5 = 5 green in a row, -3 = 3 red, etc.

# SQZMOM trajectory — momentum-of-momentum
df["SQZMOM Delta3"] = (df["SQZMOM"] - df["SQZMOM"].shift(3)).round(4)
# 5-bar return
df["Close Ret 5"] = ((c - c.shift(5)) / c.shift(5)).round(5)

# Lag features for previous bar's behaviour
df["Prev Body%"] = body_pct_of_price.shift(1).round(5)
df["Prev Bar Color"] = direction_sign.shift(1).fillna(0).astype("int8")  # +1/-1/0
df["Prev Range/ATR"] = df["Range/ATR"].shift(1)

# Time-of-day cyclic encoding (crypto liquidity is sessional)
dt_utc = df["datetime"].dt.tz_convert("UTC")
hour = dt_utc.dt.hour
dow = dt_utc.dt.dayofweek
df["Hour Sin"] = np.sin(2 * np.pi * hour / 24).round(4)
df["Hour Cos"] = np.cos(2 * np.pi * hour / 24).round(4)
df["DoW Sin"]  = np.sin(2 * np.pi * dow / 7).round(4)
df["DoW Cos"]  = np.cos(2 * np.pi * dow / 7).round(4)
df["Hour UTC"] = hour.astype("int8")
df["DoW"]      = dow.astype("int8")

# Volatility regime: ATR percentile rank (rolling 100 bars)
df["ATR Pct100"] = atr.rolling(100).rank(pct=True).round(3)

# Body% percentile (so model knows if THIS bar's body is large vs recent history)
df["Body% Pct100"] = body_pct_of_price.rolling(100).rank(pct=True).round(3)


# ----------------------------------------------------------------------
# 3a-bis) Analisa.ipynb-style indicators + 6-factor scoring (vectorised)
# ----------------------------------------------------------------------
# Trend EMAs
ema21 = c.ewm(span=21, adjust=False).mean()
ema50 = c.ewm(span=50, adjust=False).mean()

# MACD (12/26/9) — note: Analisa.ipynb uses SMA for signal line (kept identical)
ema12 = c.ewm(span=12, adjust=False).mean()
ema26 = c.ewm(span=26, adjust=False).mean()
macd_line = ema12 - ema26
macd_sig = macd_line.rolling(9).mean()

# RSI(14) — simple rolling mean of gains/losses (Analisa style)
delta = c.diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
rsi14 = 100 - (100 / (1 + rs))

# ADX(14) — rolling-mean style (matches Analisa.ipynb)
up_move = h - h.shift(1)
down_move = l.shift(1) - l
plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
tr_roll = tr.rolling(14).mean()
plus_dm_smooth = plus_dm.rolling(14).mean()
minus_dm_smooth = minus_dm.rolling(14).mean()
plus_di = 100 * (plus_dm_smooth / tr_roll)
minus_di = 100 * (minus_dm_smooth / tr_roll)
dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
adx14 = dx.rolling(14).mean()

# HTF 4H trend (resample 1h -> 4h, EMA50 + slope; shift(1) to avoid lookahead)
df_4h = (df.set_index("datetime")[["close"]]
         .resample("4h").last().dropna())
ema50_4h = df_4h["close"].ewm(span=50, adjust=False).mean()
slope_4h = ema50_4h.diff()
trend_4h = pd.Series(0, index=df_4h.index, dtype="int8")
trend_4h[(df_4h["close"] > ema50_4h) & (slope_4h > 0)] = 1
trend_4h[(df_4h["close"] < ema50_4h) & (slope_4h < 0)] = -1
# shift(1) so a 1h bar uses the PREVIOUS completed 4h trend (no future leak)
trend_4h_shifted = trend_4h.shift(1)
htf_trend = (trend_4h_shifted.reindex(df["datetime"], method="ffill")
             .fillna(0).astype("int8").values)

# 6-factor score (vectorised version of compute_signal_from_indicators)
ADX_TREND = 20.0
ADX_NO_TRADE = 15.0
EXTREME_TR_MULT = 1.6
MIN_SCORE = 1

s1 = np.where(c > ema21, 1, -1)
s2 = np.where(ema21 > ema50, 1, -1)
s3 = np.where(macd_line > macd_sig, 1, np.where(macd_line < macd_sig, -1, 0))
s4 = np.where(macd_line > 0, 1, np.where(macd_line < 0, -1, 0))
s5 = np.where(rsi14 > 55, 1, np.where(rsi14 < 45, -1, 0))
adx_high = adx14 >= ADX_TREND
s6 = np.where(adx_high & (ema21 > ema50), 1,
              np.where(adx_high & (ema21 < ema50), -1, 0))
score = s1 + s2 + s3 + s4 + s5 + s6

raw_posisi = np.where(score >= MIN_SCORE, "LONG",
                      np.where(score <= -MIN_SCORE, "SHORT", "NO TRADE"))

# Filters
filter_adx_low = (adx14 < ADX_NO_TRADE).fillna(False).values
filter_tr_spike = (tr > EXTREME_TR_MULT * atr).fillna(False).values
filter_any = filter_adx_low | filter_tr_spike
posisi_final = np.where(filter_any, "NO TRADE", raw_posisi)

# Confidence (50–90) and quality label
strength = np.minimum(np.abs(score) / 6.0, 1.0)
confidence_pct = 50 + strength * 40
quality = np.where((confidence_pct >= 75) & (adx14.fillna(0) >= 20), "STRONG",
           np.where(confidence_pct >= 60, "MEDIUM", "WEAK"))

df["EMA 21"]       = ema21.round(4)
df["EMA 50"]       = ema50.round(4)
df["MACD"]         = macd_line.round(4)
df["MACD Signal"]  = macd_sig.round(4)
df["RSI 14"]       = rsi14.round(2)
df["ADX 14"]       = adx14.round(2)
df["ATR 14"]       = atr.round(4)
df["Last TR"]      = tr.round(4)
df["HTF 4H Trend"] = htf_trend
df["Score"]        = score
df["Raw Posisi"]   = raw_posisi
df["Posisi Final"] = posisi_final
df["Confidence %"] = np.round(confidence_pct, 1)
df["Quality"]      = quality


# ----------------------------------------------------------------------
# 3b) TP/SL hit-tracking over next 48 hours
# ----------------------------------------------------------------------
# Fib levels are body-based, mirrored symmetrically:
#   body_top    = max(open, close)
#   body_bottom = min(open, close)
#   body        = |close - open|
#   Fib X.XX Up   = body_top    + (X.XX - 1) * body     (above body)
#   Fib X.XX Down = body_bottom - (X.XX - 1) * body     (below body)
# Lookahead: next 48 hourly bars (excluding the anchor bar itself).
# Hit detection: high[j] >= Up level  OR  low[j] <= Down level.
# Order: all levels first-touched in the same hour share the same order number.
LOOKAHEAD = 48
fib_levels_spec = [
    ("1.61", 0.61),
    ("2.5",  1.5),
    ("3.6",  2.6),
]

n = len(df)
o_arr = df["open"].values
h_arr = df["high"].values
l_arr = df["low"].values
c_arr = df["close"].values

# pre-allocate 12 result columns
results = {}
for tag, _ in fib_levels_spec:
    results[f"Fib {tag} Up"] = np.zeros(n, dtype=bool)
    results[f"Fib {tag} Up Order"] = np.zeros(n, dtype=np.int16)
    results[f"Fib {tag} Down"] = np.zeros(n, dtype=bool)
    results[f"Fib {tag} Down Order"] = np.zeros(n, dtype=np.int16)
lookahead_count = np.zeros(n, dtype=np.int16)

for i in range(n):
    body_top = max(o_arr[i], c_arr[i])
    body_bot = min(o_arr[i], c_arr[i])
    body_len = body_top - body_bot
    if body_len == 0:
        continue  # doji, skip

    # 6 levels for this anchor bar
    up_levels = [(tag, body_top + mult * body_len) for tag, mult in fib_levels_spec]
    dn_levels = [(tag, body_bot - mult * body_len) for tag, mult in fib_levels_spec]

    # track which level slots already filled (so we record FIRST hit only)
    up_done = {tag: False for tag, _ in fib_levels_spec}
    dn_done = {tag: False for tag, _ in fib_levels_spec}

    end = min(i + 1 + LOOKAHEAD, n)
    lookahead_count[i] = end - (i + 1)
    event_counter = 0

    for j in range(i + 1, end):
        new_hits = []  # list of ("Up"/"Down", tag)
        for tag, lvl in up_levels:
            if not up_done[tag] and h_arr[j] >= lvl:
                new_hits.append(("Up", tag))
        for tag, lvl in dn_levels:
            if not dn_done[tag] and l_arr[j] <= lvl:
                new_hits.append(("Down", tag))
        if new_hits:
            event_counter += 1
            for side, tag in new_hits:
                if side == "Up":
                    up_done[tag] = True
                    results[f"Fib {tag} Up"][i] = True
                    results[f"Fib {tag} Up Order"][i] = event_counter
                else:
                    dn_done[tag] = True
                    results[f"Fib {tag} Down"][i] = True
                    results[f"Fib {tag} Down Order"][i] = event_counter

for col, arr in results.items():
    df[col] = arr
df["Lookahead Bars"] = lookahead_count


# ----------------------------------------------------------------------
# 4) Export
# ----------------------------------------------------------------------
out = df[[
    "datetime", "open", "high", "low", "close",
    "Volume", "Vol/MA20", "Vol Pct100",
    "Bar Color", "Fib Zone", "Fib Position",
    "Body %", "Upper Wick %", "Lower Wick %", "Range", "Range/ATR",
    "Body% MA5", "Body% Pct100", "ATR Pct100",
    "Streak", "Prev Body%", "Prev Bar Color", "Prev Range/ATR",
    "Close Ret 5", "SQZMOM Delta3",
    "Hour UTC", "DoW", "Hour Sin", "Hour Cos", "DoW Sin", "DoW Cos",
    "BTC Ret 1h", "BTC Ret 4h", "BTC Ret 24h",
    "BTC Body Dir", "BTC Range/ATR", "BTC ATR Pct100",
    "SQZMOM", "Momentum Color", "Squeeze Status",
    "EMA 21", "EMA 50", "MACD", "MACD Signal", "RSI 14", "ADX 14", "ATR 14",
    "Last TR", "HTF 4H Trend",
    "Score", "Raw Posisi", "Posisi Final", "Confidence %", "Quality",
    "Fib 1.61 Up", "Fib 1.61 Up Order",
    "Fib 1.61 Down", "Fib 1.61 Down Order",
    "Fib 2.5 Up", "Fib 2.5 Up Order",
    "Fib 2.5 Down", "Fib 2.5 Down Order",
    "Fib 3.6 Up", "Fib 3.6 Up Order",
    "Fib 3.6 Down", "Fib 3.6 Down Order",
    "Lookahead Bars",
]].copy()

# Apply requested filter (warm-up rows are dropped here, indicator already computed)
if filter_start is not None:
    out = out[out["datetime"] >= filter_start]
if filter_end is not None:
    out = out[out["datetime"] <= filter_end]

out["datetime"] = out["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)  # Excel cannot store tz-aware
out.columns = [
    "Datetime (UTC)", "Open", "High", "Low", "Close",
    "Volume", "Vol/MA20", "Vol Pct100",
    "Bar Color", "Fib Zone", "Fib Position",
    "Body %", "Upper Wick %", "Lower Wick %", "Range", "Range/ATR",
    "Body% MA5", "Body% Pct100", "ATR Pct100",
    "Streak", "Prev Body%", "Prev Bar Color", "Prev Range/ATR",
    "Close Ret 5", "SQZMOM Delta3",
    "Hour UTC", "DoW", "Hour Sin", "Hour Cos", "DoW Sin", "DoW Cos",
    "BTC Ret 1h", "BTC Ret 4h", "BTC Ret 24h",
    "BTC Body Dir", "BTC Range/ATR", "BTC ATR Pct100",
    "SQZMOM Value", "Momentum Color", "Squeeze Status",
    "EMA 21", "EMA 50", "MACD", "MACD Signal", "RSI 14", "ADX 14", "ATR 14",
    "Last TR", "HTF 4H Trend",
    "Score", "Raw Posisi", "Posisi Final", "Confidence %", "Quality",
    "Fib 1.61 Up", "Fib 1.61 Up Order",
    "Fib 1.61 Down", "Fib 1.61 Down Order",
    "Fib 2.5 Up", "Fib 2.5 Up Order",
    "Fib 2.5 Down", "Fib 2.5 Down Order",
    "Fib 3.6 Up", "Fib 3.6 Up Order",
    "Fib 3.6 Down", "Fib 3.6 Down Order",
    "Lookahead Bars",
]

if args.out:
    out_path = args.out
elif filter_start is not None and filter_end is not None:
    out_path = f"ETH_SQZMOM_{args.interval}_{filter_start:%Y%m%d}_{filter_end:%Y%m%d}.xlsx"
else:
    out_path = f"ETH_SQZMOM_{args.interval}.xlsx"
out.to_excel(out_path, index=False)
print(f"\nWrote {len(out)} rows to {out_path}")


# ----------------------------------------------------------------------
# 4) Sanity print: bars around 20 May 2026 07:00 UTC (the value Anda lihat = 4.71)
# ----------------------------------------------------------------------
print("\nBars around 2026-05-20 07:00 UTC:")
target = pd.Timestamp("2026-05-20 07:00", tz="UTC")
mask = (df["datetime"] >= target - pd.Timedelta(hours=4)) & \
       (df["datetime"] <= target + pd.Timedelta(hours=2))
show = df.loc[mask, ["datetime", "open", "high", "low", "close", "SQZMOM", "Squeeze Status"]]
with pd.option_context("display.max_columns", None, "display.width", 160):
    print(show.to_string(index=False))

last = df.iloc[-1]
print(f"\nLast bar in dataset: {last['datetime']}  close={last['close']}  SQZMOM={last['SQZMOM']:.4f}")
