"""
Fib Dataset Export
==================
Regenerate the FULL "Dataset Analisis Trading" schema (input features + the 6
fib-rank OUTPUT columns) automatically from Binance ETHUSDT 1h, one sheet per
month. No more manual labelling of the output columns.

Output columns (exact order of 'Dataset Analisis Trading.xlsx'):
  Trend, SQZMOM 1 Value/Momentum/Squeeze, SQZMOM 2 Value/Momentum/Squeeze,
  Bar 1, Bar 2, Raw Position, Final Position, Last TR, Score,
  Fib 1,61 Up .. Fib 3,6 Down  (hit ✅/❌),
  Date, Clock,
  Fib 1,61 Up2 .. Fib 3,6 Down7 (first-hit RANK, the auto-computed OUTPUT).

OUTPUT logic (validated; reproduces user's May labels 81% row / 95% cell):
  - Fib drawn on the BODY of the anchor bar, mirrored both ways:
        Up_X   = max(open,close) + (X-1)*body
        Down_X = min(open,close) - (X-1)*body      for X in {1.61, 2.5, 3.6}
  - Scan the NEXT 48 hourly bars (anchor excluded). Hit when high>=Up or low<=Down.
  - Rank = event-bucket: all levels first-touched in the SAME hour share the same
    number; rank increments per hour-with-new-hits. 0 = not touched within 48h.
  - Timezone = UTC. Clock = UTC hour (0-23).

Column formulas (verified against Binance-sourced April 2026 = 100% match):
  Trend = "Long" if MACD(12,26) > 0 else "Short".
  SQZMOM = Squeeze Momentum [LazyBear]; SQZMOM 2 = previous row.
  Bar = "{Green/Red} Bar Line {zone 1-6}"  (open->high green / open->low red fib zone).
  Score / Raw / Final = 6-factor score (EMA21/50, MACD, RSI, ADX) + spike/ADX filters.

Usage:
  python fib_dataset_export.py                      # Maret..current month, 2026
  python fib_dataset_export.py --start 2026-03 --end 2026-06 --out my.xlsx
"""
import argparse, json, ssl, time, urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- Binance fetch
BINANCE_HOSTS = ["data-api.binance.vision", "api.binance.com",
                 "api1.binance.com", "api2.binance.com"]
_INTERVAL_MS = {"1h": 3_600_000}


def fetch_klines(symbol, interval, start_ms, end_ms, limit=1000):
    ctx = ssl.create_default_context()
    last_err = None
    for host in BINANCE_HOSTS:
        url = (f"https://{host}/api/v3/klines?symbol={symbol}&interval={interval}"
               f"&limit={limit}&startTime={start_ms}&endTime={end_ms}")
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                    return json.loads(resp.read())
            except Exception as e:  # noqa
                last_err = e
                time.sleep(1.0)
    raise RuntimeError(f"All Binance hosts failed. Last error: {last_err}")


def fetch_range(symbol, interval, start_ms, end_ms):
    step = _INTERVAL_MS.get(interval, 3_600_000)
    out, cursor, it = [], start_ms, 0
    while cursor < end_ms:
        it += 1
        if it > 300:
            raise RuntimeError("fetch_range exceeded 300 iterations")
        chunk = fetch_klines(symbol, interval, cursor, end_ms)
        if not chunk:
            break
        out.extend(chunk)
        last_open = chunk[-1][0]
        if last_open + step > end_ms or len(chunk) < 1000:
            break
        cursor = last_open + step
    return out


def load_ohlc(symbol, interval, start_dt, end_dt):
    raw = fetch_range(symbol, interval,
                      int(start_dt.timestamp() * 1000),
                      int(end_dt.timestamp() * 1000))
    df = pd.DataFrame(raw, columns=["open_time", "open", "high", "low", "close", "volume",
                                    "close_time", "qav", "n", "tb", "tq", "ig"])
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c])
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df[["datetime", "open", "high", "low", "close"]]


# ---------------------------------------------------------------- indicators
def rolling_linreg(arr, n, offset=0):
    arr = np.asarray(arr, dtype=float)
    out = np.full(arr.shape, np.nan)
    x = np.arange(n, dtype=float)
    sx, sx2 = x.sum(), (x * x).sum()
    den = n * sx2 - sx * sx
    for i in range(n - 1, len(arr)):
        y = arr[i - n + 1: i + 1]
        if np.isnan(y).any():
            continue
        sy, sxy = y.sum(), (x * y).sum()
        b = (n * sxy - sx * sy) / den
        a = (sy - b * sx) / n
        out[i] = a + b * (n - 1 - offset)
    return out


MONTHS_ID = {1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
             7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November",
             12: "Desember"}

FIBSPEC = [("1.61", 0.61), ("2.5", 1.5), ("3.6", 2.6)]
ORDER_KEYS = ["1.61_UP", "1.61_DOWN", "2.5_UP", "2.5_DOWN", "3.6_UP", "3.6_DOWN"]


def compute_features(df):
    df = df.sort_values("datetime").reset_index(drop=True).copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    src = c

    # --- Squeeze Momentum [LazyBear] ---
    length = lengthKC = 20
    multKC = 1.5
    pc = src.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    ma = src.rolling(lengthKC).mean()
    hh, ll = h.rolling(lengthKC).max(), l.rolling(lengthKC).min()
    mid = ((hh + ll) / 2.0 + ma) / 2.0
    df["SQZMOM"] = rolling_linreg((src - mid).values, lengthKC, 0)
    basis = src.rolling(length).mean()
    dev = multKC * src.rolling(length).std(ddof=0)
    upBB, loBB = basis + dev, basis - dev
    rangema = tr.rolling(lengthKC).mean()
    upKC, loKC = ma + rangema * multKC, ma - rangema * multKC
    sqzOn = (loBB > loKC) & (upBB < upKC)
    sqzOff = (loBB < loKC) & (upBB > upKC)
    noSqz = (~sqzOn) & (~sqzOff)
    prev = df["SQZMOM"].shift(1)

    def mcol(v, p):
        if np.isnan(v) or np.isnan(p):
            return ""
        if v > 0:
            return "lime" if v > p else "green"
        return "red" if v < p else "maroon"

    df["MomColor"] = [mcol(v, p) for v, p in zip(df["SQZMOM"], prev)]
    df["Squeeze"] = np.where(noSqz, "No Squeeze (blue)",
                             np.where(sqzOn, "Squeeze ON (black)", "Squeeze OFF (gray)"))

    # --- Bar zone (fib position within bar; open->high green, open->low red) ---
    bar_color = np.where(c > o, "Green", np.where(c < o, "Red", "Doji"))
    pos = pd.Series(np.nan, index=df.index, dtype=float)
    mg, mr = c > o, c < o
    pos.loc[mg] = (c[mg] - o[mg]) / (h[mg] - o[mg])
    pos.loc[mr] = (o[mr] - c[mr]) / (o[mr] - l[mr])
    zone = pd.cut(pos, bins=[0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
                  labels=[1, 2, 3, 4, 5, 6], include_lowest=True).astype("Int64")
    df["Bar"] = [f"{col} Bar Line {int(z)}" if col != "Doji" and pd.notna(z) else ""
                 for col, z in zip(bar_color, zone)]

    # --- score / position / trend ---
    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.rolling(9).mean()
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    rsi = 100 - 100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean())
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    up_m, dn_m = h.diff(), -l.diff()
    pdm = up_m.where((up_m > dn_m) & (up_m > 0), 0.0)
    mdm = dn_m.where((dn_m > up_m) & (dn_m > 0), 0.0)
    trr = tr.rolling(14).mean()
    pdi = 100 * pdm.rolling(14).mean() / trr
    mdi = 100 * mdm.rolling(14).mean() / trr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    adx = dx.rolling(14).mean()
    s1 = np.where(c > ema21, 1, -1)
    s2 = np.where(ema21 > ema50, 1, -1)
    s3 = np.where(macd > macd_sig, 1, np.where(macd < macd_sig, -1, 0))
    s4 = np.where(macd > 0, 1, np.where(macd < 0, -1, 0))
    s5 = np.where(rsi > 55, 1, np.where(rsi < 45, -1, 0))
    ah = adx >= 20
    s6 = np.where(ah & (ema21 > ema50), 1, np.where(ah & (ema21 < ema50), -1, 0))
    score = s1 + s2 + s3 + s4 + s5 + s6
    raw = np.where(score >= 1, "LONG", np.where(score <= -1, "SHORT", "NO TRADE"))
    fa = (adx < 15).fillna(False).values
    fb = (tr > 1.6 * atr).fillna(False).values
    final = np.where(fa | fb, "NO TRADE", raw)

    df["Score"] = score
    df["Raw"] = raw
    df["Final"] = final
    df["LastTR"] = tr
    df["Trend"] = np.where(macd > 0, "Long", "Short")
    return df


def compute_fib_orders(df, lookahead=48):
    o, h, l, c = (df["open"].values, df["high"].values,
                  df["low"].values, df["close"].values)
    n = len(df)
    cols = {f"{t}_UP": np.zeros(n, np.int16) for t, _ in FIBSPEC}
    cols.update({f"{t}_DOWN": np.zeros(n, np.int16) for t, _ in FIBSPEC})
    lookn = np.zeros(n, np.int16)
    for i in range(n):
        bt, bb = max(o[i], c[i]), min(o[i], c[i])
        bl = bt - bb
        end = min(i + 1 + lookahead, n)
        lookn[i] = end - (i + 1)
        if bl == 0:
            continue
        up = {t: bt + mu * bl for t, mu in FIBSPEC}
        dn = {t: bb - mu * bl for t, mu in FIBSPEC}
        fh = {k: None for k in cols}
        for off, j in enumerate(range(i + 1, end), 1):
            for t, _ in FIBSPEC:
                if fh[f"{t}_UP"] is None and h[j] >= up[t]:
                    fh[f"{t}_UP"] = off
            for t, _ in FIBSPEC:
                if fh[f"{t}_DOWN"] is None and l[j] <= dn[t]:
                    fh[f"{t}_DOWN"] = off
        hrs = sorted(set(v for v in fh.values() if v is not None))
        h2r = {hh_: ix + 1 for ix, hh_ in enumerate(hrs)}
        for k in cols:
            if fh[k] is not None:
                cols[k][i] = h2r[fh[k]]
    for k, v in cols.items():
        df["ord_" + k] = v
    df["LookN"] = lookn
    return df


# ---------------------------------------------------------------- assemble schema
SHEET_COLUMNS = [
    "Trend", "SQZMOM 1 Value", "SQZMOM 1 Momentum", "SQZMOM 1 Squeeze",
    "SQZMOM 2 Value", "SQZMOM 2 Momentum", "SQZMOM 2 Squeeze",
    "Bar 1", "Bar 2", "Raw Position", "Final Position", "Last TR", "Score",
    "Fib 1,61 Up", "Fib 1,61 Down", "Fib 2,5 Up", "Fib 2,5 Down",
    "Fib 3,6 Up", "Fib 3,6 Down", "Date", "Clock",
    "Fib 1,61 Up2", "Fib 1,61 Down3", "Fib 2,5 Up4", "Fib 2,5 Down5",
    "Fib 3,6 Up6", "Fib 3,6 Down7",
]
HIT_MAP = [("Fib 1,61 Up", "ord_1.61_UP"), ("Fib 1,61 Down", "ord_1.61_DOWN"),
           ("Fib 2,5 Up", "ord_2.5_UP"), ("Fib 2,5 Down", "ord_2.5_DOWN"),
           ("Fib 3,6 Up", "ord_3.6_UP"), ("Fib 3,6 Down", "ord_3.6_DOWN")]
RANK_MAP = [("Fib 1,61 Up2", "ord_1.61_UP"), ("Fib 1,61 Down3", "ord_1.61_DOWN"),
            ("Fib 2,5 Up4", "ord_2.5_UP"), ("Fib 2,5 Down5", "ord_2.5_DOWN"),
            ("Fib 3,6 Up6", "ord_3.6_UP"), ("Fib 3,6 Down7", "ord_3.6_DOWN")]


def build_month_sheet(df_month, incomplete_mask):
    """df_month: enriched bars for one month, in time order. Returns DataFrame in
    the exact Dataset schema; SQZMOM 2 / Bar 2 = previous row within the sheet."""
    out = pd.DataFrame(index=range(len(df_month)), columns=SHEET_COLUMNS)
    out["Trend"] = df_month["Trend"].values
    out["SQZMOM 1 Value"] = np.round(df_month["SQZMOM"].values, 6)
    out["SQZMOM 1 Momentum"] = df_month["MomColor"].values
    out["SQZMOM 1 Squeeze"] = df_month["Squeeze"].values
    out["SQZMOM 2 Value"] = np.round(df_month["SQZMOM"].shift(1).values, 6)
    out["SQZMOM 2 Momentum"] = df_month["MomColor"].shift(1).values
    out["SQZMOM 2 Squeeze"] = df_month["Squeeze"].shift(1).values
    out["Bar 1"] = df_month["Bar"].values
    out["Bar 2"] = df_month["Bar"].shift(1).values
    out["Raw Position"] = df_month["Raw"].values
    out["Final Position"] = df_month["Final"].values
    out["Last TR"] = np.round(df_month["LastTR"].values, 2)
    out["Score"] = df_month["Score"].values
    out["Date"] = [f"{d.day} {MONTHS_ID[d.month]} {d.year}" for d in df_month["datetime"]]
    out["Clock"] = [d.hour for d in df_month["datetime"]]
    for col, oc in HIT_MAP:
        out[col] = np.where(df_month[oc].values > 0, "✅", "❌")
    for col, oc in RANK_MAP:
        out[col] = df_month[oc].values.astype(int)
    # blank out OUTPUT for bars whose 48h window is incomplete (future not known yet)
    if incomplete_mask is not None and incomplete_mask.any():
        for col, _ in HIT_MAP + RANK_MAP:
            out[col] = out[col].astype(object)
            out.loc[incomplete_mask.values, col] = ""
    return out


def main():
    ap = argparse.ArgumentParser(description="Export full Fib dataset (input+output) per month")
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2026-03", help="first month YYYY-MM")
    ap.add_argument("--end", default=None, help="last month YYYY-MM (default: current month)")
    ap.add_argument("--out", default="Dataset Output Otomatis.xlsx")
    ap.add_argument("--lookahead", type=int, default=48)
    args = ap.parse_args()

    start_month = pd.Period(args.start, "M")
    end_month = pd.Period(args.end, "M") if args.end else pd.Period(
        pd.Timestamp.now(tz="UTC").strftime("%Y-%m"), "M")

    range_start = start_month.start_time.tz_localize("UTC")
    range_end = (end_month.end_time.tz_localize("UTC")).ceil("h")
    # warmup before + lookahead after
    fetch_start = (range_start - pd.Timedelta(days=40)).to_pydatetime()
    fetch_end = (range_end + pd.Timedelta(hours=args.lookahead + 2)).to_pydatetime()

    print(f"Fetching {args.symbol} {args.interval} "
          f"{fetch_start:%Y-%m-%d %H:%M} .. {fetch_end:%Y-%m-%d %H:%M} UTC ...")
    df = load_ohlc(args.symbol, args.interval, fetch_start, fetch_end)
    print(f"  {len(df)} bars: {df['datetime'].min()} .. {df['datetime'].max()}")

    df = compute_features(df)
    df = compute_fib_orders(df, lookahead=args.lookahead)

    last_complete = df["datetime"].max() - pd.Timedelta(hours=args.lookahead)
    months = pd.period_range(start_month, end_month, freq="M")
    sheets = {}
    for per in months:
        m0 = per.start_time.tz_localize("UTC")
        m1 = per.end_time.tz_localize("UTC")
        sub = df[(df["datetime"] >= m0) & (df["datetime"] <= m1)].copy()
        if sub.empty:
            continue
        incomplete = sub["datetime"] > last_complete
        sheet = build_month_sheet(sub.reset_index(drop=True),
                                  incomplete.reset_index(drop=True))
        name = f"{MONTHS_ID[per.month]} {per.year}"
        sheets[name] = sheet
        nblank = int(incomplete.sum())
        print(f"  sheet '{name}': {len(sheet)} rows"
              + (f"  ({nblank} recent rows have incomplete 48h window -> output left blank)"
                 if nblank else ""))

    with pd.ExcelWriter(args.out, engine="openpyxl") as xw:
        for name, sheet in sheets.items():
            sheet.to_excel(xw, sheet_name=name[:31], index=False)
    print(f"\nWrote {len(sheets)} sheet(s) to: {args.out}")


if __name__ == "__main__":
    main()
