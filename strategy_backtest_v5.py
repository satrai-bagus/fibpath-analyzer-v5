"""
Bar-by-bar P&L backtest of the USER's actual strategy, out-of-sample.
======================================================================
Rules (per user):
  - Direction from model: side with the higher reach probability (1.61+2.5).
  - Entry: close of the bar AFTER the anchor (anchor+1), once the 2nd bar formed.
  - Target (TP): the 2.5 fib level of the ANCHOR body, in the trade direction.
  - Stop (SL): from the 2 bars combined (anchor + next), fib 3.6 level (against trade).
               -> tested in several interpretations (pick the one matching your platform).
  - Gate: only trade when top-match similarity >= threshold (25-30%).

No leakage: walk-forward monthly, engine trained only on rows before (month start - 48h).
Fib body convention (same as the dataset): Up_X = max(O,C)+(X-1)*body, Down_X = min(O,C)-(X-1)*body.
"""
import numpy as np
import pandas as pd

from fib_dataset_export import load_ohlc
from fib_pattern_engine_v5 import (
    FibPatternEngineV5, FEATURE_CATEGORICAL, FEATURE_NUMERIC,
    RANK_COLUMNS, ACTIONABLE_TARGETS, MONTH_MAP_ID,
)
from datetime import datetime, timezone

DATA = "Dataset Output Otomatis 2025-2026.xlsx"
SYMBOL, INTERVAL = "ETHUSDT", "1h"
GAP_HOURS = 48
MIN_TRAIN_MONTHS = 6
MAX_HOLD = 48            # close trade after 48h if neither TP nor SL hit
FEE_RT = 0.001          # 0.10% round-trip (taker both sides) — adjust to your exchange
SIM_GATES = [0.0, 0.20, 0.25, 0.30, 0.35]


def parse_dt(date_val, clock):
    if pd.isna(date_val) or pd.isna(clock):
        return pd.NaT
    parts = str(date_val).strip().split()
    if len(parts) != 3 or parts[1].lower() not in MONTH_MAP_ID:
        return pd.NaT
    d = pd.Timestamp(year=int(parts[2]), month=MONTH_MAP_ID[parts[1].lower()], day=int(parts[0]))
    return (d + pd.Timedelta(hours=int(float(clock)))).tz_localize("UTC")


def fib_levels(o, c, x_up, x_dn):
    bt, bb = max(o, c), min(o, c)
    body = bt - bb
    return bt + (x_up - 1) * body, bb - (x_dn - 1) * body, body


def main():
    print("Fetching OHLC for trade simulation ...")
    start = datetime(2024, 12, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc) + pd.Timedelta(hours=MAX_HOLD + 4)
    ohlc = load_ohlc(SYMBOL, INTERVAL, start, end)
    O = ohlc["open"].values; H = ohlc["high"].values; L = ohlc["low"].values; C = ohlc["close"].values
    idx_of = {pd.Timestamp(t): i for i, t in enumerate(ohlc["datetime"])}
    print(f"  {len(ohlc)} bars {ohlc['datetime'].min()} .. {ohlc['datetime'].max()}")

    eng0 = FibPatternEngineV5()
    raw = eng0._load_workbook(DATA, "ALL")
    rk = raw[list(RANK_COLUMNS.values())].apply(pd.to_numeric, errors="coerce")
    raw = raw.loc[rk.notna().any(axis=1)].reset_index(drop=True)
    feat_ok = raw[FEATURE_CATEGORICAL + FEATURE_NUMERIC].notna().all(axis=1)
    raw = raw.loc[feat_ok].reset_index(drop=True)
    raw["__dt"] = [parse_dt(d, c) for d, c in zip(raw["Date"], raw["Clock"])]
    raw = raw.loc[raw["__dt"].notna()].sort_values("__dt").reset_index(drop=True)

    months = sorted(raw["__dt"].dt.to_period("M").unique())
    test_months = months[MIN_TRAIN_MONTHS:]

    trades = []
    for per in test_months:
        m_start = per.start_time.tz_localize("UTC")
        cut = m_start - pd.Timedelta(hours=GAP_HOURS)
        train = raw.loc[raw["__dt"] < cut]
        test = raw.loc[(raw["__dt"] >= m_start) & (raw["__dt"] < per.end_time.tz_localize("UTC"))]
        if len(train) < 200 or test.empty:
            continue
        eng = FibPatternEngineV5()
        eng.fit_dataframe(train.copy())

        for _, r in test.iterrows():
            dt = r["__dt"]
            i = idx_of.get(dt)
            if i is None or i + 1 + MAX_HOLD >= len(ohlc):
                continue
            setup = {c: r[c] for c in FEATURE_CATEGORICAL}
            for c in FEATURE_NUMERIC:
                setup[c] = float(r[c])
            try:
                res = eng.predict(setup, top_k_matches=1)
            except Exception:
                continue
            sim = res.top_matches[0]["similarity"] if res.top_matches else 0.0
            rp = res.reach_probs
            down = rp.get("1.61_DOWN", 0) + rp.get("2.5_DOWN", 0)
            up = rp.get("1.61_UP", 0) + rp.get("2.5_UP", 0)
            direction = "SHORT" if down > up else "LONG"

            # anchor body fib (TP=2.5) ; combined 2-bar fib (SL=3.6)
            up25, dn25, body = fib_levels(O[i], C[i], 2.5, 2.5)
            cO, cC = O[i], C[i + 1]
            cH = max(H[i], H[i + 1]); cL = min(L[i], L[i + 1])
            cbt, cbb = max(cO, cC), min(cO, cC); cbody = cbt - cbb
            E = C[i + 1]                                  # entry = close of next bar
            if body <= 0 or cbody <= 0:
                continue

            if direction == "SHORT":
                TP = dn25
                SL = {
                    "wick2bar": cH,                                    # combined 2-bar high
                    "fib36_body": cbt + 2.6 * cbody,                   # 3.6 up of combined body
                    "fib36_range": cH + 2.6 * (cH - cL),              # 3.6 up of combined range
                }
            else:
                TP = up25
                SL = {
                    "wick2bar": cL,
                    "fib36_body": cbb - 2.6 * cbody,
                    "fib36_range": cL - 2.6 * (cH - cL),
                }

            for sl_name, SLv in SL.items():
                # validity: entry must sit between TP and SL in the right order
                if direction == "SHORT":
                    if not (SLv > E > TP):
                        continue
                else:
                    if not (SLv < E < TP):
                        continue
                risk = abs(SLv - E); reward = abs(E - TP)
                outcome, exit_px, bars = "TIMEOUT", C[i + 1 + MAX_HOLD], MAX_HOLD
                for k in range(i + 2, i + 2 + MAX_HOLD):
                    hi, lo = H[k], L[k]
                    if direction == "SHORT":
                        hit_tp = lo <= TP; hit_sl = hi >= SLv
                    else:
                        hit_tp = hi >= TP; hit_sl = lo <= SLv
                    if hit_tp and hit_sl:
                        outcome, exit_px, bars = "LOSS", SLv, k - (i + 1); break   # conservative: SL first
                    if hit_tp:
                        outcome, exit_px, bars = "WIN", TP, k - (i + 1); break
                    if hit_sl:
                        outcome, exit_px, bars = "LOSS", SLv, k - (i + 1); break
                # P&L
                if direction == "SHORT":
                    ret = (E - exit_px) / E
                else:
                    ret = (exit_px - E) / E
                ret_net = ret - FEE_RT
                rmult = ((E - exit_px) if direction == "SHORT" else (exit_px - E)) / risk if risk > 0 else 0
                trades.append({
                    "dt": dt, "month": str(per), "dir": direction, "sl": sl_name,
                    "sim": sim, "E": E, "TP": TP, "SL": SLv, "risk_pct": risk / E * 100,
                    "reward_pct": reward / E * 100, "outcome": outcome, "bars": bars,
                    "ret_net": ret_net, "R": rmult,
                })
        print(f"  {per}: trades so far = {len(trades)}")

    T = pd.DataFrame(trades)
    T.to_csv("strategy_trades_v5.csv", index=False)
    print(f"\nSaved {len(T)} simulated trade-legs -> strategy_trades_v5.csv\n")

    for sl_name in ["wick2bar", "fib36_body", "fib36_range"]:
        S = T[T["sl"] == sl_name]
        print("=" * 78)
        print(f"SL variant: {sl_name}   (TP=2.5 anchor body, entry=close anchor+1, fee={FEE_RT:.2%} RT)")
        print("=" * 78)
        print(f"{'sim_gate':>8} | {'trades':>6} | {'win%':>5} | {'loss%':>5} | {'timeout%':>8} | "
              f"{'avgR':>6} | {'exp.R/trade':>11} | {'avg%net':>8} | {'total%':>8}")
        for g in SIM_GATES:
            G = S[S["sim"] >= g]
            if len(G) == 0:
                continue
            n = len(G)
            win = (G["outcome"] == "WIN").mean() * 100
            loss = (G["outcome"] == "LOSS").mean() * 100
            to = (G["outcome"] == "TIMEOUT").mean() * 100
            avgR = G["R"].mean()
            expR = G["R"].mean()
            avgpct = G["ret_net"].mean() * 100
            totpct = G["ret_net"].sum() * 100
            print(f"{g:>8.2f} | {n:>6} | {win:>5.1f} | {loss:>5.1f} | {to:>8.1f} | "
                  f"{avgR:>6.2f} | {expR:>11.3f} | {avgpct:>8.3f} | {totpct:>8.1f}")
        print()

    # headline: user's described setup ~ fib36 SL, gate 0.25
    print("=" * 78)
    for sl_name in ["fib36_range", "fib36_body", "wick2bar"]:
        G = T[(T["sl"] == sl_name) & (T["sim"] >= 0.25)]
        if len(G) == 0:
            continue
        win = (G["outcome"] == "WIN").mean()
        expR = G["R"].mean()
        net = G["ret_net"].mean() * 100
        verdict = "PROFIT" if expR > 0 and net > 0 else "RUGI/break-even"
        print(f"[SL={sl_name}, gate sim>=0.25] n={len(G)} win={win:.1%} exp={expR:+.3f}R "
              f"avg={net:+.3f}%/trade -> {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
