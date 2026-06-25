"""
Honest out-of-sample backtest for FibPatternEngineV5.
=====================================================
Goal: does the engine beat the BASE RATE out-of-sample? No data leakage:
  - Walk-forward by month: train on all rows strictly BEFORE (test month start - 48h gap),
    test on that month. The 48h gap prevents the train tail's lookahead window from
    overlapping test rows.
  - k-NN neighbours come only from past (train) rows.

Metrics (all model-vs-baseline, out-of-sample):
  A) First-hit 8-class accuracy        vs  always-predict-majority (TIE_SAME_BAR)
  B) Reach Brier skill per fib level    vs  train marginal P(reach)   (skill = 1 - brier_model/brier_base; >0 = better)
  C) Direction accuracy (UP vs DOWN first) vs majority direction      (only rows with a clear winner)
  D) Direction accuracy on the model's most-confident decile          (does confidence = edge?)
"""
import sys
import numpy as np
import pandas as pd

from fib_pattern_engine_v5 import (
    FibPatternEngineV5, FEATURE_CATEGORICAL, FEATURE_NUMERIC,
    RANK_COLUMNS, ACTIONABLE_TARGETS, ALL_FIRST_HIT_TARGETS, MONTH_MAP_ID,
)

DATA = "Dataset Output Otomatis 2025-2026.xlsx"
GAP_HOURS = 48
MIN_TRAIN_MONTHS = 6
UP = ["1.61_UP", "2.5_UP", "3.6_UP"]
DN = ["1.61_DOWN", "2.5_DOWN", "3.6_DOWN"]


def parse_dt(date_val, clock):
    if pd.isna(date_val) or pd.isna(clock):
        return pd.NaT
    if isinstance(date_val, pd.Timestamp):
        d = date_val
    else:
        parts = str(date_val).strip().split()
        if len(parts) != 3 or parts[1].lower() not in MONTH_MAP_ID:
            return pd.NaT
        d = pd.Timestamp(year=int(parts[2]), month=MONTH_MAP_ID[parts[1].lower()], day=int(parts[0]))
    return d + pd.Timedelta(hours=int(float(clock)))


def first_hit_label(ranks):
    pos = {k: v for k, v in ranks.items() if v > 0}
    if not pos:
        return "NO_HIT_48H"
    mn = min(pos.values())
    winners = [k for k, v in pos.items() if v == mn]
    return winners[0] if len(winners) == 1 else "TIE_SAME_BAR"


def main():
    eng0 = FibPatternEngineV5()
    raw = eng0._load_workbook(DATA, "ALL")

    # numeric ranks; drop incomplete-window rows (all-blank output)
    rk = raw[list(RANK_COLUMNS.values())].apply(pd.to_numeric, errors="coerce")
    keep = rk.notna().any(axis=1)
    raw = raw.loc[keep].reset_index(drop=True)
    rk = rk.loc[keep].fillna(0).astype(int).reset_index(drop=True)

    # need all features present too
    feat_ok = raw[FEATURE_CATEGORICAL + FEATURE_NUMERIC].notna().all(axis=1)
    raw = raw.loc[feat_ok].reset_index(drop=True)
    rk = rk.loc[feat_ok].reset_index(drop=True)

    raw["__dt"] = [parse_dt(d, c) for d, c in zip(raw["Date"], raw["Clock"])]
    raw = raw.loc[raw["__dt"].notna()].reset_index(drop=True)
    rk = rk.loc[raw.index].reset_index(drop=True)

    # labels
    ranks_by_key = {t: rk[RANK_COLUMNS[t]].values for t in ACTIONABLE_TARGETS}
    fh_label, dir_label, reach = [], [], {t: [] for t in ACTIONABLE_TARGETS}
    for i in range(len(raw)):
        rrank = {t: int(ranks_by_key[t][i]) for t in ACTIONABLE_TARGETS}
        fh_label.append(first_hit_label(rrank))
        up_ranks = [rrank[t] for t in UP if rrank[t] > 0]
        dn_ranks = [rrank[t] for t in DN if rrank[t] > 0]
        mu = min(up_ranks) if up_ranks else 1e9
        md = min(dn_ranks) if dn_ranks else 1e9
        dir_label.append("UP" if mu < md else "DOWN" if md < mu else "TIE")
        for t in ACTIONABLE_TARGETS:
            reach[t].append(int(rrank[t] > 0))
    raw["__fh"] = fh_label
    raw["__dir"] = dir_label
    for t in ACTIONABLE_TARGETS:
        raw["__reach_" + t] = reach[t]

    raw = raw.sort_values("__dt").reset_index(drop=True)
    months = sorted(raw["__dt"].dt.to_period("M").unique())
    test_months = months[MIN_TRAIN_MONTHS:]
    print(f"Total rows: {len(raw)}   months: {len(months)}   test months: {len(test_months)} "
          f"({test_months[0]}..{test_months[-1]})")

    rows_out = []
    for per in test_months:
        m_start = per.start_time
        cut = m_start - pd.Timedelta(hours=GAP_HOURS)
        train = raw.loc[raw["__dt"] < cut]
        test = raw.loc[(raw["__dt"] >= m_start) & (raw["__dt"] < per.end_time)]
        if len(train) < 200 or test.empty:
            continue
        eng = FibPatternEngineV5()
        eng.fit_dataframe(train.copy())
        train_majority_fh = train["__fh"].value_counts().idxmax()
        train_majority_dir = train.loc[train["__dir"] != "TIE", "__dir"].value_counts().idxmax()
        train_reach_marg = {t: train["__reach_" + t].mean() for t in ACTIONABLE_TARGETS}

        for _, r in test.iterrows():
            setup = {c: r[c] for c in FEATURE_CATEGORICAL}
            for c in FEATURE_NUMERIC:
                setup[c] = float(r[c])
            try:
                res = eng.predict(setup, top_k_matches=1)
            except Exception:
                continue
            fhp = res.first_hit_probs
            model_fh = max(ALL_FIRST_HIT_TARGETS, key=lambda k: fhp.get(k, 0.0))
            p_up = sum(fhp.get(k, 0.0) for k in UP)
            p_dn = sum(fhp.get(k, 0.0) for k in DN)
            model_dir = "UP" if p_up >= p_dn else "DOWN"
            conf = abs(p_up - p_dn)
            rec = {
                "dt": r["__dt"], "fh_actual": r["__fh"], "dir_actual": r["__dir"],
                "model_fh": model_fh, "base_fh": train_majority_fh,
                "model_dir": model_dir, "base_dir": train_majority_dir, "conf": conf,
                "trend": str(r["Trend"]), "raw": str(r["Raw Position"]),
            }
            for t in ACTIONABLE_TARGETS:
                rec["pm_" + t] = res.reach_probs.get(t, 0.0)
                rec["pb_" + t] = train_reach_marg[t]
                rec["y_" + t] = int(r["__reach_" + t])
            rows_out.append(rec)
        print(f"  {per}: train={len(train)} test={len(test)} -> cumulative preds={len(rows_out)}")

    R = pd.DataFrame(rows_out)
    print("\n" + "=" * 70)
    print(f"OUT-OF-SAMPLE EVAL  (n={len(R)} predictions)")
    print("=" * 70)

    # A) first-hit 8-class
    accA_m = (R["model_fh"] == R["fh_actual"]).mean()
    accA_b = (R["base_fh"] == R["fh_actual"]).mean()
    print(f"\nA) First-hit 8-class accuracy:")
    print(f"     model     = {accA_m:.3f}")
    print(f"     base-rate = {accA_b:.3f}   (always predict majority class)")
    print(f"     EDGE      = {accA_m - accA_b:+.3f}")

    # B) reach Brier skill
    print(f"\nB) Reach Brier skill per level (>0 = model better than marginal):")
    skills = []
    for t in ACTIONABLE_TARGETS:
        y = R["y_" + t].values
        bm = np.mean((R["pm_" + t].values - y) ** 2)
        bb = np.mean((R["pb_" + t].values - y) ** 2)
        skill = 1 - bm / bb if bb > 0 else 0.0
        skills.append(skill)
        print(f"     {t:<10}: brier_model={bm:.4f} brier_base={bb:.4f}  skill={skill:+.4f}")
    print(f"     mean skill = {np.mean(skills):+.4f}")

    # C) direction (clear-winner rows only)
    clear = R[R["dir_actual"] != "TIE"]
    accC_m = (clear["model_dir"] == clear["dir_actual"]).mean()
    accC_b = (clear["base_dir"] == clear["dir_actual"]).mean()
    print(f"\nC) Direction accuracy (UP vs DOWN first, n={len(clear)} clear-winner rows):")
    print(f"     model     = {accC_m:.3f}")
    print(f"     base-rate = {accC_b:.3f}   (always predict majority direction)")
    print(f"     coin-flip = 0.500")
    print(f"     EDGE vs base = {accC_m - accC_b:+.3f}")

    # D) does confidence help? top-decile by conf
    if len(clear) > 50:
        thr = clear["conf"].quantile(0.9)
        top = clear[clear["conf"] >= thr]
        accD = (top["model_dir"] == top["dir_actual"]).mean()
        print(f"\nD) Direction accuracy on model's most-confident 10% (n={len(top)}):")
        print(f"     model on confident subset = {accD:.3f}  (vs {accC_m:.3f} overall)")

    # E) naive baselines vs model (does the kNN add value beyond 'follow the trend'?)
    naive_trend = np.where(clear["trend"].str.lower() == "long", "UP", "DOWN")
    naive_raw = np.where(clear["raw"].str.upper() == "LONG", "UP",
                         np.where(clear["raw"].str.upper() == "SHORT", "DOWN", clear["base_dir"]))
    accE_trend = (naive_trend == clear["dir_actual"].values).mean()
    accE_raw = (naive_raw == clear["dir_actual"].values).mean()
    print(f"\nE) Naive baselines on clear-winner rows (vs model {accC_m:.3f}):")
    print(f"     follow Trend (MACD>0 -> UP)      = {accE_trend:.3f}")
    print(f"     follow Raw Position (LONG -> UP) = {accE_raw:.3f}")
    print(f"     model EDGE over best naive       = {accC_m - max(accE_trend, accE_raw):+.3f}")

    # F) per-month consistency of the direction edge
    print(f"\nF) Per-month direction accuracy (model vs coin-flip 0.5):")
    clear = clear.copy()
    clear["month"] = clear["dt"].dt.to_period("M").astype(str)
    wins = 0
    for mo, g in clear.groupby("month"):
        acc = (g["model_dir"] == g["dir_actual"]).mean()
        nt = (np.where(g["trend"].str.lower() == "long", "UP", "DOWN") == g["dir_actual"].values).mean()
        flag = "OK " if acc > 0.5 else "<<<"
        wins += acc > 0.5
        print(f"     {mo}: model={acc:.3f}  naive_trend={nt:.3f}  n={len(g):4d}  {flag}")
    print(f"     months where model>0.5: {wins}/{clear['month'].nunique()}")

    # standard error / significance for the headline direction number
    n = len(clear)
    se = (accC_m * (1 - accC_m) / n) ** 0.5
    z = (accC_m - 0.5) / se
    print(f"\n  Direction: {accC_m:.3f}  (n={n}, SE={se:.4f}, z vs 0.5 = {z:.1f})")

    R.to_csv("backtest_v5_predictions.csv", index=False)
    print("  saved -> backtest_v5_predictions.csv")

    print("\n" + "=" * 70)
    real_dir_edge = (accC_m - max(accE_trend, accE_raw) > 0.01) and wins >= int(0.6 * clear["month"].nunique())
    edge = real_dir_edge or (np.mean(skills) > 0.02) or (accA_m - accA_b > 0.02)
    print("VERDICT:", "Ada edge ARAH yang nyata & konsisten (di atas baseline naif)." if real_dir_edge else
          ("Ada sinyal arah tapi TIDAK jelas mengalahkan 'ikut tren'." if accC_m > 0.53 else
           "TIDAK ada edge berarti vs base-rate."))
    print("=" * 70)


if __name__ == "__main__":
    main()
