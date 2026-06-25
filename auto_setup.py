"""
Auto-fetch setup lengkap (Bar 1/2 + SQZMOM 1/2 + Score/Posisi/Last TR +
indikator diagnostic) untuk ticker+date+hour. Menjalankan sqzmom_export.py
sebagai subprocess lalu membaca xlsx hasilnya.

Single source of truth = Binance, supaya match dengan dataset training
(yang juga dibangun dari sqzmom_export.py).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone, date as date_cls
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# Lokasi sqzmom_export.py — di folder yang sama dengan auto_setup.py
# (di-bundle bersama app supaya deployment Streamlit Cloud tetap jalan)
SQZMOM_EXPORT_PATH = Path(__file__).resolve().parent / "sqzmom_export.py"


def _yahoo_to_binance(ticker: str) -> str:
    t = ticker.strip().upper()
    if t.endswith("-USD"):
        return t.replace("-USD", "") + "USDT"
    if t.endswith("USDT"):
        return t
    return t + "USDT"


def fetch_setup(
    ticker: str,
    target_date: date_cls,
    target_hour: int,
    interval: str = "1h",
    warmup_days: int = 90,
) -> Dict[str, object]:
    """Return dict with Bar 1/2 + SQZMOM 1/2 fields (or 'error' key).

    Calls sqzmom_export.py as subprocess (idle ~5-10s for Binance fetch).
    """
    if not SQZMOM_EXPORT_PATH.exists():
        return {"error": f"sqzmom_export.py tidak ditemukan di {SQZMOM_EXPORT_PATH}"}

    symbol = _yahoo_to_binance(ticker)
    target_dt_utc = datetime(target_date.year, target_date.month, target_date.day,
                             target_hour, 0, tzinfo=timezone.utc)
    target_dt_naive = target_dt_utc.replace(tzinfo=None)
    fetch_start = (target_dt_utc - timedelta(days=warmup_days)).strftime("%Y-%m-%dT%H:%M")
    fetch_end = (target_dt_utc + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / f"{symbol}_{interval}_setup.xlsx"
        cmd = [
            sys.executable, str(SQZMOM_EXPORT_PATH),
            "--symbol", symbol,
            "--interval", interval,
            "--start", fetch_start,
            "--end", fetch_end,
            "--out", str(out_path),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return {"error": "Timeout saat fetch data (>120s)"}
        if res.returncode != 0 or not out_path.exists():
            return {"error": f"sqzmom_export gagal: {res.stderr[-400:] or res.stdout[-400:]}"}

        df = pd.read_excel(out_path)

    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
    df = df.sort_values("Datetime (UTC)").reset_index(drop=True)

    cur = df[df["Datetime (UTC)"] == target_dt_naive]
    if cur.empty:
        return {"error": f"Bar untuk {target_dt_utc:%Y-%m-%d %H:%M} UTC tidak tersedia di Binance"}
    cur_idx = int(cur.index[0])
    if cur_idx == 0:
        return {"error": "Tidak ada bar sebelumnya untuk Bar 2 / SQZMOM 2"}
    prev = df.iloc[cur_idx - 1]
    cur = df.iloc[cur_idx]

    def _bar_label(row) -> Optional[str]:
        c, z = row.get("Bar Color"), row.get("Fib Zone")
        if pd.isna(c) or pd.isna(z):
            return None
        return f"{c} Bar Line {int(z)}"

    def _opt_float(row, col: str) -> Optional[float]:
        if col not in row or pd.isna(row[col]):
            return None
        return float(row[col])

    def _opt_str(row, col: str) -> Optional[str]:
        if col not in row or pd.isna(row[col]):
            return None
        return str(row[col])

    def _filter_reason(row) -> str:
        # Sinkron dengan logika di sqzmom_export.py: ADX < 15 atau Last TR > 1.6 * ATR
        reasons = []
        adx = row.get("ADX 14")
        atr = row.get("ATR 14")
        last_tr = row.get("Last TR")
        if adx is not None and not pd.isna(adx) and adx < 15:
            reasons.append("ADX < 15")
        if (atr is not None and not pd.isna(atr) and atr > 0
                and last_tr is not None and not pd.isna(last_tr)
                and last_tr > 1.6 * atr):
            reasons.append("Last TR > 1.6 * ATR (spike)")
        return "; ".join(reasons) if reasons else "-"

    setup = {
        # Fitur engine — Bar + SQZMOM (sumber: Binance)
        "Bar 1": _bar_label(cur),
        "Bar 2": _bar_label(prev),
        "SQZMOM 1 Value": _opt_float(cur, "SQZMOM Value"),
        "SQZMOM 1 Momentum": _opt_str(cur, "Momentum Color"),
        "SQZMOM 1 Squeeze": _opt_str(cur, "Squeeze Status"),
        "SQZMOM 2 Value": _opt_float(prev, "SQZMOM Value"),
        "SQZMOM 2 Momentum": _opt_str(prev, "Momentum Color"),
        "SQZMOM 2 Squeeze": _opt_str(prev, "Squeeze Status"),
        # Score / posisi (Binance, uppercase — match dataset training)
        "Score": int(cur["Score"]) if not pd.isna(cur["Score"]) else None,
        "Last TR": _opt_float(cur, "Last TR"),
        "Raw Position": _opt_str(cur, "Raw Posisi"),
        "Final Position": _opt_str(cur, "Posisi Final"),
        # Diagnostic untuk panel "Detail Indikator Market"
        "last_close": _opt_float(cur, "Close"),
        "rsi_last": _opt_float(cur, "RSI 14"),
        "adx_last": _opt_float(cur, "ADX 14"),
        "atr_last": _opt_float(cur, "ATR 14"),
        "ema_fast_last": _opt_float(cur, "EMA 21"),
        "ema_slow_last": _opt_float(cur, "EMA 50"),
        "macd_last": _opt_float(cur, "MACD"),
        "filter_reason": _filter_reason(cur),
        # Untuk informasi/debug
        "_target_dt": str(target_dt_utc),
        "_close": _opt_float(cur, "Close"),
    }

    required = ("Bar 1", "Bar 2", "SQZMOM 1 Value", "SQZMOM 2 Value",
                "Score", "Last TR", "Raw Position", "Final Position")
    missing = [k for k in required if setup.get(k) is None]
    if missing:
        setup["error"] = f"Field tidak lengkap (warmup belum cukup?): {missing}"
    return setup


if __name__ == "__main__":
    import json
    # Smoke test
    out = fetch_setup("ETH-USD", date_cls(2026, 5, 15), 10)
    print(json.dumps(out, indent=2, default=str))
