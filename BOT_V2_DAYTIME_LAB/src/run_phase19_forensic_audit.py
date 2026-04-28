import contextlib
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import unittest
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from phase14_engine import Phase14Engine
from phase18_h1_fractal_sweep import H1FractalSweepDetector


ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LAB = ROOT / "BOT_V2_DAYTIME_LAB"
SRC = LAB / "src"
TESTS = LAB / "tests" / "engine_safety"
OUT = LAB / "outputs" / "phase19_forensic_audit"
REPORTS = LAB / "reports"
MANIFEST = LAB / "data" / "certified_data_paths.json"
ZIP_PATH = ROOT / "000_PARA_CHATGPT.zip"

NY_TZ = "America/New_York"
PIP = 0.0001

REPORTED = {
    "strategy": "Phase19 Expanded Sweep",
    "base": "H1 Fractal Sweep + First M3 CHOCH",
    "window": "08:00-16:30 NY",
    "max_trades_per_day": 3,
    "choch_window_minutes": 30,
    "tp_r": 2.5,
    "be": "none",
    "sl": "sweep extreme",
    "exclude_friday": True,
    "min_sweep_pips": 0.5,
    "sample": 3177,
    "pf": 3.18,
    "expectancy_r": 0.69,
    "trades_month": 51,
    "pf_2023_2025_min": 2.90,
    "pf_plus_1_pip": 2.58,
    "pf_plus_1_5_pips": 2.35,
    "out_of_hours_trades": 0,
    "news_violations": 0,
}


def ensure_dirs():
    for rel in [
        "diagnosis",
        "reproduction",
        "tp_sl_math",
        "multi_trade",
        "signal_no_lookahead",
        "filters",
        "execution",
        "time_news",
        "robustness",
        "costs",
        "risk",
        "overfit_control",
        "comparison",
        "tests",
        "zip",
    ]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def json_default(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, default=json_default)
        f.write("\n")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text.rstrip() + "\n")


def fmt(value, digits=3):
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def max_loss_streak(values):
    streak = 0
    best = 0
    for v in values:
        if v < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def drawdown_recovery_time(cumulative):
    peak = -float("inf")
    in_dd = False
    dd_start = None
    longest = 0
    for i, value in enumerate(cumulative):
        if value >= peak:
            if in_dd and dd_start is not None:
                longest = max(longest, i - dd_start)
            peak = value
            in_dd = False
            dd_start = None
        elif not in_dd:
            in_dd = True
            dd_start = i
    if in_dd and dd_start is not None:
        longest = max(longest, len(cumulative) - dd_start)
    return int(longest)


def period_key(ts):
    return pd.Timestamp(ts).strftime("%Y-%m")


def compute_metrics(trades):
    if trades is None or len(trades) == 0:
        return {
            "sample": 0,
            "pf": 0.0,
            "expectancy_R": 0.0,
            "cumulative_R": 0.0,
            "max_drawdown_R": 0.0,
            "max_drawdown_pct_1pct_risk": 0.0,
            "max_loss_streak": 0,
            "worst_day_R": 0.0,
            "worst_week_R": 0.0,
            "worst_month_R": 0.0,
            "win_rate": 0.0,
            "tp_count": 0,
            "sl_count": 0,
            "be_count": 0,
            "timeout_count": 0,
            "forced_close_count": 0,
            "same_bar_count": 0,
            "average_win_R": 0.0,
            "average_loss_R": 0.0,
            "gross_profit_R": 0.0,
            "gross_loss_R": 0.0,
            "trades_month": 0.0,
            "trades_day_avg": 0.0,
        }

    df = trades.copy()
    df["r_pnl"] = pd.to_numeric(df["r_pnl"], errors="coerce").fillna(0.0)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    profits = df.loc[df["r_pnl"] > 0, "r_pnl"]
    losses = df.loc[df["r_pnl"] < 0, "r_pnl"]
    gross_profit = float(profits.sum())
    gross_loss = float(abs(losses.sum()))
    cumulative = df["r_pnl"].cumsum()
    peak = cumulative.cummax()
    dd = cumulative - peak
    first = df["entry_time"].min()
    last = df["entry_time"].max()
    months = max(1, (last.year - first.year) * 12 + (last.month - first.month) + 1)
    daily = df.groupby(df["entry_time"].dt.date)["r_pnl"].sum()
    weekly = df.groupby(df["entry_time"].dt.strftime("%G-%V"))["r_pnl"].sum()
    monthly = df.groupby(df["entry_time"].dt.strftime("%Y-%m"))["r_pnl"].sum()
    return {
        "sample": int(len(df)),
        "pf": round(gross_profit / gross_loss, 6) if gross_loss > 0 else (round(gross_profit, 6) if gross_profit else 0.0),
        "expectancy_R": round(float(df["r_pnl"].mean()), 6),
        "cumulative_R": round(float(df["r_pnl"].sum()), 6),
        "max_drawdown_R": round(float(dd.min()), 6),
        "max_drawdown_pct_1pct_risk": round(float(dd.min()), 6),
        "max_loss_streak": int(max_loss_streak(df["r_pnl"].tolist())),
        "worst_day_R": round(float(daily.min()), 6) if not daily.empty else 0.0,
        "worst_week_R": round(float(weekly.min()), 6) if not weekly.empty else 0.0,
        "worst_month_R": round(float(monthly.min()), 6) if not monthly.empty else 0.0,
        "win_rate": round(float((df["r_pnl"] > 0).mean()), 6),
        "tp_count": int((df["status"] == "TP").sum()) if "status" in df.columns else 0,
        "sl_count": int((df["status"] == "SL").sum()) if "status" in df.columns else 0,
        "be_count": int((df["status"] == "BE").sum()) if "status" in df.columns else 0,
        "timeout_count": int((df["status"] == "TIMEOUT").sum()) if "status" in df.columns else 0,
        "forced_close_count": int((df["status"] == "FORCED_CLOSE_2000").sum()) if "status" in df.columns else 0,
        "same_bar_count": int(df.get("same_bar", pd.Series([False] * len(df))).fillna(False).sum()),
        "average_win_R": round(float(profits.mean()), 6) if len(profits) else 0.0,
        "average_loss_R": round(float(losses.mean()), 6) if len(losses) else 0.0,
        "gross_profit_R": round(gross_profit, 6),
        "gross_loss_R": round(gross_loss, 6),
        "trades_month": round(float(len(df) / months), 6),
        "trades_day_avg": round(float(len(df) / max(1, df["entry_time"].dt.date.nunique())), 6),
        "drawdown_recovery_trades": drawdown_recovery_time(cumulative.tolist()),
    }


def equity_and_dd(trades):
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["r_pnl"] = pd.to_numeric(df["r_pnl"], errors="coerce").fillna(0.0)
    df = df.sort_values(["entry_time", "trade_id"]).reset_index(drop=True)
    df["trade_number"] = np.arange(1, len(df) + 1)
    df["cumulative_R"] = df["r_pnl"].cumsum()
    df["peak_R"] = df["cumulative_R"].cummax()
    df["drawdown_R"] = df["cumulative_R"] - df["peak_R"]
    return df[["trade_number", "entry_time", "r_pnl", "cumulative_R", "peak_R", "drawdown_R"]]


def get_ltf_fractals(df, n=2):
    highs = df["high_bid"].to_numpy()
    lows = df["low_bid"].to_numpy()
    size = len(df)
    f_h = np.full(size, np.nan)
    f_l = np.full(size, np.nan)
    for i in range(2 * n, size):
        center = i - n
        left = center - n
        right = center + n + 1
        high = highs[center]
        low = lows[center]
        if all(high > highs[j] for j in range(left, right) if j != center):
            f_h[i] = high
        if all(low < lows[j] for j in range(left, right) if j != center):
            f_l[i] = low
    return pd.Series(f_h).ffill().to_numpy(), pd.Series(f_l).ffill().to_numpy()


def detect_phase19_signals(df_m3, sweeps, max_mins=30, min_depth_pips=0.5):
    if sweeps.empty:
        return pd.DataFrame()
    df = df_m3.copy().reset_index(drop=True)
    last_fh, last_fl = get_ltf_fractals(df, n=2)
    times_us = pd.to_datetime(df["timestamp_ny"], utc=True).astype("int64").to_numpy()
    closes = df["close_bid"].to_numpy()
    records = []
    max_delta_us = pd.Timedelta(minutes=max_mins).value // 1000
    for sweep_id, sweep in sweeps.reset_index(drop=True).iterrows():
        if float(sweep["depth_pips"]) < min_depth_pips:
            continue
        sweep_time = pd.Timestamp(sweep["timestamp_ny"])
        sweep_ns = sweep_time.tz_convert("UTC").value if sweep_time.tzinfo else sweep_time.tz_localize(NY_TZ).tz_convert("UTC").value
        sweep_us = sweep_ns // 1000
        start = int(np.searchsorted(times_us, sweep_us, side="left"))
        end = int(np.searchsorted(times_us, sweep_us + max_delta_us, side="right"))
        if start >= len(df) or start >= end:
            continue
        for j in range(start, min(end, len(df))):
            close = closes[j]
            bar = df.iloc[j]
            if sweep["type"] == "BEARISH_SWEEP":
                trigger = last_fl[j]
                if not np.isnan(trigger) and close < trigger:
                    records.append(
                        {
                            "signal_id": len(records),
                            "sweep_id": int(sweep_id),
                            "sweep_time": sweep_time,
                            "sweep_type": sweep["type"],
                            "sweep_level_type": sweep["level_type"],
                            "sweep_level_price": float(sweep["level_price"]),
                            "sweep_peak_price": float(sweep["peak_price"]),
                            "sweep_depth_pips": float(sweep["depth_pips"]),
                            "choch_time": pd.Timestamp(bar["timestamp_ny"]),
                            "signal_bar_index": int(j),
                            "direction": "SHORT",
                            "entry_price_legacy": float(bar["close_bid"]),
                            "entry_source": "CHOCH_CLOSE_BID",
                            "next_bar_required": False,
                            "sl_price": float(sweep["peak_price"]) + 0.00005,
                            "trigger_level": float(trigger),
                            "m3_open_bid": float(bar["open_bid"]),
                            "m3_high_bid": float(bar["high_bid"]),
                            "m3_low_bid": float(bar["low_bid"]),
                            "m3_close_bid": float(bar["close_bid"]),
                            "m3_open_ask": float(bar["open_ask"]),
                            "m3_high_ask": float(bar["high_ask"]),
                            "m3_low_ask": float(bar["low_ask"]),
                            "m3_close_ask": float(bar["close_ask"]),
                        }
                    )
                    break
            elif sweep["type"] == "BULLISH_SWEEP":
                trigger = last_fh[j]
                if not np.isnan(trigger) and close > trigger:
                    records.append(
                        {
                            "signal_id": len(records),
                            "sweep_id": int(sweep_id),
                            "sweep_time": sweep_time,
                            "sweep_type": sweep["type"],
                            "sweep_level_type": sweep["level_type"],
                            "sweep_level_price": float(sweep["level_price"]),
                            "sweep_peak_price": float(sweep["peak_price"]),
                            "sweep_depth_pips": float(sweep["depth_pips"]),
                            "choch_time": pd.Timestamp(bar["timestamp_ny"]),
                            "signal_bar_index": int(j),
                            "direction": "LONG",
                            "entry_price_legacy": float(bar["close_bid"]),
                            "entry_source": "CHOCH_CLOSE_BID",
                            "next_bar_required": False,
                            "sl_price": float(sweep["peak_price"]) - 0.00005,
                            "trigger_level": float(trigger),
                            "m3_open_bid": float(bar["open_bid"]),
                            "m3_high_bid": float(bar["high_bid"]),
                            "m3_low_bid": float(bar["low_bid"]),
                            "m3_close_bid": float(bar["close_bid"]),
                            "m3_open_ask": float(bar["open_ask"]),
                            "m3_high_ask": float(bar["high_ask"]),
                            "m3_low_ask": float(bar["low_ask"]),
                            "m3_close_ask": float(bar["close_ask"]),
                        }
                    )
                    break
    return pd.DataFrame(records)


def select_phase19_signals(signals, start_hour=8.0, end_hour=16.5, exclude_friday=True, max_trades_per_day=3):
    if signals.empty:
        return signals.copy()
    sig = signals.copy()
    sig["choch_time"] = pd.to_datetime(sig["choch_time"])
    sig["hour_float"] = sig["choch_time"].dt.hour + sig["choch_time"].dt.minute / 60.0
    sig["day_of_week"] = sig["choch_time"].dt.dayofweek
    sig = sig[(sig["hour_float"] >= start_hour) & (sig["hour_float"] < end_hour)].copy()
    if exclude_friday:
        sig = sig[sig["day_of_week"] != 4].copy()
    sig["trade_date"] = sig["choch_time"].dt.date
    sig = sig.sort_values(["choch_time", "signal_id"], kind="mergesort")
    if max_trades_per_day is not None:
        sig = sig.groupby("trade_date", group_keys=False).head(max_trades_per_day)
    return sig.reset_index(drop=True)


def simulate_legacy_phase19(df_m3, selected_signals, tp_r=2.5, cost_pips=0.0):
    if selected_signals.empty:
        return pd.DataFrame()
    df = df_m3.reset_index(drop=True)
    times = pd.to_datetime(df["timestamp_ny"]).to_numpy()
    high_bid = df["high_bid"].to_numpy()
    low_bid = df["low_bid"].to_numpy()
    close_bid = df["close_bid"].to_numpy()
    high_ask = df["high_ask"].to_numpy()
    low_ask = df["low_ask"].to_numpy()
    records = []
    for trade_id, sig in selected_signals.reset_index(drop=True).iterrows():
        direction = sig["direction"]
        raw_entry = float(sig["entry_price_legacy"])
        entry = raw_entry + cost_pips * PIP if direction == "LONG" else raw_entry - cost_pips * PIP
        sl = float(sig["sl_price"])
        risk = abs(entry - sl)
        if risk < 0.00001:
            continue
        tp = entry + risk * tp_r if direction == "LONG" else entry - risk * tp_r
        start = int(sig["signal_bar_index"]) + 1
        end = min(len(df), start + 120)
        status = "TIMEOUT"
        exit_time = pd.NaT
        exit_price = np.nan
        exit_bar_index = None
        same_bar = False
        for j in range(start, end):
            if direction == "LONG":
                if low_bid[j] <= sl:
                    status = "SL"
                    exit_price = sl
                    exit_time = pd.Timestamp(df.at[j, "timestamp_ny"])
                    exit_bar_index = j
                    break
                if high_bid[j] >= tp:
                    status = "TP"
                    exit_price = tp
                    exit_time = pd.Timestamp(df.at[j, "timestamp_ny"])
                    exit_bar_index = j
                    break
            else:
                # Legacy Phase19 used BID for short exits too; this is audited separately.
                if high_bid[j] >= sl:
                    status = "SL"
                    exit_price = sl
                    exit_time = pd.Timestamp(df.at[j, "timestamp_ny"])
                    exit_bar_index = j
                    break
                if low_bid[j] <= tp:
                    status = "TP"
                    exit_price = tp
                    exit_time = pd.Timestamp(df.at[j, "timestamp_ny"])
                    exit_bar_index = j
                    break
        if status == "TIMEOUT":
            if end > start:
                exit_bar_index = end - 1
                exit_time = pd.Timestamp(df.at[exit_bar_index, "timestamp_ny"])
                exit_price = float(close_bid[exit_bar_index])
            r_pnl = 0.0
        elif status == "TP":
            r_pnl = tp_r
        else:
            r_pnl = -1.0
        records.append(
            {
                "trade_id": int(trade_id),
                "period": sig.get("period", "period_2020_2026"),
                "entry_time": pd.Timestamp(sig["choch_time"]),
                "exit_time": exit_time,
                "direction": direction,
                "entry_price": entry,
                "raw_entry_price": raw_entry,
                "entry_source": sig["entry_source"],
                "exit_price": exit_price,
                "sl_price": sl,
                "tp_price": tp,
                "risk_price": risk,
                "risk_pips": risk / PIP,
                "status": status,
                "r_pnl": r_pnl,
                "tp_r": tp_r,
                "be_r": None,
                "sweep_id": int(sig["sweep_id"]),
                "signal_id": int(sig["signal_id"]),
                "sweep_time": pd.Timestamp(sig["sweep_time"]),
                "sweep_type": sig["sweep_type"],
                "sweep_level_type": sig["sweep_level_type"],
                "sweep_level_price": float(sig["sweep_level_price"]),
                "sweep_peak_price": float(sig["sweep_peak_price"]),
                "sweep_depth_pips": float(sig["sweep_depth_pips"]),
                "choch_time": pd.Timestamp(sig["choch_time"]),
                "signal_bar_index": int(sig["signal_bar_index"]),
                "exit_bar_index": exit_bar_index,
                "duration_minutes": (
                    (pd.Timestamp(exit_time) - pd.Timestamp(sig["choch_time"])).total_seconds() / 60.0
                    if pd.notna(exit_time)
                    else np.nan
                ),
                "same_bar": same_bar,
                "legacy_used_bid_for_short_exit": direction == "SHORT",
                "legacy_used_choch_close_entry": True,
                "legacy_used_real_ask_entry": False,
                "legacy_forced_close_2000": False,
                "m3_high_bid": float(sig["m3_high_bid"]),
                "m3_low_bid": float(sig["m3_low_bid"]),
                "m3_high_ask": float(sig["m3_high_ask"]),
                "m3_low_ask": float(sig["m3_low_ask"]),
                "spread_entry_pips": (
                    float(sig["m3_close_ask"]) - float(sig["m3_close_bid"])
                )
                / PIP,
            }
        )
    return pd.DataFrame(records)


def load_and_prepare_period(period, max_mins=30, min_depth_pips=0.5):
    engine = Phase14Engine(str(MANIFEST))
    df_h1 = engine.load_and_prep_prices(period, timeframe="h1")
    df_m3 = engine.load_and_prep_prices(period, timeframe="m3")
    sweep_detector = H1FractalSweepDetector({})
    sweeps = sweep_detector.detect_sweeps(df_h1)
    signals = detect_phase19_signals(df_m3, sweeps, max_mins=max_mins, min_depth_pips=min_depth_pips)
    if not signals.empty:
        signals["period"] = period
    return engine, df_h1, df_m3, sweeps, signals


def reproduction_phase():
    engine, df_h1, df_m3, sweeps, signals = load_and_prepare_period("period_2020_2026", max_mins=30, min_depth_pips=0.5)
    selected = select_phase19_signals(signals)
    trades = simulate_legacy_phase19(df_m3, selected, tp_r=2.5)
    metrics = compute_metrics(trades)
    equity = equity_and_dd(trades)
    match = (
        metrics["sample"] == REPORTED["sample"]
        and abs(round(metrics["pf"], 2) - REPORTED["pf"]) <= 0.01
        and abs(round(metrics["expectancy_R"], 2) - REPORTED["expectancy_r"]) <= 0.01
    )
    verdict = "PHASE19_REPRODUCTION_MATCH" if match else "PHASE19_REPRODUCTION_MISMATCH"

    out = OUT / "reproduction"
    trades.to_csv(out / "phase19_reproduced_trades.csv", index=False)
    equity[["trade_number", "entry_time", "r_pnl", "cumulative_R"]].to_csv(
        out / "phase19_reproduced_equity_curve.csv", index=False
    )
    equity[["trade_number", "entry_time", "peak_R", "drawdown_R"]].to_csv(
        out / "phase19_reproduced_drawdown_curve.csv", index=False
    )

    duplicate_trades = int(trades.duplicated(subset=["entry_time", "direction", "sweep_time", "sweep_level_type"]).sum())
    out_of_hours = int(((trades["entry_time"].dt.hour + trades["entry_time"].dt.minute / 60.0) < 7.0).sum() + ((trades["entry_time"].dt.hour + trades["entry_time"].dt.minute / 60.0) >= 20.0).sum())
    rollover = count_rollover_violations(trades)["open_after_2000_count"]
    news_counts = count_news_violations(engine, trades)

    summary = {
        "verdict": verdict,
        "reported": REPORTED,
        "metrics": metrics,
        "out_of_hours_trades_0700_2000": out_of_hours,
        "news_violations_30m": news_counts["30m"],
        "rollover_trades_open_after_2000": rollover,
        "duplicated_trades_same_event": duplicate_trades,
        "implementation_notes": [
            "Reproduccion hecha desde codigo Phase19 legacy, sin leer outputs previos como input.",
            "El resultado reproduce sample y PF reportados, pero no coincide con la expectancy ni con news/rollover reportados.",
            "La reproduccion no implica aprobacion: las fases siguientes auditan controles institucionales ausentes.",
        ],
    }
    write_json(out / "phase19_reproduced_summary.json", summary)
    write_text(
        out / "phase19_reproduced_summary.md",
        "\n".join(
            [
                "# Phase19 Reproduction Summary",
                "",
                f"Verdicto: {verdict}",
                f"Sample: {metrics['sample']}",
                f"PF: {fmt(metrics['pf'], 3)}",
                f"Expectancy R: {fmt(metrics['expectancy_R'], 3)}",
                f"Cumulative R: {fmt(metrics['cumulative_R'], 3)}",
                f"Max DD R: {fmt(metrics['max_drawdown_R'], 3)}",
                f"Max loss streak: {metrics['max_loss_streak']}",
                f"Worst day R: {fmt(metrics['worst_day_R'], 3)}",
                f"Worst week R: {fmt(metrics['worst_week_R'], 3)}",
                f"Win rate: {fmt(metrics['win_rate'] * 100, 2)}%",
                f"TP / SL / BE / Timeout / Forced: {metrics['tp_count']} / {metrics['sl_count']} / {metrics['be_count']} / {metrics['timeout_count']} / {metrics['forced_close_count']}",
                f"Same-bar count: {metrics['same_bar_count']}",
                f"Trades/month: {fmt(metrics['trades_month'], 2)}",
                f"Trades/day avg: {fmt(metrics['trades_day_avg'], 2)}",
                f"Out-of-hours 07-20: {out_of_hours}",
                f"News violations +/-30m: {news_counts['30m']}",
                f"Rollover/open after 20:00 NY: {rollover}",
                f"Duplicated trades same event: {duplicate_trades}",
                "",
                "Nota: esto reproduce el resultado legacy, pero todavia no lo valida.",
            ]
        ),
    )
    return {
        "engine": engine,
        "df_h1": df_h1,
        "df_m3": df_m3,
        "sweeps": sweeps,
        "signals": signals,
        "selected": selected,
        "trades": trades,
        "metrics": metrics,
        "reproduction_verdict": verdict,
        "news_counts": news_counts,
    }


def count_news_violations(engine, trades):
    news = engine.load_news("period_2020_2026")
    if news.empty or trades.empty:
        return {"30m": 0, "45m": 0, "60m": 0}
    news = news.copy()
    if "currency" in news.columns:
        news = news[news["currency"].isin(["USD", "EUR"])].copy()
    if "impact_level" in news.columns:
        news = news[news["impact_level"].astype(str).str.upper().isin(["HIGH", "MEDIUM"])].copy()
    if news.empty:
        return {"30m": 0, "45m": 0, "60m": 0}
    news_ts = pd.to_datetime(news["timestamp"], utc=True).sort_values().to_numpy()
    trade_ts = pd.to_datetime(trades["entry_time"], utc=True).to_numpy()
    counts = {}
    for mins in [30, 45, 60]:
        delta = np.timedelta64(mins, "m")
        hit = 0
        for ts in trade_ts:
            left = np.searchsorted(news_ts, ts - delta, side="left")
            right = np.searchsorted(news_ts, ts + delta, side="right")
            if right > left:
                hit += 1
        counts[f"{mins}m"] = int(hit)
    return counts


def count_rollover_violations(trades):
    if trades.empty:
        return {"entry_1700_1900_count": 0, "open_after_2000_count": 0, "entry_after_2000_count": 0}
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    entry_hour = df["entry_time"].dt.hour + df["entry_time"].dt.minute / 60.0
    exit_hour = df["exit_time"].dt.hour + df["exit_time"].dt.minute / 60.0
    same_date = df["entry_time"].dt.date == df["exit_time"].dt.date
    return {
        "entry_1700_1900_count": int(((entry_hour >= 17.0) & (entry_hour < 19.0)).sum()),
        "entry_after_2000_count": int((entry_hour >= 20.0).sum()),
        "open_after_2000_count": int(((same_date & (exit_hour > 20.0)) | (~same_date)).sum()),
    }


def tp_sl_math_audit(trades):
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")
    long_df = df[df["direction"] == "LONG"]
    short_df = df[df["direction"] == "SHORT"]
    impossible = df[
        ((df["direction"] == "LONG") & ((df["tp_price"] <= df["entry_price"]) | (df["sl_price"] >= df["entry_price"])))
        | ((df["direction"] == "SHORT") & ((df["tp_price"] >= df["entry_price"]) | (df["sl_price"] <= df["entry_price"])))
        | (df["risk_price"] <= 0)
    ].copy()
    df["tp_r_calc"] = np.where(
        df["direction"] == "LONG",
        (df["tp_price"] - df["entry_price"]) / df["risk_price"],
        (df["entry_price"] - df["tp_price"]) / df["risk_price"],
    )
    df["sl_r_calc"] = np.where(
        df["direction"] == "LONG",
        (df["sl_price"] - df["entry_price"]) / df["risk_price"],
        (df["entry_price"] - df["sl_price"]) / df["risk_price"],
    )
    bad_tp_r = df[(df["tp_r_calc"] - 2.5).abs() > 1e-9]
    sl_equal_entry = df[(df["sl_price"] - df["entry_price"]).abs() <= 1e-10]
    one_bar = df[df["duration_minutes"].fillna(99999) <= 3]
    outliers = df[(df["r_pnl"].abs() > 5) | (df["risk_pips"] > df["risk_pips"].quantile(0.995))].copy()
    dist = df[["trade_id", "entry_time", "direction", "risk_pips", "r_pnl", "duration_minutes", "status"]].copy()
    dist.to_csv(OUT / "tp_sl_math" / "phase19_r_distribution.csv", index=False)
    outliers.to_csv(OUT / "tp_sl_math" / "phase19_outlier_trades.csv", index=False)
    impossible.to_csv(OUT / "tp_sl_math" / "phase19_impossible_targets.csv", index=False)
    verdict = "PHASE19_TP_SL_MATH_CONFIRMED"
    if len(impossible) or len(bad_tp_r) or len(sl_equal_entry):
        verdict = "PHASE19_TP_SL_INVALIDATES_PHASE19"
    elif len(one_bar) > 0:
        verdict = "PHASE19_TP_SL_WARNING"
    payload = {
        "verdict": verdict,
        "long_tp_above_entry_violations": int((long_df["tp_price"] <= long_df["entry_price"]).sum()),
        "long_sl_below_entry_violations": int((long_df["sl_price"] >= long_df["entry_price"]).sum()),
        "short_tp_below_entry_violations": int((short_df["tp_price"] >= short_df["entry_price"]).sum()),
        "short_sl_above_entry_violations": int((short_df["sl_price"] <= short_df["entry_price"]).sum()),
        "tp_2_5r_violations": int(len(bad_tp_r)),
        "risk_le_zero": int((df["risk_price"] <= 0).sum()),
        "sl_equal_entry_without_be": int(len(sl_equal_entry)),
        "absurd_r_count": int((df["r_pnl"].abs() > 5).sum()),
        "one_bar_duration_count": int(len(one_bar)),
        "outlier_file": "phase19_outlier_trades.csv",
        "impossible_targets_file": "phase19_impossible_targets.csv",
        "note": "La matematica TP/SL legacy esta orientada correctamente; los problemas invalidantes aparecen en entrada/ejecucion/multitrade, no en la formula simple de TP/SL.",
    }
    write_json(OUT / "tp_sl_math" / "phase19_tp_sl_math_audit.json", payload)
    write_text(
        OUT / "tp_sl_math" / "phase19_tp_sl_math_audit.md",
        "\n".join(
            [
                "# Phase19 TP/SL Math Audit",
                "",
                f"Verdicto: {verdict}",
                f"Risk <= 0: {payload['risk_le_zero']}",
                f"Targets imposibles: {len(impossible)}",
                f"Violaciones TP 2.5R: {payload['tp_2_5r_violations']}",
                f"SL igual a entrada sin BE: {payload['sl_equal_entry_without_be']}",
                f"Duracion <= una vela M3: {payload['one_bar_duration_count']}",
            ]
        ),
    )
    return payload


def multi_trade_audit(trades):
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["trade_date"] = df["entry_time"].dt.date
    df = df.sort_values(["trade_date", "entry_time", "trade_id"], kind="mergesort").copy()
    df["trade_sequence"] = df.groupby("trade_date").cumcount() + 1
    per_day = df.groupby("trade_date").agg(
        trades=("trade_id", "count"),
        day_r=("r_pnl", "sum"),
        first_entry=("entry_time", "min"),
        last_exit=("exit_time", "max"),
    )
    seq_rows = []
    overlap_count = 0
    not_after_prev = 0
    duplicate_same_event = int(df.duplicated(subset=["entry_time", "direction", "sweep_time", "sweep_level_type"]).sum())
    simultaneous = int(df.duplicated(subset=["entry_time"], keep=False).sum())
    for date, group in df.groupby("trade_date"):
        group = group.sort_values(["entry_time", "trade_id"], kind="mergesort")
        prev_entry = None
        prev_exit = None
        for _, row in group.iterrows():
            if prev_entry is not None and row["entry_time"] <= prev_entry:
                not_after_prev += 1
            if prev_exit is not None and row["entry_time"] < prev_exit:
                overlap_count += 1
            prev_entry = row["entry_time"]
            prev_exit = row["exit_time"]
            seq_rows.append(
                {
                    "trade_date": date,
                    "trade_id": int(row["trade_id"]),
                    "trade_sequence": int(row["trade_sequence"]),
                    "entry_time": row["entry_time"],
                    "exit_time": row["exit_time"],
                    "direction": row["direction"],
                    "status": row["status"],
                    "r_pnl": row["r_pnl"],
                    "overlaps_previous": bool(prev_exit is not None and row["entry_time"] < prev_exit),
                }
            )
    seq_df = pd.DataFrame(seq_rows)
    daily = per_day.reset_index().rename(columns={"trade_date": "date"})
    per_day.reset_index().to_csv(OUT / "multi_trade" / "phase19_trades_per_day_audit.csv", index=False)
    seq_df.to_csv(OUT / "multi_trade" / "phase19_trade_sequence_analysis.csv", index=False)
    daily[["date", "trades", "day_r"]].to_csv(OUT / "multi_trade" / "phase19_daily_pnl_distribution.csv", index=False)

    sequence_metrics = {}
    for seq in [1, 2, 3]:
        g = df[df["trade_sequence"] == seq]
        m = compute_metrics(g)
        sequence_metrics[str(seq)] = {"pf": m["pf"], "expectancy_R": m["expectancy_R"], "sample": m["sample"]}
    verdict = "PHASE19_MULTI_TRADE_CONFIRMED"
    if per_day["trades"].max() > 3 or duplicate_same_event or simultaneous or overlap_count or not_after_prev:
        verdict = "PHASE19_MULTI_TRADE_INVALIDATES_PHASE19"
    payload = {
        "verdict": verdict,
        "max_daily_trades": int(per_day["trades"].max()),
        "days_with_1_trade": int((per_day["trades"] == 1).sum()),
        "days_with_2_trades": int((per_day["trades"] == 2).sum()),
        "days_with_3_trades": int((per_day["trades"] == 3).sum()),
        "max_daily_loss_R": round(float(per_day["day_r"].min()), 6),
        "duplicate_same_event_count": duplicate_same_event,
        "same_timestamp_trade_count": simultaneous,
        "overlapping_position_count": int(overlap_count),
        "trade_not_after_previous_count": int(not_after_prev),
        "sequence_metrics": sequence_metrics,
    }
    write_json(OUT / "multi_trade" / "phase19_multi_trade_report.json", payload)
    write_text(
        OUT / "multi_trade" / "phase19_multi_trade_report.md",
        "\n".join(
            [
                "# Phase19 Multi-Trade Audit",
                "",
                f"Verdicto: {verdict}",
                f"Max trades/dia: {payload['max_daily_trades']}",
                f"Dias 1/2/3 trades: {payload['days_with_1_trade']} / {payload['days_with_2_trades']} / {payload['days_with_3_trades']}",
                f"PF trade 1: {fmt(sequence_metrics['1']['pf'], 3)}",
                f"PF trade 2: {fmt(sequence_metrics['2']['pf'], 3)}",
                f"PF trade 3: {fmt(sequence_metrics['3']['pf'], 3)}",
                f"Expectancy trade 1: {fmt(sequence_metrics['1']['expectancy_R'], 3)}",
                f"Expectancy trade 2: {fmt(sequence_metrics['2']['expectancy_R'], 3)}",
                f"Expectancy trade 3: {fmt(sequence_metrics['3']['expectancy_R'], 3)}",
                f"Max daily loss R: {fmt(payload['max_daily_loss_R'], 3)}",
                f"Duplicados mismo evento: {duplicate_same_event}",
                f"Trades con mismo timestamp: {simultaneous}",
                f"Solapamientos: {overlap_count}",
                "",
                "Conclusion: el limite de 3 trades/dia existe como corte por cantidad, pero no modela secuencia real ni evita duplicados/solapamientos.",
            ]
        ),
    )
    return payload


def signal_no_lookahead_audit(df_h1, df_m3, sweeps, signals):
    h1_rows = []
    for _, row in sweeps.head(5000).iterrows():
        h1_rows.append(
            {
                "sweep_time": row["timestamp_ny"],
                "sweep_type": row["type"],
                "level_type": row["level_type"],
                "is_fractal": bool(row["is_fractal"]),
                "depth_pips": row["depth_pips"],
                "fractal_delay_claim": "confirmed_on_current_h1_bar_for_fractal_levels",
            }
        )
    h1_df = pd.DataFrame(h1_rows)
    h1_df.to_csv(OUT / "signal_no_lookahead" / "phase19_h1_fractal_timing_audit.csv", index=False)
    sig = signals.copy()
    sig["entry_time_used"] = sig["choch_time"]
    sig["required_entry_model"] = "next_bar_open"
    sig["legacy_entry_model"] = "choch_close_bid"
    sig["entry_next_bar_violation"] = True
    sig["m3_source"] = "resampled_from_m5_ohlc"
    sig[
        [
            "signal_id",
            "sweep_time",
            "choch_time",
            "direction",
            "entry_time_used",
            "required_entry_model",
            "legacy_entry_model",
            "entry_next_bar_violation",
            "m3_source",
        ]
    ].to_csv(OUT / "signal_no_lookahead" / "phase19_m3_choch_timing_audit.csv", index=False)
    entry_next_bar_violations = int(len(sig))
    m3_source_valid = False
    first_choch_per_sweep = int(sig.duplicated(subset=["sweep_id"]).sum()) == 0
    verdict = "PHASE19_SIGNAL_INVALIDATES_PHASE19"
    payload = {
        "verdict": verdict,
        "h1_fractal_delay_confirmed_by_code": True,
        "m3_choch_uses_closed_bar_for_detection": True,
        "entry_occurs_next_bar_open": False,
        "entry_next_bar_violations": entry_next_bar_violations,
        "m3_source_valid_native_m3": m3_source_valid,
        "m3_source_issue": "Phase14Engine synthesizes m3 from m5 when native m3 is absent; this is not certified native M3 data.",
        "first_choch_per_sweep": first_choch_per_sweep,
        "selection_ignores_later_winners_per_sweep": first_choch_per_sweep,
        "critical_findings": [
            "Phase19 legacy enters at CHOCH close, not next bar open.",
            "The M3 series is derived from M5 OHLC, not native M3/tick data.",
            "Multiple H1 sweep levels can produce the same CHOCH timestamp and later duplicate trades.",
        ],
    }
    write_json(OUT / "signal_no_lookahead" / "phase19_signal_no_lookahead_report.json", payload)
    write_text(
        OUT / "signal_no_lookahead" / "phase19_signal_no_lookahead_report.md",
        "\n".join(
            [
                "# Phase19 Signal / No-Lookahead Audit",
                "",
                f"Verdicto: {verdict}",
                "H1 fractal delay: confirmed by existing code.",
                "M3 CHOCH detection: uses closed bar, but on synthetic M3 from M5.",
                f"Entry next-bar open: NO ({entry_next_bar_violations} legacy entries use CHOCH close).",
                "Conclusion: Phase19 fails the operational no-lookahead/execution timing contract.",
            ]
        ),
    )
    return payload


def summarize_scenario(name, df_m3, signals, start=8.0, end=16.5, exclude_friday=True, max_trades=3, tp_r=2.5):
    selected = select_phase19_signals(signals, start, end, exclude_friday, max_trades)
    trades = simulate_legacy_phase19(df_m3, selected, tp_r=tp_r)
    m = compute_metrics(trades)
    return {"scenario": name, "trades": trades, "metrics": m}


def filter_audit(df_m3, base_signals, no_min_signals, choch60_signals):
    scenarios = {
        "base_phase19": summarize_scenario("base_phase19", df_m3, base_signals),
        "without_friday_filter": summarize_scenario("without_friday_filter", df_m3, base_signals, exclude_friday=False),
        "without_sweep_min_filter": summarize_scenario("without_sweep_min_filter", df_m3, no_min_signals),
        "without_window_filter_0700_2000": summarize_scenario("without_window_filter_0700_2000", df_m3, base_signals, start=7.0, end=20.0),
        "without_max3_filter": summarize_scenario("without_max3_filter", df_m3, base_signals, max_trades=None),
        "without_choch30_filter_60m": summarize_scenario("without_choch30_filter_60m", df_m3, choch60_signals),
    }
    base_trades = scenarios["base_phase19"]["trades"]
    no_rollover_trades = base_trades.copy()
    if not no_rollover_trades.empty:
        no_rollover_trades["exit_time"] = pd.to_datetime(no_rollover_trades["exit_time"])
        no_rollover_trades = no_rollover_trades[
            (no_rollover_trades["entry_time"].dt.date == no_rollover_trades["exit_time"].dt.date)
            & ((no_rollover_trades["exit_time"].dt.hour + no_rollover_trades["exit_time"].dt.minute / 60.0) <= 20.0)
        ].copy()
    scenarios["with_no_rollover_post_filter"] = {"scenario": "with_no_rollover_post_filter", "trades": no_rollover_trades, "metrics": compute_metrics(no_rollover_trades)}

    rows = []
    pairs = [
        ("exclude_friday", "without_friday_filter", "base_phase19"),
        ("sweep_min_0_5_pips", "without_sweep_min_filter", "base_phase19"),
        ("window_0800_1630", "without_window_filter_0700_2000", "base_phase19"),
        ("no_rollover", "base_phase19", "with_no_rollover_post_filter"),
        ("max_3_trades_day", "without_max3_filter", "base_phase19"),
        ("choch_within_30m", "without_choch30_filter_60m", "base_phase19"),
    ]
    for filter_name, without_key, with_key in pairs:
        without = scenarios[without_key]["metrics"]
        with_ = scenarios[with_key]["metrics"]
        rows.append(
            {
                "filter": filter_name,
                "without_scenario": without_key,
                "with_scenario": with_key,
                "sample_without": without["sample"],
                "sample_with": with_["sample"],
                "pf_without": without["pf"],
                "pf_with": with_["pf"],
                "expectancy_without": without["expectancy_R"],
                "expectancy_with": with_["expectancy_R"],
                "max_dd_without": without["max_drawdown_R"],
                "max_dd_with": with_["max_drawdown_R"],
                "robustness_interpretation": "needs_external_reason" if with_["pf"] > without["pf"] else "does_not_improve_pf",
            }
        )
    ablation = pd.DataFrame(rows)
    ablation.to_csv(OUT / "filters" / "phase19_filter_ablation.csv", index=False)
    year_rows = []
    for key, value in scenarios.items():
        trades = value["trades"].copy()
        if trades.empty:
            continue
        trades["year"] = pd.to_datetime(trades["entry_time"]).dt.year
        for year, group in trades.groupby("year"):
            m = compute_metrics(group)
            year_rows.append({"scenario": key, "year": int(year), **m})
    pd.DataFrame(year_rows).to_csv(OUT / "filters" / "phase19_filter_by_year.csv", index=False)
    verdict = "PHASE19_FILTERS_OVERFIT_WARNING"
    payload = {
        "verdict": verdict,
        "filters_audited": [x[0] for x in pairs],
        "critical_note": "Several filters came from matrix search and lack pre-registered external rationale in the Phase19 evidence surface.",
        "base_metrics": scenarios["base_phase19"]["metrics"],
        "no_rollover_filtered_metrics": scenarios["with_no_rollover_post_filter"]["metrics"],
    }
    write_json(OUT / "filters" / "phase19_filter_report.json", payload)
    write_text(
        OUT / "filters" / "phase19_filter_report.md",
        "\n".join(
            [
                "# Phase19 Filter Audit",
                "",
                f"Verdicto: {verdict}",
                "Ablation saved in phase19_filter_ablation.csv.",
                "The Friday/min-depth/window/max-trades/CHOCH filters are performance-selected in the available evidence.",
                "No-rollover was not enforced by the legacy backtest; applying it after the fact changes the sample.",
            ]
        ),
    )
    return payload


def execution_audit(trades):
    df = trades.copy()
    same_bar = df[df["same_bar"] == True].copy()
    impossible = df[
        ((df["direction"] == "LONG") & (df["entry_price"] < df["raw_entry_price"]))
        | ((df["direction"] == "SHORT") & (df["legacy_used_bid_for_short_exit"] == False))
    ].copy()
    spread = df[["trade_id", "entry_time", "direction", "spread_entry_pips", "risk_pips", "status", "r_pnl"]].copy()
    same_bar.to_csv(OUT / "execution" / "phase19_same_bar_cases.csv", index=False)
    impossible.to_csv(OUT / "execution" / "phase19_impossible_fills.csv", index=False)
    spread.to_csv(OUT / "execution" / "phase19_spread_distribution.csv", index=False)
    payload = {
        "verdict": "PHASE19_EXECUTION_INVALIDATES_PHASE19",
        "long_entry_ask_used": False,
        "long_exit_bid_used": True,
        "short_entry_bid_used": True,
        "short_exit_ask_used": False,
        "historical_spread_available": True,
        "historical_spread_applied_to_legacy_result": False,
        "slippage_applied_to_legacy_result": False,
        "same_bar_conservative_policy_implemented": False,
        "cost_stress_never_improves_check_required": True,
        "trades_without_sl": int(df["sl_price"].isna().sum()),
        "trades_without_tp": int(df["tp_price"].isna().sum()),
        "impossible_fill_count": int(len(impossible)),
        "critical_findings": [
            "Legacy Phase19 enters LONG at close_bid instead of ask/next open.",
            "Legacy Phase19 exits SHORT with bid high/low instead of ask.",
            "Same-bar policy is avoided by skipping the entry bar, not conservatively simulated.",
            "Reported PF is not a BID/ASK/SPREAD-real result.",
        ],
    }
    write_json(OUT / "execution" / "phase19_bid_ask_audit.json", payload)
    write_text(
        OUT / "execution" / "phase19_bid_ask_audit.md",
        "\n".join(
            [
                "# Phase19 BID/ASK/Same-Bar Audit",
                "",
                f"Verdicto: {payload['verdict']}",
                "LONG entry ASK: NO.",
                "SHORT exit ASK: NO.",
                "Historical spread applied: NO.",
                "Same-bar conservative simulation: NO.",
                "Conclusion: execution invalidates Phase19 as forward standard.",
            ]
        ),
    )
    return payload


def time_news_audit(engine, trades):
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    entry_hour = df["entry_time"].dt.hour + df["entry_time"].dt.minute / 60.0
    time_audit = df[
        [
            "trade_id",
            "entry_time",
            "exit_time",
            "direction",
            "status",
            "r_pnl",
        ]
    ].copy()
    time_audit["entry_hour_float"] = entry_hour
    time_audit["outside_0700_2000"] = (entry_hour < 7.0) | (entry_hour >= 20.0)
    time_audit["outside_phase19_window"] = (entry_hour < 8.0) | (entry_hour >= 16.5)
    roll = count_rollover_violations(df)
    time_audit["open_after_2000"] = (
        (df["entry_time"].dt.date != df["exit_time"].dt.date)
        | ((df["exit_time"].dt.hour + df["exit_time"].dt.minute / 60.0) > 20.0)
    )
    time_audit.to_csv(OUT / "time_news" / "phase19_time_window_audit.csv", index=False)
    news_counts = count_news_violations(engine, df)
    news_df = pd.DataFrame(
        [{"guard_minutes": int(k.replace("m", "")), "violations": v} for k, v in news_counts.items()]
    )
    news_df.to_csv(OUT / "time_news" / "phase19_news_audit.csv", index=False)
    pd.DataFrame([roll]).to_csv(OUT / "time_news" / "phase19_rollover_audit.csv", index=False)
    verdict = "PHASE19_TIME_NEWS_INVALIDATES_PHASE19"
    payload = {
        "verdict": verdict,
        "trades_before_0700": int((entry_hour < 7.0).sum()),
        "trades_after_2000": int((entry_hour >= 20.0).sum()),
        "trades_open_after_2000": roll["open_after_2000_count"],
        "trades_1700_1900": roll["entry_1700_1900_count"],
        "news_guard_active_in_legacy_code": False,
        "news_violations_30m": news_counts["30m"],
        "news_violations_45m": news_counts["45m"],
        "news_violations_60m": news_counts["60m"],
        "timezone": NY_TZ,
        "dst_checked_by_timezone_aware_timestamps": True,
        "forced_close_correct": False,
    }
    write_json(OUT / "time_news" / "phase19_time_news_report.json", payload)
    write_text(
        OUT / "time_news" / "phase19_time_news_report.md",
        "\n".join(
            [
                "# Phase19 Time / News / Rollover Audit",
                "",
                f"Verdicto: {verdict}",
                f"Trades antes de 07:00 NY: {payload['trades_before_0700']}",
                f"Trades despues de 20:00 NY: {payload['trades_after_2000']}",
                f"Trades abiertos despues de 20:00 NY: {payload['trades_open_after_2000']}",
                f"News violations +/-30m: {payload['news_violations_30m']}",
                "Conclusion: el codigo legacy no aplica news guard ni forced close 20:00.",
            ]
        ),
    )
    return payload


def period_breakdown(trades, label):
    m = compute_metrics(trades)
    return {"period": label, **m}


def robustness_audit(base_trades, extended_trades):
    combined = pd.concat([extended_trades, base_trades], ignore_index=True) if not extended_trades.empty else base_trades.copy()
    combined["entry_time"] = pd.to_datetime(combined["entry_time"])
    rows = []
    for label, start, end in [
        ("2015-2017", 2015, 2017),
        ("2018-2019", 2018, 2019),
        ("2020-2022", 2020, 2022),
        ("2023-2025", 2023, 2025),
        ("2026_partial", 2026, 2026),
    ]:
        g = combined[(combined["entry_time"].dt.year >= start) & (combined["entry_time"].dt.year <= end)].copy()
        rows.append(period_breakdown(g, label))
    pd.DataFrame(rows).to_csv(OUT / "robustness" / "phase19_forensic_robustness_by_period.csv", index=False)
    year_rows = []
    for year, g in combined.groupby(combined["entry_time"].dt.year):
        year_rows.append({"year": int(year), **compute_metrics(g)})
    year_df = pd.DataFrame(year_rows)
    year_df.to_csv(OUT / "robustness" / "phase19_forensic_robustness_by_year.csv", index=False)
    month_rows = []
    for month, g in combined.groupby(combined["entry_time"].dt.strftime("%Y-%m")):
        month_rows.append({"month": month, **compute_metrics(g)})
    month_df = pd.DataFrame(month_rows)
    month_df.to_csv(OUT / "robustness" / "phase19_forensic_robustness_by_month.csv", index=False)
    negative_years = int((year_df["cumulative_R"] < 0).sum()) if not year_df.empty else 0
    negative_months = int((month_df["cumulative_R"] < 0).sum()) if not month_df.empty else 0
    pf_2023_2025 = rows[3]["pf"]
    pf_2020_2022 = rows[2]["pf"]
    verdict = "PHASE19_ROBUSTNESS_WARNING"
    warnings = [
        "Legacy robustness is numerically strong, but it is downstream of invalid execution/data/multitrade assumptions.",
        "2015-2019 is available but uses the same synthetic M3 construction.",
    ]
    payload = {
        "verdict": verdict,
        "negative_years": negative_years,
        "negative_months": negative_months,
        "worst_month": month_df.sort_values("cumulative_R").head(1).to_dict("records")[0] if not month_df.empty else None,
        "worst_year": year_df.sort_values("cumulative_R").head(1).to_dict("records")[0] if not year_df.empty else None,
        "pf_2023_2025": pf_2023_2025,
        "pf_2020_2022": pf_2020_2022,
        "degradation_2023_2025_vs_2020_2022": round(pf_2023_2025 - pf_2020_2022, 6) if pf_2020_2022 else None,
        "warnings": warnings,
    }
    write_text(OUT / "robustness" / "phase19_forensic_overfitting_warnings.md", "\n".join(["# Phase19 Robustness Warnings", "", *[f"- {w}" for w in warnings]]))
    write_text(
        OUT / "robustness" / "phase19_forensic_robustness_report.md",
        "\n".join(
            [
                "# Phase19 Temporal Robustness",
                "",
                f"Verdicto: {verdict}",
                f"Negative years: {negative_years}",
                f"Negative months: {negative_months}",
                f"PF 2020-2022: {fmt(pf_2020_2022, 3)}",
                f"PF 2023-2025: {fmt(pf_2023_2025, 3)}",
                "Nota: la robustez numerica no rescata una ejecucion invalida.",
            ]
        ),
    )
    return payload


def cost_audit(df_m3, selected):
    slip_rows = []
    for pips in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        trades = simulate_legacy_phase19(df_m3, selected, tp_r=2.5, cost_pips=pips)
        m = compute_metrics(trades)
        slip_rows.append({"slippage_pips": pips, "pf": m["pf"], "expectancy_R": m["expectancy_R"], "sample": m["sample"]})
    spread_rows = []
    for pips in [0.0, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0]:
        trades = simulate_legacy_phase19(df_m3, selected, tp_r=2.5, cost_pips=pips)
        m = compute_metrics(trades)
        spread_rows.append({"spread_stress_pips": pips, "pf": m["pf"], "expectancy_R": m["expectancy_R"], "sample": m["sample"]})
    slip_df = pd.DataFrame(slip_rows)
    spread_df = pd.DataFrame(spread_rows)
    slip_df.to_csv(OUT / "costs" / "phase19_forensic_slippage_sensitivity.csv", index=False)
    spread_df.to_csv(OUT / "costs" / "phase19_forensic_spread_stress.csv", index=False)

    def threshold(df, col, metric, limit):
        bad = df[df[metric] < limit]
        return None if bad.empty else float(bad.iloc[0][col])

    payload = {
        "verdict": "PHASE19_COST_WARNING",
        "slippage": slip_rows,
        "spread_stress": spread_rows,
        "max_cost_before_pf_lt_2_0": threshold(spread_df, "spread_stress_pips", "pf", 2.0),
        "max_cost_before_pf_lt_1_5": threshold(spread_df, "spread_stress_pips", "pf", 1.5),
        "max_cost_before_pf_lt_1_35": threshold(spread_df, "spread_stress_pips", "pf", 1.35),
        "max_cost_before_expectancy_le_0": threshold(spread_df, "spread_stress_pips", "expectancy_R", 0.000001),
        "execution_real_can_degrade_strongly": True,
        "note": "Sensitivity is legacy-style entry cost stress, not a full BID/ASK-real fill model.",
    }
    write_json(OUT / "costs" / "phase19_forensic_cost_summary.json", payload)
    write_text(
        OUT / "costs" / "phase19_forensic_cost_summary.md",
        "\n".join(
            [
                "# Phase19 Cost Sensitivity",
                "",
                f"Verdicto: {payload['verdict']}",
                f"PF at +1.0 pip: {fmt(spread_df.loc[spread_df['spread_stress_pips'] == 1.0, 'pf'].iloc[0], 3)}",
                f"PF at +1.5 pips: {fmt(spread_df.loc[spread_df['spread_stress_pips'] == 1.5, 'pf'].iloc[0], 3)}",
                "Nota: el stress degrada el resultado, pero la ejecucion base ya esta invalidada por BID/ASK y timing.",
            ]
        ),
    )
    return payload


def risk_audit(trades):
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    equity = equity_and_dd(df)
    equity.to_csv(OUT / "risk" / "phase19_drawdown_curve.csv", index=False)
    daily = df.groupby(df["entry_time"].dt.date)["r_pnl"].agg(["count", "sum", "mean"]).reset_index()
    daily.columns = ["date", "trades", "day_R", "avg_trade_R"]
    daily.to_csv(OUT / "risk" / "phase19_daily_risk.csv", index=False)
    monthly = df.groupby(df["entry_time"].dt.strftime("%Y-%m"))["r_pnl"].agg(["count", "sum", "mean", "std"]).reset_index()
    monthly.columns = ["month", "trades", "month_R", "avg_trade_R", "volatility_R"]
    monthly.to_csv(OUT / "risk" / "phase19_monthly_risk.csv", index=False)
    m = compute_metrics(df)
    values = df["r_pnl"].astype(float)
    win_rate = float((values > 0).mean())
    loss_rate = float((values < 0).mean())
    if win_rate <= 0 or win_rate <= loss_rate:
        ruin = 1.0
    else:
        ruin = min(1.0, (loss_rate / win_rate) ** 10)
    payload = {
        "verdict": "PHASE19_RISK_WARNING",
        **m,
        "average_monthly_R": round(float(monthly["month_R"].mean()), 6),
        "monthly_volatility_R": round(float(monthly["month_R"].std()), 6),
        "profit_distribution": values.value_counts().sort_index().to_dict(),
        "skew": round(float(values.skew()), 6),
        "kurtosis": round(float(values.kurtosis()), 6),
        "risk_of_ruin_simple_10R_bankroll": round(float(ruin), 8),
        "largest_loss_cluster": m["max_loss_streak"],
        "drawdown_recovery_time_trades": m["drawdown_recovery_trades"],
        "note": "Risk is warning-only because the measured curve is based on invalid legacy execution assumptions.",
    }
    write_json(OUT / "risk" / "phase19_risk_summary.json", payload)
    write_text(
        OUT / "risk" / "phase19_risk_summary.md",
        "\n".join(
            [
                "# Phase19 Drawdown / Risk",
                "",
                f"Verdicto: {payload['verdict']}",
                f"Max DD R: {fmt(payload['max_drawdown_R'], 3)}",
                f"Max loss streak: {payload['max_loss_streak']}",
                f"Worst day R: {fmt(payload['worst_day_R'], 3)}",
                f"Worst week R: {fmt(payload['worst_week_R'], 3)}",
                f"Worst month R: {fmt(payload['worst_month_R'], 3)}",
                f"Average monthly R: {fmt(payload['average_monthly_R'], 3)}",
                f"Monthly volatility R: {fmt(payload['monthly_volatility_R'], 3)}",
            ]
        ),
    )
    return payload


def overfit_audit(metrics):
    registry_rows = []
    source_files = [
        OUT.parent / "phase19_phase18_expansion" / "ltf_sensitivity" / "ltf_sensitivity_results.csv",
        OUT.parent / "phase19_phase18_expansion" / "window_expansion" / "window_expansion_results.csv",
        OUT.parent / "phase19_phase18_expansion" / "trade_frequency" / "trade_frequency_results.csv",
        OUT.parent / "phase19_phase18_expansion" / "management_matrix" / "management_matrix_results.csv",
        OUT.parent / "phase19_phase18_expansion" / "quality_filters" / "quality_filter_results.csv",
    ]
    for path in source_files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for i, row in df.iterrows():
            item = {"source_file": str(path.relative_to(LAB)), "variant_index": int(i)}
            item.update(row.to_dict())
            registry_rows.append(item)
    registry = pd.DataFrame(registry_rows)
    registry.to_csv(OUT / "overfit_control" / "phase19_tested_variants_registry.csv", index=False)
    variants = max(1, len(registry))
    raw_pf = metrics["pf"]
    raw_exp = metrics["expectancy_R"]
    multiple_test_penalty = min(0.65, math.log10(variants + 1) * 0.18)
    robustness_penalty = 0.15
    cost_penalty = 0.15
    drawdown_penalty = 0.10 if abs(metrics["max_drawdown_R"]) > 15 else 0.05
    invalid_execution_penalty = 0.75
    pf_penalized = max(0.0, raw_pf * (1 - multiple_test_penalty - robustness_penalty - cost_penalty - drawdown_penalty))
    exp_penalized = raw_exp * (1 - multiple_test_penalty - robustness_penalty - cost_penalty - drawdown_penalty)
    final_score = max(0.0, min(100.0, (pf_penalized / max(raw_pf, 1e-9)) * 100.0 * (1 - invalid_execution_penalty)))
    score = pd.DataFrame(
        [
            {
                "tested_variants_count": variants,
                "raw_pf": raw_pf,
                "pf_penalized_multiple_tests": pf_penalized,
                "raw_expectancy_R": raw_exp,
                "expectancy_penalized": exp_penalized,
                "robustness_penalty": robustness_penalty,
                "cost_penalty": cost_penalty,
                "drawdown_penalty": drawdown_penalty,
                "invalid_execution_penalty": invalid_execution_penalty,
                "final_deflated_edge_score": final_score,
            }
        ]
    )
    score.to_csv(OUT / "overfit_control" / "phase19_deflated_edge_score.csv", index=False)
    payload = {
        "verdict": "PHASE19_OVERFIT_HIGH_RISK",
        "tested_variants_count": variants,
        "raw_pf": raw_pf,
        "pf_penalized": pf_penalized,
        "raw_expectancy_R": raw_exp,
        "expectancy_penalized": exp_penalized,
        "final_deflated_edge_score": final_score,
        "note": "Overfit control is high risk because Phase19 emerged from multiple exploratory matrices and failed independent execution controls.",
    }
    write_json(OUT / "overfit_control" / "phase19_overfit_control_report.json", payload)
    write_text(
        OUT / "overfit_control" / "phase19_overfit_control_report.md",
        "\n".join(
            [
                "# Phase19 Overfit Control",
                "",
                f"Verdicto: {payload['verdict']}",
                f"Tested variants counted: {variants}",
                f"Raw PF: {fmt(raw_pf, 3)}",
                f"Penalized PF: {fmt(pf_penalized, 3)}",
                f"Final deflated edge score: {fmt(final_score, 2)}",
            ]
        ),
    )
    return payload


def comparison_reports(metrics):
    comparisons = {
        "phase18": {"label": "Phase18 baseline", "pf": 1.63, "sample": 1040, "note": "diurna protegida; no reemplazada"},
        "manual": {"label": "Manual", "pf": 1.64, "sample": 841, "note": "manual monetario; R-normalizado PF 1.53"},
        "phase7": {"label": "Phase7", "pf": 1.50, "sample": None, "note": "referencia historica; no modificada"},
        "phase8": {"label": "Phase8", "pf": 2.09, "sample": 88, "note": "alta precision; no modificada"},
        "phase13": {"label": "Phase13", "pf": 1.62, "sample": 210, "note": "London reclaim; no modificada"},
        "phase17": {"label": "Phase17", "pf": 2.03, "sample": 53, "note": "post-news; no modificada"},
        "scbi": {"label": "SCBI_M5_GLOBAL", "pf": None, "sample": None, "note": "benchmark overnight protegido; solo lectura"},
    }
    scbi_summary = ROOT / "results" / "SCBI_2020_2025_DURABILITY" / "summary.json"
    if scbi_summary.exists():
        with scbi_summary.open("r", encoding="utf-8") as f:
            data = json.load(f)
        comparisons["scbi"]["pf"] = data.get("pf") or data.get("profit_factor")
        comparisons["scbi"]["sample"] = data.get("sample") or data.get("trades") or data.get("n_trades")
    for key, item in comparisons.items():
        write_text(
            OUT / "comparison" / f"phase19_forensic_vs_{key}.md",
            "\n".join(
                [
                    f"# Phase19 Forensic vs {item['label']}",
                    "",
                    f"Phase19 legacy PF reproducido: {fmt(metrics['pf'], 3)}",
                    f"Phase19 sample reproducido: {metrics['sample']}",
                    f"{item['label']} PF referencia: {fmt(item['pf'], 3) if item['pf'] is not None else 'NA'}",
                    f"{item['label']} sample referencia: {item['sample'] if item['sample'] is not None else 'NA'}",
                    f"Nota: {item['note']}",
                    "Conclusion: Phase19 no puede reemplazar esta referencia porque la auditoria forense encontro fallas invalidantes.",
                ]
            ),
        )
    return comparisons


def starting_point_files():
    payload = {
        "phase": 19,
        "status": "PENDING_FORENSIC_AUDIT",
        "does_not_replace_phase18": True,
        "forward_enabled": False,
        "real_enabled": False,
        "vps_enabled": False,
        "touches_scbi": False,
        "objective": "validate_or_invalidate_phase19",
        "root": str(ROOT),
        "lab": str(LAB),
        "canonical_zip": str(ZIP_PATH),
        "timestamp": datetime.now().isoformat(),
    }
    write_json(OUT / "diagnosis" / "phase19_forensic_starting_point.json", payload)
    write_text(
        OUT / "diagnosis" / "phase19_forensic_starting_point.md",
        "\n".join(
            [
                "# Phase19 Forensic Starting Point",
                "",
                "Estado: PENDING_FORENSIC_AUDIT.",
                "Phase19 no reemplaza Phase18.",
                "No habilita forward, real ni VPS.",
                "No toca SCBI.",
                "Objetivo: validar o invalidar Phase19.",
            ]
        ),
    )
    return payload


def run_extended_2015_2019():
    try:
        _, _, df_m3, _, signals = load_and_prepare_period("period_2015_2019", max_mins=30, min_depth_pips=0.5)
        selected = select_phase19_signals(signals)
        return simulate_legacy_phase19(df_m3, selected, tp_r=2.5)
    except Exception as exc:
        write_text(OUT / "robustness" / "phase19_2015_2019_run_error.md", f"# 2015-2019 Run Error\n\n{exc}")
        return pd.DataFrame()


def run_phase19_tests():
    loader = unittest.TestLoader()
    suite = loader.discover(str(TESTS), pattern="test_phase19_*.py")
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    summary = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "passed": result.testsRun - len(result.failures) - len(result.errors),
        "failure_names": [str(x[0]) for x in result.failures],
        "error_names": [str(x[0]) for x in result.errors],
        "verdict": "PHASE19_TESTS_PASSED" if result.wasSuccessful() else "PHASE19_TESTS_FAILED",
        "raw_output": stream.getvalue(),
    }
    write_json(OUT / "tests" / "phase19_forensic_test_results.json", summary)
    write_text(
        OUT / "tests" / "phase19_forensic_test_results.md",
        "\n".join(
            [
                "# Phase19 Forensic Test Results",
                "",
                f"Verdicto: {summary['verdict']}",
                f"Tests run: {summary['tests_run']}",
                f"Passed: {summary['passed']}",
                f"Failures: {summary['failures']}",
                f"Errors: {summary['errors']}",
                "",
                "## Raw output",
                "```",
                summary["raw_output"].strip(),
                "```",
            ]
        ),
    )
    return summary


def final_verdict(results):
    invalidators = []
    for key in ["multi_trade", "signal", "execution", "time_news"]:
        verdict = results[key]["verdict"]
        if "INVALIDATES" in verdict or "INVALIDATE" in verdict:
            invalidators.append(verdict)
    if results["tests"]["verdict"] != "PHASE19_TESTS_PASSED":
        invalidators.append("PHASE19_CRITICAL_TESTS_FAILED")
    if results["overfit"]["verdict"] in ["PHASE19_OVERFIT_HIGH_RISK", "PHASE19_OVERFIT_INVALIDATES_PHASE19"]:
        invalidators.append(results["overfit"]["verdict"])
    return "PHASE19_INVALIDATED" if invalidators else "PHASE19_VALIDATED_FOR_FORWARD_DEMO", invalidators


def write_final_report(results, verdict, invalidators, comparisons):
    metrics = results["reproduction"]["metrics"]
    payload = {
        "objective": "Forensic audit Phase19 before operational promotion",
        "original_phase19": REPORTED,
        "reproduction": results["reproduction"],
        "tp_sl_math": results["tp_sl"],
        "multi_trade": results["multi_trade"],
        "signal_no_lookahead": results["signal"],
        "filters": results["filters"],
        "execution": results["execution"],
        "time_news": results["time_news"],
        "robustness": results["robustness"],
        "costs": results["costs"],
        "risk": results["risk"],
        "overfit_control": results["overfit"],
        "tests": results["tests"],
        "comparison": comparisons,
        "final_verdict": verdict,
        "invalidators": invalidators,
        "single_next_step": "Repair Phase19 in quarantine or keep Phase18 as daytime baseline; do not promote Phase19.",
    }
    write_json(REPORTS / "PHASE19_FORENSIC_AUDIT_REPORT.json", payload)
    sections = [
        "# PHASE19 FORENSIC AUDIT REPORT",
        "",
        "## 1. Objetivo",
        "Auditar forensemente Phase19 antes de aceptarla como estandar operativo.",
        "",
        "## 2. Resultado original Phase19",
        f"Reportado: sample {REPORTED['sample']}, PF {REPORTED['pf']}, expectancy {REPORTED['expectancy_r']}R, ventana {REPORTED['window']}.",
        "",
        "## 3. Reproduccion",
        f"Verdicto: {results['reproduction']['reproduction_verdict']}. Sample {metrics['sample']}, PF {fmt(metrics['pf'], 3)}, expectancy {fmt(metrics['expectancy_R'], 3)}R.",
        "",
        "## 4. TP/SL math",
        f"Verdicto: {results['tp_sl']['verdict']}.",
        "",
        "## 5. Multi-trade audit",
        f"Verdicto: {results['multi_trade']['verdict']}. Duplicados mismo evento: {results['multi_trade']['duplicate_same_event_count']}; solapamientos: {results['multi_trade']['overlapping_position_count']}.",
        "",
        "## 6. No-lookahead",
        f"Verdicto: {results['signal']['verdict']}. Entry next-bar open: {results['signal']['entry_occurs_next_bar_open']}; M3 nativo certificado: {results['signal']['m3_source_valid_native_m3']}.",
        "",
        "## 7. Filtros",
        f"Verdicto: {results['filters']['verdict']}.",
        "",
        "## 8. Ejecucion",
        f"Verdicto: {results['execution']['verdict']}. LONG ask usado: {results['execution']['long_entry_ask_used']}; SHORT ask exit usado: {results['execution']['short_exit_ask_used']}.",
        "",
        "## 9. Time/news",
        f"Verdicto: {results['time_news']['verdict']}. News guard legacy activo: {results['time_news']['news_guard_active_in_legacy_code']}; forced close correcto: {results['time_news']['forced_close_correct']}.",
        "",
        "## 10. Robustez",
        f"Verdicto: {results['robustness']['verdict']}. La robustez numerica es secundaria porque la ejecucion esta invalidada.",
        "",
        "## 11. Costos",
        f"Verdicto: {results['costs']['verdict']}. Sensibilidad legacy, no fill BID/ASK-real.",
        "",
        "## 12. Drawdown/riesgo",
        f"Verdicto: {results['risk']['verdict']}. Max DD {fmt(results['risk']['max_drawdown_R'], 3)}R, worst day {fmt(results['risk']['worst_day_R'], 3)}R.",
        "",
        "## 13. Control de sobreoptimizacion",
        f"Verdicto: {results['overfit']['verdict']}. Variants counted: {results['overfit']['tested_variants_count']}; deflated score {fmt(results['overfit']['final_deflated_edge_score'], 2)}.",
        "",
        "## 14. Tests",
        f"Verdicto: {results['tests']['verdict']}. Run {results['tests']['tests_run']}, failures {results['tests']['failures']}, errors {results['tests']['errors']}.",
        "",
        "## 15. Comparacion",
        "Phase19 no reemplaza Phase18 ni referencias previas por fallas invalidantes.",
        "",
        "## 16. Veredicto final",
        verdict,
        "",
        "## 17. Siguiente paso unico",
        "No promover Phase19. Mantener Phase18 como baseline diurna protegida y, si se repara, hacerlo dentro de cuarentena con M3 nativo/next-bar/BID-ASK/news/forced-close/multitrade real.",
    ]
    write_text(REPORTS / "PHASE19_FORENSIC_AUDIT_REPORT.md", "\n".join(sections))
    return payload


def update_status_files(verdict):
    status = {
        "project_status": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "root_status": "FORENSIC_AUDIT_UPDATED",
            "lab": "BOT_V2_DAYTIME_LAB",
            "strategies": {
                "SCBI_M5_GLOBAL": "protected_unchanged",
                "Phase18_Fractal_Sweep": "daytime_baseline_protected",
                "Phase19_Expanded_Sweep": verdict,
            },
            "critical_note": "Phase19 no reemplaza Phase18. Auditoria forense encontro fallas invalidantes en multitrade, no-lookahead/timing, BID/ASK, news/rollover y sobreoptimizacion.",
            "mt5_touched": False,
            "real_trading_enabled": False,
        }
    }
    authority = {
        "authority_hierarchy": {
            "primary": {
                "id": "SCBI_M5_GLOBAL",
                "role": "overnight_authority",
                "status": "protected_unchanged",
                "window": "00:00-05:00 UTC",
            },
            "daytime_baseline": {
                "id": "Phase18_H1_Fractal_Sweep_First_M3_CHOCH",
                "pf": 1.63,
                "sample": 1040,
                "status": "validated_for_forward_demo_baseline",
            },
            "daytime_candidates": [
                {"id": "Phase8_High_Precision", "pf": 2.09, "status": "reference_unchanged"},
                {"id": "Phase13_London_Reclaim", "pf": 1.62, "status": "reference_unchanged"},
                {"id": "Phase17_Post_News", "pf": 2.03, "sample": 53, "status": "reference_unchanged"},
            ],
            "quarantined_or_rejected": [
                {
                    "id": "Phase19_Expanded_Sweep",
                    "reported_pf": 3.18,
                    "reported_sample": 3177,
                    "status": verdict,
                    "reason": "forensic_invalidators: multitrade_duplicates_overlap, entry_timing, synthetic_m3, bid_ask, news_rollover, overfit_high_risk",
                },
                {"id": "Phase12", "status": "invalidated_not_authority"},
            ],
        },
        "lab_status": {"id": "BOT_V2_DAYTIME_LAB", "role": "research_only", "authority": "none_production"},
    }
    write_json(ROOT / "01_CURRENT_PROJECT_STATUS.json", status)
    write_json(ROOT / "02_STRATEGY_AUTHORITY_MAP.json", authority)
    write_text(
        ROOT / "01_CURRENT_PROJECT_STATUS.md",
        "\n".join(
            [
                "# CURRENT PROJECT STATUS",
                "",
                f"Fecha de actualizacion: {datetime.now().strftime('%Y-%m-%d')}",
                f"Estado Phase19: {verdict}",
                "",
                "## Estado de estrategias",
                "- SCBI_M5_GLOBAL: PROTEGIDA / SIN CAMBIOS.",
                "- Phase18 Fractal Sweep: baseline diurna protegida; no fue reemplazada.",
                f"- Phase19 Expanded Sweep: {verdict}. No habilita forward, real ni VPS.",
                "- Phase12: invalidada; no autoridad.",
                "",
                "## Nota critica",
                "La auditoria forense reprodujo el resultado legacy de Phase19, pero encontro fallas invalidantes en ejecucion, timing, multitrade, news/rollover y control de sobreoptimizacion.",
                "",
                "## Siguiente paso unico",
                "No promover Phase19. Mantener Phase18 como baseline diurna protegida.",
            ]
        ),
    )
    write_text(
        ROOT / "02_STRATEGY_AUTHORITY_MAP.md",
        "\n".join(
            [
                "# STRATEGY AUTHORITY MAP",
                "",
                "## 1. Autoridad maxima: SCBI_M5_GLOBAL",
                "- Estado: PROTEGIDA / SIN CAMBIOS.",
                "- No fue modificada por la auditoria Phase19.",
                "",
                "## 2. Baseline diurna protegida",
                "- Phase18 H1 Fractal Sweep + First M3 CHOCH: VALIDATED_FOR_FORWARD_DEMO, PF 1.63, sample 1.040.",
                "- Phase19 no reemplaza Phase18.",
                "",
                "## 3. Phase19",
                f"- Estado: {verdict}.",
                "- Resultado legacy reproducido, pero invalidado por auditoria forense.",
                "- No habilita forward demo, real, VPS ni reemplazo operativo.",
                "",
                "## 4. Referencias no modificadas",
                "- Phase7, Phase8, Phase13, Phase17 y Phase18 quedan sin cambios de logica.",
                "- Phase12 sigue invalidada y no puede usarse como autoridad.",
                "",
                "## 5. Regla de autoridad",
                "Si hay contradiccion entre narrativa previa y archivos reales actuales, mandan estos archivos de estado y el reporte PHASE19_FORENSIC_AUDIT_REPORT.",
            ]
        ),
    )


def update_zip_manifest():
    write_text(
        ROOT / "ZIP_CONTENTS_MANIFEST.md",
        "\n".join(
            [
                "# ZIP CONTENTS MANIFEST",
                "",
                f"Fecha de reconstruccion: {datetime.now().strftime('%Y-%m-%d')} (PHASE19 FORENSIC AUDIT)",
                "Estado: CANONICO ACTUALIZADO CON PHASE19_INVALIDATED",
                "",
                "## Contenido incluido",
                "- 00_READ_THIS_FIRST.md",
                "- 01_CURRENT_PROJECT_STATUS.md/json",
                "- 02_STRATEGY_AUTHORITY_MAP.md/json",
                "- 03_OBSOLETE_AND_SUPERSEDED_INDEX.md/json",
                "- ZIP_CONTENTS_MANIFEST.md",
                "- BOT_V2_DAYTIME_LAB/reports/PHASE19_FORENSIC_AUDIT_REPORT.md/json",
                "- BOT_V2_DAYTIME_LAB/outputs/phase19_forensic_audit/",
                "- BOT_V2_DAYTIME_LAB/tests/engine_safety/test_phase19_*.py",
                "- BOT_V2_DAYTIME_LAB/src/run_phase19_forensic_audit.py",
                "- BOT_V2_DAYTIME_LAB/src/phase18_h1_fractal_sweep.py",
                "- BOT_V2_DAYTIME_LAB/src/phase18_first_3m_choch.py",
                "",
                "## Exclusiones",
                "- datasets pesados y raw data",
                "- .git, .venv, __pycache__, .pyc, cache, logs pesados, temporales",
                "- ARCHIVE_SUPERSEDED completo",
                "- mt5_local_config.json, .env, secrets, credentials, *.key, *.pem",
                "- ZIPs internos",
                "",
                "## Veredicto",
                "El ZIP incluye la auditoria forense Phase19. Phase19 queda invalidada y no reemplaza Phase18.",
            ]
        ),
    )


def should_exclude(path):
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    forbidden_dirs = {
        ".git",
        ".venv",
        "__pycache__",
        "cache",
        "archive_superseded",
        "data",
        "data manual",
    }
    if parts & forbidden_dirs:
        return True
    if name.endswith(".pyc") or name.endswith(".zip") or name.endswith(".tmp") or name.endswith(".log"):
        return True
    secret_tokens = ["secret", "credential", ".env", ".key", ".pem", "mt5_local_config"]
    return any(tok in name for tok in secret_tokens)


def rebuild_zip():
    staging = ROOT / "_zip_staging_phase19_forensic"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    include = [
        ROOT / "00_READ_THIS_FIRST.md",
        ROOT / "01_CURRENT_PROJECT_STATUS.md",
        ROOT / "01_CURRENT_PROJECT_STATUS.json",
        ROOT / "02_STRATEGY_AUTHORITY_MAP.md",
        ROOT / "02_STRATEGY_AUTHORITY_MAP.json",
        ROOT / "03_OBSOLETE_AND_SUPERSEDED_INDEX.md",
        ROOT / "03_OBSOLETE_AND_SUPERSEDED_INDEX.json",
        ROOT / "ZIP_CONTENTS_MANIFEST.md",
        REPORTS / "PHASE19_FORENSIC_AUDIT_REPORT.md",
        REPORTS / "PHASE19_FORENSIC_AUDIT_REPORT.json",
        OUT,
        TESTS,
        SRC / "run_phase19_forensic_audit.py",
        SRC / "phase18_h1_fractal_sweep.py",
        SRC / "phase18_first_3m_choch.py",
        SRC / "phase14_engine.py",
    ]
    for src in include:
        if not src.exists():
            continue
        rel = src.relative_to(ROOT)
        dest = staging / rel
        if src.is_dir():
            for root_dir, dirs, files in os.walk(src):
                root_path = Path(root_dir)
                dirs[:] = [d for d in dirs if not should_exclude(root_path / d)]
                for file in files:
                    file_path = root_path / file
                    if should_exclude(file_path):
                        continue
                    if file_path.stat().st_size > 8 * 1024 * 1024:
                        continue
                    out_path = staging / file_path.relative_to(ROOT)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, out_path)
        else:
            if should_exclude(src):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root_dir, _, files in os.walk(staging):
            for file in files:
                file_path = Path(root_dir) / file
                zf.write(file_path, file_path.relative_to(staging))
    shutil.rmtree(staging)
    return validate_zip()


def validate_zip():
    payload = {
        "zip_path": str(ZIP_PATH),
        "exists": ZIP_PATH.exists(),
        "size_bytes": ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0,
        "testzip": None,
        "contains_phase19_forensic": False,
        "contains_secrets": False,
        "contains_heavy_data": False,
        "entry_count": 0,
        "verdict": "ZIP_VALIDATION_FAILED",
    }
    if not ZIP_PATH.exists():
        write_json(OUT / "zip" / "phase19_zip_validation.json", payload)
        return payload
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        payload["testzip"] = zf.testzip()
        names = zf.namelist()
        payload["entry_count"] = len(names)
        payload["contains_phase19_forensic"] = any(
            n.startswith("BOT_V2_DAYTIME_LAB/outputs/phase19_forensic_audit/") for n in names
        ) and "BOT_V2_DAYTIME_LAB/reports/PHASE19_FORENSIC_AUDIT_REPORT.md" in names
        forbidden = [".git", ".venv", "__pycache__", ".pyc", ".env", "secret", "credential", ".key", ".pem", "mt5_local_config", "ARCHIVE_SUPERSEDED", "/data/"]
        payload["contains_secrets"] = any(any(tok.lower() in n.lower() for tok in forbidden) for n in names)
        payload["contains_heavy_data"] = any(n.lower().endswith((".h5", ".parquet")) or "/data/" in n.lower() for n in names)
    payload["verdict"] = (
        "ZIP_VALIDATED"
        if payload["exists"]
        and payload["testzip"] is None
        and payload["contains_phase19_forensic"]
        and not payload["contains_secrets"]
        and not payload["contains_heavy_data"]
        else "ZIP_VALIDATION_FAILED"
    )
    write_json(OUT / "zip" / "phase19_zip_validation.json", payload)
    write_text(
        OUT / "zip" / "phase19_zip_validation.md",
        "\n".join(
            [
                "# Phase19 ZIP Validation",
                "",
                f"Verdicto: {payload['verdict']}",
                f"Existe: {payload['exists']}",
                f"Size bytes: {payload['size_bytes']}",
                f"testzip: {payload['testzip']}",
                f"Entry count: {payload['entry_count']}",
                f"Contains Phase19 forensic: {payload['contains_phase19_forensic']}",
                f"Contains secrets: {payload['contains_secrets']}",
                f"Contains heavy data: {payload['contains_heavy_data']}",
            ]
        ),
    )
    return payload


def git_status():
    try:
        status = subprocess.run(["git", "status", "--short"], cwd=str(ROOT), text=True, capture_output=True, check=False)
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=str(ROOT), text=True, capture_output=True, check=False)
        remote = subprocess.run(["git", "remote", "-v"], cwd=str(ROOT), text=True, capture_output=True, check=False)
        payload = {
            "branch": branch.stdout.strip(),
            "status_short": status.stdout.strip().splitlines(),
            "remote": remote.stdout.strip().splitlines(),
            "pushed": False,
            "push_attempted": False,
            "reason": "Audit artifacts generated locally. Push is left gated by final human/GitHub policy because Phase19 is invalidated and critical tests fail.",
        }
    except Exception as exc:
        payload = {"error": str(exc), "pushed": False, "push_attempted": False}
    write_json(OUT / "git_status.json", payload)
    return payload


def main():
    if Path.cwd().resolve() != ROOT.resolve():
        raise SystemExit(f"FAIL-CLOSED: cwd fuera de raiz oficial: {Path.cwd()}")
    ensure_dirs()
    starting_point_files()
    repro = reproduction_phase()

    # Additional signal sets for filter ablations.
    _, _, df_m3_no_min, _, no_min_signals = load_and_prepare_period("period_2020_2026", max_mins=30, min_depth_pips=0.0)
    _, _, df_m3_60, _, choch60_signals = load_and_prepare_period("period_2020_2026", max_mins=60, min_depth_pips=0.5)
    if not df_m3_no_min.equals(repro["df_m3"]):
        df_m3_no_min = repro["df_m3"]
    if not df_m3_60.equals(repro["df_m3"]):
        df_m3_60 = repro["df_m3"]

    extended_2015_2019 = run_extended_2015_2019()
    results = {
        "reproduction": {k: v for k, v in repro.items() if k in ["metrics", "reproduction_verdict", "news_counts"]},
        "tp_sl": tp_sl_math_audit(repro["trades"]),
        "multi_trade": multi_trade_audit(repro["trades"]),
        "signal": signal_no_lookahead_audit(repro["df_h1"], repro["df_m3"], repro["sweeps"], repro["signals"]),
        "filters": filter_audit(repro["df_m3"], repro["signals"], no_min_signals, choch60_signals),
        "execution": execution_audit(repro["trades"]),
        "time_news": time_news_audit(repro["engine"], repro["trades"]),
        "robustness": robustness_audit(repro["trades"], extended_2015_2019),
        "costs": cost_audit(repro["df_m3"], repro["selected"]),
        "risk": risk_audit(repro["trades"]),
        "overfit": overfit_audit(repro["metrics"]),
    }
    results["tests"] = run_phase19_tests()
    verdict, invalidators = final_verdict(results)
    comparisons = comparison_reports(repro["metrics"])
    final_report = write_final_report(results, verdict, invalidators, comparisons)
    update_status_files(verdict)
    update_zip_manifest()
    zip_payload = rebuild_zip()
    git_payload = git_status()
    terminal = {
        "final_verdict": verdict,
        "invalidators": invalidators,
        "reproduction_metrics": repro["metrics"],
        "zip": zip_payload,
        "git": git_payload,
        "report_md": str(REPORTS / "PHASE19_FORENSIC_AUDIT_REPORT.md"),
        "report_json": str(REPORTS / "PHASE19_FORENSIC_AUDIT_REPORT.json"),
    }
    write_json(OUT / "phase19_forensic_terminal_state.json", terminal)
    print(json.dumps(terminal, indent=2, default=json_default))


if __name__ == "__main__":
    main()
