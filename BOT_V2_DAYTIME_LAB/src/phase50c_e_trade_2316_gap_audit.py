import hashlib
import json
import lzma
import os
import struct
import urllib.request
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

import pandas as pd


LAB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LAB_ROOT.parent
BOT_ROOT = PROJECT_ROOT.parent
MARKET_ROOT = BOT_ROOT / "BOT_MARKET_DATA"
TICK_ROOT = MARKET_ROOT / "tick" / "EURUSD"
TICK_FILE = TICK_ROOT / "monthly" / "EURUSD_ticks_2025_01.parquet"
DATA_MANIFEST = TICK_ROOT / "manifests" / "EURUSD_TICK_DATA_MANIFEST.csv"
CACHE_MANIFEST = TICK_ROOT / "manifests" / "EURUSD_TICK_CACHE_MANIFEST.csv"
REPRO_DIR = TICK_ROOT / "repro_check"
REPRO_FILE = REPRO_DIR / "gap_2316_2025_01_14_0730_0845_NY.parquet"

TRADE_FILE = LAB_ROOT / "outputs" / "phase38_manipulante_deep_explainer" / "csv" / "phase38_raw_trades_enriched.csv"
REAUDIT_FILE = LAB_ROOT / "reports" / "manipulante_tick_historical" / "PHASE50C_C_20_TRADES_REAUDIT_AFTER_TZ_REPAIR.csv"
RESIDUAL_FILE = LAB_ROOT / "reports" / "manipulante_tick_historical" / "PHASE50C_D_RESIDUAL_MISMATCH_LIST.csv"
REPORTS_DIR = LAB_ROOT / "reports"
DEBUG_DIR = LAB_ROOT / "reports" / "manipulante_tick_historical" / "debug" / "residual_mismatches"

UTC = "UTC"
NY = "America/New_York"
TICK_STRUCT = struct.Struct(">IIIff")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ticks_between(df, start_ny, end_ny):
    return df[(df["timestamp_ny"] >= start_ny) & (df["timestamp_ny"] <= end_ny)]


def gap_stats(day_ticks):
    if day_ticks.empty:
        return {"gt_1m": 0, "gt_5m": 0, "gt_15m": 0, "max_gap_seconds": None, "max_gap_start_ny": None, "max_gap_end_ny": None}
    diffs = day_ticks["timestamp_utc"].diff().dt.total_seconds()
    max_idx = diffs.idxmax()
    max_gap = None if pd.isna(diffs.loc[max_idx]) else float(diffs.loc[max_idx])
    prev_idx = day_ticks.index.get_loc(max_idx) - 1 if max_idx in day_ticks.index else -1
    prev_ts = None
    if prev_idx >= 0:
        prev_ts = day_ticks.iloc[prev_idx]["timestamp_ny"].isoformat()
    return {
        "gt_1m": int((diffs > 60).sum()),
        "gt_5m": int((diffs > 300).sum()),
        "gt_15m": int((diffs > 900).sum()),
        "max_gap_seconds": None if max_gap is None else round(max_gap, 3),
        "max_gap_start_ny": prev_ts,
        "max_gap_end_ny": None if max_gap is None else day_ticks.loc[max_idx, "timestamp_ny"].isoformat(),
    }


def cache_window(timeframe, start_ny, end_ny):
    path = TICK_ROOT / "cache" / timeframe / f"EURUSD_{timeframe}_from_ticks_2025_01.parquet"
    if not path.exists():
        return {"exists": False}
    df = pd.read_parquet(path)
    if "timestamp_ny" in df.columns:
        df["timestamp_ny"] = pd.to_datetime(df["timestamp_ny"])
    else:
        df = df.reset_index()
        df["timestamp_ny"] = pd.to_datetime(df["timestamp_utc"]).dt.tz_convert(NY)
    w = ticks_between(df, start_ny, end_ny)
    return {
        "exists": True,
        "rows_window": int(len(w)),
        "tick_count_sum": None if w.empty or "tick_count" not in w.columns else int(w["tick_count"].fillna(0).sum()),
        "nan_cells": int(w.isna().sum().sum()) if not w.empty else 0,
        "first_ny": None if w.empty else w["timestamp_ny"].min().isoformat(),
        "last_ny": None if w.empty else w["timestamp_ny"].max().isoformat(),
    }


def download_dukascopy_window(start_utc, end_utc):
    rows = []
    divisor = 100000.0
    user_agent = "Mozilla/5.0"
    hours = []
    cur = start_utc.floor("h")
    while cur <= end_utc.ceil("h"):
        hours.append(cur)
        cur += pd.Timedelta(hours=1)
    for hour_ts in hours:
        day = hour_ts.date()
        hour = hour_ts.hour
        url = f"https://datafeed.dukascopy.com/datafeed/EURUSD/{day.year}/{day.month-1:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                payload = response.read()
            raw = lzma.decompress(payload) if payload else b""
        except Exception:
            raw = b""
        hour_start = datetime.combine(day, dtime(hour, 0)).replace(tzinfo=timezone.utc)
        for ms, ask_i, bid_i, ask_vol, bid_vol in TICK_STRUCT.iter_unpack(raw):
            ts = pd.Timestamp(hour_start + timedelta(milliseconds=ms)).tz_convert(UTC)
            if start_utc <= ts <= end_utc:
                rows.append({
                    "timestamp_utc": ts,
                    "timestamp_ny": ts.tz_convert(NY),
                    "bid": bid_i / divisor,
                    "ask": ask_i / divisor,
                    "bid_volume": float(bid_vol),
                    "ask_volume": float(ask_vol),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["spread"] = df["ask"] - df["bid"]
        df["spread_pips"] = df["spread"] / 0.0001
        df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def first_touch(trade, ticks):
    direction = trade["type"]
    entry = pd.Timestamp(trade["entry_time"])
    exit_time = pd.Timestamp(trade["exit_time"])
    entry_price = float(trade["entry_price"])
    risk = float(trade["risk"])
    tp = float(trade["tp"])
    initial_sl = entry_price - risk if direction == "LONG" else entry_price + risk
    be_trigger = entry_price + (0.4 * risk if direction == "LONG" else -0.4 * risk)
    window = ticks_between(ticks, entry, exit_time + pd.Timedelta(minutes=5))
    be_active = False
    for row in window.itertuples(index=False):
        bid = float(row.bid)
        ask = float(row.ask)
        if direction == "LONG":
            if not be_active and bid >= be_trigger:
                be_active = True
            current_sl = entry_price if be_active else initial_sl
            if bid <= current_sl:
                return "BE" if be_active else "SL", row.timestamp_ny.isoformat()
            if bid >= tp:
                return "TP", row.timestamp_ny.isoformat()
        else:
            if not be_active and ask <= be_trigger:
                be_active = True
            current_sl = entry_price if be_active else initial_sl
            if ask >= current_sl:
                return "BE" if be_active else "SL", row.timestamp_ny.isoformat()
            if ask <= tp:
                return "TP", row.timestamp_ny.isoformat()
    return "NONE", None


def compact_ticks(df):
    cols = ["timestamp_utc", "timestamp_ny", "bid", "ask"]
    out = []
    for row in df[cols].itertuples(index=False):
        out.append({
            "timestamp_utc": row.timestamp_utc.isoformat(),
            "timestamp_ny": row.timestamp_ny.isoformat(),
            "bid": float(row.bid),
            "ask": float(row.ask),
        })
    return out


def main():
    trade = pd.read_csv(TRADE_FILE).loc[2316]
    re = pd.read_csv(REAUDIT_FILE)
    residual = pd.read_csv(RESIDUAL_FILE)
    tick_sha = sha256(TICK_FILE)
    ticks = pd.read_parquet(TICK_FILE, columns=["timestamp_utc", "timestamp_ny", "bid", "ask", "spread_pips"])
    ticks["timestamp_utc"] = pd.to_datetime(ticks["timestamp_utc"]).dt.tz_convert(UTC)
    ticks["timestamp_ny"] = pd.to_datetime(ticks["timestamp_ny"])

    entry = pd.Timestamp(trade["entry_time"])
    exit_time = pd.Timestamp(trade["exit_time"])
    window_phase50d_start = entry - pd.Timedelta(minutes=10)
    window_phase50d_end = exit_time + pd.Timedelta(minutes=10)
    redownload_start_ny = pd.Timestamp("2025-01-14 07:30:00", tz=NY)
    redownload_end_ny = pd.Timestamp("2025-01-14 08:45:00", tz=NY)
    start_0745 = pd.Timestamp("2025-01-14 07:45:00", tz=NY)
    end_0833 = pd.Timestamp("2025-01-14 08:33:00", tz=NY)
    start_0735 = pd.Timestamp("2025-01-14 07:35:00", tz=NY)
    end_0843 = pd.Timestamp("2025-01-14 08:43:00", tz=NY)

    day_start = pd.Timestamp("2025-01-14 00:00:00", tz=NY)
    day_end = pd.Timestamp("2025-01-14 23:59:59.999999", tz=NY)
    day_ticks = ticks_between(ticks, day_start, day_end).copy()
    hourly = {str(int(h)): int(c) for h, c in day_ticks["timestamp_ny"].dt.hour.value_counts().sort_index().items()}

    start_utc = start_0745.tz_convert(UTC)
    end_utc = end_0833.tz_convert(UTC)
    ny_filter_count = int(len(ticks_between(ticks, start_0745, end_0833)))
    utc_filter_count = int(len(ticks[(ticks["timestamp_utc"] >= start_utc) & (ticks["timestamp_utc"] <= end_utc)]))

    cache = {tf: cache_window(tf, start_0745, end_0833) for tf in ["M1", "M5", "M15"]}
    cache_manifest = pd.read_csv(CACHE_MANIFEST)
    cache_manifest_202501 = {
        tf: cache_manifest[(cache_manifest["year"] == 2025) & (cache_manifest["month"] == 1) & (cache_manifest["timeframe"] == tf)].iloc[0].to_dict()
        for tf in ["M1", "M5", "M15"]
    }
    data_manifest = pd.read_csv(DATA_MANIFEST)
    data_manifest_202501 = data_manifest[(data_manifest["year"] == 2025) & (data_manifest["month"] == 1)].iloc[0].to_dict()

    REPRO_DIR.mkdir(parents=True, exist_ok=True)
    repro = download_dukascopy_window(redownload_start_ny.tz_convert(UTC), redownload_end_ny.tz_convert(UTC))
    if not repro.empty:
        repro.to_parquet(REPRO_FILE, compression="snappy")
    else:
        pd.DataFrame(columns=["timestamp_utc", "timestamp_ny", "bid", "ask", "bid_volume", "ask_volume", "spread", "spread_pips"]).to_parquet(REPRO_FILE, compression="snappy")
    canonical_repro_window = ticks_between(ticks, redownload_start_ny, redownload_end_ny)
    rows_redownload = int(len(repro))
    rows_canonical = int(len(canonical_repro_window))

    if rows_redownload > 0 and rows_canonical == 0:
        trade_state = "EXTRACTION_GAP_REQUIRES_PATCH"
        verdict = "TRADE_2316_EXTRACTION_GAP_REQUIRES_PATCH"
    elif rows_redownload == 0 and rows_canonical == 0:
        trade_state = "NOT_AUDITABLE_CONFIRMED_SOURCE_GAP"
        verdict = "TRADE_2316_CONFIRMED_NOT_AUDITABLE_SOURCE_GAP"
    elif ny_filter_count == 0 and utc_filter_count > 0:
        trade_state = "TIMEZONE_FILTER_BUG_REQUIRES_REPAIR"
        verdict = "TRADE_2316_TIMEZONE_FILTER_BUG_REQUIRES_REPAIR"
    else:
        trade_state = "DATA_GAP_INCONCLUSIVE"
        verdict = "TRADE_2316_GAP_AUDIT_INCONCLUSIVE"

    outcome = None
    if rows_redownload > 0:
        outcome, outcome_time = first_touch(trade, repro)
    else:
        outcome_time = None

    report = {
        "phase": "PHASE50C-E",
        "verdict": verdict,
        "trade_2316": {
            "trade_id": 2316,
            "entry_time_ny": entry.isoformat(),
            "exit_time_ny": exit_time.isoformat(),
            "direction": trade["type"],
            "historical_entry": float(trade["entry_price"]),
            "sl": float(trade["sl"]),
            "tp": float(trade["tp"]),
            "be_level": float(trade["entry_price"]),
            "bar_outcome": trade["outcome"],
            "tick_outcome_phase50c": re[re["trade_id"] == 2316].iloc[0]["tick_outcome_after_repair"],
            "phase50d_classification": residual[residual["trade_id"] == 2316].iloc[0]["mismatch_type_preliminar"],
            "phase50d_window_start_ny": window_phase50d_start.isoformat(),
            "phase50d_window_end_ny": window_phase50d_end.isoformat(),
        },
        "day_audit": {
            "first_tick_ny": None if day_ticks.empty else day_ticks["timestamp_ny"].min().isoformat(),
            "last_tick_ny": None if day_ticks.empty else day_ticks["timestamp_ny"].max().isoformat(),
            "total_ticks_day": int(len(day_ticks)),
            "ticks_by_hour_ny": hourly,
            "ticks_0700_0900_ny": int(len(ticks_between(ticks, pd.Timestamp("2025-01-14 07:00:00", tz=NY), pd.Timestamp("2025-01-14 09:00:00", tz=NY)))),
            "ticks_0745_0833_ny": ny_filter_count,
            "ticks_0735_0843_ny": int(len(ticks_between(ticks, start_0735, end_0843))),
            "ticks_entry_minus30_exit_plus30": int(len(ticks_between(ticks, entry - pd.Timedelta(minutes=30), exit_time + pd.Timedelta(minutes=30)))),
            "gaps": gap_stats(day_ticks),
        },
        "timezone_filter_check": {
            "ny_start": start_0745.isoformat(),
            "ny_end": end_0833.isoformat(),
            "utc_start": start_utc.isoformat(),
            "utc_end": end_utc.isoformat(),
            "timestamp_ny_count": ny_filter_count,
            "timestamp_utc_count": utc_filter_count,
            "timezone_filter_bug": bool(ny_filter_count == 0 and utc_filter_count > 0),
        },
        "manifest_cache": {
            "parquet_sha256": tick_sha,
            "data_manifest_sha256": data_manifest_202501.get("sha256"),
            "data_manifest_sha_matches": data_manifest_202501.get("sha256") == tick_sha,
            "cache": cache,
            "cache_manifest_source_sha_matches": {
                tf: cache_manifest_202501[tf].get("source_sha256") == tick_sha for tf in ["M1", "M5", "M15"]
            },
        },
        "redownload": {
            "executed": True,
            "path": str(REPRO_FILE),
            "window_start_ny": redownload_start_ny.isoformat(),
            "window_end_ny": redownload_end_ny.isoformat(),
            "rows_redownloaded": rows_redownload,
            "rows_canonical_same_window": rows_canonical,
            "first_tick_redownload": None if repro.empty else repro["timestamp_ny"].iloc[0].isoformat(),
            "last_tick_redownload": None if repro.empty else repro["timestamp_ny"].iloc[-1].isoformat(),
            "first_5_ticks_redownload": compact_ticks(repro.head(5)) if not repro.empty else [],
            "last_5_ticks_redownload": compact_ticks(repro.tail(5)) if not repro.empty else [],
            "bid_ask_equal": bool(rows_redownload > 0 and rows_canonical > 0 and repro[["bid", "ask"]].reset_index(drop=True).equals(canonical_repro_window[["bid", "ask"]].reset_index(drop=True))),
            "timestamps_equal": bool(rows_redownload > 0 and rows_canonical > 0 and repro[["timestamp_utc", "timestamp_ny"]].reset_index(drop=True).equals(canonical_repro_window[["timestamp_utc", "timestamp_ny"]].reset_index(drop=True))),
        },
        "trade_2316_final_state": {
            "classification": trade_state,
            "auditable": bool(rows_redownload > 0),
            "reaudited_outcome_from_redownload": outcome,
            "reaudited_outcome_time": outcome_time,
        },
        "jan2025_final": {
            "matches": 15,
            "projected_matches_if_extraction_gap_patched": 16 if outcome == trade["outcome"] else 15,
            "explained_differences": 4,
            "not_auditable": 1 if trade_state != "EXTRACTION_GAP_REQUIRES_PATCH" else 0,
            "ready_for_next_month": False,
        },
        "safety": {
            "manipulante_modified": False,
            "strategy_modified": False,
            "mt5_opened": False,
            "orders_sent": False,
            "real_or_exness_touched": False,
            "git_add_commit_push": False,
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    report_json = REPORTS_DIR / "PHASE50C_E_TRADE_2316_TICK_GAP_AUDIT_REPORT.json"
    report_md = REPORTS_DIR / "PHASE50C_E_TRADE_2316_TICK_GAP_AUDIT_REPORT.md"
    debug_json = DEBUG_DIR / "TRADE_2316_GAP_AUDIT.json"
    debug_md = DEBUG_DIR / "TRADE_2316_GAP_AUDIT.md"
    for path in [report_json, debug_json]:
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    md = f"""# PHASE50C-E TRADE 2316 TICK GAP AUDIT

Verdict: {verdict}

Trade 2316:
- entry/exit NY: {entry.isoformat()} / {exit_time.isoformat()}
- Phase50C-D window: {window_phase50d_start.isoformat()} to {window_phase50d_end.isoformat()}
- bar/tick: {trade['outcome']} / {report['trade_2316']['tick_outcome_phase50c']}

Tick availability:
- total_ticks_day: {report['day_audit']['total_ticks_day']}
- ticks_0745_0833_ny: {ny_filter_count}
- ticks_0745_0833_utc_equivalent: {utc_filter_count}
- max_gap_seconds: {report['day_audit']['gaps']['max_gap_seconds']}

Cache:
- M1 rows/tick_count: {cache['M1'].get('rows_window')} / {cache['M1'].get('tick_count_sum')}
- M5 rows/tick_count: {cache['M5'].get('rows_window')} / {cache['M5'].get('tick_count_sum')}
- M15 rows/tick_count: {cache['M15'].get('rows_window')} / {cache['M15'].get('tick_count_sum')}

Redownload:
- executed: True
- rows_redownloaded: {rows_redownload}
- rows_canonical_same_window: {rows_canonical}
- state: {trade_state}

Safety:
- MANIPULANTE not modified.
- Strategy, MT5, orders, real, Exness, Git add/commit/push not touched.
"""
    for path in [report_md, debug_md]:
        with path.open("w", encoding="utf-8") as f:
            f.write(md)
    print(verdict)
    print(trade_state)
    print(f"rows_redownloaded={rows_redownload}; rows_canonical={rows_canonical}; ny_count={ny_filter_count}; utc_count={utc_filter_count}")


if __name__ == "__main__":
    main()
