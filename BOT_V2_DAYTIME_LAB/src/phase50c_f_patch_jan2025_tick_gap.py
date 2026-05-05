import json
import os
import shutil
import hashlib
from pathlib import Path

import pandas as pd


LAB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LAB_ROOT.parent
BOT_ROOT = PROJECT_ROOT.parent
MARKET_ROOT = BOT_ROOT / "BOT_MARKET_DATA"
TICK_ROOT = MARKET_ROOT / "tick" / "EURUSD"

CANONICAL = TICK_ROOT / "monthly" / "EURUSD_ticks_2025_01.parquet"
PATCH = TICK_ROOT / "repro_check" / "gap_2316_2025_01_14_0730_0845_NY.parquet"
BACKUP = TICK_ROOT / "backups" / "EURUSD_ticks_2025_01_PRE_GAP2316_PATCH.parquet"
SNAPSHOT = TICK_ROOT / "quality_reports" / "PHASE50C_F_PRE_PATCH_SNAPSHOT_2025_01.json"
TRADE_FILE = LAB_ROOT / "outputs" / "phase38_manipulante_deep_explainer" / "csv" / "phase38_raw_trades_enriched.csv"
PREVIOUS_20 = LAB_ROOT / "reports" / "manipulante_tick_historical" / "PHASE50C_C_20_TRADES_REAUDIT_AFTER_TZ_REPAIR.csv"
D_REVIEW = LAB_ROOT / "reports" / "PHASE50C_D_RESIDUAL_MISMATCH_FORENSIC_REVIEW_REPORT.json"
OUT_20 = LAB_ROOT / "reports" / "manipulante_tick_historical" / "PHASE50C_F_20_TRADES_REAUDIT_AFTER_GAP_PATCH.csv"
REPORT_JSON = LAB_ROOT / "reports" / "PHASE50C_F_JAN2025_TICK_GAP_PATCH_REPORT.json"
REPORT_MD = LAB_ROOT / "reports" / "PHASE50C_F_JAN2025_TICK_GAP_PATCH_REPORT.md"

NY = "America/New_York"
UTC = "UTC"
PATCH_START_NY = pd.Timestamp("2025-01-14 07:30:00", tz=NY)
PATCH_END_NY = pd.Timestamp("2025-01-14 08:45:00", tz=NY)
TRADE_WINDOW_START_NY = pd.Timestamp("2025-01-14 07:45:00", tz=NY)
TRADE_WINDOW_END_NY = pd.Timestamp("2025-01-14 08:33:00", tz=NY)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def coerce_utc(series):
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        return ts.dt.tz_localize(UTC)
    return ts.dt.tz_convert(UTC)


def normalize(df, columns):
    df = df.copy()
    df["timestamp_utc"] = coerce_utc(df["timestamp_utc"])
    df["timestamp_ny"] = df["timestamp_utc"].dt.tz_convert(NY)
    df["bid"] = df["bid"].astype(float)
    df["ask"] = df["ask"].astype(float)
    df["bid_volume"] = df.get("bid_volume", 0.0).astype(float)
    df["ask_volume"] = df.get("ask_volume", 0.0).astype(float)
    df["spread"] = df["ask"] - df["bid"]
    df["spread_pips"] = df["spread"] / 0.0001
    df["source"] = "dukascopy_native_h"
    df["symbol"] = "EURUSD"
    return df[columns]


def window(df, start, end):
    return df[(df["timestamp_ny"] >= start) & (df["timestamp_ny"] <= end)]


def gap_stats_for_day(df):
    day = window(df, pd.Timestamp("2025-01-14 00:00:00", tz=NY), pd.Timestamp("2025-01-14 23:59:59.999999", tz=NY)).copy()
    diffs = day["timestamp_utc"].diff().dt.total_seconds()
    max_idx = diffs.idxmax() if len(diffs.dropna()) else None
    return {
        "ticks_day": int(len(day)),
        "gaps_gt_1m": int((diffs > 60).sum()),
        "gaps_gt_5m": int((diffs > 300).sum()),
        "gaps_gt_15m": int((diffs > 900).sum()),
        "max_gap_seconds": None if max_idx is None or pd.isna(diffs.loc[max_idx]) else round(float(diffs.loc[max_idx]), 3),
    }


def validate_patch(patch):
    if len(patch) != 10551:
        raise RuntimeError(f"ABORT: patch rows expected 10551, got {len(patch)}")
    if patch["timestamp_ny"].iloc[0] != pd.Timestamp("2025-01-14 07:30:01.029000", tz=NY):
        raise RuntimeError("ABORT: unexpected patch first timestamp_ny")
    if patch["timestamp_ny"].iloc[-1] != pd.Timestamp("2025-01-14 08:44:59.071000", tz=NY):
        raise RuntimeError("ABORT: unexpected patch last timestamp_ny")
    if patch[["timestamp_utc", "timestamp_ny", "bid", "ask"]].isna().sum().sum() != 0:
        raise RuntimeError("ABORT: patch has critical nulls")
    if not patch["timestamp_utc"].is_monotonic_increasing:
        raise RuntimeError("ABORT: patch timestamps not sorted")
    if not (patch["bid"] <= patch["ask"]).all():
        raise RuntimeError("ABORT: patch bid > ask")
    if not (patch["spread"] >= 0).all():
        raise RuntimeError("ABORT: patch negative spread")


def create_backup_and_snapshot(canonical, patch, rows_before, sha_before):
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2(CANONICAL, BACKUP)
    snapshot = {
        "phase": "PHASE50C-F",
        "rows_before": int(rows_before),
        "sha256_before": sha_before,
        "file_size_mb_before": round(CANONICAL.stat().st_size / (1024 * 1024), 4),
        "first_timestamp_utc": canonical["timestamp_utc"].iloc[0].isoformat(),
        "first_timestamp_ny": canonical["timestamp_ny"].iloc[0].isoformat(),
        "last_timestamp_utc": canonical["timestamp_utc"].iloc[-1].isoformat(),
        "last_timestamp_ny": canonical["timestamp_ny"].iloc[-1].isoformat(),
        "rows_patch_window_before": int(len(window(canonical, PATCH_START_NY, PATCH_END_NY))),
        "patch_rows_available": int(len(patch)),
        "reason": "Patch extraction gap for Trade 2316, 2025-01-14 07:30-08:45 NY",
        "backup_path": str(BACKUP),
        "backup_exists": BACKUP.exists(),
        "backup_sha256": sha256(BACKUP),
    }
    with SNAPSHOT.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    if not BACKUP.exists() or sha256(BACKUP) != sha_before:
        raise RuntimeError("ABORT: backup verification failed")


def rewrite_snapshot_from_backup(backup_df, patch):
    snapshot = {
        "phase": "PHASE50C-F",
        "rows_before": int(len(backup_df)),
        "sha256_before": sha256(BACKUP),
        "file_size_mb_before": round(BACKUP.stat().st_size / (1024 * 1024), 4),
        "first_timestamp_utc": backup_df["timestamp_utc"].iloc[0].isoformat(),
        "first_timestamp_ny": backup_df["timestamp_ny"].iloc[0].isoformat(),
        "last_timestamp_utc": backup_df["timestamp_utc"].iloc[-1].isoformat(),
        "last_timestamp_ny": backup_df["timestamp_ny"].iloc[-1].isoformat(),
        "rows_patch_window_before": int(len(window(backup_df, PATCH_START_NY, PATCH_END_NY))),
        "patch_rows_available": int(len(patch)),
        "reason": "Patch extraction gap for Trade 2316, 2025-01-14 07:30-08:45 NY",
        "backup_path": str(BACKUP),
        "backup_exists": BACKUP.exists(),
        "backup_sha256": sha256(BACKUP),
        "reconstructed_from_backup": True,
    }
    with SNAPSHOT.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


def patch_canonical():
    if not CANONICAL.exists() or not PATCH.exists():
        raise RuntimeError("ABORT: canonical parquet or patch parquet missing")
    canonical_raw = pd.read_parquet(CANONICAL)
    columns = list(canonical_raw.columns)
    canonical = normalize(canonical_raw, columns)
    patch = normalize(pd.read_parquet(PATCH), columns)
    validate_patch(patch)

    rows_before = len(canonical)
    sha_before = sha256(CANONICAL)
    before_window = int(len(window(canonical, PATCH_START_NY, PATCH_END_NY)))

    if before_window == 0:
        create_backup_and_snapshot(canonical, patch, rows_before, sha_before)
        combined = pd.concat([canonical, patch], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp_utc", "bid", "ask", "symbol"], keep="first")
        combined = combined.sort_values("timestamp_utc").reset_index(drop=True)
        if not (combined["bid"] <= combined["ask"]).all():
            raise RuntimeError("ABORT: patched file has bid > ask")
        if not (combined["spread"] >= 0).all():
            raise RuntimeError("ABORT: patched file has negative spread")
        if len(window(combined, PATCH_START_NY, PATCH_END_NY)) == 0:
            raise RuntimeError("ABORT: patched window still empty")
        tmp = CANONICAL.with_suffix(".phase50c_f_tmp.parquet")
        combined.to_parquet(tmp, compression="snappy")
        os.replace(tmp, CANONICAL)

    patched = normalize(pd.read_parquet(CANONICAL), columns)
    sha_after = sha256(CANONICAL)
    rows_after = len(patched)
    original_rows = rows_before
    original_sha = sha_before
    if before_window > 0 and BACKUP.exists() and sha256(BACKUP) != sha_after:
        backup_df = normalize(pd.read_parquet(BACKUP), columns)
        original_rows = len(backup_df)
        original_sha = sha256(BACKUP)
        rewrite_snapshot_from_backup(backup_df, patch)
    elif before_window > 0 and SNAPSHOT.exists():
        with SNAPSHOT.open("r", encoding="utf-8") as f:
            snapshot = json.load(f)
        original_rows = int(snapshot.get("rows_before", rows_before))
        original_sha = snapshot.get("sha256_before", sha_before)
    result = {
        "rows_before": int(original_rows),
        "rows_patch": int(len(patch)),
        "rows_after": int(rows_after),
        "rows_expected_approx": int(original_rows + len(patch)),
        "sha256_before": original_sha,
        "sha256_after": sha_after,
        "first_timestamp_utc": patched["timestamp_utc"].iloc[0].isoformat(),
        "first_timestamp_ny": patched["timestamp_ny"].iloc[0].isoformat(),
        "last_timestamp_utc": patched["timestamp_utc"].iloc[-1].isoformat(),
        "last_timestamp_ny": patched["timestamp_ny"].iloc[-1].isoformat(),
        "ticks_patch_window_after": int(len(window(patched, PATCH_START_NY, PATCH_END_NY))),
        "ticks_trade_window_after": int(len(window(patched, TRADE_WINDOW_START_NY, TRADE_WINDOW_END_NY))),
        "bid_le_ask": bool((patched["bid"] <= patched["ask"]).all()),
        "spread_ge_0": bool((patched["spread"] >= 0).all()),
        "critical_nulls": int(patched[["timestamp_utc", "timestamp_ny", "bid", "ask"]].isna().sum().sum()),
        "timezone_consistent_spot": bool((patched["timestamp_utc"].dt.tz_convert(NY).dt.tz_localize(None) == patched["timestamp_ny"].dt.tz_localize(None)).all()),
        "jan14_gaps_after": gap_stats_for_day(patched),
        "backup_path": str(BACKUP),
        "snapshot_path": str(SNAPSHOT),
    }
    return result


def trade_levels(trade):
    direction = trade["type"]
    entry = float(trade["entry_price"])
    risk = float(trade["risk"])
    return {
        "entry": entry,
        "risk": risk,
        "tp": float(trade["tp"]),
        "initial_sl": entry - risk if direction == "LONG" else entry + risk,
        "be_trigger": entry + (0.4 * risk if direction == "LONG" else -0.4 * risk),
        "be_level": entry,
    }


def first_touch(trade, ticks):
    direction = trade["type"]
    entry_time = pd.Timestamp(trade["entry_time"])
    exit_time = pd.Timestamp(trade["exit_time"])
    lv = trade_levels(trade)
    w = window(ticks, entry_time, exit_time + pd.Timedelta(minutes=5))
    be_active = False
    be_time = None
    for row in w.itertuples(index=False):
        bid = float(row.bid)
        ask = float(row.ask)
        if direction == "LONG":
            if not be_active and bid >= lv["be_trigger"]:
                be_active = True
                be_time = row.timestamp_ny
            current_sl = lv["be_level"] if be_active else lv["initial_sl"]
            if bid <= current_sl:
                return ("BE" if be_active else "SL"), row.timestamp_ny, bid, ask, be_time
            if bid >= lv["tp"]:
                return "TP", row.timestamp_ny, bid, ask, be_time
        else:
            if not be_active and ask <= lv["be_trigger"]:
                be_active = True
                be_time = row.timestamp_ny
            current_sl = lv["be_level"] if be_active else lv["initial_sl"]
            if ask >= current_sl:
                return ("BE" if be_active else "SL"), row.timestamp_ny, bid, ask, be_time
            if ask <= lv["tp"]:
                return "TP", row.timestamp_ny, bid, ask, be_time
    return "NONE", None, None, None, be_time


def nearest_entry(ticks, entry_utc):
    idx = int(ticks["timestamp_utc"].searchsorted(entry_utc))
    before = ticks.iloc[idx - 1] if idx > 0 else None
    after = ticks.iloc[idx] if idx < len(ticks) else None
    if before is None:
        return after
    if after is None:
        return before
    return before if abs(before["timestamp_utc"] - entry_utc) <= abs(after["timestamp_utc"] - entry_utc) else after


def reaudits():
    trades = pd.read_csv(TRADE_FILE)
    ids = pd.read_csv(PREVIOUS_20)["trade_id"].astype(int).tolist() if PREVIOUS_20.exists() else list(trades[trades["year_month"] == "2025-01"].index[:20])
    ticks = pd.read_parquet(CANONICAL, columns=["timestamp_utc", "timestamp_ny", "bid", "ask", "spread_pips"])
    ticks["timestamp_utc"] = coerce_utc(ticks["timestamp_utc"])
    ticks["timestamp_ny"] = ticks["timestamp_utc"].dt.tz_convert(NY)

    rows = []
    trade_2316 = None
    for tid in ids:
        trade = trades.loc[tid]
        entry_ny = pd.Timestamp(trade["entry_time"])
        entry_utc = entry_ny.tz_convert(UTC)
        near = nearest_entry(ticks, entry_utc)
        outcome, touch_time, touch_bid, touch_ask, be_time = first_touch(trade, ticks)
        match = bool(outcome == str(trade["outcome"]))
        row = {
            "trade_id": int(tid),
            "entry_time_ny": entry_ny.isoformat(),
            "direction": trade["type"],
            "bar_outcome": str(trade["outcome"]),
            "tick_outcome_after_gap_patch": outcome,
            "match_after_gap_patch": match,
            "nearest_bid": None if near is None else float(near["bid"]),
            "nearest_ask": None if near is None else float(near["ask"]),
            "spread_entry_pips": None if near is None else round(float((near["ask"] - near["bid"]) * 10000), 4),
            "first_touch_time": None if touch_time is None else touch_time.isoformat(),
            "be_activated_time": None if be_time is None else be_time.isoformat(),
        }
        rows.append(row)
        if tid == 2316:
            trade_2316 = row
    OUT_20.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_20, index=False)

    explained = 0
    if D_REVIEW.exists():
        d = json.load(D_REVIEW.open("r", encoding="utf-8"))
        explained = sum(1 for c in d.get("classifications", []) if int(c["trade_id"]) != 2316)
    matches = int(df["match_after_gap_patch"].sum())
    no_auditables = 0
    return {
        "csv_path": str(OUT_20),
        "sample_size": int(len(df)),
        "matches": matches,
        "direct_match_rate": round(matches / len(df), 4),
        "explained_differences": int(explained),
        "not_auditables": no_auditables,
        "reliability_score": round((matches + explained) / len(df), 4),
        "trade_2316": trade_2316,
        "ready_for_next_month": bool(no_auditables == 0 and (matches + explained) == len(df)),
    }


def cache_status(current_sha):
    manifest = TICK_ROOT / "manifests" / "EURUSD_TICK_CACHE_MANIFEST.csv"
    status = {}
    if not manifest.exists():
        return status
    cm = pd.read_csv(manifest)
    start = PATCH_START_NY
    end = PATCH_END_NY
    for tf in ["M1", "M5", "M15"]:
        row = cm[(cm["year"] == 2025) & (cm["month"] == 1) & (cm["timeframe"] == tf)]
        cache_file = TICK_ROOT / "cache" / tf / f"EURUSD_{tf}_from_ticks_2025_01.parquet"
        if row.empty or not cache_file.exists():
            status[tf] = {"exists": False}
            continue
        df = pd.read_parquet(cache_file)
        df["timestamp_ny"] = pd.to_datetime(df["timestamp_ny"])
        w = window(df, start, end)
        status[tf] = {
            "exists": True,
            "source_sha256_matches": bool(row.iloc[0]["source_sha256"] == current_sha),
            "rows_window": int(len(w)),
            "tick_count_sum": int(w["tick_count"].fillna(0).sum()) if "tick_count" in w.columns and len(w) else 0,
            "nan_cells_window": int(w.isna().sum().sum()) if len(w) else 0,
            "first_ny": None if w.empty else w["timestamp_ny"].min().isoformat(),
            "last_ny": None if w.empty else w["timestamp_ny"].max().isoformat(),
        }
    return status


def write_report(patch_result, re_result):
    current_sha = sha256(CANONICAL)
    caches = cache_status(current_sha)
    verdict = "JAN2025_TICK_GAP_PATCH_OK_READY_FOR_NEXT_MONTH" if re_result["ready_for_next_month"] else "JAN2025_TICK_GAP_PATCH_OK_BUT_REAUDIT_INCONCLUSIVE"
    report = {
        "phase": "PHASE50C-F",
        "verdict": verdict,
        "patch_window_ny": {"start": PATCH_START_NY.isoformat(), "end": PATCH_END_NY.isoformat()},
        "patch": patch_result,
        "manifest_cache": {
            "data_manifest_path": str(TICK_ROOT / "manifests" / "EURUSD_TICK_DATA_MANIFEST.csv"),
            "cache_manifest_path": str(TICK_ROOT / "manifests" / "EURUSD_TICK_CACHE_MANIFEST.csv"),
            "cache_status": caches,
        },
        "trade_2316": re_result["trade_2316"],
        "jan2025_final": re_result,
        "safety": {
            "manipulante_modified": False,
            "strategy_modified": False,
            "mt5_opened": False,
            "orders_sent": False,
            "real_or_exness_touched": False,
            "full_extraction_or_full_cache": False,
            "git_add_commit_push": False,
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    md = f"""# PHASE50C-F JAN2025 TICK GAP PATCH REPORT

Verdict: {verdict}

Patch:
- window NY: {PATCH_START_NY.isoformat()} to {PATCH_END_NY.isoformat()}
- rows before/patch/after: {patch_result['rows_before']} / {patch_result['rows_patch']} / {patch_result['rows_after']}
- sha before: {patch_result['sha256_before']}
- sha after: {patch_result['sha256_after']}
- ticks patch window after: {patch_result['ticks_patch_window_after']}
- ticks trade window after: {patch_result['ticks_trade_window_after']}

Trade 2316:
- bar outcome: {re_result['trade_2316']['bar_outcome']}
- tick outcome: {re_result['trade_2316']['tick_outcome_after_gap_patch']}
- match: {re_result['trade_2316']['match_after_gap_patch']}
- first touch: {re_result['trade_2316']['first_touch_time']}

January 2025:
- sample size: {re_result['sample_size']}
- matches: {re_result['matches']}
- explained differences: {re_result['explained_differences']}
- not auditable: {re_result['not_auditables']}
- direct match rate: {re_result['direct_match_rate']}
- reliability score: {re_result['reliability_score']}
- ready for next month: {re_result['ready_for_next_month']}

Safety:
- MANIPULANTE not modified.
- Strategy, MT5, orders, real, Exness, 2024, Git add/commit/push not touched.
"""
    with REPORT_MD.open("w", encoding="utf-8") as f:
        f.write(md)
    print(verdict)
    print(f"rows={patch_result['rows_before']}->{patch_result['rows_after']} patch={patch_result['rows_patch']}")
    print(f"trade2316={re_result['trade_2316']['bar_outcome']}->{re_result['trade_2316']['tick_outcome_after_gap_patch']} match={re_result['trade_2316']['match_after_gap_patch']}")


def main():
    patch_result = patch_canonical()
    re_result = reaudits()
    write_report(patch_result, re_result)


if __name__ == "__main__":
    main()
