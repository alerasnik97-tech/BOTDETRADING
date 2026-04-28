from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PIP = 0.0001
NY_TZ = "America/New_York"
CERTIFIED = "M3_FROM_M1_BID_ASK_CERTIFIED"
BLOCKED = "M3_CERTIFICATION_BLOCKED"


class M3CertificationError(RuntimeError):
    pass


def read_price_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise M3CertificationError(f"missing columns in {path}: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.reset_index(drop=True)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_source(path: str | Path) -> str:
    name = Path(path).name.upper()
    if "M1" in name:
        return "m1"
    if "TICK" in name or Path(path).suffix.lower() == ".bi5":
        return "tick_or_raw_cache"
    if "M3" in name:
        return "m3"
    if "M5" in name:
        return "m5"
    if "H1" in name:
        return "h1"
    if "NEWS" in name:
        return "news"
    return "unknown"


def reject_source_for_m3(source_type: str, bid_present: bool, ask_present: bool) -> str:
    if source_type == "m1" and bid_present and ask_present:
        return "SOURCE_VALID_FOR_M3_CERTIFICATION"
    if source_type == "tick_or_raw_cache" and bid_present and ask_present:
        return "SOURCE_REQUIRES_REPAIR"
    if source_type in {"m5", "h1", "m3"}:
        return "SOURCE_REJECTED"
    return "SOURCE_MISSING"


def ohlc_integrity(df: pd.DataFrame) -> dict:
    high_bad = (df["high"] < df[["open", "close", "low"]].max(axis=1)).sum()
    low_bad = (df["low"] > df[["open", "close", "high"]].min(axis=1)).sum()
    return {
        "rows": int(len(df)),
        "invalid_high_count": int(high_bad),
        "invalid_low_count": int(low_bad),
        "valid": int(high_bad) == 0 and int(low_bad) == 0,
    }


def critical_gap_mask(prev_ts: pd.Series, next_ts: pd.Series) -> pd.Series:
    prev_ny = prev_ts.dt.tz_convert(NY_TZ)
    next_ny = next_ts.dt.tz_convert(NY_TZ)
    same_date = prev_ny.dt.date == next_ny.dt.date
    weekday = prev_ny.dt.dayofweek < 5
    prev_hour = prev_ny.dt.hour + prev_ny.dt.minute / 60.0
    next_hour = next_ny.dt.hour + next_ny.dt.minute / 60.0
    overlaps_day_window = (prev_hour < 20.0) & (next_hour >= 7.0)
    return same_date & weekday & overlaps_day_window


def gap_report(ts: pd.Series, expected_minutes: int) -> pd.DataFrame:
    ts = pd.to_datetime(ts, utc=True).sort_values().reset_index(drop=True)
    delta_min = ts.diff().dt.total_seconds().div(60)
    gaps = pd.DataFrame(
        {
            "prev_timestamp": ts.shift(1),
            "next_timestamp": ts,
            "gap_minutes": delta_min,
        }
    )
    gaps = gaps[gaps["gap_minutes"] > expected_minutes].copy()
    if gaps.empty:
        gaps["critical_session_gap"] = []
        return gaps
    gaps["critical_session_gap"] = critical_gap_mask(gaps["prev_timestamp"], gaps["next_timestamp"])
    gaps["prev_timestamp_ny"] = gaps["prev_timestamp"].dt.tz_convert(NY_TZ)
    gaps["next_timestamp_ny"] = gaps["next_timestamp"].dt.tz_convert(NY_TZ)
    return gaps


def audit_m1_source(bid: pd.DataFrame, ask: pd.DataFrame) -> dict:
    merged = pd.merge(
        bid[["timestamp", "open", "high", "low", "close"]].rename(
            columns={c: f"{c}_bid" for c in ["open", "high", "low", "close"]}
        ),
        ask[["timestamp", "open", "high", "low", "close"]].rename(
            columns={c: f"{c}_ask" for c in ["open", "high", "low", "close"]}
        ),
        on="timestamp",
        how="inner",
    )
    spread_cols = {}
    for col in ["open", "high", "low", "close"]:
        spread_cols[col] = (merged[f"{col}_ask"] - merged[f"{col}_bid"]) / PIP
    spread = pd.DataFrame(spread_cols)
    bid_integrity = ohlc_integrity(bid)
    ask_integrity = ohlc_integrity(ask)
    bid_dupes = int(bid["timestamp"].duplicated().sum())
    ask_dupes = int(ask["timestamp"].duplicated().sum())
    merged_gaps = gap_report(merged["timestamp"], 1)
    invalid_spread = int((spread < 0).any(axis=1).sum())
    absurd_spread = int((spread["close"] > 50).sum())
    summary = {
        "source_type": "m1_derived",
        "bid_rows": int(len(bid)),
        "ask_rows": int(len(ask)),
        "merged_rows": int(len(merged)),
        "start": merged["timestamp"].min().isoformat() if not merged.empty else None,
        "end": merged["timestamp"].max().isoformat() if not merged.empty else None,
        "timezone": "UTC",
        "bid_monotonic": bool(bid["timestamp"].is_monotonic_increasing),
        "ask_monotonic": bool(ask["timestamp"].is_monotonic_increasing),
        "bid_duplicates": bid_dupes,
        "ask_duplicates": ask_dupes,
        "timestamp_mismatches": int(max(len(bid), len(ask)) - len(merged)),
        "bid_ohlc_valid": bid_integrity["valid"],
        "ask_ohlc_valid": ask_integrity["valid"],
        "spread_negative_rows": invalid_spread,
        "spread_absurd_close_gt_50_pips": absurd_spread,
        "spread_close_mean_pips": float(spread["close"].mean()) if not spread.empty else None,
        "spread_close_p95_pips": float(spread["close"].quantile(0.95)) if not spread.empty else None,
        "spread_close_max_pips": float(spread["close"].max()) if not spread.empty else None,
        "gap_count": int(len(merged_gaps)),
        "critical_gap_count": int(merged_gaps["critical_session_gap"].sum()) if not merged_gaps.empty else 0,
    }
    critical_ok = (
        summary["bid_monotonic"]
        and summary["ask_monotonic"]
        and bid_dupes == 0
        and ask_dupes == 0
        and summary["timestamp_mismatches"] == 0
        and summary["bid_ohlc_valid"]
        and summary["ask_ohlc_valid"]
        and invalid_spread == 0
    )
    summary["verdict"] = "SOURCE_VALID_FOR_M3_CERTIFICATION" if critical_ok else "SOURCE_REQUIRES_REPAIR"
    summary["gap_warning"] = summary["critical_gap_count"] > 0
    return summary


def resample_m1_to_m3(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.set_index("timestamp").sort_index()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in d.columns:
        agg["volume"] = "sum"
    out = d.resample("3min", label="left", closed="left").agg(agg).dropna(subset=["open", "high", "low", "close"])
    out = out.reset_index()
    return out


def build_m3_from_m1(bid_path: str | Path, ask_path: str | Path, output_dir: str | Path) -> dict:
    bid = read_price_csv(bid_path)
    ask = read_price_csv(ask_path)
    source_audit = audit_m1_source(bid, ask)
    if source_audit["verdict"] != "SOURCE_VALID_FOR_M3_CERTIFICATION":
        raise M3CertificationError("M1 source failed critical audit; M3 generation blocked")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    m3_bid = resample_m1_to_m3(bid)
    m3_ask = resample_m1_to_m3(ask)
    merged = pd.merge(
        m3_bid[["timestamp", "open", "high", "low", "close"]].rename(
            columns={c: f"{c}_bid" for c in ["open", "high", "low", "close"]}
        ),
        m3_ask[["timestamp", "open", "high", "low", "close"]].rename(
            columns={c: f"{c}_ask" for c in ["open", "high", "low", "close"]}
        ),
        on="timestamp",
        how="inner",
    )
    spread = pd.DataFrame({"timestamp": merged["timestamp"]})
    for col in ["open", "high", "low", "close"]:
        spread[f"spread_{col}_pips"] = (merged[f"{col}_ask"] - merged[f"{col}_bid"]) / PIP
    bid_out = out / "EURUSD_M3_BID_2020_2026.csv"
    ask_out = out / "EURUSD_M3_ASK_2020_2026.csv"
    spread_out = out / "EURUSD_M3_SPREAD_2020_2026.csv"
    m3_bid.to_csv(bid_out, index=False)
    m3_ask.to_csv(ask_out, index=False)
    spread.to_csv(spread_out, index=False)
    metadata = {
        "certification_name": CERTIFIED,
        "source_type": "m1_derived",
        "source_bid": str(bid_path),
        "source_ask": str(ask_path),
        "bid_path": str(bid_out),
        "ask_path": str(ask_out),
        "spread_path": str(spread_out),
        "rows": int(len(m3_bid)),
        "start": m3_bid["timestamp"].min().isoformat(),
        "end": m3_bid["timestamp"].max().isoformat(),
        "timezone": "UTC",
        "source_audit": source_audit,
        "bid_sha256": file_sha256(bid_out),
        "ask_sha256": file_sha256(ask_out),
        "spread_sha256": file_sha256(spread_out),
    }
    with (out / "M3_CERTIFICATION_METADATA.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    return metadata


def validate_m3_files(bid_path: str | Path, ask_path: str | Path) -> dict:
    bid = read_price_csv(bid_path)
    ask = read_price_csv(ask_path)
    merged = pd.merge(
        bid[["timestamp", "open", "high", "low", "close"]].rename(
            columns={c: f"{c}_bid" for c in ["open", "high", "low", "close"]}
        ),
        ask[["timestamp", "open", "high", "low", "close"]].rename(
            columns={c: f"{c}_ask" for c in ["open", "high", "low", "close"]}
        ),
        on="timestamp",
        how="inner",
    )
    spread_cols = {}
    for col in ["open", "high", "low", "close"]:
        spread_cols[col] = (merged[f"{col}_ask"] - merged[f"{col}_bid"]) / PIP
    spread = pd.DataFrame(spread_cols)
    gaps = gap_report(pd.to_datetime(bid["timestamp"], utc=True), 3)
    bid_integrity = ohlc_integrity(bid)
    ask_integrity = ohlc_integrity(ask)
    bid_dupes = int(bid["timestamp"].duplicated().sum())
    ask_dupes = int(ask["timestamp"].duplicated().sum())
    invalid_spread = int((spread < 0).any(axis=1).sum())
    m3_critical_gap_count = int(gaps["critical_session_gap"].sum()) if not gaps.empty else 0
    price_ok = (
        bool(bid["timestamp"].is_monotonic_increasing)
        and bool(ask["timestamp"].is_monotonic_increasing)
        and bid_dupes == 0
        and ask_dupes == 0
        and int(max(len(bid), len(ask)) - len(merged)) == 0
        and bid_integrity["valid"]
        and ask_integrity["valid"]
        and invalid_spread == 0
    )
    return {
        "source_type": "m1_derived",
        "timeframe": "m3",
        "bid_rows": int(len(bid)),
        "ask_rows": int(len(ask)),
        "merged_rows": int(len(merged)),
        "start": merged["timestamp"].min().isoformat() if not merged.empty else None,
        "end": merged["timestamp"].max().isoformat() if not merged.empty else None,
        "timezone": "UTC",
        "bid_monotonic": bool(bid["timestamp"].is_monotonic_increasing),
        "ask_monotonic": bool(ask["timestamp"].is_monotonic_increasing),
        "bid_duplicates": bid_dupes,
        "ask_duplicates": ask_dupes,
        "timestamp_mismatches": int(max(len(bid), len(ask)) - len(merged)),
        "bid_ohlc_valid": bid_integrity["valid"],
        "ask_ohlc_valid": ask_integrity["valid"],
        "spread_negative_rows": invalid_spread,
        "spread_absurd_close_gt_50_pips": int((spread["close"] > 50).sum()),
        "spread_close_mean_pips": float(spread["close"].mean()) if not spread.empty else None,
        "spread_close_p95_pips": float(spread["close"].quantile(0.95)) if not spread.empty else None,
        "spread_close_max_pips": float(spread["close"].max()) if not spread.empty else None,
        "m3_gap_count": int(len(gaps)),
        "m3_critical_gap_count": m3_critical_gap_count,
        "verdict": "M3_BID_ASK_CERTIFIED" if price_ok and m3_critical_gap_count == 0 else "M3_REQUIRES_REPAIR",
    }
