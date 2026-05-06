import os
import sys
import pandas as pd
import numpy as np
import pytz
import json
import hashlib
import lzma
import struct
import argparse
from datetime import datetime, timedelta, date, time as pytime
from pathlib import Path

ROOT_PROJECT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
MARKET_DATA_ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA"
TICK_PATH = os.path.join(MARKET_DATA_ROOT, "tick", "EURUSD")
CHECKPOINT_PATH = os.path.join(TICK_PATH, "checkpoints")
PARTIAL_PATH = os.path.join(TICK_PATH, "partial")
SUBFOLDERS = ["raw", "processed", "monthly", "checkpoints", "partial", "manifests", "quality_reports", "logs"]

UTC = pytz.UTC
NY = pytz.timezone("America/New_York")
TICK_STRUCT = struct.Struct(">IIIff")

def _ensure_dirs():
    for sub in SUBFOLDERS:
        os.makedirs(os.path.join(TICK_PATH, sub), exist_ok=True)

def _coerce_timestamp_utc(series):
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert("UTC")

def _get_sha256(file_path):
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha.update(block)
    return sha.hexdigest()

class DukascopyResumable:
    def __init__(self, symbol="EURUSD"):
        self.symbol = symbol
        self.divisor = 100000.0 if "JPY" not in symbol else 1000.0
        
    def download_hour(self, day, hour):
        url = f"https://datafeed.dukascopy.com/datafeed/{self.symbol}/{day.year}/{day.month-1:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                return lzma.decompress(data) if data else None
        except: return None

    def download_day(self, day):
        all_ticks = []
        for hour in range(24):
            raw = self.download_hour(day, hour)
            if not raw: continue
            hour_start = datetime.combine(day, pytime(hour, 0), tzinfo=UTC)
            try:
                for ms, ask_i, bid_i, ask_vol, bid_vol in TICK_STRUCT.iter_unpack(raw):
                    all_ticks.append({
                        "timestamp_utc": hour_start + timedelta(milliseconds=ms),
                        "bid": bid_i / self.divisor,
                        "ask": ask_i / self.divisor,
                        "bid_volume": bid_vol,
                        "ask_volume": ask_vol
                    })
            except: continue
        return pd.DataFrame(all_ticks)

def load_checkpoint(year, month):
    ck_file = os.path.join(CHECKPOINT_PATH, f"{year}_{month:02d}_checkpoint.json")
    if os.path.exists(ck_file):
        with open(ck_file, "r") as f:
            return json.load(f)
    return {"year": year, "month": month, "days": {}}

def save_checkpoint(year, month, data):
    ck_file = os.path.join(CHECKPOINT_PATH, f"{year}_{month:02d}_checkpoint.json")
    with open(ck_file, "w") as f:
        json.dump(data, f, indent=2)

def download_partial(year, month, day):
    day_str = day.strftime("%Y%m%d")
    partial_file = os.path.join(PARTIAL_PATH, f"{year}_{month:02d}", f"{day_str}.parquet")
    return partial_file, os.path.exists(partial_file)

def mode_diagnostic(year, month):
    print(f"[DIAGNOSTIC] Year: {year}, Month: {month}")
    ck = load_checkpoint(year, month)
    completed = sum(1 for v in ck.get("days", {}).values() if v.get("status") == "OK")
    pending = 0
    for d in range(1, 32):
        try:
            dt = date(year, month, d)
            if dt.month != month: continue
            day_str = dt.strftime("%Y%m%d")
            if day_str not in ck.get("days", {}) or ck["days"][day_str].get("status") != "OK":
                pending += 1
        except: continue
    print(f"Days completed: {completed}")
    print(f"Days pending: {pending}")
    return completed, pending

def mode_extract(year, month, max_days):
    _ensure_dirs()
    dl = DukascopyResumable("EURUSD")
    ck = load_checkpoint(year, month)
    if "days" not in ck: ck["days"] = {}
    
    downloaded = 0
    errors = []
    month_dir = os.path.join(PARTIAL_PATH, f"{year}_{month:02d}")
    os.makedirs(month_dir, exist_ok=True)
    
    for d in range(1, max_days + 1):
        try:
            dt = date(year, month, d)
            if dt.month != month: continue
        except: continue
        day_str = dt.strftime("%Y%m%d")
        
        if ck.get("days", {}).get(day_str, {}).get("status") == "OK":
            print(f"[SKIP] {dt} ya OK")
            continue
        
        print(f"[DOWNLOAD] {dt}...")
        df = dl.download_day(dt)
        
        if df.empty:
            errors.append(f"{day_str}: empty")
            ck["days"][day_str] = {"status": "EMPTY", "rows": 0}
            continue
        
        out_file = os.path.join(month_dir, f"{day_str}.parquet")
        df.to_parquet(out_file, compression="snappy")
        ck["days"][day_str] = {"status": "OK", "rows": len(df)}
        save_checkpoint(year, month, ck)
        downloaded += 1
        print(f"[OK] {dt}: {len(df)} rows")
    
    save_checkpoint(year, month, ck)
    return downloaded, errors

def mode_resume(year, month):
    _ensure_dirs()
    dl = DukascopyResumable("EURUSD")
    ck = load_checkpoint(year, month)
    if "days" not in ck: ck["days"] = {}
    
    downloaded = 0
    errors = []
    month_dir = os.path.join(PARTIAL_PATH, f"{year}_{month:02d}")
    os.makedirs(month_dir, exist_ok=True)
    
    total_days = 31
    for d in range(1, total_days + 1):
        try:
            dt = date(year, month, d)
            if dt.month != month: continue
        except: continue
        day_str = dt.strftime("%Y%m%d")
        
        if ck.get("days", {}).get(day_str, {}).get("status") == "OK":
            continue
        
        print(f"[RESUME] {dt}...")
        df = dl.download_day(dt)
        
        if df.empty:
            errors.append(f"{day_str}: empty")
            ck["days"][day_str] = {"status": "EMPTY", "rows": 0}
            save_checkpoint(year, month, ck)
            continue
        
        out_file = os.path.join(month_dir, f"{day_str}.parquet")
        df.to_parquet(out_file, compression="snappy")
        ck["days"][day_str] = {"status": "OK", "rows": len(df)}
        save_checkpoint(year, month, ck)
        downloaded += 1
        print(f"[OK] {dt}: {len(df)} rows")
    
    save_checkpoint(year, month, ck)
    return downloaded, errors

def mode_finalize(year, month):
    month_dir = os.path.join(PARTIAL_PATH, f"{year}_{month:02d}")
    if not os.path.exists(month_dir):
        print("[ERROR] No partial directory")
        return None
    
    all_dfs = []
    for f in os.listdir(month_dir):
        if f.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(month_dir, f))
            all_dfs.append(df)
    
    if not all_dfs:
        print("[ERROR] No day files")
        return None
    
    df = pd.concat(all_dfs, ignore_index=True)
    df["timestamp_utc"] = _coerce_timestamp_utc(df["timestamp_utc"])
    df.sort_values("timestamp_utc", inplace=True)
    df["timestamp_ny"] = df["timestamp_utc"].dt.tz_convert(NY)
    df["spread"] = df["ask"] - df["bid"]
    df["spread_pips"] = df["spread"] / 0.0001
    df["source"] = "dukascopy_resumable"
    df["symbol"] = "EURUSD"
    
    out_file = os.path.join(TICK_PATH, "monthly", f"EURUSD_ticks_{year}_{month:02d}.parquet")
    df.to_parquet(out_file, compression="snappy")
    print(f"[SAVED] {out_file}: {len(df)} rows")
    
    return out_file

def mode_validate(year, month):
    file_name = os.path.join(TICK_PATH, "monthly", f"EURUSD_ticks_{year}_{month:02d}.parquet")
    if not os.path.exists(file_name):
        print("[ERROR] File not found")
        return None
    
    df = pd.read_parquet(file_name)
    df["timestamp_utc"] = _coerce_timestamp_utc(df["timestamp_utc"])
    
    rows = len(df)
    neg_spread = (df["spread"] < 0).sum()
    invalid = (df["bid"] > df["ask"]).sum()
    first = df["timestamp_utc"].min()
    last = df["timestamp_utc"].max()
    sha = _get_sha256(file_name)
    size = os.path.getsize(file_name) / (1024 * 1024)
    
    print(f"Rows: {rows}")
    print(f"Size: {size:.2f} MB")
    print(f"SHA256: {sha[:16]}...")
    print(f"First: {first}")
    print(f"Last: {last}")
    print(f"Neg spread: {neg_spread}")
    print(f"Invalid bid>ask: {invalid}")
    
    return {
        "rows": rows, "size_mb": size, "sha256": sha,
        "first": str(first), "last": str(last),
        "neg_spread": neg_spread, "invalid": invalid
    }

if __name__ == "__main__":
    import urllib.request
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--mode", choices=["diagnostic", "extract", "resume", "finalize", "validate"], required=True)
    parser.add_argument("--chunk", default="day")
    parser.add_argument("--max-days", type=int, default=3)
    args = parser.parse_args()
    
    if args.mode == "diagnostic":
        mode_diagnostic(args.year, args.month)
    elif args.mode == "extract":
        mode_extract(args.year, args.month, args.max_days)
    elif args.mode == "resume":
        mode_resume(args.year, args.month)
    elif args.mode == "finalize":
        mode_finalize(args.year, args.month)
    elif args.mode == "validate":
        mode_validate(args.year, args.month)