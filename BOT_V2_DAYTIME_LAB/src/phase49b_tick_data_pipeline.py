import os
import sys
import pandas as pd
import numpy as np
import pytz
import json
import hashlib
import lzma
import struct
from datetime import datetime, timedelta, date, time as pytime
from pathlib import Path
import urllib.request
from urllib.error import HTTPError, URLError

# --- CONFIGURACIÓN DE RUTAS ---
ROOT_PROJECT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
MARKET_DATA_ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA"
TICK_PATH = os.path.join(MARKET_DATA_ROOT, "tick", "EURUSD")

SUBFOLDERS = ["raw", "processed", "monthly", "manifests", "quality_reports", "logs"]

# --- TIMEZONES ---
UTC = pytz.UTC
NY = pytz.timezone("America/New_York")

# --- DUKASCOPY BINARY FORMAT (TICKS) ---
TICK_STRUCT = struct.Struct(">IIIff")

def _coerce_timestamp_utc(series):
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert("UTC")

def _timestamp_ny_from_utc(series):
    return _coerce_timestamp_utc(series).dt.tz_convert("America/New_York")

def _ensure_directories():
    for sub in SUBFOLDERS:
        os.makedirs(os.path.join(TICK_PATH, sub), exist_ok=True)

def _get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class DukascopyTickDownloader:
    def __init__(self, symbol="EURUSD"):
        self.symbol = symbol
        self.divisor = 100000.0 if "JPY" not in symbol else 1000.0
        self.user_agent = "Mozilla/5.0"

    def _download_hour_type(self, day: date, hour: int):
        url = f"https://datafeed.dukascopy.com/datafeed/{self.symbol}/{day.year}/{day.month-1:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                data = response.read()
                return lzma.decompress(data) if data else None
        except: return None

    def get_day_ticks(self, day: date):
        print(f"[*] Descargando: {day}")
        all_day_ticks = []
        for hour in range(24):
            raw_data = self._download_hour_type(day, hour)
            if not raw_data: continue
            hour_start = datetime.combine(day, pytime(hour, 0), tzinfo=UTC)
            for ms, ask_i, bid_i, ask_vol, bid_vol in TICK_STRUCT.iter_unpack(raw_data):
                all_day_ticks.append({
                    "timestamp_utc": hour_start + timedelta(milliseconds=ms),
                    "bid": bid_i / self.divisor,
                    "ask": ask_i / self.divisor,
                    "bid_volume": bid_vol,
                    "ask_volume": ask_vol
                })
        return pd.DataFrame(all_day_ticks)

def extract_month(symbol, year, month, limit_days=None):
    downloader = DukascopyTickDownloader(symbol)
    all_ticks = []
    
    start_date = date(year, month, 1)
    if month == 12:
        end_date = date(year + 1, 1, 1)
    else:
        end_date = date(year, month + 1, 1)
        
    if limit_days:
        end_date = start_date + timedelta(days=limit_days)
        
    curr = start_date
    while curr < end_date:
        df_day = downloader.get_day_ticks(curr)
        if not df_day.empty:
            all_ticks.append(df_day)
        curr += timedelta(days=1)
        
    if not all_ticks:
        return False, "no_data"
        
    df = pd.concat(all_ticks, ignore_index=True)
    df['timestamp_utc'] = _coerce_timestamp_utc(df['timestamp_utc'])
    df.sort_values("timestamp_utc", inplace=True)
    
    # Enriquecer
    df['timestamp_ny'] = _timestamp_ny_from_utc(df['timestamp_utc'])
    df['spread'] = df['ask'] - df['bid']
    df['spread_pips'] = df['spread'] / 0.0001
    df['source'] = "dukascopy_native_h"
    df['symbol'] = symbol
    
    # Guardar
    file_name = f"{symbol}_ticks_{year}_{month:02d}.parquet"
    file_path = os.path.join(TICK_PATH, "monthly", file_name)
    
    print(f"[*] Guardando {len(df)} ticks en {file_path}...")
    df.to_parquet(file_path, compression='snappy')
    
    return True, file_path

def compute_detailed_quality(file_path):
    if not os.path.exists(file_path): return None
    df = pd.read_parquet(file_path)
    df['timestamp_utc'] = _coerce_timestamp_utc(df['timestamp_utc'])
    df['timestamp_ny'] = _timestamp_ny_from_utc(df['timestamp_utc'])
    
    # Metricas exactas
    rows = len(df)
    neg_spread = (df['spread'] < 0).sum()
    invalid_bid_ask = (df['bid'] > df['ask']).sum()
    dups = df['timestamp_utc'].duplicated().sum()
    
    # Gaps (más de 1 hora sin ticks en dias de semana)
    diffs = df['timestamp_utc'].diff()
    gaps = (diffs > pd.Timedelta(hours=1)).sum()
    
    # Outliers (spread > 50 pips)
    outliers = (df['spread_pips'] > 50).sum()
    
    # Timestamps
    first_utc = df['timestamp_utc'].min()
    last_utc = df['timestamp_utc'].max()
    first_ny = df['timestamp_ny'].min()
    last_ny = df['timestamp_ny'].max()
    
    trading_days = df['timestamp_utc'].dt.date.nunique()
    
    # Stats de spread
    avg_spread = df['spread_pips'].mean()
    med_spread = df['spread_pips'].median()
    max_spread = df['spread_pips'].max()
    
    sha = _get_sha256(file_path)
    file_size = os.path.getsize(file_path) / (1024*1024)
    
    stats = {
        "symbol": str(df['symbol'].iloc[0]),
        "rows": int(rows),
        "first_timestamp_utc": first_utc.isoformat(),
        "last_timestamp_utc": last_utc.isoformat(),
        "first_timestamp_ny": first_ny.isoformat(),
        "last_timestamp_ny": last_ny.isoformat(),
        "min_bid": float(df['bid'].min()),
        "max_bid": float(df['bid'].max()),
        "min_ask": float(df['ask'].min()),
        "max_ask": float(df['ask'].max()),
        "avg_spread_pips": round(float(avg_spread), 6),
        "median_spread_pips": round(float(med_spread), 6),
        "max_spread_pips": round(float(max_spread), 6),
        "duplicate_timestamps": int(dups),
        "negative_spread_count": int(neg_spread),
        "invalid_bid_ask_count": int(invalid_bid_ask),
        "gaps_count": int(gaps),
        "outlier_count": int(outliers),
        "trading_days": int(trading_days),
        "file_size_mb": round(file_size, 4),
        "sha256": sha,
        "source": "dukascopy_native_h"
    }
    return stats

def update_manifest(stats, year, month):
    manifest_path = os.path.join(TICK_PATH, "manifests", "EURUSD_TICK_DATA_MANIFEST.csv")
    
    new_entry = {
        "symbol": stats["symbol"],
        "year": year,
        "month": month,
        "file_path": f"monthly/EURUSD_ticks_{year}_{month:02d}.parquet",
        "format": "parquet",
        "rows": stats["rows"],
        "first_timestamp_utc": stats["first_timestamp_utc"],
        "last_timestamp_utc": stats["last_timestamp_utc"],
        "first_timestamp_ny": stats["first_timestamp_ny"],
        "last_timestamp_ny": stats["last_timestamp_ny"],
        "min_bid": stats["min_bid"],
        "max_bid": stats["max_bid"],
        "min_ask": stats["min_ask"],
        "max_ask": stats["max_ask"],
        "avg_spread_pips": stats["avg_spread_pips"],
        "median_spread_pips": stats["median_spread_pips"],
        "max_spread_pips": stats["max_spread_pips"],
        "duplicate_timestamps": stats["duplicate_timestamps"],
        "negative_spread_count": stats["negative_spread_count"],
        "invalid_bid_ask_count": stats["invalid_bid_ask_count"],
        "gaps_count": stats["gaps_count"],
        "outlier_count": stats["outlier_count"],
        "trading_days": stats["trading_days"],
        "missing_days": 0,
        "file_size_mb": stats["file_size_mb"],
        "sha256": stats["sha256"],
        "source": stats["source"],
        "extraction_status": "COMPLETED"
    }
    
    df_new = pd.DataFrame([new_entry])
    if os.path.exists(manifest_path):
        try:
            df_old = pd.read_csv(manifest_path)
            df_old = df_old[~((df_old['year'] == year) & (df_old['month'] == month))]
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        except:
            df_final = df_new
    else:
        df_final = df_new
        
    df_final.to_csv(manifest_path, index=False)
    print(f"[*] Manifest actualizado: {manifest_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--pilot-days", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    _ensure_directories()

    if args.validate_only:
        file_name = f"EURUSD_ticks_{args.year}_{args.month:02d}.parquet"
        path = os.path.join(TICK_PATH, "monthly", file_name)
        if os.path.exists(path):
            print(f"[*] Validando archivo existente: {path}")
            stats = compute_detailed_quality(path)
            quality_path = os.path.join(TICK_PATH, "quality_reports", f"EURUSD_tick_quality_{args.year}_{args.month:02d}.json")
            with open(quality_path, 'w') as f:
                json.dump(stats, f, indent=2)
            update_manifest(stats, args.year, args.month)
            print("[SUCCESS] Validacion y manifiesto completados.")
        else:
            print(f"[ERROR] No existe el archivo {path}")
        return

    if args.pilot:
        print(f"[*] Iniciando piloto: {args.year}-{args.month:02d}")
        success, path = extract_month("EURUSD", args.year, args.month, limit_days=args.pilot_days)
        if success:
            stats = compute_detailed_quality(path)
            quality_path = os.path.join(TICK_PATH, "quality_reports", f"EURUSD_tick_quality_{args.year}_{args.month:02d}.json")
            with open(quality_path, 'w') as f:
                json.dump(stats, f, indent=2)
            update_manifest(stats, args.year, args.month)
        else:
            print(f"[ERROR] Fallo en la extraccion.")

if __name__ == "__main__":
    main()
