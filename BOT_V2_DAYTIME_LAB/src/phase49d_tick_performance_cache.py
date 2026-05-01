import os
import sys
import pandas as pd
import numpy as np
import pytz
import json
import time
from datetime import datetime, timedelta, date
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
MARKET_DATA_ROOT = r"C:\Users\alera\Desktop\Bot\BOT_MARKET_DATA"
TICK_PATH = os.path.join(MARKET_DATA_ROOT, "tick", "EURUSD")
CACHE_PATH = os.path.join(TICK_PATH, "cache")
MANIFESTS_PATH = os.path.join(TICK_PATH, "manifests")

# --- TIMEZONES ---
UTC = pytz.UTC
NY = pytz.timezone("America/New_York")

def _get_sha256(file_path):
    import hashlib
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class TickPerformanceEngine:
    def __init__(self, symbol="EURUSD"):
        self.symbol = symbol
        self.tick_file_pattern = "{symbol}_ticks_{year}_{month:02d}.parquet"
        self.tf_map = {"M1": "1min", "M5": "5min", "M15": "15min"}

    def load_tick_data(self, year, month, columns=None, ny_window=None):
        file_name = self.tick_file_pattern.format(symbol=self.symbol, year=year, month=month)
        file_path = os.path.join(TICK_PATH, "monthly", file_name)
        if not os.path.exists(file_path): return None, "File not found"
        start_t = time.time()
        df = pd.read_parquet(file_path, columns=columns)
        if ny_window:
            start_h, end_h = ny_window
            df = df[(df['timestamp_ny'].dt.hour >= start_h) & (df['timestamp_ny'].dt.hour < end_h)]
        end_t = time.time()
        return df, end_t - start_t

    def build_ohlc_cache(self, df, timeframe='M1'):
        if df is None or df.empty: return None
        pd_tf = self.tf_map.get(timeframe, timeframe)
        df = df.set_index('timestamp_utc')
        resampler = df.resample(pd_tf)
        ohlc = pd.DataFrame()
        ohlc['bid_open'] = resampler['bid'].first()
        ohlc['bid_high'] = resampler['bid'].max()
        ohlc['bid_low'] = resampler['bid'].min()
        ohlc['bid_close'] = resampler['bid'].last()
        ohlc['ask_open'] = resampler['ask'].first()
        ohlc['ask_high'] = resampler['ask'].max()
        ohlc['ask_low'] = resampler['ask'].min()
        ohlc['ask_close'] = resampler['ask'].last()
        ohlc['spread_mean'] = resampler['spread_pips'].mean()
        ohlc['tick_count'] = resampler['bid'].count()
        ohlc = ohlc.dropna(subset=['bid_open'])
        ohlc['timestamp_ny'] = ohlc.index.tz_convert(NY)
        return ohlc

    def save_cache(self, ohlc, year, month, timeframe):
        folder = os.path.join(CACHE_PATH, timeframe)
        os.makedirs(folder, exist_ok=True)
        file_name = f"{self.symbol}_{timeframe}_from_ticks_{year}_{month:02d}.parquet"
        file_path = os.path.join(folder, file_name)
        ohlc.to_parquet(file_path, compression='snappy')
        return file_path

    def update_cache_manifest(self, year, month, timeframe, file_path, source_sha256):
        manifest_path = os.path.join(MANIFESTS_PATH, "EURUSD_TICK_CACHE_MANIFEST.csv")
        new_entry = {
            "symbol": self.symbol, "year": year, "month": month, "timeframe": timeframe,
            "cache_file": f"cache/{timeframe}/{os.path.basename(file_path)}",
            "source_sha256": source_sha256,
            "rows": len(pd.read_parquet(file_path)),
            "file_size_mb": round(os.path.getsize(file_path) / (1024*1024), 4),
            "cache_sha256": _get_sha256(file_path),
            "created_at": datetime.now().isoformat()
        }
        df_new = pd.DataFrame([new_entry])
        if os.path.exists(manifest_path):
            df_old = pd.read_csv(manifest_path)
            df_old = df_old[~((df_old['year'] == year) & (df_old['month'] == month) & (df_old['timeframe'] == timeframe))]
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_final = df_new
        df_final.to_csv(manifest_path, index=False)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--timeframes", type=str, default="M1,M5,M15")
    parser.add_argument("--validate-cache", action="store_true")
    args = parser.parse_args()

    engine = TickPerformanceEngine("EURUSD")
    if args.dry_run: print("[DRY-RUN] OK"); return

    if args.build_cache:
        tfs = args.timeframes.split(",")
        file_name = f"EURUSD_ticks_{args.year}_{args.month:02d}.parquet"
        tick_path = os.path.join(TICK_PATH, "monthly", file_name)
        src_sha = _get_sha256(tick_path)
        df, _ = engine.load_tick_data(args.year, args.month)
        for tf in tfs:
            ohlc = engine.build_ohlc_cache(df, tf)
            path = engine.save_cache(ohlc, args.year, args.month, tf)
            engine.update_cache_manifest(args.year, args.month, tf, path, src_sha)
            print(f"[SUCCESS] Cache {tf} guardado.")

    if args.validate_cache:
        manifest_path = os.path.join(MANIFESTS_PATH, "EURUSD_TICK_CACHE_MANIFEST.csv")
        if os.path.exists(manifest_path):
            print(pd.read_csv(manifest_path).to_string())
        else: print("[!] No manifest.")

if __name__ == "__main__":
    main()
