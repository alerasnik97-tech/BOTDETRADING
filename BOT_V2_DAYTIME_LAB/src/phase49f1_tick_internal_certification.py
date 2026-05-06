import os
import pandas as pd
import numpy as np
import json
import hashlib
from datetime import datetime, timedelta

# --- CONFIGURACIÓN ---
PROJECT_ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
MARKET_DATA_ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA"
TICK_PATH = os.path.join(MARKET_DATA_ROOT, "tick", "EURUSD")
REPORTS_PATH = os.path.join(TICK_PATH, "quality_reports")
REPO_REPORTS_PATH = os.path.join(PROJECT_ROOT, "BOT_V2_DAYTIME_LAB", "reports")
NY_TZ = "America/New_York"

os.makedirs(REPORTS_PATH, exist_ok=True)
os.makedirs(REPO_REPORTS_PATH, exist_ok=True)

CANONICAL_PARQUET = os.path.join(TICK_PATH, "monthly", "EURUSD_ticks_2025_01.parquet")

def coerce_timestamp_utc(series):
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert("UTC")

def drop_timezone(series):
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        return ts
    return ts.dt.tz_localize(None)

def timezone_name(series):
    ts = pd.to_datetime(series)
    return str(ts.dt.tz) if ts.dt.tz is not None else "naive"

def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

class TickInternalCertifier:
    def __init__(self, parquet_path):
        self.path = parquet_path
        self.df = pd.read_parquet(parquet_path)
        self.timestamp_utc_timezone = timezone_name(self.df['timestamp_utc'])
        self.timestamp_ny_timezone = timezone_name(self.df['timestamp_ny'])
        self.df['timestamp_utc'] = coerce_timestamp_utc(self.df['timestamp_utc'])
        self.df['timestamp_ny'] = pd.to_datetime(self.df['timestamp_ny'])
        self.expected_timestamp_ny = self.df['timestamp_utc'].dt.tz_convert(NY_TZ)

    def run_audit(self):
        # 1. File Info
        file_size = os.path.getsize(self.path) / (1024 * 1024)
        sha = get_sha256(self.path)
        
        # 2. Consistency
        null_ts = int(self.df['timestamp_utc'].isna().sum())
        null_bid = int(self.df['bid'].isna().sum())
        null_ask = int(self.df['ask'].isna().sum())
        bid_gt_ask = int((self.df['bid'] > self.df['ask']).sum())
        neg_spread = int((self.df['spread'] < 0).sum())
        duplicates = int(self.df['timestamp_utc'].duplicated().sum())
        unsorted = int((self.df['timestamp_utc'].diff().dt.total_seconds() < 0).sum())
        
        # 3. Timezone
        # January 2025 NY must be UTC-5. Compare local clock values, not only instants.
        utc_nominal = self.df['timestamp_utc'].dt.tz_localize(None)
        ny_nominal = drop_timezone(self.df['timestamp_ny'])
        expected_ny_nominal = self.expected_timestamp_ny.dt.tz_localize(None)
        nominal_offsets = (utc_nominal - ny_nominal).dt.total_seconds() / 3600
        expected_offsets = (utc_nominal - expected_ny_nominal).dt.total_seconds() / 3600
        timezone_name_ok = self.timestamp_ny_timezone == NY_TZ
        nominal_clock_ok = bool(ny_nominal.equals(expected_ny_nominal))
        offset_ok = bool((expected_offsets == 5.0).all() and (nominal_offsets == 5.0).all())
        tz_consistent = bool(timezone_name_ok and nominal_clock_ok and offset_ok)
        
        # 4. Spread
        spread_stats = {
            "mean": round(float(self.df['spread_pips'].mean()), 4),
            "median": round(float(self.df['spread_pips'].median()), 2),
            "p95": round(float(self.df['spread_pips'].quantile(0.95)), 2),
            "p99": round(float(self.df['spread_pips'].quantile(0.99)), 2),
            "max": round(float(self.df['spread_pips'].max()), 2),
            "spikes_gt_2": int((self.df['spread_pips'] > 2).sum()),
            "spikes_gt_5": int((self.df['spread_pips'] > 5).sum()),
            "spikes_gt_10": int((self.df['spread_pips'] > 10).sum())
        }
        
        # 5. Price Sanity
        price_diffs = self.df['bid'].diff().abs() * 10000
        jumps_gt_10 = int((price_diffs > 10).sum())
        
        results = {
            "file": {
                "path": self.path,
                "rows": len(self.df),
                "size_mb": round(file_size, 2),
                "sha256": sha,
                "columns": list(self.df.columns),
                "first_utc": str(self.df['timestamp_utc'].min()),
                "first_ny": str(self.df['timestamp_ny'].min()),
                "last_utc": str(self.df['timestamp_utc'].max()),
                "last_ny": str(self.df['timestamp_ny'].max())
            },
            "consistency": {
                "null_ts": null_ts,
                "null_bid": null_bid,
                "null_ask": null_ask,
                "bid_gt_ask": bid_gt_ask,
                "neg_spread": neg_spread,
                "duplicates": duplicates,
                "unsorted": unsorted
            },
            "timezone": {
                "expected_utc_minus_ny_hours": 5.0,
                "timestamp_utc_timezone": self.timestamp_utc_timezone,
                "timestamp_ny_timezone": self.timestamp_ny_timezone,
                "timezone_name_ok": timezone_name_ok,
                "nominal_clock_ok": nominal_clock_ok,
                "offset_ok": offset_ok,
                "nominal_utc_minus_ny_hours_min": float(nominal_offsets.min()),
                "nominal_utc_minus_ny_hours_max": float(nominal_offsets.max()),
                "consistent": tz_consistent
            },
            "spread": spread_stats,
            "sanity": {
                "min_bid": float(self.df['bid'].min()),
                "max_bid": float(self.df['bid'].max()),
                "jumps_gt_10pips": jumps_gt_10,
                "max_jump_pips": round(float(price_diffs.max()), 2)
            }
        }
        return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=str, default="2025-01")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY-RUN: Phase49F1 Script Ready.")
        return

    certifier = TickInternalCertifier(CANONICAL_PARQUET)
    summary = certifier.run_audit()
    
    # Guardar reportes
    with open(os.path.join(REPORTS_PATH, f"PHASE49F1_INTERNAL_CERTIFICATION_{args.month}.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    with open(os.path.join(REPO_REPORTS_PATH, "PHASE49F1_TICK_INTERNAL_CERTIFICATION_REPORT.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Generar MD compacto
    md_content = f"""# PHASE 49F1 — TICK INTERNAL CERTIFICATION REPORT
- **Status**: {"CERTIFIED" if summary['consistency']['null_ts'] == 0 and summary['timezone']['consistent'] else "WARNING"}
- **Rows**: {summary['file']['rows']}
- **SHA256**: {summary['file']['sha256']}
- **Spread Mean**: {summary['spread']['mean']} pips
- **Timezone Consistent**: {summary['timezone']['consistent']} (America/New_York, UTC-5 nominal clock)
"""
    with open(os.path.join(REPO_REPORTS_PATH, "PHASE49F1_TICK_INTERNAL_CERTIFICATION_REPORT.md"), "w") as f:
        f.write(md_content)

    print("PHASE49F1_COMPLETED_SUCCESSFULLY")

if __name__ == "__main__":
    main()
