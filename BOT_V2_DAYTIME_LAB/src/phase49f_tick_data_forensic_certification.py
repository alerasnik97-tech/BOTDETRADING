import os
import pandas as pd
import numpy as np
import json
import hashlib
from datetime import datetime, timedelta
import pytz

# --- CONFIGURACIÓN ---
PROJECT_ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
MARKET_DATA_ROOT = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_MARKET_DATA"
TICK_PATH = os.path.join(MARKET_DATA_ROOT, "tick", "EURUSD")
REPORTS_PATH = os.path.join(TICK_PATH, "quality_reports")
REPRO_PATH = os.path.join(TICK_PATH, "repro_check")

os.makedirs(REPORTS_PATH, exist_ok=True)
os.makedirs(REPRO_PATH, exist_ok=True)

CANONICAL_PARQUET = os.path.join(TICK_PATH, "monthly", "EURUSD_ticks_2025_01.parquet")

class TickForensicCertifier:
    def __init__(self, parquet_path):
        self.path = parquet_path
        self.df = pd.read_parquet(parquet_path)
        self.df['timestamp_ny'] = pd.to_datetime(self.df['timestamp_ny'])
        self.df['timestamp_utc'] = pd.to_datetime(self.df['timestamp_utc'])

    def internal_audit(self):
        """Auditoría de integridad técnica."""
        audit = {
            "rows": len(self.df),
            "duplicates": int(self.df.duplicated().sum()),
            "negative_spreads": len(self.df[self.df.spread < 0]),
            "null_values": int(self.df.isnull().sum().sum()),
            "temporal_order_ok": bool(self.df.timestamp_utc.is_monotonic_increasing),
            "avg_spread": round(float(self.df.spread_pips.mean()), 4),
            "max_spread": round(float(self.df.spread_pips.max()), 2),
            "p95_spread": round(float(self.df.spread_pips.quantile(0.95)), 2)
        }
        
        # Gaps de tiempo (> 60s en horario operativo)
        # Operativo: Lun-Vie 07:00-17:00 NY
        op_mask = (self.df.timestamp_ny.dt.hour >= 7) & (self.df.timestamp_ny.dt.hour < 17) & (self.df.timestamp_ny.dt.dayofweek < 5)
        df_op = self.df[op_mask].copy()
        df_op['diff'] = df_op.timestamp_utc.diff().dt.total_seconds()
        gaps = df_op[df_op['diff'] > 60]
        audit["gaps_count_op"] = len(gaps)
        audit["max_gap_op_sec"] = float(df_op['diff'].max()) if not df_op.empty else 0
        
        return audit

    def timezone_audit(self):
        """Auditoría de alineación horaria NY/UTC."""
        # Enero 2025: NY es UTC-5
        sample_ticks = self.df.sample(100)
        offsets = (sample_ticks.timestamp_utc - sample_ticks.timestamp_ny).dt.total_seconds() / 3600
        
        audit = {
            "expected_offset": 5.0,
            "min_offset_found": float(offsets.min()),
            "max_offset_found": float(offsets.max()),
            "consistent": bool(offsets.min() == 5.0 and offsets.max() == 5.0)
        }
        return audit

    def price_sanity_audit(self):
        """Auditoría de saltos de precio irracionales."""
        self.df['price_diff'] = self.df.bid.diff().abs() * 10000 # pips
        jumps = self.df[self.df.price_diff > 10] # Saltos > 10 pips entre ticks
        
        audit = {
            "jumps_gt_10pips": len(jumps),
            "max_jump_pips": round(float(self.df.price_diff.max()), 2)
        }
        return audit

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=str, default="2025-01")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    certifier = TickForensicCertifier(CANONICAL_PARQUET)
    
    results = {
        "internal": certifier.internal_audit(),
        "timezone": certifier.timezone_audit(),
        "price_sanity": certifier.price_sanity_audit()
    }
    
    print(json.dumps(results, indent=4))
    
    # Guardar reportes
    with open(os.path.join(REPORTS_PATH, f"PHASE49F_FORENSIC_CERTIFICATION_{args.month}.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
