"""
Operational Stack Historical Replay
Simulates the institutional machinery (News, DHL, Tribunal) on 2020-2025 data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
TRADES_LEDGER = ROOT / "results" / "SCBI_2020_2025_DURABILITY" / "trades_baseline.csv"
OUTPUT_REPORT = ROOT / "results" / "SCBI_OPERATIONAL_REPLAY_REPORT.json"

# Institutional Rules
DHL_LIMIT = -3.0
NEWS_PROTECTION = True # Assumed already filtered in baseline or we filter by hour if news data available
# For this replay, we focus on DHL and Tribunal logic over the durability ledger.

def simulate_institutional_layer(df):
    df = df.sort_values("entry_time").copy()
    df["date"] = df["session_date"]
    
    processed_trades = []
    daily_pnl = {}
    blocked_days = set()
    
    for i, row in df.iterrows():
        date = row["date"]
        pnl = row["pnl_r"]
        
        # 1. Check if day is already blocked by DHL
        if date in blocked_days:
            continue
            
        # 2. Daily PnL tracking
        current_daily_pnl = daily_pnl.get(date, 0.0)
        
        # 3. Simulate Prop Firm Guard (DHL)
        if current_daily_pnl <= DHL_LIMIT:
            blocked_days.add(date)
            continue
            
        # 4. Accept Trade
        processed_trades.append(row)
        daily_pnl[date] = current_daily_pnl + pnl
        
    return pd.DataFrame(processed_trades)

def run_replay():
    if not TRADES_LEDGER.exists():
        return {"status": "ERROR", "msg": "Ledger not found"}
        
    df_raw = pd.read_csv(TRADES_LEDGER)
    df_inst = simulate_institutional_layer(df_raw)
    
    # Metrics Comparison
    def get_metrics(df):
        if df.empty: return {}
        pnls = df["pnl_r"]
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        pf = wins.sum() / abs(losses.sum()) if not losses.empty else 999
        dd = (pnls.cumsum() - pnls.cumsum().cummax()).min()
        return {
            "n": len(df),
            "pf": round(pf, 3),
            "max_dd": round(dd, 3),
            "total_r": round(pnls.sum(), 3)
        }
        
    report = {
        "raw": get_metrics(df_raw),
        "institutional": get_metrics(df_inst),
        "impact": {
            "trades_removed": len(df_raw) - len(df_inst),
            "pnl_diff": round(df_inst["pnl_r"].sum() - df_raw["pnl_r"].sum(), 3)
        }
    }
    
    # Tribunal Emulation
    n_inst = len(df_inst)
    if n_inst >= 60 and report["institutional"]["pf"] > 2.0:
        report["tribunal_verdict"] = "DEMO_ELIGIBLE_HISTORICAL"
    else:
        report["tribunal_verdict"] = "FOLLOW_ON_REQUIRED"
        
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    res = run_replay()
    print(json.dumps(res, indent=2))
