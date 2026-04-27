"""
Dual Line Operational Stack Historical Replay
Simulates the dual institutional machinery (GLOBAL & CORE) on historical data.
"""
import pandas as pd
from pathlib import Path
import json

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
GLOBAL_LEDGER = ROOT / "results" / "SCBI_2020_2025_DURABILITY" / "trades_baseline.csv"
CORE_LEDGER = ROOT / "results" / "SCBI_CORE_STAGE2" / "core_stage2_trades.csv"
OUTPUT_REPORT = ROOT / "results" / "SCBI_DUAL_OPERATIONAL_REPLAY_REPORT.json"

DHL_LIMIT = -3.0

def process_line(df, name):
    if df.empty: return pd.DataFrame()
    # Normalize columns if needed
    time_col = "entry_time" if "entry_time" in df.columns else "timestamp_ny"
    date_col = "session_date" if "session_date" in df.columns else "date"
    
    df = df.sort_values(time_col).copy()
    df["line"] = name
    
    processed_trades = []
    daily_pnl = {}
    blocked_days = set()
    
    for _, row in df.iterrows():
        date = row[date_col]
        pnl = row["pnl_r"]
        
        if date in blocked_days: continue
        
        current_daily_pnl = daily_pnl.get(date, 0.0)
        if current_daily_pnl <= DHL_LIMIT:
            blocked_days.add(date)
            continue
            
        processed_trades.append(row)
        daily_pnl[date] = current_daily_pnl + pnl
        
    return pd.DataFrame(processed_trades)

def get_metrics(df):
    if df.empty: return {"n": 0, "pf": 0.0, "max_dd": 0.0, "total_r": 0.0}
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

def run_dual_replay():
    if not GLOBAL_LEDGER.exists() or not CORE_LEDGER.exists():
        return {"status": "ERROR", "msg": "One or more ledgers not found"}
        
    df_global_raw = pd.read_csv(GLOBAL_LEDGER)
    df_core_raw = pd.read_csv(CORE_LEDGER)
    
    df_global_inst = process_line(df_global_raw, "GLOBAL")
    df_core_inst = process_line(df_core_raw, "CORE")
    
    report = {
        "GLOBAL": {
            "raw": get_metrics(df_global_raw),
            "institutional": get_metrics(df_global_inst)
        },
        "CORE": {
            "raw": get_metrics(df_core_raw),
            "institutional": get_metrics(df_core_inst)
        },
        "COMPARISON": {
            "pf_delta": round(get_metrics(df_core_inst)["pf"] - get_metrics(df_global_inst)["pf"], 3),
            "n_ratio": round(len(df_core_inst) / len(df_global_inst), 3) if not df_global_inst.empty else 0
        }
    }
    
    # Tribunal Emulation
    for line in ["GLOBAL", "CORE"]:
        m = report[line]["institutional"]
        if m["n"] >= 40 and m["pf"] > 2.0:
            report[line]["tribunal_verdict"] = "DEMO_ELIGIBLE_HISTORICAL"
        elif m["n"] >= 10:
            report[line]["tribunal_verdict"] = "FOLLOW_ON_REQUIRED"
        else:
            report[line]["tribunal_verdict"] = "PAPER_ONLY_INSUFFICIENT_SAMPLE"

    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    res = run_dual_replay()
    print(json.dumps(res, indent=2))
