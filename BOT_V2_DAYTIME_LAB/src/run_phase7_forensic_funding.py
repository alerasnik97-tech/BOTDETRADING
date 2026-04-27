
import pandas as pd
import numpy as np
import json
from pathlib import Path

def run_funding_audit():
    print("Phase 8: Funding Risk Audit - STRONG_CANDIDATE_PHASE7_V1")
    
    trades_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase7_forensic_audit\reproduction\reproduced_trades.csv"
    trades = pd.read_csv(trades_path)
    trades['date'] = pd.to_datetime(trades['date'])
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase7_forensic_audit\funding")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Daily Risk Distribution
    daily_r = trades.groupby('date')['r_value'].sum()
    daily_stats = {
        "max_daily_loss_r": round(daily_r.min(), 2),
        "max_daily_win_r": round(daily_r.max(), 2),
        "avg_daily_r": round(daily_r.mean(), 3),
        "worst_day_date": str(daily_r.idxmin().date()) if not daily_r.empty else "N/A"
    }
    
    # Funding Scenarios (0.5% Risk)
    risk_per_trade = 0.005 # 0.5%
    trades['pnl_pct'] = trades['r_value'] * risk_per_trade
    trades['cum_pnl'] = (1 + trades['pnl_pct']).cumprod()
    
    daily_pnl = trades.groupby('date')['pnl_pct'].sum()
    funding_stats = {
        "max_daily_drawdown_pct": round(float(daily_pnl.min() * 100), 2),
        "total_return_pct": round(float((trades['cum_pnl'].iloc[-1] - 1) * 100), 2) if not trades.empty else 0,
        "is_plausible_for_funding": bool(daily_pnl.min() > -0.05) # Max 5% daily loss
    }
    
    with open(out_dir / "funding_risk_audit.json", 'w') as f:
        json.dump({**daily_stats, **funding_stats}, f, indent=2)
    
    daily_r.to_csv(out_dir / "daily_risk_distribution.csv")
    print("Funding Risk Audit Complete.")

if __name__ == "__main__":
    run_funding_audit()


