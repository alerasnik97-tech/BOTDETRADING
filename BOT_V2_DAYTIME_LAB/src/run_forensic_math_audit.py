
import pandas as pd
import numpy as np
import json
from pathlib import Path

def run_math_audit():
    print("FASE 2: AUDITORÍA DE CÁLCULO DE PF")
    
    trades_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase12_forensic_audit\reproduction\reproduced_trades.csv"
    df = pd.read_csv(trades_path)
    
    # 1. Distribution of Risk
    risk_stats = df['risk_pips'].describe()
    micro_risk_count = len(df[df['risk_pips'] < 0.2]) # trades with less than 0.2 pips of risk
    
    # 2. Wins/Losses Breakdown
    tp_r = df[df['result'] == 'TP']['r_val'].sum()
    sl_r = abs(df[df['result'] == 'SL']['r_val'].sum())
    to_r = abs(df[df['result'] == 'TIMEOUT']['r_val'].sum())
    
    gl = sl_r + to_r
    pf = tp_r / gl if gl > 0 else 999
    
    # 3. Trade Duration
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
    df['exit_time'] = pd.to_datetime(df['exit_time'], utc=True)
    df['duration_mins'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
    
    # 4. Same-bar TP/SL
    # If duration is 0 (same bar in M5), it's highly suspicious
    same_bar_count = len(df[df['duration_mins'] == 0])
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase12_forensic_audit\pf_math")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Distribution CSV
    df[['risk_pips', 'duration_mins', 'r_val']].to_csv(out_dir / "trade_r_distribution.csv", index=False)
    
    # Audit Report
    report = f"""# PF Math Audit Report

## Risk Distribution
- Mean Risk: {risk_stats['mean']:.4f} pips
- Median Risk: {risk_stats['50%']:.4f} pips
- Min Risk: {risk_stats['min']:.4f} pips
- Max Risk: {risk_stats['max']:.4f} pips
- **Micro-Risk Trades (<0.2 pips):** {micro_risk_count} ({micro_risk_count/len(df)*100:.1f}%)

## Execution Realism
- **Same-bar Trades (M5):** {same_bar_count} ({same_bar_count/len(df)*100:.1f}%)
- Avg Duration: {df['duration_mins'].mean():.1f} mins

## Mathematical Verification
- Gross Profit (R): {tp_r:.2f}
- Gross Loss (R): {gl:.2f}
- Calculated PF: {pf:.3f}

## Findings
"""
    if micro_risk_count > 0:
        report += "- **WARNING:** Detected trades with extremely small risk (under 0.2 pips). This artificially inflates PF because a 2.0R target is hit by noise.\n"
    if same_bar_count > 0:
        report += "- **CAUTION:** {same_bar_count} trades resolved within the same 5-minute bar. This requires tick-data verification or a conservative same-bar assumption.\n"
    
    if pf > 10 and risk_stats['mean'] < 1.0:
        report += "- **INVALIDATION RISK:** The high PF is likely an artifact of sub-pip risk modeling and lack of execution friction.\n"
    
    with open(out_dir / "pf_math_audit.md", 'w') as f:
        f.write(report)
        
    print("Math Audit Complete.")

if __name__ == "__main__":
    run_math_audit()
