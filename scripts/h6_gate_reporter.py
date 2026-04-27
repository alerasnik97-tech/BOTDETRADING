import pandas as pd
import numpy as np
from pathlib import Path

# Paths
LEDGER_OFFICIAL = Path("results/H6_SHADOW_LEDGER_OFFICIAL.csv")
LEDGER_OBSERVED = Path("results/H6_SHADOW_LEDGER_OBSERVED.csv")
DAILY_STATUS = Path("results/H6_FORWARD_ONLY_DAILY_STATUS.csv")

def calculate_metrics(df):
    if df.empty:
        return {
            "n": 0, "sum_r": 0.0, "pf": 0.0, "expectancy": 0.0, "max_dd": 0.0
        }
    
    trades = df[df['event_type'] == 'PAPER_EXIT'].copy()
    if trades.empty:
        return {
            "n": 0, "sum_r": 0.0, "pf": 0.0, "expectancy": 0.0, "max_dd": 0.0
        }
    
    r_values = trades['pnl_r'].astype(float)
    n = len(r_values)
    sum_r = r_values.sum()
    winners = r_values[r_values > 0]
    losers = r_values[r_values <= 0]
    pf = abs(winners.sum() / losers.sum()) if not losers.empty and losers.sum() != 0 else 100.0
    expectancy = sum_r / n if n > 0 else 0.0
    
    pnl_curve = r_values.cumsum()
    peak = pnl_curve.cummax()
    dd = pnl_curve - peak
    max_dd = dd.min()
    
    return {
        "n": n,
        "sum_r": round(sum_r, 4),
        "pf": round(pf, 2),
        "expectancy": round(expectancy, 4),
        "max_dd": round(max_dd, 4)
    }

def generate_report():
    print("="*60)
    print("H6 APRIL GATE AUTOMATION ENGINE")
    print("="*60)
    
    if not LEDGER_OFFICIAL.exists() or not LEDGER_OBSERVED.exists():
        print("ERROR: Ledgers missing.")
        return

    df_off = pd.read_csv(LEDGER_OFFICIAL)
    df_obs = pd.read_csv(LEDGER_OBSERVED)
    
    # Split by provenance
    off_backfill = df_off[df_off['provenance'] == 'BACKFILL']
    off_forward = df_off[df_off['provenance'] == 'FORWARD']
    
    obs_backfill = df_obs[df_obs['provenance'] == 'BACKFILL']
    obs_forward = df_obs[df_obs['provenance'] == 'FORWARD']
    
    m_off_bf = calculate_metrics(off_backfill)
    m_off_fw = calculate_metrics(off_forward)
    m_obs_bf = calculate_metrics(obs_backfill)
    m_obs_fw = calculate_metrics(obs_forward)
    
    print(f"\n[1] BASELINE BACKFILL (N={m_off_bf['n']})")
    print(f"OFFICIAL  | PF: {m_off_bf['pf']} | EXP: {m_off_bf['expectancy']} | DD: {m_off_bf['max_dd']}")
    print(f"OBSERVED  | PF: {m_obs_bf['pf']} | EXP: {m_obs_bf['expectancy']} | DD: {m_obs_bf['max_dd']}")
    
    print(f"\n[2] EVIDENCE FORWARD (N={m_off_fw['n']})")
    print(f"OFFICIAL  | PF: {m_off_fw['pf']} | EXP: {m_off_fw['expectancy']} | DD: {m_off_fw['max_dd']}")
    print(f"OBSERVED  | PF: {m_obs_fw['pf']} | EXP: {m_obs_fw['expectancy']} | DD: {m_obs_fw['max_dd']}")
    
    print("\n[3] COMPARISON & EDGE RETENTION")
    if m_off_bf['expectancy'] != 0:
        retention = (m_off_fw['expectancy'] / m_off_bf['expectancy']) * 100 if m_off_fw['n'] > 0 else 0
        print(f"Edge Retention (Fwd/Bkf): {round(retention, 2)}%")
    
    div = m_obs_fw['expectancy'] - m_off_fw['expectancy'] if m_off_fw['n'] > 0 else 0
    print(f"Model Divergence (Obs - Off): {round(div, 4)} R/trade")
    
    print("\n" + "="*60)
    print("END OF AUTOMATED REPORT")
    print("="*60)

if __name__ == "__main__":
    generate_report()
