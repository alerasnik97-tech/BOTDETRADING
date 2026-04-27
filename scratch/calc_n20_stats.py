import pandas as pd
off = pd.read_csv('results/H6_RESEARCH_VS_SHADOW_OFFICIAL.csv')
diag = pd.read_csv('results/H6_RESEARCH_VS_SHADOW_DIAGNOSTIC.csv')
cal = pd.read_csv('results/H6_SPREAD_SLIPPAGE_CALIBRATION.csv')

def get_stats(df, name):
    total_r = df['shadow_pnl_r'].sum()
    total_res = df['research_pnl_r'].sum()
    wins = df[df['shadow_pnl_r'] > 0]['shadow_pnl_r'].sum()
    losses = abs(df[df['shadow_pnl_r'] < 0]['shadow_pnl_r'].sum())
    pf = wins / losses if losses > 0 else float('inf')
    return {
        "name": name,
        "res_r": total_res,
        "shadow_r": total_r,
        "pf": pf
    }

s_off = get_stats(off, "OFFICIAL")
s_diag = get_stats(diag, "DIAGNOSTIC")

print(f"RESEARCH Total: {s_off['res_r']:.4f}")
print(f"OFFICIAL Total: {s_off['shadow_r']:.4f}, PF: {s_off['pf']:.2f}")
print(f"DIAGNOSTIC Total: {s_diag['shadow_r']:.4f}, PF: {s_diag['pf']:.2f}")
print(f"AVG Entry Spread: {cal['observed_spread_entry_pips'].mean():.4f}")
print(f"Max Entry Spread: {cal['observed_spread_entry_pips'].max():.4f}")
