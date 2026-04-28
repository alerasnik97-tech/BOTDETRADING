
import pandas as pd
import os

path = r'BOT_V2_DAYTIME_LAB\outputs\phase23_phase22_forensic_readiness\be_audit\phase22_be_05_audit_full.csv'
if os.path.exists(path):
    df = pd.read_csv(path)
    profits = df[df['pnl_r'] > 0]['pnl_r'].sum()
    losses = abs(df[df['pnl_r'] < 0]['pnl_r'].sum())
    pf = profits / losses if losses > 0 else profits
    wr = len(df[df['pnl_r'] > 0]) / len(df)
    print(f"SAMPLE: {len(df)}")
    print(f"PF (Audit): {pf:.4f}")
    print(f"WR (Audit): {wr:.4f}")
    
    ambiguous = df[df['ambiguous'] == True]
    print(f"AMBIGUOUS CASES: {len(ambiguous)}")
    
    # Worst case: Assume all ambiguous that resulted in TP/BE are SL
    df_worst = df.copy()
    df_worst.loc[df_worst['ambiguous'] == True, 'pnl_r'] = -1.0
    
    p_w = df_worst[df_worst['pnl_r'] > 0]['pnl_r'].sum()
    l_w = abs(df_worst[df_worst['pnl_r'] < 0]['pnl_r'].sum())
    pf_w = p_w / l_w if l_w > 0 else p_w
    print(f"PF (FAIL-CLOSED): {pf_w:.4f}")
    df['year'] = pd.to_datetime(df['entry_time'], utc=True).dt.year
    for year, group in df.groupby('year'):
        p = group[group['pnl_r'] > 0]['pnl_r'].sum()
        l = abs(group[group['pnl_r'] < 0]['pnl_r'].sum())
        pf_y = p / l if l > 0 else p
        print(f"YEAR {year} PF: {pf_y:.4f} | Trades: {len(group)}")
else:
    print("FILE NOT FOUND")
