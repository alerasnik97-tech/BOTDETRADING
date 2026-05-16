import pandas as pd
import numpy as np
from pathlib import Path

def analyze():
    csv_path = Path("results/final_audit_evidence_OOS.csv")
    if not csv_path.exists():
        print("CSV NOT FOUND")
        return

    df = pd.read_csv(csv_path)
    # Ensure numeric
    df['pf'] = pd.to_numeric(df['pf'], errors='coerce')
    df['expectancy'] = pd.to_numeric(df['expectancy'], errors='coerce')
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
    df['trades'] = pd.to_numeric(df['trades'], errors='coerce')

    print(f"Total Rows: {len(df)}")
    
    for label in df['finalist'].unique():
        sdf = df[df['finalist'] == label]
        std = sdf[sdf['stress'] == 'NO']
        stress = sdf[sdf['stress'].str.startswith('YES', na=False)]
        
        print(f"\n=== FINALIST: {label} ===")
        print(f"Standard Mode:")
        print(f"  Sample Months: {len(std)}")
        print(f"  Total Trades:  {std['trades'].sum():.0f}")
        print(f"  Avg PF:        {std['pf'].mean():.2f}")
        print(f"  Avg Expectancy:{std['expectancy'].mean():.4f}")
        print(f"  Total PnL (R): {std['pnl'].sum():.2f}")
        
        # Max DD
        if not std.empty:
            std_sorted = std.sort_values('period')
            cum_pnl = std_sorted['pnl'].cumsum().values
            peak = np.maximum.accumulate(cum_pnl)
            dd = peak - cum_pnl
            print(f"  Max Monthly DD: {np.max(dd):.2f}")
            
        if not stress.empty:
            print(f"Stress Mode (Friction):")
            print(f"  Avg PF:        {stress['pf'].mean():.2f}")
            print(f"  Avg Expectancy:{stress['expectancy'].mean():.4f}")
            print(f"  Total PnL (R): {stress['pnl'].sum():.2f}")
            
    print("\n--- ANNUAL DISTRIBUTION ---")
    print(df['period'].astype(str).str[:4].value_counts().sort_index())

if __name__ == "__main__":
    analyze()
