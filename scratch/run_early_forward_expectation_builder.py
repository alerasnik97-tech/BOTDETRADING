import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

def calculate_metrics(pnl_series):
    if len(pnl_series) == 0:
        return None
    
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series <= 0]
    
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else (999.0 if wins.sum() > 0 else 0.0)
    exp = pnl_series.mean()
    wr = (len(wins) / len(pnl_series)) * 100
    
    # Drawdown
    cum_pnl = pnl_series.cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()
    
    # Losing streak
    is_loss = (pnl_series <= 0).astype(int)
    losing_streak = is_loss * (is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumcount() + 1)
    max_losing_streak = losing_streak.max()
    
    return {
        "pf": float(pf),
        "expectancy": float(exp),
        "win_rate": float(wr),
        "max_dd": float(max_dd),
        "max_losing_streak": int(max_losing_streak)
    }

def build_envelopes(trades_path, line_name, checkpoints=[5, 10, 20, 40]):
    if not os.path.exists(trades_path):
        print(f"Error: {trades_path} not found")
        return None
    
    df = pd.read_csv(trades_path)
    # Ensure chronological order if possible
    if 'session_date' in df.columns:
        df = df.sort_values('session_date')
    
    pnls = df['pnl_r'].values
    results = {}
    
    for n in checkpoints:
        if len(pnls) < n:
            continue
            
        metrics_list = []
        # Rolling window
        for i in range(len(pnls) - n + 1):
            window = pnls[i:i+n]
            m = calculate_metrics(pd.Series(window))
            metrics_list.append(m)
            
        m_df = pd.DataFrame(metrics_list)
        
        envelope = {}
        for col in m_df.columns:
            percentiles = m_df[col].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
            envelope[col] = {str(k): float(v) for k, v in percentiles.items()}
        
        results[str(n)] = envelope
        
    return results

def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    config = [
        {
            "line": "SCBI_M5_GLOBAL",
            "path": RESULTS_DIR / "SCBI_2020_2025_DURABILITY" / "trades_baseline.csv"
        },
        {
            "line": "SCBI_CORE",
            "path": RESULTS_DIR / "SCBI_CORE_STAGE2" / "core_stage2_trades.csv"
        }
    ]
    
    all_envelopes = {}
    for entry in config:
        print(f"Building envelopes for {entry['line']}...")
        env = build_envelopes(entry['path'], entry['line'])
        if env:
            all_envelopes[entry['line']] = env
            
    output_path = RESULTS_DIR / "SCBI_EARLY_FORWARD_EXPECTATION_ENVELOPES.json"
    with open(output_path, 'w') as f:
        json.dump(all_envelopes, f, indent=2)
        
    print(f"Envelopes saved to {output_path}")

if __name__ == "__main__":
    main()
