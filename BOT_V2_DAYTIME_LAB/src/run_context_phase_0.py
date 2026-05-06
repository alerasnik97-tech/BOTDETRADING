
import pandas as pd
import json
import numpy as np
from research_v2_engine import ResearchV2Engine, calculate_metrics
from pathlib import Path

def resample_ohlc(df, tf_str):
    resampled = df.resample(tf_str, closed='left', label='right').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).shift(1).dropna()
    return resampled

def run_phase_0_baseline():
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    engine = ResearchV2Engine(manifest_path)
    
    # 1. Load Data (2015-2026)
    print("Loading H1 data...")
    df_h1 = pd.concat([engine.load_prices('period_2015_2019', 'h1'), engine.load_prices('period_2020_2026', 'h1')]).sort_values('timestamp').set_index('timestamp')
    levels = engine.get_levels(df_h1)
    
    print("Loading M1 data for M3 baseline...")
    df_m1_raw = pd.concat([engine.load_prices('period_2015_2019', 'm1'), engine.load_prices('period_2020_2026', 'm1')]).sort_values('timestamp').set_index('timestamp')
    df_m3 = resample_ohlc(df_m1_raw, '3min')
    
    # 2. Config Baseline
    config = {
        'level_type': 'pdh', # Prioritario para Short
        'start_time': '08:30',
        'end_time': '11:00',
        'tp_r': 2.0,
        'sl_value': 1.5,
        'entry_model': 'fvg',
        'slippage_pips': 0.0,
        'atr_window': 14
    }
    
    print("Running Baseline Simulation (2015-2026)...")
    trades = engine.run_simulation(df_m3, levels, config)
    
    # 3. Calculate Metrics
    metrics = calculate_metrics(trades)
    
    # Metrics por año
    trades['year'] = trades['entry_time'].dt.year
    yearly = trades.groupby('year').apply(lambda x: calculate_metrics(x)['pf']).to_dict()
    metrics['yearly_performance'] = yearly
    
    # Max loss streak
    trades['is_win'] = trades['result'] == 'TP'
    metrics['max_loss_streak'] = (trades['is_win'] == False).astype(int).groupby(trades['is_win'].cumsum()).cumsum().max()
    
    # 4. Save Outputs
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\context_phase\baseline_m3_fvg")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "baseline_m3_fvg_summary.json", 'w') as f:
        json.dump(metrics, f, indent=4, default=str)
        
    trades.to_csv(out_dir / "baseline_m3_fvg_trades.csv", index=False)
    
    # MD Report
    md_content = f"""# Baseline M3 FVG Reconfirmation (2015-2026)

* **Sample:** {metrics['sample_size']}
* **PF:** {metrics['pf']:.4f}
* **Expectancy (R):** {metrics['expectancy']:.4f}
* **Win Rate:** {metrics['win_rate']:.2%}
* **TP/SL Count:** {metrics['tp_count']} / {metrics['sl_count']}
* **Max Loss Streak:** {metrics['max_loss_streak']}
* **Yearly Performance (PF):**
{json.dumps(yearly, indent=2)}

---
*Veredicto Actual:* **NO_CANDIDATE_FOUND_PHASE3**
"""
    with open(out_dir / "baseline_m3_fvg_summary.md", 'w') as f:
        f.write(md_content)
        
    print("Phase 0 Baseline complete.")

if __name__ == "__main__":
    run_phase_0_baseline()


