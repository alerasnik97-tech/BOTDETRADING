
import pandas as pd
import numpy as np
import json
from pathlib import Path
from phase6_engine import Phase6Engine

def run_improvement_variant(variant_name, config_patch):
    print(f"  Testing Variant: {variant_name}...")
    engine = Phase6Engine()
    
    # Baseline Config
    config = {
        'entry_type': 1, 'timeframe': 'm3', 'fractal_n': 8,
        'start_hour': '08:30', 'end_hour': '11:00',
        'tp_val': 1.5, 'be_r': None, 'sl_type': 'sweep',
        'sl_plus_pips': 0.5, 'news_block_mins': 30,
        'one_trade_per_day': True, 'first_sweep_only': True,
        'min_atr': 0.0012, 
        'trend_exhaustion': True
    }
    config.update(config_patch)
    
    # Load manifest
    manifest_path = r"C:\Users\alera\Desktop\Bot\Bot V1\data_manifest\certified_data_paths.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    periods = ['period_2015_2019', 'period_2020_2026']
    news_list = []
    for p in periods:
        news_list.append(pd.read_csv(manifest[p]['news']))
    news_df = pd.concat(news_list)
    
    all_trades = []
    for p in periods:
        df_src = pd.read_csv(manifest[p]['m5_bid'])
        df_src['timestamp'] = pd.to_datetime(df_src['timestamp'], utc=True)
        df_src.set_index('timestamp', inplace=True)
        df_m3 = df_src.resample('3min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
        df_m3['timestamp_ny'] = pd.to_datetime(df_m3['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        df_m3['is_high_fractal'], df_m3['is_low_fractal'] = engine.get_fractals(df_m3, n=8)
        
        df_h1 = pd.read_csv(manifest[p]['h1_bid'])
        df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'], utc=True).dt.tz_convert(engine.tz_ny)
        levels = engine.get_levels(df_h1)
        
        trades = engine.run_phase6_backtest(df_m3, levels, news_df, config)
        
        # Apply weekday filter if in config
        if 'exclude_weekdays' in config:
            trades['weekday'] = pd.to_datetime(trades['entry_time'], utc=True).dt.day_name()
            trades = trades[~trades['weekday'].isin(config['exclude_weekdays'])]
            
        # Apply level filter if in config
        if 'exclude_levels' in config:
            trades = trades[~trades['level'].isin(config['exclude_levels'])]

        # Apply min/max depth filter (Fase 2A)
        if 'min_depth_pips' in config:
            trades = trades[trades['max_depth_pips'] >= config['min_depth_pips']]
        if 'max_depth_pips' in config:
            trades = trades[trades['max_depth_pips'] <= config['max_depth_pips']]

        all_trades.append(trades)
    
    full_trades = pd.concat(all_trades)
    
    # Calculate Metrics
    gp = full_trades[full_trades['r_value'] > 0]['r_value'].sum()
    gl = abs(full_trades[full_trades['r_value'] < 0]['r_value'].sum())
    pf = gp / gl if gl > 0 else 0
    
    return {
        "variant": variant_name,
        "sample": len(full_trades),
        "pf": round(pf, 3),
        "expectancy": round(full_trades['r_value'].mean(), 4),
        "cumulative_r": round(full_trades['r_value'].sum(), 2)
    }

def run_all_improvements():
    print("Phase 2: Individual Improvements Laboratory")
    results = []
    
    # 1. Baseline
    results.append(run_improvement_variant("Baseline", {}))
    
    # 2. Sweep Quality (Fase 2A)
    results.append(run_improvement_variant("Sweep_Min_1_Pip", {"min_depth_pips": 1.0}))
    results.append(run_improvement_variant("Sweep_Max_15_Pips", {"max_depth_pips": 15.0}))
    results.append(run_improvement_variant("Sweep_Max_20_Pips", {"max_depth_pips": 20.0}))
    
    # 3. CHoCH Quality (Fase 2B)
    results.append(run_improvement_variant("CHoCH_Body_50", {"min_body_pct": 0.50}))
    results.append(run_improvement_variant("CHoCH_Body_60", {"min_body_pct": 0.60}))
    
    # 4. Level/Weekday Filters (Fase 2C)
    results.append(run_improvement_variant("Exclude_Friday", {"exclude_weekdays": ["Friday"]}))
    results.append(run_improvement_variant("Exclude_PDH", {"exclude_levels": ["pdh"]}))
    
    # 5. Volatility (Fase 2F)
    results.append(run_improvement_variant("ATR_GT_14", {"min_atr": 0.0014}))
    
    # 6. Management (Fase 2H)
    results.append(run_improvement_variant("TP_2.0R", {"tp_val": 2.0}))
    
    out_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\outputs\phase8_improvement_lab\individual_improvements")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "individual_improvement_results.csv", index=False)
    
    with open(out_dir / "individual_improvement_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Individual Improvements Complete.")

if __name__ == "__main__":
    run_all_improvements()


